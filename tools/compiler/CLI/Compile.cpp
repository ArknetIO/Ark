// tools/compiler/CLI/Compile.cpp
#include "Subcommands.h"
#include "Workspace.h"
#include "Config.h"

#include <CLI/CLI.hpp>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>

// --- Ark Pipeline Headers ---
#include "../hud.h"
#include "../Frontend/ModuleRegistry.h"
#include "../Pipeline/Compiler.h"
#include "../Pipeline/Linker.h"
#include "../Pipeline/JIT.h"
#include "../Integration/CapsuleBackend.h"

// --- MLIR Headers ---
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "ark/IR/ArkMirOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

// --- Translation Headers ---
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// --- Conversion Headers ---
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef ARK_VERSION_STRING
  #define ARK_VERSION_STRING "dev"
#endif


#if !defined(_WIN32)
#include <unistd.h>
extern "C" {
    extern char** environ;
}
#else
#include <io.h>
#endif

namespace ark::cli {

struct CompileOptions {
    std::string inputFile;
    std::string outputFile;
    bool bare = false;
    bool jit = false;
    bool keepTmp = false;
    bool verbose = false;
    bool noSeal = false;

    bool seal = false;
    bool requireSeal = false;

    std::string runtimePath = "tools/compiler/Runtime";
    std::string llvmBinDir = "";
    std::string targetTriple = "";

    bool stageCompile = false;
    bool stageLower = false;
    bool stageTranslate = false;
    bool stageLinkOnly = false;
    std::string inLl = "";

    std::string sealKeyHex = "";
    std::string sealKeyFile = "";
    bool noVaultPrompt = false;
};

static CompileOptions opts;

// -----------------------------------------------------------------------------
// Pipeline Helpers
// -----------------------------------------------------------------------------
static std::string fmtBytes(std::uint64_t bytes) {
    static constexpr std::array<const char*, 5> units = {"B", "KiB", "MiB", "GiB", "TiB"};
    long double v = static_cast<long double>(bytes);
    std::size_t u = 0;
    while (u + 1 < units.size() && v >= 1024.0L) {
        v /= 1024.0L;
        ++u;
    }

    std::ostringstream o;
    o.setf(std::ios::fixed);

    if (u == 0) {
        o.unsetf(std::ios::floatfield);
        o << bytes << ' ' << units[u];
        return o.str();
    }

    const int prec = (v < 10.0L) ? 2 : 1;
    o << std::setprecision(prec) << static_cast<double>(v) << ' ' << units[u];
    return o.str();
}

static std::string resolveRuntimePath() {
    if (!opts.runtimePath.empty() && opts.runtimePath != "tools/compiler/Runtime") {
        return opts.runtimePath;
    }

    void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(resolveRuntimePath));
    auto mainExe = llvm::sys::fs::getMainExecutable(nullptr, mainAddr);

    llvm::SmallString<256> p = llvm::sys::path::parent_path(mainExe);
    llvm::sys::path::append(p, "../tools/compiler/Runtime");
    if (llvm::sys::fs::exists(p)) return std::string(p.str());

    p = llvm::sys::path::parent_path(mainExe);
    llvm::sys::path::append(p, "../../tools/compiler/Runtime");
    if (llvm::sys::fs::exists(p)) return std::string(p.str());

    return "tools/compiler/Runtime";
}

template <typename Fn>
static bool runStep(arklang::hud::Hud& hud, arklang::hud::StepMeta meta, Fn&& fn) {
    arklang::hud::Step s(hud, std::move(meta));
    if (!fn()) {
        s.fail();
        return false;
    }
    s.ok();
    return true;
}

static std::string readAllFile(llvm::StringRef p) {
    auto mb = llvm::MemoryBuffer::getFile(p);
    if (!mb) return {};
    return std::string(mb.get()->getBuffer());
}

static void removeQuietly(llvm::StringRef p) {
    if (!p.empty()) (void)llvm::sys::fs::remove(p);
}

static void removeDirQuietly(llvm::StringRef p) {
    if (p.empty()) return;
    (void)llvm::sys::fs::remove_directories(p, /*IgnoreErrors=*/true);
}

static void bestEffortAuditFatbinExport(arklang::hud::Hud& hud, llvm::StringRef binaryPath, bool enabled) {
    if (!enabled) return;

    auto nm = llvm::sys::findProgramByName("nm");
    if (!nm) return;

    llvm::SmallString<256> outPath;
    llvm::SmallString<256> errPath;
    (void)llvm::sys::fs::createTemporaryFile("ark_nm_out", "log", outPath);
    (void)llvm::sys::fs::createTemporaryFile("ark_nm_err", "log", errPath);

    llvm::SmallVector<llvm::StringRef, 8> argv;
    argv.push_back(*nm);
    argv.push_back("-D");
    argv.push_back(binaryPath);

    std::string errMsg;
    const llvm::StringRef outRef(outPath);
    const llvm::StringRef errRef(errPath);
    std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};

    (void)llvm::sys::ExecuteAndWait(*nm, argv, std::nullopt, redirects, 0, 0, &errMsg);

    const std::string so = readAllFile(outPath);
    removeQuietly(outPath);
    removeQuietly(errPath);

    if (!errMsg.empty()) {
        hud.pushLogBlock("nm exec error", errMsg);
        return;
    }

    const bool found = (so.find("ark_gpu_fatbin_query_v1") != std::string::npos);
    hud.note(std::string("fatbin registry export: ") + (found ? "present" : "missing"));
}

static std::optional<llvm::ArrayRef<llvm::StringRef>>
buildInheritedEnv(llvm::SmallVectorImpl<std::string>& storage,
                  llvm::SmallVectorImpl<llvm::StringRef>& refs) {
    storage.clear();
    refs.clear();

#if defined(_WIN32)
    char*** penv = __p__environ();
    if (!penv || !*penv) return std::nullopt;

    for (char** e = *penv; *e != nullptr; ++e) {
        storage.emplace_back(*e);
    }
#else
    if (!::environ) return std::nullopt;

    for (char** e = ::environ; *e != nullptr; ++e) {
        storage.emplace_back(*e);
    }
#endif
    

    refs.reserve(storage.size());
    for (auto& s : storage) {
        refs.push_back(s);
    }

    return llvm::ArrayRef<llvm::StringRef>(refs);
}
static std::optional<std::string> getenvStr(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    std::string s(v);
    if (s.empty()) return std::nullopt;
    return s;
}

static bool stdinIsTty() {
#if defined(_WIN32)
    return ::_isatty(::_fileno(stdin)) != 0;
#else
    return ::isatty(STDIN_FILENO) != 0;
#endif
}

static std::string_view trimAscii(std::string_view s) {
    auto isWs = [](unsigned char c) -> bool { return std::isspace(c) != 0; };

    while (!s.empty() && isWs(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && isWs(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

static std::string trimOwned(std::string s) {
    const std::string_view v = trimAscii(std::string_view(s));
    return std::string(v);
}

static bool parseHexByte(char hi, char lo, std::uint8_t& out) {
    auto hx = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };

    const int a = hx(hi);
    const int b = hx(lo);
    if (a < 0 || b < 0) return false;

    out = static_cast<std::uint8_t>((a << 4) | b);
    return true;
}

static bool parseKeyHex32(std::string_view in, std::vector<std::uint8_t>& outBytes, std::string& err) {
    in = trimAscii(in);
    if (in.rfind("0x", 0) == 0 || in.rfind("0X", 0) == 0) in.remove_prefix(2);

    std::string compact;
    compact.reserve(in.size());
    for (char c : in) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) continue;
        compact.push_back(c);
    }

    if (compact.size() != 64) {
        err = "seal key must be 64 hex characters (32 bytes).";
        return false;
    }

    outBytes.clear();
    outBytes.reserve(32);

    for (std::size_t i = 0; i < 64; i += 2) {
        std::uint8_t b = 0;
        if (!parseHexByte(compact[i], compact[i + 1], b)) {
            err = "seal key contains non-hex characters.";
            outBytes.clear();
            return false;
        }
        outBytes.push_back(b);
    }

    if (outBytes.size() != 32) {
        err = "seal key decode failed (expected 32 bytes).";
        outBytes.clear();
        return false;
    }

    return true;
}

static std::optional<std::string> readTextFileStrict(std::string_view path, std::string& err) {
    auto mb = llvm::MemoryBuffer::getFile(std::string(path));
    if (!mb) {
        err = "failed to read file";
        return std::nullopt;
    }
    return std::string(mb.get()->getBuffer());
}

static std::string buildCapsuleSealKeySetupHelp() {
    std::string msg;
    msg += "Capsule sealing is enabled but no seal key is configured.\n";
    msg += "\n";
    msg += "Accepted sources (first match wins):\n";
    msg += "  1) --seal-key-hex <64-hex>\n";
    msg += "  2) --seal-key-file <path>\n";
    msg += "  3) ARK_CAPSULE_SEAL_KEY_HEX (or ARKNET_CAPSULE_SEAL_KEY_HEX)\n";
    msg += "  4) ARK_CAPSULE_SEAL_KEY_FILE (or ARKNET_CAPSULE_SEAL_KEY_FILE)\n";
    msg += "  5) Vault secret: [capsule].seal_key_hex\n";
    msg += "\n";

#if defined(_WIN32)
    msg += "Generate a 32-byte key (Windows PowerShell):\n";
    msg += "  $b = New-Object byte[] 32; [System.Security.Cryptography.RandomNumberGenerator]::Fill($b); ";
    msg += "$k = ($b | ForEach-Object { $_.ToString('x2') }) -join ''\n";
    msg += "  $env:ARK_CAPSULE_SEAL_KEY_HEX = $k\n";
    msg += "\n";
    msg += "Or file-based:\n";
    msg += "  New-Item -ItemType Directory -Force \"$env:USERPROFILE\\.arknet\" | Out-Null\n";
    msg += "  Set-Content -Path \"$env:USERPROFILE\\.arknet\\capsule_seal_key.hex\" -Value $k -NoNewline\n";
    msg += "  $env:ARK_CAPSULE_SEAL_KEY_FILE = \"$env:USERPROFILE\\.arknet\\capsule_seal_key.hex\"\n";
#elif defined(__APPLE__)
    msg += "Generate a 32-byte key (macOS zsh/bash):\n";
    msg += "  export ARK_CAPSULE_SEAL_KEY_HEX=\"$(openssl rand -hex 32)\"\n";
    msg += "\n";
    msg += "Or file-based:\n";
    msg += "  mkdir -p ~/.arknet && umask 077 && openssl rand -hex 32 > ~/.arknet/capsule_seal_key.hex\n";
    msg += "  export ARK_CAPSULE_SEAL_KEY_FILE=\"$HOME/.arknet/capsule_seal_key.hex\"\n";
#elif defined(__linux__)
    msg += "Generate a 32-byte key (Linux bash/zsh):\n";
    msg += "  export ARK_CAPSULE_SEAL_KEY_HEX=\"$(openssl rand -hex 32)\"\n";
    msg += "\n";
    msg += "Or file-based:\n";
    msg += "  mkdir -p ~/.arknet && umask 077 && openssl rand -hex 32 > ~/.arknet/capsule_seal_key.hex\n";
    msg += "  export ARK_CAPSULE_SEAL_KEY_FILE=\"$HOME/.arknet/capsule_seal_key.hex\"\n";
#else
    msg += "Generate a 32-byte key:\n";
    msg += "  Use any CSPRNG to produce 32 random bytes and encode as 64 hex chars.\n";
    msg += "  Then set ARK_CAPSULE_SEAL_KEY_HEX or ARK_CAPSULE_SEAL_KEY_FILE.\n";
#endif

    msg += "\n";
    msg += "Vault path (recommended for local dev):\n";
    msg += "  arknet config --set-secret capsule.seal_key_hex <64-hex>\n";
    return msg;
}

static std::string buildCapsuleVaultPasswordHelp() {
    std::string msg;
    msg += "Capsule seal key exists in the vault, but no vault password was provided.\n";
    msg += "\n";
    msg += "Provide one of:\n";
    msg += "  - ARKNET_VAULT_PASSWORD\n";
    msg += "  - ARK_CAPSULE_VAULT_PASSWORD\n";
    msg += "  - interactive TTY prompt (default)\n";
    msg += "\n";
    msg += "Or bypass the vault for this build using:\n";
    msg += "  --seal-key-hex / --seal-key-file\n";
    return msg;
}

static bool tryLoadCapsuleSealKey(std::vector<std::uint8_t>& outKey, std::string& outErr) {
    std::optional<std::string> hexOverride;
    if (!opts.sealKeyHex.empty()) hexOverride = trimOwned(opts.sealKeyHex);

    if (!hexOverride || hexOverride->empty()) {
        if (auto v = getenvStr("ARK_CAPSULE_SEAL_KEY_HEX")) hexOverride = trimOwned(*v);
        else if (auto v2 = getenvStr("ARKNET_CAPSULE_SEAL_KEY_HEX")) hexOverride = trimOwned(*v2);
    }

    std::optional<std::string> fileOverride;
    if (!opts.sealKeyFile.empty()) fileOverride = trimOwned(opts.sealKeyFile);

    if (!fileOverride || fileOverride->empty()) {
        if (auto v = getenvStr("ARK_CAPSULE_SEAL_KEY_FILE")) fileOverride = trimOwned(*v);
        else if (auto v2 = getenvStr("ARKNET_CAPSULE_SEAL_KEY_FILE")) fileOverride = trimOwned(*v2);
    }

    std::string parseErr;

    if (hexOverride && !hexOverride->empty()) {
        if (!parseKeyHex32(*hexOverride, outKey, parseErr)) {
            outErr = "Invalid capsule seal key from --seal-key-hex / env: " + parseErr;
            return false;
        }
        return true;
    }

    if (fileOverride && !fileOverride->empty()) {
        std::string ioErr;
        auto txt = readTextFileStrict(*fileOverride, ioErr);
        if (!txt) {
            outErr = "Failed to load capsule seal key file '" + *fileOverride + "': " + ioErr;
            return false;
        }

        if (!parseKeyHex32(trimAscii(*txt), outKey, parseErr)) {
            outErr = "Invalid capsule seal key in file '" + *fileOverride + "': " + parseErr;
            return false;
        }
        return true;
    }

    if (!GlobalConfig::hasEncryptedSecret("capsule", "seal_key_hex")) {
        outErr = buildCapsuleSealKeySetupHelp();
        return false;
    }

    std::optional<std::string> vaultPwd = getenvStr("ARKNET_VAULT_PASSWORD");
    if (!vaultPwd) vaultPwd = getenvStr("ARK_CAPSULE_VAULT_PASSWORD");
    if (vaultPwd) *vaultPwd = trimOwned(*vaultPwd);

    if (!vaultPwd || vaultPwd->empty()) {
        if (opts.noVaultPrompt || !stdinIsTty()) {
            outErr = buildCapsuleVaultPasswordHelp();
            return false;
        }

        std::string prompted = promptForPassword("Vault password (to decrypt capsule seal key)");
        prompted = trimOwned(prompted);
        if (prompted.empty()) {
            outErr = "Vault password cannot be empty.";
            return false;
        }
        vaultPwd = std::move(prompted);
    }

    auto hex = GlobalConfig::getDecryptedSecret("capsule", "seal_key_hex", *vaultPwd);
    if (!hex || hex->empty()) {
        outErr = "Failed to decrypt capsule seal key from vault.";
        return false;
    }

    if (!parseKeyHex32(trimAscii(*hex), outKey, parseErr)) {
        outErr = "Invalid capsule seal key stored in vault: " + parseErr;
        return false;
    }

    return true;
}

static std::vector<std::uint8_t> loadCapsuleSealKeyOrExit(arklang::hud::Hud& hud) {
    std::vector<std::uint8_t> key;
    std::string err;
    if (tryLoadCapsuleSealKey(key, err)) return key;

    hud.error(err);
    hud.finish(false);
    std::exit(1);
}

// -----------------------------------------------------------------------------
// Core Engine
// -----------------------------------------------------------------------------
static void executePipeline(bool isRunMode, arklang::hud::Hud& hud) {
    // Policy:
    // - run: unsealed by default
    // - compile: sealed by default (unless bare/jit)
    bool wantCapsuleDefault = (!isRunMode && !opts.bare && !opts.jit);
    bool wantCapsule = wantCapsuleDefault;

    if (opts.seal) wantCapsule = true;
    if (opts.noSeal || opts.bare || opts.jit) wantCapsule = false;

    const bool strictSeal = (opts.seal || opts.requireSeal);

    std::vector<std::uint8_t> capsuleSealKey;
    if (wantCapsule) {
        std::string keyErr;
        if (!tryLoadCapsuleSealKey(capsuleSealKey, keyErr)) {
            if (strictSeal) {
                hud.error(keyErr);
                hud.finish(false);
                std::exit(1);
            }

            if (isRunMode) hud.note("No capsule seal key configured; running unsealed local build.");
            else hud.note("No capsule seal key configured; continuing with unsealed build.");

            wantCapsule = false;
        }
    }

    if (opts.verbose) {
        hud.banner("arknet", ARK_VERSION_STRING, wantCapsule ? "Secure Pipeline" : "Raw Pipeline");
    }

    std::optional<std::string> explicitInput = opts.inputFile.empty()
        ? std::nullopt
        : std::make_optional(opts.inputFile);

    WorkspaceConfig config = Workspace::discover(explicitInput, hud);
    if (!opts.targetTriple.empty()) config.profile.targetTriple = opts.targetTriple;

    std::string cacheKey = Workspace::computeCacheKey(config, hud);
    std::string stem = llvm::sys::path::stem(config.entryFile).str();

    if (opts.outputFile.empty()) {
        if (opts.stageCompile) opts.outputFile = stem + ".mlir";
        else if (opts.stageLower) opts.outputFile = stem + ".llvm.mlir";
        else if (opts.stageTranslate) opts.outputFile = stem + ".ll";
    }

    std::string finalOutput = opts.outputFile;
    if (finalOutput.empty()) {
        llvm::SmallString<256> outPath(config.buildDir);
        llvm::sys::fs::create_directories(outPath);

        std::string baseName = config.projectName.empty() ? stem : config.projectName;
        llvm::sys::path::append(outPath, baseName + (wantCapsule ? "" : ".raw"));
        finalOutput = std::string(outPath.str());

#if defined(_WIN32)
        if (wantCapsule && !llvm::StringRef(finalOutput).ends_with(".exe")) finalOutput += ".exe";
#endif
    }

    llvm::SmallString<256> cacheLayerDir(config.buildDir);
    llvm::sys::path::append(cacheLayerDir, ".ark", "cache");
    llvm::sys::fs::create_directories(cacheLayerDir);

    llvm::SmallString<256> cachedMlirPath(cacheLayerDir);
    llvm::sys::path::append(cachedMlirPath, cacheKey + ".mlir");

    bool isCacheHit = llvm::sys::fs::exists(cachedMlirPath) && llvm::sys::fs::exists(finalOutput);
    bool performBuild = !isCacheHit && !opts.stageLinkOnly;

    int steps = 0;
    if (opts.stageLinkOnly) {
        steps = 1;
        if (wantCapsule) steps++;
    } else if (performBuild) {
        steps = 1;
        if (!opts.stageCompile) steps++;
        if (!opts.stageCompile && !opts.stageLower) {
            if (opts.jit) {
                steps++;
            } else {
                steps++;
                if (!opts.stageTranslate) {
                    steps++;
                    if (wantCapsule) steps++;
                }
            }
        }
    }
    if (isRunMode) steps++;
    hud.setTotalSteps(steps);

    std::string shortKey = cacheKey.substr(0, 8);
    std::string displayName = config.projectName.empty() ? stem : config.projectName;

    if (performBuild) hud.note("Compiling " + displayName + " [" + shortKey + "]");
    else if (!opts.stageLinkOnly) hud.note("Cached    " + displayName + " [" + shortKey + "]");

    const std::string validRuntime = resolveRuntimePath();
    if (!llvm::sys::fs::exists(validRuntime)) {
        hud.error(std::string("Runtime not found at: ") + validRuntime);
        hud.finish(false);
        std::exit(1);
    }

    arklang::ModuleRegistry registry;
    ark::compiler::pipeline::Compiler compiler(hud, registry);

    ark::compiler::pipeline::LinkerConfig linkCfg;
    linkCfg.runtimePath = validRuntime;
    linkCfg.llvmBinDir = opts.llvmBinDir;
    linkCfg.keepTmp = opts.keepTmp;
    ark::compiler::pipeline::Linker linker(hud, linkCfg);

    mlir::ModuleOp module;
    std::vector<ark::compiler::pipeline::CompiledGpuModule> gpuMods;

    // =========================================================================
    // LINK ONLY FAST-PATH
    // =========================================================================
    if (opts.stageLinkOnly) {
        const std::string tmpDir = linker.makeTempDir();
        if (opts.keepTmp) hud.note(std::string("Build Temp: ") + tmpDir);

        const std::string binaryPath = wantCapsule ? (tmpDir + "/payload.bin") : finalOutput;

        if (!runStep(hud, {"Linking", "LLVM IR → Native Binary"}, [&] {
            return linker.linkToBinary(opts.inLl, binaryPath, gpuMods);
        })) {
            hud.finish(false);
            if (!opts.keepTmp) removeDirQuietly(tmpDir);
            std::exit(1);
        }

        bestEffortAuditFatbinExport(hud, binaryPath, opts.verbose);

        if (wantCapsule) {
            arklang::hud::Step s(hud, {"Sealing", "Creating Self-Executing Capsule"});
            void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(resolveRuntimePath));
            auto mainExe = llvm::sys::fs::getMainExecutable(nullptr, mainAddr);
            llvm::SmallString<256> stubPath = llvm::sys::path::parent_path(mainExe);
            llvm::sys::path::append(stubPath, "ark-stub");

            std::uint64_t sealedSize = 0;
            bool ok = ark::compiler::integration::CapsuleBackend::CreateSelfExecutingCapsule(
                hud,
                config.buildDir,
                std::string(stubPath.str()),
                binaryPath,
                finalOutput,
                capsuleSealKey,
                &sealedSize);

            if (!ok) {
                s.fail();
                hud.finish(false);
                std::exit(1);
            }

            (void)llvm::sys::fs::setPermissions(
                finalOutput,
                llvm::sys::fs::owner_exe | llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);

            s.ok();
            hud.pushLogBlock("Sealed", finalOutput + " (" + fmtBytes(sealedSize) + ")");
            if (!opts.keepTmp) removeQuietly(binaryPath);
        }

        if (!opts.keepTmp) removeDirQuietly(tmpDir);
        goto EXECUTION_PHASE;
    }

    // =========================================================================
    // STANDARD BUILD PHASE
    // =========================================================================
    if (performBuild) {
        mlir::DialectRegistry dialectRegistry;
        dialectRegistry.insert<
            arklang::mir::ArkMirDialect,
            mlir::func::FuncDialect,
            mlir::LLVM::LLVMDialect,
            mlir::arith::ArithDialect,
            mlir::memref::MemRefDialect,
            mlir::scf::SCFDialect,
            mlir::cf::ControlFlowDialect,
            mlir::vector::VectorDialect,
            mlir::index::IndexDialect,
            mlir::gpu::GPUDialect,
            mlir::ub::UBDialect,
            mlir::NVVM::NVVMDialect,
            mlir::ROCDL::ROCDLDialect>();

        mlir::arith::registerConvertArithToLLVMInterface(dialectRegistry);
        mlir::cf::registerConvertControlFlowToLLVMInterface(dialectRegistry);
        mlir::registerConvertFuncToLLVMInterface(dialectRegistry);
        mlir::index::registerConvertIndexToLLVMInterface(dialectRegistry);
        mlir::registerConvertMemRefToLLVMInterface(dialectRegistry);
        mlir::vector::registerConvertVectorToLLVMInterface(dialectRegistry);
        mlir::ub::registerConvertUBToLLVMInterface(dialectRegistry);
        mlir::registerConvertNVVMToLLVMInterface(dialectRegistry);

        mlir::registerBuiltinDialectTranslation(dialectRegistry);
        mlir::registerLLVMDialectTranslation(dialectRegistry);
        mlir::registerNVVMDialectTranslation(dialectRegistry);
        mlir::registerROCDLDialectTranslation(dialectRegistry);
        mlir::registerGPUDialectTranslation(dialectRegistry);

        mlir::MLIRContext ctx(dialectRegistry);
        ctx.loadAllAvailableDialects();

        if (!runStep(hud, {"Compiling", "Source → MIR"}, [&] {
            return compiler.compileToMLIR(config.entryFile, ctx, module);
        })) {
            hud.finish(false);
            std::exit(1);
        }

        if (opts.stageCompile) {
            compiler.writeModuleToFile(module, opts.outputFile);
            hud.note("wrote MLIR: " + opts.outputFile);
            hud.finish(true);
            return;
        }

        compiler.writeModuleToFile(module, std::string(cachedMlirPath.str()));

        if (!runStep(hud, {"Lowering", "MIR → LLVM Dialect"}, [&] {
            if (!compiler.lowerToLLVM(ctx, module)) return false;

            if (!opts.stageLower) {
                gpuMods = compiler.extractCompiledGpuModules(module);
                if (opts.verbose) {
                    hud.note("GPU modules captured: " + std::to_string(gpuMods.size()));
                    for (const auto& m : gpuMods) {
                        hud.note("  gpu: key=" + m.moduleKey + " bytes=" + std::to_string(m.blob.size()));
                    }
                }
            } else if (opts.verbose) {
                hud.note("stop after LLVM dialect: preserving ark.gpu.modules in MLIR");
            }

            return true;
        })) {
            hud.finish(false);
            std::exit(1);
        }

        if (opts.stageLower) {
            compiler.writeModuleToFile(module, opts.outputFile);
            hud.note("wrote LLVM dialect MLIR: " + opts.outputFile);
            hud.finish(true);
            return;
        }

        if (opts.jit) {
            arklang::hud::Step s(hud, {"JIT Execution", "Running in-memory"});
            int rc = ark::compiler::pipeline::JIT::Run(module, validRuntime);
            if (rc == 0) s.ok(); else s.fail();
            hud.finish(rc == 0);
            std::exit(rc);
        }

        const std::string tmpDir = linker.makeTempDir();
        if (opts.keepTmp) hud.note(std::string("Build Temp: ") + tmpDir);

        const std::string binaryPath = wantCapsule ? (tmpDir + "/payload.bin") : finalOutput;
        const std::string tmpLl = opts.stageTranslate ? opts.outputFile : (tmpDir + "/out.ll");

        if (!runStep(hud, {"Translating", "MLIR → LLVM IR"}, [&] {
            llvm::LLVMContext llvmContext;
            auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
            if (!llvmModule) return false;

            std::error_code ec;
            llvm::ToolOutputFile out(tmpLl, ec, llvm::sys::fs::OF_None);
            if (ec) return false;

            llvmModule->print(out.os(), nullptr);
            out.keep();
            return true;
        })) {
            hud.finish(false);
            if (!opts.keepTmp) removeDirQuietly(tmpDir);
            std::exit(1);
        }

        if (opts.stageTranslate) {
            hud.note("wrote LLVM IR: " + tmpLl);
            hud.finish(true);
            return;
        }

        if (!runStep(hud, {"Linking", "LLVM IR → Native Binary"}, [&] {
            return linker.linkToBinary(tmpLl, binaryPath, gpuMods);
        })) {
            hud.finish(false);
            if (!opts.keepTmp) removeDirQuietly(tmpDir);
            std::exit(1);
        }

        bestEffortAuditFatbinExport(hud, binaryPath, opts.verbose && !gpuMods.empty());

        if (wantCapsule) {
            arklang::hud::Step s(hud, {"Sealing", "Creating Self-Executing Capsule"});
            void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(resolveRuntimePath));
            auto mainExe = llvm::sys::fs::getMainExecutable(nullptr, mainAddr);
            llvm::SmallString<256> stubPath = llvm::sys::path::parent_path(mainExe);
            llvm::sys::path::append(stubPath, "ark-stub");

            std::uint64_t sealedSize = 0;
            bool ok = ark::compiler::integration::CapsuleBackend::CreateSelfExecutingCapsule(
                hud,
                config.buildDir,
                std::string(stubPath.str()),
                binaryPath,
                finalOutput,
                capsuleSealKey,
                &sealedSize);

            if (!ok) {
                s.fail();
                hud.finish(false);
                std::exit(1);
            }

            (void)llvm::sys::fs::setPermissions(
                finalOutput,
                llvm::sys::fs::owner_exe | llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);

            s.ok();
            hud.pushLogBlock("Sealed", finalOutput + " (" + fmtBytes(sealedSize) + ")");
            if (!opts.keepTmp) removeQuietly(binaryPath);
        }

        if (!opts.keepTmp) removeDirQuietly(tmpDir);
    }

EXECUTION_PHASE:
    if (isRunMode) {
        arklang::hud::Step s(hud, {performBuild ? "Executing" : "Executing (Cached)", finalOutput});

        llvm::SmallString<256> outPath;
        llvm::SmallString<256> errPath;
        (void)llvm::sys::fs::createTemporaryFile("ark_run_out", "log", outPath);
        (void)llvm::sys::fs::createTemporaryFile("ark_run_err", "log", errPath);

        std::string errMsg;
        int ret = 0;

        {
            llvm::SmallVector<llvm::StringRef, 4> args = {finalOutput};
            const llvm::StringRef outRef(outPath), errRef(errPath);
            std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};

            llvm::SmallVector<std::string, 64> envStorage;
            llvm::SmallVector<llvm::StringRef, 64> envRefs;
            std::optional<llvm::ArrayRef<llvm::StringRef>> env = buildInheritedEnv(envStorage, envRefs);

            ret = llvm::sys::ExecuteAndWait(finalOutput, args, env, redirects, 0, 0, &errMsg);
        }

        const std::string progOut = readAllFile(outPath);
        const std::string progErr = readAllFile(errPath);
        removeQuietly(outPath);
        removeQuietly(errPath);

        if (ret == 0 && errMsg.empty()) s.ok(); else s.fail();

        if (!progOut.empty()) hud.pushLogBlock("Program stdout", progOut);
        if (!progErr.empty()) hud.pushLogBlock("Program stderr", progErr);
        if (!progOut.empty() || !progErr.empty()) hud.flush();

        if (!errMsg.empty()) hud.error(std::string("Execution failed: ") + errMsg);
        else if (ret != 0) hud.error(std::string("Execution failed: non-zero exit code ") + std::to_string(ret));

        hud.finish(ret == 0);
        std::exit(ret == 0 ? 0 : 1);
    }
}

// -----------------------------------------------------------------------------
// Subcommand Registration
// -----------------------------------------------------------------------------
void setupCompileCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("compile", "Compile an Ark project or file");

    sub->add_option("input", opts.inputFile, "Entry file or directory");
    sub->add_option("-o,--output", opts.outputFile, "Output file path");
    sub->add_flag("--bare", opts.bare, "Disable capsule security wrapper");
    sub->add_flag("--jit", opts.jit, "Run immediately in LLVM JIT (no binary emitted)");
    sub->add_flag("--keep-tmp", opts.keepTmp, "Keep temporary build artifacts");
    sub->add_flag("-v,--verbose", opts.verbose, "Enable verbose output");
    sub->add_flag("--no-seal", opts.noSeal, "Do not create a sealed capsule (emit raw binary)");
    sub->add_flag("--seal", opts.seal, "Request sealed capsule output");
    sub->add_flag("--require-seal", opts.requireSeal, "Fail if a seal key is unavailable");
    sub->add_option("--target", opts.targetTriple, "Target architecture triple");

    sub->add_flag("--emit-mlir", opts.stageCompile, "Stop after MIR is built");
    sub->add_flag("--emit-llvm-dialect", opts.stageLower, "Stop after lowering to LLVM dialect");
    sub->add_flag("--emit-llvm-ir", opts.stageTranslate, "Stop after translating to LLVM IR");

    sub->add_flag("--link-only", opts.stageLinkOnly, "Link an existing .ll from --in-ll into a binary");
    sub->add_option("--in-ll", opts.inLl, "Input LLVM IR (.ll) for --link-only");

    sub->add_option("--seal-key-hex", opts.sealKeyHex, "Capsule sealing key (64 hex chars). Overrides vault/env");
    sub->add_option("--seal-key-file", opts.sealKeyFile, "File containing capsule sealing key hex. Overrides vault/env");
    sub->add_flag("--no-vault-prompt", opts.noVaultPrompt, "Never prompt for vault password (CI-safe)");

    sub->callback([]() {
        arklang::hud::Theme theme;
        theme.verbose = opts.verbose;
        arklang::hud::Hud hud(theme, 1);

        if (opts.stageLinkOnly && opts.inLl.empty()) {
            hud.error("--link-only requires --in-ll=<path-to.ll>");
            std::exit(1);
        }

        if (opts.stageLinkOnly && (opts.stageCompile || opts.stageLower || opts.stageTranslate)) {
            hud.error("--link-only cannot be combined with emit flags");
            std::exit(1);
        }

        if (opts.seal && opts.noSeal) {
            hud.error("--seal and --no-seal cannot be used together");
            std::exit(1);
        }

        if (opts.requireSeal && opts.noSeal) {
            hud.error("--require-seal and --no-seal cannot be used together");
            std::exit(1);
        }

        if (!opts.sealKeyHex.empty() && !opts.sealKeyFile.empty()) {
            hud.error("Use only one of --seal-key-hex or --seal-key-file");
            std::exit(1);
        }

        executePipeline(false, hud);
    });
}

void setupRunCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("run", "Compile and immediately execute an Ark project");

    sub->add_option("input", opts.inputFile, "Entry file or directory");
    sub->add_flag("--bare", opts.bare, "Disable capsule security wrapper");
    sub->add_flag("--jit", opts.jit, "Run immediately in LLVM JIT (no binary emitted)");
    sub->add_flag("--keep-tmp", opts.keepTmp, "Keep temporary build artifacts");
    sub->add_flag("-v,--verbose", opts.verbose, "Enable verbose output");

    sub->add_flag("--seal", opts.seal, "Run with sealed capsule output");
    sub->add_flag("--no-seal", opts.noSeal, "Force unsealed local run (default)");
    sub->add_flag("--require-seal", opts.requireSeal, "Fail if a seal key is unavailable");

    sub->add_option("--seal-key-hex", opts.sealKeyHex, "Capsule sealing key (64 hex chars). Overrides vault/env");
    sub->add_option("--seal-key-file", opts.sealKeyFile, "File containing capsule sealing key hex. Overrides vault/env");
    sub->add_flag("--no-vault-prompt", opts.noVaultPrompt, "Never prompt for vault password (CI-safe)");

    sub->callback([]() {
        arklang::hud::Theme theme;
        theme.verbose = opts.verbose;
        arklang::hud::Hud hud(theme, 1);

        if (opts.seal && opts.noSeal) {
            hud.error("--seal and --no-seal cannot be used together");
            std::exit(1);
        }

        if (opts.requireSeal && opts.noSeal) {
            hud.error("--require-seal and --no-seal cannot be used together");
            std::exit(1);
        }

        if (!opts.sealKeyHex.empty() && !opts.sealKeyFile.empty()) {
            hud.error("Use only one of --seal-key-hex or --seal-key-file");
            std::exit(1);
        }

        executePipeline(true, hud);
    });
}

} // namespace ark::cli