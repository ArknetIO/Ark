#include <llvm/ADT/SmallString.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/ToolOutputFile.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "Frontend/ModuleRegistry.h"
#include "Integration/CapsuleBackend.h"
#include "Pipeline/Compiler.h"
#include "Pipeline/JIT.h"
#include "Pipeline/Linker.h"

#include "ark_config.h"
#include "hud.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef ARK_VERSION_STRING
  #define ARK_VERSION_STRING "dev"
#endif

#if !defined(_WIN32)
#include <unistd.h>
extern "C" {
    extern char** environ;
}
#endif

namespace cl = llvm::cl;

namespace {

static cl::OptionCategory ArkCategory("Ark Compiler Options");

static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input .ark file>"),
    cl::init(""),
    cl::cat(ArkCategory));

static cl::opt<std::string> outputFilename(
    "o",
    cl::desc("Output file"),
    cl::value_desc("filename"),
    cl::init(""),
    cl::cat(ArkCategory));

static cl::opt<bool> bareMode(
    "bare",
    cl::desc("Disable capsule security"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> useJit(
    "jit",
    cl::desc("Use LLVM JIT"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> runMode(
    "run",
    cl::desc("Compile, link, seal, and execute"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<std::string> runtimePath(
    "runtime",
    cl::desc("Path to Runtime"),
    cl::init(""),
    cl::cat(ArkCategory));

static cl::opt<std::string> llvmBinDir(
    "llvm-bin",
    cl::desc("Directory containing clang / clang++ / mlir tools"),
    cl::init(""),
    cl::cat(ArkCategory));

static cl::opt<bool> keepTmp(
    "keep-tmp",
    cl::desc("Keep temporary build artifacts"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> debugMode(
    "verbose",
    cl::desc("Verbose logging"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> stageCompile(
    "emit-mlir",
    cl::desc("Stop after MLIR is built"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> stageLower(
    "emit-llvm-dialect",
    cl::desc("Stop after lowering to LLVM dialect"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> stageTranslate(
    "emit-llvm-ir",
    cl::desc("Emit LLVM IR (.ll) and stop"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<bool> stageLinkOnly(
    "link-only",
    cl::desc("Link an existing .ll from -in-ll into a binary"),
    cl::init(false),
    cl::cat(ArkCategory));

static cl::opt<std::string> inLl(
    "in-ll",
    cl::desc("Input LLVM IR (.ll) for -link-only"),
    cl::init(""),
    cl::cat(ArkCategory));

static cl::opt<bool> stageNoSeal(
    "no-seal",
    cl::desc("Do not create a capsule even if default behavior would"),
    cl::init(false),
    cl::cat(ArkCategory));

static const std::vector<std::uint8_t> MOCK_KEY = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
};

enum class StopAfter {
    None,
    MLIR,
    LLVMDialect,
    LLVMIR
};

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

static std::string readAllFile(llvm::StringRef p) {
    auto mb = llvm::MemoryBuffer::getFile(p);
    if (!mb) return {};
    return std::string(mb.get()->getBuffer());
}

static void removeQuietly(llvm::StringRef p) {
    if (!p.empty()) {
        (void)llvm::sys::fs::remove(p);
    }
}

static void removeDirQuietly(llvm::StringRef p) {
    if (!p.empty()) {
        (void)llvm::sys::fs::remove_directories(p, /*IgnoreErrors=*/true);
    }
}

static bool pathExists(llvm::StringRef p) {
    return !p.empty() && llvm::sys::fs::exists(p);
}

static std::string joinPath(llvm::StringRef a, llvm::StringRef b) {
    llvm::SmallString<256> p(a);
    llvm::sys::path::append(p, b);
    return std::string(p.str());
}

static bool runtimeDirLooksValid(llvm::StringRef dir) {
    if (!pathExists(dir)) return false;
    if (pathExists(joinPath(dir, "include/ark_protocol.h"))) return true;
    if (pathExists(joinPath(dir, "include/ark_config.h"))) return true;
    if (pathExists(joinPath(dir, "core/async.cpp"))) return true;
    return false;
}

static bool fileLooksExecutable(llvm::StringRef p) {
    return pathExists(p);
}

static std::string getEnvVar(const char* name) {
    if (const char* v = std::getenv(name)) {
        return std::string(v);
    }
    return {};
}

static std::string getMainExecutablePath(const char* argv0) {
    void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(getMainExecutablePath));
    return llvm::sys::fs::getMainExecutable(argv0, mainAddr);
}

static std::string resolveRuntimePath(const char* argv0) {
#if defined(ARK_RUNTIME_SOURCE_DIR)
    if (runtimeDirLooksValid(ARK_RUNTIME_SOURCE_DIR)) {
        return ARK_RUNTIME_SOURCE_DIR;
    }
#endif

    if (runtimePath.getNumOccurrences() > 0 && runtimeDirLooksValid(runtimePath)) {
        return runtimePath;
    }

    if (const std::string envRuntime = getEnvVar("ARK_RUNTIME_DIR"); runtimeDirLooksValid(envRuntime)) {
        return envRuntime;
    }

    const std::string mainExe = getMainExecutablePath(argv0);
    llvm::SmallString<256> cur = llvm::sys::path::parent_path(mainExe);

    for (;;) {
        const std::string candidate = joinPath(cur, "tools/compiler/Runtime");
        if (runtimeDirLooksValid(candidate)) {
            return candidate;
        }

        const llvm::SmallString<256> parent = llvm::sys::path::parent_path(cur);
        if (parent == cur || parent.empty()) {
            break;
        }
        cur = parent;
    }

    if (runtimeDirLooksValid("tools/compiler/Runtime")) {
        return "tools/compiler/Runtime";
    }

    return {};
}

static std::string resolveStubPath(const char* argv0) {
    const std::string mainExe = getMainExecutablePath(argv0);
    llvm::SmallString<256> exeDir = llvm::sys::path::parent_path(mainExe);

    std::vector<std::string> candidates;
#if defined(_WIN32)
    candidates.push_back(joinPath(exeDir, "ark-stub.exe"));
    candidates.push_back(joinPath(exeDir, "../ark-stub.exe"));
    candidates.push_back(joinPath(exeDir, "../../ark-stub.exe"));
#else
    candidates.push_back(joinPath(exeDir, "ark-stub"));
    candidates.push_back(joinPath(exeDir, "../ark-stub"));
    candidates.push_back(joinPath(exeDir, "../../ark-stub"));
#endif

    for (const auto& c : candidates) {
        if (fileLooksExecutable(c)) {
            return c;
        }
    }

    return {};
}

static std::string resolveGpuPluginDir(const char* argv0) {
    if (const std::string envDir = getEnvVar("ARK_GPU_PLUGIN_DIR"); pathExists(envDir)) {
        return envDir;
    }

    const std::string mainExe = getMainExecutablePath(argv0);
    llvm::SmallString<256> exeDir = llvm::sys::path::parent_path(mainExe);

    const std::array<std::string, 6> candidates = {
        joinPath(exeDir, "lib"),
        joinPath(exeDir, "../lib"),
        joinPath(exeDir, "../../lib"),
        joinPath(exeDir, "Runtime/gpu"),
        joinPath(exeDir, "../Runtime/gpu"),
        joinPath(exeDir, "../../Runtime/gpu")
    };

    for (const auto& c : candidates) {
        if (pathExists(c)) {
            return c;
        }
    }

    return {};
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

static void upsertEnvVar(llvm::SmallVectorImpl<std::string>& storage,
                         llvm::SmallVectorImpl<llvm::StringRef>& refs,
                         llvm::StringRef key,
                         llvm::StringRef value) {
    const std::string prefix = (key + "=").str();

    for (auto& entry : storage) {
        if (llvm::StringRef(entry).starts_with(prefix)) {
            entry = prefix + value.str();
            refs.clear();
            refs.reserve(storage.size());
            for (auto& s : storage) {
                refs.push_back(s);
            }
            return;
        }
    }

    storage.push_back(prefix + value.str());
    refs.clear();
    refs.reserve(storage.size());
    for (auto& s : storage) {
        refs.push_back(s);
    }
}

static bool anyStageFlagsSpecified() {
    return stageCompile || stageLower || stageTranslate || stageLinkOnly;
}

static StopAfter computeStopAfter() {
    if (!anyStageFlagsSpecified()) return StopAfter::None;
    if (stageLinkOnly) return StopAfter::None;
    if (stageTranslate) return StopAfter::LLVMIR;
    if (stageLower) return StopAfter::LLVMDialect;
    if (stageCompile) return StopAfter::MLIR;
    return StopAfter::None;
}

static std::string defaultStageOutputPath(llvm::StringRef input, StopAfter stopAfter) {
    const llvm::StringRef stem = llvm::sys::path::stem(input);

    switch (stopAfter) {
        case StopAfter::MLIR:
            return (stem + ".mlir").str();
        case StopAfter::LLVMDialect:
            return (stem + ".llvm.mlir").str();
        case StopAfter::LLVMIR:
            return (stem + ".ll").str();
        case StopAfter::None:
            break;
    }

#if defined(_WIN32)
    return "a.exe";
#else
    return "a.out";
#endif
}

static std::string defaultBinaryOutputPath() {
#if defined(_WIN32)
    return "a.exe";
#else
    return "a.out";
#endif
}

static std::string tempBinarySuffix() {
#if defined(_WIN32)
    return "exe";
#else
    return "bin";
#endif
}

static int computeSteps(bool useJitLocal,
                        bool wantCapsule,
                        bool runLocal,
                        StopAfter stopAfter,
                        bool linkOnly) {
    int steps = 0;

    if (linkOnly) {
        steps += 1;
        if (wantCapsule && !stageNoSeal) steps += 1;
        if (runLocal) steps += 1;
        return steps;
    }

    steps += 1;
    if (stopAfter == StopAfter::MLIR) return steps;

    steps += 1;
    if (stopAfter == StopAfter::LLVMDialect) return steps;

    if (useJitLocal) {
        steps += 1;
        return steps;
    }

    steps += 1;
    if (stopAfter == StopAfter::LLVMIR) return steps;

    steps += 1;
    if (wantCapsule && !stageNoSeal) steps += 1;
    if (runLocal) steps += 1;
    return steps;
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

static void bestEffortAuditFatbinExport(arklang::hud::Hud& hud,
                                        llvm::StringRef binaryPath,
                                        bool enabled) {
    if (!enabled) return;

#if defined(_WIN32)
    return;
#else
    auto nm = llvm::sys::findProgramByName("nm");
    if (!nm) return;

    llvm::SmallString<256> outPath;
    llvm::SmallString<256> errPath;
    (void)llvm::sys::fs::createTemporaryFile("ark_nm_out", "log", outPath);
    (void)llvm::sys::fs::createTemporaryFile("ark_nm_err", "log", errPath);

    llvm::SmallVector<llvm::StringRef, 8> argv;
    argv.push_back(*nm);
  #if defined(__APPLE__)
    argv.push_back("-gU");
  #else
    argv.push_back("-D");
  #endif
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
#endif
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init(argc, argv);
    cl::HideUnrelatedOptions(ArkCategory);
    cl::ParseCommandLineOptions(argc, argv, "Ark Compiler\n");

    if (stageLinkOnly && inLl.empty()) {
        llvm::errs() << "error: -link-only requires -in-ll=<path-to.ll>\n";
        return 1;
    }

    if (stageLinkOnly && (stageCompile || stageLower || stageTranslate)) {
        llvm::errs() << "error: -link-only cannot be combined with -emit-mlir/-emit-llvm-dialect/-emit-llvm-ir\n";
        return 1;
    }

    if (!stageLinkOnly && inputFilename.empty()) {
        llvm::errs() << "error: missing input .ark file\n";
        return 1;
    }

    const StopAfter stopAfter = computeStopAfter();

    const bool wantCapsuleDefault = (runMode && !bareMode) ? true : (!bareMode && !useJit);
    const bool wantCapsule = wantCapsuleDefault && !stageNoSeal;

    std::string finalOutput = outputFilename;
    bool isTempRunOutput = false;

    if (runMode && finalOutput.empty()) {
        llvm::SmallString<256> tmp;
        if (llvm::sys::fs::createTemporaryFile("ark_run", tempBinarySuffix(), tmp)) {
            llvm::errs() << "error: failed to allocate temporary run output\n";
            return 1;
        }
        finalOutput = std::string(tmp.str());
        isTempRunOutput = true;
    } else if (finalOutput.empty() && stopAfter == StopAfter::None) {
        finalOutput = defaultBinaryOutputPath();
    }

    arklang::hud::Theme theme;
    theme.verbose = debugMode;

    const int totalSteps = computeSteps(useJit, wantCapsuleDefault, runMode, stopAfter, stageLinkOnly);
    arklang::hud::Hud hud(theme, totalSteps);
    hud.setLabel("arkc");

    if (debugMode) {
        hud.banner("arkc", ARK_VERSION_STRING, wantCapsuleDefault ? "Secure Pipeline" : "Raw Pipeline");
    }

    const std::string validRuntime = resolveRuntimePath(argv[0]);
    if (validRuntime.empty() || !pathExists(validRuntime)) {
        hud.error("Runtime not found. Set --runtime or ARK_RUNTIME_DIR, or run from a valid repo/build layout.");
        hud.finish(false);
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
    }

    arklang::ModuleRegistry registry;

    ark::compiler::pipeline::CompilerConfig compilerCfg{};
    ark::compiler::pipeline::Compiler compiler(hud, registry, compilerCfg);

    ark::compiler::pipeline::LinkerConfig linkCfg{};
    linkCfg.runtimePath = validRuntime;
    linkCfg.llvmBinDir = llvmBinDir;
    linkCfg.keepTmp = keepTmp;

#if defined(_WIN32) && !defined(__MINGW32__)
    linkCfg.toolchain = ark::compiler::pipeline::ToolchainKind::MSVC;
#elif defined(__MINGW32__)
    linkCfg.toolchain = ark::compiler::pipeline::ToolchainKind::MinGW;
#elif defined(__APPLE__)
    linkCfg.toolchain = ark::compiler::pipeline::ToolchainKind::Apple;
#else
    linkCfg.toolchain = ark::compiler::pipeline::ToolchainKind::Generic;
#endif

#if defined(ARK_DEFAULT_LLVM_BIN_DIR)
    linkCfg.llvmBinDir = llvmBinDir.empty() ? std::string(ARK_DEFAULT_LLVM_BIN_DIR) : llvmBinDir;
#else
    linkCfg.llvmBinDir = llvmBinDir;
#endif

    ark::compiler::pipeline::Linker linker(hud, linkCfg);

    // Keep main as a pure driver. Compiler.cpp owns MLIR bootstrap.
    mlir::MLIRContext ctx;
    ctx.disableMultithreading();

    mlir::ModuleOp module;
    std::vector<ark::compiler::pipeline::CompiledGpuModule> gpuMods;

    if (stageLinkOnly) {
        const std::string tmpDir = linker.makeTempDir();
        if (keepTmp) hud.note(std::string("Build Temp: ") + tmpDir);

        const std::string binaryPath = wantCapsule ? joinPath(tmpDir, "payload.bin") : finalOutput;

        if (!runStep(hud, {"Linking", "LLVM IR → Native Binary"}, [&] {
                return linker.linkToBinary(inLl, binaryPath, gpuMods);
            })) {
            hud.finish(false);
            if (!keepTmp) removeDirQuietly(tmpDir);
            if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
            return 1;
        }

        bestEffortAuditFatbinExport(hud, binaryPath, debugMode && !gpuMods.empty());

        if (wantCapsule) {
            arklang::hud::Step s(hud, {"Sealing", "Creating Self-Executing Capsule"});

            const std::string stubPath = resolveStubPath(argv[0]);
            if (stubPath.empty()) {
                hud.error("Stub not found");
                s.fail();
                hud.finish(false);
                if (!keepTmp) removeDirQuietly(tmpDir);
                if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
                return 1;
            }

            llvm::SmallString<256> srcDir(".");
            bool ok = false;
            std::uint64_t sealedSize = 0;

            {
                arklang::hud::Hud::ScopedDiagnostics diag(hud);
                ok = ark::compiler::integration::CapsuleBackend::CreateSelfExecutingCapsule(
                    hud,
                    std::string(srcDir.str()),
                    stubPath,
                    binaryPath,
                    finalOutput,
                    MOCK_KEY,
                    &sealedSize
                );
            }

            if (!ok) {
                s.fail();
                hud.finish(false);
                if (!keepTmp) removeDirQuietly(tmpDir);
                if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
                return 1;
            }

            (void)llvm::sys::fs::setPermissions(
                finalOutput,
                llvm::sys::fs::owner_exe | llvm::sys::fs::owner_read | llvm::sys::fs::owner_write
            );

            s.ok();
            hud.pushLogBlock("Sealed", finalOutput + " (" + fmtBytes(sealedSize) + ")");
            hud.flush();

            if (!keepTmp) removeQuietly(binaryPath);
        }

        if (runMode) {
            arklang::hud::Step s(hud, {"Executing", finalOutput});

            llvm::SmallString<256> outPath;
            llvm::SmallString<256> errPath;
            (void)llvm::sys::fs::createTemporaryFile("ark_run_out", "log", outPath);
            (void)llvm::sys::fs::createTemporaryFile("ark_run_err", "log", errPath);

            std::string errMsg;
            int ret = 0;

            {
                arklang::hud::Hud::ScopedDiagnostics diag(hud);
                llvm::SmallVector<llvm::StringRef, 4> args = {finalOutput};

                const llvm::StringRef outRef(outPath);
                const llvm::StringRef errRef(errPath);
                std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};

                llvm::SmallVector<std::string, 256> envStorage;
                llvm::SmallVector<llvm::StringRef, 256> envRefs;
                std::optional<llvm::ArrayRef<llvm::StringRef>> env = buildInheritedEnv(envStorage, envRefs);

                if (const std::string pluginDir = resolveGpuPluginDir(argv[0]); !pluginDir.empty()) {
                    if (!env.has_value()) {
                        envStorage.clear();
                        envRefs.clear();
                        env = llvm::ArrayRef<llvm::StringRef>(envRefs);
                    }
                    upsertEnvVar(envStorage, envRefs, "ARK_GPU_PLUGIN_DIR", pluginDir);
                    env = llvm::ArrayRef<llvm::StringRef>(envRefs);
                }

                ret = llvm::sys::ExecuteAndWait(finalOutput, args, env, redirects, 0, 0, &errMsg);
            }

            const std::string progOut = readAllFile(outPath);
            const std::string progErr = readAllFile(errPath);

            removeQuietly(outPath);
            removeQuietly(errPath);

            if (ret == 0 && errMsg.empty()) s.ok();
            else s.fail();

            if (!progOut.empty()) hud.pushLogBlock("Program stdout", progOut);
            if (!progErr.empty()) hud.pushLogBlock("Program stderr", progErr);
            if (!progOut.empty() || !progErr.empty()) hud.flush();

            if (!errMsg.empty()) hud.error(std::string("Execution failed: ") + errMsg);
            else if (ret != 0) hud.error(std::string("Execution failed: non-zero exit code ") + std::to_string(ret));

            if (!keepTmp) removeQuietly(finalOutput);

            hud.finish(ret == 0);
            return ret == 0 ? 0 : 1;
        }

        if (!keepTmp) removeDirQuietly(tmpDir);
        hud.finish(true);
        return 0;
    }

    if (!runStep(hud, {"Compiling", "Source → MIR"}, [&] {
            return compiler.compileToMLIR(inputFilename, ctx, module);
        })) {
        hud.finish(false);
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
    }

    if (stopAfter == StopAfter::MLIR) {
        const std::string outPath = outputFilename.empty()
            ? defaultStageOutputPath(inputFilename, StopAfter::MLIR)
            : outputFilename;

        if (!compiler.writeModuleToFile(module, outPath)) {
            hud.error(std::string("Failed to write MLIR to: ") + outPath);
            hud.finish(false);
            return 1;
        }

        hud.note(std::string("wrote MLIR: ") + outPath);
        hud.finish(true);
        return 0;
    }

    if (!runStep(hud, {"Lowering", "MIR → LLVM Dialect"}, [&] {
            const bool ok = compiler.lowerToLLVM(ctx, module);
            if (!ok) return false;

            if (stopAfter != StopAfter::LLVMDialect) {
                gpuMods = compiler.extractCompiledGpuModules(module);
                if (debugMode) {
                    hud.note(std::string("GPU modules captured: ") + std::to_string(gpuMods.size()));
                    for (const auto& m : gpuMods) {
                        hud.note(std::string("  gpu: key=") + m.moduleKey +
                                 " kind=" + std::to_string(m.moduleKind) +
                                 " bytes=" + std::to_string(m.blob.size()) +
                                 " kernels=" + std::to_string(m.kernels.size()));
                    }
                }
            } else if (debugMode) {
                hud.note("stop after LLVM dialect: preserving ark.gpu.modules in MLIR");
            }

            return true;
        })) {
        hud.finish(false);
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
    }

    if (stopAfter == StopAfter::LLVMDialect) {
        const std::string outPath = outputFilename.empty()
            ? defaultStageOutputPath(inputFilename, StopAfter::LLVMDialect)
            : outputFilename;

        if (!compiler.writeModuleToFile(module, outPath)) {
            hud.error(std::string("Failed to write LLVM dialect MLIR to: ") + outPath);
            hud.finish(false);
            return 1;
        }

        hud.note(std::string("wrote LLVM dialect MLIR: ") + outPath);
        hud.finish(true);
        return 0;
    }

    if (useJit) {
        arklang::hud::Step s(hud, {"JIT Execution", "Running in-memory"});

        int rc = 0;
        {
            arklang::hud::Hud::ScopedDiagnostics diag(hud);
            rc = ark::compiler::pipeline::JIT::Run(module, validRuntime);
        }

        if (rc == 0) s.ok();
        else s.fail();

        hud.finish(rc == 0);
        return rc;
    }

    const std::string tmpDir = linker.makeTempDir();
    if (keepTmp) hud.note(std::string("Build Temp: ") + tmpDir);

    const std::string binaryPath = wantCapsule ? joinPath(tmpDir, "payload.bin") : finalOutput;
    const std::string tmpLl =
        (stopAfter == StopAfter::LLVMIR)
            ? (outputFilename.empty() ? joinPath(tmpDir, "out.ll") : outputFilename)
            : joinPath(tmpDir, "out.ll");

    if (!runStep(hud, {"Translating", "MLIR → LLVM IR (In-Process)"}, [&] {
            llvm::LLVMContext llvmContext;
            auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
            if (!llvmModule) {
                hud.error("Translation to LLVM IR failed internally.");
                return false;
            }

            std::error_code ec;
            llvm::ToolOutputFile out(tmpLl, ec, llvm::sys::fs::OF_None);
            if (ec) {
                hud.error(std::string("Failed to open output file: ") + tmpLl);
                return false;
            }

            llvmModule->print(out.os(), nullptr);
            out.keep();
            return true;
        })) {
        hud.finish(false);
        if (!keepTmp) {
            removeQuietly(tmpLl);
            removeDirQuietly(tmpDir);
        }
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
    }

    if (stopAfter == StopAfter::LLVMIR) {
        if (debugMode) hud.note(std::string("wrote LLVM IR: ") + tmpLl);
        hud.finish(true);
        if (!keepTmp && outputFilename.empty()) removeDirQuietly(tmpDir);
        return 0;
    }

    if (!runStep(hud, {"Linking", "LLVM IR → Native Binary"}, [&] {
            return linker.linkToBinary(tmpLl, binaryPath, gpuMods);
        })) {
        hud.finish(false);
        if (!keepTmp) {
            removeQuietly(tmpLl);
            removeDirQuietly(tmpDir);
        }
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
    }

    bestEffortAuditFatbinExport(hud, binaryPath, debugMode && !gpuMods.empty());

    if (wantCapsule) {
        arklang::hud::Step s(hud, {"Sealing", "Creating Self-Executing Capsule"});

        const std::string stubPath = resolveStubPath(argv[0]);
        if (stubPath.empty()) {
            hud.error("Stub not found");
            s.fail();
            hud.finish(false);
            if (!keepTmp) {
                removeQuietly(tmpLl);
                removeDirQuietly(tmpDir);
            }
            if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
            return 1;
        }

        llvm::SmallString<256> srcDir = llvm::sys::path::parent_path(inputFilename);
        if (srcDir.empty()) {
            srcDir = llvm::StringRef(".");
        }

        bool ok = false;
        std::uint64_t sealedSize = 0;

        {
            arklang::hud::Hud::ScopedDiagnostics diag(hud);
            ok = ark::compiler::integration::CapsuleBackend::CreateSelfExecutingCapsule(
                hud,
                std::string(srcDir.str()),
                stubPath,
                binaryPath,
                finalOutput,
                MOCK_KEY,
                &sealedSize
            );
        }

        if (!ok) {
            s.fail();
            hud.finish(false);
            if (!keepTmp) {
                removeQuietly(tmpLl);
                removeDirQuietly(tmpDir);
            }
            if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
            return 1;
        }

        (void)llvm::sys::fs::setPermissions(
            finalOutput,
            llvm::sys::fs::owner_exe | llvm::sys::fs::owner_read | llvm::sys::fs::owner_write
        );

        s.ok();
        hud.pushLogBlock("Sealed", finalOutput + " (" + fmtBytes(sealedSize) + ")");
        hud.flush();

        if (!keepTmp) removeQuietly(binaryPath);
    } else if (wantCapsuleDefault && stageNoSeal) {
        if (!outputFilename.empty() && finalOutput != binaryPath) {
            std::error_code ec = llvm::sys::fs::copy_file(binaryPath, finalOutput);
            if (ec) {
                hud.error(std::string("Failed to copy raw binary to output: ") + ec.message());
                hud.finish(false);
                if (!keepTmp) {
                    removeQuietly(tmpLl);
                    removeDirQuietly(tmpDir);
                }
                if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
                return 1;
            }
        }
    }

    if (runMode) {
        arklang::hud::Step s(hud, {"Executing", finalOutput});

        llvm::SmallString<256> outPath;
        llvm::SmallString<256> errPath;
        (void)llvm::sys::fs::createTemporaryFile("ark_run_out", "log", outPath);
        (void)llvm::sys::fs::createTemporaryFile("ark_run_err", "log", errPath);

        std::string errMsg;
        int ret = 0;

        {
            arklang::hud::Hud::ScopedDiagnostics diag(hud);
            llvm::SmallVector<llvm::StringRef, 4> args = {finalOutput};

            const llvm::StringRef outRef(outPath);
            const llvm::StringRef errRef(errPath);
            std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};

            llvm::SmallVector<std::string, 256> envStorage;
            llvm::SmallVector<llvm::StringRef, 256> envRefs;
            std::optional<llvm::ArrayRef<llvm::StringRef>> env = buildInheritedEnv(envStorage, envRefs);

            if (const std::string pluginDir = resolveGpuPluginDir(argv[0]); !pluginDir.empty()) {
                if (!env.has_value()) {
                    envStorage.clear();
                    envRefs.clear();
                    env = llvm::ArrayRef<llvm::StringRef>(envRefs);
                }
                upsertEnvVar(envStorage, envRefs, "ARK_GPU_PLUGIN_DIR", pluginDir);
                env = llvm::ArrayRef<llvm::StringRef>(envRefs);
            }

            ret = llvm::sys::ExecuteAndWait(finalOutput, args, env, redirects, 0, 0, &errMsg);
        }

        const std::string progOut = readAllFile(outPath);
        const std::string progErr = readAllFile(errPath);

        removeQuietly(outPath);
        removeQuietly(errPath);

        if (ret == 0 && errMsg.empty()) s.ok();
        else s.fail();

        if (!progOut.empty()) hud.pushLogBlock("Program stdout", progOut);
        if (!progErr.empty()) hud.pushLogBlock("Program stderr", progErr);
        if (!progOut.empty() || !progErr.empty()) hud.flush();

        if (!errMsg.empty()) hud.error(std::string("Execution failed: ") + errMsg);
        else if (ret != 0) hud.error(std::string("Execution failed: non-zero exit code ") + std::to_string(ret));

        if (!keepTmp) removeQuietly(finalOutput);

        hud.finish(ret == 0);
        return ret == 0 ? 0 : 1;
    }

    if (!keepTmp) {
        removeQuietly(tmpLl);
        removeDirQuietly(tmpDir);
    }

    hud.finish(true);
    return 0;
}