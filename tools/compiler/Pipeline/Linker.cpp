#include "Pipeline/Linker.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace ark::compiler::pipeline {

// ---------------------------------------------
// Small file helpers
// ---------------------------------------------
static std::optional<std::string> readFileToString(llvm::StringRef path) {
    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) return std::nullopt;
    return std::string((*buf)->getBuffer());
}

static void removeDirTree(llvm::StringRef path) {
    if (path.empty()) return;
    (void)llvm::sys::fs::remove_directories(path, /*IgnoreErrors=*/true);
}

// ---------------------------------------------
// Toolchain helpers
// ---------------------------------------------
static bool looksLikeClangDriver(llvm::StringRef exe) {
    const llvm::StringRef stem = llvm::sys::path::stem(exe);
    return stem.contains("clang");
}

static std::string msvcJoinedPathArg(const std::string& flag, const std::string& path) {
    const bool needsQuotes =
        (path.find(' ') != std::string::npos) || (path.find('\t') != std::string::npos);
    if (needsQuotes) return flag + "\"" + path + "\"";
    return flag + path;
}

static std::string escapeForAsm(const std::string& path) {
    std::string out;
    out.reserve(path.size());
    for (char c : path) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    return out;
}

static bool endsWithAnyCpp(const std::vector<std::string>& files) {
    for (const auto& f : files) {
        if (llvm::StringRef(f).ends_with(".cpp")) return true;
    }
    return false;
}

// ---------------------------------------------
// Linker
// ---------------------------------------------
Linker::Linker(arklang::hud::Hud& hud, const LinkerConfig& config)
    : hud_(hud), cfg_(config) {}

std::vector<std::string> Linker::getRuntimeSources() const {
    // Core runtime + GPU HAL. Vendor GPU backends are plugins (dlopen), not linked here.
    std::vector<std::string> v = {
        "core/async.cpp",
        "core/memory.cpp",
        "core/string.cpp",
        "core/panic.cpp",
        "core/print.cpp",
        "core/hash.cpp",
        "core/vector.cpp",
        "core/gpu_hal.cpp",
        "net/remote.cpp",
    };

    if (cfg_.toolchain == ToolchainKind::MSVC || cfg_.toolchain == ToolchainKind::MinGW) {
        v.push_back("fs/fs_win32.cpp");
        v.push_back("net/socket_win32.cpp");
        // v.push_back("sys/sys_win32.cpp"); // (Add this later when you build Windows support!)
    } else {
        v.push_back("fs/fs_posix.cpp");
        v.push_back("net/socket_posix.cpp");
        
        // [FIX] Tell the AOT Linker to compile our new SYS kernel!
        v.push_back("sys/sys_posix.cpp"); 
    }

    return v;
}


std::string Linker::findTool(const std::string& name) {
    if (name == "clang" && !cfg_.clangOverride.empty()) {
        if (llvm::sys::fs::exists(cfg_.clangOverride)) return cfg_.clangOverride;
    }
    if (name == "mlir-translate" && !cfg_.mlirTranslateOverride.empty()) {
        if (llvm::sys::fs::exists(cfg_.mlirTranslateOverride)) return cfg_.mlirTranslateOverride;
    }

    if (!cfg_.llvmBinDir.empty()) {
        llvm::SmallString<256> p(cfg_.llvmBinDir);
        llvm::sys::path::append(p, name);
        if (llvm::sys::fs::exists(p)) return std::string(p.str());
    }

    if (auto found = llvm::sys::findProgramByName(name)) return *found;
    return "";
}

std::string Linker::findCompilerExe() {
    if (cfg_.toolchain == ToolchainKind::MSVC) {
        if (!cfg_.clangOverride.empty() && llvm::sys::fs::exists(cfg_.clangOverride)) return cfg_.clangOverride;
        if (auto p = llvm::sys::findProgramByName("clang-cl")) return *p;
        return "";
    }
    return findTool("clang");
}

// Runs a child tool and captures stdout/stderr into HUD blocks.
// This is used for: mlir-translate, clang/clang++ link driver, etc.
int Linker::runCmd(const std::string& exe, const std::vector<std::string>& args, const std::string& label) {
    llvm::SmallVector<llvm::StringRef, 32> argv;
    argv.push_back(exe);
    for (const auto& a : args) argv.push_back(a);

    llvm::SmallString<256> outPath;
    llvm::SmallString<256> errPath;
    (void)llvm::sys::fs::createTemporaryFile("ark-link", "out", outPath);
    (void)llvm::sys::fs::createTemporaryFile("ark-link", "err", errPath);

    const llvm::StringRef outRef(outPath);
    const llvm::StringRef errRef(errPath);

    std::optional<llvm::StringRef> redirects[] = {
        std::nullopt, // stdin
        outRef,       // stdout
        errRef        // stderr
    };

    std::string errMsg;
    const int rc = llvm::sys::ExecuteAndWait(exe, argv, std::nullopt, redirects, 0, 0, &errMsg);

    const std::string stdoutText = readFileToString(outRef).value_or("");
    const std::string stderrText = readFileToString(errRef).value_or("");

    if (!cfg_.keepTmp) {
        (void)llvm::sys::fs::remove(outPath);
        (void)llvm::sys::fs::remove(errPath);
    } else {
        if (!stdoutText.empty() || !stderrText.empty()) {
            hud_.debug("Preserved temp logs: " + std::string(outPath) + ", " + std::string(errPath));
        }
    }

    if (!stdoutText.empty()) hud_.pushLogBlock(label + " stdout", stdoutText);
    if (!stderrText.empty()) hud_.pushLogBlock(label + " stderr", stderrText);
    if (!errMsg.empty()) hud_.pushLogBlock(label + " exec error", errMsg);

    return rc;
}

bool Linker::translateMlirToLlvmIr(const std::string& inputMlir, const std::string& outputLl) {
    std::string exe = findTool("mlir-translate");
    if (exe.empty()) {
        hud_.error("Tool not found: mlir-translate");
        return false;
    }
    const int rc = runCmd(exe, {"--mlir-to-llvmir", inputMlir, "-o", outputLl}, "mlir-translate");
    return rc == 0;
}

// Emits a source file that embeds `filePath` as:
// - MSVC: C++ array + size symbol
// - Apple: Mach-O incbin + _start/_end
// - ELF/MinGW: .incbin + start/end/size
EmbeddingResult Linker::createEmbeddingSource(const std::string& symbolPrefix, const std::string& filePath) {
    if (cfg_.toolchain == ToolchainKind::MSVC) {
        auto buf = llvm::MemoryBuffer::getFile(filePath);
        if (!buf) return {"", EmbeddingKind::Error};

        llvm::StringRef data = (*buf)->getBuffer();

        std::string s;
        s += "#include <cstdint>\n";
        s += "extern \"C\" {\n";
        s += "  __declspec(align(16)) __declspec(dllexport) const unsigned char " + symbolPrefix + "_start[] = {";

        for (size_t i = 0; i < data.size(); ++i) {
            if ((i % 16) == 0) s += "\n    ";
            char hex[8];
            std::snprintf(hex, sizeof(hex), "0x%02X,", static_cast<unsigned char>(data[i]));
            s += hex;
        }

        s += "\n  };\n";
        s += "  __declspec(dllexport) const std::uint64_t " + symbolPrefix + "_size = " +
             std::to_string(static_cast<std::uint64_t>(data.size())) + "ULL;\n";
        s += "}\n";

        return {s, EmbeddingKind::Cpp};
    }

    std::string s;
    const std::string escapedPath = escapeForAsm(filePath);

    if (cfg_.toolchain == ToolchainKind::Apple) {
        s += ".section __DATA,__const\n";
        s += ".global _" + symbolPrefix + "_start\n";
        s += ".global _" + symbolPrefix + "_end\n";
        s += ".global _" + symbolPrefix + "_size\n";
        s += ".p2align 4\n";
        s += "_" + symbolPrefix + "_start:\n";
        s += ".incbin \"" + escapedPath + "\"\n";
        s += "_" + symbolPrefix + "_end:\n";
        s += ".p2align 3\n";
        s += "_" + symbolPrefix + "_size:\n";
        s += ".quad _" + symbolPrefix + "_end - _" + symbolPrefix + "_start\n";
        return {s, EmbeddingKind::Asm};
    }

    const char* section =
        (cfg_.toolchain == ToolchainKind::MinGW) ? ".section .rdata,\"dr\"" : ".section .rodata";
    s += std::string(section) + "\n";
    s += ".global " + symbolPrefix + "_start\n";
    s += ".global " + symbolPrefix + "_end\n";
    s += ".global " + symbolPrefix + "_size\n";
    s += ".p2align 4\n";
    s += symbolPrefix + "_start:\n";
    s += ".incbin \"" + escapedPath + "\"\n";
    s += symbolPrefix + "_end:\n";
    s += ".p2align 3\n";
    s += symbolPrefix + "_size:\n";
    s += ".quad " + symbolPrefix + "_end - " + symbolPrefix + "_start\n";
    return {s, EmbeddingKind::Asm};
}

bool Linker::runtimeNeedsCxxLink() const {
    for (const auto& src : getRuntimeSources()) {
        if (llvm::StringRef(src).ends_with(".cpp")) return true;
    }
    return false;
}

std::string Linker::chooseLinkerExe(const std::string& compilerPath) {
    if (cfg_.toolchain == ToolchainKind::MSVC) return compilerPath;

    llvm::SmallString<256> p(compilerPath);
    const llvm::StringRef filename = llvm::sys::path::filename(p);
    const llvm::StringRef stem = llvm::sys::path::stem(filename);
    const llvm::StringRef ext  = llvm::sys::path::extension(filename);

    auto checkSibling = [&](llvm::StringRef newStem) -> std::string {
        llvm::SmallString<256> q(compilerPath);
        llvm::sys::path::remove_filename(q);
        llvm::SmallString<128> newName(newStem);
        newName += ext;
        llvm::sys::path::append(q, newName);
        if (llvm::sys::fs::exists(q)) return std::string(q.str());
        return "";
    };

    if (stem == "clang") {
        if (auto s = checkSibling("clang++"); !s.empty()) return s;
    }
    if (stem.starts_with("clang-")) {
        llvm::SmallString<64> cxxStem(stem);
        cxxStem += "++";
        if (auto s = checkSibling(cxxStem); !s.empty()) return s;
    }

    std::string cxx = findTool("clang++");
    return cxx.empty() ? compilerPath : cxx;
}

bool Linker::linkToBinary(const std::string& inputLl, const std::string& outputBin) {
    return linkToBinary(inputLl, outputBin, {});
}

bool Linker::linkToBinary(const std::string& inputLl,
                          const std::string& outputBin,
                          const std::vector<ark::compiler::pipeline::CompiledGpuModule>& gpuMods) {
    const std::string compilerExe = findCompilerExe();
    if (compilerExe.empty()) {
        hud_.error("Compiler driver not found");
        return false;
    }

    const std::vector<std::string> sources = getRuntimeSources();

    // Always emit a fatbin registry TU:
    // - If gpuMods is empty: registry exports ark_gpu_fatbin_query_v1 and returns an empty fatbin.
    // - If gpuMods non-empty: registry also references embedded blobs and enumerates entries.
    std::vector<std::string> fatbinSources;
    std::string fatbinTmpDir;
    {
        auto art = emitGpuFatbinArtifacts(gpuMods);
        if (!art) return false;
        fatbinSources = std::move(art->extraSources);
        fatbinTmpDir  = art->tmpDir;
        if (cfg_.keepTmp) hud_.debug("GPU fatbin temp dir: " + fatbinTmpDir);
    }

    // Registry TU is always C++ => use a C++ driver.
    const bool needCxx = runtimeNeedsCxxLink() || endsWithAnyCpp(fatbinSources);
    const std::string driverExe = needCxx ? chooseLinkerExe(compilerExe) : compilerExe;

    std::vector<std::string> args;

    // Ensure ark_gpu_fatbin_query_v1 is:
    // - present in the dynamic symbol table (so dlsym(RTLD_DEFAULT) can find it)
    // - not dead-stripped by section GC / aggressive LTO
    auto addFatbinExportLinkFlags = [&](std::vector<std::string>& a) {
        if (cfg_.toolchain == ToolchainKind::MSVC) {
            a.push_back("/link");
            a.push_back("/INCLUDE:ark_gpu_fatbin_query_v1");
            return;
        }

        if (cfg_.toolchain == ToolchainKind::Apple) {
            a.push_back("-Wl,-export_dynamic");
            a.push_back("-Wl,-u,_ark_gpu_fatbin_query_v1");
            return;
        }

        a.push_back("-rdynamic");
        a.push_back("-Wl,-export-dynamic");
        a.push_back("-Wl,-u,ark_gpu_fatbin_query_v1");
        a.push_back("-Wl,--no-gc-sections");
    };

    if (cfg_.toolchain == ToolchainKind::MSVC) {
        args.push_back("/clang:-x");
        args.push_back("/clang:ir");
        args.push_back(inputLl);

        args.push_back("/std:c++20");

        args.push_back(msvcJoinedPathArg("/I", cfg_.runtimePath));
        args.push_back(msvcJoinedPathArg("/I", cfg_.runtimePath + "/include"));

        for (const auto& src : sources) {
            const std::string p = cfg_.runtimePath + "/" + src;
            if (llvm::sys::fs::exists(p)) args.push_back(p);
        }
        for (const auto& src : fatbinSources) args.push_back(src);

        args.push_back("/Od");
        args.push_back("/Zi");
        args.push_back("/MDd");

        args.push_back(msvcJoinedPathArg("/Fe:", outputBin));

        addFatbinExportLinkFlags(args);
    } else {
        args.push_back(inputLl);
        args.push_back("-std=c++20");

        args.push_back("-I");
        args.push_back(cfg_.runtimePath);
        args.push_back("-I");
        args.push_back(cfg_.runtimePath + "/include");

        for (const auto& src : sources) {
            const std::string p = cfg_.runtimePath + "/" + src;
            if (llvm::sys::fs::exists(p)) args.push_back(p);
        }
        for (const auto& src : fatbinSources) args.push_back(src);

        args.push_back("-g");
        args.push_back("-O0");

        if (cfg_.toolchain == ToolchainKind::Generic) args.push_back("-pthread");

        addFatbinExportLinkFlags(args);

        if (cfg_.toolchain == ToolchainKind::Generic) args.push_back("-ldl");

        args.push_back("-o");
        args.push_back(outputBin);

        if (looksLikeClangDriver(driverExe)) args.push_back("-Wno-override-module");
    }

    const int rc = runCmd(driverExe, args, "linker");

    // If we’re not keeping tmp artifacts, tear down the fatbin temp dir too.
    if (!cfg_.keepTmp && !fatbinTmpDir.empty()) removeDirTree(fatbinTmpDir);

    return rc == 0;
}

std::string Linker::makeTempDir() {
    llvm::SmallString<256> p;
    llvm::sys::path::system_temp_directory(true, p);
    llvm::sys::path::append(p, "arkc");
    (void)llvm::sys::fs::create_directories(p);

    llvm::SmallString<256> unique;
    (void)llvm::sys::fs::createUniqueDirectory(p, unique);
    return std::string(unique.str());
}

} // namespace ark::compiler::pipeline
