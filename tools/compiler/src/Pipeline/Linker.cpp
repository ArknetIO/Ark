#include "ark/compiler/Pipeline/Linker.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace ark::compiler::pipeline {

// ---------------------------------------------
// Small file helpers
// ---------------------------------------------
static std::optional<std::string> readFileToString(llvm::StringRef path) {
    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) {
        return std::nullopt;
    }
    return std::string((*buf)->getBuffer());
}

static void removeDirTree(llvm::StringRef path) {
    if (path.empty()) {
        return;
    }
    (void)llvm::sys::fs::remove_directories(path, true);
}

static bool fileExists(llvm::StringRef path) {
    return !path.empty() && llvm::sys::fs::exists(path);
}

static bool dirExists(llvm::StringRef path) {
    return !path.empty() && llvm::sys::fs::is_directory(path);
}

static std::string joinPath(llvm::StringRef a, llvm::StringRef b) {
    llvm::SmallString<256> p(a);
    llvm::sys::path::append(p, b);
    return std::string(p.str());
}

static std::string canonicalizePath(llvm::StringRef in) {
    if (in.empty()) {
        return {};
    }

    llvm::SmallString<256> p(in);
    std::error_code ec = llvm::sys::fs::make_absolute(p);
    if (ec) {
        llvm::sys::path::remove_dots(p, true);
        return std::string(p.str());
    }

    llvm::sys::path::remove_dots(p, true);
    return std::string(p.str());
}

static std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static bool isCppLikeSource(llvm::StringRef path) {
    const std::string ext = lowerCopy(std::string(llvm::sys::path::extension(path)));
    return ext == ".cc" || ext == ".cp" || ext == ".cpp" || ext == ".cxx" || ext == ".c++";
}

static bool isLlvmIrInput(llvm::StringRef path) {
    const std::string ext = lowerCopy(std::string(llvm::sys::path::extension(path)));
    return ext == ".ll" || ext == ".bc";
}

static bool endsWithAnyCppLike(const std::vector<std::string>& files) {
    for (const auto& f : files) {
        if (isCppLikeSource(f)) {
            return true;
        }
    }
    return false;
}

static std::string parentDirOf(llvm::StringRef path) {
    if (path.empty()) {
        return {};
    }

    llvm::SmallString<256> p(path);
    if (!llvm::sys::fs::is_directory(p)) {
        llvm::sys::path::remove_filename(p);
    }
    return std::string(p.str());
}

static std::string grandParentDirOf(llvm::StringRef path) {
    const std::string parent = parentDirOf(path);
    if (parent.empty()) {
        return {};
    }
    return parentDirOf(parent);
}

static void pushUnique(std::vector<std::string>& out, llvm::StringRef value) {
    if (value.empty()) {
        return;
    }

    const std::string s = canonicalizePath(value);
    if (s.empty()) {
        return;
    }

    if (std::find(out.begin(), out.end(), s) == out.end()) {
        out.push_back(s);
    }
}

static std::vector<std::string> executableCandidates(const std::string& name) {
#if defined(_WIN32)
    if (llvm::StringRef(name).ends_with(".exe")) {
        return {name};
    }
    return {name + ".exe", name};
#else
    return {name};
#endif
}

static std::string findProgramByNamePortable(const std::string& name) {
    for (const auto& candidate : executableCandidates(name)) {
        if (auto found = llvm::sys::findProgramByName(candidate)) {
            return *found;
        }
    }
    return {};
}

static std::string findFileInDir(llvm::StringRef dir, const std::string& name) {
    if (dir.empty()) {
        return {};
    }

    const std::string p = joinPath(dir, name);
    return fileExists(p) ? canonicalizePath(p) : std::string();
}

static bool copyFileReplace(llvm::StringRef from, llvm::StringRef to) {
    if (!fileExists(from)) {
        return false;
    }

    const std::string dstParent = parentDirOf(to);
    if (!dstParent.empty()) {
        (void)llvm::sys::fs::create_directories(dstParent);
    }

    (void)llvm::sys::fs::remove(to);
    return !llvm::sys::fs::copy_file(from, to);
}

// ---------------------------------------------
// Toolchain helpers
// ---------------------------------------------
static bool looksLikeClangDriver(llvm::StringRef exe) {
    const llvm::StringRef stem = llvm::sys::path::stem(exe);
    return lowerCopy(std::string(stem)).find("clang") != std::string::npos;
}

static bool looksLikeClangCl(llvm::StringRef exe) {
    const std::string stem = lowerCopy(std::string(llvm::sys::path::stem(exe)));
    return stem == "clang-cl";
}

static bool pathLooksLikeVcpkgBundledLlvm(llvm::StringRef path) {
    const std::string s = lowerCopy(std::string(path));
#if defined(_WIN32)
    return s.find("\\vcpkg\\installed\\") != std::string::npos &&
           s.find("\\tools\\llvm\\") != std::string::npos;
#else
    return s.find("/vcpkg/installed/") != std::string::npos &&
           s.find("/tools/llvm/") != std::string::npos;
#endif
}

static std::string escapeForAsm(const std::string& path) {
    std::string out;
    out.reserve(path.size());

    for (char c : path) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else {
            out.push_back(c);
        }
    }

    return out;
}

static bool isSourceOnlyFlag(llvm::StringRef arg) {
    return
        arg == "/std:c++20" ||
        arg == "/std:c++23" ||
        arg == "/std:c++latest" ||
        arg == "/std:c++17" ||
        arg == "/EHsc" ||
        arg == "/EHs" ||
        arg == "/EHc" ||
        arg == "/GR" ||
        arg == "/GR-" ||
        arg == "/TC" ||
        arg == "/TP" ||
        arg == "/Od" ||
        arg == "/O2" ||
        arg == "/Zi" ||
        arg == "/MD" ||
        arg == "/MDd" ||
        arg == "-std=c++20" ||
        arg == "-std=c++23" ||
        arg == "-std=gnu++20" ||
        arg == "-std=gnu++23" ||
        arg == "-fexceptions" ||
        arg == "-fcxx-exceptions";
}

static std::vector<std::string> filterIrLinkFlags(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    out.reserve(in.size());

    for (size_t i = 0; i < in.size(); ++i) {
        const llvm::StringRef arg(in[i]);

        if (arg == "-x") {
            if (i + 1 < in.size()) {
                ++i;
            }
            continue;
        }

        if (arg == "/clang:-x") {
            if (i + 1 < in.size()) {
                ++i;
            }
            continue;
        }

        if (arg == "ir" || arg == "/clang:ir") {
            continue;
        }

        if (isSourceOnlyFlag(arg)) {
            continue;
        }

        out.push_back(in[i]);
    }

    return out;
}

static void appendMsvcLinkerFlag(std::vector<std::string>& args, llvm::StringRef flag) {
    args.push_back("-Xlinker");
    args.push_back(flag.str());
}

static void appendMsvcDefaultLib(std::vector<std::string>& args, llvm::StringRef libName) {
    appendMsvcLinkerFlag(args, ("/DEFAULTLIB:" + libName).str());
}

static void appendFatbinExportFlags(std::vector<std::string>& args, ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::MSVC) {
        appendMsvcLinkerFlag(args, "/INCLUDE:ark_gpu_fatbin_query_v1");
        return;
    }

    if (toolchain == ToolchainKind::Apple) {
        args.push_back("-Wl,-export_dynamic");
        args.push_back("-Wl,-u,_ark_gpu_fatbin_query_v1");
        return;
    }

    args.push_back("-rdynamic");
    args.push_back("-Wl,-export-dynamic");
    args.push_back("-Wl,-u,ark_gpu_fatbin_query_v1");
    args.push_back("-Wl,--no-gc-sections");
}

static void appendPlatformLinkFlags(std::vector<std::string>& args, ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::MSVC) {
        appendMsvcDefaultLib(args, "ws2_32.lib");
        appendMsvcDefaultLib(args, "advapi32.lib");
        return;
    }

    if (toolchain == ToolchainKind::MinGW) {
        args.push_back("-lws2_32");
        args.push_back("-ladvapi32");
        return;
    }

    if (toolchain == ToolchainKind::Generic) {
        args.push_back("-pthread");
        args.push_back("-ldl");
        return;
    }

    if (toolchain == ToolchainKind::Apple) {
        return;
    }
}

static void appendRuntimeRPathFlags(std::vector<std::string>& args, ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::Apple) {
        args.push_back("-Wl,-rpath,@loader_path");
        return;
    }

    if (toolchain == ToolchainKind::Generic || toolchain == ToolchainKind::MinGW) {
        args.push_back("-Wl,-rpath,$ORIGIN");
    }
}

static std::vector<std::string> configuredToolRoots(const LinkerConfig& cfg) {
    std::vector<std::string> roots;

    if (!cfg.toolchainRoot.empty()) {
        pushUnique(roots, joinPath(cfg.toolchainRoot, "bin"));
    }

    if (!cfg.llvmBinDir.empty()) {
        pushUnique(roots, cfg.llvmBinDir);
    }

    if (!cfg.clangOverride.empty()) {
        pushUnique(roots, parentDirOf(cfg.clangOverride));
    }

    if (!cfg.mlirTranslateOverride.empty()) {
        pushUnique(roots, parentDirOf(cfg.mlirTranslateOverride));
    }

    return roots;
}

static std::string findNamedToolInRoots(const std::vector<std::string>& roots, const std::string& name) {
    for (const auto& root : roots) {
        for (const auto& candidate : executableCandidates(name)) {
            if (auto p = findFileInDir(root, candidate); !p.empty()) {
                return p;
            }
        }
    }
    return {};
}

static bool hasConfiguredToolRoots(const LinkerConfig& cfg) {
    return
        !cfg.toolchainRoot.empty() ||
        !cfg.llvmBinDir.empty() ||
        !cfg.clangOverride.empty() ||
        !cfg.mlirTranslateOverride.empty();
}

static std::string findConfiguredCompilerInRoots(const std::vector<std::string>& roots,
                                                 ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::MSVC) {
        if (auto p = findNamedToolInRoots(roots, "clang++"); !p.empty()) return p;
        if (auto p = findNamedToolInRoots(roots, "clang"); !p.empty()) return p;
        if (auto p = findNamedToolInRoots(roots, "clang-cl"); !p.empty()) return p;
        return {};
    }

    if (auto p = findNamedToolInRoots(roots, "clang++"); !p.empty()) return p;
    if (auto p = findNamedToolInRoots(roots, "clang"); !p.empty()) return p;
    return {};
}

static std::string chooseWindowsIrDriver(const std::string& compilerPath,
                                         const LinkerConfig& cfg,
                                         bool needCxx) {
    if (compilerPath.empty()) {
        return {};
    }

    if (!looksLikeClangCl(compilerPath)) {
        return compilerPath;
    }

    llvm::SmallString<256> dir(compilerPath);
    llvm::sys::path::remove_filename(dir);

    auto sibling = [&](const char* name) -> std::string {
        return findFileInDir(dir, name);
    };

    if (needCxx) {
        if (auto p = sibling("clang++.exe"); !p.empty()) return p;
        if (auto p = sibling("clang++.bat"); !p.empty()) return p;
    }

    if (auto p = sibling("clang.exe"); !p.empty()) return p;
    if (auto p = sibling("clang.bat"); !p.empty()) return p;

    const std::vector<std::string> roots = configuredToolRoots(cfg);

    if (needCxx) {
        if (auto p = findNamedToolInRoots(roots, "clang++"); !p.empty()) return p;
    }

    if (auto p = findNamedToolInRoots(roots, "clang"); !p.empty()) return p;

    if (needCxx) {
        if (auto p = findProgramByNamePortable("clang++"); !p.empty()) return p;
    }

    if (auto p = findProgramByNamePortable("clang"); !p.empty()) return p;
    return {};
}

static std::string defaultStaticRuntimeLibName(ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::MSVC) {
        return "ark_runtime_static.lib";
    }
    if (toolchain == ToolchainKind::Apple) {
        return "libark_runtime.a";
    }
    if (toolchain == ToolchainKind::MinGW) {
        return "libark_runtime.a";
    }
    return "libark_runtime.a";
}

static std::string defaultSharedRuntimeLibName(ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::MSVC) {
        return "ark_runtime.dll";
    }
    if (toolchain == ToolchainKind::Apple) {
        return "libark_runtime.dylib";
    }
    if (toolchain == ToolchainKind::MinGW) {
        return "libark_runtime.dll";
    }
    return "libark_runtime.so";
}

static std::string defaultImportRuntimeLibName(ToolchainKind toolchain) {
    if (toolchain == ToolchainKind::MSVC) {
        return "ark_runtime.lib";
    }
    if (toolchain == ToolchainKind::MinGW) {
        return "libark_runtime.dll.a";
    }
    return {};
}

// ---------------------------------------------
// Linker
// ---------------------------------------------
Linker::Linker(arklang::hud::Hud& hud, const LinkerConfig& config)
    : hud_(hud), cfg_(config) {}

std::optional<ToolchainLayout> Linker::resolveToolchainLayout() const {
    ToolchainLayout layout;

    if (!cfg_.toolchainRoot.empty()) {
        layout.rootDir = canonicalizePath(cfg_.toolchainRoot);
    } else if (!cfg_.llvmBinDir.empty()) {
        layout.rootDir = canonicalizePath(parentDirOf(cfg_.llvmBinDir));
    } else if (!cfg_.clangOverride.empty()) {
        layout.rootDir = canonicalizePath(grandParentDirOf(cfg_.clangOverride));
    } else if (!cfg_.mlirTranslateOverride.empty()) {
        layout.rootDir = canonicalizePath(grandParentDirOf(cfg_.mlirTranslateOverride));
    }

    if (!cfg_.llvmBinDir.empty()) {
        layout.binDir = canonicalizePath(cfg_.llvmBinDir);
    } else if (!layout.rootDir.empty()) {
        layout.binDir = canonicalizePath(joinPath(layout.rootDir, "bin"));
    }

    if (!cfg_.runtimeLibDirOverride.empty()) {
        layout.libDir = canonicalizePath(cfg_.runtimeLibDirOverride);
    } else if (!layout.rootDir.empty()) {
        layout.libDir = canonicalizePath(joinPath(layout.rootDir, "lib"));
    }

    if (!cfg_.runtimeIncludeDirOverride.empty()) {
        layout.includeDir = canonicalizePath(cfg_.runtimeIncludeDirOverride);
    } else if (!layout.rootDir.empty()) {
        layout.includeDir = canonicalizePath(joinPath(layout.rootDir, "include"));
    }

    if (!cfg_.runtimeBinDirOverride.empty()) {
        layout.runtimeBinDir = canonicalizePath(cfg_.runtimeBinDirOverride);
    } else if (!layout.binDir.empty()) {
        layout.runtimeBinDir = layout.binDir;
    }

    if (!cfg_.clangOverride.empty() && fileExists(cfg_.clangOverride)) {
        layout.clangPath = canonicalizePath(cfg_.clangOverride);
    } else if (!layout.binDir.empty()) {
        layout.clangPath = findNamedToolInRoots({layout.binDir}, "clang");
    }

    if (!layout.binDir.empty()) {
        layout.clangxxPath = findNamedToolInRoots({layout.binDir}, "clang++");
        layout.clangClPath = findNamedToolInRoots({layout.binDir}, "clang-cl");
    }

    if (!cfg_.mlirTranslateOverride.empty() && fileExists(cfg_.mlirTranslateOverride)) {
        layout.mlirTranslatePath = canonicalizePath(cfg_.mlirTranslateOverride);
    } else if (!layout.binDir.empty()) {
        layout.mlirTranslatePath = findNamedToolInRoots({layout.binDir}, "mlir-translate");
    }

    if (!layout.binDir.empty()) {
        if (cfg_.toolchain == ToolchainKind::MSVC) {
            layout.lldPath = findNamedToolInRoots({layout.binDir}, "lld-link");
        } else {
            if (auto p = findNamedToolInRoots({layout.binDir}, "ld.lld"); !p.empty()) layout.lldPath = p;
            else layout.lldPath = findNamedToolInRoots({layout.binDir}, "lld");
        }
    }

    if (!cfg_.runtimeStaticLibOverride.empty()) {
        layout.runtimeStaticLibPath = canonicalizePath(cfg_.runtimeStaticLibOverride);
    } else if (!layout.libDir.empty()) {
        layout.runtimeStaticLibPath = findFileInDir(layout.libDir, defaultStaticRuntimeLibName(cfg_.toolchain));
    }

    if (!cfg_.runtimeSharedLibOverride.empty()) {
        layout.runtimeSharedLibPath = canonicalizePath(cfg_.runtimeSharedLibOverride);
    } else {
        const std::string dir =
            !layout.runtimeBinDir.empty() ? layout.runtimeBinDir : layout.libDir;
        if (!dir.empty()) {
            layout.runtimeSharedLibPath = findFileInDir(dir, defaultSharedRuntimeLibName(cfg_.toolchain));
        }
    }

    if (!cfg_.runtimeImportLibOverride.empty()) {
        layout.runtimeImportLibPath = canonicalizePath(cfg_.runtimeImportLibOverride);
    } else if (!layout.libDir.empty()) {
        const std::string name = defaultImportRuntimeLibName(cfg_.toolchain);
        if (!name.empty()) {
            layout.runtimeImportLibPath = findFileInDir(layout.libDir, name);
        }
    }

    if (!cfg_.linkerConfigOverride.empty()) {
        layout.linkerConfigPath = canonicalizePath(cfg_.linkerConfigOverride);
    } else if (!layout.libDir.empty()) {
        layout.linkerConfigPath = findFileInDir(layout.libDir, "linker-config.json");
    }

    if (!layout.rootDir.empty()) {
        layout.manifestPath = findFileInDir(layout.rootDir, "manifest.json");
    }

    if (layout.rootDir.empty() &&
        layout.binDir.empty() &&
        layout.libDir.empty() &&
        layout.includeDir.empty()) {
        return std::nullopt;
    }

    return layout;
}

std::optional<RuntimeArtifacts> Linker::resolveRuntimeArtifacts(const ToolchainLayout& layout) const {
    RuntimeArtifacts out;

    out.includeDir = !cfg_.runtimeIncludeDirOverride.empty()
        ? canonicalizePath(cfg_.runtimeIncludeDirOverride)
        : layout.includeDir;

    if (!dirExists(out.includeDir)) {
        hud_.error("Runtime include directory missing: " + out.includeDir);
        return std::nullopt;
    }

    out.staticLibPath = !cfg_.runtimeStaticLibOverride.empty()
        ? canonicalizePath(cfg_.runtimeStaticLibOverride)
        : layout.runtimeStaticLibPath;

    out.sharedLibPath = !cfg_.runtimeSharedLibOverride.empty()
        ? canonicalizePath(cfg_.runtimeSharedLibOverride)
        : layout.runtimeSharedLibPath;

    out.importLibPath = !cfg_.runtimeImportLibOverride.empty()
        ? canonicalizePath(cfg_.runtimeImportLibOverride)
        : layout.runtimeImportLibPath;

    out.binDir = !cfg_.runtimeBinDirOverride.empty()
        ? canonicalizePath(cfg_.runtimeBinDirOverride)
        : layout.runtimeBinDir;

    const bool preferShared = cfg_.preferSharedRuntime;

    if (cfg_.toolchain == ToolchainKind::MSVC) {
        if (preferShared) {
            if (fileExists(out.importLibPath)) {
                out.extraLinkInputs.push_back(out.importLibPath);
                if (fileExists(out.sharedLibPath)) {
                    out.runtimeFilesToStage.push_back(out.sharedLibPath);
                }
                return out;
            }

            if (fileExists(out.staticLibPath)) {
                out.extraLinkInputs.push_back(out.staticLibPath);
                return out;
            }
        } else {
            if (fileExists(out.staticLibPath)) {
                out.extraLinkInputs.push_back(out.staticLibPath);
                return out;
            }

            if (fileExists(out.importLibPath)) {
                out.extraLinkInputs.push_back(out.importLibPath);
                if (fileExists(out.sharedLibPath)) {
                    out.runtimeFilesToStage.push_back(out.sharedLibPath);
                }
                return out;
            }
        }

        hud_.error("No usable packaged runtime library found");
        return std::nullopt;
    }

    if (preferShared) {
        if (fileExists(out.sharedLibPath)) {
            out.extraLinkInputs.push_back(out.sharedLibPath);
            out.runtimeFilesToStage.push_back(out.sharedLibPath);
            return out;
        }

        if (fileExists(out.staticLibPath)) {
            out.extraLinkInputs.push_back(out.staticLibPath);
            return out;
        }
    } else {
        if (fileExists(out.staticLibPath)) {
            out.extraLinkInputs.push_back(out.staticLibPath);
            return out;
        }

        if (fileExists(out.sharedLibPath)) {
            out.extraLinkInputs.push_back(out.sharedLibPath);
            out.runtimeFilesToStage.push_back(out.sharedLibPath);
            return out;
        }
    }

    hud_.error("No usable packaged runtime library found");
    return std::nullopt;
}

std::string Linker::findTool(const std::string& name) {
    if (name == "clang" && !cfg_.clangOverride.empty() && fileExists(cfg_.clangOverride)) {
        return canonicalizePath(cfg_.clangOverride);
    }

    if (name == "mlir-translate" &&
        !cfg_.mlirTranslateOverride.empty() &&
        fileExists(cfg_.mlirTranslateOverride)) {
        return canonicalizePath(cfg_.mlirTranslateOverride);
    }

    if (auto layout = resolveToolchainLayout()) {
        if (name == "clang" && !layout->clangPath.empty()) {
            return layout->clangPath;
        }
        if (name == "clang++" && !layout->clangxxPath.empty()) {
            return layout->clangxxPath;
        }
        if (name == "clang-cl" && !layout->clangClPath.empty()) {
            return layout->clangClPath;
        }
        if (name == "mlir-translate" && !layout->mlirTranslatePath.empty()) {
            return layout->mlirTranslatePath;
        }
        if ((name == "lld" || name == "ld.lld" || name == "lld-link") && !layout->lldPath.empty()) {
            return layout->lldPath;
        }
    }

    const std::vector<std::string> roots = configuredToolRoots(cfg_);

    if (auto p = findNamedToolInRoots(roots, name); !p.empty()) {
        return p;
    }

    if (hasConfiguredToolRoots(cfg_)) {
        hud_.error("Configured LLVM tool not found: " + name);
        return {};
    }

    return findProgramByNamePortable(name);
}

std::string Linker::findCompilerExe() {
    hud_.debug("toolchainRoot=[" + cfg_.toolchainRoot + "]");
    hud_.debug("llvmBinDir=[" + cfg_.llvmBinDir + "]");
    hud_.debug("clangOverride=[" + cfg_.clangOverride + "]");
    hud_.debug("mlirTranslateOverride=[" + cfg_.mlirTranslateOverride + "]");

    const std::vector<std::string> roots = configuredToolRoots(cfg_);
    for (const auto& root : roots) {
        hud_.debug("toolRoot=[" + root + "]");
    }

    if (!cfg_.clangOverride.empty() && fileExists(cfg_.clangOverride)) {
        return canonicalizePath(cfg_.clangOverride);
    }

    if (auto layout = resolveToolchainLayout()) {
        if (cfg_.toolchain == ToolchainKind::MSVC) {
            if (!layout->clangClPath.empty()) return layout->clangClPath;
            if (!layout->clangxxPath.empty()) return layout->clangxxPath;
            if (!layout->clangPath.empty()) return layout->clangPath;
        } else {
            if (!layout->clangxxPath.empty()) return layout->clangxxPath;
            if (!layout->clangPath.empty()) return layout->clangPath;
        }
    }

    if (auto p = findConfiguredCompilerInRoots(roots, cfg_.toolchain); !p.empty()) {
        return p;
    }

    if (cfg_.toolchain == ToolchainKind::MSVC) {
        if (auto p = findProgramByNamePortable("clang-cl"); !p.empty()) {
            if (pathLooksLikeVcpkgBundledLlvm(p) && cfg_.llvmBinDir.empty() && cfg_.toolchainRoot.empty()) {
                hud_.error("Selected vcpkg-bundled Clang toolchain: " + p);
                return {};
            }
            return p;
        }

        if (auto p = findProgramByNamePortable("clang++"); !p.empty()) return p;
        if (auto p = findProgramByNamePortable("clang"); !p.empty()) return p;
        return {};
    }

    if (auto p = findProgramByNamePortable("clang++"); !p.empty()) return p;
    if (auto p = findProgramByNamePortable("clang"); !p.empty()) return p;
    return {};
}

std::string Linker::chooseLinkerExe(const std::string& compilerPath) {
    if (compilerPath.empty()) {
        return {};
    }

    if (cfg_.toolchain == ToolchainKind::MSVC) {
        const std::string gnu = chooseWindowsIrDriver(compilerPath, cfg_, true);
        return gnu.empty() ? compilerPath : gnu;
    }

    llvm::SmallString<256> p(compilerPath);
    const llvm::StringRef filename = llvm::sys::path::filename(p);
    const llvm::StringRef stem = llvm::sys::path::stem(filename);
    const llvm::StringRef ext = llvm::sys::path::extension(filename);

    auto checkSibling = [&](llvm::StringRef newStem) -> std::string {
        llvm::SmallString<256> q(compilerPath);
        llvm::sys::path::remove_filename(q);
        llvm::SmallString<128> newName(newStem);
        newName += ext;
        llvm::sys::path::append(q, newName);
        if (llvm::sys::fs::exists(q)) {
            return std::string(q.str());
        }
        return {};
    };

    if (stem == "clang") {
        if (auto s = checkSibling("clang++"); !s.empty()) {
            return s;
        }
    }

    if (stem.starts_with("clang-")) {
        llvm::SmallString<64> cxxStem(stem);
        cxxStem += "++";
        if (auto s = checkSibling(cxxStem); !s.empty()) {
            return s;
        }
    }

    if (auto cxx = findTool("clang++"); !cxx.empty()) {
        return cxx;
    }

    return compilerPath;
}

int Linker::runCmd(const std::string& exe,
                   const std::vector<std::string>& args,
                   const std::string& label) {
    llvm::SmallVector<llvm::StringRef, 64> argv;
    argv.push_back(exe);
    for (const auto& a : args) {
        argv.push_back(a);
    }

    llvm::SmallString<256> outPath;
    llvm::SmallString<256> errPath;

    std::optional<llvm::StringRef> redirects[3];

    if (llvm::sys::fs::createTemporaryFile("ark-link", "out", outPath)) {
        hud_.debug("Failed to create temporary stdout log for " + label);
    } else {
        redirects[1] = llvm::StringRef(outPath);
    }

    if (llvm::sys::fs::createTemporaryFile("ark-link", "err", errPath)) {
        hud_.debug("Failed to create temporary stderr log for " + label);
    } else {
        redirects[2] = llvm::StringRef(errPath);
    }

    std::string errMsg;
    const int rc = llvm::sys::ExecuteAndWait(exe, argv, std::nullopt, redirects, 0, 0, &errMsg);

    const std::string stdoutText =
        outPath.empty() ? std::string() : readFileToString(outPath).value_or("");
    const std::string stderrText =
        errPath.empty() ? std::string() : readFileToString(errPath).value_or("");

    if (!cfg_.keepTmp) {
        if (!outPath.empty()) {
            (void)llvm::sys::fs::remove(outPath);
        }
        if (!errPath.empty()) {
            (void)llvm::sys::fs::remove(errPath);
        }
    } else if (!stdoutText.empty() || !stderrText.empty()) {
        hud_.debug("Preserved temp logs: " + std::string(outPath) + ", " + std::string(errPath));
    }

    if (!stdoutText.empty()) {
        hud_.pushLogBlock(label + " stdout", stdoutText);
    }
    if (!stderrText.empty()) {
        hud_.pushLogBlock(label + " stderr", stderrText);
    }
    if (!errMsg.empty()) {
        hud_.pushLogBlock(label + " exec error", errMsg);
    }

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

EmbeddingResult Linker::createEmbeddingSource(const std::string& symbolPrefix,
                                             const std::string& filePath) {
    if (cfg_.toolchain == ToolchainKind::MSVC) {
        auto buf = llvm::MemoryBuffer::getFile(filePath);
        if (!buf) {
            return {"", EmbeddingKind::Error};
        }

        llvm::StringRef data = (*buf)->getBuffer();

        std::string s;
        s += "#include <cstdint>\n";
        s += "extern \"C\" {\n";
        s += "  __declspec(align(16)) __declspec(dllexport) const unsigned char " + symbolPrefix + "_start[] = {";

        for (size_t i = 0; i < data.size(); ++i) {
            if ((i % 16) == 0) {
                s += "\n    ";
            }
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

bool Linker::linkToBinary(const std::string& inputLl, const std::string& outputBin) {
    return linkToBinary(inputLl, outputBin, {});
}

bool Linker::linkToBinary(
    const std::string& inputLl,
    const std::string& outputBin,
    const std::vector<ark::compiler::pipeline::CompiledGpuModule>& gpuMods
) {
    const std::string compilerExe = findCompilerExe();
    if (compilerExe.empty()) {
        hud_.error("Compiler driver not found");
        return false;
    }

    const auto layout = resolveToolchainLayout();
    if (!layout) {
        hud_.error("Toolchain layout could not be resolved");
        return false;
    }

    const auto runtime = resolveRuntimeArtifacts(*layout);
    if (!runtime) {
        if (cfg_.devSourceRuntimeFallback) {
            hud_.error("Source-runtime fallback is disabled in the packaged linker path");
        }
        return false;
    }

    std::vector<std::string> fatbinSources;
    std::string fatbinTmpDir;
    {
        auto art = emitGpuFatbinArtifacts(gpuMods);
        if (!art) {
            return false;
        }

        fatbinSources = std::move(art->extraSources);
        fatbinTmpDir = art->tmpDir;

        if (cfg_.keepTmp && !fatbinTmpDir.empty()) {
            hud_.debug("GPU fatbin temp dir: " + fatbinTmpDir);
        }
    }

    const bool needCxx = endsWithAnyCppLike(fatbinSources);

    std::string driverExe = chooseLinkerExe(compilerExe);
    if (driverExe.empty()) {
        hud_.error("Link driver not found");
        if (!cfg_.keepTmp && !fatbinTmpDir.empty()) {
            removeDirTree(fatbinTmpDir);
        }
        return false;
    }

    std::vector<std::string> args;
    args.reserve(
        16 +
        runtime->extraLinkInputs.size() +
        fatbinSources.size()
    );

    if (isLlvmIrInput(inputLl)) {
        if (cfg_.toolchain == ToolchainKind::MSVC) {
            const std::string irDriver = chooseWindowsIrDriver(driverExe, cfg_, needCxx);
            if (irDriver.empty()) {
                hud_.error(
                    "LLVM IR linking on Windows requires clang++.exe or clang.exe "
                    "next to clang-cl.exe, or reachable on PATH."
                );
                if (!cfg_.keepTmp && !fatbinTmpDir.empty()) {
                    removeDirTree(fatbinTmpDir);
                }
                return false;
            }
            driverExe = irDriver;
        }

        args.push_back("-x");
        args.push_back("ir");
        args.push_back(inputLl);
    } else {
        args.push_back(inputLl);
    }

    if (!runtime->includeDir.empty()) {
        args.push_back("-I");
        args.push_back(runtime->includeDir);
    }

    for (const auto& src : fatbinSources) {
        args.push_back(src);
    }

    for (const auto& lib : runtime->extraLinkInputs) {
        args.push_back(lib);
    }

    args = filterIrLinkFlags(args);

    if (cfg_.toolchain == ToolchainKind::MSVC) {
        args.push_back("-fms-runtime-lib=dll");
    }

    if (needCxx) {
        args.push_back("-std=c++20");
    }

    args.push_back("-g");
    args.push_back("-O0");

    appendFatbinExportFlags(args, cfg_.toolchain);
    appendPlatformLinkFlags(args, cfg_.toolchain);

    if (!runtime->runtimeFilesToStage.empty()) {
        appendRuntimeRPathFlags(args, cfg_.toolchain);
    }

    args.push_back("-o");
    args.push_back(outputBin);

    if (looksLikeClangDriver(driverExe)) {
        args.push_back("-Wno-override-module");
    }

    const int rc = runCmd(driverExe, args, "linker");

    bool stagedOk = true;
    if (rc == 0 && !runtime->runtimeFilesToStage.empty()) {
        const std::string outDir = parentDirOf(outputBin);
        for (const auto& src : runtime->runtimeFilesToStage) {
            const std::string dst = joinPath(outDir, llvm::sys::path::filename(src));
            if (!copyFileReplace(src, dst)) {
                hud_.error("Failed to stage runtime artifact: " + src);
                stagedOk = false;
            }
        }
    }

    if (!cfg_.keepTmp && !fatbinTmpDir.empty()) {
        removeDirTree(fatbinTmpDir);
    }

    return rc == 0 && stagedOk;
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