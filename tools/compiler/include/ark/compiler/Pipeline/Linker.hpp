#pragma once

#include "ark/compiler/Pipeline/Compiler.hpp"
#include "ark/compiler/Support/Hud.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>

namespace ark::compiler::pipeline {
struct CompiledGpuModule;
}

namespace ark::compiler::pipeline {

// =============================================================================
// Configuration & Enums
// =============================================================================

enum class ToolchainKind {
    Generic,
    Apple,
    MinGW,
    MSVC
};

struct ToolchainLayout {
    std::string rootDir;
    std::string binDir;
    std::string libDir;
    std::string includeDir;

    std::string clangPath;
    std::string clangxxPath;
    std::string clangClPath;
    std::string mlirTranslatePath;
    std::string lldPath;

    std::string runtimeStaticLibPath;
    std::string runtimeSharedLibPath;
    std::string runtimeImportLibPath;
    std::string runtimeBinDir;
    std::string linkerConfigPath;
    std::string manifestPath;

    bool valid() const {
        return !rootDir.empty() && !binDir.empty() && !libDir.empty() && !includeDir.empty();
    }
};

struct LinkerConfig {
    std::string toolchainRoot;
    std::string llvmBinDir;
    std::string clangOverride;
    std::string mlirTranslateOverride;
    std::string linkerConfigOverride;
    std::string runtimeIncludeDirOverride;
    std::string runtimeLibDirOverride;
    std::string runtimeBinDirOverride;
    std::string runtimeStaticLibOverride;
    std::string runtimeSharedLibOverride;
    std::string runtimeImportLibOverride;

    bool keepTmp = false;
    bool preferSharedRuntime = false;
    bool devSourceRuntimeFallback = false;

#if defined(_MSC_VER)
    ToolchainKind toolchain = ToolchainKind::MSVC;
#elif defined(__APPLE__)
    ToolchainKind toolchain = ToolchainKind::Apple;
#elif defined(__MINGW32__)
    ToolchainKind toolchain = ToolchainKind::MinGW;
#else
    ToolchainKind toolchain = ToolchainKind::Generic;
#endif
};

enum class EmbeddingKind {
    Asm,
    Cpp,
    Error
};

struct EmbeddingResult {
    std::string text;
    EmbeddingKind kind = EmbeddingKind::Error;
};

// =============================================================================
// GPU Fatbin Emission
// =============================================================================

struct FatbinArtifacts {
    std::vector<std::string> extraSources;
    std::string tmpDir;
};

// =============================================================================
// Packaged Runtime Artifacts
// =============================================================================

struct RuntimeArtifacts {
    std::string includeDir;
    std::string staticLibPath;
    std::string sharedLibPath;
    std::string importLibPath;
    std::string binDir;
    std::vector<std::string> extraLinkInputs;
    std::vector<std::string> runtimeFilesToStage;

    bool empty() const {
        return includeDir.empty() &&
               staticLibPath.empty() &&
               sharedLibPath.empty() &&
               importLibPath.empty() &&
               extraLinkInputs.empty();
    }
};

// =============================================================================
// The Linker Class
// =============================================================================

class Linker {
public:
    Linker(arklang::hud::Hud& hud, const LinkerConfig& config);

    bool translateMlirToLlvmIr(const std::string& inputMlir, const std::string& outputLl);

    bool linkToBinary(const std::string& inputLl, const std::string& outputBin);

    bool linkToBinary(const std::string& inputLl,
                      const std::string& outputBin,
                      const std::vector<CompiledGpuModule>& gpuMods);

    EmbeddingResult createEmbeddingSource(const std::string& symbolPrefix, const std::string& filePath);

    std::optional<FatbinArtifacts> emitGpuFatbinArtifacts(const std::vector<CompiledGpuModule>& mods);

    std::string makeTempDir();

private:
    std::string findTool(const std::string& name);
    std::string findCompilerExe();
    std::string chooseLinkerExe(const std::string& compilerPath);

    std::optional<ToolchainLayout> resolveToolchainLayout() const;
    std::optional<RuntimeArtifacts> resolveRuntimeArtifacts(const ToolchainLayout& layout) const;

    int runCmd(const std::string& exe,
               const std::vector<std::string>& args,
               const std::string& label);

    arklang::hud::Hud& hud_;
    LinkerConfig cfg_;
};

} // namespace ark::compiler::pipeline