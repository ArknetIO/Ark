#pragma once

#include "ark/compiler/Support/Hud.hpp"
#include <llvm/ADT/StringRef.h>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include "ark/compiler/Pipeline/Compiler.hpp"

// Forward decl to avoid pulling compiler headers into Linker.h
namespace ark::compiler::pipeline {
    struct CompiledGpuModule;
}

namespace ark::compiler::pipeline {

// =============================================================================
// Configuration & Enums
// =============================================================================

enum class ToolchainKind {
    Generic, // Linux/Unix default (GCC/Clang on ELF)
    Apple,   // macOS (Clang/ld64 on Mach-O)
    MinGW,   // Windows (GCC/Clang on COFF, GNU-like flags)
    MSVC     // Windows (clang-cl on COFF, MSVC-like flags)
};

struct LinkerConfig {
    std::string llvmBinDir;            // Path to LLVM tools (optional)
    std::string runtimePath;           // Root path of ArkRuntime
    std::string clangOverride;         // Specific clang executable path
    std::string mlirTranslateOverride; // Specific mlir-translate path

    bool keepTmp = false;              // Preserve temporary files for debugging

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
    Asm,   // Generated Assembly (.s/.S) using .incbin (GCC/Clang/Apple)
    Cpp,   // Generated C++ source (.cpp) using byte arrays (MSVC fallback)
    Error
};

struct EmbeddingResult {
    std::string text;   // [FIX] matches Linker.cpp usage
    EmbeddingKind kind;
};

// =============================================================================
// GPU Fatbin Emission
// =============================================================================

struct FatbinArtifacts {
    std::vector<std::string> extraSources; // embed_*.{s|cpp} + registry TU
    std::string tmpDir;                    // directory holding emitted artifacts
};

// =============================================================================
// The Linker Class
// =============================================================================

class Linker {
public:
    Linker(arklang::hud::Hud& hud, const LinkerConfig& config);

    // -- Primary Entry Points --

    // Translates MLIR to LLVM IR (.ll)
    bool translateMlirToLlvmIr(const std::string& inputMlir, const std::string& outputLl);

    // Links objects and runtime into a final executable
    bool linkToBinary(const std::string& inputLl, const std::string& outputBin);

    // Links objects and runtime into a final executable, embedding GPU fatbins.
    bool linkToBinary(const std::string& inputLl,
                      const std::string& outputBin,
                      const std::vector<CompiledGpuModule>& gpuMods);

    // -- Fat Binary Support --

    // Generates source code to embed a binary file (GPU blob) into the host executable.
    // ABI Contract:
    //  - ELF/MinGW:   exports '{prefix}_start', '{prefix}_end', '{prefix}_size'
    //  - Mach-O:      exports '_{prefix}_start', '_{prefix}_end', '_{prefix}_size'
    //  - MSVC:        exports '{prefix}_start[]' and '{prefix}_size' (uint64)
    EmbeddingResult createEmbeddingSource(const std::string& symbolPrefix, const std::string& filePath);

    // Emits:
    //  - blob_*.bin
    //  - embed_*.{s|cpp}
    //  - gpu_fatbin_registry.cpp
    // Returns the list of sources to add to the final link invocation.
    std::optional<FatbinArtifacts> emitGpuFatbinArtifacts(const std::vector<CompiledGpuModule>& mods);

    // -- Utilities --
    std::string makeTempDir();

private:
    // Helper Methods
    std::string findTool(const std::string& name);
    std::string findCompilerExe();
    std::string chooseLinkerExe(const std::string& compilerPath);
    bool runtimeNeedsCxxLink() const;
    std::vector<std::string> getRuntimeSources() const;

    int runCmd(const std::string& exe, const std::vector<std::string>& args, const std::string& label);

    arklang::hud::Hud& hud_;
    LinkerConfig cfg_;
};

} // namespace ark::compiler::pipeline
