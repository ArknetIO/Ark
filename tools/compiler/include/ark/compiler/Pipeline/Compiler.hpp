// tools/compiler/Pipeline/Compiler.h
#pragma once

#include "ark/compiler/Support/Hud.hpp"
#include <llvm/ADT/StringRef.h>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
class ModuleOp;
}
namespace arklang {
class ModuleRegistry;
}

namespace ark::compiler::pipeline {

struct CompilerConfig {
    // Enable/Disable backends per build
    bool enableCudaPtx  = true;
    bool enableHipHsaco = false;
    bool enableMetal    = false;

    // CUDA PTX
    std::string cudaArch     = "sm_60";
    std::string cudaFeatures = "+ptx60";

    // HIP HSACO (AMDGPU)
    std::string hipArch      = "gfx900";
    std::string hipFeatures  = "";

    // Metal: phase-1 uses source blob injection (MSL) if present as an attribute
    // - If enableMetal=true and a gpu.module carries `ark.metal.msl` (StringAttr),
    //   we emit moduleKind=ARK_GPU_MODULE_SRC_MSL (7) with that string as the blob.
    bool metalUseSourceBlob = true;
};

struct CompiledGpuModule {
    std::string moduleKey;

    // ark_gpu_module_kind (runtime/gpu/gpu_backend.h):
    // 1=CUDA_PTX, 2=CUDA_CUBIN, 3=HIP_HSACO, 4=METAL_LIB, 5=SRC_CUDA, 6=SRC_HIP, 7=SRC_MSL
    std::uint32_t moduleKind;

    std::string blob;
    std::vector<std::string> kernels;
};

class Compiler {
public:
    Compiler(arklang::hud::Hud& hud, arklang::ModuleRegistry& registry, const CompilerConfig& config = CompilerConfig{});

    bool compileToMLIR(llvm::StringRef inputFilename, mlir::MLIRContext& ctx, mlir::ModuleOp& outModule);
    bool lowerToLLVM(mlir::MLIRContext& ctx, mlir::ModuleOp module);

    std::vector<CompiledGpuModule> extractCompiledGpuModules(mlir::ModuleOp module);
    bool writeModuleToFile(mlir::ModuleOp module, llvm::StringRef path);

private:
    arklang::hud::Hud& hud_;
    arklang::ModuleRegistry& registry_;
    CompilerConfig cfg_;
};

} // namespace ark::compiler::pipeline
