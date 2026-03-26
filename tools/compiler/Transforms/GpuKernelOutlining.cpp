#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ark/IR/ArkMirOps.h"

using namespace mlir;

namespace arklang {

struct GpuKernelOutliningPass : public PassWrapper<GpuKernelOutliningPass, OperationPass<ModuleOp>> {
    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder builder(module.getBodyRegion());

        // 1. Create a gpu.module to hold our kernels
        // We name it 'ark_gpu_kernels'
        auto gpuModule = builder.create<gpu::GPUModuleOp>(module.getLoc(), "ark_gpu_kernels");
        SymbolTable gpuSymbolTable(gpuModule);

        // 2. Find functions tagged with 'ark.domain = "gpu"'
        SmallVector<func::FuncOp, 4> gpuFunctions;
        module.walk([&](func::FuncOp func) {
            if (auto attr = func->getAttrOfType<StringAttr>("ark.domain")) {
                if (attr.getValue() == "gpu") {
                    gpuFunctions.push_back(func);
                }
            }
        });

        for (auto func : gpuFunctions) {
            // 3. Clone function into gpu.module
            builder.setInsertionPointToStart(gpuModule.getBody());
            auto gpuFunc = builder.create<gpu::GPUFuncOp>(
                func.getLoc(), 
                func.getName(), 
                func.getFunctionType()
            );
            
            // Mark as kernel entry point
            gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

            // Move body
            gpuFunc.getBody().takeBody(func.getBody());
            
            // 4. Transform ark.slot/load/store inside kernel to raw memory ops?
            // Actually, we need to run standard lowering inside the GPU module later.
            
            // 5. Remove original function from host module 
            // (Strictly speaking we should keep a stub, but for now we remove it 
            // because ark.launch refers to it by name string, not symbol reference logic yet)
            func.erase();
        }

        // If no kernels, remove the empty module
        if (gpuFunctions.empty()) {
            gpuModule.erase();
        }
    }
};

std::unique_ptr<Pass> createGpuKernelOutliningPass() {
    return std::make_unique<GpuKernelOutliningPass>();
}

} // namespace arklang