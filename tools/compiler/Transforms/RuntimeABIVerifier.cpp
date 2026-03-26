// tools/compiler/Transforms/RuntimeABIVerifier.cpp
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace ark::compiler::transforms {

struct RuntimeABIVerifierPass : public mlir::PassWrapper<RuntimeABIVerifierPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeABIVerifierPass)

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();

        module.walk([&](mlir::LLVM::CallOp callOp) {
            auto calleeAttr = callOp.getCalleeAttr();
            if (!calleeAttr) return; // Ignore indirect calls

            llvm::StringRef calleeName = calleeAttr.getValue();

            // Check if this call is targeting our GPU launch runtime hook.
            if (calleeName == "ark_gpu_launch" || calleeName == "cuda_launch_v1") {
                
                // 1. Enforce the Arity (Argument Count) Contract
                if (callOp.getNumOperands() != 10) {
                    auto diag = callOp.emitError("Fatal ABI Misalignment: Runtime Contract Violation");
                    
                    diag.attachNote() << "The C++ backend hook '" << calleeName << "' requires exactly 10 arguments:\n"
                                      << "  (kernel_name, args_array, arg_count, gx, gy, gz, bx, by, bz, stream)";
                    diag.attachNote() << "But GenMIR emitted a call with " << callOp.getNumOperands() << " arguments.";
                    diag.attachNote() << "Allowing this to compile will cause C-stack corruption and silent GPU failures.";
                    
                    signalPassFailure();
                    return;
                }

                // 2. Enforce the Type Contract (e.g., Stream must be a pointer)
                auto streamArgTy = callOp.getOperand(9).getType();
                // [FIX] Modern MLIR requires llvm::isa instead of .isa()
                if (!llvm::isa<mlir::LLVM::LLVMPointerType>(streamArgTy)) {
                    auto diag = callOp.emitError("Fatal ABI Misalignment: Invalid Stream Argument");
                    diag.attachNote() << "Argument 10 (stream) must be a raw pointer (!llvm.ptr), but got: " << streamArgTy;
                    signalPassFailure();
                }
            }
        });
    }
};

// Hook to create the pass
std::unique_ptr<mlir::Pass> createRuntimeABIVerifierPass() {
    return std::make_unique<RuntimeABIVerifierPass>();
}

} // namespace ark::compiler::transforms