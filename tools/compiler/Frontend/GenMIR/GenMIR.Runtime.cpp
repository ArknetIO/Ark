#include "Frontend/GenMIR/GenMIR.Runtime.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

namespace arklang {

namespace {

static bool isLlvmPtr(mlir::Type ty) {
    return llvm::isa<mlir::LLVM::LLVMPointerType>(ty);
}

static mlir::LLVM::LLVMPointerType asLlvmPtr(mlir::Type ty) {
    return llvm::cast<mlir::LLVM::LLVMPointerType>(ty);
}

static mlir::Type getVoidPtrTy(mlir::OpBuilder &b) {
    return mlir::LLVM::LLVMPointerType::get(b.getContext());
}

} // namespace

// =============================================================================
// Small value helpers
// =============================================================================

mlir::Value constBool(mlir::OpBuilder &builder, mlir::Location loc, bool val) {
    return builder.create<mlir::LLVM::ConstantOp>(
        loc,
        builder.getI1Type(),
        builder.getBoolAttr(val)
    );
}

mlir::Type getUnitType(mlir::OpBuilder &b) {
    return mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {});
}

mlir::Value getUnitUndef(mlir::OpBuilder &b, mlir::Location loc) {
    return b.create<mlir::LLVM::UndefOp>(loc, getUnitType(b));
}

// =============================================================================
// Pointer adaptation helpers
// =============================================================================

mlir::Value castPtrTo(mlir::OpBuilder &b,
                      mlir::Location loc,
                      mlir::Value ptr,
                      mlir::Type expectedPtrTy) {
    if (!ptr || !isLlvmPtr(ptr.getType()) || !isLlvmPtr(expectedPtrTy)) {
        return {};
    }

    if (ptr.getType() == expectedPtrTy) {
        return ptr;
    }

    auto srcPtrTy = asLlvmPtr(ptr.getType());
    auto dstPtrTy = asLlvmPtr(expectedPtrTy);

    if (srcPtrTy.getAddressSpace() != dstPtrTy.getAddressSpace()) {
        return b.create<mlir::LLVM::AddrSpaceCastOp>(loc, dstPtrTy, ptr);
    }

    return ptr;
}

mlir::Value castToExpectedPtr(mlir::OpBuilder &b,
                              mlir::Location loc,
                              mlir::Value v,
                              mlir::Type expectedPtrTy) {
    if (!v || !isLlvmPtr(expectedPtrTy)) {
        return {};
    }

    if (v.getType() == expectedPtrTy) {
        return v;
    }

    if (isLlvmPtr(v.getType())) {
        return castPtrTo(b, loc, v, expectedPtrTy);
    }

    if (llvm::isa<mlir::IntegerType>(v.getType())) {
        return b.create<mlir::LLVM::IntToPtrOp>(loc, expectedPtrTy, v);
    }

    if (auto structTy = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(v.getType())) {
        if (!structTy.isOpaque()) {
            auto body = structTy.getBody();
            if (!body.empty() && isLlvmPtr(body[0])) {
                mlir::Value p = b.create<mlir::LLVM::ExtractValueOp>(
                    loc,
                    body[0],
                    v,
                    b.getDenseI64ArrayAttr({0})
                );
                return castPtrTo(b, loc, p, expectedPtrTy);
            }
        }
    }

    return {};
}

// =============================================================================
// Runtime declarations
// =============================================================================

mlir::LLVM::LLVMFuncOp getOrDeclRuntimeFn(mlir::ModuleOp module,
                                          mlir::OpBuilder &b,
                                          mlir::Location loc,
                                          llvm::StringRef name,
                                          mlir::Type retTy,
                                          llvm::ArrayRef<mlir::Type> argTys,
                                          bool isVarArg) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());

    auto fnTy = mlir::LLVM::LLVMFunctionType::get(retTy, argTys, isVarArg);
    return b.create<mlir::LLVM::LLVMFuncOp>(loc, name, fnTy);
}

mlir::LLVM::LLVMFuncOp getOrDeclareArkLaunch(mlir::ModuleOp module,
                                             mlir::OpBuilder &builder,
                                             mlir::Location loc) {
    static constexpr llvm::StringLiteral kName = "__ark_launch";

    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(kName)) {
        return fn;
    }

    auto voidPtrTy = getVoidPtrTy(builder);
    auto i64Ty = builder.getI64Type();

    return getOrDeclRuntimeFn(
        module,
        builder,
        loc,
        kName,
        i64Ty,
        {
            voidPtrTy, // grid
            voidPtrTy, // kernel
            i64Ty,     // uid_lo
            i64Ty,     // uid_hi
            voidPtrTy, // args
            i64Ty,     // args_size
            i64Ty,     // grid_dim
            voidPtrTy  // config
        },
        /*isVarArg=*/false
    );
}

mlir::LLVM::LLVMFuncOp getOrDeclareArkGpuLaunch(mlir::ModuleOp module,
                                                mlir::OpBuilder &builder,
                                                mlir::Location loc) {
    static constexpr llvm::StringLiteral kName = "__ark_gpu_launch";

    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(kName)) {
        return fn;
    }

    auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto voidPtrTy = getVoidPtrTy(builder);
    auto i32Ty = builder.getI32Type();

    return getOrDeclRuntimeFn(
        module,
        builder,
        loc,
        kName,
        voidTy,
        {
            voidPtrTy, // kernel_name
            voidPtrTy, // args
            i32Ty,     // arg_count
            i32Ty,     // grid_x
            i32Ty,     // grid_y
            i32Ty,     // grid_z
            i32Ty,     // block_x
            i32Ty,     // block_y
            i32Ty,     // block_z
            voidPtrTy  // stream
        },
        /*isVarArg=*/false
    );
}

} // namespace arklang