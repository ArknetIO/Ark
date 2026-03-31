#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace arklang {

mlir::Value constBool(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    bool val
);

mlir::Type getUnitType(
    mlir::OpBuilder& b
);

mlir::Value getUnitUndef(
    mlir::OpBuilder& b,
    mlir::Location loc
);

mlir::Value castPtrTo(
    mlir::OpBuilder& b,
    mlir::Location loc,
    mlir::Value ptr,
    mlir::Type expectedPtrTy
);

mlir::Value castToExpectedPtr(
    mlir::OpBuilder& b,
    mlir::Location loc,
    mlir::Value v,
    mlir::Type expectedPtrTy
);

mlir::LLVM::LLVMFuncOp getOrDeclRuntimeFn(
    mlir::ModuleOp module,
    mlir::OpBuilder& b,
    mlir::Location loc,
    llvm::StringRef name,
    mlir::Type retTy,
    llvm::ArrayRef<mlir::Type> argTys,
    bool isVarArg = false
);

mlir::LLVM::LLVMFuncOp getOrDeclareArkLaunch(
    mlir::ModuleOp module,
    mlir::OpBuilder& builder,
    mlir::Location loc
);

mlir::LLVM::LLVMFuncOp getOrDeclareArkGpuLaunch(
    mlir::ModuleOp module,
    mlir::OpBuilder& builder,
    mlir::Location loc
);

} // namespace arklang