#pragma once

#include "ark/compiler/Frontend/AST.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace arklang::mir {

arklang::RValue unitAlive(
    mlir::OpBuilder& b,
    mlir::Location loc
);

mlir::Value getOrCreateGlobalString(
    mlir::Location loc,
    mlir::OpBuilder& funcBuilder,
    mlir::ModuleOp module,
    llvm::StringRef content
);

mlir::Value constBool(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    bool val
);

mlir::LLVM::LLVMFuncOp getOrDeclPrintf(
    mlir::ModuleOp module,
    mlir::OpBuilder& builder
);

void emitPrintf(
    mlir::OpBuilder& b,
    mlir::Location loc,
    mlir::ModuleOp mod,
    llvm::StringRef fmt,
    mlir::ValueRange args
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

mlir::Type getUnitType(
    mlir::OpBuilder& b
);

mlir::Value getUnitUndef(
    mlir::OpBuilder& b,
    mlir::Location loc
);

bool containsReturn(
    const Expr& e
);

mlir::FailureOr<mlir::Block*> splitBlockAt(
    mlir::OpBuilder& b,
    mlir::Location loc,
    mlir::Block* cur
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

std::string astTypeToString(
    const arklang::Type& t
);

llvm::StringRef normalizeIntrinsicName(
    llvm::StringRef raw
);

bool isGpuSafeIntrinsic(
    llvm::StringRef rawName
);

mlir::LogicalResult mangleTypeRecursive(
    mlir::Type t,
    llvm::raw_ostream& os
);

mlir::FailureOr<std::string> mangleCanonicalType(
    mlir::Type t
);

arklang::Type substituteTypeParams(
    const arklang::Type& src,
    const llvm::StringMap<arklang::Type>& subst
);

std::string mangleArg(
    const arklang::Type& t
);

std::string mangleGenericName(llvm::StringRef baseName,
                              llvm::ArrayRef<arklang::Type> args);

std::string llvmStructNameFor(
    llvm::StringRef arkName
);


} // namespace arklang::mir