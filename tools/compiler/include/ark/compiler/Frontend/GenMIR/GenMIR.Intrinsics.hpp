#pragma once

// =============================================================================
// GenMIR Intrinsics Surface
// =============================================================================
// Class-fragment header.
// Include only from inside class GenMIR in GenMIR.hpp.
// =============================================================================

// Create or retrieve the private panic function.
mlir::func::FuncOp getOrCreatePanicFn();

// Emit a panic call and return a terminated RValue.
RValue emitPanic(mlir::Location loc, mlir::Value msgStrVal);



// Host-side assertion: if (!cond) panic(msg).
void emitHostAssert(mlir::Location loc,
                    mlir::Value condI1,
                    llvm::StringRef msg);

// Host-side validation for parallel loop bounds.
void assertParBoundsHost(mlir::Location loc,
                         mlir::Value startI64,
                         mlir::Value limitI64);


// Create a type-stable intrinsic symbol for a concrete lowered argument type.
mlir::FailureOr<mlir::func::FuncOp> getOrCreateIntrinsic(mlir::Location loc,
                                                         llvm::StringRef base,
                                                         mlir::Type argTy,
                                                         mlir::TypeRange resTys);
// Get a container length through intrinsic lowering.
mlir::FailureOr<mlir::Value> getContainerLen(mlir::Location loc,
                                             const Expr& expr);

// Get width/height-style dimensions through intrinsic lowering.
mlir::FailureOr<DimsResult> getDimsFromCall(mlir::Location loc,
                                            const CallExpr& call);

// Force or adapt a value into the lowered string representation.
mlir::FailureOr<mlir::Value> forceStrValue(mlir::Location loc,
                                           mlir::Value v,
                                           arklang::Type astTy);

// Dispatcher for built-in intrinsics such as len, allocof, dims, etc.
mlir::FailureOr<RValue> lowerIntrinsicCall(const CallExpr& call,
                                           llvm::StringRef name);