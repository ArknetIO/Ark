#include "ark/compiler/Frontend/GenMIR.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Intrinsics.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Runtime.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Types.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Casting.h"

#include <string>

namespace {

// =============================================================================
// Local Runtime Declaration Helper
// =============================================================================
// Used by intrinsic lowering to lazily declare plain LLVM runtime entrypoints
// such as free/hash helpers.
// =============================================================================
static mlir::LLVM::LLVMFuncOp getOrDeclIntrinsicRuntimeFn(mlir::ModuleOp module,
                                                          mlir::OpBuilder& builder,
                                                          llvm::StringRef name,
                                                          mlir::Type retTy,
                                                          llvm::ArrayRef<mlir::Type> args) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto fnTy = mlir::LLVM::LLVMFunctionType::get(retTy, args, /*isVarArg=*/false);
    return builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, fnTy);
}

} // namespace

namespace arklang {

// =============================================================================
// Panic Function Declaration
// =============================================================================
// Materialize the private MLIR func-level panic shim used by safety lowering.
// =============================================================================
mlir::func::FuncOp GenMIR::getOrCreatePanicFn() {
    static constexpr llvm::StringLiteral kName = "__ark_panic";

    if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(kName)) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto fnTy = builder.getFunctionType(mlir::TypeRange{i8PtrTy}, mlir::TypeRange{});

    auto fn = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), kName, fnTy);
    fn.setPrivate();
    return fn;
}

// =============================================================================
// Panic Emission
// =============================================================================
// Emit a terminating runtime panic call.
//
// Notes:
// - returns a dead/already-failed RValue shape for caller convenience
// - explicitly creates the unit value before unreachable terminates the block
// =============================================================================
RValue GenMIR::emitPanic(mlir::Location loc, mlir::Value msgStrVal) {
    arklang::Type strAst{arklang::Type::Str};
    mlir::Type strTy = convertType(strAst);
    mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    auto panicFn = getOrDeclRuntimeFn(module, builder, loc, "ark_panic", voidTy, {strTy});

    mlir::Value deadUnit = getUnitUndef(builder, loc);

    builder.create<mlir::LLVM::CallOp>(loc, panicFn, mlir::ValueRange{msgStrVal});
    builder.create<mlir::LLVM::UnreachableOp>(loc);

    return RValue{deadUnit, constBool(builder, loc, false)};
}

// =============================================================================
// Host Assertion
// =============================================================================
// Lower:
//   assert(cond, msg)
// into:
//   if (!cond) panic(msg)
// =============================================================================
void GenMIR::emitHostAssert(mlir::Location loc, mlir::Value condI1, llvm::StringRef msg) {
    auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    mlir::Value isBad = builder.create<mlir::arith::XOrIOp>(loc, condI1, one);

    auto ifOp = builder.create<mlir::scf::IfOp>(loc, isBad, /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    auto cstr = arklang::mir::getOrCreateGlobalString(loc, builder, module, msg.str());
    emitPanic(loc, cstr);

    builder.setInsertionPointAfter(ifOp);
}

// =============================================================================
// Parallel Bounds Assertions
// =============================================================================
// Enforce the minimal host-side safety invariants for par-loop bounds.
// =============================================================================
void GenMIR::assertParBoundsHost(mlir::Location loc,
                                 mlir::Value startI64,
                                 mlir::Value limitI64) {
    auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);

    auto startNonNeg = builder.create<mlir::arith::CmpIOp>(
        loc,
        mlir::arith::CmpIPredicate::sge,
        startI64,
        zero
    );
    emitHostAssert(loc, startNonNeg, "par: start must be >= 0");

    auto limitNonNeg = builder.create<mlir::arith::CmpIOp>(
        loc,
        mlir::arith::CmpIPredicate::sge,
        limitI64,
        zero
    );
    emitHostAssert(loc, limitNonNeg, "par: limit must be >= 0");

    auto limitGeStart = builder.create<mlir::arith::CmpIOp>(
        loc,
        mlir::arith::CmpIPredicate::sge,
        limitI64,
        startI64
    );
    emitHostAssert(loc, limitGeStart, "par: limit must be >= start");
}

// =============================================================================
// Type-Stable Intrinsic Symbol Creation
// =============================================================================
// Create or reuse a monomorphized intrinsic symbol based on the lowered input
// type. This keeps intrinsic dispatch type-stable at the IR level.
// =============================================================================
mlir::FailureOr<mlir::func::FuncOp> GenMIR::getOrCreateIntrinsic(mlir::Location loc,
                                                                 llvm::StringRef base,
                                                                 mlir::Type argTy,
                                                                 mlir::TypeRange resTys) {
    auto suffixOr = arklang::mir::mangleCanonicalType(argTy);
    if (mlir::failed(suffixOr)) {
        return mlir::failure();
    }

    std::string name = (base + "_" + *suffixOr).str();

    if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
        auto want = builder.getFunctionType({argTy}, resTys);
        if (existing.getFunctionType() != want) {
            return mlir::failure();
        }
        return existing;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto fnTy = builder.getFunctionType({argTy}, resTys);
    auto fn = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name, fnTy);
    fn.setPrivate();
    return fn;
}

// =============================================================================
// Dimension Intrinsic
// =============================================================================
// Lower a dimension query through the type-stable intrinsic path.
// =============================================================================
mlir::FailureOr<GenMIR::DimsResult> GenMIR::getDimsFromCall(mlir::Location loc,
                                                            const CallExpr& call) {
    if (call.args.size() != 1) {
        return mlir::failure();
    }

    auto vOr = lowerExpr(*call.args[0].value);
    if (mlir::failed(vOr)) {
        return mlir::failure();
    }

    mlir::Value v = vOr->val;
    mlir::Type i64Ty = builder.getI64Type();

    auto fnOr = getOrCreateIntrinsic(
        loc,
        "__ark_intrinsic_dims",
        v.getType(),
        mlir::TypeRange{i64Ty, i64Ty}
    );
    if (mlir::failed(fnOr)) {
        return mlir::failure();
    }

    auto callOp = builder.create<mlir::func::CallOp>(loc, *fnOr, mlir::ValueRange{v});
    return DimsResult{callOp.getResult(0), callOp.getResult(1)};
}

// =============================================================================
// String Value Enforcement
// =============================================================================
// Require that the caller already holds the lowered SSA string value, not an
// addressable place/pointer.
// =============================================================================
mlir::FailureOr<mlir::Value> GenMIR::forceStrValue(mlir::Location loc,
                                                   mlir::Value v,
                                                   arklang::Type astTy) {
    if (astTy.kind != arklang::Type::Str) {
        mlir::emitError(loc) << "Internal: forceStrValue on non-Str AST type";
        return mlir::failure();
    }

    mlir::Type strTy = convertType(astTy);

    if (v.getType() != strTy) {
        mlir::emitError(loc)
            << "Internal: String intrinsic expects value (Struct), got "
            << v.getType() << ". Missing Load?";
        return mlir::failure();
    }

    return v;
}

// =============================================================================
// Container Length Helper
// =============================================================================
// Shared intrinsic-style helper for extracting container length from lowered
// container values.
// =============================================================================
mlir::FailureOr<mlir::Value> GenMIR::getContainerLen(mlir::Location loc, const Expr& expr) {
    // 1. Kernel/runtime-sized variables
    if (auto* sym = dynamic_cast<const SymbolExpr*>(&expr)) {
        if (mir->isDeclared(sym->name)) {
            VarInfo* var = mir->lookup(sym->name);
            if (var->len) {
                mlir::Value len = var->len;
                if (len.getType() != builder.getI64Type()) {
                    len = coerce(builder, loc, len, builder.getI64Type());
                }
                return len;
            }
        }
    }

    // 2. Lower expression
    auto res = lowerExpr(expr);
    if (mlir::failed(res)) return mlir::failure();

    RValue rv = *res;
    mlir::Value val = rv.val;
    arklang::Type astTy = getExprType(expr);

    // 3. If lowered as pointer-to-container on host, load it first
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
        mlir::Type loadedTy = convertType(astTy);
        if (llvm::isa<mlir::LLVM::LLVMStructType>(loadedTy)) {
            val = builder.create<mlir::LLVM::LoadOp>(loc, loadedTy, val);
        } else {
            mlir::emitError(loc)
                << "Cannot infer length of raw pointer natively. "
                << "Use an explicit range loop or pass length explicitly.";
            return mlir::failure();
        }
    }

    // 4. Struct unpacking
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(val.getType());
    if (!st) {
        mlir::emitError(loc)
            << "Type does not support implicit length inference";
        return mlir::failure();
    }

    llvm::ArrayRef<mlir::Type> body = st.getBody();
    if (body.size() < 2) {
        mlir::emitError(loc)
            << "Container struct is too small to contain length information";
        return mlir::failure();
    }

    mlir::Value len = builder.create<mlir::LLVM::ExtractValueOp>(
        loc,
        val,
        builder.getDenseI64ArrayAttr({1})
    );

    if (!llvm::isa<mlir::IntegerType>(len.getType())) {
        mlir::emitError(loc)
            << "Container length field is not an integer";
        return mlir::failure();
    }

    if (len.getType() != builder.getI64Type()) {
        len = coerce(builder, loc, len, builder.getI64Type());
    }

    return len;
}

// =============================================================================
// Intrinsic Lowering Dispatcher
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerIntrinsicCall(const CallExpr& call,
                                                   llvm::StringRef name) {
    mlir::Location loc = toLoc(call.loc);
    auto ipTy = builder.getI64Type();
    mlir::DataLayout dl(module);

    // -------------------------------------------------------------------------
    // sizeof<T>() -> i64
    // -------------------------------------------------------------------------
    if (name == "sizeof") {
        if (call.explicitGenericArgs.empty()) {
            return fail(call.loc, "'sizeof' requires type <T>");
        }

        mlir::Type t = convertType(call.explicitGenericArgs[0]);
        uint64_t bytes = (dl.getTypeSizeInBits(t) + 7) / 8;

        return RValue{
            builder.create<mlir::LLVM::ConstantOp>(
                loc,
                ipTy,
                builder.getIntegerAttr(ipTy, bytes)
            ),
            constBool(builder, loc, true)
        };
    }

    // -------------------------------------------------------------------------
    // typeof<T>() -> String
    // -------------------------------------------------------------------------
    if (name == "typeof") {
        if (call.explicitGenericArgs.empty()) {
            return fail(call.loc, "'typeof' requires type <T>");
        }

        std::string typeName = arklang::mir::astTypeToString(call.explicitGenericArgs[0]);
        StringExpr s(call.loc, typeName);
        return lowerString(s);
    }

    // -------------------------------------------------------------------------
    // castof<T>(val) -> T
    // -------------------------------------------------------------------------
    if (name == "castof") {
        if (call.args.size() != 1) {
            return fail(call.loc, "'castof' expects 1 argument");
        }
        if (call.explicitGenericArgs.empty()) {
            return fail(call.loc, "'castof' requires target type <T>");
        }

        auto argRv = lowerExpr(*call.args[0].value);
        if (mlir::failed(argRv)) {
            return mlir::failure();
        }

        arklang::Type srcAstTy = getExprType(*call.args[0].value);
        arklang::Type dstAstTy = call.explicitGenericArgs[0];

        mlir::Value input = argRv->val;
        mlir::Type inputTy = input.getType();
        mlir::Type targetTy = convertType(dstAstTy);

        // Pointer <-> Pointer
        if (llvm::isa<mlir::LLVM::LLVMPointerType>(inputTy) &&
            llvm::isa<mlir::LLVM::LLVMPointerType>(targetTy)) {
            return RValue{
                builder.create<mlir::LLVM::BitcastOp>(loc, targetTy, input),
                argRv->state
            };
        }

        // Pointer <-> Integer
        if (llvm::isa<mlir::LLVM::LLVMPointerType>(inputTy) &&
            llvm::isa<mlir::IntegerType>(targetTy)) {
            return RValue{
                builder.create<mlir::LLVM::PtrToIntOp>(loc, targetTy, input),
                argRv->state
            };
        }

        if (llvm::isa<mlir::IntegerType>(inputTy) &&
            llvm::isa<mlir::LLVM::LLVMPointerType>(targetTy)) {
            return RValue{
                builder.create<mlir::LLVM::IntToPtrOp>(loc, targetTy, input),
                argRv->state
            };
        }

        // Integer <-> Integer
        auto intIn = llvm::dyn_cast<mlir::IntegerType>(inputTy);
        auto intOut = llvm::dyn_cast<mlir::IntegerType>(targetTy);
        if (intIn && intOut) {
            unsigned inW = intIn.getWidth();
            unsigned outW = intOut.getWidth();

            if (inW == outW) {
                return RValue{input, argRv->state};
            }
            if (inW > outW) {
                return RValue{
                    builder.create<mlir::LLVM::TruncOp>(loc, targetTy, input),
                    argRv->state
                };
            }

            if (srcAstTy.isSigned()) {
                return RValue{
                    builder.create<mlir::LLVM::SExtOp>(loc, targetTy, input),
                    argRv->state
                };
            }

            return RValue{
                builder.create<mlir::LLVM::ZExtOp>(loc, targetTy, input),
                argRv->state
            };
        }

        // Float <-> Float
        auto floatIn = llvm::dyn_cast<mlir::FloatType>(inputTy);
        auto floatOut = llvm::dyn_cast<mlir::FloatType>(targetTy);
        if (floatIn && floatOut) {
            unsigned inW = floatIn.getWidth();
            unsigned outW = floatOut.getWidth();

            if (inW > outW) {
                return RValue{
                    builder.create<mlir::LLVM::FPTruncOp>(loc, targetTy, input),
                    argRv->state
                };
            }

            return RValue{
                builder.create<mlir::LLVM::FPExtOp>(loc, targetTy, input),
                argRv->state
            };
        }

        // Integer <-> Float
        if (intIn && floatOut) {
            if (srcAstTy.isSigned()) {
                return RValue{
                    builder.create<mlir::LLVM::SIToFPOp>(loc, targetTy, input),
                    argRv->state
                };
            }

            return RValue{
                builder.create<mlir::LLVM::UIToFPOp>(loc, targetTy, input),
                argRv->state
            };
        }

        if (floatIn && intOut) {
            if (dstAstTy.isSigned()) {
                return RValue{
                    builder.create<mlir::LLVM::FPToSIOp>(loc, targetTy, input),
                    argRv->state
                };
            }

            return RValue{
                builder.create<mlir::LLVM::FPToUIOp>(loc, targetTy, input),
                argRv->state
            };
        }

        return fail(call.loc, "castof unsupported operand combination");
    }

    // -------------------------------------------------------------------------
    // free(ptr) -> Void
    // -------------------------------------------------------------------------
    if (name == "free") {
        if (call.args.size() != 1) {
            return fail(call.loc, "'free' expects 1 argument");
        }

        auto argRv = lowerExpr(*call.args[0].value);
        if (mlir::failed(argRv)) {
            return mlir::failure();
        }

        auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
        auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto freeFn = getOrDeclIntrinsicRuntimeFn(module, builder, "free", voidTy, {voidPtrTy});

        mlir::Value ptr = castToExpectedPtr(builder, loc, argRv->val, voidPtrTy);
        builder.create<mlir::LLVM::CallOp>(loc, freeFn, mlir::ValueRange{ptr});

        return RValue{getUnitUndef(builder, loc), constBool(builder, loc, true)};
    }

    // -------------------------------------------------------------------------
    // len(container) -> i64
    // -------------------------------------------------------------------------
    if (name == "len") {
        if (call.args.size() != 1) {
            return fail(call.loc, "'len' expects 1 argument");
        }

        auto lenOr = getContainerLen(loc, *call.args[0].value);
        if (mlir::failed(lenOr)) {
            return fail(call.loc, "Type does not support 'len'");
        }

        return RValue{*lenOr, constBool(builder, loc, true)};
    }

    // -------------------------------------------------------------------------
    // addr(place) -> Ptr
    // -------------------------------------------------------------------------
    if (name == "addr") {
        if (call.args.size() != 1) {
            return fail(call.loc, "'addr' expects 1 argument");
        }

        auto placeOr = lowerExprAsPlace(*call.args[0].value);
        if (mlir::failed(placeOr)) {
            return fail(call.loc, "'addr' requires an addressable place");
        }

        mlir::Value p = *placeOr;
        auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        p = castToExpectedPtr(builder, loc, p, voidPtrTy);

        return RValue{p, constBool(builder, loc, true)};
    }

    // -------------------------------------------------------------------------
    // min(a, b) -> T
    // -------------------------------------------------------------------------
    if (name == "min") {
        if (call.args.size() != 2) {
            return fail(call.loc, "'min' expects 2 arguments");
        }

        auto lhsRv = lowerExpr(*call.args[0].value);
        auto rhsRv = lowerExpr(*call.args[1].value);
        if (mlir::failed(lhsRv) || mlir::failed(rhsRv)) {
            return mlir::failure();
        }

        mlir::Value lhs = lhsRv->val;
        mlir::Value rhs = coerce(builder, loc, rhsRv->val, lhs.getType());

        if (llvm::isa<mlir::FloatType>(lhs.getType())) {
            mlir::Value res = builder.create<mlir::LLVM::MinNumOp>(loc, lhs, rhs);
            return RValue{res, constBool(builder, loc, true)};
        }

        mlir::Value pred;
        if (llvm::isa<mlir::IntegerType>(lhs.getType())) {
            bool isSigned = getExprType(*call.args[0].value).isSigned();
            auto predKind = isSigned
                ? mlir::LLVM::ICmpPredicate::slt
                : mlir::LLVM::ICmpPredicate::ult;
            pred = builder.create<mlir::LLVM::ICmpOp>(loc, predKind, lhs, rhs);
        } else {
            return fail(call.loc, "'min' only supports integer/float types");
        }

        mlir::Value res = builder.create<mlir::LLVM::SelectOp>(loc, pred, lhs, rhs);
        return RValue{res, constBool(builder, loc, true)};
    }

    // -------------------------------------------------------------------------
    // panic(msg) -> NoReturn
    // -------------------------------------------------------------------------
    if (name == "panic") {
        if (call.args.size() != 1) {
            return fail(call.loc, "'panic' expects 1 argument");
        }

        arklang::Type argTy = getExprType(*call.args[0].value);
        auto msgRv = lowerExpr(*call.args[0].value);
        if (mlir::failed(msgRv)) {
            return mlir::failure();
        }

        auto msgOr = forceStrValue(loc, msgRv->val, argTy);
        if (mlir::failed(msgOr)) {
            return mlir::failure();
        }

        return emitPanic(loc, *msgOr);
    }

    // -------------------------------------------------------------------------
    // assert(cond, msg?) -> Void
    // -------------------------------------------------------------------------
    if (name == "assert") {
        if (call.args.size() < 1 || call.args.size() > 2) {
            return fail(call.loc, "'assert' expects 1 or 2 arguments");
        }

        arklang::Type condAstTy = getExprType(*call.args[0].value);
        if (condAstTy.kind != arklang::Type::Bool) {
            return fail(call.loc, "'assert' condition must be boolean");
        }

        auto condRv = lowerExpr(*call.args[0].value);
        if (mlir::failed(condRv)) {
            return mlir::failure();
        }

        mlir::Value cond = condRv->val;
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(cond.getType())) {
            if (intTy.getWidth() != 1) {
                return fail(call.loc, "Internal: assert bool lowering mismatch");
            }
        } else {
            return fail(call.loc, "Internal: assert condition is not integer");
        }

        mlir::Block* curBlock = builder.getInsertionBlock();
        if (!curBlock) {
            return fail(call.loc, "assert outside of a block");
        }

        auto contBlockOr = arklang::mir::splitBlockAt(builder, loc, curBlock);
        if (mlir::failed(contBlockOr)) {
            return mlir::failure();
        }

        mlir::Block* contBlock = *contBlockOr;
        mlir::Region* region = curBlock->getParent();
        mlir::Block* failBlock = builder.createBlock(region, contBlock->getIterator());

        builder.setInsertionPointToEnd(curBlock);

        if (curBlock->getTerminator()) {
            return fail(call.loc, "Internal Error: terminator conflict in assert");
        }

        builder.create<mlir::LLVM::CondBrOp>(loc, cond, contBlock, failBlock);

        // Fail path
        builder.setInsertionPointToStart(failBlock);
        mlir::Value msg;

        if (call.args.size() == 2) {
            arklang::Type msgTy = getExprType(*call.args[1].value);
            auto msgRv = lowerExpr(*call.args[1].value);
            if (mlir::failed(msgRv)) {
                return mlir::failure();
            }

            auto msgOr = forceStrValue(loc, msgRv->val, msgTy);
            if (mlir::failed(msgOr)) {
                return mlir::failure();
            }

            msg = *msgOr;
        } else {
            StringExpr s(call.loc, "Assertion failed");
            auto sRes = lowerString(s);
            if (mlir::failed(sRes)) {
                return mlir::failure();
            }
            msg = sRes->val;
        }

        emitPanic(loc, msg);

        // Resume path
        builder.setInsertionPointToStart(contBlock);
        return RValue{getUnitUndef(builder, loc), constBool(builder, loc, true)};
    }

    // -------------------------------------------------------------------------
    // hash(val), shash(val), stable_hash(val) -> i64
    // -------------------------------------------------------------------------
    if (name == "hash" || name == "shash" || name == "stable_hash") {
        if (call.args.size() != 1) {
            return fail(call.loc, "'" + std::string(name) + "' expects 1 argument");
        }

        auto argRvOr = lowerExpr(*call.args[0].value);
        if (mlir::failed(argRvOr)) {
            return mlir::failure();
        }

        mlir::Value val = argRvOr->val;
        arklang::Type astTy = getExprType(*call.args[0].value);

        mlir::Value ptr;
        mlir::Value lenBytes;
        mlir::Type i64Ty = builder.getI64Type();
        auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        // Strategy A: containers
        if (astTy.kind == arklang::Type::Str ||
            astTy.kind == arklang::Type::Slice ||
            astTy.kind == arklang::Type::Vec) {
            if (llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
                val = builder.create<mlir::LLVM::LoadOp>(loc, convertType(astTy), val);
            }

            ptr = builder.create<mlir::LLVM::ExtractValueOp>(
                loc,
                val,
                builder.getDenseI64ArrayAttr({0})
            ).getResult();

            mlir::Value lenElems = builder.create<mlir::LLVM::ExtractValueOp>(
                loc,
                val,
                builder.getDenseI64ArrayAttr({1})
            ).getResult();

            mlir::Type elemTy = builder.getI8Type();
            if (!astTy.genericArgs.empty()) {
                elemTy = convertType(astTy.genericArgs[0]);
            }

            uint64_t elemSize = (dl.getTypeSizeInBits(elemTy) + 7) / 8;
            if (elemSize > 1) {
                mlir::Value elemSzVal = builder.create<mlir::LLVM::ConstantOp>(
                    loc,
                    i64Ty,
                    builder.getI64IntegerAttr(elemSize)
                );
                lenBytes = builder.create<mlir::LLVM::MulOp>(loc, lenElems, elemSzVal).getResult();
            } else {
                lenBytes = lenElems;
            }

            ptr = castToExpectedPtr(builder, loc, ptr, voidPtrTy);
        }
        // Strategy B: primitives / inline structs
        else {
            uint64_t bytes = (dl.getTypeSizeInBits(val.getType()) + 7) / 8;
            lenBytes = builder.create<mlir::LLVM::ConstantOp>(
                loc,
                i64Ty,
                builder.getI64IntegerAttr(bytes)
            );

            mlir::Value spilled = mir->spillTemp(loc, val.getType(), val);
            ptr = castToExpectedPtr(builder, loc, spilled, voidPtrTy);
        }

        llvm::StringRef runtimeFnName =
            (name == "hash") ? "__ark_hash_bytes" : "__ark_stable_hash_bytes";

        auto hashFn = getOrDeclRuntimeFn(module, builder, loc, runtimeFnName, i64Ty, {voidPtrTy, i64Ty});
        auto callOp = builder.create<mlir::LLVM::CallOp>(
            loc,
            hashFn,
            mlir::ValueRange{ptr, lenBytes}
        );

        return RValue{callOp.getOperation()->getResult(0), arklang::mir::unitAlive(builder, loc).state};
    }

    return fail(call.loc, "Unknown intrinsic: " + std::string(name));
}

} // namespace arklang