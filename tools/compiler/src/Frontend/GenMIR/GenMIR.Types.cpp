#include "ark/compiler/Frontend/GenMIR/GenMIR.Types.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace arklang::mir {

// =============================================================================
// MLIR Type Formatting
// =============================================================================
// Small diagnostic helper used by the type layer to stringify MLIR types
// exactly as MLIR renders them.
std::string typeToString(mlir::Type type) {
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    type.print(os);
    return os.str();
}

// =============================================================================
// Tensor-Type Classification
// =============================================================================
// Centralized type-policy helper for deciding whether an Ark type should be
// treated as tensor-like by the lowering layer.
//
// Current policy:
// - tensor<T> is tensor-like
// - Alloc<T> is treated as tensor-like for lowering purposes
// =============================================================================
bool isTensorType(const arklang::Type& ty) {
    return
        ty.kind == arklang::Type::Tensor ||
        (ty.kind == arklang::Type::Generic && ty.schemaName == "Alloc");
}

} // namespace arklang::mir

namespace arklang {

// =============================================================================
// Type Conversion (AST -> LLVM Dialect / MemRef)
// =============================================================================
// Converts Ark AST types into their lowered MLIR representation.
//
// Key policy decisions preserved here:
// - Alloc<T> lowers to an opaque LLVM pointer
// - Tensor-like values lower differently depending on function domain
//   * GPU domain  -> dynamic 1D memref
//   * Host domain -> opaque LLVM pointer
// - Composite host types (str, vec, slice, tuple) lower to LLVM structs
// - Schema/generic user types resolve through schema instantiation
// =============================================================================
mlir::Type GenMIR::convertType(const arklang::Type& astType) {
    mlir::MLIRContext* ctx = module.getContext();

    // -------------------------------------------------------------------------
    // Intrinsic generic Alloc<T>
    // -------------------------------------------------------------------------
    // Alloc<T> is currently treated as a raw pointer-like type. This keeps it
    // usable for low-level address manipulation and matches current lowering
    // expectations unless a richer container policy is introduced later.
    // -------------------------------------------------------------------------
    if (astType.kind == arklang::Type::Generic && astType.schemaName == "Alloc") {
        return mlir::LLVM::LLVMPointerType::get(ctx);
    }

    // -------------------------------------------------------------------------
    // Domain-aware tensor lowering
    // -------------------------------------------------------------------------
    // GPU kernels need tensors in MemRef form so GPU ABI lowering can
    // materialize the expected pointer/shape-style calling convention.
    //
    // Host-side lowering keeps tensors as opaque pointers for now.
    // -------------------------------------------------------------------------
    if (arklang::mir::isTensorType(astType)) {
        if (currentFnDomain == Domain::GPU) {
            // Current policy:
            // - 1D dynamic memref
            // - f32 default element type
            //
            // Future improvement:
            // - derive element type from genericArgs[0] when available
            mlir::Type elemTy = builder.getF32Type();
            return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
        }

        return mlir::LLVM::LLVMPointerType::get(ctx);
    }

    switch (astType.kind) {
        case arklang::Type::Void:
            return mlir::LLVM::LLVMStructType::getLiteral(ctx, {});

        case arklang::Type::Bool:
            return mlir::IntegerType::get(ctx, 1);

        case arklang::Type::I8:
            return mlir::IntegerType::get(ctx, 8);

        case arklang::Type::I16:
            return mlir::IntegerType::get(ctx, 16);

        case arklang::Type::I32:
            return mlir::IntegerType::get(ctx, 32);

        case arklang::Type::I64:
            return mlir::IntegerType::get(ctx, 64);

        case arklang::Type::U8:
            return mlir::IntegerType::get(ctx, 8);

        case arklang::Type::U16:
            return mlir::IntegerType::get(ctx, 16);

        case arklang::Type::U32:
            return mlir::IntegerType::get(ctx, 32);

        case arklang::Type::U64:
            return mlir::IntegerType::get(ctx, 64);

        case arklang::Type::F32:
            return mlir::Float32Type::get(ctx);

        case arklang::Type::F64:
            return mlir::Float64Type::get(ctx);

        case arklang::Type::Str: {
            // Lowered as:
            //   { i8*, i64 }
            // where:
            //   field 0 = data pointer
            //   field 1 = length
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
            auto i64Ty = mlir::IntegerType::get(ctx, 64);
            return mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
        }

        case arklang::Type::Func:
            // Functions lower to opaque callable/code pointers for now.
            return mlir::LLVM::LLVMPointerType::get(ctx);

        case arklang::Type::Vec: {
            // Lowered as:
            //   { data_ptr, len, cap }
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
            auto lenTy = mlir::IntegerType::get(ctx, 64);
            return mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, lenTy, lenTy});
        }

        case arklang::Type::Slice: {
            // Lowered as:
            //   { data_ptr, len }
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
            auto lenTy = mlir::IntegerType::get(ctx, 64);
            return mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, lenTy});
        }

        case arklang::Type::Tuple: {
            llvm::SmallVector<mlir::Type, 4> elements;
            for (const auto& sub : astType.subtypes) {
                elements.push_back(convertType(sub));
            }
            return mlir::LLVM::LLVMStructType::getLiteral(ctx, elements);
        }

        case arklang::Type::Schema:
        case arklang::Type::Generic: {
            // Schema and instantiated generic types are resolved through the
            // schema registry / instantiation path.
            const auto* info = getOrInstantiateSchema(astType.schemaName, astType.genericArgs);
            if (info) return info->loweredType;

            llvm::errs() << "[ERROR] convertType: Failed to resolve schema '"
                         << astType.schemaName << "'\n";
            return mlir::LLVM::LLVMPointerType::get(ctx);
        }

        default:
            break;
    }

    llvm::errs() << "[WARNING] convertType: Unknown type kind "
                 << static_cast<int>(astType.kind)
                 << ". Returning Unit/Void.\n";

    return mlir::LLVM::LLVMStructType::getLiteral(ctx, {});
}

// =============================================================================
// Type Coercion Helper
// =============================================================================
// Implements implicit value adaptation between already-lowered MLIR values.
//
// Preserved behavior:
// - null input -> hard error
// - same type -> no-op
// - MemRef involvement -> leave value unchanged
// - void source -> error
// - void target -> drop value and return empty
// - pointer/pointer supports addrspace adaptation
// - pointer/int and int/pointer conversions supported
// - float width conversions supported
// - integer extend/truncation supported
// - int/float conversions supported
// - equal-bitwidth LLVM-compatible values may bitcast
//
// NOTE:
// This is a type-policy function, not a generic helper. Changes here directly
// affect lowering semantics and ABI behavior.
// =============================================================================
mlir::Value GenMIR::coerce(mlir::OpBuilder& b,
                           mlir::Location loc,
                           mlir::Value val,
                           mlir::Type targetType) {
    if (!val) {
        mlir::emitError(loc) << "coerce: null input value (to " << targetType << ")";
        return {};
    }

    mlir::Type currentType = val.getType();
    if (currentType == targetType) return val;

    // -------------------------------------------------------------------------
    // MemRef guard
    // -------------------------------------------------------------------------
    // MemRefs are intentionally left alone here. They are not treated as generic
    // primitive LLVM values, and mismatches should be handled by specialized
    // container/tensor lowering logic rather than this coercion layer.
    // -------------------------------------------------------------------------
    if (llvm::isa<mlir::MemRefType>(currentType) || llvm::isa<mlir::MemRefType>(targetType)) {
        return val;
    }

    auto isVoidTy = [](mlir::Type ty) -> bool {
        return llvm::isa<mlir::LLVM::LLVMVoidType>(ty);
    };

    if (isVoidTy(currentType)) {
        mlir::emitError(loc) << "coerce: source type is void (invalid SSA value) (to "
                             << targetType << ")";
        return {};
    }

    if (isVoidTy(targetType)) {
        return {};
    }

    auto emitFail = [&](mlir::Twine msg) -> mlir::Value {
        mlir::emitError(loc) << "coerce: " << msg
                             << " (from " << currentType
                             << " to " << targetType << ")";
        return {};
    };

    // -------------------------------------------------------------------------
    // Pointer -> Pointer
    // -------------------------------------------------------------------------
    // Handle address-space adaptation explicitly. Under opaque pointers, if the
    // address spaces already match, no further cast is required.
    // -------------------------------------------------------------------------
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(currentType) &&
        llvm::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
        auto srcPtrTy = llvm::cast<mlir::LLVM::LLVMPointerType>(currentType);
        auto dstPtrTy = llvm::cast<mlir::LLVM::LLVMPointerType>(targetType);

        if (srcPtrTy.getAddressSpace() != dstPtrTy.getAddressSpace()) {
            return b.create<mlir::LLVM::AddrSpaceCastOp>(loc, dstPtrTy, val);
        }

        return val;
    }

    // Pointer -> Integer
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(currentType) && targetType.isInteger()) {
        return b.create<mlir::LLVM::PtrToIntOp>(loc, targetType, val);
    }

    // Integer -> Pointer
    if (currentType.isInteger() && llvm::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
        return b.create<mlir::LLVM::IntToPtrOp>(loc, targetType, val);
    }

    // F32 -> F64
    if (currentType.isF32() && targetType.isF64()) {
        return b.create<mlir::LLVM::FPExtOp>(loc, targetType, val);
    }

    // F64 -> F32
    if (currentType.isF64() && targetType.isF32()) {
        return b.create<mlir::LLVM::FPTruncOp>(loc, targetType, val);
    }

    // -------------------------------------------------------------------------
    // Integer -> Integer
    // -------------------------------------------------------------------------
    // Preserve current sign-extension policy:
    // - i1 extends with zero-extension
    // - wider signed integers extend with sign-extension
    // - narrowing truncates
    // -------------------------------------------------------------------------
    if (currentType.isInteger() && targetType.isInteger()) {
        unsigned srcW = currentType.getIntOrFloatBitWidth();
        unsigned dstW = targetType.getIntOrFloatBitWidth();

        if (dstW > srcW) {
            if (srcW == 1) {
                return b.create<mlir::LLVM::ZExtOp>(loc, targetType, val);
            }
            return b.create<mlir::LLVM::SExtOp>(loc, targetType, val);
        }

        if (dstW < srcW) {
            return b.create<mlir::LLVM::TruncOp>(loc, targetType, val);
        }

        return val;
    }

    // Integer -> Float
    if (llvm::isa<mlir::IntegerType>(currentType) && llvm::isa<mlir::FloatType>(targetType)) {
        return b.create<mlir::LLVM::SIToFPOp>(loc, targetType, val);
    }

    // Float -> Integer
    if (llvm::isa<mlir::FloatType>(currentType) && llvm::isa<mlir::IntegerType>(targetType)) {
        return b.create<mlir::LLVM::FPToSIOp>(loc, targetType, val);
    }

    // -------------------------------------------------------------------------
    // Equal-bitwidth LLVM-compatible fallback
    // -------------------------------------------------------------------------
    // This preserves the existing bitcast policy for LLVM-native or signless
    // scalar values only.
    // -------------------------------------------------------------------------
    auto isLLVMCompatible = [](mlir::Type t) {
        return t.getDialect().getNamespace() == "llvm" || t.isSignlessIntOrFloat();
    };

    if (isLLVMCompatible(currentType) && isLLVMCompatible(targetType)) {
        const unsigned srcBits = mlir::LLVM::getPrimitiveTypeSizeInBits(currentType);
        const unsigned dstBits = mlir::LLVM::getPrimitiveTypeSizeInBits(targetType);

        if (srcBits > 0 && srcBits == dstBits) {
            return b.create<mlir::LLVM::BitcastOp>(loc, targetType, val);
        }
    }

    return emitFail("no valid implicit coercion");
}

// =============================================================================
// Copy Semantics Check
// =============================================================================
// Classifies an Ark type as either:
//
// - Copy:
//     trivially duplicable value semantics
//
// - Move:
//     resource/container/object semantics that should not be blindly copied
//
// Preserved policy:
// - scalar / integer / float -> Copy
// - slice                    -> Copy
// - function                 -> Copy
// - tuple                    -> Copy iff every subtype is Copy
// - everything else          -> Move
// =============================================================================
bool GenMIR::isCopyType(const arklang::Type& t) {
    // Primitive value types are trivially copyable.
    if (t.isScalar() || t.isInteger() || t.isFloat()) return true;

    // Slices are views / references and are therefore copyable by handle.
    if (t.kind == arklang::Type::Slice) return true;

    // Functions lower as pointer-like callable references.
    if (t.kind == arklang::Type::Func) return true;

    // Tuples are copyable only when all components are copyable.
    if (t.kind == arklang::Type::Tuple) {
        for (const auto& sub : t.subtypes) {
            if (!isCopyType(sub)) return false;
        }
        return true;
    }

    // Default policy: containers, schemas, tensors, strings, etc. are move-only.
    return false;
}

} // namespace arklang