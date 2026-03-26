#include "ark/IR/ArkMirOps.h"
#include "ark/IR/ArkMirTypes.h" // Definitions for VecType, StructType, etc.

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
// [NEW] Required for traits defined in .td (CallInterfaces, SideEffectInterfaces)
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace arklang::mir;

// =============================================================================
// Helper: Extract Pointee
// =============================================================================
static mlir::Type getPointee(mlir::Type t) {
  if (auto pt = llvm::dyn_cast<PlaceType>(t)) return pt.getPointee();
  return {};
}

static mlir::LogicalResult verifyPlaceValueType(mlir::Operation *op,
                                                mlir::Type placeTy,
                                                mlir::Type valueTy,
                                                llvm::StringRef what) {
  mlir::Type p = getPointee(placeTy);
  if (!p) return op->emitOpError() << what << ": operand must be !ark.mir.place<...>";
  
  // Strict type checking: GenMIR is responsible for inserting casts/coercions.
  if (p != valueTy)
    return op->emitOpError() << what << ": type mismatch: place<" << p << "> vs value " << valueTy;
    
  return mlir::success();
}

// =============================================================================
// Op Verifiers
// =============================================================================

mlir::LogicalResult ReadOp::verify() {
  return verifyPlaceValueType(getOperation(), getPlace().getType(), getValue().getType(), "read");
}

mlir::LogicalResult MoveOutOp::verify() {
  return verifyPlaceValueType(getOperation(), getPlace().getType(), getValue().getType(), "move_out");
}

mlir::LogicalResult StoreOp::verify() {
  return verifyPlaceValueType(getOperation(), getPlace().getType(), getValue().getType(), "store");
}

mlir::LogicalResult DropOp::verify() {
  mlir::Type p = getPointee(getPlace().getType());
  if (!p) return emitOpError("drop: operand must be a valid place");
  return mlir::success();
}

mlir::LogicalResult IndexOp::verify() {
  mlir::Type baseP = getPointee(getBase().getType());
  if (!baseP) return emitOpError() << "index: base must be !ark.mir.place<...>";

  mlir::Type expectedElemTy;
  bool isStruct = false;

  if (auto v = llvm::dyn_cast<VecType>(baseP)) {
    expectedElemTy = v.getElementType();
  } else if (auto s = llvm::dyn_cast<SliceType>(baseP)) {
    expectedElemTy = s.getElementType();
  } else if (auto st = llvm::dyn_cast<StructType>(baseP)) {
    // For structs, the dialect type (!ark.struct<"Name">) does not contain field info.
    // The field info lives in the GenMIR registry. We assume the generator is correct here.
    isStruct = true;
  } else {
    return emitOpError() << "index: requires !ark.vec<T>, !ark.slice<T> or !ark.struct<...>, got " << baseP;
  }

  // Only verify element type equality for Vec/Slice where the inner type is explicit in the IR
  if (!isStruct) {
    mlir::Type actualElemTy = getPointee(getElem().getType());
    if (expectedElemTy != actualElemTy) {
      return emitOpError() << "index: result place type mismatch: expected " << expectedElemTy
                           << ", got " << actualElemTy;
    }
  }
  
  return mlir::success();
}

mlir::LogicalResult AllocOp::verify() {
  auto memRefTy = llvm::dyn_cast<mlir::MemRefType>(getResult().getType());
  if (!memRefTy) return emitOpError("result must be a memref type");

  if (memRefTy.getNumDynamicDims() != getSizes().size()) {
    return emitOpError() << "dimension mismatch: result type " << memRefTy
                         << " has " << memRefTy.getNumDynamicDims()
                         << " dynamic dimensions, but " << getSizes().size()
                         << " size operands were provided";
  }
  return mlir::success();
}

#define GET_OP_CLASSES
#include "ArkMirOps.cpp.inc" 
#undef GET_OP_CLASSES