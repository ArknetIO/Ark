#include "ark/IR/ArkMirOps.h" // Source file: Full path is correct

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace arklang::mir;

// --- FIX START ---
// Drop "ark/IR/" from all .inc includes below

#include "ArkMirDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ArkMirTypes.cpp.inc"

void ArkMirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ArkMirOps.cpp.inc"
  >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "ArkMirTypes.cpp.inc"
  >();
}
// --- FIX END ---