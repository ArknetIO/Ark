#pragma once

#include "ark/IR/ArkMirTypes.h"

// Core op infra
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

// Interfaces used by generated code / traits
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"

// Bytecode hooks
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectResourceBlobManager.h"

// --- THE FIX ---
// The build system puts this file in the root of the "IR" build artifact folder.
// We added that folder to the search path, so we include it by name only.
#define GET_OP_CLASSES
#include "ArkMirOps.h.inc"   // <--- Path Removed
#undef GET_OP_CLASSES