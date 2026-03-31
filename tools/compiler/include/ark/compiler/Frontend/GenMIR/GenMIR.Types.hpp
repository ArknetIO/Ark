#pragma once

#include "ark/compiler/Frontend/GenMIR.hpp"

#include "mlir/IR/BuiltinTypes.h"

#include <string>

namespace arklang::mir {

// =============================================================================
// GenMIR Type Helpers
// =============================================================================
// This header owns free helper functions used by the extracted type-lowering
// layer. GenMIR member function declarations remain in GenMIR.hpp.
//
// Implementations live in:
//   src/Frontend/GenMIR/GenMIR.Types.cpp
// =============================================================================

// -----------------------------------------------------------------------------
// MLIR Type Formatting
// -----------------------------------------------------------------------------
// Diagnostic helper that renders an MLIR type exactly as MLIR prints it.
std::string typeToString(mlir::Type type);

// -----------------------------------------------------------------------------
// Ark Type Classification
// -----------------------------------------------------------------------------
// Returns true when the Ark type should be treated as tensor-like by the type
// lowering layer. This currently includes:
// - tensor<T>
// - Alloc<T> intrinsic container
bool isTensorType(const arklang::Type& ty);

} // namespace arklang::mir