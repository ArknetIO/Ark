#pragma once

#include "mlir/Pass/Pass.h"
#include "ark/IR/ArkMirOps.h"

namespace arklang {
namespace mir {

// The "Borrow Checker" Pass
// verifies that no Read/Move/Drop operation consumes an Uninit or MaybeInit token.
std::unique_ptr<mlir::Pass> createOwnershipVerifierPass();

} // namespace mir
} // namespace arklang