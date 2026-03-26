// tools/compiler/Transforms/Passes.h
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace arklang {
namespace mir {

// Creates the RAII Pass: Automatically inserts drops for initialized values
// going out of scope or being overwritten.
std::unique_ptr<mlir::Pass> createDropInsertionPass();

} // namespace mir
} // namespace arklang

// --- NEW: Ark Compiler Transforms ---
namespace ark::compiler::transforms {

// Creates the pass that strictly enforces the host-device C ABI
// contract before lowering to LLVM IR.
std::unique_ptr<mlir::Pass> createRuntimeABIVerifierPass();

} // namespace ark::compiler::transforms