#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace arklang {

// Creates the pass that lowers Ark MIR (erasing ownership tokens) to LLVM Dialect.
std::unique_ptr<mlir::Pass> createArkToLLVMPass();

} // namespace arklang