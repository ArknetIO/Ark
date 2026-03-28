#include "Pipeline/JIT.h"

#include <cerrno>

namespace ark::compiler::pipeline {

int JIT::Run(mlir::ModuleOp, llvm::StringRef) {
    return ENOTSUP;
}

} // namespace ark::compiler::pipeline