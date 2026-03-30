#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <llvm/ADT/StringRef.h>

namespace ark::compiler::pipeline {

class JIT {
public:
    // Runs the given MLIR module using the LLVM JIT engine.
    // - runtimeDir: Path to look for runtime shared libraries (if needed).
    // Returns the exit code of the program's 'main' function.
    static int Run(mlir::ModuleOp module, llvm::StringRef runtimeDir);
};

} // namespace ark::compiler::pipeline