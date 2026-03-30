#pragma once

#include "ark/IR/ArkMirOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace arklang::mir {

enum class TokState : uint8_t { Unknown, Init, Uninit, Maybe };

// Memoized classifier for SSA tokens
struct TokClassifier {
    enum class Mark : uint8_t { InProgress, Done };

    llvm::DenseMap<mlir::Value, Mark> marks;
    llvm::DenseMap<mlir::Value, TokState> memo;

    mlir::FailureOr<TokState> classify(mlir::Value tok, mlir::Operation *at);

private:
    mlir::FailureOr<TokState> classifyBlockArg(mlir::BlockArgument barg, mlir::Operation *at);
};

// Helper to check ID compatibility
mlir::LogicalResult verifyIdMatch(mlir::Value place, mlir::Value stateTok, mlir::Operation *at);

} // namespace arklang::mir