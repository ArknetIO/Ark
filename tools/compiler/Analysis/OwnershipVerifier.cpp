// Analysis/OwnershipVerifier.cpp
#include "Analysis/OwnershipVerifier.h"

#include "ark/IR/ArkMirOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace arklang::mir;

namespace {

enum class TokState : uint8_t { Bottom, Init, Uninit, Maybe };
enum class Mark : uint8_t { InProgress, Done };

static FailureOr<int64_t> getStateId(Value stateTok, Operation *at) {
    auto stTy = dyn_cast<StateType>(stateTok.getType());
    if (!stTy) {
        at->emitError("expected !ark.mir.state, got ") << stateTok.getType();
        return failure();
    }
    return stTy.getId();
}

static FailureOr<int64_t> getPlaceId(Value place, Operation *at) {
    auto pTy = dyn_cast<PlaceType>(place.getType());
    if (!pTy) {
        at->emitError("expected !ark.mir.place, got ") << place.getType();
        return failure();
    }
    return pTy.getId();
}

static LogicalResult verifyIdMatch(Value place, Value stateTok, Operation *at) {
    auto pid = getPlaceId(place, at);
    auto sid = getStateId(stateTok, at);
    if (failed(pid) || failed(sid)) return failure();
    if (*pid != *sid) {
        at->emitError("ownership id mismatch: place<") << *pid << "> with state<" << *sid << ">";
        return failure();
    }
    return success();
}

static TokState joinTok(TokState a, TokState b) {
    if (a == TokState::Bottom) return b;
    if (b == TokState::Bottom) return a;
    if (a == b) return a;
    return TokState::Maybe;
}

static FailureOr<OperandRange> getEdgeOperands(Operation *term, unsigned succIndex, Operation *at) {
    if (auto branch = dyn_cast<BranchOpInterface>(term)) {
        return branch.getSuccessorOperands(succIndex).getForwardedOperands();
    }
    at->emitError("terminator does not implement BranchOpInterface: ") << term->getName();
    return failure();
}

static FailureOr<TokState> classifyToken(Value tok,
                                         Operation *userOp,
                                         llvm::DenseMap<Value, Mark> &marks,
                                         llvm::DenseMap<Value, TokState> &memo);

static FailureOr<TokState> classifyBlockArg(BlockArgument barg,
                                            Operation *userOp,
                                            llvm::DenseMap<Value, Mark> &marks,
                                            llvm::DenseMap<Value, TokState> &memo) {
    Block *blk = barg.getOwner();
    const unsigned argNo = barg.getArgNumber();

    if (blk->isEntryBlock()) return TokState::Bottom;

    TokState acc = TokState::Bottom;
    bool sawAnyEdge = false;
    bool sawAnyIncoming = false;

    for (Block *pred : blk->getPredecessors()) {
        Operation *term = pred->getTerminator();
        if (!term) {
            userOp->emitError("CFG malformed: predecessor has no terminator");
            return failure();
        }

        bool mappedPred = false;

        for (unsigned si = 0, se = term->getNumSuccessors(); si != se; ++si) {
            if (term->getSuccessor(si) != blk) continue;
            mappedPred = true;
            sawAnyEdge = true;

            auto edgeOpsOr = getEdgeOperands(term, si, userOp);
            if (failed(edgeOpsOr)) return failure();
            OperandRange edgeOps = *edgeOpsOr;

            if (argNo >= edgeOps.size()) {
                userOp->emitError("phi operand index out of range on incoming edge from terminator ")
                    << term->getName();
                return failure();
            }

            Value incoming = edgeOps[argNo];

            if (incoming == barg) continue;

            auto st = classifyToken(incoming, userOp, marks, memo);
            if (failed(st)) return failure();

            acc = joinTok(acc, *st);
            sawAnyIncoming = true;
        }

        if (!mappedPred) {
            userOp->emitError("CFG edge mapping failed for predecessor terminator ")
                << term->getName();
            return failure();
        }
    }

    if (!sawAnyEdge || !sawAnyIncoming) return TokState::Bottom;
    return acc;
}

static FailureOr<TokState> classifyToken(Value tok,
                                         Operation *userOp,
                                         llvm::DenseMap<Value, Mark> &marks,
                                         llvm::DenseMap<Value, TokState> &memo) {
    if (!tok) return TokState::Bottom;
    if (auto it = memo.find(tok); it != memo.end()) return it->second;

    auto mIt = marks.find(tok);
    if (mIt != marks.end() && mIt->second == Mark::InProgress) {
        return TokState::Bottom;
    }

    marks[tok] = Mark::InProgress;

    TokState result = TokState::Bottom;

    if (auto barg = dyn_cast<BlockArgument>(tok)) {
        auto r = classifyBlockArg(barg, userOp, marks, memo);
        if (failed(r)) return failure();
        result = *r;
    } else {
        Operation *def = tok.getDefiningOp();
        if (!def) {
            result = TokState::Bottom;
        } else if (isa<SlotOp>(def)) {
            result = TokState::Uninit;
        } else if (isa<StoreOp>(def)) {
            result = TokState::Init;
        } else if (isa<MoveOutOp>(def) || isa<DropOp>(def)) {
            result = TokState::Uninit;
        } else if (auto rd = dyn_cast<ReadOp>(def)) {
            auto r = classifyToken(rd.getStateIn(), userOp, marks, memo);
            if (failed(r)) return failure();
            result = *r;
        } else {
            result = TokState::Maybe;
        }
    }

    marks[tok] = Mark::Done;
    memo[tok] = result;
    return result;
}

static LogicalResult requireInit(Value place, Value stateIn, Operation *op, StringRef opName) {
    if (failed(verifyIdMatch(place, stateIn, op))) return failure();

    llvm::DenseMap<Value, Mark> marks;
    llvm::DenseMap<Value, TokState> memo;

    auto stOr = classifyToken(stateIn, op, marks, memo);
    if (failed(stOr)) return failure();

    switch (*stOr) {
        case TokState::Init:
            return success();
        case TokState::Uninit:
            op->emitError() << "use of uninitialized/moved-from value in '" << opName << "'";
            return failure();
        case TokState::Maybe:
        case TokState::Bottom:
            op->emitError() << "use of conditionally-initialized value (MaybeInit) in '" << opName << "'";
            return failure();
    }

    op->emitError() << "internal error: unknown token state for '" << opName << "'";
    return failure();
}

class OwnershipVerifierPass final
    : public PassWrapper<OwnershipVerifierPass, OperationPass<func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipVerifierPass)

    StringRef getArgument() const override { return "ark-verify-ownership"; }
    StringRef getDescription() const override { return "Verifies ark_mir ownership state tokens"; }

    void runOnOperation() override {
        func::FuncOp fn = getOperation();
        bool anyFail = false;

        fn.walk([&](Operation *op) {
            if (auto rd = dyn_cast<ReadOp>(op)) {
                if (failed(requireInit(rd.getPlace(), rd.getStateIn(), op, "read"))) anyFail = true;
                return;
            }
            if (auto mv = dyn_cast<MoveOutOp>(op)) {
                if (failed(requireInit(mv.getPlace(), mv.getStateIn(), op, "move_out"))) anyFail = true;
                return;
            }
            if (auto dr = dyn_cast<DropOp>(op)) {
                if (failed(requireInit(dr.getPlace(), dr.getStateIn(), op, "drop"))) anyFail = true;
                return;
            }
            if (auto st = dyn_cast<StoreOp>(op)) {
                if (failed(verifyIdMatch(st.getPlace(), st.getStateIn(), op))) anyFail = true;
                return;
            }
        });

        if (anyFail) signalPassFailure();
    }
};

} // namespace

namespace arklang::mir {

std::unique_ptr<mlir::Pass> createOwnershipVerifierPass() {
    return std::make_unique<OwnershipVerifierPass>();
}

} // namespace arklang::mir
