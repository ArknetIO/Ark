// Analysis/OwnershipUtils.cpp

#include "ark/compiler/Analysis/OwnershipUtils.hpp"

#include "ark/IR/ArkMirOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

using namespace mlir;
using namespace arklang::mir;

namespace {

static FailureOr<int64_t> getStateId(Value stateTok, Operation *at) {
    auto stTy = dyn_cast<StateType>(stateTok.getType());
    if (!stTy) {
        at->emitError("expected !ark.mir.state token, got ") << stateTok.getType();
        return failure();
    }
    return stTy.getId();
}

static FailureOr<int64_t> getPlaceId(Value place, Operation *at) {
    auto pTy = dyn_cast<PlaceType>(place.getType());
    if (!pTy) {
        at->emitError("expected !ark.mir.place value, got ") << place.getType();
        return failure();
    }
    return pTy.getId();
}

static TokState joinTok(TokState a, TokState b) {
    if (a == TokState::Unknown) return b;
    if (b == TokState::Unknown) return a;
    if (a == b) return a;
    return TokState::Maybe;
}

} // namespace

namespace arklang::mir {

LogicalResult verifyIdMatch(Value place, Value stateTok, Operation *at) {
    auto pid = getPlaceId(place, at);
    auto sid = getStateId(stateTok, at);
    if (failed(pid) || failed(sid)) return failure();

    if (*pid != *sid) {
        at->emitError("ownership ID mismatch: place<") << *pid << "> used with state<" << *sid << ">";
        return failure();
    }
    return success();
}

FailureOr<TokState> TokClassifier::classify(Value tok, Operation *at) {
    if (!tok) return TokState::Maybe;

    auto [it, inserted] = memo.try_emplace(tok, TokState::Unknown);
    TokState &slot = it->second;

    auto mit = marks.find(tok);
    if (mit != marks.end() && mit->second == Mark::InProgress) {
        return slot;
    }

    marks[tok] = Mark::InProgress;

    TokState computed = TokState::Maybe;

    if (auto barg = dyn_cast<BlockArgument>(tok)) {
        auto r = classifyBlockArg(barg, at);
        if (failed(r)) return failure();
        computed = *r;
    } else {
        Operation *def = tok.getDefiningOp();
        if (!def) {
            computed = TokState::Maybe;
        } else if (isa<SlotOp>(def)) {
            computed = TokState::Uninit;
        } else if (isa<StoreOp>(def)) {
            computed = TokState::Init;
        } else if (isa<MoveOutOp>(def) || isa<DropOp>(def)) {
            computed = TokState::Uninit;
        } else if (auto rd = dyn_cast<ReadOp>(def)) {
            auto r = classify(rd.getStateIn(), at);
            if (failed(r)) return failure();
            computed = *r;
        } else {
            computed = TokState::Maybe;
        }
    }

    marks[tok] = Mark::Done;

    slot = joinTok(slot, computed);
    return slot;
}

FailureOr<TokState> TokClassifier::classifyBlockArg(BlockArgument barg, Operation *at) {
    Block *blk = barg.getOwner();
    const unsigned argNo = barg.getArgNumber();

    if (blk->isEntryBlock()) return TokState::Maybe;

    TokState acc = TokState::Maybe;
    bool sawEdge = false;
    bool first = true;

    for (Block *pred : blk->getPredecessors()) {
        Operation *term = pred->getTerminator();
        if (!term) {
            at->emitError("malformed CFG: predecessor has no terminator");
            return failure();
        }

        auto branchOp = dyn_cast<BranchOpInterface>(term);
        if (!branchOp) {
            at->emitError("malformed CFG: terminator does not implement BranchOpInterface");
            return failure();
        }

        bool mappedThisPred = false;

        for (unsigned si = 0, se = term->getNumSuccessors(); si != se; ++si) {
            if (term->getSuccessor(si) != blk) continue;

            OperandRange edgeOps = branchOp.getSuccessorOperands(si).getForwardedOperands();
            if (argNo >= edgeOps.size()) {
                at->emitError("malformed CFG: block-arg index out of range on edge");
                return failure();
            }

            Value incoming = edgeOps[argNo];

            // Ignore trivial self-cycle on backedges: %arg coming from itself.
            // This prevents loops that carry an unchanged token from collapsing to Maybe.
            if (incoming == barg) {
                mappedThisPred = true;
                sawEdge = true;
                continue;
            }

            auto st = classify(incoming, at);
            if (failed(st)) return failure();

            if (first) {
                acc = *st;
                first = false;
            } else {
                acc = joinTok(acc, *st);
            }

            mappedThisPred = true;
            sawEdge = true;
        }

        if (!mappedThisPred) {
            at->emitError("CFG mapping failed: predecessor does not link to phi owner");
            return failure();
        }
    }

    if (!sawEdge) return TokState::Maybe;
    if (first) return TokState::Maybe;
    return acc;
}

} // namespace arklang::mir
