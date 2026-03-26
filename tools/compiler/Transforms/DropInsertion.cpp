#include "Transforms/Passes.h"

#include "Analysis/OwnershipUtils.h"
#include "ark/IR/ArkMirOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h" // [NEW] For logging

using namespace mlir;
using namespace arklang::mir;

namespace {

using SlotId = int64_t;
using StateMap = llvm::DenseMap<SlotId, Value>;
using IdSet = llvm::DenseSet<SlotId>;

// =============================================================================
// Logging Helper
// =============================================================================
static void logState(StringRef phase, Block *b, SlotId id, Value v) {
    // Uncomment to enable verbose logging
    // llvm::errs() << "[" << phase << "] Block " << b << " | Slot " << id << " -> ";
    // if (!v) llvm::errs() << "null\n";
    // else llvm::errs() << v << "\n";
}

// =============================================================================
// Helpers
// =============================================================================

static FailureOr<SlotId> getStateTypeId(Type ty, Operation *at) {
    auto stTy = dyn_cast<StateType>(ty);
    if (!stTy) {
        at->emitError("expected !ark.mir.state, got ") << ty;
        return failure();
    }
    return stTy.getId();
}

static FailureOr<SlotId> getStateTokId(Value tok, Operation *at) {
    if (!tok) {
        at->emitError("missing state token");
        return failure();
    }
    return getStateTypeId(tok.getType(), at);
}

static bool mapsEqualBySet(const StateMap &a, const StateMap &b, const IdSet &ids) {
    for (SlotId id : ids) {
        auto ia = a.find(id);
        auto ib = b.find(id);
        Value va = (ia == a.end()) ? Value{} : ia->second;
        Value vb = (ib == b.end()) ? Value{} : ib->second;
        if (va != vb) return false;
    }
    return true;
}

static void scanInstructions(Block &blk, Operation *limitOp, StateMap &cur) {
    for (Operation &op : blk) {
        if (&op == limitOp) break;

        if (auto slot = dyn_cast<SlotOp>(op)) {
            auto idOr = getStateTokId(slot.getState(), &op);
            if (succeeded(idOr)) cur[*idOr] = slot.getState();
            continue;
        }
        if (auto st = dyn_cast<StoreOp>(op)) {
            auto idOr = getStateTokId(st.getStateOut(), &op);
            if (succeeded(idOr)) cur[*idOr] = st.getStateOut();
            continue;
        }
        if (auto rd = dyn_cast<ReadOp>(op)) {
            auto idOr = getStateTokId(rd.getStateOut(), &op);
            if (succeeded(idOr)) cur[*idOr] = rd.getStateOut();
            continue;
        }
        if (auto mv = dyn_cast<MoveOutOp>(op)) {
            auto idOr = getStateTokId(mv.getStateOut(), &op);
            if (succeeded(idOr)) cur[*idOr] = mv.getStateOut();
            continue;
        }
        if (auto dr = dyn_cast<DropOp>(op)) {
            auto idOr = getStateTokId(dr.getStateOut(), &op);
            if (succeeded(idOr)) cur[*idOr] = dr.getStateOut();
            continue;
        }
    }
}

// =============================================================================
// State Liveness
// =============================================================================

static void computeStateLiveness(func::FuncOp fn,
                                 llvm::ArrayRef<SlotId> allIds,
                                 llvm::DenseMap<Block *, IdSet> &liveIn,
                                 llvm::DenseMap<Block *, IdSet> &liveOut) {
    llvm::DenseMap<Block *, IdSet> use;
    llvm::DenseMap<Block *, IdSet> def;

    for (Block &blk : fn.getBody()) {
        IdSet seen;

        // Block args define (phi-like defs).
        for (BlockArgument a : blk.getArguments()) {
            if (auto stTy = dyn_cast<StateType>(a.getType())) {
                SlotId id = stTy.getId();
                def[&blk].insert(id);
                seen.insert(id);
            }
        }

        // Scan ops in order: operand uses before result defs.
        for (Operation &op : blk) {
            for (Value v : op.getOperands()) {
                if (auto stTy = dyn_cast<StateType>(v.getType())) {
                    SlotId id = stTy.getId();
                    if (!seen.contains(id)) use[&blk].insert(id);
                }
            }
            for (Value r : op.getResults()) {
                if (auto stTy = dyn_cast<StateType>(r.getType())) {
                    SlotId id = stTy.getId();
                    def[&blk].insert(id);
                    seen.insert(id);
                }
            }
        }

        liveIn[&blk].clear();
        liveOut[&blk].clear();
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (Block &blk : llvm::reverse(fn.getBody())) {
            IdSet newOut;
            if (Operation *term = blk.getTerminator()) {
                for (Block *succ : term->getSuccessors()) {
                    auto it = liveIn.find(succ);
                    if (it == liveIn.end()) continue;
                    for (SlotId id : it->second) newOut.insert(id);
                }
            }

            IdSet newIn = use[&blk];
            for (SlotId id : newOut) {
                if (!def[&blk].contains(id)) newIn.insert(id);
            }

            if (newIn != liveIn[&blk]) {
                liveIn[&blk] = std::move(newIn);
                changed = true;
            }
            if (newOut != liveOut[&blk]) {
                liveOut[&blk] = std::move(newOut);
                changed = true;
            }
        }
    }

    (void)allIds;
}

// =============================================================================
// Dataflow
// =============================================================================

struct BlockSummary {
    StateMap lastDefs;
};

static LogicalResult computeBlockStates(func::FuncOp fn,
                                        llvm::ArrayRef<SlotId> slotIds,
                                        llvm::DenseMap<Block *, StateMap> &inState,
                                        llvm::DenseMap<Block *, StateMap> &outState) {
    if (fn.isExternal() || fn.getBody().empty()) return success();

    llvm::DenseMap<Block *, IdSet> liveIn;
    llvm::DenseMap<Block *, IdSet> liveOut;
    computeStateLiveness(fn, slotIds, liveIn, liveOut);

    llvm::DenseMap<Block *, BlockSummary> summaries;
    llvm::DenseMap<Block *, llvm::DenseSet<SlotId>> phiIds;
    phiIds.clear();

    for (Block &block : fn.getBody()) {
        BlockSummary &summary = summaries[&block];

        llvm::DenseSet<SlotId> &phis = phiIds[&block];
        for (BlockArgument arg : block.getArguments()) {
            if (auto stTy = dyn_cast<StateType>(arg.getType())) {
                const SlotId id = stTy.getId();
                phis.insert(id);
                summary.lastDefs[id] = arg;
            }
        }

        scanInstructions(block, nullptr, summary.lastDefs);

        inState[&block]  = StateMap{};
        outState[&block] = StateMap{};
    }

    llvm::SmallVector<Block *, 32> worklist;
    llvm::DenseSet<Block *> inWorklist;

    auto pushWL = [&](Block *b) {
        if (inWorklist.insert(b).second) worklist.push_back(b);
    };

    pushWL(&fn.getBody().front());

    llvm::DenseSet<Block *> processed;

    while (!worklist.empty()) {
        Block *blk = worklist.pop_back_val();
        inWorklist.erase(blk);

        const IdSet &needIn  = liveIn[blk];
        const IdSet &needOut = liveOut[blk];

        StateMap newIn;

        if (!blk->isEntryBlock()) {
            llvm::SmallVector<Block *, 8> preds(blk->getPredecessors());
            const llvm::DenseSet<SlotId> &phis = phiIds.lookup(blk);

            for (SlotId id : slotIds) {
                if (!needIn.contains(id)) continue;
                if (phis.contains(id)) continue;
                if (preds.empty()) continue;

                bool allReady = true;
                bool firstFound = false;
                bool anyMissing = false;
                Value v0;

                for (Block *p : preds) {
                    if (!processed.contains(p)) { allReady = false; break; }

                    auto &pOut = outState[p];
                    auto it = pOut.find(id);
                    Value v = (it == pOut.end()) ? Value{} : it->second;

                    if (!v) { anyMissing = true; continue; }

                    if (!firstFound) {
                        v0 = v;
                        firstFound = true;
                    } else if (v != v0) {
                        blk->getParentOp()->emitError("missing state phi for slot id ") << id
                            << " (implicit flow mismatch)";
                        return failure();
                    }
                }

                if (!allReady) continue;

                if (firstFound) {
                    if (anyMissing) {
                        blk->getParentOp()->emitError("missing state for slot id ") << id
                            << " on one predecessor";
                        return failure();
                    }
                    newIn[id] = v0;
                }
            }
        }

        bool inChanged = !mapsEqualBySet(newIn, inState[blk], needIn);
        if (inChanged) inState[blk] = newIn;

        StateMap newOut;
        const BlockSummary &summary = summaries[blk];

        for (SlotId id : slotIds) {
            if (!needOut.contains(id)) continue;

            if (summary.lastDefs.count(id)) newOut[id] = summary.lastDefs.lookup(id);
            else {
                auto it = newIn.find(id);
                newOut[id] = (it == newIn.end()) ? Value{} : it->second;
            }
        }

        bool outChanged = !mapsEqualBySet(newOut, outState[blk], needOut);
        if (outChanged) {
            outState[blk] = newOut;
            if (auto *term = blk->getTerminator()) {
                for (auto *succ : term->getSuccessors()) pushWL(succ);
            }
        }

        processed.insert(blk);
    }

    return success();
}

// =============================================================================
// Pass
// =============================================================================

class DropInsertionPass : public PassWrapper<DropInsertionPass, OperationPass<func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropInsertionPass)

    StringRef getArgument() const override { return "ark-drop-insertion"; }
    StringRef getDescription() const override {
        return "Inserts RAII drops (overwrite + return) for ark_mir token SSA";
    }

    void runOnOperation() override {
        func::FuncOp fn = getOperation();
        if (fn.isExternal()) return;

        // [LOG] Start of Pass
        // llvm::errs() << "\n=== DropInsertionPass: " << fn.getName() << " ===\n";

        OpBuilder b(fn.getContext());

        llvm::DenseMap<SlotId, Value> slotPlace;
        llvm::DenseMap<SlotId, Value> slotInitState;
        llvm::SmallVector<SlotId, 32> slotIds;

        bool failedAny = false;

        fn.walk([&](SlotOp slot) {
            auto idOr = getStateTokId(slot.getState(), slot);
            if (failed(idOr)) { failedAny = true; return; }
            const SlotId id = *idOr;

            if (!slotPlace.count(id)) slotIds.push_back(id);
            slotPlace[id] = slot.getPlace();
            slotInitState[id] = slot.getState();
            
            // [LOG] Registered Slot
            // llvm::errs() << "  Slot[" << id << "] defined by " << slot << "\n";
        });

        if (failedAny) { signalPassFailure(); return; }

        llvm::sort(slotIds);
        slotIds.erase(std::unique(slotIds.begin(), slotIds.end()), slotIds.end());

        llvm::DenseMap<Block *, StateMap> inState;
        llvm::DenseMap<Block *, StateMap> outState;

        if (failed(computeBlockStates(fn, slotIds, inState, outState))) {
            signalPassFailure();
            return;
        }

        // Phase 1: Drop-on-Overwrite
        fn.walk([&](StoreOp st) {
            // [LOG] Checking Store
            // llvm::errs() << "  StoreOp: " << st << "\n";
            
            if (failed(verifyIdMatch(st.getPlace(), st.getStateIn(), st))) {
                failedAny = true; return;
            }

            TokClassifier cls;
            auto status = cls.classify(st.getStateIn(), st);
            
            if (failed(status)) { 
                // llvm::errs() << "    -> Classification Failed!\n";
                failedAny = true; return; 
            }

            // llvm::errs() << "    -> State: " << (int)*status << "\n";

            if (*status == TokState::Init) {
                b.setInsertionPoint(st);
                auto drop = b.create<DropOp>(st.getLoc(), st.getStateIn().getType(),
                                             st.getPlace(), st.getStateIn());
                st.getStateInMutable().assign(drop.getStateOut());
            }
        });

        if (failedAny) { signalPassFailure(); return; }

        // Phase 2: Drop-on-Return
        fn.walk([&](func::ReturnOp ret) {
            Block *blk = ret->getBlock();

            StateMap cur;
            if (auto it = inState.find(blk); it != inState.end()) cur = it->second;

            scanInstructions(*blk, ret, cur);

            b.setInsertionPoint(ret);

            for (SlotId id : slotIds) {
                Value place = slotPlace.lookup(id);
                if (!place) continue;

                Value tok;
                if (auto it = cur.find(id); it != cur.end()) tok = it->second;
                if (!tok) tok = slotInitState.lookup(id);
                if (!tok) continue;

                if (failed(verifyIdMatch(place, tok, ret))) { failedAny = true; continue; }

                TokClassifier cls;
                auto status = cls.classify(tok, ret);
                if (failed(status)) { failedAny = true; continue; }

                if (*status == TokState::Init) {
                    // llvm::errs() << "  Inserting Drop at Return for Slot[" << id << "]\n";
                    b.create<DropOp>(ret.getLoc(), tok.getType(), place, tok);
                }
            }
        });

        if (failedAny) signalPassFailure();
    }
};

} // namespace

namespace arklang::mir {
std::unique_ptr<mlir::Pass> createDropInsertionPass() {
    return std::make_unique<DropInsertionPass>();
}
} // namespace arklang::mir