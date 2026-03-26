#include "Frontend/MirBuilder.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace arklang{

// =============================================================================
// Static Helpers
// =============================================================================

static mlir::LLVM::LLVMPointerType ptrTy(mlir::MLIRContext *ctx) {
    return mlir::LLVM::LLVMPointerType::get(ctx);
}

// Strict DataLayout check: No silent fallback to 64-bit.
static mlir::Type intptrTy(mlir::ModuleOp m, mlir::MLIRContext *ctx) {
    mlir::DataLayout dl(m);
    const uint64_t bits = dl.getTypeSizeInBits(ptrTy(ctx));
    
    if (bits != 32 && bits != 64) {
        llvm::report_fatal_error("MirBuilder: missing or unsupported DataLayout (pointer size must be 32 or 64)");
    }
    return mlir::IntegerType::get(ctx, static_cast<unsigned>(bits));
}

// Create a constant integer of the CORRECT target width
static mlir::Value constInt(mlir::OpBuilder &b, mlir::Location loc, mlir::Type ty, uint64_t v) {
    auto it = llvm::dyn_cast<mlir::IntegerType>(ty);
    if (!it) llvm::report_fatal_error("MirBuilder: constInt requires integer type");
    return b.create<mlir::LLVM::ConstantOp>(loc, ty, b.getIntegerAttr(ty, v));
}

static mlir::Value constI1(mlir::OpBuilder &b, mlir::Location loc, bool v) {
    return b.create<mlir::LLVM::ConstantOp>(loc, b.getI1Type(), b.getBoolAttr(v));
}

static void requireI1OrDie(mlir::Value v, const char *msg) {
    if (!v || !v.getType().isInteger(1)) llvm::report_fatal_error(msg);
}

// Validates that the lowered type matches the AST expectation for Vec/Slice.
static void validateVecLikeLayoutOrDie(mlir::ModuleOp module,
                                       mlir::Type ipTy,
                                       const arklang::Type &astTy,
                                       mlir::Type valueTy) {
    if (astTy.kind != arklang::Type::Slice && astTy.kind != arklang::Type::Vec) return;

    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(valueTy);
    if (!st) llvm::report_fatal_error("MirBuilder: Vec/Slice must lower to LLVM struct");

    auto body = st.getBody();
    const bool isSlice = (astTy.kind == arklang::Type::Slice);
    const size_t want = isSlice ? 2 : 3;

    if (body.size() != want) {
        llvm::report_fatal_error("MirBuilder: Vec/Slice layout arity mismatch");
    }

    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(body[0])) {
        llvm::report_fatal_error("MirBuilder: Vec/Slice field 0 must be a pointer");
    }

    if (body[1] != ipTy) llvm::report_fatal_error("MirBuilder: Vec/Slice len must be intptr");
    if (!isSlice && body[2] != ipTy) llvm::report_fatal_error("MirBuilder: Vec cap must be intptr");
}

// =============================================================================
// MirBuilder Implementation
// =============================================================================

MirBuilder::MirBuilder(mlir::OpBuilder &builder,
                       mlir::ModuleOp module,
                       ConvertTypeFn convertTypeFn,
                       CoerceFn coerceFn,
                       IsCopyFn isCopyTypeFn)
    : builder(builder),
      module(module),
      convertType(std::move(convertTypeFn)),
      coerce(std::move(coerceFn)),
      isCopyType(std::move(isCopyTypeFn)),
      ipTy(intptrTy(module, builder.getContext()))
{
    if (!convertType || !coerce || !isCopyType)
        llvm::report_fatal_error("MirBuilder: null callback(s) passed to constructor");

    pushScope();
}


// --- Scope Management ---

void MirBuilder::enterFunction() {
    scopes.clear();
    pushScope();
}

void MirBuilder::exitFunction() {
    scopes.clear();
}

void MirBuilder::pushScope() { 
    scopes.emplace_back(); 
}

void MirBuilder::popScope() {
    if (scopes.empty()) llvm::report_fatal_error("MirBuilder: popScope underflow");
    scopes.pop_back();
}

bool MirBuilder::isDeclared(llvm::StringRef name) const {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        if (it->vars.count(name.str())) return true;
    }
    return false;
}

VarInfo* MirBuilder::lookup(llvm::StringRef name) {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        auto f = it->vars.find(name.str());
        if (f != it->vars.end()) return &f->second;
    }
    return nullptr;
}

// --- Core Operations ---

// =============================================================================
// [FIXED] Create Stack Slot (Supports CPU and GPU Functions)
// =============================================================================
mlir::Value MirBuilder::createSlot(mlir::Location loc, mlir::Type valueTy) {
    mlir::OpBuilder::InsertionGuard g(builder);

    mlir::Operation *parentOp = builder.getInsertionBlock()->getParentOp();
    mlir::Block *entryBlock = nullptr;

    // 1. Check for Standard LLVM Function (CPU)
    if (auto llvmFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(parentOp)) {
        entryBlock = &llvmFunc.getBody().front();
    } 
    // 2. Check for GPU Kernel Function (GPU) [FIX]
    // The GPU dialect uses 'gpu.func' which has a body region just like LLVM func.
    // Allocas here will be lowered to NVVM/ROCDL allocas (Local Memory).
    else if (auto gpuFunc = llvm::dyn_cast<mlir::gpu::GPUFuncOp>(parentOp)) {
        entryBlock = &gpuFunc.getBody().front();
    }
    // 3. Check for nested regions (e.g. scf.if / scf.for) inside a function
    else {
        auto llvmParent = parentOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
        if (llvmParent) {
            entryBlock = &llvmParent.getBody().front();
        } else {
            auto gpuParent = parentOp->getParentOfType<mlir::gpu::GPUFuncOp>();
            if (gpuParent) {
                entryBlock = &gpuParent.getBody().front();
            }
        }
    }
    
    if (!entryBlock) {
        llvm::report_fatal_error("MirBuilder: createSlot called outside LLVMFuncOp or GPUFuncOp");
    }

    // Insert allocas at the start of the entry block
    auto it = entryBlock->begin();
    while (it != entryBlock->end() && llvm::isa<mlir::LLVM::AllocaOp>(&*it)) ++it;
    
    builder.setInsertionPoint(entryBlock, it);

    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
        
    // Note: Use ptrTy(ctx) helper if available, or get from builder
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    return builder.create<mlir::LLVM::AllocaOp>(loc, ptrType, valueTy, one);
}

mlir::Value MirBuilder::normalizeIndex(mlir::Location loc, mlir::Value idx) {
    auto src = llvm::dyn_cast<mlir::IntegerType>(idx.getType());
    if (!src) llvm::report_fatal_error("MirBuilder: index must be integer");

    unsigned tgtW = llvm::cast<mlir::IntegerType>(ipTy).getWidth();
    unsigned srcW = src.getWidth();

    if (srcW == tgtW) return idx;
    if (srcW < tgtW) return builder.create<mlir::LLVM::ZExtOp>(loc, ipTy, idx);
    
    return builder.create<mlir::LLVM::TruncOp>(loc, ipTy, idx);
}

mlir::Type MirBuilder::deriveElemTy(const arklang::Type &astTy, mlir::Type valueTy) {
    if (auto arr = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(valueTy))
        return arr.getElementType();

    if (astTy.kind == arklang::Type::Vec || astTy.kind == arklang::Type::Slice) {
        if (astTy.genericArgs.empty()) {
            llvm::report_fatal_error("MirBuilder: Vec/Slice AST missing genericArgs (element type)");
        }
        return convertType(astTy.genericArgs[0]);
    }
    return {};
}

VarInfo& MirBuilder::declareLocal(mlir::Location loc,
                                  llvm::StringRef name,
                                  const arklang::Type &astTy,
                                  RValue init) {
    mlir::Type valueTy = convertType(astTy);
    
    validateVecLikeLayoutOrDie(module, ipTy, astTy, valueTy);
    
    mlir::Type elemTy = deriveElemTy(astTy, valueTy);

    mlir::Value place = createSlot(loc, valueTy);

    mlir::Value stored = coerce(loc, init.val, valueTy);
    builder.create<mlir::LLVM::StoreOp>(loc, stored, place);

    VarInfo info;
    info.place = place;
    info.valueTy = valueTy;
    info.elemTy = elemTy;
    info.astTy = astTy;
    info.state = init.state;
    
    return scopes.back().vars[name.str()] = info;
}

// =============================================================================
// Read Variable (Use-After-Move Protection)
// =============================================================================
RValue MirBuilder::readVar(mlir::Location loc, VarInfo &v) {
    requireI1OrDie(v.state, "MirBuilder: VarInfo.state must be i1");

    mlir::Value val;
    // [CRITICAL FIX] Check if the variable lives in a stack slot (pointer).
    // If it's a direct SSA value (like our new MemRef kernel arguments), use it directly.
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(v.place.getType())) {
        val = builder.create<mlir::LLVM::LoadOp>(loc, v.valueTy, v.place);
    } else {
        val = v.place; // It's already the value we need!
    }
    
    if (isCopyType(v.astTy)) {
        return RValue{val, constI1(builder, loc, true)};
    }

    // Move: Consume the variable
    v.state = constI1(builder, loc, false);
    return RValue{val, constI1(builder, loc, true)};
}

// =============================================================================
// Write Variable (Drop + Coerce + Store)
// =============================================================================
void MirBuilder::writeVar(mlir::Location loc, VarInfo &v, RValue rhs) {
    requireI1OrDie(v.state, "MirBuilder: VarInfo.state must be i1");
    requireI1OrDie(rhs.state, "MirBuilder: RValue.state must be i1");

    // [CRITICAL FIX] Prevent storing into a direct value (MemRef).
    // You cannot reassign the `out` argument itself inside a kernel. 
    // (You write to out[i], which is handled correctly by lowerAssign, not here).
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(v.place.getType())) {
        llvm::report_fatal_error("Cannot re-assign a direct GPU Kernel argument");
    }

    dropPlaceIfOwned(loc, v.astTy, v.place, v.state);
    
    mlir::Value stored = coerce(loc, rhs.val, v.valueTy);
    builder.create<mlir::LLVM::StoreOp>(loc, stored, v.place);
    
    v.state = rhs.state;
}


// [NEW] Manual Registration
void MirBuilder::defineVar(llvm::StringRef name, VarInfo info) {
    if (scopes.empty()) llvm::report_fatal_error("MirBuilder: defineVar with no active scope");
    scopes.back().vars[name.str()] = info;
}

// =============================================================================
// Element Access (Unified for Host Stack, Host Structs, and GPU Registers)
// =============================================================================
mlir::FailureOr<mlir::Value> MirBuilder::elementPtr(mlir::Location loc,
                                                    const VarInfo &v,
                                                    mlir::Value idx) {
    mlir::Value idxVal = coerce(loc, idx, builder.getI64Type());

    // 1. Determine Target Element Type
    mlir::Type elemLLVMTy = builder.getI8Type(); // Default fallback
    if (v.elemTy) {
        elemLLVMTy = v.elemTy;
    } else if (!v.astTy.genericArgs.empty()) {
        elemLLVMTy = convertType(v.astTy.genericArgs[0]);
    }

    // 2. Case A: Struct Containers (e.g., Vec<T>, Slice<T>)
    if (auto structTy = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(v.valueTy)) {
        if (structTy.getBody().empty() || !llvm::isa<mlir::LLVM::LLVMPointerType>(structTy.getBody()[0])) {
            return mlir::failure(); 
        }

        // [FIX] Extract the EXACT pointer type from the struct to preserve Address Space!
        mlir::Type actualPtrTy = structTy.getBody()[0];
        auto defaultPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        auto zero = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
        mlir::Value dataPtrPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, defaultPtrTy, v.valueTy, v.place, mlir::ValueRange{zero, zero}
        ).getResult();
        
        mlir::Value dataPtr = builder.create<mlir::LLVM::LoadOp>(loc, actualPtrTy, dataPtrPtr);
        
        return builder.create<mlir::LLVM::GEPOp>(
            loc, actualPtrTy, elemLLVMTy, dataPtr, mlir::ValueRange{idxVal}
        ).getResult(); 
    }

    // 3. Case B: Raw Pointers (e.g., Alloc<T>, Tensor kernel arguments)
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(v.valueTy)) {
        mlir::Value rawPtr;

        if (v.place.getDefiningOp<mlir::LLVM::AllocaOp>()) {
            // [FIX] Use v.valueTy to preserve the AS1 Global Memory marker during the Load!
            rawPtr = builder.create<mlir::LLVM::LoadOp>(loc, v.valueTy, v.place);
        } else {
            rawPtr = v.place;
        }

        // [FIX] Use v.valueTy to ensure the GEP result remains AS1 Global Memory!
        return builder.create<mlir::LLVM::GEPOp>(
            loc, v.valueTy, elemLLVMTy, rawPtr, mlir::ValueRange{idxVal}
        ).getResult();
    }

    return mlir::failure();
}


// =============================================================================
// Read Element (Copy-Only Safety)
// =============================================================================
RValue MirBuilder::readElem(mlir::Location loc, const VarInfo &v, mlir::Value idx) {
    // 1. Validate Type
    if (v.astTy.kind != arklang::Type::Vec && v.astTy.kind != arklang::Type::Slice)
        llvm::report_fatal_error("MirBuilder: readElem requires Vec/Slice AST type");

    if (v.astTy.genericArgs.empty())
        llvm::report_fatal_error("MirBuilder: readElem missing Vec/Slice element type");

    // 2. Enforce Copy Semantics (Partial Move not supported on vector access yet)
    const arklang::Type &elemAstTy = v.astTy.genericArgs[0];
    if (!isCopyType(elemAstTy))
        llvm::report_fatal_error("MirBuilder: moving/reading owned element by value is forbidden (use borrow/swap)");

    // 3. Calculate Address
    auto ptrOr = elementPtr(loc, v, idx);
    if (failed(ptrOr)) llvm::report_fatal_error("MirBuilder: elementPtr failed in readElem");

    // 4. Load
    //    We explicitly use the cached element type or fallback to i8.
    mlir::Type loadTy = v.elemTy ? v.elemTy : builder.getI8Type();
    mlir::Value val = builder.create<mlir::LLVM::LoadOp>(loc, loadTy, *ptrOr);
    
    // Return with Alive state
    mlir::Value trueState = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(true));
    return RValue{val, trueState};
}

// =============================================================================
// Write to Element (Assignment with Drop)
// =============================================================================
void MirBuilder::writeElem(mlir::Location loc, const VarInfo &baseVar, mlir::Value idx, RValue newVal) {
    // 1. Calculate Address (Internally)
    //    [FIX] We take 'idx', not 'elemAddr'. We must compute the address first.
    auto ptrOr = elementPtr(loc, baseVar, idx);
    if (failed(ptrOr)) llvm::report_fatal_error("MirBuilder: elementPtr failed in writeElem");
    mlir::Value addr = *ptrOr;

    // 2. Identify Element Type for Drop
    arklang::Type elemAstTy;
    if ((baseVar.astTy.kind == arklang::Type::Vec || baseVar.astTy.kind == arklang::Type::Slice) 
        && !baseVar.astTy.genericArgs.empty()) {
        elemAstTy = baseVar.astTy.genericArgs[0];
    } else {
        elemAstTy = {arklang::Type::I32}; // Fallback
    }

    // 3. Drop OLD Value (Assume Alive)
    //    If the container exists, its slots are initialized (or we are overwriting).
    mlir::Value aliveState = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(true));
    
    dropPlaceIfOwned(loc, elemAstTy, addr, aliveState);

    // 4. Store NEW Value
    //    StoreOp(Value, Pointer)
    builder.create<mlir::LLVM::StoreOp>(loc, newVal.val, addr);
}

// =============================================================================
// Helper: Check if type needs dropping (RAII)
// =============================================================================
bool MirBuilder::isDropNeeded(const arklang::Type &t) {
    // 1. Primitives and Views are trivial (Copy/No-op drop)
    if (t.isScalar() || t.isInteger() || t.isFloat()) return false;
    if (t.kind == arklang::Type::Func) return false;
    if (t.kind == arklang::Type::Slice) return false;

    // 2. Tuples: Need drop if any field needs drop
    if (t.kind == arklang::Type::Tuple) {
        for (const auto &sub : t.subtypes) {
            if (isDropNeeded(sub)) return true;
        }
        return false;
    }

    // 3. Resources (Vec, String, Schema)
    return true;
}

// =============================================================================
// Borrow Variable
// =============================================================================
RValue MirBuilder::borrowVar(mlir::Location loc, VarInfo &v) {
    if (!v.state || !v.state.getType().isInteger(1))
        llvm::report_fatal_error("MirBuilder::borrowVar: VarInfo.state must be i1");

    mlir::Value val;
    // [CRITICAL FIX] Same stack slot vs direct value check.
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(v.place.getType())) {
        val = builder.create<mlir::LLVM::LoadOp>(loc, v.valueTy, v.place).getResult();
    } else {
        val = v.place;
    }
    
    return RValue{val, v.state};
}

// =============================================================================
// Drop Glue Generator (Type-Directed)
// Host-only today. If invoked in GPU/device context, we hard-fail with a
// concrete diagnostic (no silent skipping).
// =============================================================================
void MirBuilder::dropPlaceIfOwned(mlir::Location loc,
                                  const arklang::Type &ty,
                                  mlir::Value place,
                                  mlir::Value state) {
    // 1. Triviality Check
    if (!isDropNeeded(ty)) return;

    auto die = [&](llvm::Twine msg) -> void {
        llvm::report_fatal_error(msg);
    };

    auto isInGpu = [&]() -> bool {
        if (place && place.getParentRegion()) {
            if (place.getParentRegion()->getParentOfType<mlir::gpu::GPUModuleOp>()) return true;
        }
        if (mlir::Block *b = builder.getBlock()) {
            if (mlir::Operation *op = b->getParentOp()) {
                if (op->getParentOfType<mlir::gpu::GPUModuleOp>()) return true;
            }
        }
        return false;
    };

    auto typeStr = [&]() -> std::string {
        llvm::SmallString<128> s;
        llvm::raw_svector_ostream os(s);
        os << ty.toString();
        return std::string(os.str());
    };

    if (isInGpu()) {
        die(llvm::Twine("MirBuilder: illegal drop requested in GPU/device context. ")
            + "type=" + typeStr()
            + " reason=host-runtime-drop-glue-not-supported-in-kernels");
    }

    // 2. Dead Code Optimization
    if (auto cst = llvm::dyn_cast_or_null<mlir::LLVM::ConstantOp>(state.getDefiningOp())) {
        if (auto attr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
            if (attr.getValue() == 0) return;
        }
    }

    // 3. Create Control Flow ( If(alive) { Drop } )
    mlir::Block *curBlock = builder.getBlock();
    if (!curBlock) die("MirBuilder: dropPlaceIfOwned with null insertion block");

    mlir::Region *region = curBlock->getParent();
    if (!region) die("MirBuilder: dropPlaceIfOwned with null parent region");

    mlir::Block *dropBlock = builder.createBlock(region, region->end());
    mlir::Block *contBlock = builder.createBlock(region, region->end());

    builder.setInsertionPointToEnd(curBlock);
    mlir::Value cond = coerce(loc, state, builder.getI1Type());
    if (!cond || !cond.getType().isInteger(1)) {
        die("MirBuilder: dropPlaceIfOwned state must coerce to i1");
    }
    builder.create<mlir::LLVM::CondBrOp>(loc, cond, dropBlock, contBlock);

    // 4. Generate Specific Drop Logic
    builder.setInsertionPointToStart(dropBlock);

    if (ty.kind == arklang::Type::Vec) {
        std::string fnName = "__ark_drop_vec_opaque";
        if (!ty.genericArgs.empty()) {
            const auto &elem = ty.genericArgs[0];
            if (elem.kind == arklang::Type::I32) fnName = "__ark_drop_vec_i32";
            else if (elem.kind == arklang::Type::I64) fnName = "__ark_drop_vec_i64";
            else if (elem.kind == arklang::Type::F32) fnName = "__ark_drop_vec_f32";
            else if (elem.kind == arklang::Type::F64) fnName = "__ark_drop_vec_f64";
            else if (elem.kind == arklang::Type::Str) fnName = "__ark_drop_vec_str";
        }

        auto dropFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fnName);
        if (!dropFn) {
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
            auto pTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
            auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {pTy}, false);
            dropFn = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), fnName, fnTy);
        }

        if (!place) die("MirBuilder: dropPlaceIfOwned(Vec) requires non-null place");
        builder.create<mlir::LLVM::CallOp>(loc, dropFn, mlir::ValueRange{place});
    }
    else if (ty.kind == arklang::Type::Str) {
        auto pTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        if (!place) die("MirBuilder: dropPlaceIfOwned(Str) requires non-null place");

        mlir::Type strTy = convertType(ty);
        if (!llvm::isa<mlir::LLVM::LLVMStructType>(strTy)) {
            die("MirBuilder: dropPlaceIfOwned(Str) expected string to lower to LLVM struct");
        }

        mlir::Value ptrAddr = builder.create<mlir::LLVM::GEPOp>(
            loc, pTy, strTy, place,
            mlir::ValueRange{
                builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0)),
                builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0))
            }).getResult();

        mlir::Value rawPtr = builder.create<mlir::LLVM::LoadOp>(loc, pTy, ptrAddr);

        auto freeFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_free");
        if (!freeFn) {
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
            auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {pTy}, false);
            freeFn = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "__ark_free", fnTy);
        }

        builder.create<mlir::LLVM::CallOp>(loc, freeFn, mlir::ValueRange{rawPtr});
    }
    else if (ty.kind == arklang::Type::Tuple) {
        mlir::Type structTy = convertType(ty);
        auto pTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        mlir::Value trueState = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI1Type(), builder.getBoolAttr(true));

        for (size_t i = 0; i < ty.subtypes.size(); ++i) {
            if (!isDropNeeded(ty.subtypes[i])) continue;

            if (!place) die("MirBuilder: dropPlaceIfOwned(Tuple) requires non-null place");

            mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
                loc, pTy, structTy, place,
                mlir::ValueRange{
                    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0)),
                    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(i))
                }).getResult();

            dropPlaceIfOwned(loc, ty.subtypes[i], fieldPtr, trueState);
        }
    }
    else {
        die(llvm::Twine("MirBuilder: dropPlaceIfOwned has no drop implementation for type=")
            + typeStr());
    }

    builder.create<mlir::LLVM::BrOp>(loc, contBlock);
    builder.setInsertionPointToStart(contBlock);
}


// =============================================================================
// Spill Temp (Helper)
// =============================================================================
mlir::Value MirBuilder::spillTemp(mlir::Location loc, mlir::Type ty, mlir::Value v) {
    // 1. Create Slot (Alloca in Entry Block)
    mlir::Value slot = createSlot(loc, ty);
    
    // 2. Coerce Value to Storage Type
    mlir::Value stored = coerce(loc, v, ty);
    
    // 3. Store
    builder.create<mlir::LLVM::StoreOp>(loc, stored, slot);
    
    return slot;
}

} // namespace arklang