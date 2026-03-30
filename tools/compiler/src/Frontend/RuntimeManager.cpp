#include "ark/compiler/Frontend/RuntimeManager.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace arklang {

RuntimeManager::RuntimeManager(mlir::ModuleOp m, mlir::OpBuilder &b) 
    : module(m), builder(b) {}

// =========================================================
// Type Accessors (Lazy Initialization)
// =========================================================

mlir::Type RuntimeManager::getStatusTy() { 
    if (arkStatusTy) return arkStatusTy;
    arkStatusTy = builder.getI32Type();
    return arkStatusTy;
}

mlir::Type RuntimeManager::getVoidPtrTy() {
    if (voidPtrTy) return voidPtrTy;
    voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    return voidPtrTy;
}

mlir::Type RuntimeManager::getArkStrTy() {
    if (arkStrTy) return arkStrTy;
    auto ptr = getVoidPtrTy();
    auto i64 = builder.getI64Type();
    // struct ArkStr { char* ptr; int64_t len; }
    arkStrTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), {ptr, i64});
    return arkStrTy;
}

mlir::Type RuntimeManager::getIoErrorTy() {
    if (arkIoErrorTy) return arkIoErrorTy;
    auto i32 = getStatusTy();
    // struct ArkIoError { int32_t code; int32_t _pad; ArkStr msg; }
    arkIoErrorTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), 
        {i32, i32, getArkStrTy()});
    return arkIoErrorTy;
}

mlir::Type RuntimeManager::getArkVecStructTy() {
    if (arkVecStructTy) return arkVecStructTy;
    auto ptr = getVoidPtrTy();
    auto i64 = builder.getI64Type();
    // struct ark_vec_t { void* ptr; int64_t len; int64_t cap; }
    arkVecStructTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), {ptr, i64, i64});
    return arkVecStructTy;
}

// =========================================================
// Core Error Handling API
// =========================================================

mlir::Value RuntimeManager::callWithCheck(
    mlir::Location loc, 
    llvm::StringRef funcName, 
    mlir::Type outputValueTy, 
    mlir::ValueRange inputs) 
{
    auto ctx = builder.getContext();
    auto ptrTy = getVoidPtrTy();
    auto i64 = builder.getI64Type();
    auto one = builder.create<mlir::LLVM::ConstantOp>(loc, i64, builder.getI64IntegerAttr(1));

    // 1. Prepare Argument Types
    llvm::SmallVector<mlir::Type, 8> fnArgTypes;
    for (auto v : inputs) fnArgTypes.push_back(v.getType());

    // 2. Allocate Output Slots (if needed)
    mlir::Value resultPtr = nullptr;
    if (outputValueTy) {
        resultPtr = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, outputValueTy, one, 8);
        fnArgTypes.push_back(ptrTy); // Add result* to signature
    }

    // 3. Allocate Error Slot
    auto errPtr = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, getIoErrorTy(), one, 8);
    fnArgTypes.push_back(ptrTy); // Add err* to signature

    // 4. Resolve Function & Call
    auto funcOp = getOrInsert(funcName, getStatusTy(), fnArgTypes);
    
    llvm::SmallVector<mlir::Value, 8> callArgs(inputs.begin(), inputs.end());
    if (resultPtr) callArgs.push_back(resultPtr);
    callArgs.push_back(errPtr);

    auto status = builder.create<mlir::func::CallOp>(loc, funcOp, callArgs).getResult(0);

    // 5. Inject Safety Check
    emitTrapIfError(loc, status, errPtr);

    // 6. Return Result
    if (resultPtr) {
        return builder.create<mlir::LLVM::LoadOp>(loc, outputValueTy, resultPtr);
    }
    return nullptr;
}

// =========================================================
// Memory Management API
// =========================================================

mlir::Value RuntimeManager::arkAlloc(mlir::Location loc, mlir::Value size) {
    auto i64 = builder.getI64Type();
    return callPtr(loc, "ark_alloc", {size});
}

mlir::Value RuntimeManager::arkRealloc(mlir::Location loc, mlir::Value ptr, mlir::Value newSize) {
    return callPtr(loc, "ark_realloc", {ptr, newSize});
}

void RuntimeManager::arkFree(mlir::Location loc, mlir::Value ptr) {
    callVoid(loc, "ark_free", {ptr});
}

mlir::Value RuntimeManager::callPtr(mlir::Location loc, llvm::StringRef funcName, mlir::ValueRange inputs) {
    auto ptrTy = getVoidPtrTy();
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto v : inputs) argTypes.push_back(v.getType());

    auto funcOp = getOrInsert(funcName, ptrTy, argTypes);
    return builder.create<mlir::func::CallOp>(loc, funcOp, inputs).getResult(0);
}

// =========================================================
// Void / Side-Effect API
// =========================================================

void RuntimeManager::callVoid(mlir::Location loc, llvm::StringRef funcName, mlir::ValueRange inputs) {
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto v : inputs) argTypes.push_back(v.getType());

    // Void return type is null/empty list
    auto funcOp = getOrInsert(funcName, nullptr, argTypes);
    builder.create<mlir::func::CallOp>(loc, funcOp, inputs);
}

// =========================================================
// Vector Lifecycle API
// =========================================================

void RuntimeManager::callVectorVoid(mlir::Location loc, llvm::StringRef funcName, mlir::Value vectorStructVal) {
    auto ptrTy = getVoidPtrTy();
    auto vecTy = getArkVecStructTy();
    auto i64 = builder.getI64Type();
    
    // Spill SSA struct to Stack so we can pass a pointer
    auto one = builder.create<mlir::LLVM::ConstantOp>(loc, i64, builder.getI64IntegerAttr(1));
    auto stackSlot = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, vecTy, one, 8);
    builder.create<mlir::LLVM::StoreOp>(loc, vectorStructVal, stackSlot);

    callVoid(loc, funcName, {stackSlot});
}

mlir::Value RuntimeManager::arkVecGrow(mlir::Location loc, mlir::Value vecStructVal, mlir::Value elemSize,
                                       llvm::StringRef debugFile, int debugLine, int debugCol) {
    auto ptrTy = getVoidPtrTy();
    auto i64 = builder.getI64Type();
    auto i32 = builder.getI32Type();

    // 1. Spill
    auto one = builder.create<mlir::LLVM::ConstantOp>(loc, i64, builder.getI64IntegerAttr(1));
    auto stackSlot = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, getArkVecStructTy(), one, 8);
    builder.create<mlir::LLVM::StoreOp>(loc, vecStructVal, stackSlot);

    // 2. Prepare Args
    llvm::SmallVector<mlir::Value, 5> args;
    args.push_back(stackSlot);
    args.push_back(elemSize);
    getDebugArgs(loc, debugFile, debugLine, debugCol, args);

    // 3. Call: void ark_vec_grow(ark_vec_t* v, int64_t elem_size, char* f, i32 l, i32 c)
    auto fn = getOrInsert("ark_vec_grow", nullptr, {ptrTy, i64, ptrTy, i32, i32});
    builder.create<mlir::func::CallOp>(loc, fn, args);

    // 4. Reload
    return builder.create<mlir::LLVM::LoadOp>(loc, getArkVecStructTy(), stackSlot);
}

mlir::Value RuntimeManager::arkVecReserveAt(mlir::Location loc, mlir::Value vecStructVal, mlir::Value minCap, mlir::Value elemSize,
                                            llvm::StringRef debugFile, int debugLine, int debugCol) {
    auto ptrTy = getVoidPtrTy();
    auto i64 = builder.getI64Type();
    auto i32 = builder.getI32Type();

    // 1. Spill
    auto one = builder.create<mlir::LLVM::ConstantOp>(loc, i64, builder.getI64IntegerAttr(1));
    auto stackSlot = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, getArkVecStructTy(), one, 8);
    builder.create<mlir::LLVM::StoreOp>(loc, vecStructVal, stackSlot);

    // 2. Prepare Args
    llvm::SmallVector<mlir::Value, 6> args;
    args.push_back(stackSlot);
    args.push_back(minCap);
    args.push_back(elemSize);
    getDebugArgs(loc, debugFile, debugLine, debugCol, args);

    // 3. Call: void ark_vec_reserve_at(...)
    auto fn = getOrInsert("ark_vec_reserve_at", nullptr, {ptrTy, i64, i64, ptrTy, i32, i32});
    builder.create<mlir::func::CallOp>(loc, fn, args);

    // 4. Reload
    return builder.create<mlir::LLVM::LoadOp>(loc, getArkVecStructTy(), stackSlot);
}

void RuntimeManager::arkVecFree(mlir::Location loc, mlir::Value vecStructVal) {
    auto ptrTy = getVoidPtrTy();
    auto i64 = builder.getI64Type();

    // 1. Spill (Free takes a pointer to the struct)
    auto one = builder.create<mlir::LLVM::ConstantOp>(loc, i64, builder.getI64IntegerAttr(1));
    auto stackSlot = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, getArkVecStructTy(), one, 8);
    builder.create<mlir::LLVM::StoreOp>(loc, vecStructVal, stackSlot);

    // 2. Call
    callVoid(loc, "ark_vec_free", {stackSlot});
    
    // No reload, vector is dead
}

// =========================================================
// Internal Helpers
// =========================================================

void RuntimeManager::emitTrapIfError(mlir::Location loc, mlir::Value statusVal, mlir::Value errStructPtr) {
    auto i32 = getStatusTy();
    auto zero = builder.create<mlir::LLVM::ConstantOp>(loc, i32, builder.getI32IntegerAttr(0));
    
    // Check if status != 0
    auto isErr = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, statusVal, zero);

    builder.create<mlir::scf::IfOp>(loc, isErr, [&](mlir::OpBuilder &b, mlir::Location l) {
        b.create<mlir::LLVM::Trap>(l);
        b.create<mlir::scf::YieldOp>(l);
    });
}

mlir::func::FuncOp RuntimeManager::getOrInsert(llvm::StringRef name, mlir::Type retTy, llvm::ArrayRef<mlir::Type> args) {
    if (auto f = module.lookupSymbol<mlir::func::FuncOp>(name)) return f;
    
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    // Handle Void return (empty list) vs Value return
    auto fnTy = retTy 
        ? builder.getFunctionType(args, {retTy}) 
        : builder.getFunctionType(args, {});
        
    auto f = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name, fnTy);
    f.setSymVisibility("private");

    // [CRITICAL FIX]
    // Do NOT set "llvm.emit_c_interface" here. 
    // Your runtime functions are standard C functions, not MLIR-wrapped interfaces.
    // f->setAttr("llvm.emit_c_interface", builder.getUnitAttr()); // <--- DELETE THIS LINE

    return f;
}

void RuntimeManager::getDebugArgs(mlir::Location loc, llvm::StringRef file, int line, int col, llvm::SmallVectorImpl<mlir::Value>& out) {
    auto i32 = builder.getI32Type();
    out.push_back(getGlobalStringPtr(loc, file));
    out.push_back(builder.create<mlir::LLVM::ConstantOp>(loc, i32, builder.getI32IntegerAttr(line)));
    out.push_back(builder.create<mlir::LLVM::ConstantOp>(loc, i32, builder.getI32IntegerAttr(col)));
}

mlir::Value RuntimeManager::getGlobalStringPtr(mlir::Location loc, llvm::StringRef str) {
    std::string globalName = "str_" + std::to_string(llvm::hash_value(str));
    
    if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());
        
        std::string content = str.str();
        content.push_back('\0'); 
        
        auto type = mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), content.size());
        builder.create<mlir::LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, 
            globalName, builder.getStringAttr(content));
    }
    
    auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName);
    return builder.create<mlir::LLVM::AddressOfOp>(loc, global);
}

mlir::Value RuntimeManager::getGlobalString(mlir::Location loc, llvm::StringRef str) {
    return getGlobalStringPtr(loc, str);
}

mlir::Value RuntimeManager::asCString(mlir::Location loc, mlir::Value arkStrVal) {
    auto ptrTy = getVoidPtrTy();
    auto i64Ty = builder.getI64Type();
    auto i8Ty  = builder.getI8Type();

    // 1. Unpack ArkStr {ptr, len}
    auto srcPtr = builder.create<mlir::LLVM::ExtractValueOp>(loc, arkStrVal, 0);
    auto len = builder.create<mlir::LLVM::ExtractValueOp>(loc, arkStrVal, 1);

    // 2. Calculate Size (len + 1)
    auto one = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(1));
    auto sizeWithNull = builder.create<mlir::LLVM::AddOp>(loc, len, one);

    // 3. Allocate Stack Buffer (Dynamic Alloca)
    // Note: Standard LLVM alloca handles dynamic sizes efficiently in the function prelude/stack adjustments.
    auto dstPtr = builder.create<mlir::LLVM::AllocaOp>(loc, ptrTy, i8Ty, sizeWithNull, 1);

    // 4. MemCpy (src -> dst, len bytes)
    // "isVolatile"=false
    builder.create<mlir::LLVM::MemcpyOp>(loc, dstPtr, srcPtr, len, false);

    // 5. Null Terminate (dst[len] = 0)
    auto zero = builder.create<mlir::LLVM::ConstantOp>(loc, i8Ty, builder.getI8IntegerAttr(0));
    auto nullTermAddr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrTy, i8Ty, dstPtr, mlir::ValueRange{len});
    builder.create<mlir::LLVM::StoreOp>(loc, zero, nullTermAddr);

    return dstPtr;
}

} // namespace arklang