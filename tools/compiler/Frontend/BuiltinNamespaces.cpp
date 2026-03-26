#include "BuiltinNamespaces.h"

#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"

namespace arklang::frontend {

// -----------------------------------------------------------------------------
// Local helpers (LLVM::CallOp has getResults(), not getResult(i))
// -----------------------------------------------------------------------------
static inline mlir::Value llvmCallRes(mlir::LLVM::CallOp op, unsigned i = 0) {
    return op.getOperation()->getResult(i);
}

static inline void emitMemcpyP0P0I64(mlir::ModuleOp module,
                                     mlir::OpBuilder &b,
                                     mlir::Location loc,
                                     mlir::Value dst,
                                     mlir::Value src,
                                     mlir::Value lenI64) {
    // [FIX] Call standard libc 'memcpy' instead of the LLVM intrinsic.
    // This avoids all LLVM version mismatch issues with 'captures(none)'.
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i64Ty = b.getI64Type();
    
    auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("memcpy");
    if (!fn) {
        mlir::OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(module.getBody());
        // Signature: void* memcpy(void* dest, const void* src, size_t count);
        auto fnTy = mlir::LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy, i64Ty}, false);
        fn = b.create<mlir::LLVM::LLVMFuncOp>(loc, "memcpy", fnTy);
    }
    
    b.create<mlir::LLVM::CallOp>(
        loc, 
        mlir::TypeRange{ptrTy},
        mlir::SymbolRefAttr::get(b.getContext(), "memcpy"), 
        mlir::ValueRange{dst, src, lenI64}
    );
}

BuiltinNsLowering::BuiltinNsLowering(mlir::ModuleOp module, mlir::OpBuilder &b)
    : module(module), b(b) {}

// =============================================================================
// Type Helpers
// =============================================================================

mlir::LLVM::LLVMStructType BuiltinNsLowering::arkStrTy() {
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i64Ty = b.getI64Type();
    return mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {ptrTy, i64Ty});
}

mlir::LLVM::LLVMStructType BuiltinNsLowering::arkBytesTy() {
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i64Ty = b.getI64Type();
    return mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {ptrTy, i64Ty, i64Ty});
}

mlir::LLVM::LLVMStructType BuiltinNsLowering::arkIoErrTy() {
    auto i32Ty = b.getI32Type();
    auto strTy = arkStrTy();
    return mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {i32Ty, i32Ty, strTy});
}

// =============================================================================
// Function Declaration Helpers
// =============================================================================

mlir::func::FuncOp BuiltinNsLowering::externFn(llvm::StringRef name,
                                               mlir::TypeRange args,
                                               mlir::TypeRange rets) {
    if (auto op = module.lookupSymbol(name)) {
        if (auto f = llvm::dyn_cast<mlir::func::FuncOp>(op)) return f;
        return mlir::func::FuncOp();
    }

    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(module.getBody());

    auto fnTy = b.getFunctionType(args, rets);
    auto fn = b.create<mlir::func::FuncOp>(b.getUnknownLoc(), name, fnTy);
    fn.setPrivate();
    return fn;
}

mlir::LLVM::LLVMFuncOp BuiltinNsLowering::externLLVMFn(llvm::StringRef name,
                                                       mlir::Type ret,
                                                       mlir::TypeRange args) {
    if (auto existing = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        llvm::SmallVector<mlir::Type, 8> argTys(args.begin(), args.end());
        auto want = mlir::LLVM::LLVMFunctionType::get(ret, argTys, /*isVarArg=*/false);
        if (existing.getFunctionType() != want) {
            mlir::emitError(existing.getLoc())
                << "BuiltinNsLowering externLLVMFn type mismatch for '" << name
                << "': existing=" << existing.getFunctionType() << " want=" << want;
            return mlir::LLVM::LLVMFuncOp();
        }
        return existing;
    }

    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(module.getBody());

    llvm::SmallVector<mlir::Type, 8> argTys(args.begin(), args.end());
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(ret, argTys, /*isVarArg=*/false);

    auto fn = b.create<mlir::LLVM::LLVMFuncOp>(b.getUnknownLoc(), name, fnTy);
    fn.setLinkage(mlir::LLVM::Linkage::External);
    return fn;
}

// =============================================================================
// Intrinsics
// =============================================================================

void BuiltinNsLowering::ensureMemcpy() {
    // No-op: We now use MLIR's native MemcpyOp which handles imports safely.
}

void BuiltinNsLowering::ensureMemset() {
    // No-op: Same reason as MemcpyOp.
}

// =============================================================================
// Low-Level Helpers
// =============================================================================

mlir::Value BuiltinNsLowering::alloca1(mlir::Location loc, mlir::Type ty, int64_t align) {
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto one64 = b.create<mlir::arith::ConstantIntOp>(loc, 1, 64).getResult();
    return b.create<mlir::LLVM::AllocaOp>(loc, ptrTy, ty, one64, align).getResult();
}

mlir::Value BuiltinNsLowering::zero(mlir::Location loc, mlir::Type ty) {
    return b.create<mlir::LLVM::ZeroOp>(loc, ty).getResult();
}

void BuiltinNsLowering::storeZero(mlir::Location loc, mlir::Value ptr, mlir::Type ty) {
    b.create<mlir::LLVM::StoreOp>(loc, zero(loc, ty), ptr);
}

bool BuiltinNsLowering::isArkStrLike(mlir::Type ty) const {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(ty);
    if (!st) return false;
    auto body = st.getBody();
    if (body.size() != 2) return false;
    return llvm::isa<mlir::LLVM::LLVMPointerType>(body[0]) && body[1].isInteger(64);
}

bool BuiltinNsLowering::isArkBytesLike(mlir::Type ty) const {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(ty);
    if (!st) return false;
    auto body = st.getBody();
    if (body.size() != 3) return false;
    return llvm::isa<mlir::LLVM::LLVMPointerType>(body[0]) && body[1].isInteger(64) && body[2].isInteger(64);
}

void BuiltinNsLowering::unpackArkStr(mlir::Location loc, mlir::Value s, mlir::Value &outPtr, mlir::Value &outLen) {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(s.getType());
    if (!st || st.getBody().size() != 2) {
        mlir::emitError(loc) << "unpackArkStr: not ArkStr-like type: " << s.getType();
        outPtr = {};
        outLen = {};
        return;
    }
    outPtr = b.create<mlir::LLVM::ExtractValueOp>(loc, st.getBody()[0], s, b.getDenseI64ArrayAttr({0})).getResult();
    outLen = b.create<mlir::LLVM::ExtractValueOp>(loc, st.getBody()[1], s, b.getDenseI64ArrayAttr({1})).getResult();
}

void BuiltinNsLowering::unpackArkBytes(mlir::Location loc, mlir::Value s, mlir::Value &outPtr, mlir::Value &outLen) {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(s.getType());
    if (!st || st.getBody().size() != 3) {
        mlir::emitError(loc) << "unpackArkBytes: not ArkBytes-like type: " << s.getType();
        outPtr = {};
        outLen = {};
        return;
    }
    outPtr = b.create<mlir::LLVM::ExtractValueOp>(loc, st.getBody()[0], s, b.getDenseI64ArrayAttr({0})).getResult();
    outLen = b.create<mlir::LLVM::ExtractValueOp>(loc, st.getBody()[1], s, b.getDenseI64ArrayAttr({1})).getResult();
}

mlir::Value BuiltinNsLowering::materializeCString(mlir::Location loc, mlir::Value arkStr, mlir::Value &outLenI64) {
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i8Ty  = b.getI8Type();
    auto i64Ty = b.getI64Type();

    mlir::Value srcPtr, srcLen;
    unpackArkStr(loc, arkStr, srcPtr, srcLen);
    if (!srcPtr || !srcLen) return mlir::Value();
    outLenI64 = srcLen;

    mlir::Value one64 = b.create<mlir::arith::ConstantIntOp>(loc, 1, 64).getResult();
    mlir::Value n = b.create<mlir::arith::AddIOp>(loc, srcLen, one64).getResult();

    // ark_protocol.h: void* __ark_alloc(uint64_t bytes, uint64_t align);
    mlir::Value align1 = b.create<mlir::arith::ConstantIntOp>(loc, 1, 64).getResult();
    auto alloc = externLLVMFn("__ark_alloc", ptrTy, {i64Ty, i64Ty});
    if (!alloc) return mlir::Value();

    auto call = b.create<mlir::LLVM::CallOp>(loc, alloc, mlir::ValueRange{n, align1});
    mlir::Value dst = llvmCallRes(call, 0);

    // Call the native MemcpyOp wrapper
    emitMemcpyP0P0I64(module, b, loc, dst, srcPtr, srcLen);

    // dst[len] = 0
    mlir::Value z8 = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
    mlir::Value endPtr = b.create<mlir::LLVM::GEPOp>(loc, ptrTy, i8Ty, dst, mlir::ValueRange{srcLen}).getResult();
    b.create<mlir::LLVM::StoreOp>(loc, z8, endPtr);

    return dst;
}

void BuiltinNsLowering::freeCString(mlir::Location loc, mlir::Value cstrPtr) {
    auto ptrTy  = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto voidTy = mlir::LLVM::LLVMVoidType::get(b.getContext());

    auto f = externLLVMFn("__ark_free", voidTy, {ptrTy});
    if (!f) return;

    mlir::Value p = cstrPtr;
    if (p.getType() != ptrTy) p = b.create<mlir::LLVM::BitcastOp>(loc, ptrTy, p).getResult();
    b.create<mlir::LLVM::CallOp>(loc, f, mlir::ValueRange{p});
}

mlir::Value BuiltinNsLowering::packCallResult(mlir::Location loc,
                                              mlir::Value status,
                                              llvm::ArrayRef<mlir::Value> outs,
                                              mlir::Value errVal) {
    llvm::SmallVector<mlir::Type, 8> tys;
    tys.reserve(2 + outs.size());
    tys.push_back(status.getType());
    for (auto v : outs) tys.push_back(v.getType());
    tys.push_back(errVal.getType());

    auto retTy = mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), tys);
    mlir::Value agg = b.create<mlir::LLVM::UndefOp>(loc, retTy).getResult();

    agg = b.create<mlir::LLVM::InsertValueOp>(loc, agg, status, b.getDenseI64ArrayAttr({0})).getResult();
    for (size_t i = 0; i < outs.size(); ++i) {
        agg = b.create<mlir::LLVM::InsertValueOp>(
                  loc, agg, outs[i], b.getDenseI64ArrayAttr({(int64_t)(i + 1)}))
                  .getResult();
    }
    agg = b.create<mlir::LLVM::InsertValueOp>(
              loc, agg, errVal, b.getDenseI64ArrayAttr({(int64_t)(outs.size() + 1)}))
              .getResult();
    return agg;
}

// =============================================================================
// Table Definition
// =============================================================================

llvm::ArrayRef<BuiltinNsLowering::MapEntry> BuiltinNsLowering::table(mlir::MLIRContext *ctx) {
    static llvm::SmallVector<MapEntry, 48> T;
    if (!T.empty()) return T;

    auto i32Ty = mlir::IntegerType::get(ctx, 32);
    auto i64Ty = mlir::IntegerType::get(ctx, 64);
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

    auto addConst = [&](Ns ns, llvm::StringRef name, mlir::Type ty, int64_t v) {
        MapEntry e{};
        e.key = MapKey{ns, Kind::Constant, name};
        e.cst = ConstSpec{ty, v};
        T.push_back(e);
    };

    auto addCall = [&](Ns ns, llvm::StringRef name, llvm::StringRef abi,
                       llvm::ArrayRef<mlir::Type> args,
                       llvm::ArrayRef<mlir::Type> rets) {
        MapEntry e{};
        e.key = MapKey{ns, Kind::Call, name};
        e.call.abi = abi;
        e.call.argTys.assign(args.begin(), args.end());
        e.call.retTys.assign(rets.begin(), rets.end());
        T.push_back(e);
    };

    addConst(Ns::NET, "ARK_NET_MAX_PACKET", i64Ty, (int64_t)(64LL * 1024 * 1024));
    addConst(Ns::IO,  "ARK_AT_FDCWD",       i64Ty, (int64_t)(-100));

    addCall(Ns::FS,  "writeAtomic", "__ark_file_write_atomic", {ptrTy, ptrTy, i64Ty, ptrTy}, {i32Ty});
    addCall(Ns::FS,  "append",      "__ark_file_append",       {ptrTy, ptrTy, i64Ty, ptrTy}, {i32Ty});
    addCall(Ns::FS,  "readAll",     "__ark_file_read_all",     {ptrTy, ptrTy, ptrTy},        {i32Ty});
    addCall(Ns::FS,  "exists",      "__ark_file_exists",       {ptrTy, ptrTy, ptrTy},        {i32Ty});

    addCall(Ns::IO,  "open",        "__ark_fd_open",           {ptrTy, i32Ty, i32Ty, ptrTy, ptrTy}, {i32Ty});
    addCall(Ns::IO,  "close",       "__ark_fd_close",          {i64Ty, ptrTy},                      {i32Ty});

    addCall(Ns::NET, "connect",     "__ark_net_connect",       {ptrTy, i32Ty, i64Ty, ptrTy, ptrTy}, {i32Ty});
    addCall(Ns::NET, "send",        "__ark_net_send",          {i32Ty, ptrTy, i64Ty, ptrTy},        {i32Ty});
    addCall(Ns::NET, "recv",        "__ark_net_recv",          {i32Ty, i64Ty, ptrTy, ptrTy},        {i32Ty});

    // SYS is handled via dedicated wrappers (value/option/slice packing)
    addCall(Ns::SYS, "cwd",         "__ark_sys_cwd",           {ptrTy, ptrTy},                      {i32Ty});
    addCall(Ns::SYS, "args",        "__ark_sys_args",          {ptrTy, ptrTy, ptrTy},               {i32Ty});
    addCall(Ns::SYS, "env_get",     "__ark_sys_env_get",       {ptrTy, ptrTy, ptrTy, ptrTy},        {i32Ty});

    return T;
}

// =============================================================================
// Dispatchers
// =============================================================================

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerConstant(mlir::Location loc, Ns ns, llvm::StringRef name) {
    for (auto &e : table(b.getContext())) {
        if (e.key.kind != Kind::Constant) continue;
        if (e.key.ns != ns) continue;
        if (e.key.name != name) continue;

        if (e.cst.ty.isInteger(64))
            return b.create<mlir::arith::ConstantIntOp>(loc, e.cst.i64, 64).getResult();
        if (e.cst.ty.isInteger(32))
            return b.create<mlir::arith::ConstantIntOp>(loc, e.cst.i64, 32).getResult();

        return mlir::failure();
    }
    return mlir::failure();
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerCall(mlir::Location loc,
                                                          Ns ns,
                                                          llvm::StringRef name,
                                                          llvm::ArrayRef<mlir::Value> args) {
    if (ns == Ns::FS) {
        if (name == "writeAtomic") return lowerFS_writeAtomic(loc, args);
        if (name == "append")      return lowerFS_append(loc, args);
        if (name == "readAll")     return lowerFS_readAll(loc, args);
        if (name == "exists")      return lowerFS_exists(loc, args);
    }
    if (ns == Ns::IO) {
        if (name == "open")  return lowerIO_open(loc, args);
        if (name == "close") return lowerIO_close(loc, args);
    }
    if (ns == Ns::NET) {
        if (name == "connect") return lowerNET_connect(loc, args);
        if (name == "send")    return lowerNET_send(loc, args);
        if (name == "recv")    return lowerNET_recv(loc, args);
    }
    if (ns == Ns::SYS) {
        if (name == "cwd") return lowerSYS_cwd(loc, args);
        if (name == "args") return lowerSYS_args(loc, args);
        if (name == "env_get" || name == "env.get" || name == "getenv" || name == "get")
            return lowerSYS_env_get(loc, args);
    }
    return mlir::failure();
}

// =============================================================================
// Implementations: FS
// =============================================================================

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerFS_writeAtomic(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 2) return mlir::failure(); // Path, Data

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    if (!isArkStrLike(args[0].getType())) return mlir::failure();
    mlir::Value pathLen;
    mlir::Value cPath = materializeCString(loc, args[0], pathLen);
    if (!cPath) return mlir::failure();

    mlir::Value dataPtr, dataLen;
    if (isArkStrLike(args[1].getType())) unpackArkStr(loc, args[1], dataPtr, dataLen);
    else if (isArkBytesLike(args[1].getType())) unpackArkBytes(loc, args[1], dataPtr, dataLen);
    else return mlir::failure();

    auto fn = externLLVMFn("__ark_file_write_atomic", i32Ty, {ptrTy, ptrTy, i64Ty, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cPath, dataPtr, dataLen, errPtr}), 0);

    freeCString(loc, cPath);

    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {}, errVal);
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerFS_append(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 2) return mlir::failure();

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    if (!isArkStrLike(args[0].getType())) return mlir::failure();
    mlir::Value pathLen;
    mlir::Value cPath = materializeCString(loc, args[0], pathLen);
    if (!cPath) return mlir::failure();

    mlir::Value dataPtr, dataLen;
    if (isArkStrLike(args[1].getType())) unpackArkStr(loc, args[1], dataPtr, dataLen);
    else if (isArkBytesLike(args[1].getType())) unpackArkBytes(loc, args[1], dataPtr, dataLen);
    else return mlir::failure();

    auto fn = externLLVMFn("__ark_file_append", i32Ty, {ptrTy, ptrTy, i64Ty, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cPath, dataPtr, dataLen, errPtr}), 0);

    freeCString(loc, cPath);

    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {}, errVal);
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerFS_readAll(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 1) return mlir::failure();

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    if (!isArkStrLike(args[0].getType())) return mlir::failure();
    mlir::Value pathLen;
    mlir::Value cPath = materializeCString(loc, args[0], pathLen);
    if (!cPath) return mlir::failure();

    auto outTy = arkStrTy();
    mlir::Value outPtr = alloca1(loc, outTy, 8);
    storeZero(loc, outPtr, outTy);

    auto fn = externLLVMFn("__ark_file_read_all", i32Ty, {ptrTy, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cPath, outPtr, errPtr}), 0);

    freeCString(loc, cPath);

    mlir::Value outVal = b.create<mlir::LLVM::LoadOp>(loc, outTy, outPtr).getResult();
    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {outVal}, errVal);
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerFS_exists(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 1) return mlir::failure();

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();
    auto i8Ty  = b.getI8Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    if (!isArkStrLike(args[0].getType())) return mlir::failure();
    mlir::Value pathLen;
    mlir::Value cPath = materializeCString(loc, args[0], pathLen);
    if (!cPath) return mlir::failure();

    mlir::Value existsPtr = alloca1(loc, i8Ty, 1);
    b.create<mlir::LLVM::StoreOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult(), existsPtr);

    auto fn = externLLVMFn("__ark_file_exists", i32Ty, {ptrTy, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cPath, existsPtr, errPtr}), 0);

    freeCString(loc, cPath);

    mlir::Value exists8 = b.create<mlir::LLVM::LoadOp>(loc, i8Ty, existsPtr).getResult();
    mlir::Value zero8   = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
    mlir::Value exists1 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, exists8, zero8).getResult();

    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {exists1}, errVal);
}

// =============================================================================
// Implementations: IO (Formerly FD)
// =============================================================================

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerIO_open(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 3) return mlir::failure(); // Path, Flags, Mode

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    if (!isArkStrLike(args[0].getType())) return mlir::failure();
    mlir::Value pathLen;
    mlir::Value cPath = materializeCString(loc, args[0], pathLen);
    if (!cPath) return mlir::failure();

    mlir::Value flags = args[1];
    if (!flags.getType().isInteger(32)) flags = b.create<mlir::arith::TruncIOp>(loc, i32Ty, flags).getResult();

    mlir::Value mode = args[2];
    if (!mode.getType().isInteger(32)) mode = b.create<mlir::arith::TruncIOp>(loc, i32Ty, mode).getResult();

    mlir::Value outFdPtr = alloca1(loc, i64Ty, 8);
    b.create<mlir::LLVM::StoreOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult(), outFdPtr);

    auto fn = externLLVMFn("__ark_fd_open", i32Ty, {ptrTy, i32Ty, i32Ty, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cPath, flags, mode, outFdPtr, errPtr}), 0);

    freeCString(loc, cPath);

    mlir::Value fdVal = b.create<mlir::LLVM::LoadOp>(loc, i64Ty, outFdPtr).getResult();
    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {fdVal}, errVal);
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerIO_close(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 1) return mlir::failure();

    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    mlir::Value fd = args[0];
    if (!fd.getType().isInteger(64)) fd = b.create<mlir::arith::ExtSIOp>(loc, i64Ty, fd).getResult();

    auto fn = externLLVMFn("__ark_fd_close", i32Ty, {i64Ty, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{fd, errPtr}), 0);

    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {}, errVal);
}

// =============================================================================
// Implementations: NET
// =============================================================================

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerNET_connect(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 3) return mlir::failure(); // Host, Port, Timeout

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    if (!isArkStrLike(args[0].getType())) return mlir::failure();
    mlir::Value hostLen;
    mlir::Value cHost = materializeCString(loc, args[0], hostLen);
    if (!cHost) return mlir::failure();

    mlir::Value port = args[1];
    if (!port.getType().isInteger(32)) port = b.create<mlir::arith::TruncIOp>(loc, i32Ty, port).getResult();

    mlir::Value toMs = args[2];
    if (!toMs.getType().isInteger(64)) toMs = b.create<mlir::arith::ExtSIOp>(loc, i64Ty, toMs).getResult();

    mlir::Value outSockPtr = alloca1(loc, i32Ty, 4);
    b.create<mlir::LLVM::StoreOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, -1, 32).getResult(), outSockPtr);

    auto fn = externLLVMFn("__ark_net_connect", i32Ty, {ptrTy, i32Ty, i64Ty, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cHost, port, toMs, outSockPtr, errPtr}), 0);

    freeCString(loc, cHost);

    mlir::Value sockVal = b.create<mlir::LLVM::LoadOp>(loc, i32Ty, outSockPtr).getResult();
    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {sockVal}, errVal);
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerNET_send(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 2) return mlir::failure();

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());
    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    mlir::Value sock = args[0];
    if (!sock.getType().isInteger(32)) sock = b.create<mlir::arith::TruncIOp>(loc, i32Ty, sock).getResult();

    mlir::Value dataPtr, dataLen;
    if (isArkStrLike(args[1].getType())) unpackArkStr(loc, args[1], dataPtr, dataLen);
    else if (isArkBytesLike(args[1].getType())) unpackArkBytes(loc, args[1], dataPtr, dataLen);
    else return mlir::failure();

    auto fn = externLLVMFn("__ark_net_send", i32Ty, {i32Ty, ptrTy, i64Ty, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{sock, dataPtr, dataLen, errPtr}), 0);

    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {}, errVal);
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerNET_recv(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 2) return mlir::failure();

    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    mlir::Value sock = args[0];
    if (!sock.getType().isInteger(32)) sock = b.create<mlir::arith::TruncIOp>(loc, i32Ty, sock).getResult();

    mlir::Value maxL = args[1];
    if (!maxL.getType().isInteger(64)) maxL = b.create<mlir::arith::ExtSIOp>(loc, i64Ty, maxL).getResult();

    auto outTy = arkBytesTy();
    mlir::Value outPtr = alloca1(loc, outTy, 8);
    storeZero(loc, outPtr, outTy);

    auto fn = externLLVMFn("__ark_net_recv", i32Ty, {i32Ty, i64Ty, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value status = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{sock, maxL, outPtr, errPtr}), 0);

    mlir::Value outVal = b.create<mlir::LLVM::LoadOp>(loc, outTy, outPtr).getResult();
    auto errVal = b.create<mlir::LLVM::LoadOp>(loc, arkIoErrTy(), errPtr).getResult();
    return packCallResult(loc, status, {outVal}, errVal);
}

// =============================================================================
// Implementations: SYS
// =============================================================================

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerSYS_cwd(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (!args.empty()) return mlir::failure();

    auto i32Ty = b.getI32Type();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    auto outTy = arkStrTy();
    mlir::Value outStrPtr = alloca1(loc, outTy, 8);
    storeZero(loc, outStrPtr, outTy);

    auto fn = externLLVMFn("__ark_sys_cwd", i32Ty, {ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    (void)b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{outStrPtr, errPtr});

    return b.create<mlir::LLVM::LoadOp>(loc, outTy, outStrPtr).getResult();
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerSYS_args(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (!args.empty()) return mlir::failure();

    auto i32Ty = b.getI32Type();
    auto i64Ty = b.getI64Type();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());

    mlir::Value outPtrPtr = alloca1(loc, ptrTy, 8);
    b.create<mlir::LLVM::StoreOp>(loc, b.create<mlir::LLVM::ZeroOp>(loc, ptrTy).getResult(), outPtrPtr);

    mlir::Value outLenPtr = alloca1(loc, i64Ty, 8);
    b.create<mlir::LLVM::StoreOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult(), outLenPtr);

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    auto fn = externLLVMFn("__ark_sys_args", i32Ty, {ptrTy, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value st = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{outPtrPtr, outLenPtr, errPtr}), 0);

    mlir::Value p = b.create<mlir::LLVM::LoadOp>(loc, ptrTy, outPtrPtr).getResult();
    mlir::Value n = b.create<mlir::LLVM::LoadOp>(loc, i64Ty, outLenPtr).getResult();

    auto sliceTy = mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {ptrTy, i64Ty});
    mlir::Value slice = b.create<mlir::LLVM::UndefOp>(loc, sliceTy).getResult();

    mlir::Value z32 = b.create<mlir::arith::ConstantIntOp>(loc, 0, 32).getResult();
    mlir::Value ok  = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, st, z32).getResult();

    mlir::Value nullPtr = b.create<mlir::LLVM::ZeroOp>(loc, ptrTy).getResult();
    mlir::Value zeroLen = b.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult();

    mlir::Value finalP = b.create<mlir::LLVM::SelectOp>(loc, ok, p, nullPtr).getResult();
    mlir::Value finalN = b.create<mlir::LLVM::SelectOp>(loc, ok, n, zeroLen).getResult();

    slice = b.create<mlir::LLVM::InsertValueOp>(loc, slice, finalP, b.getDenseI64ArrayAttr({0})).getResult();
    slice = b.create<mlir::LLVM::InsertValueOp>(loc, slice, finalN, b.getDenseI64ArrayAttr({1})).getResult();
    return slice;
}

mlir::FailureOr<mlir::Value> BuiltinNsLowering::lowerSYS_env_get(mlir::Location loc, llvm::ArrayRef<mlir::Value> args) {
    if (args.size() != 1) return mlir::failure();
    if (!isArkStrLike(args[0].getType())) return mlir::failure();

    auto i32Ty = b.getI32Type();
    auto i8Ty  = b.getI8Type();
    auto i1Ty  = b.getI1Type();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());

    mlir::Value keyLen;
    mlir::Value cKey = materializeCString(loc, args[0], keyLen);
    if (!cKey) return mlir::failure();

    auto strTy = arkStrTy();
    mlir::Value outStrPtr = alloca1(loc, strTy, 8);
    storeZero(loc, outStrPtr, strTy);

    mlir::Value outOkPtr = alloca1(loc, i8Ty, 1);
    b.create<mlir::LLVM::StoreOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult(), outOkPtr);

    mlir::Value errPtr = alloca1(loc, arkIoErrTy(), 8);
    storeZero(loc, errPtr, arkIoErrTy());

    auto fn = externLLVMFn("__ark_sys_env_get", i32Ty, {ptrTy, ptrTy, ptrTy, ptrTy});
    if (!fn) return mlir::failure();

    mlir::Value st = llvmCallRes(b.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{cKey, outStrPtr, outOkPtr, errPtr}), 0);

    freeCString(loc, cKey);

    mlir::Value v   = b.create<mlir::LLVM::LoadOp>(loc, strTy, outStrPtr).getResult();
    mlir::Value ok8 = b.create<mlir::LLVM::LoadOp>(loc, i8Ty, outOkPtr).getResult();

    mlir::Value z8  = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
    mlir::Value ok1 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, ok8, z8).getResult();

    mlir::Value z32  = b.create<mlir::arith::ConstantIntOp>(loc, 0, 32).getResult();
    mlir::Value stOk = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, st, z32).getResult();

    mlir::Value emptyStr = [&]() -> mlir::Value {
        mlir::Value s = b.create<mlir::LLVM::UndefOp>(loc, strTy).getResult();
        mlir::Value nullPtr = b.create<mlir::LLVM::ZeroOp>(loc, ptrTy).getResult();
        mlir::Value zeroLen = b.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult();
        s = b.create<mlir::LLVM::InsertValueOp>(loc, s, nullPtr, b.getDenseI64ArrayAttr({0})).getResult();
        s = b.create<mlir::LLVM::InsertValueOp>(loc, s, zeroLen, b.getDenseI64ArrayAttr({1})).getResult();
        return s;
    }();

    mlir::Value finalStr = b.create<mlir::LLVM::SelectOp>(loc, stOk, v, emptyStr).getResult();
    mlir::Value finalOk  = b.create<mlir::LLVM::SelectOp>(
                               loc, stOk, ok1, b.create<mlir::arith::ConstantIntOp>(loc, 0, 1).getResult())
                               .getResult();

    auto optTy = mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {strTy, i1Ty});
    mlir::Value opt = b.create<mlir::LLVM::UndefOp>(loc, optTy).getResult();
    opt = b.create<mlir::LLVM::InsertValueOp>(loc, opt, finalStr, b.getDenseI64ArrayAttr({0})).getResult();
    opt = b.create<mlir::LLVM::InsertValueOp>(loc, opt, finalOk,  b.getDenseI64ArrayAttr({1})).getResult();
    return opt;
}

} // namespace arklang::frontend