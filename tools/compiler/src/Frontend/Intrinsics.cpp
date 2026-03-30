#include "ark/compiler/Frontend/Intrinsics.hpp"
#include "ark/compiler/Frontend/AST.hpp"
#include <set> // [FIX] Required for std::set

namespace arklang {

// =============================================================================
// Shared Intrinsic Helper
// =============================================================================

// [FIX] Implementation of the helper declared in Intrinsics.h
bool isIntrinsicFn(llvm::StringRef name) {
    static const std::set<std::string> registry = {
        "allocof", "sizeof", "typeof", "castof", "free", 
        "len", "min", "addr", "assert", "panic", "hash",
        "shash"
    };
    return registry.count(name.str());
}

// =============================================================================
// Helpers
// =============================================================================

// Construct "Alloc<T>" (Canonical Pointer Type)
static arklang::Type makeAllocPtrType(const arklang::Type &elem) {
    arklang::Type t;
    t.kind = arklang::Type::Generic;
    t.schemaName = "Alloc"; 
    t.genericArgs = {elem};
    return t;
}

// =============================================================================
// Inference Logic
// =============================================================================

// 1. allocof<T>(size) -> Alloc<T>
static std::optional<arklang::Type> inferAllocOf(const CallExpr &call, const Intrinsic::GetTypeFn&) {
    if (call.explicitGenericArgs.empty()) return std::nullopt;
    return makeAllocPtrType(call.explicitGenericArgs[0]);
}

// 2. sizeof<T>() -> i64
static std::optional<arklang::Type> inferSizeOf(const CallExpr &, const Intrinsic::GetTypeFn&) {
    return arklang::Type{arklang::Type::I64};
}

// 3. typeof<T>() -> str
static std::optional<arklang::Type> inferTypeOf(const CallExpr &, const Intrinsic::GetTypeFn&) {
    return arklang::Type{arklang::Type::Str};
}

// 4. castof<T>(ptr) -> T
static std::optional<arklang::Type> inferCastOf(const CallExpr &call, const Intrinsic::GetTypeFn&) {
    if (call.explicitGenericArgs.empty()) return std::nullopt;
    return call.explicitGenericArgs[0];
}

// 5. free(ptr) -> void
static std::optional<arklang::Type> inferFree(const CallExpr &, const Intrinsic::GetTypeFn&) {
    return arklang::Type{arklang::Type::Void};
}

// 6. len(container) -> i64
static std::optional<arklang::Type> inferLen(const CallExpr &, const Intrinsic::GetTypeFn&) {
    return arklang::Type{arklang::Type::I64};
}

// 7. addr(val) -> Alloc<T>
static std::optional<arklang::Type> inferAddr(const CallExpr &call, const Intrinsic::GetTypeFn &getTy) {
    if (call.args.empty()) return std::nullopt;
    const arklang::Type valTy = getTy(*call.args[0].value);
    return makeAllocPtrType(valTy);
}

// 8. min(a, b) -> T
static std::optional<arklang::Type> inferMin(const CallExpr &call, const Intrinsic::GetTypeFn &getTy) {
    if (call.args.size() != 2) return std::nullopt;
    const arklang::Type aTy = getTy(*call.args[0].value);
    const arklang::Type bTy = getTy(*call.args[1].value);
    if (aTy.kind != bTy.kind) return std::nullopt;
    return aTy;
}

// 9. assert(cond, msg?) -> void
static std::optional<arklang::Type> inferAssert(const CallExpr &, const Intrinsic::GetTypeFn&) {
    return arklang::Type{arklang::Type::Void};
}

// 10. panic(msg) -> void
static std::optional<arklang::Type> inferPanic(const CallExpr &, const Intrinsic::GetTypeFn&) {
    return arklang::Type{arklang::Type::Void};
}

// =============================================================================
// Registration
// =============================================================================

void registerIntrinsics(IntrinsicRegistry &r) {
    r.add("allocof", Intrinsic{inferAllocOf, 1, 1, true});
    r.add("sizeof",  Intrinsic{inferSizeOf,  0, 0, true});
    r.add("typeof",  Intrinsic{inferTypeOf,  0, 0, true});
    r.add("castof",  Intrinsic{inferCastOf,  1, 1, true});
    r.add("free",    Intrinsic{inferFree,    1, 1, false});
    r.add("addr",    Intrinsic{inferAddr,    1, 1, false});
    r.add("len",     Intrinsic{inferLen,     1, 1, false});
    r.add("min",     Intrinsic{inferMin,     2, 2, false});
    r.add("assert",  Intrinsic{inferAssert,  1, 2, false});
    r.add("panic",   Intrinsic{inferPanic,   1, 1, false});
}

} // namespace arklang