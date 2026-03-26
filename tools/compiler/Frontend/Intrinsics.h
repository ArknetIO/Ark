#pragma once

#include "Frontend/AST.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <optional>
#include <cstdint>

namespace arklang {
// [CRITICAL FIX] This declaration was missing!
// It allows GenMIR to see the helper function defined in Intrinsics.cpp.
bool isIntrinsicFn(llvm::StringRef name);

struct Intrinsic final {
    // Callback signature: generic AST expression -> Resolved Type
    using GetTypeFn = std::function<arklang::Type(const Expr&)>;
    
    // Inference signature: Call AST + Type Resolver -> Result Type (or nullopt)
    using InferFn = std::function<std::optional<arklang::Type>(const CallExpr&, const GetTypeFn&)>;

    InferFn infer;
    uint8_t minArgs = 0;
    uint8_t maxArgs = 0;
    
    // Does this intrinsic require explicit generic parameters? (e.g. sizeof<T>)
    bool requiresGeneric = false;

    // Helper to validate argument count
    bool acceptsArgCount(size_t n) const { 
        return n >= minArgs && n <= maxArgs; 
    }
};

class IntrinsicRegistry final {
public:
    void add(llvm::StringRef name, Intrinsic i) { 
        map[name] = std::move(i); 
    }

    const Intrinsic* lookup(llvm::StringRef name) const {
        auto it = map.find(name);
        return it == map.end() ? nullptr : &it->second;
    }

private:
    llvm::StringMap<Intrinsic> map;
};

// Global registration entry point
void registerIntrinsics(IntrinsicRegistry &r);

} // namespace arklang