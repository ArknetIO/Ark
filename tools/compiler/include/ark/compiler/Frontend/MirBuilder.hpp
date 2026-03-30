#pragma once

#include "ark/compiler/Frontend/AST.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/FunctionExtras.h" // llvm::unique_function
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h" // [FIX] Required for GPU Kernel support
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <string>
#include <vector>

namespace arklang {

// =============================================================================
// RValue: The Atomic Unit of Linear Data Flow
// =============================================================================
struct RValue final {
    mlir::Value val;   // The LLVM value
    mlir::Value state; // i1: true = alive/owned, false = dropped/moved
};

// =============================================================================
// VarInfo: Stack Slot Metadata
// =============================================================================
struct VarInfo final {
    mlir::Value place;     // !llvm.ptr (Stack Slot Address)
    mlir::Value len;       // [NEW] Holds dynamic size for Kernel Tensors

    // Type Cache (Authoritative)
    mlir::Type   valueTy;  // The lowered LLVM type of the value
    mlir::Type   elemTy;   // The element type (if indexable), else null
    arklang::Type astTy;   // The semantic source type

    // State Tracking
    mlir::Value state;     // i1: Current ownership token (SSA)
    bool escapes = false;  // Optimization hint
};

// =============================================================================
// Dependency Injection (Owning Callbacks)
// =============================================================================
using ConvertTypeFn = llvm::unique_function<mlir::Type(const arklang::Type&)>;
using CoerceFn      = llvm::unique_function<mlir::Value(mlir::Location, mlir::Value, mlir::Type)>;
using IsCopyFn      = llvm::unique_function<bool(const arklang::Type&)>;

class MirBuilder {
public:
    MirBuilder(mlir::OpBuilder &builder,
               mlir::ModuleOp module,
               ConvertTypeFn convertType,
               CoerceFn coerce,
               IsCopyFn isCopyType);

    MirBuilder(const MirBuilder&) = delete;
    MirBuilder& operator=(const MirBuilder&) = delete;
    MirBuilder(MirBuilder&&) = default;
    MirBuilder& operator=(MirBuilder&&) = default;

    // --- Scope Management ---
    void enterFunction();
    void exitFunction();
    void pushScope();
    void popScope();

    bool isDeclared(llvm::StringRef name) const;
    VarInfo* lookup(llvm::StringRef name);

    // --- Target / ABI ---
    mlir::Type getIntPtrType() const { return ipTy; }
    bool isDropNeeded(const arklang::Type &t);
    // --- Core Variable Operations ---

    // Declare: Alloc Slot -> Store Init -> Register Var
    VarInfo& declareLocal(mlir::Location loc,
                          llvm::StringRef name,
                          const arklang::Type &astTy,
                          RValue init);
    // Manual Registration (For GPU Kernel Arguments / MemRefs)
    // Directly inserts a fully constructed VarInfo into the current scope.
    void defineVar(llvm::StringRef name, VarInfo info);

    // Read: Handles Move vs Copy semantics.
    // - Copy Type: Returns {Load(place), state=true}, Var state remains true.
    // - Move Type: Returns {Load(place), state=true}, Var state becomes false.
    RValue readVar(mlir::Location loc, VarInfo &v);

    // Write: Handles Drop-on-Overwrite.
    // - If v.state is true, Drop(v.place).
    // - Coerce rhs -> Store new value.
    // - v.state becomes rhs.state.
    void writeVar(mlir::Location loc, VarInfo &v, RValue rhs);

    // --- Element Access (Vectors/Slices) ---

    // Calculate Address: Validates types + GEP
    mlir::FailureOr<mlir::Value> elementPtr(mlir::Location loc,
                                            const VarInfo &v,
                                            mlir::Value idx);

    // Read Element: Equivalent to readVar but for v[i]
    // STRICT: Only allows reading 'Copy' types. Moving out of a vector element
    // requires specific swap/replace logic not handled by a simple read.
    RValue readElem(mlir::Location loc, const VarInfo &v, mlir::Value idx);

    // Write Element: Equivalent to writeVar but for v[i] = rhs
    // CRITICAL: Drops the old element at v[i] (assuming it is initialized).
    void writeElem(mlir::Location loc, const VarInfo &v, mlir::Value idx, RValue rhs);

    // --- Low Level / Memory ---
    mlir::Value createSlot(mlir::Location loc, mlir::Type ty);

    // Spill RValue to temp stack slot (e.g. for passing by-ref to runtime)
    mlir::Value spillTemp(mlir::Location loc, mlir::Type ty, mlir::Value v);

    // Generates conditional drop glue
    void dropPlaceIfOwned(mlir::Location loc,
                          const arklang::Type &ty,
                          mlir::Value place,
                          mlir::Value state);

    mlir::OpBuilder &getBuilder() { return builder; }

    // Returns pointers to all active variables in the current scope stack.
    // Used for CFG construction (Snapshotting state for Phi nodes).
    std::vector<VarInfo*> getActiveVars() {
        std::vector<VarInfo*> active;
        for (auto &scope : scopes) {
            for (auto &kv : scope.vars) active.push_back(&kv.second);
        }
        return active;
    }
    RValue borrowVar(mlir::Location loc, VarInfo &v);
    
private:
    mlir::OpBuilder &builder;
    mlir::ModuleOp module;

    ConvertTypeFn convertType;
    CoerceFn coerce;
    IsCopyFn isCopyType;

    mlir::Type ipTy; // Cached DataLayout intptr_t

    struct Scope { std::map<std::string, VarInfo> vars; };
    std::vector<Scope> scopes;

    mlir::Value normalizeIndex(mlir::Location loc, mlir::Value idx);
    mlir::Type deriveElemTy(const arklang::Type &astTy, mlir::Type valueTy);
};

} // namespace arklang
