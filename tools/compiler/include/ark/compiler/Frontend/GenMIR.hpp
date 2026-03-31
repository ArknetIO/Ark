#pragma once

#include "ark/compiler/Frontend/AST.hpp"
#include "ark/compiler/Frontend/MirBuilder.hpp"
#include "ark/compiler/Frontend/BuiltinNamespaces.hpp"
#include "ark/compiler/Frontend/Intrinsics.hpp"
#include "ark/IR/ArkMirOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace arklang {

// Forward declaration for the HUD sink used during lowering.
namespace hud {
class Hud;
}

// =============================================================================
// GenMIR
// =============================================================================
// Primary AST -> MIR lowering driver.
//
// Responsibilities:
// - register modules and imports
// - resolve functions and schemas across module boundaries
// - lower Ark AST into MLIR/LLVM-compatible MIR
// - delegate storage/scope/drop mechanics to MirBuilder
//
// Design notes:
// - MirBuilder owns variable/scoping mechanics
// - GenMIR owns semantic lowering policy
// - schema-specific declarations are split into a fragment header to keep this
//   file manageable without duplicating declarations
// =============================================================================
class GenMIR {
public:
    GenMIR(mlir::ModuleOp module, mlir::OpBuilder& builder, arklang::hud::Hud& hud_ref);

    // Register a module into the global symbol/schema registries.
    void registerModule(const Module& astMod, bool isRoot = false);

    // Compile one module after registration/import wiring.
    mlir::LogicalResult compileModule(const Module& astMod, bool isRoot = false);

    // Import visibility management.
    void registerImport(const std::string& alias, const Module* mod);
    void clearImports();

    // Legacy wrapper kept for compatibility with older call sites.
    mlir::LogicalResult lowerModule(const Module& astMod);

private:
    // =========================================================================
    // GPU Lowering State
    // =========================================================================

    // Cache of emitted GPU container modules.
    llvm::StringMap<mlir::gpu::GPUModuleOp> gpuModules;

    // =========================================================================
    // Shared Indexing / Data Pointer Extraction
    // =========================================================================

    struct IndexBaseInfo final {
        mlir::Value dataPtr;       // underlying data pointer
        mlir::Type elemLlvmTy;     // lowered element type
        arklang::Type elemAstTy;   // semantic element type
        mlir::Type indexLlvmTy;    // DL-derived integer index type
    };

    // DataLayout-derived index integer type.
    mlir::Type getIndexLlvmTyOrDie(mlir::Location loc);

    // Normalize an integer index value to the requested integer type.
    mlir::Value normalizeIndexTo(mlir::Location loc, mlir::Value idx, mlir::Type indexTy);

    // Extract data pointer + element metadata for Vec/Slice/Tensor/Alloc bases.
    mlir::FailureOr<IndexBaseInfo> extractDataPtr(mlir::Location loc,
                                                  VarInfo& baseVar,
                                                  const RValue& baseRv);

    // Compute element address from base metadata + index.
    mlir::FailureOr<mlir::Value> indexGep(mlir::Location loc,
                                          const IndexBaseInfo& info,
                                          mlir::Value idxVal);

    // =========================================================================
    // Core Lowering State
    // =========================================================================

    arklang::hud::Hud& hud;

    // Dependency-injection callbacks passed to MirBuilder.
    std::function<mlir::Type(const arklang::Type&)> convertTypeCb;
    std::function<mlir::Value(mlir::Location, mlir::Value, mlir::Type)> coerceCb;
    std::function<bool(const arklang::Type&)> isCopyTypeCb;

    // Registry used by expression/type inference for intrinsic lowering.
    IntrinsicRegistry intrinsicRegistry;

    // Engine that handles slots, scopes, drops, and address computations.
    std::unique_ptr<MirBuilder> mir;

    // Builtin namespace lowering frontend.
    std::unique_ptr<frontend::BuiltinNsLowering> builtinLowering;

    // Current AST module being compiled.
    const Module* astModule = nullptr;

    // Imported-module visibility map: alias -> module.
    llvm::StringMap<const Module*> importedModules;

    // Global registries built during registration.
    llvm::StringMap<const Function*> globalFunctionMap;
    llvm::StringMap<const SchemaDecl*> globalSchemaMap;

    // Per-module function name mangling prefixes.
    std::map<const Module*, std::string> modulePrefixes;
    std::string currentContextPrefix;

    // Current lowering domain (host/cpu/gpu).
    Domain currentFnDomain = Domain::Host;

    // Singleton schema globals indexed by schema name.
    llvm::StringMap<mlir::LLVM::GlobalOp> globalSingletons;

    // =========================================================================
    // Type / Schema Internals
    // =========================================================================

    struct SchemaInfo {
        std::string name;
        llvm::StringMap<int64_t> fieldIndices;
        std::vector<Type> fieldTypes;
        bool isPacked = false;
        mlir::Type loweredType;
        bool isEnum = false;
    };

    // Function parameter name table keyed by mangled function symbol.
    llvm::StringMap<std::vector<std::string>> funcParamNames;

    // Concrete instantiated schema layouts keyed by mangled schema name.
    llvm::StringMap<SchemaInfo> schemaRegistry;

    // =========================================================================
    // Small Helper Structs
    // =========================================================================

    struct DimsResult {
        mlir::Value w;
        mlir::Value h;
    };

    // -------------------------------------------------------------------------
    // Schema lowering surface
    // -------------------------------------------------------------------------
    // IMPORTANT:
    // - This fragment must stay inside class GenMIR.
    // - It must appear after SchemaInfo is declared.
    // - Do not duplicate these declarations elsewhere in this class.
    #include "ark/compiler/Frontend/GenMIR/GenMIR.Schema.hpp"

    
    #include "ark/compiler/Frontend/GenMIR/GenMIR.GPU.hpp"
    #include "ark/compiler/Frontend/GenMIR/GenMIR.Intrinsics.hpp"


    // =========================================================================
    // MLIR Context
    // =========================================================================

    mlir::ModuleOp module;
    mlir::OpBuilder& builder;

    // =========================================================================
    // Symbol Resolution / Naming
    // =========================================================================

    // Resolve the final emitted symbol name for a function in a given module.
    std::string mangleFunction(const std::string& name, const Module* mod);

    // Resolve the symbolic base used for static enum member access.
    std::string resolveStaticEnumBase(const Expr& obj);

    // =========================================================================
    // High-Level Function Lowering
    // =========================================================================

    mlir::LogicalResult lowerFunction(const Function& fn);
    mlir::LogicalResult lowerHostFunction(const Function& fn);
    mlir::LogicalResult lowerCpuKernel(const Function& fn);

    // Attach capability metadata / policy to the current lowered function.
    void injectCapabilities(const Function& fn);

    // =========================================================================
    // Statement Lowering
    // =========================================================================

    mlir::LogicalResult lowerStmt(const Expr& stmt);
    mlir::LogicalResult lowerBlock(const BlockExpr& blk);

    // Control flow
    mlir::LogicalResult lowerReturn(const ReturnStmt& ret);
    mlir::LogicalResult lowerFor(const ForStmt& stmt);
    mlir::LogicalResult lowerWhile(const WhileStmt& stmt);
    mlir::LogicalResult lowerParLoop(const ParLoop& loop);

    // Variables / assignment
    mlir::LogicalResult lowerVarDecl(const VarDecl& decl);
    mlir::LogicalResult lowerAssign(const AssignStmt& asn);

    // =========================================================================
    // Expression Lowering
    // =========================================================================
    // All expression handlers return RValue { value, state }.
    // =========================================================================

    mlir::FailureOr<RValue> lowerExpr(const Expr& expr);

    mlir::FailureOr<RValue> lowerCall(const CallExpr& call);
    mlir::FailureOr<RValue> lowerLiteral(const LiteralExpr& lit);
    mlir::FailureOr<RValue> lowerString(const StringExpr& str);
    mlir::FailureOr<RValue> lowerBinary(const BinaryExpr& bin);
    mlir::FailureOr<RValue> lowerSymbol(const SymbolExpr& sym);
    mlir::FailureOr<RValue> lowerIndex(const IndexExpr& idx);
    mlir::FailureOr<RValue> lowerLaunch(const LaunchExpr& ln);
    mlir::FailureOr<RValue> lowerAwait(const AwaitExpr& aw);
    mlir::FailureOr<RValue> lowerTuple(const TupleExpr& expr);
    mlir::FailureOr<RValue> lowerArrayLiteral(const ArrayLiteral& expr);
    mlir::FailureOr<RValue> lowerLambda(const LambdaExpr& expr);

    // Control-flow expressions
    mlir::FailureOr<RValue> lowerIf(const IfStmt& stmt);
    mlir::FailureOr<RValue> lowerMatch(const MatchStmt& stmt);
    mlir::FailureOr<RValue> lowerPrint(const PrintStmt& stmt);

    // Block-as-value helper
    mlir::FailureOr<RValue> lowerBlockAsValue(const BlockExpr& blk, mlir::Type expectedRetTy);

    // Allocations / containers
    mlir::FailureOr<RValue> lowerAlloc(const AllocExpr& allocExpr,
                                       const std::vector<CallArg>& args);

    // Member access / calls
    mlir::FailureOr<RValue> lowerMemberAccess(const MemberExpr& expr);
    mlir::FailureOr<RValue> lowerMemberCall(const MemberCallNode& expr);

    // Builtin namespace router
    bool isBuiltinNamespace(llvm::StringRef baseName);
    mlir::FailureOr<RValue> lowerBuiltin(mlir::Location loc,
                                         llvm::StringRef baseName,
                                         llvm::StringRef member,
                                         llvm::ArrayRef<mlir::Value> args);

    // L-value / place helpers
    mlir::FailureOr<mlir::Value> lowerMemberPlace(const MemberExpr& expr);
    mlir::FailureOr<mlir::Value> lowerExprAsPlace(const Expr& expr);

    // Slice lowering
    mlir::FailureOr<RValue> lowerSlice(const IndexExpr& expr, const RangeExpr& range);

    // Vector method lowering
    mlir::FailureOr<RValue> lowerVectorMethod(const MemberCallNode& expr,
                                              const VarInfo& var,
                                              const arklang::Type& elemAstTy);

    // Prepare concrete lowered arguments for a call.
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> prepareCallArgs(
        SourceLoc loc,
        mlir::LLVM::LLVMFuncOp callee,
        const std::vector<CallArg>& args,
        const std::string& funcName
    );

    // =========================================================================
    // Type / Inference / Diagnostics Utilities
    // =========================================================================

    // Lower an AST type into the current MLIR/LLVM representation.
    mlir::Type convertType(const Type& t);

    // Strict coercion entry point used by lowerers and MirBuilder.
    mlir::Value coerce(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::Type target);

    // Convenience overload bound to this->builder.
    mlir::Value coerce(mlir::Location loc, mlir::Value v, mlir::Type target);

    // Copy-vs-move classification used by lowering and drop logic.
    bool isCopyType(const Type& t);

    // Infer semantic AST type for an expression.
    arklang::Type getExprType(const Expr& expr);

    // Source location conversion helpers.
    mlir::Location toLoc(SourceLoc loc);
    mlir::LogicalResult fail(SourceLoc loc, const llvm::Twine& msg);

    // Internal helper: compute address of a vector element for indexing.
    mlir::FailureOr<mlir::Value> getVectorElementAddress(mlir::Location loc,
                                                         VarInfo& baseVar,
                                                         mlir::Value indexVal);

    // Deprecated: prefer getExprType or direct semantic access.
    Type inferASTType(const Expr* expr);
};

} // namespace arklang