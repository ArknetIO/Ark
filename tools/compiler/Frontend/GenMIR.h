#pragma once

#include "Frontend/AST.h"
#include "Frontend/MirBuilder.h" // The Engine
#include "Frontend/BuiltinNamespaces.h"
#include "Frontend/Intrinsics.h" // [CRITICAL] Needed for isIntrinsicFn
#include "ark/IR/ArkMirOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // [FIX] Added for LLVM::GlobalOp
#include "mlir/Dialect/Func/IR/FuncOps.h"   // [FIX] Added for func::FuncOp

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <vector>
#include <map>
#include <functional>

namespace arklang {

// Forward Declaration for Hud
namespace hud { class Hud; }


/// The primary AST -> MIR lowering driver.
/// Focusing on Logic traversal, delegating plumbing to MirBuilder.
class GenMIR {
public:
    GenMIR(mlir::ModuleOp module, mlir::OpBuilder &builder, arklang::hud::Hud &hud_ref);

    // Register a module into the global registry (Header Scan).
    void registerModule(const Module &astMod, bool isRoot = false);

    // Main entry point to compile a specific module.
    mlir::LogicalResult compileModule(const Module &astMod, bool isRoot = false);

    void registerImport(const std::string& alias, const Module* mod);
    void clearImports();
    
    // Legacy wrapper
    mlir::LogicalResult lowerModule(const Module &astMod);

private:
    llvm::StringMap<mlir::gpu::GPUModuleOp> gpuModules;
    
    mlir::LogicalResult emitGpuDeviceKernel(const Function &fn);
    mlir::LogicalResult emitGpuHostStub(const Function &fn);
    mlir::gpu::GPUModuleOp getOrCreateGpuModule();
    // --- Indexing / data pointer extraction (shared) ---
    struct IndexBaseInfo final {
        mlir::Value dataPtr;      // !llvm.ptr (void*)
        mlir::Type  elemLlvmTy;   // lowered element type
        arklang::Type elemAstTy;  // semantic element type
        mlir::Type  indexLlvmTy;  // intptr (DL-derived)
    };
    // DataLayout-derived intptr type (no guessing i64).
    mlir::Type getIndexLlvmTyOrDie(mlir::Location loc);

    // Normalize any integer index to a requested integer type (zext/trunc).
    mlir::Value normalizeIndexTo(mlir::Location loc, mlir::Value idx, mlir::Type indexTy);

    // Extract base data pointer + element type for Vec/Slice/Tensor/Alloc.
    // The VarInfo is non-const because some cases may need to materialize loads/spills uniformly.
    mlir::FailureOr<IndexBaseInfo> extractDataPtr(mlir::Location loc, VarInfo &baseVar, const RValue &baseRv);

    mlir::FailureOr<mlir::Value> indexGep(mlir::Location loc, const IndexBaseInfo &info, mlir::Value idxVal);

    // =========================================================================
    // Core State & Context
    // =========================================================================
    arklang::hud::Hud &hud;
    
    // Callbacks for MirBuilder dependency injection
    std::function<mlir::Type(const arklang::Type&)> convertTypeCb;
    std::function<mlir::Value(mlir::Location, mlir::Value, mlir::Type)> coerceCb;
    std::function<bool(const arklang::Type&)> isCopyTypeCb;
    
    // Registry to store type inference rules
    IntrinsicRegistry intrinsicRegistry;

    // The Engine: Handles Allocas, Scopes, Variables, Drops, and GEPs.
    std::unique_ptr<MirBuilder> mir; 

    std::unique_ptr<frontend::BuiltinNsLowering> builtinLowering;
    const Module* astModule = nullptr;

    // Import Visibility
    llvm::StringMap<const Module*> importedModules;
    
    // Global Registries
    llvm::StringMap<const Function*> globalFunctionMap;
    llvm::StringMap<const SchemaDecl*> globalSchemaMap;
    std::map<const Module*, std::string> modulePrefixes;
    std::string currentContextPrefix;
    Domain currentFnDomain = Domain::Host; // [NEW] Track current execution domain

    // Singleton Registry: Maps schema names to their global variable definitions.
    llvm::StringMap<mlir::LLVM::GlobalOp> globalSingletons;

    // =========================================================================
    // Type System Internals
    // =========================================================================
    struct SchemaInfo {
        std::string name; 
        llvm::StringMap<int64_t> fieldIndices;
        std::vector<Type> fieldTypes;
        bool isPacked = false;
        mlir::Type loweredType;
        bool isEnum = false;
    };

    llvm::StringMap<std::vector<std::string>> funcParamNames;
    llvm::StringMap<SchemaInfo> schemaRegistry; 

    // MLIR Context
    mlir::ModuleOp module;
    mlir::OpBuilder &builder;

    // =========================================================================
    // Helper Structs
    // =========================================================================
    // [FIX] Define DimsResult inside class to match .cpp scope
    struct DimsResult { mlir::Value w; mlir::Value h; };

    // =========================================================================
    // Safety & Intrinsic Helpers (The Hardened Core)
    // =========================================================================
    
    // Create or retrieve the private panic function
    mlir::func::FuncOp getOrCreatePanicFn();
    
    // Emit a call to panic and an unreachable instruction
    RValue emitPanic(mlir::Location loc, mlir::Value cstrI8Ptr);
    
    // Emit a host-side runtime assertion (if (!cond) panic(msg))
    void emitHostAssert(mlir::Location loc, mlir::Value condI1, llvm::StringRef msg);
    
    // Verify parallel loop bounds on host before launch
    void assertParBoundsHost(mlir::Location loc, mlir::Value startI64, mlir::Value limitI64);
    
    // Create a type-stable intrinsic symbol
    mlir::FailureOr<mlir::func::FuncOp> getOrCreateIntrinsic(
        mlir::Location loc, llvm::StringRef base, mlir::Type argTy, mlir::TypeRange resTys);

    // Get container length via dialect-pure intrinsic call
    mlir::FailureOr<mlir::Value> getContainerLen(mlir::Location loc, const Expr &expr);
    
    // Get dimensions via dialect-pure intrinsic call
    mlir::FailureOr<DimsResult> getDimsFromCall(mlir::Location loc, const CallExpr& call);
    
    // Recursive scanner for illegal GPU operations
    bool checkGpuLegality(const Expr& e, std::string& outError);

    // =========================================================================
    // Mangles & Resolvers
    // =========================================================================
    std::string mangleFunction(const std::string& name, const Module* mod);
    const SchemaDecl* resolveSchemaAST(const std::string& name);
    std::string resolveStaticEnumBase(const Expr &obj);

    // Generates the underlying !llvm.struct definition if needed.
    const SchemaInfo *getOrInstantiateSchema(llvm::StringRef baseName,
                                             llvm::ArrayRef<Type> args);

    // =========================================================================
    // High-Level Lowering Methods
    // =========================================================================

    mlir::LogicalResult lowerFunction(const Function &fn);
    mlir::LogicalResult lowerHostFunction(const Function &fn);
    mlir::LogicalResult lowerCpuKernel(const Function &fn);
    mlir::LogicalResult lowerGpuKernel(const Function &fn);
    
    void injectCapabilities(const Function &fn);

    // =========================================================================
    // Statement Handlers (Return LogicalResult)
    // =========================================================================

    mlir::LogicalResult lowerStmt(const Expr &stmt);
    mlir::LogicalResult lowerBlock(const BlockExpr &blk);
    
    // Control Flow
    mlir::LogicalResult lowerReturn(const ReturnStmt &ret);
    mlir::LogicalResult lowerFor(const ForStmt &stmt);
    mlir::LogicalResult lowerWhile(const WhileStmt &stmt);
    
    // [NEW] Parallel Loop Lowering (The 'par' Keyword)
    mlir::LogicalResult lowerParLoop(const ParLoop &loop);

    // Variable Declaration & Assignment
    mlir::LogicalResult lowerVarDecl(const VarDecl &decl);
    mlir::LogicalResult lowerAssign(const AssignStmt &asn);

    // =========================================================================
    // Expression Handlers (Return RValue)
    // NOTE: All expression handlers return RValue {Value, State}
    // =========================================================================
    
    mlir::FailureOr<RValue> lowerExpr(const Expr &expr);

    mlir::FailureOr<RValue> lowerCall(const CallExpr &call);
    mlir::FailureOr<RValue> lowerLiteral(const LiteralExpr &lit);
    mlir::FailureOr<RValue> lowerString(const StringExpr &str);
    mlir::FailureOr<RValue> lowerBinary(const BinaryExpr &bin);
    mlir::FailureOr<RValue> lowerSymbol(const SymbolExpr &sym);
    mlir::FailureOr<RValue> lowerIndex(const IndexExpr &idx);
    mlir::FailureOr<RValue> lowerLaunch(const LaunchExpr &ln);
    mlir::FailureOr<RValue> lowerAwait(const AwaitExpr &aw);
    mlir::FailureOr<RValue> lowerTuple(const TupleExpr &expr);
    mlir::FailureOr<RValue> lowerArrayLiteral(const ArrayLiteral &expr);
    mlir::FailureOr<RValue> lowerLambda(const LambdaExpr &expr);

    // Control Flow Expressions
    mlir::FailureOr<RValue> lowerIf(const IfStmt &stmt);
    mlir::FailureOr<RValue> lowerMatch(const MatchStmt &stmt);
    mlir::FailureOr<RValue> lowerPrint(const PrintStmt &stmt);
    
    // Special Lowering Helpers
    mlir::FailureOr<RValue> lowerBlockAsValue(const BlockExpr &blk, mlir::Type expectedRetTy);

    // Complex Types & Allocations
    mlir::FailureOr<RValue> lowerSchemaInit(const SchemaExpr &expr);
    mlir::FailureOr<RValue> lowerAlloc(const AllocExpr &allocExpr, const std::vector<CallArg> &args);

    // Member Access & Methods
    mlir::FailureOr<RValue> lowerMemberAccess(const MemberExpr &expr);
    mlir::FailureOr<RValue> lowerMemberCall(const MemberCallNode &expr);
    
    // [NEW] Builtin Namespace Router
    bool isBuiltinNamespace(llvm::StringRef baseName);
    mlir::FailureOr<RValue> lowerBuiltin(mlir::Location loc, llvm::StringRef baseName, llvm::StringRef member, llvm::ArrayRef<mlir::Value> args);

    // Helper: Address calculation for member assignment (L-Value)
    mlir::FailureOr<mlir::Value> lowerMemberPlace(const MemberExpr &expr);
    mlir::FailureOr<mlir::Value> lowerExprAsPlace(const Expr &expr);

    mlir::FailureOr<RValue> lowerSlice(const IndexExpr &expr, const RangeExpr &range);
    
    mlir::FailureOr<RValue> lowerVariantConstructor(const MemberCallNode &expr, 
                                                    const SchemaDecl* schemaDecl,
                                                    int tag,
                                                    const std::vector<Type>& payloadTypes);

    mlir::FailureOr<RValue> lowerVectorMethod(const MemberCallNode &expr, 
                                        const VarInfo &var, 
                                        const arklang::Type &elemAstTy);
    
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> prepareCallArgs(
            SourceLoc loc, 
            mlir::LLVM::LLVMFuncOp callee, 
            const std::vector<CallArg> &args,
            const std::string &funcName
        );

    // --- Intrinsic & Helper Methods ---
    
    // Dispatcher for built-in intrinsics (len, allocof, etc.)
    mlir::FailureOr<RValue> lowerIntrinsicCall(const CallExpr &call, llvm::StringRef name);

    // Enforce string type and handle implicit conversions if needed
    mlir::FailureOr<mlir::Value> forceStrValue(mlir::Location loc, mlir::Value v, arklang::Type astTy);

    // --- Utilities ---
    // [Dependency Injection Targets for MirBuilder]
    mlir::Type convertType(const Type &t);
    
    // Coerce with OpBuilder (Strict Signature)
    mlir::Value coerce(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v, mlir::Type target);
    // Overload for internal use (calls above)
    mlir::Value coerce(mlir::Location loc, mlir::Value v, mlir::Type target);
    
    bool isCopyType(const Type &t);
    arklang::Type getExprType(const Expr &expr);

    mlir::Location toLoc(SourceLoc loc);
    mlir::LogicalResult fail(SourceLoc loc, const llvm::Twine &msg);
    
    // Internal Helper: get address of vector element for indexing
    mlir::FailureOr<mlir::Value>
    getVectorElementAddress(mlir::Location loc, VarInfo &baseVar, mlir::Value indexVal);

    
    // Deprecated: Prefer getExprType or direct access
    Type inferASTType(const Expr* expr);
};

} // namespace arklang