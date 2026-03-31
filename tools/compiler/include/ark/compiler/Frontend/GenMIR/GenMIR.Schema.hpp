#pragma once

// =============================================================================
// GenMIR Schema Surface
// =============================================================================
// Class-fragment header.
//
// IMPORTANT:
// - Include this ONLY from inside class GenMIR in GenMIR.hpp.
// - Do not duplicate these declarations manually in GenMIR.hpp.
// =============================================================================

// -----------------------------------------------------------------------------
// Schema Resolution / Instantiation
// -----------------------------------------------------------------------------
const SchemaDecl* resolveSchemaAST(const std::string& name);

const SchemaInfo* getOrInstantiateSchema(llvm::StringRef userProvidedName,
                                         llvm::ArrayRef<arklang::Type> args);

// -----------------------------------------------------------------------------
// Schema Registration / Materialization
// -----------------------------------------------------------------------------
void registerModuleSchemas(const Module& astMod);

mlir::LogicalResult materializeSingletonSchema(const Module& astMod,
                                               const SchemaDecl& decl);

// -----------------------------------------------------------------------------
// Schema / Enum Construction
// -----------------------------------------------------------------------------
mlir::FailureOr<RValue> lowerSchemaInit(const SchemaExpr& expr);

mlir::FailureOr<RValue> lowerVariantConstructor(const MemberCallNode& expr,
                                                const SchemaDecl* schemaDecl,
                                                int tag,
                                                const std::vector<Type>& payloadTypes);

// -----------------------------------------------------------------------------
// Schema / Enum Member Access
// -----------------------------------------------------------------------------
mlir::FailureOr<RValue> lowerStaticEnumAccess(mlir::Location loc,
                                              const MemberExpr& expr);

mlir::FailureOr<RValue> lowerEnumNameReflection(mlir::Location loc,
                                                const MemberExpr& expr,
                                                RValue base,
                                                const arklang::Type& baseTy,
                                                const SchemaInfo& info);

mlir::FailureOr<RValue> lowerSchemaFieldAccess(mlir::Location loc,
                                               const MemberExpr& expr,
                                               RValue base,
                                               const SchemaInfo& info);

mlir::FailureOr<RValue> lowerSchemaMemberAccess(mlir::Location loc,
                                                const MemberExpr& expr,
                                                RValue base,
                                                const arklang::Type& baseTy);