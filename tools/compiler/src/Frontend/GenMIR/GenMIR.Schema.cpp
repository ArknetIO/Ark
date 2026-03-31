#include "ark/compiler/Frontend/GenMIR.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Schema.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Runtime.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Types.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Utils.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace arklang {

// =============================================================================
// Schema AST Resolution
// =============================================================================
// Resolve a schema declaration by:
// 1. current module
// 2. qualified import path (alias.Type)
// 3. global schema registry
// =============================================================================
const SchemaDecl* GenMIR::resolveSchemaAST(const std::string& name) {
    // 1. Local module
    if (astModule) {
        for (const auto& s : astModule->schemas) {
            if (s->name == name) return s.get();
        }
    }

    // 2. Qualified import match: alias.Type
    size_t dotPos = name.find('.');
    if (dotPos != std::string::npos) {
        std::string alias = name.substr(0, dotPos);
        std::string typeName = name.substr(dotPos + 1);

        if (importedModules.count(alias)) {
            const Module* mod = importedModules[alias];
            for (const auto& s : mod->schemas) {
                if (s->name == typeName) return s.get();
            }
        }
    }

    // 3. Global fallback
    if (globalSchemaMap.count(name)) return globalSchemaMap[name];

    return nullptr;
}

// =============================================================================
// Schema Instantiation
// =============================================================================
// Instantiate a concrete lowered schema layout and cache it in schemaRegistry.
//
// Responsibilities:
// - resolve AST declaration
// - bind generic parameters
// - materialize field metadata
// - construct lowered LLVM struct layout
// =============================================================================
const GenMIR::SchemaInfo*
GenMIR::getOrInstantiateSchema(llvm::StringRef userProvidedName,
                               llvm::ArrayRef<arklang::Type> args) {
    // 1. Resolve AST
    const SchemaDecl* decl = resolveSchemaAST(userProvidedName.str());
    if (!decl) {
        llvm::errs() << "[GenMIR] Schema '" << userProvidedName << "' not found.\n";
        return nullptr;
    }

    // 2. Build canonical instantiated name
    std::string canonicalBaseName = decl->name;
    std::string lookupKey = arklang::mir::mangleGenericName(canonicalBaseName, args);

    // 3. Reuse cached instantiation
    if (auto it = schemaRegistry.find(lookupKey); it != schemaRegistry.end()) {
        return &it->second;
    }

    // 4. Bind generic substitutions
    llvm::StringMap<arklang::Type> subst;
    if (!decl->genericParams.empty()) {
        if (args.size() != decl->genericParams.size()) {
            return nullptr;
        }

        for (size_t i = 0; i < decl->genericParams.size(); ++i) {
            subst[decl->genericParams[i]] = args[i];
        }
    }

    // 5. Initialize schema info
    SchemaInfo info;
    info.name = lookupKey;
    info.isPacked = false;
    info.isEnum = (decl->kind == SchemaDecl::Enum);

    // 6. Materialize record fields
    if (decl->kind == SchemaDecl::Record) {
        info.fieldTypes.reserve(decl->fields.size());

        for (size_t i = 0; i < decl->fields.size(); ++i) {
            const auto& f = decl->fields[i];
            info.fieldIndices[f.name] = static_cast<int64_t>(i);

            Type concrete = arklang::mir::substituteTypeParams(f.type, subst);
            info.fieldTypes.push_back(concrete);
        }
    }

    // 7. Define lowered LLVM struct
    const std::string llvmName = arklang::mir::llvmStructNameFor(lookupKey);
    mlir::LLVM::LLVMStructType llvmStruct =
        mlir::LLVM::LLVMStructType::getIdentified(builder.getContext(), llvmName);

    if (!llvmStruct.isInitialized()) {
        llvm::SmallVector<mlir::Type, 8> bodyTypes;

        if (decl->kind == SchemaDecl::Record) {
            bodyTypes.reserve(info.fieldTypes.size());
            for (const auto& ft : info.fieldTypes) {
                bodyTypes.push_back(convertType(ft));
            }
        } else if (decl->kind == SchemaDecl::Enum) {
            // Enum layout: { tag: i32, payload: [32 x i8] }
            // Current payload policy is fixed-width.
            bodyTypes.push_back(builder.getI32Type());

            auto byteType = builder.getI8Type();
            auto payloadType = mlir::LLVM::LLVMArrayType::get(byteType, 32);
            bodyTypes.push_back(payloadType);
        }

        if (mlir::failed(llvmStruct.setBody(bodyTypes, info.isPacked))) {
            return nullptr;
        }
    }

    info.loweredType = llvmStruct;
    schemaRegistry[lookupKey] = std::move(info);
    return &schemaRegistry[lookupKey];
}

// =============================================================================
// Schema Expression Construction
// =============================================================================
// Lower a record-style schema literal into an SSA struct value.
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerSchemaInit(const SchemaExpr& expr) {
    mlir::Location loc = toLoc(expr.loc);

    // 1. Resolve schema layout
    const auto* info = getOrInstantiateSchema(expr.name, expr.genericArgs);
    if (!info) return fail(expr.loc, "Unknown schema: " + expr.name);

    mlir::Type structTy = info->loweredType;
    if (!structTy) return fail(expr.loc, "Internal error: Schema type was not lowered");

    // 2. Start from undef and fill fields
    mlir::Value structVal = builder.create<mlir::LLVM::UndefOp>(loc, structTy);

    // 3. Populate explicitly initialized fields
    for (const auto& fieldInit : expr.fields) {
        auto it = info->fieldIndices.find(fieldInit.name);
        if (it == info->fieldIndices.end()) {
            return fail(expr.loc, "Field '" + fieldInit.name + "' not found in schema");
        }

        int64_t fieldIdx = it->second;

        auto valRes = lowerExpr(*fieldInit.value);
        if (mlir::failed(valRes)) return mlir::failure();

        auto llvmStruct = mlir::cast<mlir::LLVM::LLVMStructType>(structTy);
        mlir::Type fieldTy = llvmStruct.getBody()[fieldIdx];
        mlir::Value coerced = coerce(builder, loc, valRes->val, fieldTy);

        structVal = builder.create<mlir::LLVM::InsertValueOp>(
            loc,
            structVal,
            coerced,
            builder.getDenseI64ArrayAttr({fieldIdx})
        );
    }

    return RValue{structVal, arklang::mir::unitAlive(builder, loc).state};
}

// =============================================================================
// Enum Variant Construction
// =============================================================================
// Lower:
//   Option.Some(10) -> { tag = 1, payload = {10} }
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerVariantConstructor(
    const MemberCallNode& expr,
    const SchemaDecl* schemaDecl,
    int tag,
    const std::vector<Type>& payloadTypes) {

    mlir::Location loc = toLoc(expr.loc);
    mlir::Type indexTy = builder.getI64Type();

    // 1. Resolve concrete enum layout
    const SchemaInfo* schemaInfo = getOrInstantiateSchema(schemaDecl->name, {});
    if (!schemaInfo) return fail(expr.loc, "Variant schema instantiation failed");

    mlir::Type variantTy = schemaInfo->loweredType;
    auto variantSt = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(variantTy);

    // Expected invariant: { tag, payload }
    if (!variantSt || variantSt.getBody().size() < 2) {
        return fail(expr.loc, "Variant must lower to {tag, payload, ...} struct");
    }

    mlir::Type tagFieldTy = variantSt.getBody()[0];
    auto payloadArr = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(variantSt.getBody()[1]);

    if (!payloadArr || payloadArr.getElementType() != builder.getI8Type()) {
        return fail(expr.loc, "Variant payload field must be [N x i8]");
    }

    const uint64_t payloadBytes = payloadArr.getNumElements();

    auto cIndex = [&](uint64_t v) {
        return builder.create<mlir::LLVM::ConstantOp>(
            loc,
            indexTy,
            builder.getIntegerAttr(indexTy, v)
        );
    };

    auto cI32 = [&](int32_t v) {
        return builder.create<mlir::LLVM::ConstantOp>(
            loc,
            builder.getI32Type(),
            builder.getI32IntegerAttr(v)
        );
    };

    auto cI8 = [&](int8_t v) {
        return builder.create<mlir::LLVM::ConstantOp>(
            loc,
            builder.getI8Type(),
            builder.getIntegerAttr(builder.getI8Type(), v)
        );
    };

    // 2. Allocate temporary slot for the final variant
    mlir::Value variantSlot = mir->createSlot(loc, variantTy);
    auto slotPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    // 3. Zero the payload region
    {
        mlir::Value payloadBasePtr = builder.create<mlir::LLVM::GEPOp>(
            loc,
            slotPtrTy,
            variantTy,
            variantSlot,
            mlir::ValueRange{cIndex(0), cIndex(1)}
        ).getResult();

        mlir::Value lenVal = cIndex(payloadBytes);
        mlir::Value zeroByte = cI8(0);

        builder.create<mlir::LLVM::MemsetOp>(
            loc,
            payloadBasePtr,
            zeroByte,
            lenVal,
            /*isVolatile=*/false
        );
    }

    // 4. Store the tag
    {
        mlir::Value tagPtr = builder.create<mlir::LLVM::GEPOp>(
            loc,
            slotPtrTy,
            variantTy,
            variantSlot,
            mlir::ValueRange{cIndex(0), cIndex(0)}
        ).getResult();

        mlir::Value tagVal = coerce(builder, loc, cI32(tag), tagFieldTy);
        builder.create<mlir::LLVM::StoreOp>(loc, tagVal, tagPtr);
    }

    // 5. Build and copy payload, if present
    if (!payloadTypes.empty()) {
        if (expr.args.size() != payloadTypes.size()) {
            return fail(expr.loc, "Variant argument count mismatch");
        }

        llvm::SmallVector<mlir::Type, 8> caseFieldTys;
        caseFieldTys.reserve(payloadTypes.size());
        for (const auto& t : payloadTypes) {
            caseFieldTys.push_back(convertType(t));
        }

        mlir::Type caseTy =
            mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), caseFieldTys);

        mlir::DataLayout dl(module);
        const uint64_t caseBits = dl.getTypeSizeInBits(caseTy);
        if (caseBits == 0) return fail(expr.loc, "Cannot compute variant case size");

        const uint64_t caseBytes = (caseBits + 7) / 8;
        if (caseBytes > payloadBytes) {
            return fail(expr.loc, "Variant payload overflow");
        }

        mlir::Value caseSlot = mir->createSlot(loc, caseTy);

        for (size_t i = 0; i < expr.args.size(); ++i) {
            auto argRvOr = lowerExpr(*expr.args[i].value);
            if (mlir::failed(argRvOr)) return mlir::failure();

            mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
                loc,
                slotPtrTy,
                caseTy,
                caseSlot,
                mlir::ValueRange{cIndex(0), cIndex(static_cast<int64_t>(i))}
            ).getResult();

            mlir::Value stored = coerce(builder, loc, argRvOr->val, caseFieldTys[i]);
            builder.create<mlir::LLVM::StoreOp>(loc, stored, fieldPtr);
        }

        mlir::Value payloadBasePtr = builder.create<mlir::LLVM::GEPOp>(
            loc,
            slotPtrTy,
            variantTy,
            variantSlot,
            mlir::ValueRange{cIndex(0), cIndex(1)}
        ).getResult();

        builder.create<mlir::LLVM::MemcpyOp>(
            loc,
            payloadBasePtr,
            caseSlot,
            cIndex(static_cast<int64_t>(caseBytes)),
            /*isVolatile=*/false
        );
    }

    // 6. Return the fully materialized enum value
    mlir::Value result = builder.create<mlir::LLVM::LoadOp>(loc, variantTy, variantSlot);
    return RValue{result, arklang::mir::unitAlive(builder, loc).state};
}

// =============================================================================
// Singleton Schema Materialization
// =============================================================================
mlir::LogicalResult GenMIR::materializeSingletonSchema(const Module& astMod,
                                                       const SchemaDecl& decl) {
    if (!decl.genericParams.empty()) {
        return fail(decl.loc, "Singleton schema cannot be generic");
    }

    if (decl.kind == SchemaDecl::Enum) {
        return fail(decl.loc, "Singleton enums not supported yet");
    }

    // 1. Instantiate layout
    const SchemaInfo* info = getOrInstantiateSchema(decl.name, {});
    if (!info) {
        return mlir::failure();
    }

    mlir::Type structTy = info->loweredType;
    auto llvmStruct = mlir::cast<mlir::LLVM::LLVMStructType>(structTy);
    std::string globalName = mangleFunction(decl.name, &astMod);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    // 2. Create backing global
    auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
        toLoc(decl.loc),
        structTy,
        /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal,
        globalName,
        mlir::Attribute()
    );
    globalSingletons[decl.name] = globalOp;

    // 3. Build initializer region
    mlir::Region& region = globalOp.getInitializerRegion();
    mlir::Block* block = builder.createBlock(&region);
    (void)block;

    mlir::Value currentStruct =
        builder.create<mlir::LLVM::ZeroOp>(toLoc(decl.loc), structTy);

    for (size_t i = 0; i < decl.fields.size(); ++i) {
        const auto& field = decl.fields[i];
        mlir::Type fieldTy = llvmStruct.getBody()[i];
        mlir::Location fieldLoc = toLoc(decl.loc);

        if (!field.defaultValue) {
            continue;
        }

        auto valRes = lowerExpr(*field.defaultValue);
        if (mlir::failed(valRes)) {
            return mlir::failure();
        }

        mlir::Value val = valRes->val;
        val = coerce(builder, fieldLoc, val, fieldTy);

        currentStruct = builder.create<mlir::LLVM::InsertValueOp>(
            fieldLoc,
            currentStruct,
            val,
            llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)}
        );
    }

    builder.create<mlir::LLVM::ReturnOp>(toLoc(decl.loc), currentStruct);
    return mlir::success();
}

// =============================================================================
// Schema Registration
// =============================================================================
void GenMIR::registerModuleSchemas(const Module& astMod) {
    for (const auto& schema : astMod.schemas) {
        globalSchemaMap[schema->name] = schema.get();
    }
}

// =============================================================================
// Static Enum Access
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerStaticEnumAccess(mlir::Location loc,
                                                      const MemberExpr& expr) {
    if (auto baseName = resolveStaticEnumBase(*expr.object); !baseName.empty()) {
        const SchemaDecl* decl = resolveSchemaAST(baseName);
        if (decl && decl->kind == SchemaDecl::Enum) {
            int tag = -1;
            for (size_t i = 0; i < decl->variants.size(); ++i) {
                if (decl->variants[i].name == expr.member) {
                    tag = static_cast<int>(i);
                    break;
                }
            }

            if (tag >= 0) {
                auto dummyBase = std::make_unique<SymbolExpr>(expr.loc, baseName);
                std::vector<CallArg> emptyArgs;
                MemberCallNode call(
                    expr.loc,
                    std::move(dummyBase),
                    expr.member,
                    std::move(emptyArgs)
                );
                return lowerVariantConstructor(call, decl, tag, {});
            }
        }
    }

    return mlir::failure();
}

// =============================================================================
// Enum Reflection
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerEnumNameReflection(mlir::Location loc,
                                                        const MemberExpr& expr,
                                                        RValue base,
                                                        const arklang::Type& baseTy,
                                                        const GenMIR::SchemaInfo& info) {
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(base.val.getType())) {
        base.val = builder.create<mlir::LLVM::LoadOp>(loc, info.loweredType, base.val);
    }

    auto tag = builder.create<mlir::LLVM::ExtractValueOp>(
        loc,
        base.val,
        builder.getDenseI64ArrayAttr({0})
    ).getResult();

    const SchemaDecl* decl = resolveSchemaAST(baseTy.schemaName);
    if (!decl) {
        return fail(expr.loc, "Missing AST for enum: " + baseTy.schemaName);
    }

    auto resOr = lowerString(StringExpr(expr.loc, "?"));
    if (mlir::failed(resOr)) {
        return mlir::failure();
    }

    mlir::Value acc = resOr->val;

    for (int i = static_cast<int>(decl->variants.size()) - 1; i >= 0; --i) {
        auto sOr = lowerString(StringExpr(expr.loc, decl->variants[i].name));
        if (mlir::failed(sOr)) {
            return mlir::failure();
        }

        mlir::Value idx = builder.create<mlir::LLVM::ConstantOp>(
            loc,
            tag.getType(),
            builder.getIntegerAttr(tag.getType(), i)
        );

        mlir::Value cond = builder.create<mlir::LLVM::ICmpOp>(
            loc,
            mlir::LLVM::ICmpPredicate::eq,
            tag,
            idx
        );

        acc = builder.create<mlir::LLVM::SelectOp>(loc, cond, sOr->val, acc).getResult();
    }

    return RValue{acc, arklang::mir::unitAlive(builder, loc).state};
}

// =============================================================================
// Schema Field Access
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerSchemaFieldAccess(mlir::Location loc,
                                                       const MemberExpr& expr,
                                                       RValue base,
                                                       const GenMIR::SchemaInfo& info) {
    auto it = info.fieldIndices.find(expr.member);
    if (it == info.fieldIndices.end()) {
        return mlir::failure();
    }

    if (llvm::isa<mlir::LLVM::LLVMPointerType>(base.val.getType())) {
        base.val = builder.create<mlir::LLVM::LoadOp>(loc, info.loweredType, base.val);
    }

    auto ex = builder.create<mlir::LLVM::ExtractValueOp>(
        loc,
        base.val,
        builder.getDenseI64ArrayAttr({it->second})
    );

    return RValue{ex.getResult(), base.state};
}

// =============================================================================
// Schema Member Access Dispatcher
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerSchemaMemberAccess(mlir::Location loc,
                                                        const MemberExpr& expr,
                                                        RValue base,
                                                        const arklang::Type& baseTy) {
    if (baseTy.kind != arklang::Type::Schema && baseTy.kind != arklang::Type::Generic) {
        return mlir::failure();
    }

    const SchemaInfo* info = getOrInstantiateSchema(baseTy.schemaName, baseTy.genericArgs);
    if (!info) {
        return fail(expr.loc, "Unknown schema type: " + baseTy.schemaName);
    }

    if (info->isEnum && expr.member == "name") {
        return lowerEnumNameReflection(loc, expr, base, baseTy, *info);
    }

    return lowerSchemaFieldAccess(loc, expr, base, *info);
}

} // namespace arklang