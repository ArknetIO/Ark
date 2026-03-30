#include "ark/compiler/Frontend/GenMIR.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace mlir;
using namespace arklang;
using namespace arklang::mir;

// =============================================================================
// Helper: Global String Constant Creation
// Creates a [N x i8] global constant and returns an i8* pointer to it.
// Deduplicates identical strings using a hash of their content.
// =============================================================================
mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                    mlir::OpBuilder &funcBuilder,
                                    mlir::ModuleOp module,
                                    llvm::StringRef content) {
    auto* ctx = funcBuilder.getContext();
    std::string name = ".str." + std::to_string(llvm::hash_value(content));

    if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
        mlir::OpBuilder modBuilder(ctx);
        modBuilder.setInsertionPointToStart(module.getBody());

        std::string sData = content.str();
        sData.push_back('\0');

        auto i8 = mlir::IntegerType::get(ctx, 8);
        auto arrTy = mlir::LLVM::LLVMArrayType::get(i8, sData.size());

        modBuilder.create<mlir::LLVM::GlobalOp>(
            loc,
            arrTy,
            /*isConstant=*/true,
            mlir::LLVM::Linkage::Private,
            name,
            modBuilder.getStringAttr(sData)
        );
    }

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    auto symRef = mlir::SymbolRefAttr::get(ctx, name);

    mlir::Value base = funcBuilder.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, symRef);

    auto glob = module.lookupSymbol<mlir::LLVM::GlobalOp>(name);
    mlir::Type arrTy = glob.getType();

    auto i64Ty = funcBuilder.getI64Type();
    mlir::Value zero = funcBuilder.create<mlir::LLVM::ConstantOp>(
        loc,
        i64Ty,
        funcBuilder.getIntegerAttr(i64Ty, 0)
    );

    return funcBuilder.create<mlir::LLVM::GEPOp>(
        loc,
        ptrTy,
        arrTy,
        base,
        mlir::ValueRange{zero, zero}
    );
}

// Helper: Create a boolean constant (i1)
mlir::Value constBool(mlir::OpBuilder& builder, mlir::Location loc, bool val) {
    return builder.create<mlir::LLVM::ConstantOp>(
        loc,
        builder.getI1Type(),
        builder.getIntegerAttr(builder.getI1Type(), val ? 1 : 0)
    );
}

mlir::LLVM::LLVMFuncOp getOrDeclPrintf(mlir::ModuleOp module, mlir::OpBuilder& builder) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto i32Ty = builder.getI32Type();
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(i32Ty, {i8PtrTy}, true);
    return builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "printf", fnTy);
}

// Helper to emit a printf call
void emitPrintf(mlir::OpBuilder& b,
                mlir::Location loc,
                mlir::ModuleOp mod,
                llvm::StringRef fmt,
                mlir::ValueRange args) {
    auto printfFn = getOrDeclPrintf(mod, b);
    auto fmtStr = getOrCreateGlobalString(loc, b, mod, fmt);

    llvm::SmallVector<mlir::Value, 4> callArgs;
    callArgs.push_back(fmtStr);
    for (auto arg : args) {
        callArgs.push_back(arg);
    }

    b.create<mlir::LLVM::CallOp>(loc, printfFn, callArgs);
}

// =============================================================================
// Helper: Get or Declare Runtime Launch Function
// Signature: i64 __ark_launch(ptr grid, ptr kernel, i64 uid_lo, i64 uid_hi, ptr args, i64 size, i64 dim, ptr config)
// =============================================================================
mlir::LLVM::LLVMFuncOp getOrDeclareArkLaunch(mlir::ModuleOp module,
                                             mlir::OpBuilder& builder,
                                             mlir::Location loc) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_launch")) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto* ctx = builder.getContext();
    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = builder.getI64Type();

    auto fnTy = mlir::LLVM::LLVMFunctionType::get(
        i64Ty,
        {
            voidPtrTy,
            voidPtrTy,
            i64Ty,
            i64Ty,
            i64Ty,
            voidPtrTy,
            i64Ty,
            i64Ty,
            voidPtrTy
        },
        /*isVarArg=*/false
    );

    return builder.create<mlir::LLVM::LLVMFuncOp>(
        loc,
        "__ark_launch",
        fnTy,
        mlir::LLVM::Linkage::External
    );
}


llvm::StringRef normalizeIntrinsicName(llvm::StringRef raw) {
    llvm::StringRef s = raw;
    if (!s.consume_front("__ark_intrinsic_")) {
        return raw;
    }

    static constexpr llvm::StringLiteral kBases[] = {
        "len", "dims", "sizeof", "castof",
        "sin", "cos", "max", "min",
        "atomic_add", "atomic_sub",
        "thread_id_x", "block_id_x"
    };

    for (auto b : kBases) {
        if (s.starts_with(b) && (s.size() == b.size() || s[b.size()] == '_')) {
            return b;
        }
    }

    return s;
}

// Helper: Check whether an intrinsic is allowed inside GPU codegen.
bool isGpuSafeIntrinsic(llvm::StringRef rawName) {
    const llvm::StringRef name = normalizeIntrinsicName(rawName);

    return
        name == "len" ||
        name == "dims" ||
        name == "sizeof" ||
        name == "castof" ||
        name == "hash" ||
        name == "shash" ||
        name == "stable_hash" ||
        name == "sin" ||
        name == "cos" ||
        name == "max" ||
        name == "min" ||
        name == "atomic_add" ||
        name == "atomic_sub" ||
        name == "thread_id_x" ||
        name == "block_id_x";
}


mlir::LogicalResult mangleTypeRecursive(mlir::Type t, llvm::raw_ostream& os) {
    if (auto i = llvm::dyn_cast<mlir::IntegerType>(t)) {
        os << "i" << i.getWidth();
        return mlir::success();
    }
    if (llvm::isa<mlir::Float32Type>(t)) {
        os << "f32";
        return mlir::success();
    }
    if (llvm::isa<mlir::Float64Type>(t)) {
        os << "f64";
        return mlir::success();
    }
    if (llvm::isa<mlir::IndexType>(t)) {
        os << "idx";
        return mlir::success();
    }

    if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(t)) {
        if (!st.isIdentified()) {
            os << "sl";
            for (auto f : st.getBody()) {
                os << "_";
                if (mlir::failed(mangleTypeRecursive(f, os))) {
                    return mlir::failure();
                }
            }
        } else {
            os << "sn_" << st.getName();
        }
        return mlir::success();
    }

    if (auto ptr = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(t)) {
        os << "p" << ptr.getAddressSpace();
        return mlir::success();
    }

    if (auto arr = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(t)) {
        os << "a" << arr.getNumElements() << "_";
        return mangleTypeRecursive(arr.getElementType(), os);
    }

    return mlir::failure();
}

mlir::FailureOr<std::string> mangleCanonicalType(mlir::Type t) {
    std::string out;
    llvm::raw_string_ostream os(out);
    if (mlir::failed(mangleTypeRecursive(t, os))) {
        return mlir::failure();
    }
    return os.str();
}

// Helper: Recursively scan for forbidden Return statements in GPU regions
bool containsReturn(const Expr& e) {
    if (e.kind == ExprKind::Return) {
        return true;
    }

    if (auto* b = dynamic_cast<const BlockExpr*>(&e)) {
        for (const auto& s : b->stmts) {
            if (containsReturn(*s)) {
                return true;
            }
        }
        return false;
    }

    if (auto* ifStmt = dynamic_cast<const IfStmt*>(&e)) {
        if (containsReturn(*ifStmt->condition)) {
            return true;
        }
        if (ifStmt->thenBranch && containsReturn(*ifStmt->thenBranch)) {
            return true;
        }
        if (ifStmt->elseBranch && containsReturn(*ifStmt->elseBranch)) {
            return true;
        }
        return false;
    }

    if (auto* matchStmt = dynamic_cast<const MatchStmt*>(&e)) {
        if (containsReturn(*matchStmt->target)) {
            return true;
        }
        for (const auto& c : matchStmt->cases) {
            if (c.body && containsReturn(*c.body)) {
                return true;
            }
        }
        return false;
    }

    if (auto* w = dynamic_cast<const WhileStmt*>(&e)) {
        return containsReturn(*w->body);
    }
    if (auto* f = dynamic_cast<const ForStmt*>(&e)) {
        return containsReturn(*f->body);
    }
    if (auto* p = dynamic_cast<const ParLoop*>(&e)) {
        return containsReturn(*p->body);
    }

    if (auto* bin = dynamic_cast<const BinaryExpr*>(&e)) {
        return containsReturn(*bin->lhs) || containsReturn(*bin->rhs);
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&e)) {
        for (const auto& arg : call->args) {
            if (containsReturn(*arg.value)) {
                return true;
            }
        }
        return false;
    }
    if (auto* assign = dynamic_cast<const AssignStmt*>(&e)) {
        return containsReturn(*assign->target) || containsReturn(*assign->value);
    }

    if (dynamic_cast<const LambdaExpr*>(&e)) {
        return false;
    }

    return false;
}

mlir::LLVM::LLVMFuncOp getOrDeclareArkGpuLaunch(mlir::ModuleOp module,
                                                mlir::OpBuilder& builder,
                                                mlir::Location loc) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_gpu_launch")) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto i32Ty = builder.getI32Type();

    auto fnTy = mlir::LLVM::LLVMFunctionType::get(
        voidTy,
        {
            voidPtrTy,
            voidPtrTy,
            i32Ty,
            i32Ty, i32Ty, i32Ty,
            i32Ty, i32Ty, i32Ty,
            voidPtrTy
        },
        false
    );

    return builder.create<mlir::LLVM::LLVMFuncOp>(
        loc,
        "__ark_gpu_launch",
        fnTy,
        mlir::LLVM::Linkage::External
    );
}

// Helper: Robust Pointer Casting (Address Space Aware)
mlir::Value castPtrTo(mlir::OpBuilder& b,
                      mlir::Location loc,
                      mlir::Value ptr,
                      mlir::Type expectedPtrTy) {
    if (ptr.getType() == expectedPtrTy) {
        return ptr;
    }

    auto srcPtrTy = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(ptr.getType());
    auto dstPtrTy = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(expectedPtrTy);

    if (!srcPtrTy || !dstPtrTy) {
        return b.create<mlir::LLVM::BitcastOp>(loc, expectedPtrTy, ptr);
    }

    if (srcPtrTy.getAddressSpace() != dstPtrTy.getAddressSpace()) {
        ptr = b.create<mlir::LLVM::AddrSpaceCastOp>(loc, expectedPtrTy, ptr);
        if (ptr.getType() == expectedPtrTy) {
            return ptr;
        }
    }

    return b.create<mlir::LLVM::BitcastOp>(loc, expectedPtrTy, ptr);
}

// Helper: Cast any pointer to the expected runtime pointer type (handling AddrSpace)
mlir::Value castToExpectedPtr(mlir::OpBuilder& b,
                              mlir::Location loc,
                              mlir::Value v,
                              mlir::Type expectedPtrTy) {
    if (v.getType() == expectedPtrTy) {
        return v;
    }

    auto vPtr = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(v.getType());
    auto ePtr = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(expectedPtrTy);

    if (vPtr && ePtr && vPtr.getAddressSpace() != ePtr.getAddressSpace()) {
        v = b.create<mlir::LLVM::AddrSpaceCastOp>(loc, expectedPtrTy, v);
        return v;
    }

    return b.create<mlir::LLVM::BitcastOp>(loc, expectedPtrTy, v);
}

// Helper: Canonical Unit Type
mlir::Type getUnitType(mlir::OpBuilder& b) {
    return mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), {});
}

// Helper: Canonical Unit Value (Undef)
mlir::Value getUnitUndef(mlir::OpBuilder& b, mlir::Location loc) {
    return b.create<mlir::LLVM::UndefOp>(loc, getUnitType(b));
}

bool isTensorType(const arklang::Type& ty) {
    return (ty.kind == arklang::Type::Tensor) ||
           (ty.kind == arklang::Type::Generic && ty.schemaName == "Alloc");
}

// Safely splits a block for control flow insertion.
mlir::FailureOr<mlir::Block*> splitBlockAt(mlir::OpBuilder& b,
                                           mlir::Location loc,
                                           mlir::Block* cur) {
    if (b.getInsertionBlock() != cur) {
        return mlir::emitError(loc, "Internal: Builder insertion block mismatch in splitBlockAt");
    }

    mlir::Block::iterator ip = b.getInsertionPoint();
    mlir::Operation* term = cur->getTerminator();

    if (term) {
        if (term->getNextNode() != nullptr) {
            return mlir::emitError(loc, "Internal: Block has operations after terminator (malformed IR)");
        }

        assert((cur->back().hasTrait<mlir::OpTrait::IsTerminator>()) &&
               "Internal: Block terminator does not have Terminator trait");

        if (ip == cur->end()) {
            ip = term->getIterator();
        } else {
            mlir::Operation* op = &*ip;
            if (op != term && term->isBeforeInBlock(op)) {
                return mlir::emitError(loc, "Internal: Insertion point is physically past the terminator");
            }
        }
    } else {
        assert((cur->empty() || !cur->back().hasTrait<mlir::OpTrait::IsTerminator>()) &&
               "Internal: Block has trailing terminator but getTerminator() returned null");
    }

    if (ip == cur->end()) {
        mlir::Region* region = cur->getParent();
        return b.createBlock(region, std::next(cur->getIterator()));
    }

    return cur->splitBlock(ip);
}

mlir::LLVM::LLVMFuncOp getOrDeclRuntimeFn(mlir::ModuleOp module,
                                          mlir::OpBuilder& b,
                                          mlir::Location loc,
                                          llvm::StringRef name,
                                          mlir::Type retTy,
                                          llvm::ArrayRef<mlir::Type> argTys,
                                          bool isVarArg) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(module.getBody());
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(retTy, argTys, isVarArg);
    return b.create<mlir::LLVM::LLVMFuncOp>(loc, name, fnTy);
}

std::string astTypeToString(const arklang::Type& t) {
    switch (t.kind) {
        case arklang::Type::I32: return "i32";
        case arklang::Type::F32: return "f32";
        case arklang::Type::I64: return "i64";
        case arklang::Type::F64: return "f64";
        case arklang::Type::Bool: return "bool";
        case arklang::Type::Schema: return t.schemaName;
        default: return "unknown";
    }
}

arklang::Type substituteTypeParams(const arklang::Type& src,
                                   const llvm::StringMap<arklang::Type>& subst) {
    arklang::Type out = src;

    if (out.kind == arklang::Type::Schema) {
        auto it = subst.find(out.schemaName);
        if (it != subst.end()) {
            return it->second;
        }
    }

    if (!out.genericArgs.empty()) {
        for (auto& arg : out.genericArgs) {
            arg = substituteTypeParams(arg, subst);
        }
    }

    if (out.kind == arklang::Type::Tuple) {
        for (auto& sub : out.subtypes) {
            sub = substituteTypeParams(sub, subst);
        }
    }

    if (out.kind == arklang::Type::Func) {
        for (auto& param : out.paramTypes) {
            param = substituteTypeParams(param, subst);
        }
        if (out.funcReturnType) {
            *out.funcReturnType = substituteTypeParams(*out.funcReturnType, subst);
        }
    }

    return out;
}

std::string mangleGenericName(llvm::StringRef baseName,
                              llvm::ArrayRef<arklang::Type> args);

std::string mangleArg(const arklang::Type& t) {
    switch (t.kind) {
        case arklang::Type::I8: return "i8";
        case arklang::Type::I16: return "i16";
        case arklang::Type::I32: return "i32";
        case arklang::Type::I64: return "i64";
        case arklang::Type::U8: return "u8";
        case arklang::Type::U16: return "u16";
        case arklang::Type::U32: return "u32";
        case arklang::Type::U64: return "u64";
        case arklang::Type::F32: return "f32";
        case arklang::Type::F64: return "f64";
        case arklang::Type::Bool: return "b1";
        case arklang::Type::Str: return "str";
        case arklang::Type::Schema: return t.schemaName;
        case arklang::Type::Generic: return mangleGenericName(t.schemaName, t.genericArgs);
        case arklang::Type::Vec:
            return "vec_" + (t.genericArgs.empty() ? "void" : mangleArg(t.genericArgs[0]));
        case arklang::Type::Slice:
            return "slice_" + (t.genericArgs.empty() ? "void" : mangleArg(t.genericArgs[0]));
        case arklang::Type::Tensor:
            return "tensor_" + (t.genericArgs.empty() ? "void" : mangleArg(t.genericArgs[0]));
        case arklang::Type::Tuple: {
            std::string out = "t";
            for (auto& s : t.subtypes) {
                out += "_";
                out += mangleArg(s);
            }
            return out;
        }
        default:
            break;
    }
    return "T";
}

std::string mangleGenericName(llvm::StringRef baseName,
                              llvm::ArrayRef<arklang::Type> args) {
    if (args.empty()) {
        return baseName.str();
    }

    std::string out = baseName.str();
    for (const auto& a : args) {
        out += "_";
        out += mangleArg(a);
    }
    return out;
}

std::string llvmStructNameFor(llvm::StringRef arkName) {
    std::string out;
    out.reserve(11 + arkName.size());
    out += "ark.struct.";
    out += arkName;
    return out;
}