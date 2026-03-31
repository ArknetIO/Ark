#include "ark/compiler/Frontend/GenMIR.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Utils.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Types.hpp"


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

namespace {


mlir::LogicalResult mangleTypeRecursiveImpl(mlir::Type t, llvm::raw_ostream& os) {
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
                if (mlir::failed(mangleTypeRecursiveImpl(f, os))) {
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
        return mangleTypeRecursiveImpl(arr.getElementType(), os);
    }

    return mlir::failure();
}

} // namespace

namespace arklang::mir {

mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                    mlir::OpBuilder& funcBuilder,
                                    mlir::ModuleOp module,
                                    llvm::StringRef content) {
    auto* ctx = funcBuilder.getContext();
    std::string name = ".str." + std::to_string(llvm::hash_value(content));

    if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
        mlir::OpBuilder modBuilder(ctx);
        modBuilder.setInsertionPointToStart(module.getBody());

        std::string sData = content.str();
        sData.push_back('\0');

        auto i8Ty = mlir::IntegerType::get(ctx, 8);
        auto arrTy = mlir::LLVM::LLVMArrayType::get(i8Ty, sData.size());

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

mlir::LLVM::LLVMFuncOp getOrDeclPrintf(mlir::ModuleOp module, mlir::OpBuilder& builder) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
        return fn;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto i32Ty = builder.getI32Type();
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(i32Ty, {i8PtrTy}, true);

    return builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(),
        "printf",
        fnTy
    );
}

void emitPrintf(mlir::OpBuilder& b,
                mlir::Location loc,
                mlir::ModuleOp mod,
                llvm::StringRef fmt,
                mlir::ValueRange args) {
    auto printfFn = getOrDeclPrintf(mod, b);
    auto fmtStr = getOrCreateGlobalString(loc, b, mod, fmt);

    llvm::SmallVector<mlir::Value, 8> callArgs;
    callArgs.push_back(fmtStr);
    callArgs.append(args.begin(), args.end());

    b.create<mlir::LLVM::CallOp>(loc, printfFn, callArgs);
}

llvm::StringRef normalizeIntrinsicName(llvm::StringRef raw) {
    llvm::StringRef s = raw;
    if (!s.consume_front("__ark_intrinsic_")) {
        return raw;
    }

    static constexpr llvm::StringLiteral kBases[] = {
        "len",
        "dims",
        "sizeof",
        "castof",
        "sin",
        "cos",
        "max",
        "min",
        "atomic_add",
        "atomic_sub",
        "thread_id_x",
        "block_id_x"
    };

    for (auto base : kBases) {
        if (s.starts_with(base) && (s.size() == base.size() || s[base.size()] == '_')) {
            return base;
        }
    }

    return s;
}

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
    return mangleTypeRecursiveImpl(t, os);
}

mlir::FailureOr<std::string> mangleCanonicalType(mlir::Type t) {
    std::string out;
    llvm::raw_string_ostream os(out);

    if (mlir::failed(mangleTypeRecursiveImpl(t, os))) {
        return mlir::failure();
    }

    return os.str();
}

bool containsReturn(const Expr& e) {
    if (e.kind == ExprKind::Return) {
        return true;
    }

    if (auto* b = dynamic_cast<const BlockExpr*>(&e)) {
        for (const auto& stmt : b->stmts) {
            if (containsReturn(*stmt)) {
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

        assert(cur->back().hasTrait<mlir::OpTrait::IsTerminator>() &&
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

std::string astTypeToString(const arklang::Type& t) {
    switch (t.kind) {
        case arklang::Type::Void:
            return "void";
        case arklang::Type::I8:
            return "i8";
        case arklang::Type::I16:
            return "i16";
        case arklang::Type::I32:
            return "i32";
        case arklang::Type::I64:
            return "i64";
        case arklang::Type::U8:
            return "u8";
        case arklang::Type::U16:
            return "u16";
        case arklang::Type::U32:
            return "u32";
        case arklang::Type::U64:
            return "u64";
        case arklang::Type::F32:
            return "f32";
        case arklang::Type::F64:
            return "f64";
        case arklang::Type::Bool:
            return "bool";
        case arklang::Type::Str:
            return "str";
        case arklang::Type::Schema:
            return t.schemaName;
        case arklang::Type::Generic:
            return mangleGenericName(t.schemaName, t.genericArgs);
        case arklang::Type::Vec:
            return "vec<" + (t.genericArgs.empty() ? std::string("void") : astTypeToString(t.genericArgs[0])) + ">";
        case arklang::Type::Slice:
            return "slice<" + (t.genericArgs.empty() ? std::string("void") : astTypeToString(t.genericArgs[0])) + ">";
        case arklang::Type::Tensor:
            return "tensor<" + (t.genericArgs.empty() ? std::string("void") : astTypeToString(t.genericArgs[0])) + ">";
        case arklang::Type::Tuple: {
            std::string out = "(";
            for (size_t i = 0; i < t.subtypes.size(); ++i) {
                if (i) out += ", ";
                out += astTypeToString(t.subtypes[i]);
            }
            out += ")";
            return out;
        }
        case arklang::Type::Func:
            return "fn";
        default:
            return "unknown";
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
        case arklang::Type::I8:
            return "i8";
        case arklang::Type::I16:
            return "i16";
        case arklang::Type::I32:
            return "i32";
        case arklang::Type::I64:
            return "i64";
        case arklang::Type::U8:
            return "u8";
        case arklang::Type::U16:
            return "u16";
        case arklang::Type::U32:
            return "u32";
        case arklang::Type::U64:
            return "u64";
        case arklang::Type::F32:
            return "f32";
        case arklang::Type::F64:
            return "f64";
        case arklang::Type::Bool:
            return "b1";
        case arklang::Type::Str:
            return "str";
        case arklang::Type::Schema:
            return t.schemaName;
        case arklang::Type::Generic:
            return mangleGenericName(t.schemaName, t.genericArgs);
        case arklang::Type::Vec:
            return "vec_" + (t.genericArgs.empty() ? std::string("void") : mangleArg(t.genericArgs[0]));
        case arklang::Type::Slice:
            return "slice_" + (t.genericArgs.empty() ? std::string("void") : mangleArg(t.genericArgs[0]));
        case arklang::Type::Tensor:
            return "tensor_" + (t.genericArgs.empty() ? std::string("void") : mangleArg(t.genericArgs[0]));
        case arklang::Type::Tuple: {
            std::string out = "t";
            for (const auto& s : t.subtypes) {
                out += "_";
                out += mangleArg(s);
            }
            return out;
        }
        default:
            return "T";
    }
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

arklang::RValue unitAlive(mlir::OpBuilder& b, mlir::Location loc) {
    auto trueVal = b.create<mlir::LLVM::ConstantOp>(
        loc,
        b.getI1Type(),
        b.getBoolAttr(true)
    );

    auto voidVal = b.create<mlir::LLVM::UndefOp>(
        loc,
        mlir::LLVM::LLVMVoidType::get(b.getContext())
    );

    return arklang::RValue{voidVal, trueVal};
}

} // namespace arklang::mir