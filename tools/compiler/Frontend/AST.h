#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/StringMap.h"

namespace arklang {

// =============================================================================
// Effect Flags
// =============================================================================
// These constants are used throughout parsing, semantic analysis, and codegen
// to represent function capabilities as a compact bitmask.
static constexpr uint32_t EFF_FS  = 1u << 0;
static constexpr uint32_t EFF_NET = 1u << 1;
static constexpr uint32_t EFF_IO  = 1u << 2;
static constexpr uint32_t EFF_SYS = 1u << 3;

// Forward declaration so ImportDecl can hold a unique_ptr<Module>.
struct Module;

// =============================================================================
// 1. Source Location & Execution Context
// =============================================================================

/**
 * Precise source position used for diagnostics and AST node provenance.
 */
struct SourceLoc {
    std::string file;
    int line = 0;
    int col = 0;
};

/**
 * Execution domain for functions and kernels.
 */
enum class Domain {
    Host,
    CPU,
    GPU
};

/**
 * Memory / placement descriptor.
 *
 * Compatibility goals:
 * - Preserve the old RAM/GPU + deviceId model
 * - Extend it to support:
 *     @gpu:0
 *     @gpu:"route-id"
 *     @gpu:route
 *     @runtime preset
 *
 * Notes:
 * - kind describes the broad memory family
 * - addressKind refines how the target is addressed
 * - address stores route strings, symbol names, or runtime preset names
 */
struct Space {
    enum Kind {
        RAM,
        GPU
    } kind = RAM;

    enum class AddressKind : uint8_t {
        Default,
        DeviceId,
        RouteLiteral,
        RouteSymbol,
        RuntimePreset
    };

    AddressKind addressKind = AddressKind::Default;
    int deviceId = 0;
    std::string address;

    static Space host() {
        return Space{};
    }

    static Space gpuDevice(int id) {
        Space s;
        s.kind = GPU;
        s.addressKind = AddressKind::DeviceId;
        s.deviceId = id;
        return s;
    }

    static Space gpuRouteLiteral(std::string route) {
        Space s;
        s.kind = GPU;
        s.addressKind = AddressKind::RouteLiteral;
        s.address = std::move(route);
        return s;
    }

    static Space gpuRouteSymbol(std::string symbol) {
        Space s;
        s.kind = GPU;
        s.addressKind = AddressKind::RouteSymbol;
        s.address = std::move(symbol);
        return s;
    }

    static Space runtimePreset(std::string presetName) {
        Space s;
        s.kind = GPU;
        s.addressKind = AddressKind::RuntimePreset;
        s.address = std::move(presetName);
        return s;
    }

    bool isDefault() const {
        return kind == RAM && addressKind == AddressKind::Default && deviceId == 0 && address.empty();
    }

    bool operator==(const Space& other) const {
        return kind == other.kind &&
               addressKind == other.addressKind &&
               deviceId == other.deviceId &&
               address == other.address;
    }

    bool operator!=(const Space& other) const {
        return !(*this == other);
    }
};

/**
 * Security capability model.
 * Functions must explicitly declare these effects to perform sensitive actions.
 */
enum Effect : uint32_t {
    None = 0,
    IO   = 1 << 0,
    NET  = 1 << 1,
    FS   = 1 << 2,
    SYS  = 1 << 3,

    All  = IO | NET | FS | SYS
};

inline const char* domainToCString(Domain d) {
    switch (d) {
        case Domain::Host: return "host";
        case Domain::CPU:  return "cpu";
        case Domain::GPU:  return "gpu";
    }
    return "unknown";
}

inline const char* spaceToCString(Space::Kind k) {
    switch (k) {
        case Space::RAM: return "host";
        case Space::GPU: return "gpu";
    }
    return "unknown";
}

inline std::string spaceToString(const Space& sp) {
    if (sp.kind == Space::RAM) {
        return "@host";
    }

    switch (sp.addressKind) {
        case Space::AddressKind::Default:
            return "@gpu";
        case Space::AddressKind::DeviceId:
            return "@gpu:" + std::to_string(sp.deviceId);
        case Space::AddressKind::RouteLiteral:
            return "@gpu:\"" + sp.address + "\"";
        case Space::AddressKind::RouteSymbol:
            return "@gpu:" + sp.address;
        case Space::AddressKind::RuntimePreset:
            return "@runtime " + sp.address;
    }

    return "@gpu";
}

// =============================================================================
// 2. Type System
// =============================================================================

struct Type {
    enum Kind {
        Void,

        // Primitive scalars
        I8, I16, I32, I64,
        U8, U16, U32, U64,
        F32, F64,
        Bool,
        Str,

        // Compound / callable
        Func,
        Ptr,

        // Containers
        Vec,
        Slice,
        Tensor,

        // User / generic / tuple
        Schema,
        Generic,
        Tuple
    } kind = Void;

    // Function type shape
    std::vector<Type> paramTypes;
    std::shared_ptr<Type> funcReturnType;

    // Generic arguments:
    //   vec<i32>      -> { i32 }
    //   tensor<f32>   -> { f32 }
    //   Box<i32, f32> -> { i32, f32 }
    std::vector<Type> genericArgs;

    // User-defined type name or generic base name
    std::string schemaName;

    // Optional shape metadata for tensors / shaped user constructs
    std::vector<std::string> shape;

    // Tuple element types
    std::vector<Type> subtypes;

    // Resolved / declared memory placement
    Space space;

    bool isScalar() const {
        return (kind >= I8 && kind <= Bool) || kind == Ptr;
    }

    bool isContainer() const {
        return kind == Vec || kind == Slice || kind == Tensor;
    }

    bool isInteger() const {
        return (kind >= I8 && kind <= I64) || (kind >= U8 && kind <= U64);
    }

    bool isFloat() const {
        return kind == F32 || kind == F64;
    }

    bool isSigned() const {
        return kind == I8 || kind == I16 || kind == I32 || kind == I64;
    }

    Type getInnerType() const {
        if (!genericArgs.empty()) return genericArgs[0];
        return {Void};
    }

    bool operator==(const Type& other) const {
        if (kind != other.kind) return false;
        if (space != other.space) return false;
        if (schemaName != other.schemaName) return false;
        if (shape != other.shape) return false;

        if (genericArgs.size() != other.genericArgs.size()) return false;
        for (std::size_t i = 0; i < genericArgs.size(); ++i) {
            if (genericArgs[i] != other.genericArgs[i]) return false;
        }

        if (subtypes.size() != other.subtypes.size()) return false;
        for (std::size_t i = 0; i < subtypes.size(); ++i) {
            if (subtypes[i] != other.subtypes[i]) return false;
        }

        if (paramTypes.size() != other.paramTypes.size()) return false;
        for (std::size_t i = 0; i < paramTypes.size(); ++i) {
            if (paramTypes[i] != other.paramTypes[i]) return false;
        }

        if (static_cast<bool>(funcReturnType) != static_cast<bool>(other.funcReturnType)) return false;
        if (funcReturnType && other.funcReturnType && *funcReturnType != *other.funcReturnType) return false;

        return true;
    }

    bool operator!=(const Type& other) const {
        return !(*this == other);
    }

    std::string toString() const {
        struct Printer {
            std::size_t maxList;
            std::string out;

            explicit Printer(std::size_t limit) : maxList(limit) {}

            static const char* kindName(Type::Kind k) {
                switch (k) {
                    case Type::Void:   return "void";
                    case Type::I8:     return "i8";
                    case Type::I16:    return "i16";
                    case Type::I32:    return "i32";
                    case Type::I64:    return "i64";
                    case Type::U8:     return "u8";
                    case Type::U16:    return "u16";
                    case Type::U32:    return "u32";
                    case Type::U64:    return "u64";
                    case Type::F32:    return "f32";
                    case Type::F64:    return "f64";
                    case Type::Bool:   return "bool";
                    case Type::Str:    return "str";
                    case Type::Func:   return "fn";
                    case Type::Ptr:    return "ptr";
                    case Type::Vec:    return "vec";
                    case Type::Slice:  return "slice";
                    case Type::Tensor: return "tensor";
                    case Type::Schema: return "schema";
                    case Type::Generic:return "generic";
                    case Type::Tuple:  return "tuple";
                }
                return "unknown";
            }

            void printSpace(const Space& sp) {
                out += spaceToString(sp);
            }

            void print(const Type& t) {
                switch (t.kind) {
                    case Type::Func: {
                        out += "fn(";
                        printTypeList(t.paramTypes, ", ");
                        out += ") -> ";
                        if (t.funcReturnType) print(*t.funcReturnType);
                        else out += "void";
                        printSpace(t.space);
                        return;
                    }

                    case Type::Ptr: {
                        out += "*";
                        if (!t.genericArgs.empty()) print(t.genericArgs[0]);
                        else out += "void";
                        printSpace(t.space);
                        return;
                    }

                    case Type::Vec:
                    case Type::Slice:
                    case Type::Tensor: {
                        out += kindName(t.kind);
                        printGenericArgs(t.genericArgs);
                        if (!t.shape.empty()) printShape(t.shape);
                        printSpace(t.space);
                        return;
                    }

                    case Type::Schema:
                    case Type::Generic: {
                        out += (!t.schemaName.empty() ? t.schemaName : kindName(t.kind));
                        if (!t.genericArgs.empty()) printGenericArgs(t.genericArgs);
                        if (!t.shape.empty()) printShape(t.shape);
                        printSpace(t.space);
                        return;
                    }

                    case Type::Tuple: {
                        out += "(";
                        printTypeList(t.subtypes, ", ");
                        out += ")";
                        printSpace(t.space);
                        return;
                    }

                    default: {
                        out += kindName(t.kind);
                        printSpace(t.space);
                        return;
                    }
                }
            }

            void printGenericArgs(const std::vector<Type>& args) {
                out += "<";
                if (args.empty()) {
                    out += "?";
                } else {
                    printTypeList(args, ", ");
                }
                out += ">";
            }

            void printShape(const std::vector<std::string>& dims) {
                out += "[";
                const std::size_t n = dims.size();
                const std::size_t take = (n > maxList) ? maxList : n;

                for (std::size_t i = 0; i < take; ++i) {
                    if (i) out += ", ";
                    out += dims[i].empty() ? "?" : dims[i];
                }

                if (take < n) {
                    out += ", ...(+";
                    out += std::to_string(n - take);
                    out += ")";
                }

                out += "]";
            }

            void printTypeList(const std::vector<Type>& ts, const char* sep) {
                const std::size_t n = ts.size();
                const std::size_t take = (n > maxList) ? maxList : n;

                for (std::size_t i = 0; i < take; ++i) {
                    if (i) out += sep;
                    print(ts[i]);
                }

                if (take < n) {
                    out += sep;
                    out += "...(+";
                    out += std::to_string(n - take);
                    out += ")";
                }
            }
        };

        Printer p(/*limit=*/12);
        p.print(*this);
        return p.out;
    }
};

// =============================================================================
// 3. AST Node Kinds
// =============================================================================

enum class ExprKind {
    // Basic values
    Symbol,
    Literal,
    String,
    Tuple,

    // Operators / access
    Binary,
    Unary,
    Index,

    // Calls / async / runtime
    Alloc,
    Launch,
    Call,
    Await,
    Import,
    RuntimeLiteral,

    // Control flow
    Block,
    If,
    Match,
    Return,
    Break,
    Continue,

    // Loops
    While,
    For,
    Iter,
    ParLoop,

    // Variables / side effects
    Let,
    Assign,
    Print,

    // Data construction / member ops
    ArrayLiteral,
    Range,
    SchemaExpr,
    MemberAccess,
    MemberCall,
    Lambda
};

// =============================================================================
// 4. Base AST Nodes
// =============================================================================

struct Expr {
    ExprKind kind;
    SourceLoc loc;

    virtual std::string toString() const {
        switch (kind) {
            case ExprKind::Symbol:         return "Symbol";
            case ExprKind::Literal:        return "Literal";
            case ExprKind::String:         return "String";
            case ExprKind::Tuple:          return "Tuple";
            case ExprKind::Binary:         return "Binary";
            case ExprKind::Unary:          return "Unary";
            case ExprKind::Index:          return "Index";
            case ExprKind::Alloc:          return "Alloc";
            case ExprKind::Launch:         return "Launch";
            case ExprKind::Call:           return "Call";
            case ExprKind::Await:          return "Await";
            case ExprKind::Import:         return "Import";
            case ExprKind::RuntimeLiteral: return "RuntimeLiteral";
            case ExprKind::Block:          return "Block";
            case ExprKind::If:             return "If";
            case ExprKind::Match:          return "Match";
            case ExprKind::Return:         return "Return";
            case ExprKind::Break:          return "Break";
            case ExprKind::Continue:       return "Continue";
            case ExprKind::While:          return "While";
            case ExprKind::For:            return "For";
            case ExprKind::Iter:           return "Iter";
            case ExprKind::ParLoop:        return "ParLoop";
            case ExprKind::Let:            return "Let";
            case ExprKind::Assign:         return "Assign";
            case ExprKind::Print:          return "Print";
            case ExprKind::ArrayLiteral:   return "ArrayLiteral";
            case ExprKind::Range:          return "Range";
            case ExprKind::SchemaExpr:     return "SchemaExpr";
            case ExprKind::MemberAccess:   return "MemberAccess";
            case ExprKind::MemberCall:     return "MemberCall";
            case ExprKind::Lambda:         return "Lambda";
        }
        return "Expr";
    }

    virtual ~Expr() = default;

protected:
    Expr(ExprKind k, SourceLoc l) : kind(k), loc(std::move(l)) {}
};

struct Decl {
    virtual ~Decl() = default;
};

// =============================================================================
// 5. Module / Import System
// =============================================================================

struct ImportDecl : public Expr, public Decl {
    std::string path;
    std::string alias;

    // Populated by the parser or import loader with the imported AST.
    std::unique_ptr<Module> importedModule;

    ImportDecl(SourceLoc l, std::string p, std::string a)
        : Expr(ExprKind::Import, l), path(std::move(p)), alias(std::move(a)) {}

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Import;
    }
};

// =============================================================================
// 6. Values & Primitive Expressions
// =============================================================================

struct SymbolExpr : public Expr {
    std::string name;

    SymbolExpr(SourceLoc l, std::string n)
        : Expr(ExprKind::Symbol, l), name(std::move(n)) {}

    std::string toString() const override {
        return name;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Symbol;
    }
};

struct LiteralExpr : public Expr {
    std::string value;
    Type type;

    LiteralExpr(SourceLoc l, std::string v, Type t)
        : Expr(ExprKind::Literal, l), value(std::move(v)), type(std::move(t)) {}

    std::string toString() const override {
        return value;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Literal;
    }
};

struct StringExpr : public Expr {
    std::string value;

    StringExpr(SourceLoc l, std::string v)
        : Expr(ExprKind::String, l), value(std::move(v)) {}

    std::string toString() const override {
        return "\"" + value + "\"";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::String;
    }
};

struct TupleExpr : public Expr {
    std::vector<std::unique_ptr<Expr>> elements;

    TupleExpr(SourceLoc l, std::vector<std::unique_ptr<Expr>> e)
        : Expr(ExprKind::Tuple, l), elements(std::move(e)) {}

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Tuple;
    }
};

// =============================================================================
// 7. Operators & Access
// =============================================================================

struct BinaryExpr : public Expr {
    std::string op;
    std::unique_ptr<Expr> lhs;
    std::unique_ptr<Expr> rhs;

    BinaryExpr(SourceLoc l, std::string o, std::unique_ptr<Expr> L, std::unique_ptr<Expr> R)
        : Expr(ExprKind::Binary, l), op(std::move(o)), lhs(std::move(L)), rhs(std::move(R)) {}

    std::string toString() const override {
        return "Binary(" + op + ")";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Binary;
    }
};

struct UnaryExpr : public Expr {
    std::string op;
    std::unique_ptr<Expr> operand;

    UnaryExpr(SourceLoc l, std::string o, std::unique_ptr<Expr> expr)
        : Expr(ExprKind::Unary, l), op(std::move(o)), operand(std::move(expr)) {}

    std::string toString() const override {
        return "Unary(" + op + ")";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Unary;
    }
};

struct IndexExpr : public Expr {
    std::unique_ptr<Expr> base;
    std::unique_ptr<Expr> index;

    // Filled during typechecking / lowering for downstream consumers.
    Type resolvedType = { Type::Void };

    IndexExpr(SourceLoc l, std::unique_ptr<Expr> b, std::unique_ptr<Expr> idx)
        : Expr(ExprKind::Index, l), base(std::move(b)), index(std::move(idx)) {}

    std::string toString() const override {
        return (base ? base->toString() : std::string("<null>")) + "[...]";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Index;
    }
};

// =============================================================================
// 8. Calls
// =============================================================================

struct CallArg {
    std::string name;
    std::unique_ptr<Expr> value;

    CallArg(std::string n, std::unique_ptr<Expr> v)
        : name(std::move(n)), value(std::move(v)) {}
};

struct CallExpr : public Expr {
    std::unique_ptr<Expr> callee;
    std::vector<CallArg> args;

    // Explicit generic arguments:
    //   foo<T, U>(...)
    std::vector<Type> explicitGenericArgs;

    CallExpr(SourceLoc loc,
             std::unique_ptr<Expr> calleeExpr,
             std::vector<CallArg> a,
             std::vector<Type> genericArgs = {})
        : Expr(ExprKind::Call, loc),
          callee(std::move(calleeExpr)),
          args(std::move(a)),
          explicitGenericArgs(std::move(genericArgs)) {}

    std::string toString() const override {
        return "Call";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Call;
    }
};

// =============================================================================
// 9. Runtime Literals, Allocation & Async
// =============================================================================

struct RuntimeFieldInit {
    std::string name;
    std::unique_ptr<Expr> value;

    RuntimeFieldInit(std::string n, std::unique_ptr<Expr> v)
        : name(std::move(n)), value(std::move(v)) {}
};

struct RuntimeLiteralExpr : public Expr {
    std::vector<RuntimeFieldInit> fields;

    RuntimeLiteralExpr(SourceLoc l, std::vector<RuntimeFieldInit> f)
        : Expr(ExprKind::RuntimeLiteral, l), fields(std::move(f)) {}

    std::string toString() const override {
        return "runtime{...}";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::RuntimeLiteral;
    }
};

struct AllocExpr : public Expr {
    Type type;

    // Legacy textual spelling preserved for compatibility with existing passes.
    // Examples:
    //   ""
    //   "@gpu:0"
    //   "@gpu:\"route-id\""
    //   "@gpu:route"
    //   "@runtime preset"
    std::string location;

    // Structured placement form for new lowering paths.
    std::optional<Space> placement;

    AllocExpr(SourceLoc l, Type t, std::string locStr = "")
        : Expr(ExprKind::Alloc, l), type(std::move(t)), location(std::move(locStr)) {}

    AllocExpr(SourceLoc l, Type t, Space sp)
        : Expr(ExprKind::Alloc, l),
          type(std::move(t)),
          location(spaceToString(sp)),
          placement(std::move(sp)) {}

    std::string toString() const override {
        return location.empty() ? "alloc" : "alloc " + location;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Alloc;
    }
};

struct LaunchExpr : public Expr {
    std::string destVar;
    std::string kernelName;
    std::vector<CallArg> args;
    std::string tokenName;

    // Optional runtime routing annotation.
    // For current grammar this is typically:
    //   @runtime preset
    std::string runtimeName;
    std::optional<Space> runtime;

    LaunchExpr(SourceLoc l,
               std::string d,
               std::string k,
               std::vector<CallArg> a,
               std::string t,
               std::string rt = "")
        : Expr(ExprKind::Launch, l),
          destVar(std::move(d)),
          kernelName(std::move(k)),
          args(std::move(a)),
          tokenName(std::move(t)),
          runtimeName(std::move(rt)) {}

    LaunchExpr(SourceLoc l,
               std::string d,
               std::string k,
               std::vector<CallArg> a,
               std::string t,
               Space rt)
        : Expr(ExprKind::Launch, l),
          destVar(std::move(d)),
          kernelName(std::move(k)),
          args(std::move(a)),
          tokenName(std::move(t)),
          runtimeName(rt.addressKind == Space::AddressKind::RuntimePreset ? rt.address : std::string()),
          runtime(std::move(rt)) {}

    std::string toString() const override {
        std::string s = "launch " + kernelName + " -> " + destVar;
        if (!tokenName.empty()) s += " as " + tokenName;
        if (!runtimeName.empty()) s += " @runtime " + runtimeName;
        return s;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Launch;
    }
};

struct AwaitExpr : public Expr {
    std::string tokenName;

    AwaitExpr(SourceLoc l, std::string t)
        : Expr(ExprKind::Await, l), tokenName(std::move(t)) {}

    std::string toString() const override {
        return "await " + tokenName;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Await;
    }
};

// =============================================================================
// 10. Statements & Control Flow
// =============================================================================

struct BlockExpr : public Expr {
    std::vector<std::unique_ptr<Expr>> stmts;

    BlockExpr(SourceLoc l, std::vector<std::unique_ptr<Expr>> s)
        : Expr(ExprKind::Block, l), stmts(std::move(s)) {}

    std::string toString() const override {
        return "Block";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Block;
    }
};

struct VarDecl : public Expr {
    std::vector<std::string> names;
    std::optional<Type> annotation;
    std::unique_ptr<Expr> init;

    VarDecl(SourceLoc l,
            std::vector<std::string> n,
            std::optional<Type> ann,
            std::unique_ptr<Expr> i)
        : Expr(ExprKind::Let, l),
          names(std::move(n)),
          annotation(std::move(ann)),
          init(std::move(i)) {}

    std::string toString() const override {
        if (names.empty()) return "let (empty)";
        if (names.size() == 1) return "let " + names[0];
        return "let (" + names[0] + ", ...)";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Let;
    }
};

struct LambdaExpr : public Expr {
    struct Param {
        std::string name;
        Type type;
    };

    std::vector<Param> params;
    Type returnType;
    std::unique_ptr<Expr> body;
    bool explicitReturn = false;

    LambdaExpr(SourceLoc l,
               std::vector<Param> p,
               Type rt,
               std::unique_ptr<Expr> b,
               bool expl)
        : Expr(ExprKind::Lambda, l),
          params(std::move(p)),
          returnType(std::move(rt)),
          body(std::move(b)),
          explicitReturn(expl) {}

    std::string toString() const override {
        return "lambda";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Lambda;
    }
};

struct AssignStmt : public Expr {
    std::unique_ptr<Expr> target;
    std::unique_ptr<Expr> value;

    AssignStmt(SourceLoc l, std::unique_ptr<Expr> t, std::unique_ptr<Expr> v)
        : Expr(ExprKind::Assign, l), target(std::move(t)), value(std::move(v)) {}

    std::string toString() const override {
        return (target ? target->toString() : std::string("<null>")) + " = ...";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Assign;
    }
};

struct ReturnStmt : public Expr {
    std::unique_ptr<Expr> value;

    ReturnStmt(SourceLoc l, std::unique_ptr<Expr> v)
        : Expr(ExprKind::Return, l), value(std::move(v)) {}

    std::string toString() const override {
        return "return";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Return;
    }
};

struct BreakStmt : public Expr {
    BreakStmt(SourceLoc l)
        : Expr(ExprKind::Break, l) {}

    std::string toString() const override {
        return "break";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Break;
    }
};

struct ContinueStmt : public Expr {
    ContinueStmt(SourceLoc l)
        : Expr(ExprKind::Continue, l) {}

    std::string toString() const override {
        return "continue";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Continue;
    }
};

struct IfStmt : public Expr {
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> thenBranch;
    std::unique_ptr<Expr> elseBranch;

    IfStmt(SourceLoc l, std::unique_ptr<Expr> c, std::unique_ptr<Expr> t, std::unique_ptr<Expr> e)
        : Expr(ExprKind::If, l),
          condition(std::move(c)),
          thenBranch(std::move(t)),
          elseBranch(std::move(e)) {}

    std::string toString() const override {
        return "if";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::If;
    }
};

// =============================================================================
// 11. Loops
// =============================================================================

struct WhileStmt : public Expr {
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> body;

    WhileStmt(SourceLoc l, std::unique_ptr<Expr> c, std::unique_ptr<Expr> b)
        : Expr(ExprKind::While, l), condition(std::move(c)), body(std::move(b)) {}

    std::string toString() const override {
        return "while";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::While;
    }
};

struct ForStmt : public Expr {
    std::string iterVar;
    std::unique_ptr<Expr> start;
    std::unique_ptr<Expr> end;
    std::unique_ptr<Expr> body;

    ForStmt(SourceLoc l,
            std::string iv,
            std::unique_ptr<Expr> s,
            std::unique_ptr<Expr> e,
            std::unique_ptr<Expr> b)
        : Expr(ExprKind::For, l),
          iterVar(std::move(iv)),
          start(std::move(s)),
          end(std::move(e)),
          body(std::move(b)) {}

    std::string toString() const override {
        return "for " + iterVar;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::For;
    }
};

struct IterStmt : public Expr {
    std::string iterVar;
    std::unique_ptr<Expr> collection;
    std::unique_ptr<Expr> body;

    IterStmt(SourceLoc l, std::string iv, std::unique_ptr<Expr> col, std::unique_ptr<Expr> b)
        : Expr(ExprKind::Iter, l),
          iterVar(std::move(iv)),
          collection(std::move(col)),
          body(std::move(b)) {}

    std::string toString() const override {
        return "iter " + iterVar;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Iter;
    }
};

struct ParLoop : public Expr {
    std::vector<std::string> iterVars;

    enum class DomainKind : uint8_t {
        Range,
        LenSugar,
        DimsCall
    };

    struct Domain {
        DomainKind kind;
        std::unique_ptr<Expr> expr;

        Domain(DomainKind k, std::unique_ptr<Expr> e)
            : kind(k), expr(std::move(e)) {}
    };

    Domain domain;

    // Launch hints from block(...)
    std::vector<std::unique_ptr<Expr>> blockDims;

    // Body is always a block for structured lowering.
    std::unique_ptr<BlockExpr> body;

    ParLoop(SourceLoc l,
            std::vector<std::string> vars,
            Domain dom,
            std::vector<std::unique_ptr<Expr>> hints,
            std::unique_ptr<BlockExpr> b)
        : Expr(ExprKind::ParLoop, l),
          iterVars(std::move(vars)),
          domain(std::move(dom)),
          blockDims(std::move(hints)),
          body(std::move(b)) {}

    std::string toString() const override {
        return "par";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::ParLoop;
    }
};

// =============================================================================
// 12. Data Construction & Schemas
// =============================================================================

struct ArrayLiteral : public Expr {
    std::vector<std::unique_ptr<Expr>> elements;

    ArrayLiteral(SourceLoc l, std::vector<std::unique_ptr<Expr>> elms)
        : Expr(ExprKind::ArrayLiteral, l), elements(std::move(elms)) {}

    std::string toString() const override {
        return "[...]";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::ArrayLiteral;
    }
};

struct RangeExpr : public Expr {
    std::unique_ptr<Expr> start;
    std::unique_ptr<Expr> end;

    RangeExpr(SourceLoc l, std::unique_ptr<Expr> s, std::unique_ptr<Expr> e)
        : Expr(ExprKind::Range, l), start(std::move(s)), end(std::move(e)) {}

    std::string toString() const override {
        return "start..end";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Range;
    }
};

struct RecordField {
    std::string name;
    Type type;

    // Optional default value, used especially by singleton-style schemas.
    std::unique_ptr<Expr> defaultValue;
};

struct EnumVariant {
    std::string name;

    enum Kind {
        Unit,
        Tuple,
        Struct
    } kind = Unit;

    std::vector<Type> tuplePayload;
    std::vector<RecordField> structPayload;
};

struct SchemaDecl : public Decl {
    enum Kind {
        Record,
        Enum
    };

    Kind kind;
    std::string name;
    bool hasMeta = false;
    bool isSingleton = false;
    std::vector<std::string> genericParams;
    std::vector<RecordField> fields;
    std::vector<EnumVariant> variants;
    SourceLoc loc;

    SchemaDecl(std::string n, std::vector<RecordField> f, SourceLoc l, bool meta)
        : kind(Record),
          name(std::move(n)),
          hasMeta(meta),
          fields(std::move(f)),
          loc(std::move(l)) {}

    SchemaDecl(std::string n, std::vector<EnumVariant> v, SourceLoc l, bool meta)
        : kind(Enum),
          name(std::move(n)),
          hasMeta(meta),
          variants(std::move(v)),
          loc(std::move(l)) {}
};

struct SchemaInitField {
    std::string name;
    std::unique_ptr<Expr> value;
};

struct SchemaExpr : public Expr {
    std::string name;
    std::vector<Type> genericArgs;
    std::vector<SchemaInitField> fields;

    SchemaExpr(SourceLoc l, std::string n, std::vector<Type> args, std::vector<SchemaInitField> f)
        : Expr(ExprKind::SchemaExpr, l),
          name(std::move(n)),
          genericArgs(std::move(args)),
          fields(std::move(f)) {}

    std::string toString() const override {
        return name + "{...}";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::SchemaExpr;
    }
};

struct MemberExpr : public Expr {
    std::unique_ptr<Expr> object;
    std::string member;

    MemberExpr(SourceLoc l, std::unique_ptr<Expr> obj, std::string mem)
        : Expr(ExprKind::MemberAccess, l),
          object(std::move(obj)),
          member(std::move(mem)) {}

    std::string toString() const override {
        return (object ? object->toString() : std::string("<null>")) + "." + member;
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::MemberAccess;
    }
};

struct MemberCallNode : public Expr {
    std::unique_ptr<Expr> object;
    std::string methodName;
    std::vector<CallArg> args;

    MemberCallNode(SourceLoc l, std::unique_ptr<Expr> obj, std::string method, std::vector<CallArg> a)
        : Expr(ExprKind::MemberCall, l),
          object(std::move(obj)),
          methodName(std::move(method)),
          args(std::move(a)) {}

    std::string toString() const override {
        return (object ? object->toString() : std::string("<null>")) + "." + methodName + "(...)";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::MemberCall;
    }
};

// =============================================================================
// 13. Pattern Matching
// =============================================================================

struct Pattern {
    std::string schemaName;
    std::string variantName;
    std::vector<std::string> bindings;
    bool isDefault = false;
};

struct Case {
    Pattern pattern;
    std::unique_ptr<Expr> body;
};

struct MatchStmt : public Expr {
    std::unique_ptr<Expr> target;
    std::vector<Case> cases;

    MatchStmt(SourceLoc l, std::unique_ptr<Expr> t, std::vector<Case> c)
        : Expr(ExprKind::Match, l), target(std::move(t)), cases(std::move(c)) {}

    std::string toString() const override {
        return "match";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Match;
    }
};

struct PrintStmt : public Expr {
    std::vector<std::unique_ptr<Expr>> values;

    PrintStmt(SourceLoc l, std::vector<std::unique_ptr<Expr>> v)
        : Expr(ExprKind::Print, l), values(std::move(v)) {}

    std::string toString() const override {
        return "print";
    }

    static bool classof(const Expr* e) {
        return e->kind == ExprKind::Print;
    }
};

// =============================================================================
// 14. Top-Level Structures
// =============================================================================

struct Function : public Decl {
    std::string name;
    Domain domain = Domain::Host;
    uint32_t effects = 0;
    SourceLoc loc;

    // Ordered parameter list: (name, type)
    std::vector<std::pair<std::string, Type>> args;

    Type returnType;
    std::vector<std::unique_ptr<Expr>> body;
};

struct Module {
    // Stable identifier, typically derived from file stem.
    std::string id;

    std::vector<std::unique_ptr<ImportDecl>> imports;
    std::vector<std::unique_ptr<Function>> functions;
    std::vector<std::unique_ptr<SchemaDecl>> schemas;

    // Non-owning submodule links used by later passes for qualified resolution.
    llvm::StringMap<Module*> submodules;
};

} // namespace arklang