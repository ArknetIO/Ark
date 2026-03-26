#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <optional>
#include <variant>
#include "llvm/ADT/StringMap.h" 

namespace arklang {



static constexpr uint32_t EFF_FS  = 1u << 0;
static constexpr uint32_t EFF_NET = 1u << 1;
static constexpr uint32_t EFF_IO  = 1u << 2;
static constexpr uint32_t EFF_SYS = 1u << 3;


// [CRITICAL] Forward Declaration
// Allows ImportDecl to hold a unique_ptr<Module> before Module is fully defined.
struct Module;

// =============================================================================
// 1. Source Location & Execution Context
// =============================================================================

/**
 * @brief Tracks origin in source file for precise error reporting.
 */
struct SourceLoc {
    std::string file;
    int line = 0;
    int col = 0;
};

/**
 * @brief Execution Domain: Where code runs (Host CPU vs Device GPU).
 */
enum class Domain { Host, CPU, GPU };

/**
 * @brief Memory Space: Physical location of data.
 */
struct Space {
    enum Kind { RAM, GPU } kind = RAM;
    int deviceId = 0;
};

/**
 * @brief The "Trinity" Security Model.
 * Functions must explicitly declare these effects (e.g., `fn main() !IO`) to perform sensitive operations.
 */
enum Effect : uint32_t {
    None = 0,
    IO   = 1 << 0,  // !IO  -> Raw Hardware/FDs (Console, etc.)
    NET  = 1 << 1,  // !NET -> Network Sockets
    FS   = 1 << 2,  // !FS  -> Filesystem Access
    
    All  = IO | NET | FS
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
    if (sp.kind == Space::GPU) {
        std::string s = "@gpu:";
        s += std::to_string(sp.deviceId);
        return s;
    }
    return "@host";
}


// =============================================================================
// 2. Type System
// =============================================================================

struct Type {
    enum Kind { 
        Void, 
        // Primitives
        I8, I16, I32, I64, 
        U8, U16, U32, U64, 
        F32, F64, 
        Bool, 
        Str, 
        Func, // fn(A, B) -> R
        Ptr,
        // Containers (The Holy Trinity)
        Vec,    // vec<T>   -> genericArgs[0] is T
        Slice,  // slice<T> -> genericArgs[0] is T
        Tensor, // tensor<T>
        
        // Compound
        Schema,    // Concrete User-defined (Struct/Enum) e.g. Point
        Generic,   // Generic Instantiation (e.g. Box<i32>)
        Tuple      // Anonymous product: (i32, f32)

    } kind = Void; 

    // Function Types
    std::vector<Type> paramTypes;
    std::shared_ptr<Type> funcReturnType;
    
    // [FIX] Unified Generics System
    // Used by: Vec, Slice, Tensor, Generic, Schema
    // Example: vec<i32> stores {Type::I32} here.
    std::vector<Type> genericArgs; 
    
    // Identity
    std::string schemaName; 

    // Tensor Shape (Literal)
    std::vector<std::string> shape; 
    
    // Tuple Subtypes
    std::vector<Type> subtypes; 

    // Memory location (RAM/GPU)
    Space space; 
    
    // --- Helpers ---

    bool isScalar() const { 
        // [FIX] Pointers are scalars (hold a single memory address)
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

    // [ADD THIS HELPER]
    bool isSigned() const {
        return kind == I8 || kind == I16 || kind == I32 || kind == I64;
    }
    
    // [NEW] Helper to get inner type safely
    // Returns Void type if not applicable
    Type getInnerType() const {
        if (!genericArgs.empty()) return genericArgs[0];
        return {Void};
    }
    
    // Equality Check (Deep comparison)
    bool operator==(const Type& other) const {
        if (kind != other.kind) return false;
        
        // Deep Check for Containers & Generics
        if (isContainer() || kind == Schema || kind == Generic) {
            if (kind == Schema || kind == Generic) {
                if (schemaName != other.schemaName) return false;
            }
            // Recursive check for T inside vec<T>
            if (genericArgs.size() != other.genericArgs.size()) return false;
            for (size_t i = 0; i < genericArgs.size(); ++i) {
                if (genericArgs[i] != other.genericArgs[i]) return false;
            }
            return true;
        }

        if (kind == Tuple) {
            if (subtypes.size() != other.subtypes.size()) return false;
            for (size_t i = 0; i < subtypes.size(); ++i) {
                if (subtypes[i] != other.subtypes[i]) return false;
            }
            return true;
        }
        
        // For Functions
        if (kind == Func) {
             if (paramTypes.size() != other.paramTypes.size()) return false;
             for(size_t i=0; i<paramTypes.size(); ++i) 
                 if(paramTypes[i] != other.paramTypes[i]) return false;
             // Compare return types (handle nulls safely)
             if ((funcReturnType && !other.funcReturnType) || (!funcReturnType && other.funcReturnType)) return false;
             if (funcReturnType && *funcReturnType != *other.funcReturnType) return false;
             return true;
        }

        return true;
    }

    bool operator!=(const Type& other) const { return !(*this == other); }

    // inside struct Type

    std::string toString() const {
        struct Printer {
            size_t maxList;
            std::string out;

            explicit Printer(size_t limit) : maxList(limit) {}

            static const char* kindName(Kind k) {
                switch (k) {
                    case Void:   return "void";
                    case I8:     return "i8";
                    case I16:    return "i16";
                    case I32:    return "i32";
                    case I64:    return "i64";
                    case U8:     return "u8";
                    case U16:    return "u16";
                    case U32:    return "u32";
                    case U64:    return "u64";
                    case F32:    return "f32";
                    case F64:    return "f64";
                    case Bool:   return "bool";
                    case Str:    return "str";
                    case Func:   return "fn";
                    case Ptr:    return "ptr";
                    case Vec:    return "vec";
                    case Slice:  return "slice";
                    case Tensor: return "tensor";
                    case Schema: return "schema";
                    case Generic:return "generic";
                    case Tuple:  return "tuple";
                }
                return "unknown";
            }

            void printSpace(const Space& sp) {
                if (sp.kind == Space::GPU) {
                    out += "@gpu:";
                    out += std::to_string(sp.deviceId);
                } else {
                    out += "@host";
                }
            }

            void print(const Type& t) {
                switch (t.kind) {
                    case Func: {
                        out += "fn(";
                        printTypeList(t.paramTypes, ", ");
                        out += ") -> ";
                        if (t.funcReturnType) print(*t.funcReturnType);
                        else out += "void";
                        printSpace(t.space);
                        return;
                    }
                    case Ptr: {
                        out += "*";
                        if (!t.genericArgs.empty()) print(t.genericArgs[0]);
                        else out += "void";
                        printSpace(t.space);
                        return;
                    }
                    case Vec:
                    case Slice:
                    case Tensor: {
                        out += kindName(t.kind);
                        printGenericArgs(t.genericArgs);
                        printSpace(t.space);
                        return;
                    }
                    case Schema:
                    case Generic: {
                        out += (!t.schemaName.empty() ? t.schemaName : kindName(t.kind));
                        if (!t.genericArgs.empty()) printGenericArgs(t.genericArgs);
                        if (!t.shape.empty()) printShape(t.shape);
                        printSpace(t.space);
                        return;
                    }
                    case Tuple: {
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
                const size_t n = dims.size();
                const size_t take = (n > maxList) ? maxList : n;

                for (size_t i = 0; i < take; ++i) {
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
                const size_t n = ts.size();
                const size_t take = (n > maxList) ? maxList : n;

                for (size_t i = 0; i < take; ++i) {
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
// 3. AST Node Definitions
// =============================================================================
enum class ExprKind {
    // Basic
    Symbol, Literal, String, Tuple,
    
    // Operations
    Binary, Unary, Index,
    
    // Memory/Async
    Alloc, Launch, Call, Await, Import,
    
    // Control Flow
    Block, If, Match, Return,
    
    // Loops
    While, For, Iter, ParLoop,
    
    // Variables & IO
    Let, Assign, Print,
    
    // Data Construction
    ArrayLiteral, Range,
    SchemaExpr,   // Point { x: 1 }
    MemberAccess, // p.x
    MemberCall,    // vec.push(1)
    Lambda
};

// Base AST Node
struct Expr { 
    ExprKind kind;
    SourceLoc loc;
    
    // [ADD THIS] Virtual toString method
    virtual std::string toString() const {
        switch (kind) {
            case ExprKind::Symbol:   return "Symbol";
            case ExprKind::Literal:  return "Literal";
            case ExprKind::Call:     return "Call";
            case ExprKind::Alloc:    return "alloc";
            case ExprKind::MemberAccess: return "MemberAccess";
            case ExprKind::Import:   return "Import";
            case ExprKind::String:   return "String";
            case ExprKind::Tuple:    return "Tuple";
            default: return "Expr";
        }
    }
    virtual ~Expr() = default; 
    
    protected: Expr(ExprKind k, SourceLoc l) : kind(k), loc(std::move(l)) {}
};

// [NEW] Base Declaration Node
// Allows parsing Functions, Schemas, and Imports uniformly.
struct Decl {
    virtual ~Decl() = default;
};

// -----------------------------------------------------------------------------
// Module System
// -----------------------------------------------------------------------------

struct ImportDecl : public Expr, public Decl {
    std::string path;  // "math/vectors.ark"
    std::string alias; // "v"

    // [CRITICAL ADDITION] 
    // The Parser populates this with the AST of the imported file.
    // ArkCodeGen reads this to resolve types like 'v.Vector'.
    std::unique_ptr<Module> importedModule;

    ImportDecl(SourceLoc l, std::string p, std::string a) 
        : Expr(ExprKind::Import, l), path(std::move(p)), alias(std::move(a)) {}

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Import; }
};

// -----------------------------------------------------------------------------
// Values & Variables
// -----------------------------------------------------------------------------

struct SymbolExpr : public Expr {
    std::string name;
    SymbolExpr(SourceLoc l, std::string n) : Expr(ExprKind::Symbol, l), name(std::move(n)) {}
    
    std::string toString() const override { return name; }
    
    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Symbol; }
};

struct LiteralExpr : public Expr {
    std::string value; // Stored as string, parsed by CodeGen
    Type type;
    LiteralExpr(SourceLoc l, std::string v, Type t) 
        : Expr(ExprKind::Literal, l), value(std::move(v)), type(std::move(t)) {}
    
    std::string toString() const override { return value; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Literal; }
};

struct StringExpr : public Expr {
    std::string value;
    StringExpr(SourceLoc l, std::string v) 
        : Expr(ExprKind::String, l), value(std::move(v)) {}
    
    std::string toString() const override { return "\"" + value + "\""; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::String; }
};

struct TupleExpr : public Expr {
    std::vector<std::unique_ptr<Expr>> elements;
    TupleExpr(SourceLoc l, std::vector<std::unique_ptr<Expr>> e)
        : Expr(ExprKind::Tuple, l), elements(std::move(e)) {}

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Tuple; }
};
// -----------------------------------------------------------------------------
// Operations
// -----------------------------------------------------------------------------

struct BinaryExpr : public Expr {
    std::string op; 
    std::unique_ptr<Expr> lhs, rhs;
    BinaryExpr(SourceLoc l, std::string o, std::unique_ptr<Expr> L, std::unique_ptr<Expr> R) 
        : Expr(ExprKind::Binary, l), op(std::move(o)), lhs(std::move(L)), rhs(std::move(R)) {}

    std::string toString() const override { return "Binary(" + op + ")"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Binary; }
};

struct IndexExpr : public Expr {
    std::unique_ptr<Expr> base;
    std::unique_ptr<Expr> index;

    // Filled by TypeChecker/CodeGen (for method calls on arrays)
    Type resolvedType = { Type::Void };

    IndexExpr(SourceLoc l, std::unique_ptr<Expr> b, std::unique_ptr<Expr> idx)
        : Expr(ExprKind::Index, l), base(std::move(b)), index(std::move(idx)) {}

    std::string toString() const override {
        return (base ? base->toString() : std::string("<null>")) + "[...]";
    }

    static bool classof(const Expr *e) { return e->kind == ExprKind::Index; }
};

// -----------------------------------------------------------------------------
// Functions & Calls
// -----------------------------------------------------------------------------

struct CallArg {
    std::string name; // "argName" or "" if positional
    std::unique_ptr<Expr> value;
    
    CallArg(std::string n, std::unique_ptr<Expr> v) 
        : name(std::move(n)), value(std::move(v)) {}
};

struct CallExpr : public Expr {
    std::unique_ptr<Expr> callee;
    std::vector<CallArg> args;
    
    // Storage for generics like foo<T>(...)
    std::vector<Type> explicitGenericArgs; 

    CallExpr(SourceLoc loc, 
             std::unique_ptr<Expr> callee, 
             std::vector<CallArg> args,
             std::vector<Type> genericArgs = {}) // Defaults to empty
        : Expr(ExprKind::Call, loc), 
          callee(std::move(callee)), 
          args(std::move(args)), 
          explicitGenericArgs(std::move(genericArgs)) {}

    std::string toString() const override { return "Call"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Call; }
};

// -----------------------------------------------------------------------------
// Memory & Async
// -----------------------------------------------------------------------------

struct AllocExpr : public Expr {
    Type type;
    std::string location; // "gpu:0" (Optional)

    AllocExpr(SourceLoc l, Type t, std::string locStr = "") 
        : Expr(ExprKind::Alloc, l), type(std::move(t)), location(std::move(locStr)) {}
    
    std::string toString() const override { return "alloc"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Alloc; }
};

struct LaunchExpr : public Expr {
    std::string destVar;   // Result variable
    std::string kernelName;
    std::vector<CallArg> args; 
    std::string tokenName; // Optional async token handle
    
    LaunchExpr(SourceLoc l, std::string d, std::string k, std::vector<CallArg> a, std::string t) 
        : Expr(ExprKind::Launch, l), destVar(std::move(d)), kernelName(std::move(k)), args(std::move(a)), tokenName(std::move(t)) {}

    std::string toString() const override { 
        return "launch " + kernelName + " -> " + destVar; 
    }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Launch; }
};

struct AwaitExpr : public Expr {
    std::string tokenName;
    AwaitExpr(SourceLoc l, std::string t) : Expr(ExprKind::Await, l), tokenName(std::move(t)) {}

    std::string toString() const override { 
        return "await " + tokenName; 
    }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Await; }
};

// -------------// -----------------------------------------------------------------------------
// Statements & Control Flow
// -----------------------------------------------------------------------------

struct BlockExpr : public Expr {
    std::vector<std::unique_ptr<Expr>> stmts;
    BlockExpr(SourceLoc l, std::vector<std::unique_ptr<Expr>> s)
        : Expr(ExprKind::Block, l), stmts(std::move(s)) {}

    std::string toString() const override { return "Block"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Block; }
};

struct VarDecl : public Expr {
    std::vector<std::string> names; // Supports destructuring: let (a, b) = ...
    std::optional<Type> annotation;
    std::unique_ptr<Expr> init;
    
    VarDecl(SourceLoc l, std::vector<std::string> n, std::optional<Type> ann, std::unique_ptr<Expr> i)
        : Expr(ExprKind::Let, l), names(std::move(n)), annotation(std::move(ann)), init(std::move(i)) {}

    std::string toString() const override { 
        if (names.empty()) return "let (empty)";
        if (names.size() == 1) return "let " + names[0];
        return "let (" + names[0] + ", ...)"; 
    }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Let; }
};

struct LambdaExpr : public Expr {
    struct Param {
        std::string name;
        Type type;
    };
    
    std::vector<Param> params;
    Type returnType;            // Can be Void if inferred/omitted
    std::unique_ptr<Expr> body; // BlockExpr or generic Expr
    bool explicitReturn;        // True if user wrote "-> T"

    LambdaExpr(SourceLoc l, std::vector<Param> p, Type rt, std::unique_ptr<Expr> b, bool expl)
        : Expr(ExprKind::Lambda, l), 
          params(std::move(p)), 
          returnType(std::move(rt)), 
          body(std::move(b)), 
          explicitReturn(expl) {}

    std::string toString() const override { return "lambda"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Lambda; }
};

struct AssignStmt : public Expr {
    std::unique_ptr<Expr> target; // L-Value (Symbol, Index, Member)
    std::unique_ptr<Expr> value;  // R-Value
    
    AssignStmt(SourceLoc l, std::unique_ptr<Expr> t, std::unique_ptr<Expr> v)
        : Expr(ExprKind::Assign, l), target(std::move(t)), value(std::move(v)) {}
    
    std::string toString() const override { 
        return target->toString() + " = ..."; 
    }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Assign; }
};

// [FIX] Renamed ReturnExpr -> ReturnStmt to match GenMIR usage
struct ReturnStmt : public Expr {
    std::unique_ptr<Expr> value; // null for void return
    ReturnStmt(SourceLoc l, std::unique_ptr<Expr> v)
        : Expr(ExprKind::Return, l), value(std::move(v)) {}

    std::string toString() const override { return "return"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Return; }
};

struct IfStmt : public Expr {
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> thenBranch;
    std::unique_ptr<Expr> elseBranch; // Can be IfStmt (else if) or Block (else)

    IfStmt(SourceLoc l, std::unique_ptr<Expr> c, std::unique_ptr<Expr> t, std::unique_ptr<Expr> e)
        : Expr(ExprKind::If, l), condition(std::move(c)), thenBranch(std::move(t)), elseBranch(std::move(e)) {}

    std::string toString() const override { return "if"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::If; }
};

// -----------------------------------------------------------------------------
// Loops
// -----------------------------------------------------------------------------

struct WhileStmt : public Expr {
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> body;
    
    WhileStmt(SourceLoc l, std::unique_ptr<Expr> c, std::unique_ptr<Expr> b)
        : Expr(ExprKind::While, l), condition(std::move(c)), body(std::move(b)) {}

    std::string toString() const override { return "while"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::While; }
};

struct ForStmt : public Expr {
    std::string iterVar;
    std::unique_ptr<Expr> start;
    std::unique_ptr<Expr> end;
    std::unique_ptr<Expr> body;

    ForStmt(SourceLoc l, std::string iv, std::unique_ptr<Expr> s, std::unique_ptr<Expr> e, std::unique_ptr<Expr> b)
        : Expr(ExprKind::For, l), iterVar(std::move(iv)), start(std::move(s)), end(std::move(e)), body(std::move(b)) {}

    std::string toString() const override { return "for " + iterVar; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::For; }
};

struct IterStmt : public Expr {
    std::string iterVar;
    std::unique_ptr<Expr> collection; // Vector or Tensor
    std::unique_ptr<Expr> body;

    IterStmt(SourceLoc l, std::string iv, std::unique_ptr<Expr> col, std::unique_ptr<Expr> b)
        : Expr(ExprKind::Iter, l), iterVar(std::move(iv)), collection(std::move(col)), body(std::move(b)) {}

    std::string toString() const override { return "iter " + iterVar; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Iter; }
};

struct ParLoop : public Expr {
    // Supports: par i in ...  OR par (x, y) in ...
    std::vector<std::string> iterVars;

    // Domain classification.
    // Tags the intent at parse time so GenMIR doesn't have to guess.
    enum class DomainKind : uint8_t {
        Range,     // RangeExpr: 0..N
        LenSugar,  // Expr: C / member / index -> sugar for 0..len(expr)
        DimsCall   // CallExpr: dims(out)
    };

    struct Domain {
        DomainKind kind;
        std::unique_ptr<Expr> expr; // The actual expression tree

        Domain(DomainKind k, std::unique_ptr<Expr> e)
            : kind(k), expr(std::move(e)) {}
    };

    Domain domain;

    // Launch hints from block(...)
    // Parser enforces size is 0, 1, 2, or 3.
    std::vector<std::unique_ptr<Expr>> blockDims;

    // Enforced strictly as a BlockExpr ( { ... } )
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

    std::string toString() const override { return "par"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::ParLoop; }
};
// -----------------------------------------------------------------------------
// Schemas & Data Structures
// -----------------------------------------------------------------------------

struct ArrayLiteral : public Expr {
    std::vector<std::unique_ptr<Expr>> elements;
    ArrayLiteral(SourceLoc l, std::vector<std::unique_ptr<Expr>> elms)
        : Expr(ExprKind::ArrayLiteral, l), elements(std::move(elms)) {}

    std::string toString() const override { return "[...]"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::ArrayLiteral; }
};

struct RangeExpr : public Expr {
    std::unique_ptr<Expr> start; // null = 0
    std::unique_ptr<Expr> end;   // null = len
    
    RangeExpr(SourceLoc l, std::unique_ptr<Expr> s, std::unique_ptr<Expr> e)
        : Expr(ExprKind::Range, l), start(std::move(s)), end(std::move(e)) {}

    std::string toString() const override { return "start..end"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Range; }
};

// --- Schema Definition ---

struct RecordField {
    std::string name;
    Type type;
    // [NEW] Default value for Singleton fields (e.g. port: i32 = 8080)
    std::unique_ptr<Expr> defaultValue; 
};

struct EnumVariant {
    std::string name;
    enum Kind { Unit, Tuple, Struct } kind;
    std::vector<Type> tuplePayload;                 
    std::vector<RecordField> structPayload; 
};

struct SchemaDecl : public Decl {
    enum Kind { Record, Enum };
    
    Kind kind;
    std::string name;
    bool hasMeta;
    
    // [NEW] Singleton Flag
    bool isSingleton = false;

    // [NEW] Generic Parameters (e.g. "T" in struct Box<T>)
    std::vector<std::string> genericParams;

    std::vector<RecordField> fields;    
    std::vector<EnumVariant> variants;  
    SourceLoc loc;

    // Struct Ctor
    SchemaDecl(std::string n, std::vector<RecordField> f, SourceLoc l, bool meta)
        : kind(Record), name(std::move(n)), hasMeta(meta), fields(std::move(f)), loc(std::move(l)) {}

    // Enum Ctor
    SchemaDecl(std::string n, std::vector<EnumVariant> v, SourceLoc l, bool meta)
        : kind(Enum), name(std::move(n)), hasMeta(meta), variants(std::move(v)), loc(std::move(l)) {}
};

// --- Schema Usage ---

struct SchemaInitField {
    std::string name;
    std::unique_ptr<Expr> value;
};

struct SchemaExpr : public Expr {
    std::string name;
    std::vector<Type> genericArgs; // [NEW] Stores <i32>
    std::vector<SchemaInitField> fields;
    
    // Update Constructor
    SchemaExpr(SourceLoc l, std::string n, std::vector<Type> args, std::vector<SchemaInitField> f)
        : Expr(ExprKind::SchemaExpr, l), name(std::move(n)), genericArgs(std::move(args)), fields(std::move(f)) {}

    std::string toString() const override { return name + "{...}"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::SchemaExpr; }
};

struct MemberExpr : public Expr {
    std::unique_ptr<Expr> object;
    std::string member;
    
    MemberExpr(SourceLoc l, std::unique_ptr<Expr> obj, std::string mem)
        : Expr(ExprKind::MemberAccess, l), object(std::move(obj)), member(std::move(mem)) {}
    
    std::string toString() const override { 
        return object->toString() + "." + member; 
    }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::MemberAccess; }
};

struct MemberCallNode : public Expr {
    std::unique_ptr<Expr> object;
    std::string methodName;
    std::vector<CallArg> args; 

    MemberCallNode(SourceLoc l, std::unique_ptr<Expr> obj, std::string method, std::vector<CallArg> a)
        : Expr(ExprKind::MemberCall, l), object(std::move(obj)), methodName(std::move(method)), args(std::move(a)) {}

    std::string toString() const override { 
        return object->toString() + "." + methodName + "(...)"; 
    }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::MemberCall; }
};


// -----------------------------------------------------------------------------
// Pattern Matching
// -----------------------------------------------------------------------------

struct Pattern {
    std::string schemaName;   // "Event"
    std::string variantName;  // "Click"
    std::vector<std::string> bindings; // ["x", "y"]
    bool isDefault = false;   // true for "default =>"
};

struct Case {
    Pattern pattern;
    std::unique_ptr<Expr> body;
};

struct MatchStmt : public Expr {
    std::unique_ptr<Expr> target;
    std::vector<Case> cases;

    MatchStmt(SourceLoc loc, std::unique_ptr<Expr> target, std::vector<Case> cases)
        : Expr(ExprKind::Match, loc), target(std::move(target)), cases(std::move(cases)) {}

    std::string toString() const override { return "match"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Match; }
};

struct PrintStmt : public Expr {
    std::vector<std::unique_ptr<Expr>> values;
    PrintStmt(SourceLoc l, std::vector<std::unique_ptr<Expr>> v) 
        : Expr(ExprKind::Print, l), values(std::move(v)) {}

    std::string toString() const override { return "print"; }

    // [FIX] Required for llvm::dyn_cast
    static bool classof(const Expr *e) { return e->kind == ExprKind::Print; }
};

// -----------------------------------------------------------------------------
// Top Level Structures
// -----------------------------------------------------------------------------

struct Function : public Decl {
    std::string name;
    Domain domain;
    uint32_t effects; // Bitmask (IO | NET | FS)
    SourceLoc loc;
    
    std::vector<std::pair<std::string, Type>> args; 
    Type returnType;
    std::vector<std::unique_ptr<Expr>> body;
};

struct Module {
    // [NEW] Unique Identifier for the module (usually the filename stem)
    // Used for name mangling (e.g. "vectors" -> @vectors_add)
    std::string id;

    // List of imports: import "math" as m;
    std::vector<std::unique_ptr<ImportDecl>> imports;
    
    // Top-level definitions
    std::vector<std::unique_ptr<Function>> functions;
    std::vector<std::unique_ptr<SchemaDecl>> schemas; 
    
    // Maps submodule names (e.g. "vec") to loaded Module pointers
    // This allows CodeGen to resolve "math.vec.add" recursively.
    llvm::StringMap<Module*> submodules;
};

} // namespace arklang