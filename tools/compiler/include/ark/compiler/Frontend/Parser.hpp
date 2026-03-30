#pragma once

#include "ark/compiler/Frontend/Lexer.hpp"
#include "ark/compiler/Frontend/AST.hpp"

#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace arklang {

// =============================================================================
// Parser
// =============================================================================
// Recursive-descent parser for Ark source files.
//
// Responsibilities:
// - Parse top-level declarations:
//     import, schema, singleton schema, fn
// - Parse statements and expressions with precedence
// - Parse launch syntax and runtime-aware primary forms
// - Recover from syntax errors with panic-mode synchronization
// - Preserve token/source lifetime by owning TokenStream by value
//
// Lifetime model:
// TokenStream is stored by value so every token's StringRef remains valid for
// the full parser lifetime and throughout AST construction.
// =============================================================================
class Parser {
public:
    // Construct parser with an owned token stream.
    // Taking by value avoids dangling token/string storage when the caller
    // passes a temporary TokenStream.
    explicit Parser(TokenStream stream);

    // Parse a full module/file and produce a Module AST.
    std::unique_ptr<Module> parseModule();

    // Error inspection.
    bool hasErrors() const { return !errors.empty(); }
    const std::vector<std::string>& getErrors() const { return errors; }

private:
    // =========================================================================
    // Owned input
    // =========================================================================

    // Token stream is stored by value so token/source storage remains valid
    // for the entire parser + AST construction phase.
    TokenStream stream;

    // =========================================================================
    // Parser state
    // =========================================================================

    // Cursor into stream.tokens.
    std::size_t current = 0;

    // Collected human-readable syntax errors.
    std::vector<std::string> errors;

    // Panic mode suppresses cascaded error spam until a synchronization point.
    bool panicMode = false;

    // -------------------------------------------------------------------------
    // Context-sensitive parsing guard
    // -------------------------------------------------------------------------
    // When true, parsePrimary() must not reinterpret:
    //
    //   Ident { ... }
    //
    // as a schema/struct initialization expression.
    //
    // This resolves control-flow ambiguities such as:
    //
    //   for i in 0..n { ... }
    //
    // where `n {` must be parsed as range-end + loop body, not as schema init.
    bool suppressStructInitInPrimary = false;

    // RAII helper used to temporarily disable struct-init parsing inside
    // control-flow heads and other ambiguity-prone regions.
    struct ScopedStructInitSuppression {
        Parser& p;
        bool prev;

        explicit ScopedStructInitSuppression(Parser& parser)
            : p(parser), prev(parser.suppressStructInitInPrimary) {
            p.suppressStructInitInPrimary = true;
        }

        ~ScopedStructInitSuppression() {
            p.suppressStructInitInPrimary = prev;
        }
    };

    // Parse an expression while temporarily disabling struct-init
    // reinterpretation. Used in control-flow/range headers such as:
    //
    //   if ...
    //   while ...
    //   for ... in ...
    //   iter ... in ...
    //   par ... in ...
    std::unique_ptr<Expr> parseExprNoStructInit();

    // =========================================================================
    // Core token helpers
    // =========================================================================

    // Current token, or EOF token when out of range.
    const Token& peek() const;

    // One-token lookahead.
    const Token& peekNext() const;

    // Arbitrary lookahead:
    //   distance = 0 -> current
    //   distance = 1 -> next
    //   ...
    const Token& peekDistance(int distance) const;

    // Most recently consumed token.
    const Token& previous() const;

    // True when the current token is EOF.
    bool isAtEnd() const;

    // Consume the current token and advance.
    const Token& advance();

    // Check the current token kind without consuming it.
    bool check(TokenType k) const;

    // If the current token matches k, consume it and return true.
    bool match(TokenType k);

    // Consume a required token or report an error.
    const Token& consume(TokenType k, llvm::Twine msg);

    // =========================================================================
    // Error handling & recovery
    // =========================================================================

    // Report an error and return an expression-level fallback.
    std::unique_ptr<Expr> error(llvm::Twine msg);

    // Report a syntax error at a specific source location.
    void errorAt(SourceLoc loc, llvm::Twine msg);

    // Panic-mode recovery. Advances until a likely declaration or statement
    // boundary is reached.
    void synchronize();

    // Development watchdog for catching non-advancing parser loops.
    void watchdog(const char* loopName, std::size_t& lastPos);

    // =========================================================================
    // Top-level declarations
    // =========================================================================

    // Dispatch one top-level declaration:
    // - import
    // - schema / singleton schema
    // - fn
    std::unique_ptr<Decl> parseTopLevel();

    // Parse:
    //   import "path/to/file.ark" [as alias];
    std::unique_ptr<ImportDecl> parseImport();

    // Parse:
    //   fn[host] name(args...) -> T !IO { ... }
    //   fn[cpu]  name(args...) -> T { ... }
    //   fn[gpu]  name(args...) -> T { ... }
    std::unique_ptr<Function> parseFunction();

    // Parse:
    //   schema Name { ... }
    //   singleton schema Name { ... }
    //
    // isSingleton is supplied by the top-level dispatcher.
    std::unique_ptr<SchemaDecl> parseSchema(bool isSingleton = false);

    // Parse type syntax:
    // - primitives: i32, f32, bool, str, void
    // - containers: vec<T>, slice<T>, tensor<T>
    // - user types / generic instantiations: Foo<T>
    // - tuple types: (i32, f32)
    // - memory-space suffixes where supported by the type grammar
    Type parseType();

    // Parse capability/effect suffixes into a bitmask:
    //   !IO !FS !NET !SYS
    uint32_t parseEffects();

    // =========================================================================
    // Statements
    // =========================================================================

    // Parse one statement or statement-like expression.
    std::unique_ptr<Expr> parseStmt();

    // Parse:
    //   { stmt* }
    std::unique_ptr<Expr> parseBlock();

    // Parse:
    //   let x = expr;
    //   let x: i32 = expr;
    //   let (a, b) = expr;
    std::unique_ptr<Expr> parseLet();

    // Assignment-specific helper retained for compatibility.
    std::unique_ptr<Expr> parseAssign(std::unique_ptr<Expr> left);

    // Parse:
    //   return;
    //   return expr;
    std::unique_ptr<Expr> parseReturn();

    // Parse:
    //   if (cond) { ... } else { ... }
    //   if cond expr else expr
    std::unique_ptr<Expr> parseIf();

    // Parse:
    //   print a, b, c;
    std::unique_ptr<Expr> parsePrint();

    // Parse match statement / expression.
    std::unique_ptr<Expr> parseMatch();

    // Parse while loop.
    std::unique_ptr<Expr> parseWhile();

    // Parse range-based for loop.
    std::unique_ptr<Expr> parseFor();

    // Parse iter loop.
    std::unique_ptr<Expr> parseIter();

    // Parse parallel loop.
    std::unique_ptr<Expr> parseParLoop();

    // =========================================================================
    // Expressions
    // =========================================================================

    // Entry point for expression parsing.
    std::unique_ptr<Expr> parseExpr();

    // Parse launch syntax at the lowest-precedence wrapper layer:
    //
    //   dest <- kernel(args...)
    //   dest <- kernel(args...) as token
    //   dest <- kernel(args...) as token @runtime preset
    std::unique_ptr<Expr> parseLaunch();

    // Logical OR: ||
    std::unique_ptr<Expr> parseLogicOr();

    // Logical AND: &&
    std::unique_ptr<Expr> parseLogicAnd();

    // Equality: == !=
    std::unique_ptr<Expr> parseEquality();

    // Comparison: < > <= >=
    std::unique_ptr<Expr> parseComparison();

    // Additive: + -
    std::unique_ptr<Expr> parseTerm();

    // Multiplicative: * / %
    std::unique_ptr<Expr> parseFactor();

    // Unary prefix operators: ! -
    std::unique_ptr<Expr> parseUnary();

    // Calls, indexing, member access, and postfix chaining.
    std::unique_ptr<Expr> parseCall();

    // Primary expressions:
    // - identifiers
    // - literals
    // - grouped expressions / tuples
    // - arrays
    // - lambdas
    // - schema init
    // - runtime literals
    std::unique_ptr<Expr> parsePrimary();

    // Parse:
    //   [a, b, c]
    std::unique_ptr<Expr> parseArrayLiteral();

    // Parse either:
    //   (expr)
    //   (a, b, c)
    std::unique_ptr<Expr> parseTupleOrGroup();

    // Parse:
    //   (x: i32) => expr
    //   (x: i32) => { ... }
    std::unique_ptr<Expr> parseLambda();

    // Parse runtime literal syntax introduced by KwRuntime:
    //
    //   runtime{
    //       target: route,
    //       endpoint: endpoint,
    //       token: token,
    //       timeout_ms: timeout,
    //       max_burn_usd: burncap
    //   }
    std::unique_ptr<Expr> parseRuntimeLiteral();

    // =========================================================================
    // Small parsing helpers
    // =========================================================================

    // Parse call argument list:
    //   (1, foo=2, x)
    std::vector<CallArg> parseArgumentList();

    // Parse and record or skip schema meta block:
    //   meta { ... }
    void parseMetaBlock(bool& outMeta);

    // Parse record fields used by record schemas and struct-style payloads.
    std::vector<RecordField> parseRecordFields();
};

} // namespace arklang