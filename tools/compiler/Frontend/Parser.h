#pragma once

#include "Frontend/Lexer.h"
#include "Frontend/AST.h"

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
// Key responsibilities:
// - Parse top-level declarations (imports, schemas, functions)
// - Parse statements and expressions with precedence
// - Recover from syntax errors where possible (panic-mode sync)
// - Preserve token/source ownership safely by storing TokenStream by value
//
// IMPORTANT:
// We keep TokenStream by value so all token StringRef slices remain valid for the
// lifetime of the parser + produced AST construction phase.
// =============================================================================
class Parser {
public:
    // Construct parser with owned token stream.
    // Taking by value avoids dangling references when caller passes a temporary.
    explicit Parser(TokenStream stream);

    // Parse a full module/file.
    // Produces a Module AST containing imports, schemas, and functions.
    std::unique_ptr<Module> parseModule();

    // Error inspection
    bool hasErrors() const { return !errors.empty(); }
    const std::vector<std::string>& getErrors() const { return errors; }

private:
    // =========================================================================
    // Owned input
    // =========================================================================

    // Token stream is stored by value to ensure token/string lifetime safety.
    TokenStream stream;

    // =========================================================================
    // Parser state
    // =========================================================================

    // Current token cursor into stream.tokens.
    std::size_t current = 0;

    // Collected syntax errors (human-readable).
    std::vector<std::string> errors;

    // Panic mode prevents error spam until synchronization point is reached.
    bool panicMode = false;

    // -------------------------------------------------------------------------
    // Context-sensitive parsing guard
    // -------------------------------------------------------------------------
    // When true, parsePrimary() must NOT reinterpret `Ident { ... }` as a
    // schema/struct initialization expression.
    //
    // This fixes grammar ambiguities in control-flow heads like:
    //   for i in 0..n { ... }
    // where `n {` must be parsed as range-end + block, not `SchemaInit`.
    bool suppressStructInitInPrimary = false;

    // RAII helper to temporarily suppress struct-init parsing.
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

    // Parse an expression while temporarily disabling struct-init disambiguation.
    // Used in control-flow/range headers (for/while/if/iter/par).
    std::unique_ptr<Expr> parseExprNoStructInit();

    // =========================================================================
    // Core token helpers
    // =========================================================================

    // Look at current token (or EOF token if out of range).
    const Token& peek() const;

    // Look at next token (LL(2) helper).
    const Token& peekNext() const;

    // Arbitrary lookahead:
    // distance=0 => current token, distance=1 => next token, etc.
    const Token& peekDistance(int distance) const;

    // Most recently consumed token.
    const Token& previous() const;

    // End-of-stream check (true when current token is EOF).
    bool isAtEnd() const;

    // Consume current token and advance cursor.
    const Token& advance();

    // Check current token kind without consuming.
    bool check(TokenType k) const;

    // If current token matches k, consume it and return true.
    bool match(TokenType k);

    // Consume token of expected kind or report error.
    const Token& consume(TokenType k, llvm::Twine msg);

    // =========================================================================
    // Error handling / recovery
    // =========================================================================

    // Helper that reports an error and returns an error expression/null fallback.
    // (Implementation-dependent; kept for compatibility with existing parser.)
    std::unique_ptr<Expr> error(llvm::Twine msg);

    // Report syntax error at a specific source location.
    void errorAt(SourceLoc loc, llvm::Twine msg);

    // Panic-mode recovery: skip tokens until likely statement/declaration boundary.
    void synchronize();

    // Debug watchdog to catch non-advancing loops during parser development.
    void watchdog(const char* loopName, std::size_t& lastPos);

    // =========================================================================
    // Top-level declarations
    // =========================================================================

    // Dispatch global declarations:
    // - import
    // - schema / singleton schema
    // - fn
    std::unique_ptr<Decl> parseTopLevel();

    // Parse:
    //   import "path/to/file.ark" [as alias];
    std::unique_ptr<ImportDecl> parseImport();

    // Parse function declaration:
    //   fn[host] name(args...) -> T !IO { ... }
    //   fn[cpu] ...
    //   fn[gpu] ...
    std::unique_ptr<Function> parseFunction();

    // Parse schema declaration:
    //   schema Name { ... }
    //   singleton schema Name { ... }
    //
    // isSingleton is passed by the top-level dispatcher when relevant.
    std::unique_ptr<SchemaDecl> parseSchema(bool isSingleton = false);

    // Parse type syntax:
    // - primitives (i32, f32, bool, str, void)
    // - containers (vec<T>, slice<T>, tensor<T>)
    // - user types / generics (Foo<T>)
    // - tuple types ((i32, f32))
    // - memory-space suffixes (@gpu:0, @ram)
    Type parseType();

    // Parse capability/effects suffixes into a bitmask:
    //   !IO !FS !NET
    uint32_t parseEffects();

    // =========================================================================
    // Statements
    // =========================================================================

    // Parse one statement (or statement-like expression).
    std::unique_ptr<Expr> parseStmt();

    // Parse block:
    //   { stmt* }
    std::unique_ptr<Expr> parseBlock();

    // Parse variable declaration:
    //   let x = expr;
    //   let x: i32 = expr;
    //   let (a, b) = expr;
    std::unique_ptr<Expr> parseLet();

    // Optional helper for assignment-specific parsing (kept for compatibility).
    std::unique_ptr<Expr> parseAssign(std::unique_ptr<Expr> left);

    // Parse:
    //   return;
    //   return expr;
    std::unique_ptr<Expr> parseReturn();

    // Parse if-expression / if-statement:
    //   if (cond) { ... } else { ... }
    //   if cond expr else expr
    std::unique_ptr<Expr> parseIf();

    // Parse print statement:
    //   print a, b, c;
    std::unique_ptr<Expr> parsePrint();

    // Parse match statement/expression.
    std::unique_ptr<Expr> parseMatch();

    // Parse while loop.
    std::unique_ptr<Expr> parseWhile();

    // Parse for range loop.
    std::unique_ptr<Expr> parseFor();

    // Parse iter loop.
    std::unique_ptr<Expr> parseIter();

    // Parse parallel loop (`par`).
    std::unique_ptr<Expr> parseParLoop();

    // =========================================================================
    // Expressions (precedence / recursive-descent layers)
    // =========================================================================

    // Entry point for expression parsing.
    std::unique_ptr<Expr> parseExpr();

    // Launch syntax (lowest precedence wrapper in current grammar):
    //   dest <- kernel(args...) [as token]
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
    // (% support belongs here, same precedence as * and /)
    std::unique_ptr<Expr> parseFactor();

    // Unary prefix operators: ! -
    std::unique_ptr<Expr> parseUnary();

    // Calls / member access / postfix chaining (if split out in implementation)
    std::unique_ptr<Expr> parseCall();

    // Primary expressions:
    // - identifiers
    // - literals
    // - grouped / tuple
    // - arrays
    // - lambdas
    // - schema init
    std::unique_ptr<Expr> parsePrimary();

    // Parse array literal:
    //   [a, b, c]
    std::unique_ptr<Expr> parseArrayLiteral();

    // Parse either:
    //   (expr)         // grouping
    //   (a, b, c)      // tuple literal
    std::unique_ptr<Expr> parseTupleOrGroup();

    // Parse lambda:
    //   (x: i32) => expr
    //   (x: i32) => { ... }
    std::unique_ptr<Expr> parseLambda();

    // =========================================================================
    // Small parsing helpers
    // =========================================================================

    // Parse call argument list:
    //   (1, foo=2, x)
    std::vector<CallArg> parseArgumentList();

    // Parse and skip/record schema meta block:
    //   meta { ... }
    void parseMetaBlock(bool& outMeta);

    // Parse record fields inside schema record variants / struct payloads.
    std::vector<RecordField> parseRecordFields();
};

} // namespace arklang