#pragma once

#include "Frontend/AST.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <vector>

namespace arklang {

enum class TokenType {
    // --- Keywords: Control Flow ---
    KwFn, KwLet, KwConst,
    KwReturn, KwBreak, KwContinue,
    KwIf, KwElse,
    KwFor, KwWhile, KwPar,
    KwIn, KwIter,

    // --- Keywords: Modules & Visibility ---
    KwMod,
    KwPub,
    KwImport,

    // --- Keywords: Execution Domains / Memory ---
    KwGpu, KwHost, KwRam, KwCpu,
    KwAllocof,

    // --- Keywords: Async & Concurrency ---
    KwLaunch, KwAwait, KwAs,

    // --- Keywords: Capabilities ---
    KwIo, KwNet, KwFs, KwSys,

    // --- Keywords: Primitives ---
    KwVoid,
    KwU8, KwU16, KwU32, KwU64,
    KwI8, KwI16, KwI32, KwI64,
    KwF32, KwF64,
    KwBool, KwStr,

    // --- Keywords: Containers ---
    KwVec,
    KwSlice,
    KwTensor,

    // --- Keywords: Constants ---
    KwTrue, KwFalse, KwNull,

    // --- Keywords: Pattern Matching ---
    KwMatch,
    KwCase,
    KwDefault,

    // --- Keywords: Data Modeling / Misc ---
    KwSchema,
    KwMeta,
    KwPrint,
    KwSingleton,

    // --- Identifiers & Literals ---
    Identifier,
    Integer,
    Float,
    String,
    Char,

    // --- Delimiters / Punctuation ---
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,

    Colon,
    Semicolon,
    Comma,
    Dot,
    Range,      // ..
    DotDot = Range,

    // --- Assignment / Arrows / Misc ---
    Equal,      // =
    Arrow,      // ->
    FatArrow,   // =>
    ArrowL,     // <-
    At,         // @
    Question,   // ?

    // --- Arithmetic Operators ---
    Plus, Minus, Star, Slash, Percent, // + - * / %

    // --- Bitwise Operators ---
    Amp, Pipe, Caret, Tilde,
    Shl, Shr,

    // --- Logical / Comparison ---
    Less, Greater, Not,
    EqualEqual,
    BangEqual,
    LessEqual,
    GreaterEqual,
    AmpAmp,
    PipePipe,

    // --- Compound Assignments ---
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,

    // --- Control ---
    Eof,
    Error,

    // --- Aliases ---
    Semi = Semicolon
};

struct Token {
    TokenType type;
    llvm::StringRef text;
    SourceLoc loc;
};

struct TokenStream {
    std::string filename;
    std::shared_ptr<std::string> source;
    std::vector<Token> tokens;
    std::vector<std::string> errors;
};

using TokenList = TokenStream;

class Lexer {
public:
    explicit Lexer(std::string source, std::string filename = "<stdin>");
    TokenStream tokenize();

private:
    TokenStream out;

    size_t cursor = 0;
    int line = 1;
    int col = 1;

    char peek(unsigned offset = 0) const;
    bool eof(unsigned offset = 0) const;
    char advance();
    bool match(char expected);
    SourceLoc loc() const;
    Token make(TokenType kind, const char* begin, const char* end, SourceLoc at) const;
    Token lexError(const std::string& msg);

    void skipTrivia();
    void skipLineComment();
    void skipBlockComment();

    Token lexIdentifierOrKeyword();
    Token lexNumber();
    Token lexString();
    Token lexChar();
    Token lexPunctOrOp();
};

} // namespace arklang