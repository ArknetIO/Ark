#include "ark/compiler/Frontend/Lexer.hpp"
#include "llvm/ADT/StringSwitch.h"

#include <cctype>
#include <string>
#include <utility>

namespace arklang {

// =============================================================================
// Constructor & Core State
// =============================================================================

Lexer::Lexer(std::string sourceStr, std::string filename) {
    out.source = std::make_shared<std::string>(std::move(sourceStr));
    out.filename = std::move(filename);
}

char Lexer::peek(unsigned offset) const {
    const size_t i = cursor + offset;
    if (i >= out.source->size()) return '\0';
    return (*out.source)[i];
}

bool Lexer::eof(unsigned offset) const {
    return (cursor + offset) >= out.source->size();
}

char Lexer::advance() {
    const char c = eof() ? '\0' : (*out.source)[cursor++];
    if (c == '\n') {
        ++line;
        col = 1;
    } else {
        ++col;
    }
    return c;
}

bool Lexer::match(char expected) {
    if (peek() != expected) return false;
    advance();
    return true;
}

SourceLoc Lexer::loc() const {
    return SourceLoc{out.filename, line, col};
}

Token Lexer::make(TokenType kind, const char* begin, const char* end, SourceLoc at) const {
    return Token{
        kind,
        llvm::StringRef(begin, static_cast<size_t>(end - begin)),
        std::move(at)
    };
}

Token Lexer::lexError(const std::string& msg) {
    SourceLoc at = loc();
    out.errors.push_back(
        out.filename + ":" +
        std::to_string(at.line) + ":" +
        std::to_string(at.col) + ": " + msg
    );
    return Token{TokenType::Error, llvm::StringRef(), std::move(at)};
}

// =============================================================================
// Trivia & Comments
// =============================================================================

void Lexer::skipLineComment() {
    while (!eof() && peek() != '\n') advance();
}

void Lexer::skipBlockComment() {
    advance();
    advance();

    while (!eof()) {
        if (peek() == '*' && peek(1) == '/') {
            advance();
            advance();
            return;
        }
        advance();
    }

    out.errors.push_back(
        out.filename + ":" +
        std::to_string(line) + ":" +
        std::to_string(col) + ": unterminated block comment"
    );
}

void Lexer::skipTrivia() {
    for (;;) {
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) {
            advance();
        }

        if (peek() == '/' && peek(1) == '/') {
            skipLineComment();
            continue;
        }

        if (peek() == '/' && peek(1) == '*') {
            skipBlockComment();
            continue;
        }

        break;
    }
}

// =============================================================================
// Identifiers & Keywords
// =============================================================================

Token Lexer::lexIdentifierOrKeyword() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;

    advance();
    while (!eof()) {
        const char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            advance();
            continue;
        }
        break;
    }

    const char* end = out.source->data() + cursor;
    const llvm::StringRef s(begin, static_cast<size_t>(end - begin));

    const TokenType type = llvm::StringSwitch<TokenType>(s)
        // Control flow
        .Case("fn", TokenType::KwFn)
        .Case("let", TokenType::KwLet)
        .Case("const", TokenType::KwConst)
        .Case("return", TokenType::KwReturn)
        .Case("break", TokenType::KwBreak)
        .Case("continue", TokenType::KwContinue)
        .Case("if", TokenType::KwIf)
        .Case("else", TokenType::KwElse)
        .Case("for", TokenType::KwFor)
        .Case("while", TokenType::KwWhile)
        .Case("par", TokenType::KwPar)
        .Case("in", TokenType::KwIn)
        .Case("iter", TokenType::KwIter)

        // Modules & visibility
        .Case("mod", TokenType::KwMod)
        .Case("pub", TokenType::KwPub)
        .Case("import", TokenType::KwImport)

        // Execution, memory & runtime
        .Case("gpu", TokenType::KwGpu)
        .Case("host", TokenType::KwHost)
        .Case("ram", TokenType::KwRam)
        .Case("cpu", TokenType::KwCpu)
        .Case("allocof", TokenType::KwAllocof)
        .Case("runtime", TokenType::KwRuntime)

        // Async & concurrency
        .Case("launch", TokenType::KwLaunch)
        .Case("await", TokenType::KwAwait)
        .Case("as", TokenType::KwAs)

        // Capabilities
        .Case("IO", TokenType::KwIo)
        .Case("NET", TokenType::KwNet)
        .Case("FS", TokenType::KwFs)
        .Case("SYS", TokenType::KwSys)

        // Primitives
        .Case("void", TokenType::KwVoid)
        .Case("u8", TokenType::KwU8)
        .Case("u16", TokenType::KwU16)
        .Case("u32", TokenType::KwU32)
        .Case("u64", TokenType::KwU64)
        .Case("i8", TokenType::KwI8)
        .Case("i16", TokenType::KwI16)
        .Case("i32", TokenType::KwI32)
        .Case("i64", TokenType::KwI64)
        .Case("f32", TokenType::KwF32)
        .Case("f64", TokenType::KwF64)
        .Case("bool", TokenType::KwBool)
        .Case("str", TokenType::KwStr)

        // Containers
        .Case("vec", TokenType::KwVec)
        .Case("slice", TokenType::KwSlice)
        .Case("tensor", TokenType::KwTensor)

        // Constants
        .Case("true", TokenType::KwTrue)
        .Case("false", TokenType::KwFalse)
        .Case("null", TokenType::KwNull)

        // Pattern matching
        .Case("match", TokenType::KwMatch)
        .Case("case", TokenType::KwCase)
        .Case("default", TokenType::KwDefault)

        // Data modeling & misc
        .Case("schema", TokenType::KwSchema)
        .Case("meta", TokenType::KwMeta)
        .Case("print", TokenType::KwPrint)
        .Case("singleton", TokenType::KwSingleton)

        .Default(TokenType::Identifier);

    return make(type, begin, end, std::move(at));
}

// =============================================================================
// Numeric Literals
// =============================================================================

Token Lexer::lexNumber() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;

    while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    bool isFloat = false;

    if (peek() == '.' && peek(1) != '.') {
        if (std::isdigit(static_cast<unsigned char>(peek(1)))) {
            isFloat = true;
            advance();
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        }
    }

    if (peek() == 'e' || peek() == 'E') {
        const size_t saveCursor = cursor;
        const int saveLine = line;
        const int saveCol = col;

        advance();
        if (peek() == '+' || peek() == '-') advance();

        if (std::isdigit(static_cast<unsigned char>(peek()))) {
            isFloat = true;
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        } else {
            cursor = saveCursor;
            line = saveLine;
            col = saveCol;
        }
    }

    const char* end = out.source->data() + cursor;
    return make(isFloat ? TokenType::Float : TokenType::Integer, begin, end, std::move(at));
}

// =============================================================================
// String & Character Literals
// =============================================================================

Token Lexer::lexString() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;

    advance();

    while (!eof()) {
        const char c = peek();

        if (c == '"') break;

        if (c == '\\') {
            advance();
            if (!eof()) advance();
            continue;
        }

        if (c == '\n') {
            return lexError("Unterminated string literal");
        }

        advance();
    }

    if (eof()) {
        return lexError("Unterminated string literal");
    }

    advance();

    const char* end = out.source->data() + cursor;
    return make(TokenType::String, begin, end, std::move(at));
}

Token Lexer::lexChar() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;

    advance();

    if (eof() || peek() == '\n') {
        return lexError("Unterminated character literal");
    }

    if (peek() == '\\') {
        advance();
        if (eof() || peek() == '\n') {
            return lexError("Unterminated character literal");
        }
        advance();
    } else {
        advance();
    }

    if (peek() != '\'') {
        return lexError("Multi-character character literal or missing closing quote");
    }

    advance();

    const char* end = out.source->data() + cursor;
    return make(TokenType::Char, begin, end, std::move(at));
}

// =============================================================================
// Punctuation & Operators
// =============================================================================

Token Lexer::lexPunctOrOp() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;
    const char c = advance();

    if (c == '.' && match('.'))  return make(TokenType::Range, begin, out.source->data() + cursor, std::move(at));
    if (c == '-' && match('>'))  return make(TokenType::Arrow, begin, out.source->data() + cursor, std::move(at));
    if (c == '<' && match('-'))  return make(TokenType::ArrowL, begin, out.source->data() + cursor, std::move(at));
    if (c == '=' && match('>'))  return make(TokenType::FatArrow, begin, out.source->data() + cursor, std::move(at));

    if (c == '=' && match('='))  return make(TokenType::EqualEqual, begin, out.source->data() + cursor, std::move(at));
    if (c == '!' && match('='))  return make(TokenType::BangEqual, begin, out.source->data() + cursor, std::move(at));
    if (c == '<' && match('='))  return make(TokenType::LessEqual, begin, out.source->data() + cursor, std::move(at));
    if (c == '>' && match('='))  return make(TokenType::GreaterEqual, begin, out.source->data() + cursor, std::move(at));

    if (c == '&' && match('&'))  return make(TokenType::AmpAmp, begin, out.source->data() + cursor, std::move(at));
    if (c == '|' && match('|'))  return make(TokenType::PipePipe, begin, out.source->data() + cursor, std::move(at));
    if (c == '<' && match('<'))  return make(TokenType::Shl, begin, out.source->data() + cursor, std::move(at));
    if (c == '>' && match('>'))  return make(TokenType::Shr, begin, out.source->data() + cursor, std::move(at));

    if (c == '+' && match('='))  return make(TokenType::PlusEq, begin, out.source->data() + cursor, std::move(at));
    if (c == '-' && match('='))  return make(TokenType::MinusEq, begin, out.source->data() + cursor, std::move(at));
    if (c == '*' && match('='))  return make(TokenType::StarEq, begin, out.source->data() + cursor, std::move(at));
    if (c == '/' && match('='))  return make(TokenType::SlashEq, begin, out.source->data() + cursor, std::move(at));
    if (c == '%' && match('='))  return make(TokenType::PercentEq, begin, out.source->data() + cursor, std::move(at));

    const char* end = out.source->data() + cursor;

    switch (c) {
        case '(': return make(TokenType::LParen, begin, end, std::move(at));
        case ')': return make(TokenType::RParen, begin, end, std::move(at));
        case '{': return make(TokenType::LBrace, begin, end, std::move(at));
        case '}': return make(TokenType::RBrace, begin, end, std::move(at));
        case '[': return make(TokenType::LBracket, begin, end, std::move(at));
        case ']': return make(TokenType::RBracket, begin, end, std::move(at));
        case ':': return make(TokenType::Colon, begin, end, std::move(at));
        case ';': return make(TokenType::Semicolon, begin, end, std::move(at));
        case ',': return make(TokenType::Comma, begin, end, std::move(at));
        case '.': return make(TokenType::Dot, begin, end, std::move(at));
        case '=': return make(TokenType::Equal, begin, end, std::move(at));
        case '@': return make(TokenType::At, begin, end, std::move(at));
        case '?': return make(TokenType::Question, begin, end, std::move(at));

        case '+': return make(TokenType::Plus, begin, end, std::move(at));
        case '-': return make(TokenType::Minus, begin, end, std::move(at));
        case '*': return make(TokenType::Star, begin, end, std::move(at));
        case '/': return make(TokenType::Slash, begin, end, std::move(at));
        case '%': return make(TokenType::Percent, begin, end, std::move(at));

        case '&': return make(TokenType::Amp, begin, end, std::move(at));
        case '|': return make(TokenType::Pipe, begin, end, std::move(at));
        case '^': return make(TokenType::Caret, begin, end, std::move(at));
        case '~': return make(TokenType::Tilde, begin, end, std::move(at));

        case '<': return make(TokenType::Less, begin, end, std::move(at));
        case '>': return make(TokenType::Greater, begin, end, std::move(at));
        case '!': return make(TokenType::Not, begin, end, std::move(at));

        default:
            return lexError(std::string("Unexpected character: ") + c);
    }
}

// =============================================================================
// Main Tokenization Loop
// =============================================================================

TokenStream Lexer::tokenize() {
    cursor = 0;
    line = 1;
    col = 1;
    out.tokens.clear();
    out.errors.clear();

    while (true) {
        skipTrivia();

        if (eof()) {
            out.tokens.push_back(Token{
                TokenType::Eof,
                llvm::StringRef(),
                SourceLoc{out.filename, line, col}
            });
            break;
        }

        const char c = peek();

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            out.tokens.push_back(lexIdentifierOrKeyword());
            continue;
        }

        if (std::isdigit(static_cast<unsigned char>(c))) {
            Token t = lexNumber();
            if (t.type == TokenType::Integer && t.text.size() > 1 && t.text.front() == '0') {
                out.errors.push_back(
                    out.filename + ":" +
                    std::to_string(t.loc.line) + ":" +
                    std::to_string(t.loc.col) + ": invalid leading zero"
                );
            }
            out.tokens.push_back(std::move(t));
            continue;
        }

        if (c == '"') {
            out.tokens.push_back(lexString());
            continue;
        }

        if (c == '\'') {
            out.tokens.push_back(lexChar());
            continue;
        }

        out.tokens.push_back(lexPunctOrOp());
    }

    return out;
}

} // namespace arklang