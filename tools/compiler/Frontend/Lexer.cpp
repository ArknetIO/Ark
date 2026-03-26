#include "Frontend/Lexer.h"
#include "llvm/ADT/StringSwitch.h"
#include <cctype>
#include <algorithm> 

namespace arklang {

// =============================================================================
// Constructor & Core State Management
// =============================================================================

Lexer::Lexer(std::string sourceStr, std::string filename) {
    // We strictly take ownership of the source string and filename here.
    // Storing them in the shared output structure ensures that SourceLoc 
    // references (which are views into this data) remain valid throughout 
    // the entire compilation pipeline, even if the Lexer instance is destroyed.
    out.source = std::make_shared<std::string>(std::move(sourceStr));
    out.filename = std::move(filename);
}

char Lexer::peek(unsigned offset) const {
    size_t i = cursor + offset;
    if (i >= out.source->size()) return '\0';
    return (*out.source)[i];
}

bool Lexer::eof(unsigned offset) const {
    return (cursor + offset) >= out.source->size();
}

char Lexer::advance() {
    char c = eof() ? '\0' : (*out.source)[cursor++];
    if (c == '\n') {
        line += 1;
        col = 1;
    } else {
        col += 1;
    }
    return c;
}

bool Lexer::match(char expected) {
    if (peek() != expected) return false;
    advance();
    return true;
}

// Returns a source location referencing the current cursor position.
// This relies on the persistent filename stored in 'out'.
SourceLoc Lexer::loc() const {
    return SourceLoc{out.filename, line, col};
}

Token Lexer::make(TokenType type, const char* begin, const char* end, SourceLoc at) const {
    llvm::StringRef ref(begin, static_cast<size_t>(end - begin));
    return Token{type, ref, std::move(at)};
}

Token Lexer::lexError(const std::string& msg) {
    SourceLoc at = loc();
    // Standard error format: filename:line:col: message
    out.errors.push_back(out.filename + ":" + std::to_string(line) + ":" + std::to_string(col) + ": " + msg);
    return Token{TokenType::Error, llvm::StringRef(), std::move(at)};
}

// =============================================================================
// Whitespace & Comments
// =============================================================================

void Lexer::skipLineComment() {
    while (!eof() && peek() != '\n') advance();
}

void Lexer::skipBlockComment() {
    advance(); advance(); // Consume /*
    while (!eof()) {
        if (peek() == '*' && peek(1) == '/') {
            advance(); advance();
            return;
        }
        advance();
    }
    // If EOF is reached inside comment, the parser will eventually hit EOF
    // expecting a token and error out there.
}

void Lexer::skipTrivia() {
    for (;;) {
        // Skip basic whitespace
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) advance();
        
        // Check for single-line comments (//)
        if (peek() == '/' && peek(1) == '/') {
            skipLineComment();
            continue;
        }
        // Check for block comments (/* ... */)
        if (peek() == '/' && peek(1) == '*') {
            skipBlockComment();
            continue;
        }
        break;
    }
}

Token Lexer::lexChar() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;
    
    // Consume opening quote '
    advance(); 
    
    if (eof() || peek() == '\n') return lexError("Unterminated character literal");

    // Handle escape sequences or normal chars
    if (peek() == '\\') {
        advance(); // consume '\'
        if (eof()) return lexError("Unterminated character literal");
        advance(); // consume escaped char
    } else {
        advance(); // consume char
    }

    if (peek() != '\'') return lexError("Multi-character character literal or missing closing quote");
    
    // Consume closing quote '
    advance();
    
    const char* end = out.source->data() + cursor;
    return make(TokenType::Char, begin, end, std::move(at));
}

// =============================================================================
// Keyword & Identifier Logic
// =============================================================================

Token Lexer::lexIdentifierOrKeyword() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;

    // First char is known to be alpha/_ by the caller
    advance(); 
    while (!eof()) {
        char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            advance();
            continue;
        }
        break;
    }

    const char* end = out.source->data() + cursor;
    llvm::StringRef s(begin, static_cast<size_t>(end - begin));

    // Map identifiers to Keywords using LLVM's optimized StringSwitch
    TokenType type = llvm::StringSwitch<TokenType>(s)
        // --- Control Flow ---
        .Case("fn", TokenType::KwFn)
        .Case("let", TokenType::KwLet)
        .Case("return", TokenType::KwReturn)
        .Case("if", TokenType::KwIf)        
        .Case("else", TokenType::KwElse)    
        .Case("for", TokenType::KwFor)      
        .Case("while", TokenType::KwWhile)  
        .Case("par", TokenType::KwPar)
        .Case("iter", TokenType::KwIter)    
        .Case("in", TokenType::KwIn)
        .Case("await", TokenType::KwAwait)
        
        // --- Modules & Visibility ---
        .Case("mod", TokenType::KwMod)
        .Case("pub", TokenType::KwPub)
        .Case("import", TokenType::KwImport)
        .Case("as", TokenType::KwAs)

        // --- Memory & Execution ---
        .Case("allocof", TokenType::KwAllocof)
        .Case("launch", TokenType::KwLaunch)
        .Case("host", TokenType::KwHost)
        .Case("cpu", TokenType::KwCpu)
        .Case("gpu", TokenType::KwGpu)
        .Case("ram", TokenType::KwRam)
        
        // --- Primitives ---
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
        .Case("void", TokenType::KwVoid)
        
        // --- Values ---
        .Case("true", TokenType::KwTrue)
        .Case("false", TokenType::KwFalse)
        
        // --- Capabilities (Uppercase) ---
        .Case("IO", TokenType::KwIo)
        .Case("NET", TokenType::KwNet)
        .Case("FS", TokenType::KwFs)
        
        // --- Containers ---
        .Case("vec", TokenType::KwVec)
        .Case("slice", TokenType::KwSlice)
        .Case("tensor", TokenType::KwTensor)

        // --- Data Modeling ---
        .Case("schema", TokenType::KwSchema)
        .Case("singleton", TokenType::KwSingleton) // Added for Global Singleton support
        .Case("meta", TokenType::KwMeta)
        .Case("match", TokenType::KwMatch)
        .Case("case", TokenType::KwCase)
        .Case("default", TokenType::KwDefault)
        
        // --- Misc ---
        .Case("print", TokenType::KwPrint) 
        
        .Default(TokenType::Identifier);

    return make(type, begin, end, std::move(at));
}

// =============================================================================
// Numeric Literals
// =============================================================================

Token Lexer::lexNumber() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;

    // Consume integer part
    while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) advance();

    bool isFloat = false;
    // Check for fractional part
    if (peek() == '.') {
        // Lookahead: ensure it's not a range operator ".."
        // If the NEXT char is a digit, it's a float (e.g., 1.2).
        // If the NEXT char is '.', it's a range (e.g., 1..10), so we stop here.
        if (std::isdigit(static_cast<unsigned char>(peek(1)))) {
            isFloat = true;
            advance(); // Consume '.'
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) advance();
            
            // Exponent part (e.g. 1.2e-3)
            if (peek() == 'e' || peek() == 'E') {
                size_t saveCursor = cursor;
                advance(); // consume 'e'
                
                if (peek() == '+' || peek() == '-') advance();
                
                if (std::isdigit(static_cast<unsigned char>(peek()))) {
                    while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) advance();
                } else {
                    // Backtrack if 'e' is not followed by digits (e.g., variable 'e' after number)
                    cursor = saveCursor; 
                }
            }
        }
    }

    const char* end = out.source->data() + cursor;
    return make(isFloat ? TokenType::Float : TokenType::Integer, begin, end, std::move(at));
}

// =============================================================================
// String Literals
// =============================================================================

Token Lexer::lexString() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;
    
    // We already peeked '"', so advance past it
    advance(); 

    while (!eof()) {
        char current = peek();
        if (current == '"') break; // End of string
        
        if (current == '\\') {
            advance(); // Consume backslash
            if (!eof()) advance(); // Consume escaped char
        } else {
            advance(); // Consume regular char
        }
    }

    if (eof()) return lexError("Unterminated string literal");

    // Consume closing quote
    advance(); 
    const char* end = out.source->data() + cursor;
    
    return make(TokenType::String, begin, end, std::move(at));
}

// =============================================================================
// Operators & Punctuation
// =============================================================================

Token Lexer::lexPunctOrOp() {
    SourceLoc at = loc();
    const char* begin = out.source->data() + cursor;
    char c = advance();
    const char* end = out.source->data() + cursor;

    // Double char tokens
    if (c == '-' && match('>')) return make(TokenType::Arrow, begin, out.source->data() + cursor, std::move(at));
    if (c == '.') {
        if (match('.')) return make(TokenType::Range, begin, out.source->data() + cursor, std::move(at));
        return make(TokenType::Dot, begin, out.source->data() + cursor, std::move(at));
    }
    if (c == '<' && match('-')) return make(TokenType::ArrowL, begin, out.source->data() + cursor, std::move(at));
    if (c == '!' && match('=')) return make(TokenType::BangEqual, begin, out.source->data() + cursor, std::move(at));
    if (c == '=' && match('=')) return make(TokenType::EqualEqual, begin, out.source->data() + cursor, std::move(at));
    if (c == '=' && match('>')) return make(TokenType::FatArrow, begin, out.source->data() + cursor, std::move(at));
    if (c == '<' && match('=')) return make(TokenType::LessEqual, begin, out.source->data() + cursor, std::move(at));
    if (c == '>' && match('=')) return make(TokenType::GreaterEqual, begin, out.source->data() + cursor, std::move(at));

    // Single char tokens
    switch (c) {
        case '(': return make(TokenType::LParen, begin, end, std::move(at));
        case ')': return make(TokenType::RParen, begin, end, std::move(at));
        case '{': return make(TokenType::LBrace, begin, end, std::move(at));
        case '}': return make(TokenType::RBrace, begin, end, std::move(at));
        case '[': return make(TokenType::LBracket, begin, end, std::move(at));
        case ']': return make(TokenType::RBracket, begin, end, std::move(at));
        case ',': return make(TokenType::Comma, begin, end, std::move(at));
        case ';': return make(TokenType::Semicolon, begin, end, std::move(at));
        case ':': return make(TokenType::Colon, begin, end, std::move(at));
        case '@': return make(TokenType::At, begin, end, std::move(at));
        case '+': return make(TokenType::Plus, begin, end, std::move(at));
        case '-': return make(TokenType::Minus, begin, end, std::move(at));
        case '*': return make(TokenType::Star, begin, end, std::move(at));
        case '/': return make(TokenType::Slash, begin, end, std::move(at));
        case '%': return make(TokenType::Percent, begin, end, std::move(at));
        case '!': return make(TokenType::Not, begin, end, std::move(at));
        case '=': return make(TokenType::Equal, begin, end, std::move(at));
        case '<': return make(TokenType::Less, begin, end, std::move(at));
        case '>': return make(TokenType::Greater, begin, end, std::move(at));

        default:  return lexError(std::string("Unexpected character: ") + c);
    }
}

// =============================================================================
// Main Tokenizer Loop
// =============================================================================

TokenStream Lexer::tokenize() {
    cursor = 0; line = 1; col = 1;
    
    // Ensure the output is clean before we start
    out.tokens.clear();
    out.errors.clear();

    while (true) {
        // Skip whitespace/comments FIRST to ensure accurate EOF checks
        skipTrivia();
        
        if (eof()) {
            out.tokens.push_back(Token{TokenType::Eof, llvm::StringRef(), SourceLoc{out.filename, line, col}});
            break;
        }
        
        char c = peek();
        
        // Identifiers & Keywords (start with alpha or _)
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            out.tokens.push_back(lexIdentifierOrKeyword());
            continue;
        }
        
        // Numbers (start with digit)
        if (std::isdigit(static_cast<unsigned char>(c))) {
            Token t = lexNumber();
            // Strict check for invalid leading zeros on integers (e.g. 0123)
            if (t.type == TokenType::Integer && t.text.size() > 1 && t.text[0] == '0') {
                out.errors.push_back(out.filename + ":" + std::to_string(t.loc.line) + ":" +
                                     std::to_string(t.loc.col) + ": invalid leading zero");
            }
            out.tokens.push_back(t);
            continue;
        }
        
        // String Literals
        if (c == '"') {
            out.tokens.push_back(lexString());
            continue;
        }

        // [HOOKED IN] Character Literals
        if (c == '\'') {
            out.tokens.push_back(lexChar());
            continue;
        }
        
        // Punctuation & Operators
        Token t = lexPunctOrOp();
        if (t.type == TokenType::Error) {
            // Keep error token in stream for parser to handle/recover
            out.tokens.push_back(t);
            continue;
        }
        out.tokens.push_back(t);
    }
    return std::move(out);
}

} // namespace arklang