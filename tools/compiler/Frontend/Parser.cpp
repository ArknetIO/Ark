#include "Frontend/Parser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h" // [NEW] Required for implicit alias derivation
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>


namespace {

static std::string stripQuotedToken(llvm::StringRef text) {
    std::string s = text.str();
    if (s.size() >= 2) {
        const char first = s.front();
        const char last = s.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return s.substr(1, s.size() - 2);
        }
    }
    return s;
}

} // namespace

namespace arklang {

// -----------------------------------------------------------------------------
// Core Helpers
// -----------------------------------------------------------------------------

Parser::Parser(TokenStream stream) : stream(std::move(stream)) {}

const Token& Parser::peek() const {
    if (current >= stream.tokens.size()) return stream.tokens.back();
    return stream.tokens[current];
}

const Token& Parser::peekNext() const {
    if (current + 1 >= stream.tokens.size()) return stream.tokens.back();
    return stream.tokens[current + 1];
}

const Token& Parser::previous() const {
    if (current == 0) return stream.tokens[0];
    return stream.tokens[current - 1];
}

bool Parser::isAtEnd() const {
    return peek().type == TokenType::Eof;
}

const Token& Parser::advance() {
    if (!isAtEnd()) current++;
    return previous();
}

bool Parser::check(TokenType k) const {
    if (isAtEnd()) return false;
    return peek().type == k;
}

bool Parser::match(TokenType k) {
    if (check(k)) {
        advance();
        return true;
    }
    return false;
}

const Token& Parser::consume(TokenType k, llvm::Twine msg) {
    if (check(k)) return advance();
    errorAt(peek().loc, msg);
    return stream.tokens[current];
}

void Parser::errorAt(SourceLoc loc, llvm::Twine msg) {
    if (panicMode) return;
    panicMode = true;
    errors.push_back(loc.file + ":" + std::to_string(loc.line) + ":" + 
                     std::to_string(loc.col) + ": " + msg.str());
}

void Parser::synchronize() {
    panicMode = false;
    while (!isAtEnd()) {
        if (previous().type == TokenType::Semicolon) return;
        switch (peek().type) {
            case TokenType::KwFn:
            case TokenType::KwSchema:
            case TokenType::KwLet:
            case TokenType::KwFor:
            case TokenType::KwWhile:
            case TokenType::KwIf:
            case TokenType::KwReturn:
            case TokenType::KwPrint:
            case TokenType::KwMatch:
            case TokenType::KwImport: // [FIX] Use KwImport
                return;
            default: ;
        }
        advance();
    }
}
// [WATCHDOG] Infinite Loop Detection
void Parser::watchdog(const char* loopName, size_t& lastPos) {
    // Only trigger if we have NOT moved since the LAST check.
    if (current == lastPos && !panicMode) {
         // [FIX] Added peek().loc.file.c_str() to the output
         fprintf(stderr, "\n[FATAL] Infinite Loop detected in '%s'\n", loopName);
         fprintf(stderr, "  File:  %s\n", peek().loc.file.c_str());
         fprintf(stderr, "  Line:  %d\n", peek().loc.line);
         fprintf(stderr, "  Token: '%s'\n", peek().text.str().c_str());
         fprintf(stderr, "PLEASE submit a bug report with the code above.\n");
         abort();
    }
    lastPos = current;
}
// -----------------------------------------------------------------------------
// Top Level
// -----------------------------------------------------------------------------

std::unique_ptr<Module> Parser::parseModule() {
    auto mod = std::make_unique<Module>();
    size_t lastPos = -1;

    while (!isAtEnd()) {
        watchdog("parseModule", lastPos);
        
        auto decl = parseTopLevel();
        if (decl) {
            // RTTI Cast to store in specific vectors
            if (auto* fn = dynamic_cast<Function*>(decl.get())) {
                decl.release(); // Release ownership from generic unique_ptr
                mod->functions.push_back(std::unique_ptr<Function>(fn));
            }
            else if (auto* sc = dynamic_cast<SchemaDecl*>(decl.get())) {
                decl.release();
                mod->schemas.push_back(std::unique_ptr<SchemaDecl>(sc));
            }
            else if (auto* imp = dynamic_cast<ImportDecl*>(decl.get())) {
                decl.release();
                mod->imports.push_back(std::unique_ptr<ImportDecl>(imp));
            }
        } else if (!panicMode) {
            // If parseTopLevel failed but didn't advance/panic, force advance to avoid infinite loop
            if (current == lastPos) advance();
        }

        if (panicMode) synchronize();
    }
    return mod;
}


// -----------------------------------------------------------------------------
// Functions & Types
// -----------------------------------------------------------------------------

uint32_t Parser::parseEffects() {
    uint32_t effects = 0;
    size_t lastPos = static_cast<size_t>(-1);

    while (check(TokenType::Not)) {
        watchdog("parseEffects", lastPos);
        consume(TokenType::Not, "Expected '!'");

        std::string capName;

        if (match(TokenType::Identifier)) {
            capName = previous().text.str();
            for (char& ch : capName) {
                unsigned char c = static_cast<unsigned char>(ch);
                ch = static_cast<char>(std::toupper(c));
            }
        } else if (match(TokenType::KwFs)) {
            capName = "FS";
        } else if (match(TokenType::KwNet)) {
            capName = "NET";
        } else if (match(TokenType::KwIo)) {
            capName = "IO";
        } else {
            errorAt(peek().loc, "Expected capability name after '!'");
            if (current == lastPos) advance();
            continue;
        }

        if (capName == "FS") {
            effects |= EFF_FS;
        } else if (capName == "NET") {
            effects |= EFF_NET;
        } else if (capName == "IO") {
            effects |= EFF_IO;
        } else if (capName == "SYS") {
            effects |= EFF_SYS;
        } else {
            errorAt(previous().loc, "Unknown capability '" + capName + "'");
        }
    }

    return effects;
}

std::unique_ptr<Function> Parser::parseFunction() {
    SourceLoc loc = previous().loc;

    Domain domain = Domain::Host;
    if (match(TokenType::LBracket)) {
        if (match(TokenType::KwGpu)) domain = Domain::GPU;
        else if (match(TokenType::KwCpu)) domain = Domain::CPU;
        else if (match(TokenType::KwHost)) domain = Domain::Host;
        else errorAt(peek().loc, "Expected 'gpu', 'host', or 'cpu'");
        consume(TokenType::RBracket, "Expected ']'");
    }

    std::string name = consume(TokenType::Identifier, "Expected function name").text.str();
    consume(TokenType::LParen, "Expected '('");

    std::vector<std::pair<std::string, Type>> args;
    if (!check(TokenType::RParen)) {
        do {
            std::string argName = consume(TokenType::Identifier, "Expected parameter name").text.str();
            consume(TokenType::Colon, "Expected ':'");
            Type argType = parseType();
            args.push_back({argName, argType});
        } while (match(TokenType::Comma));
    }
    consume(TokenType::RParen, "Expected ')'");

    if (domain == Domain::GPU) {
        for (auto& arg : args) {
            if (arg.second.isContainer() && arg.second.space.kind == Space::RAM) {
                arg.second.space.kind = Space::GPU;
            }
        }
    }

    Type retType = {Type::Void};
    if (match(TokenType::Arrow)) {
        retType = parseType();
    }

    uint32_t effects = Effect::None;
    if (check(TokenType::Not)) {
        effects |= parseEffects();

        while (match(TokenType::Comma)) {
            if (!check(TokenType::Not)) {
                errorAt(peek().loc, "Expected '!' after ',' in effect list");
                break;
            }
            effects |= parseEffects();
        }
    }

    consume(TokenType::LBrace, "Expected '{'");
    std::vector<std::unique_ptr<Expr>> body;

    size_t lastPos = static_cast<size_t>(-1);
    size_t stuckCounter = 0;

    while (!check(TokenType::RBrace) && !isAtEnd()) {
        if (current == lastPos) {
            stuckCounter++;
            if (stuckCounter > 100) {
                fprintf(stderr, "[FATAL] Infinite Loop in %s:%d at token '%s'\n",
                        peek().loc.file.c_str(), peek().loc.line, peek().text.str().c_str());
                exit(1);
            }
        } else {
            stuckCounter = 0;
            lastPos = current;
        }

        if (check(TokenType::KwFn) || check(TokenType::KwSchema)) {
            errorAt(peek().loc, "Missing '}' before next declaration");
            break;
        }

        size_t startPos = current;
        auto stmt = parseStmt();

        if (stmt) {
            body.push_back(std::move(stmt));
            if (current == startPos && !panicMode) {
                advance();
            }
        } else if (!panicMode) {
            advance();
        }

        if (panicMode) synchronize();
    }

    consume(TokenType::RBrace, "Expected '}'");

    auto fn = std::make_unique<Function>();
    fn->name = name;
    fn->domain = domain;
    fn->effects = effects;
    fn->args = args;
    fn->returnType = retType;
    fn->body = std::move(body);
    fn->loc = loc;
    return fn;
}

// -----------------------------------------------------------------------------
// Statements
// -----------------------------------------------------------------------------


// =============================================================================
// Statement Parsing
// Adds support for:
// - break;
// - continue;
// while preserving the existing statement/expression split.
// =============================================================================
std::unique_ptr<Expr> Parser::parseStmt() {
    if (match(TokenType::KwLet))      return parseLet();
    if (match(TokenType::KwReturn))   return parseReturn();
    if (match(TokenType::KwPar))      return parseParLoop();
    if (match(TokenType::KwIf))       return parseIf();
    if (match(TokenType::KwWhile))    return parseWhile();
    if (match(TokenType::KwFor))      return parseFor();
    if (match(TokenType::KwIter))     return parseIter();
    if (match(TokenType::KwMatch))    return parseMatch();
    if (match(TokenType::KwPrint))    return parsePrint();

    if (match(TokenType::KwBreak)) {
        SourceLoc loc = previous().loc;
        consume(TokenType::Semicolon, "Expected ';' after break");
        return std::make_unique<BreakStmt>(loc);
    }

    if (match(TokenType::KwContinue)) {
        SourceLoc loc = previous().loc;
        consume(TokenType::Semicolon, "Expected ';' after continue");
        return std::make_unique<ContinueStmt>(loc);
    }

    if (check(TokenType::LBrace)) {
        return parseBlock();
    }

    std::unique_ptr<Expr> expr = parseExpr();

    if (match(TokenType::Equal)) {
        SourceLoc eqLoc = previous().loc;
        std::unique_ptr<Expr> val = parseExpr();

        const bool isLValue =
            expr &&
            (expr->kind == ExprKind::Symbol ||
             expr->kind == ExprKind::Index ||
             expr->kind == ExprKind::MemberAccess);

        if (isLValue) {
            auto assign = std::make_unique<AssignStmt>(eqLoc, std::move(expr), std::move(val));
            consume(TokenType::Semicolon, "Expected ';' after assignment");
            return assign;
        }

        errorAt(eqLoc, "Invalid assignment target. Only variables, fields, and indices can be assigned.");
        consume(TokenType::Semicolon, "Expected ';' after invalid assignment");
        return expr;
    }

    consume(TokenType::Semicolon, "Expected ';' after expression statement");
    return expr;
}



std::unique_ptr<Expr> Parser::parsePrint() {
    SourceLoc loc = previous().loc;
    std::vector<std::unique_ptr<Expr>> values;
    if (!check(TokenType::Semicolon)) {
        do { values.push_back(parseExpr()); } while (match(TokenType::Comma));
    }
    consume(TokenType::Semicolon, "Expected ';' after print");
    return std::make_unique<PrintStmt>(loc, std::move(values));
}

std::unique_ptr<Expr> Parser::parseExprNoStructInit() {
    ScopedStructInitSuppression guard(*this);
    return parseExpr();
}

std::unique_ptr<Expr> Parser::parseWhile() {
    SourceLoc loc = previous().loc;
    bool hasParen = match(TokenType::LParen);

    std::unique_ptr<Expr> cond = hasParen ? parseExpr() : parseExprNoStructInit();

    if (hasParen) consume(TokenType::RParen, "Expected ')'");
    auto body = parseBlock();
    return std::make_unique<WhileStmt>(loc, std::move(cond), std::move(body));
}


std::unique_ptr<Expr> Parser::parseFor() {
    SourceLoc loc = previous().loc;
    bool hasParen = match(TokenType::LParen);

    std::string iv = consume(TokenType::Identifier, "Expected iterator variable").text.str();
    consume(TokenType::KwIn, "Expected 'in'");

    auto start = parseExpr();
    consume(TokenType::DotDot, "Expected '..'");

    std::unique_ptr<Expr> end = hasParen ? parseExpr() : parseExprNoStructInit();

    if (hasParen) consume(TokenType::RParen, "Expected ')'");
    auto body = parseBlock();
    return std::make_unique<ForStmt>(loc, iv, std::move(start), std::move(end), std::move(body));
}


std::unique_ptr<Expr> Parser::parseIter() {
    SourceLoc loc = previous().loc;
    bool hasParen = match(TokenType::LParen);

    std::string iv = consume(TokenType::Identifier, "Expected iterator variable").text.str();
    consume(TokenType::KwIn, "Expected 'in'");

    std::unique_ptr<Expr> collection = hasParen ? parseExpr() : parseExprNoStructInit();

    if (hasParen) consume(TokenType::RParen, "Expected ')'");
    auto body = parseBlock();
    return std::make_unique<IterStmt>(loc, iv, std::move(collection), std::move(body));
}


std::unique_ptr<Expr> Parser::parseTupleOrGroup() {
    // [CRITICAL FIX] Consume the '(' that parsePrimary peeked at
    consume(TokenType::LParen, "Expected '('");
    SourceLoc loc = previous().loc; 
    
    // [FIX] Handle Unit/Empty Tuple: "()"
    if (match(TokenType::RParen)) {
        // Return an empty tuple (Unit)
        return std::make_unique<TupleExpr>(loc, std::vector<std::unique_ptr<Expr>>{});
    }

    // Parse first element
    auto first = parseExpr();
    
    // If followed by comma, it's a tuple: (1, 2)
    if (match(TokenType::Comma)) {
        std::vector<std::unique_ptr<Expr>> elements;
        elements.push_back(std::move(first));
        
        // Loop for remaining elements
        do {
            // [OPTIONAL] Handle trailing comma like (1,) by checking if RParen is next
            if (check(TokenType::RParen)) break; 
            
            elements.push_back(parseExpr());
        } while (match(TokenType::Comma));
        
        consume(TokenType::RParen, "Expected ')' after tuple literal");
        return std::make_unique<TupleExpr>(loc, std::move(elements));
    }
    
    // Otherwise, it's just a grouped expression: (a + b)
    consume(TokenType::RParen, "Expected ')' after expression");
    return first;
}

// =============================================================================
// Parse Variable Declaration (let)
// Syntax: let x = 1; OR let (x, y) = tuple;
// =============================================================================
std::unique_ptr<Expr> Parser::parseLet() {
    SourceLoc loc = previous().loc;
    
    std::vector<std::string> names;

    // Case 1: Tuple Destructuring "let (x, y) = ..."
    if (match(TokenType::LParen)) {
        do {
            if (check(TokenType::Identifier)) {
                names.push_back(advance().text.str());
            } else {
                errorAt(peek().loc, "Expect identifier in tuple destructuring");
            }
        } while (match(TokenType::Comma));
        
        consume(TokenType::RParen, "Expect ')' after destructuring list");
    } 
    // Case 2: Single Variable "let x = ..."
    else {
        Token name = consume(TokenType::Identifier, "Expected variable name");
        names.push_back(name.text.str());
    }

    // Optional Type Annotation ": i32"
    std::optional<Type> typeAnnotation;
    if (match(TokenType::Colon)) {
        typeAnnotation = parseType();
    }
    
    consume(TokenType::Equal, "Expected '=' after variable declaration");
    std::unique_ptr<Expr> init = parseExpr();
    consume(TokenType::Semicolon, "Expected ';' after variable declaration");
    
    // [FIX] Use VarDecl (was LetExpr)
    return std::make_unique<VarDecl>(loc, std::move(names), std::move(typeAnnotation), std::move(init));
}

// =============================================================================
// Parse Return Statement
// Syntax: return value; OR return;
// =============================================================================
std::unique_ptr<Expr> Parser::parseReturn() {
    SourceLoc loc = previous().loc;
    std::unique_ptr<Expr> val = nullptr;
    
    // If next token isn't semicolon, parse return value
    if (!check(TokenType::Semicolon)) {
        val = parseExpr();
    }
    
    consume(TokenType::Semicolon, "Expected ';' after return");
    
    // [FIX] Use ReturnStmt (was ReturnExpr)
    return std::make_unique<ReturnStmt>(loc, std::move(val));
}

std::unique_ptr<Expr> Parser::parseIf() {
    SourceLoc loc = previous().loc;

    bool hasParen = match(TokenType::LParen);
    std::unique_ptr<Expr> cond = hasParen ? parseExpr() : parseExprNoStructInit();
    if (hasParen) consume(TokenType::RParen, "Expected ')'");

    std::unique_ptr<Expr> thenBranch;
    if (check(TokenType::LBrace)) {
        thenBranch = parseBlock();
    } else {
        thenBranch = parseExpr();
    }

    std::unique_ptr<Expr> elseBranch = nullptr;
    if (match(TokenType::KwElse)) {
        if (check(TokenType::KwIf)) {
            elseBranch = parseStmt();
        } else if (check(TokenType::LBrace)) {
            elseBranch = parseBlock();
        } else {
            elseBranch = parseExpr();
        }
    }

    return std::make_unique<IfStmt>(loc, std::move(cond), std::move(thenBranch), std::move(elseBranch));
}

std::unique_ptr<Expr> Parser::parseBlock() {
    SourceLoc loc = peek().loc;
    consume(TokenType::LBrace, "Expected '{'");
    std::vector<std::unique_ptr<Expr>> stmts;
    size_t lastPos = -1;

    while (!check(TokenType::RBrace) && !isAtEnd()) {
        watchdog("parseBlock", lastPos);
        auto stmt = parseStmt();
        if (stmt) stmts.push_back(std::move(stmt));
        if (panicMode) synchronize();
    }
    consume(TokenType::RBrace, "Expected '}'");
    return std::make_unique<BlockExpr>(loc, std::move(stmts));
}

std::unique_ptr<Expr> Parser::parseParLoop() {
    SourceLoc loc = previous().loc;

    std::vector<std::string> iterVars;
    if (match(TokenType::LParen)) {
        do {
            Token t = consume(TokenType::Identifier, "Expected iterator name in tuple");
            iterVars.push_back(t.text.str());
        } while (match(TokenType::Comma));
        consume(TokenType::RParen, "Expected ')' after iterator tuple");
    } else {
        Token t = consume(TokenType::Identifier, "Expected iterator name");
        iterVars.push_back(t.text.str());
    }

    consume(TokenType::KwIn, "Expected 'in' after parallel iterators");

    auto lhs = parseExprNoStructInit();

    std::unique_ptr<Expr> domainExpr;
    ParLoop::DomainKind domainKind;

    if (match(TokenType::DotDot)) {
        auto rhs = parseExprNoStructInit();
        domainExpr = std::make_unique<RangeExpr>(lhs->loc, std::move(lhs), std::move(rhs));
        domainKind = ParLoop::DomainKind::Range;
    } else {
        bool isDims = false;
        if (auto* call = llvm::dyn_cast<CallExpr>(lhs.get())) {
            if (auto* sym = llvm::dyn_cast<SymbolExpr>(call->callee.get())) {
                if (sym->name == "dims") isDims = true;
            }
        }

        domainKind = isDims ? ParLoop::DomainKind::DimsCall : ParLoop::DomainKind::LenSugar;
        domainExpr = std::move(lhs);
    }

    std::vector<std::unique_ptr<Expr>> hints;
    while (check(TokenType::Identifier) && peek().text == "block") {
        advance();
        consume(TokenType::LParen, "Expected '(' after block hint");

        do {
            hints.push_back(parseExpr());
        } while (match(TokenType::Comma));

        consume(TokenType::RParen, "Expected ')' after block hint arguments");
    }

    auto bodyRaw = parseBlock();
    std::unique_ptr<BlockExpr> body(static_cast<BlockExpr*>(bodyRaw.release()));

    return std::make_unique<ParLoop>(
        loc,
        std::move(iterVars),
        ParLoop::Domain(domainKind, std::move(domainExpr)),
        std::move(hints),
        std::move(body));
}

std::unique_ptr<ImportDecl> Parser::parseImport() {
    SourceLoc loc = previous().loc;
    
    if (!match(TokenType::String)) {
        errorAt(peek().loc, "Expected string literal path after 'import'");
        return nullptr;
    }
    
    std::string rawPath = previous().text.str();
    // Strip quotes "..."
    if (rawPath.size() >= 2) rawPath = rawPath.substr(1, rawPath.size() - 2);
    
    std::string alias;

    if (match(TokenType::KwAs)) {
        if (!match(TokenType::Identifier)) {
            errorAt(peek().loc, "Expected identifier after 'as'");
            return nullptr;
        }
        alias = previous().text.str();
    } else {
        // Implicit Alias: "math/vectors.ark" -> "vectors"
        llvm::StringRef p(rawPath);
        alias = llvm::sys::path::stem(p).str();
        if (alias.empty()) alias = "unknown";
    }

    consume(TokenType::Semi, "Expected ';' after import declaration");
    
    // 1. Create the AST Node
    auto importNode = std::make_unique<ImportDecl>(loc, rawPath, alias);

    // =========================================================
    // [CRITICAL] Recursive Import Parsing
    // =========================================================
    
    // 2. Resolve File Path (Relative to current file)
    llvm::SmallString<128> currentPath(loc.file);
    llvm::sys::path::remove_filename(currentPath);
    llvm::sys::path::append(currentPath, rawPath);

    // 3. Read File
    auto bufferOrErr = llvm::MemoryBuffer::getFile(currentPath);
    if (!bufferOrErr) {
        // [FIX] Bubble up the actual system error (e.g. "No such file or directory")
        std::error_code ec = bufferOrErr.getError();
        
        errorAt(loc, "Could not open imported file '" + std::string(currentPath.str()) + "': " + ec.message());
        
        // Return the node anyway so the AST remains valid structure-wise, 
        // but the compiler knows an error occurred via errorAt.
        return importNode; 
    }

    std::string source = bufferOrErr.get()->getBuffer().str();

    // 4. Create Sub-Parser
    // We create a new Lexer/Parser instance specifically for the imported file.
    Lexer subLexer(source);
    TokenStream subTokens = subLexer.tokenize();
    
    Parser subParser(std::move(subTokens));
    
    // 5. Parse recursively!
    std::unique_ptr<Module> importedMod = subParser.parseModule();

    // 6. Handle Errors in imported file
    if (subParser.hasErrors()) {
        errorAt(loc, "Errors in imported module '" + rawPath + "'");
        for (const auto& err : subParser.getErrors()) {
            // Forward errors to the main parser's list (or just print them)
            // Using llvm::errs() directly ensures they are seen immediately.
            llvm::errs() << "\t-> " << err << "\n";
        }
    } else {
        // [SUCCESS] Attach the parsed AST to the ImportDecl
        importNode->importedModule = std::move(importedMod);
    }

    return importNode;
}

// -----------------------------------------------------------------------------
// Expressions
// -----------------------------------------------------------------------------

std::unique_ptr<Expr> Parser::parseExpr() {
    return parseLaunch();
}

std::unique_ptr<Expr> Parser::parseMatch() {
    SourceLoc loc = previous().loc;
    consume(TokenType::LParen, "Expected '(' after match");
    auto target = parseExpr();
    consume(TokenType::RParen, "Expected ')' after match target");
    consume(TokenType::LBrace, "Expected '{'");
    std::vector<Case> cases;
    size_t lastPos = -1;

    while (!check(TokenType::RBrace) && !isAtEnd()) {
        watchdog("parseMatch", lastPos);
        Pattern pat;
        if (match(TokenType::KwDefault)) {
            pat.isDefault = true;
        } else {
            consume(TokenType::KwCase, "Expected 'case' or 'default'");
            std::string schemaName = consume(TokenType::Identifier, "Expected Enum name").text.str();
            consume(TokenType::Dot, "Expected '.'");
            std::string variantName = consume(TokenType::Identifier, "Expected Variant name").text.str();
            pat.schemaName = schemaName;
            pat.variantName = variantName;
            if (match(TokenType::LParen)) {
                if (!check(TokenType::RParen)) {
                    do {
                        pat.bindings.push_back(consume(TokenType::Identifier, "Expected bind variable").text.str());
                    } while (match(TokenType::Comma));
                }
                consume(TokenType::RParen, "Expected ')'");
            }
        }
        consume(TokenType::FatArrow, "Expected '=>'");
        auto body = parseBlock();
        cases.push_back({pat, std::move(body)});
    }
    consume(TokenType::RBrace, "Expected '}'");
    return std::make_unique<MatchStmt>(loc, std::move(target), std::move(cases));
}


std::unique_ptr<Expr> Parser::parseEquality() {
    std::unique_ptr<Expr> lhs = parseComparison();

    while (match(TokenType::EqualEqual) || match(TokenType::BangEqual)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> rhs = parseComparison();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }

    return lhs;
}

std::unique_ptr<Expr> Parser::parseLogicAnd() {
    std::unique_ptr<Expr> lhs = parseEquality();

    while (match(TokenType::AmpAmp)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> rhs = parseEquality();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }

    return lhs;
}

std::unique_ptr<Expr> Parser::parseLogicOr() {
    std::unique_ptr<Expr> lhs = parseLogicAnd();

    while (match(TokenType::PipePipe)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> rhs = parseLogicAnd();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }

    return lhs;
}



std::unique_ptr<Expr> Parser::parseLaunch() {
    std::unique_ptr<Expr> lhs = parseLogicOr();

    if (!match(TokenType::ArrowL)) {
        return lhs;
    }

    SourceLoc loc = previous().loc;

    if (!lhs || lhs->kind != ExprKind::Symbol) {
        errorAt(loc, "Launch destination must be a variable symbol");
        return lhs;
    }

    std::string dest = static_cast<SymbolExpr*>(lhs.get())->name;

    std::string kernel = consume(TokenType::Identifier, "Expected kernel name").text.str();
    while (match(TokenType::Dot)) {
        kernel += ".";
        kernel += consume(TokenType::Identifier, "Expected identifier after '.' in kernel name").text.str();
    }

    auto args = parseArgumentList();

    std::string tokenName;
    if (match(TokenType::KwAs)) {
        tokenName = consume(TokenType::Identifier, "Expected token name after 'as'").text.str();
    }

    if (match(TokenType::At)) {
        consume(TokenType::KwRuntime, "Expected 'runtime' after '@' in launch annotation");
        std::string presetName = consume(TokenType::Identifier, "Expected runtime preset name after '@runtime'").text.str();
        return std::make_unique<LaunchExpr>(
            loc,
            dest,
            kernel,
            std::move(args),
            tokenName,
            Space::runtimePreset(presetName)
        );
    }

    return std::make_unique<LaunchExpr>(loc, dest, kernel, std::move(args), tokenName);
}



std::unique_ptr<Expr> Parser::parseComparison() {
    std::unique_ptr<Expr> lhs = parseTerm();

    while (match(TokenType::Less) ||
           match(TokenType::Greater) ||
           match(TokenType::LessEqual) ||
           match(TokenType::GreaterEqual)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> rhs = parseTerm();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }

    return lhs;
}

std::unique_ptr<Expr> Parser::parseTerm() {
    std::unique_ptr<Expr> lhs = parseFactor();

    while (match(TokenType::Plus) || match(TokenType::Minus)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> rhs = parseFactor();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }

    return lhs;
}

std::unique_ptr<Expr> Parser::parseUnary() {
    if (match(TokenType::Not) || match(TokenType::Minus) || match(TokenType::Plus)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> operand = parseUnary();
        return std::make_unique<UnaryExpr>(loc, op, std::move(operand));
    }

    return parseCall();
}


std::unique_ptr<Expr> Parser::parseCall() {
    std::unique_ptr<Expr> expr = parsePrimary();

    while (true) {
        if (check(TokenType::LParen)) {
            SourceLoc loc = expr ? expr->loc : peek().loc;
            auto args = parseArgumentList();
            expr = std::make_unique<CallExpr>(loc, std::move(expr), std::move(args));
            continue;
        }

        if (match(TokenType::LBracket)) {
            SourceLoc loc = previous().loc;
            std::unique_ptr<Expr> index = parseExpr();
            consume(TokenType::RBracket, "Expected ']' after index expression");
            expr = std::make_unique<IndexExpr>(loc, std::move(expr), std::move(index));
            continue;
        }

        if (match(TokenType::Dot)) {
            SourceLoc loc = previous().loc;
            std::string member = consume(TokenType::Identifier, "Expected member name after '.'").text.str();

            if (check(TokenType::LParen)) {
                auto args = parseArgumentList();
                expr = std::make_unique<MemberCallNode>(loc, std::move(expr), member, std::move(args));
            } else {
                expr = std::make_unique<MemberExpr>(loc, std::move(expr), member);
            }
            continue;
        }

        break;
    }

    return expr;
}


// =============================================================================
// Runtime Literal Parsing
// Syntax:
//   runtime{
//       target: expr,
//       endpoint: expr,
//       token: expr,
//       timeout_ms: expr,
//       max_burn_usd: expr
//   }
// =============================================================================
std::unique_ptr<Expr> Parser::parseRuntimeLiteral() {
    SourceLoc loc = previous().loc;

    consume(TokenType::LBrace, "Expected '{' after 'runtime'");

    std::vector<RuntimeFieldInit> fields;

    if (!check(TokenType::RBrace)) {
        do {
            std::string fieldName = consume(TokenType::Identifier, "Expected runtime field name").text.str();
            consume(TokenType::Colon, "Expected ':' after runtime field name");
            fields.emplace_back(fieldName, parseExpr());
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RBrace, "Expected '}' after runtime literal");
    return std::make_unique<RuntimeLiteralExpr>(loc, std::move(fields));
}


std::unique_ptr<Expr> Parser::parseFactor() {
    std::unique_ptr<Expr> lhs = parseUnary();

    while (match(TokenType::Star) || match(TokenType::Slash) || match(TokenType::Percent)) {
        SourceLoc loc = previous().loc;
        std::string op = previous().text.str();
        std::unique_ptr<Expr> rhs = parseUnary();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }

    return lhs;
}

std::unique_ptr<Expr> Parser::parseArrayLiteral() {
    SourceLoc loc = peek().loc;
    
    // [CRITICAL] You must consume the opening bracket here!
    // parsePrimary() used check(), so the token is still on the stream.
    consume(TokenType::LBracket, "Expected '['");

    std::vector<std::unique_ptr<Expr>> elements;
    
    // Check for empty array '[]' to avoid parsing ']' as an expression
    if (!check(TokenType::RBracket)) {
        do {
            elements.push_back(parseExpr());
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RBracket, "Expected ']' after array elements");
    
    return std::make_unique<ArrayLiteral>(loc, std::move(elements));
}

std::unique_ptr<Expr> Parser::parseLambda() {
    SourceLoc loc = peek().loc;
    
    // [FIX] Use LParen
    consume(TokenType::LParen, "Expect '(' to start lambda params");

    std::vector<LambdaExpr::Param> params;
    
    // [FIX] Use RParen
    if (!check(TokenType::RParen)) {
        do {
            // [FIX] Use .text.str() to convert StringRef to std::string
            std::string name = consume(TokenType::Identifier, "Expect parameter name").text.str();
            
            consume(TokenType::Colon, "Expect ':' after parameter name");
            
            Type type = parseType();
            params.push_back({name, type});
        } while (match(TokenType::Comma));
    }

    // [FIX] Use RParen
    consume(TokenType::RParen, "Expect ')' after lambda params");
    
    // Explicit return type logic
    Type returnType = {Type::Void};
    bool hasExplicitReturn = false; // [NEW] Track explicit vs implicit

    if (match(TokenType::Arrow)) { 
        returnType = parseType();
        hasExplicitReturn = true;
    }

    consume(TokenType::FatArrow, "Expect '=>' before lambda body");

    std::unique_ptr<Expr> body;
    // [FIX] Use LBrace
    if (check(TokenType::LBrace)) {
        body = parseBlock();
    } else {
        body = parseExpr();
    }

    // [FIX] Pass 5 arguments to match new AST definition
    return std::make_unique<LambdaExpr>(loc, std::move(params), returnType, std::move(body), hasExplicitReturn);
}

// =============================================================================
// Primary Expressions
// Handles:
// - allocof<T>(...)
// - runtime{...}
// - await token
// - literals
// - grouped / tuple
// - arrays
// - lambdas
// - schema init
// - generic calls
// =============================================================================
std::unique_ptr<Expr> Parser::parsePrimary() {
    if (match(TokenType::KwAllocof)) {
        SourceLoc loc = previous().loc;

        consume(TokenType::Less, "Expected '<' after allocof");
        Type t = parseType();
        consume(TokenType::Greater, "Expected '>' after allocof type");

        auto args = parseArgumentList();

        std::optional<Space> placement;

        if (match(TokenType::At)) {
            if (match(TokenType::KwGpu)) {
                if (match(TokenType::Colon)) {
                    if (match(TokenType::Integer)) {
                        placement = Space::gpuDevice(std::stoi(previous().text.str()));
                    } else if (match(TokenType::String)) {
                        placement = Space::gpuRouteLiteral(stripQuotedToken(previous().text));
                    } else if (match(TokenType::Identifier)) {
                        placement = Space::gpuRouteSymbol(previous().text.str());
                    } else {
                        errorAt(peek().loc, "Expected GPU device id, route string, or route symbol after '@gpu:'");
                    }
                } else {
                    Space sp;
                    sp.kind = Space::GPU;
                    sp.addressKind = Space::AddressKind::Default;
                    placement = sp;
                }
            } else if (match(TokenType::KwRuntime)) {
                std::string presetName = consume(TokenType::Identifier, "Expected runtime preset name after '@runtime'").text.str();
                placement = Space::runtimePreset(presetName);
            } else if (match(TokenType::KwHost) || match(TokenType::KwRam)) {
                placement = Space::host();
            } else {
                errorAt(peek().loc, "Expected placement target after '@'");
            }
        } else if (!t.space.isDefault()) {
            placement = t.space;
        }

        std::unique_ptr<Expr> alloc;
        if (placement.has_value()) {
            alloc = std::make_unique<AllocExpr>(loc, t, *placement);
        } else {
            alloc = std::make_unique<AllocExpr>(loc, t, "");
        }

        return std::make_unique<CallExpr>(loc, std::move(alloc), std::move(args));
    }

    if (match(TokenType::KwRuntime)) {
        return parseRuntimeLiteral();
    }

    if (match(TokenType::KwAwait)) {
        SourceLoc loc = previous().loc;
        std::string tokenName = consume(TokenType::Identifier, "Expected token name after 'await'").text.str();
        return std::make_unique<AwaitExpr>(loc, tokenName);
    }

    if (match(TokenType::KwIf)) {
        return parseIf();
    }

    if (match(TokenType::KwMatch)) {
        return parseMatch();
    }

    if (match(TokenType::Integer)) {
        SourceLoc loc = previous().loc;
        Type t;
        t.kind = Type::I32;
        return std::make_unique<LiteralExpr>(loc, previous().text.str(), t);
    }

    if (match(TokenType::Float)) {
        SourceLoc loc = previous().loc;
        Type t;
        t.kind = Type::F64;
        return std::make_unique<LiteralExpr>(loc, previous().text.str(), t);
    }

    if (match(TokenType::KwTrue) || match(TokenType::KwFalse)) {
        SourceLoc loc = previous().loc;
        Type t;
        t.kind = Type::Bool;
        return std::make_unique<LiteralExpr>(loc, previous().text.str(), t);
    }

    if (match(TokenType::KwNull)) {
        SourceLoc loc = previous().loc;
        Type t;
        t.kind = Type::Void;
        return std::make_unique<LiteralExpr>(loc, "null", t);
    }

    if (match(TokenType::String)) {
        SourceLoc loc = previous().loc;
        return std::make_unique<StringExpr>(loc, stripQuotedToken(previous().text));
    }

    if (match(TokenType::Char)) {
        SourceLoc loc = previous().loc;
        Type t;
        t.kind = Type::I32;
        return std::make_unique<LiteralExpr>(loc, previous().text.str(), t);
    }

    if (check(TokenType::LBracket)) {
        return parseArrayLiteral();
    }

    if (check(TokenType::LParen)) {
        bool lambdaLike = false;
        int depth = 0;

        for (std::size_t i = current; i < stream.tokens.size(); ++i) {
            const TokenType tk = stream.tokens[i].type;

            if (tk == TokenType::LParen) {
                ++depth;
            } else if (tk == TokenType::RParen) {
                --depth;
                if (depth == 0) {
                    if (i + 1 < stream.tokens.size()) {
                        const TokenType next = stream.tokens[i + 1].type;
                        if (next == TokenType::FatArrow || next == TokenType::Arrow) {
                            lambdaLike = true;
                        }
                    }
                    break;
                }
            }
        }

        if (lambdaLike) {
            return parseLambda();
        }

        return parseTupleOrGroup();
    }

    if (match(TokenType::Identifier) ||
        match(TokenType::KwFs) ||
        match(TokenType::KwNet) ||
        match(TokenType::KwIo) ||
        match(TokenType::KwSys)) {
        SourceLoc loc = previous().loc;
        std::string name = previous().text.str();

        while (check(TokenType::Dot) && peekDistance(1).type == TokenType::Identifier) {
            advance();
            name += ".";
            name += consume(TokenType::Identifier, "Expected identifier after '.'").text.str();
        }

        auto tryParseGenericArgs = [&]() -> std::optional<std::vector<Type>> {
            if (!check(TokenType::Less)) {
                return std::nullopt;
            }

            const std::size_t savedCurrent = current;
            const bool savedPanic = panicMode;
            const std::size_t savedErrorCount = errors.size();

            std::vector<Type> genericArgs;

            match(TokenType::Less);
            do {
                genericArgs.push_back(parseType());
            } while (match(TokenType::Comma));

            const bool closed = match(TokenType::Greater);
            const bool validFollower = check(TokenType::LParen) || check(TokenType::LBrace);

            if (!closed || !validFollower || panicMode) {
                current = savedCurrent;
                panicMode = savedPanic;
                errors.resize(savedErrorCount);
                return std::nullopt;
            }

            return genericArgs;
        };

        std::vector<Type> genericArgs;
        if (auto parsed = tryParseGenericArgs()) {
            genericArgs = std::move(*parsed);
        }

        bool isStructInit = false;
        if (!suppressStructInitInPrimary && check(TokenType::LBrace)) {
            if (peekDistance(1).type == TokenType::RBrace) {
                isStructInit = true;
            } else if (peekDistance(1).type == TokenType::Identifier) {
                const TokenType next2 = peekDistance(2).type;
                if (next2 == TokenType::Colon || next2 == TokenType::Equal) {
                    isStructInit = true;
                }
            }
        }

        if (isStructInit) {
            consume(TokenType::LBrace, "Expected '{'");

            std::vector<SchemaInitField> fields;
            if (!check(TokenType::RBrace)) {
                do {
                    std::string fieldName = consume(TokenType::Identifier, "Expected field name").text.str();

                    if (!(match(TokenType::Colon) || match(TokenType::Equal))) {
                        errorAt(peek().loc, "Expected ':' or '=' in schema initialization");
                        break;
                    }

                    fields.push_back({fieldName, parseExpr()});
                } while (match(TokenType::Comma));
            }

            consume(TokenType::RBrace, "Expected '}' after schema initialization");
            return std::make_unique<SchemaExpr>(loc, name, std::move(genericArgs), std::move(fields));
        }

        if (!genericArgs.empty() && check(TokenType::LParen)) {
            auto args = parseArgumentList();
            auto callee = std::make_unique<SymbolExpr>(loc, name);
            return std::make_unique<CallExpr>(loc, std::move(callee), std::move(args), std::move(genericArgs));
        }

        return std::make_unique<SymbolExpr>(loc, name);
    }

    errorAt(peek().loc, "Expected expression");
    SourceLoc loc = peek().loc;
    if (!isAtEnd()) advance();

    Type t;
    t.kind = Type::Void;
    return std::make_unique<LiteralExpr>(loc, "0", t);
}


// [FIXED] Updated to handle Move-Only RecordFields
std::vector<RecordField> Parser::parseRecordFields() {
    std::vector<RecordField> fields;
    if (!check(TokenType::RBrace)) {
        do {
            std::string name = consume(TokenType::Identifier, "Expected field name").text.str();
            consume(TokenType::Colon, "Expected ':' in record field");
            Type type = parseType();
            
            // Parse Optional Default Value
            std::unique_ptr<Expr> defVal = nullptr;
            if (match(TokenType::Equal)) {
                defVal = parseExpr();
            }

            // Construct in-place (Move semantics)
            fields.push_back({name, type, std::move(defVal)});
            
        } while (match(TokenType::Comma));
    }
    consume(TokenType::RBrace, "Expected '}'");
    return fields;
}

// [FIXED] Added std::move() for vectors containing unique_ptr
std::unique_ptr<SchemaDecl> Parser::parseSchema(bool isSingleton) {
    SourceLoc loc = previous().loc;
    
    if (!match(TokenType::Identifier)) {
        errorAt(peek().loc, "Expected schema name");
        return nullptr;
    }
    std::string name = previous().text.str();

    std::vector<std::string> genericParams;
    if (match(TokenType::Less)) {
        do {
            if (!match(TokenType::Identifier)) {
                errorAt(peek().loc, "Expect generic parameter name");
                return nullptr;
            }
            genericParams.push_back(previous().text.str());
        } while (match(TokenType::Comma));
        consume(TokenType::Greater, "Expect '>' after generic parameters");
    }

    consume(TokenType::LBrace, "Expected '{'");

    bool hasMeta = false;
    if (match(TokenType::KwMeta)) {
        parseMetaBlock(hasMeta);
        match(TokenType::Comma); 
    }

    // Empty Schema
    if (match(TokenType::RBrace)) {
        auto s = std::make_unique<SchemaDecl>(name, std::vector<RecordField>{}, loc, hasMeta);
        s->genericParams = std::move(genericParams);
        s->isSingleton = isSingleton;
        return s;
    }

    // Check if it's a Record (fields) or Enum (variants)
    // Records start with "name :"
    bool isRecord = false;
    if (check(TokenType::Identifier) && peekNext().type == TokenType::Colon) {
        isRecord = true;
    }

    if (isRecord) {
        std::vector<RecordField> fields;
        do {
            if (check(TokenType::RBrace)) break;
            std::string fName = consume(TokenType::Identifier, "Expected field name").text.str();
            consume(TokenType::Colon, "Expected ':'");
            Type type = parseType();
            
            std::unique_ptr<Expr> defVal = nullptr;
            if (match(TokenType::Equal)) { 
                defVal = parseExpr();
            }

            // Move unique_ptr correctly
            fields.push_back({fName, type, std::move(defVal)});

        } while (match(TokenType::Comma));
        consume(TokenType::RBrace, "Expected '}'");
        
        auto s = std::make_unique<SchemaDecl>(name, std::move(fields), loc, hasMeta);
        s->genericParams = std::move(genericParams);
        s->isSingleton = isSingleton;
        return s;
    } else {
        // Enum Parsing
        if (isSingleton) {
             errorAt(loc, "Singleton Enums are not supported yet.");
        }

        std::vector<EnumVariant> variants;
        do {
            if (check(TokenType::RBrace)) break;
            std::string vName = consume(TokenType::Identifier, "Expected variant name").text.str();
            EnumVariant v;
            v.name = vName;
            v.kind = EnumVariant::Unit;
            
            if (match(TokenType::LParen)) {
                v.kind = EnumVariant::Tuple;
                do { v.tuplePayload.push_back(parseType()); } while (match(TokenType::Comma));
                consume(TokenType::RParen, "Expected ')'");
            } 
            else if (match(TokenType::LBrace)) {
                v.kind = EnumVariant::Struct;
                v.structPayload = parseRecordFields(); 
            }
            
            variants.push_back(std::move(v));

        } while (match(TokenType::Comma));
        consume(TokenType::RBrace, "Expected '}'");
        
        auto s = std::make_unique<SchemaDecl>(name, std::move(variants), loc, hasMeta);
        s->genericParams = std::move(genericParams);
        return s;
    }
}

std::vector<CallArg> Parser::parseArgumentList() {
    std::vector<CallArg> args;
    consume(TokenType::LParen, "Expected '('");
    if (!check(TokenType::RParen)) {
        do {
            std::string name = "";
            if (check(TokenType::Identifier) && peekNext().type == TokenType::Equal) {
                name = advance().text.str();
                advance(); 
            }
            args.emplace_back(name, parseExpr());
        } while (match(TokenType::Comma));
    }
    consume(TokenType::RParen, "Expected ')'");
    return args;
}

void Parser::parseMetaBlock(bool& outMeta) {
    outMeta = true;
    consume(TokenType::LBrace, "Expected '{' after meta");
    int depth = 1;
    size_t lastPos = -1;
    while (depth > 0 && !isAtEnd()) {
        watchdog("parseMetaBlock", lastPos);
        if (check(TokenType::LBrace)) depth++;
        if (check(TokenType::RBrace)) depth--;
        advance();
    }
    if (isAtEnd()) errorAt(peek().loc, "Unterminated meta block");
}


const Token& Parser::peekDistance(int distance) const {
    // Ensure we don't read past the end of the token stream.
    // Assuming 'stream.tokens' is the vector holding your tokens.
    if (current + distance >= stream.tokens.size()) {
        return stream.tokens.back(); // Return the EOF token
    }
    return stream.tokens[current + distance];
}


// [NEW] Dispatcher for global declarations
// Handles: 'singleton schema', 'schema singleton', 'schema', 'fn', 'import'
std::unique_ptr<Decl> Parser::parseTopLevel() {
    // 1. 'singleton schema ...'
    if (match(TokenType::KwSingleton)) {
        if (match(TokenType::KwSchema)) {
            return parseSchema(true); // isSingleton = true
        }
        errorAt(peek().loc, "Expected 'schema' after 'singleton'");
        return nullptr;
    }

    // 2. 'schema ...' cases
    if (check(TokenType::KwSchema)) {
        // Check for 'schema singleton ...' (Lookahead)
        if (peekNext().type == TokenType::KwSingleton) {
            consume(TokenType::KwSchema, "");
            consume(TokenType::KwSingleton, "");
            return parseSchema(true); // isSingleton = true
        }
        
        // Standard 'schema ...'
        consume(TokenType::KwSchema, "");
        return parseSchema(false); // isSingleton = false
    }

    // 3. Imports
    if (check(TokenType::KwImport)) {
        if (match(TokenType::KwImport)) return parseImport();
    } 
    
    // 4. Functions
    if (match(TokenType::KwFn)) {
        return parseFunction();
    }

    errorAt(peek().loc, "Expected import, function, schema, or singleton declaration");
    advance();
    return nullptr;
}

// =============================================================================
// Type Parsing
// Supports:
// - primitives
// - tuples
// - containers
// - user types / generics
// - optional shape suffixes
// - placement suffixes:
//     @gpu
//     @gpu:0
//     @gpu:"route"
//     @gpu:route
//     @runtime preset
//     @host / @ram
// =============================================================================
Type Parser::parseType() {
    if (match(TokenType::LParen)) {
        if (match(TokenType::RParen)) {
            return Type{Type::Void};
        }

        Type first = parseType();

        if (match(TokenType::Comma)) {
            Type tupleType;
            tupleType.kind = Type::Tuple;
            tupleType.subtypes.push_back(std::move(first));

            do {
                if (check(TokenType::RParen)) break;
                tupleType.subtypes.push_back(parseType());
            } while (match(TokenType::Comma));

            consume(TokenType::RParen, "Expected ')' after tuple type");
            return tupleType;
        }

        consume(TokenType::RParen, "Expected ')' after parenthesized type");
        return first;
    }

    Type t;
    bool parsedBase = true;

    if (match(TokenType::KwVoid))      t.kind = Type::Void;
    else if (match(TokenType::KwU8))   t.kind = Type::U8;
    else if (match(TokenType::KwU16))  t.kind = Type::U16;
    else if (match(TokenType::KwU32))  t.kind = Type::U32;
    else if (match(TokenType::KwU64))  t.kind = Type::U64;
    else if (match(TokenType::KwI8))   t.kind = Type::I8;
    else if (match(TokenType::KwI16))  t.kind = Type::I16;
    else if (match(TokenType::KwI32))  t.kind = Type::I32;
    else if (match(TokenType::KwI64))  t.kind = Type::I64;
    else if (match(TokenType::KwF32))  t.kind = Type::F32;
    else if (match(TokenType::KwF64))  t.kind = Type::F64;
    else if (match(TokenType::KwBool)) t.kind = Type::Bool;
    else if (match(TokenType::KwStr))  t.kind = Type::Str;
    else if (match(TokenType::KwVec))  t.kind = Type::Vec;
    else if (match(TokenType::KwSlice)) t.kind = Type::Slice;
    else if (match(TokenType::KwTensor)) t.kind = Type::Tensor;
    else if (match(TokenType::Identifier)) {
        t.kind = Type::Schema;
        t.schemaName = previous().text.str();

        while (match(TokenType::Dot)) {
            std::string part = consume(TokenType::Identifier, "Expected identifier after '.' in type name").text.str();
            t.schemaName += ".";
            t.schemaName += part;
        }
    } else {
        parsedBase = false;
        errorAt(peek().loc, "Expected type");
        return Type{Type::Void};
    }

    if (parsedBase && match(TokenType::Less)) {
        do {
            t.genericArgs.push_back(parseType());
        } while (match(TokenType::Comma));
        consume(TokenType::Greater, "Expected '>' after generic arguments");

        if (t.kind == Type::Schema) {
            t.kind = Type::Generic;
        }
    }

    if (match(TokenType::LBracket)) {
        do {
            if (match(TokenType::Question)) {
                t.shape.push_back("?");
            } else if (match(TokenType::Integer) || match(TokenType::Identifier)) {
                t.shape.push_back(previous().text.str());
            } else {
                errorAt(peek().loc, "Expected shape dimension");
                break;
            }
        } while (match(TokenType::Comma));

        consume(TokenType::RBracket, "Expected ']' after shape suffix");
    }

    if (match(TokenType::At)) {
        if (match(TokenType::KwGpu)) {
            if (match(TokenType::Colon)) {
                if (match(TokenType::Integer)) {
                    t.space = Space::gpuDevice(std::stoi(previous().text.str()));
                } else if (match(TokenType::String)) {
                    t.space = Space::gpuRouteLiteral(stripQuotedToken(previous().text));
                } else if (match(TokenType::Identifier)) {
                    t.space = Space::gpuRouteSymbol(previous().text.str());
                } else {
                    errorAt(peek().loc, "Expected GPU device id, route string, or route symbol after '@gpu:'");
                    t.space.kind = Space::GPU;
                    t.space.addressKind = Space::AddressKind::Default;
                }
            } else {
                t.space.kind = Space::GPU;
                t.space.addressKind = Space::AddressKind::Default;
            }
        } else if (match(TokenType::KwHost) || match(TokenType::KwRam)) {
            t.space = Space::host();
        } else if (match(TokenType::KwRuntime)) {
            std::string presetName = consume(TokenType::Identifier, "Expected runtime preset name after '@runtime'").text.str();
            t.space = Space::runtimePreset(presetName);
        } else {
            errorAt(peek().loc, "Expected placement suffix after '@'");
        }
    }

    return t;
}



} // namespace arklang