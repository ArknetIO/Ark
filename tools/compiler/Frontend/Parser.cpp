#include "Frontend/Parser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h" // [NEW] Required for implicit alias derivation
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>

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

std::unique_ptr<Expr> Parser::parseStmt() {
    // 1. Statements that don't require trailing semicolons
    if (match(TokenType::KwLet))    return parseLet();
    if (match(TokenType::KwReturn)) return parseReturn();
    if (match(TokenType::KwPar))    return parseParLoop();
    
    // Control Flow (Statement Form)
    if (match(TokenType::KwIf))     return parseIf();
    if (match(TokenType::KwWhile))  return parseWhile();
    if (match(TokenType::KwFor))    return parseFor();
    if (match(TokenType::KwIter))   return parseIter();
    if (match(TokenType::KwMatch))  return parseMatch();
    
    // Debug
    if (match(TokenType::KwPrint))  return parsePrint();
    
    // Block Statement
    if (check(TokenType::LBrace))   return parseBlock();

    // 2. Expression Statements (Require Semicolons)
    // We parse the expression first. It might be a function call, 
    // or the Left-Hand Side (LHS) of an assignment.
    std::unique_ptr<Expr> expr = parseExpr();
    
    // 3. Assignment Handling (LHS = RHS)
    if (match(TokenType::Equal)) {
        SourceLoc eqLoc = previous().loc;
        std::unique_ptr<Expr> val = parseExpr();
        
        // Validate L-Value: Can we assign to this expression?
        // Valid targets: Symbols (x), Indexing (arr[i]), Member Access (obj.prop)
        bool isLValue = (expr->kind == ExprKind::Symbol || 
                         expr->kind == ExprKind::Index || 
                         expr->kind == ExprKind::MemberAccess);

        if (isLValue) {
            // [FIXED] Use generic AssignStmt constructor taking (Target, Value)
            auto assign = std::make_unique<AssignStmt>(eqLoc, std::move(expr), std::move(val));
            consume(TokenType::Semicolon, "Expected ';' after assignment");
            return assign;
        } else {
            errorAt(eqLoc, "Invalid assignment target. Only variables, fields, and indices can be assigned.");
            // Error recovery: return the LHS so parsing can continue without crashing
            return expr; 
        }
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

std::unique_ptr<Expr> Parser::parseLaunch() {
    std::unique_ptr<Expr> lhs = parseComparison();
    if (match(TokenType::ArrowL)) {
        SourceLoc loc = previous().loc;
        if (lhs->kind != ExprKind::Symbol) {
            errorAt(loc, "Launch destination must be a variable symbol");
            return lhs;
        }
        std::string dest = static_cast<SymbolExpr*>(lhs.get())->name;
        std::string kernel = consume(TokenType::Identifier, "Expected kernel name").text.str();
        auto args = parseArgumentList(); 
        std::string token = "";
        if (match(TokenType::KwAs)) {
            token = consume(TokenType::Identifier, "Expected token name").text.str();
        }
        return std::make_unique<LaunchExpr>(loc, dest, kernel, std::move(args), token);
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseComparison() {
    std::unique_ptr<Expr> lhs = parseTerm();
    while (match(TokenType::Less) || match(TokenType::Greater) ||
           match(TokenType::LessEqual) || match(TokenType::GreaterEqual) ||
           match(TokenType::EqualEqual) || match(TokenType::BangEqual)) {
        std::string op = previous().text.str();
        SourceLoc loc = previous().loc;
        std::unique_ptr<Expr> rhs = parseTerm();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseTerm() {
    std::unique_ptr<Expr> lhs = parseFactor();
    while (match(TokenType::Plus) || match(TokenType::Minus)) {
        std::string op = previous().text.str();
        SourceLoc loc = previous().loc;
        std::unique_ptr<Expr> rhs = parseFactor();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseFactor() {
    std::unique_ptr<Expr> lhs = parsePrimary();

    while (true) {
        if (match(TokenType::Star) || match(TokenType::Slash) || match(TokenType::Percent)) {
            std::string op = previous().text.str();
            SourceLoc loc = previous().loc;
            std::unique_ptr<Expr> rhs = parsePrimary();
            lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
            continue;
        }
        break;
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

std::unique_ptr<Expr> Parser::parsePrimary() {
    std::unique_ptr<Expr> expr = nullptr;

    if (match(TokenType::KwAllocof)) {
        SourceLoc loc = previous().loc;
        consume(TokenType::Less, "Expected '<' after allocof");
        Type t = parseType();
        consume(TokenType::Greater, "Expected '>' after allocof type");

        auto args = parseArgumentList();
        std::string locStr = "";
        if (t.space.kind == Space::GPU) locStr = "gpu";

        if (match(TokenType::At)) {
            if (match(TokenType::KwGpu)) locStr = "gpu";
            else if (match(TokenType::KwRam)) locStr = "ram";

            if (locStr == "gpu" && match(TokenType::Colon)) {
                Token devId = consume(TokenType::Integer, "Expected GPU ID");
                locStr += ":" + devId.text.str();
            }
        }

        auto allocExpr = std::make_unique<AllocExpr>(loc, t, locStr);
        return std::make_unique<CallExpr>(loc, std::move(allocExpr), std::move(args));
    }

    if (match(TokenType::Identifier) ||
        match(TokenType::KwFs) ||
        match(TokenType::KwNet) ||
        match(TokenType::KwIo) ||
        match(TokenType::KwSys)) {

        SourceLoc loc = previous().loc;
        std::string name = previous().text.str();

        while (check(TokenType::Dot)) {
            if (peekDistance(1).type == TokenType::Identifier) {
                TokenType ctx = peekDistance(2).type;
                if (ctx == TokenType::Dot || ctx == TokenType::LBrace || ctx == TokenType::Less) {
                    consume(TokenType::Dot, "Expected '.'");
                    std::string part = consume(TokenType::Identifier, "Expected identifier").text.str();
                    name += "." + part;
                    continue;
                }
            }
            break;
        }

        std::vector<Type> genArgs;
        if (check(TokenType::Less)) {
            consume(TokenType::Less, "Expected '<'");
            do {
                genArgs.push_back(parseType());
            } while (match(TokenType::Comma));
            consume(TokenType::Greater, "Expected '>'");
        }

        bool isStructInit = false;
        if (!suppressStructInitInPrimary && check(TokenType::LBrace)) {
            if (peekDistance(1).type == TokenType::RBrace) {
                isStructInit = true;
            } else if (peekDistance(1).type == TokenType::Identifier) {
                TokenType t2 = peekDistance(2).type;
                if (t2 == TokenType::Colon || t2 == TokenType::Equal) {
                    isStructInit = true;
                }
            }
        }

        if (isStructInit) {
            consume(TokenType::LBrace, "");
            std::vector<SchemaInitField> fields;
            if (!check(TokenType::RBrace)) {
                do {
                    std::string fName = consume(TokenType::Identifier, "Expected field name").text.str();

                    if (match(TokenType::Colon) || match(TokenType::Equal)) {
                        auto fVal = parseExpr();
                        fields.push_back({fName, std::move(fVal)});
                    } else {
                        errorAt(peek().loc, "Expected ':' or '=' in struct initialization");
                    }
                } while (match(TokenType::Comma));
            }
            consume(TokenType::RBrace, "Expected '}'");
            expr = std::make_unique<SchemaExpr>(loc, name, genArgs, std::move(fields));
        } else {
            if (!genArgs.empty()) {
                if (check(TokenType::LParen)) {
                    auto args = parseArgumentList();
                    auto callee = std::make_unique<SymbolExpr>(loc, name);
                    expr = std::make_unique<CallExpr>(loc, std::move(callee), std::move(args), std::move(genArgs));
                } else {
                    errorAt(loc, "Generic arguments <...> must be followed by call '(...)' or struct init '{...}'");
                    expr = std::make_unique<SymbolExpr>(loc, name);
                }
            } else {
                expr = std::make_unique<SymbolExpr>(loc, name);
            }
        }
    }
    else if (match(TokenType::KwIf)) {
        return parseIf();
    }
    else if (match(TokenType::KwMatch)) {
        return parseMatch();
    }
    else if (match(TokenType::Integer)) {
        std::string txt = previous().text.str();
        Type t = {Type::I32};
        try {
            long long val = std::stoll(txt);
            if (val > 2147483647 || val < -2147483648) t.kind = Type::I64;
        } catch (...) { t.kind = Type::I64; }
        expr = std::make_unique<LiteralExpr>(previous().loc, txt, t);
    }
    else if (match(TokenType::Float)) {
        expr = std::make_unique<LiteralExpr>(previous().loc, previous().text.str(), Type{Type::F32});
    }
    else if (match(TokenType::KwTrue)) {
        expr = std::make_unique<LiteralExpr>(previous().loc, "true", Type{Type::Bool});
    }
    else if (match(TokenType::KwFalse)) {
        expr = std::make_unique<LiteralExpr>(previous().loc, "false", Type{Type::Bool});
    }
    else if (match(TokenType::String)) {
        std::string s = previous().text.str();
        if (s.size() >= 2) s = s.substr(1, s.size() - 2);
        expr = std::make_unique<StringExpr>(previous().loc, s);
    }
    else if (match(TokenType::Char)) {
        std::string s = previous().text.str();
        char c = (s.size() >= 2) ? s[1] : 0;
        expr = std::make_unique<LiteralExpr>(previous().loc, std::to_string((int)c), Type{Type::U8});
    }
    else if (match(TokenType::Minus)) {
        if (match(TokenType::Integer)) {
            std::string val = "-" + previous().text.str();
            Type t = {Type::I32};
            try {
                long long v = std::stoll(val);
                if (v > 2147483647 || v < -2147483648) t.kind = Type::I64;
            } catch (...) { t.kind = Type::I64; }
            expr = std::make_unique<LiteralExpr>(previous().loc, val, t);
        }
        else if (match(TokenType::Float)) {
            std::string val = "-" + previous().text.str();
            expr = std::make_unique<LiteralExpr>(previous().loc, val, Type{Type::F32});
        }
        else {
            errorAt(previous().loc, "Expected number after unary '-'");
            expr = std::make_unique<LiteralExpr>(previous().loc, "0", Type{Type::I32});
        }
    }
    else if (check(TokenType::LBracket)) {
        expr = parseArrayLiteral();
    }
    else if (match(TokenType::KwAwait)) {
        SourceLoc loc = previous().loc;
        std::string t = consume(TokenType::Identifier, "Expected token").text.str();
        expr = std::make_unique<AwaitExpr>(loc, t);
    }
    else if (check(TokenType::LParen)) {
        bool isLambda = false;
        if (peekDistance(1).type == TokenType::RParen && peekDistance(2).type == TokenType::FatArrow) {
            isLambda = true;
        } else if (peekDistance(1).type == TokenType::Identifier && peekDistance(2).type == TokenType::Colon) {
            isLambda = true;
        }

        if (isLambda) expr = parseLambda();
        else expr = parseTupleOrGroup();
    }
    else {
        errorAt(peek().loc, "Expected expression (Found: " + std::string(peek().text) + ")");
        advance();
        return std::make_unique<SymbolExpr>(peek().loc, "__error__");
    }

    while (true) {
        if (match(TokenType::Dot)) {
            SourceLoc loc = previous().loc;
            std::string mem = consume(TokenType::Identifier, "Expected member name").text.str();

            if (check(TokenType::LParen)) {
                auto args = parseArgumentList();
                expr = std::make_unique<MemberCallNode>(loc, std::move(expr), mem, std::move(args));
            } else {
                expr = std::make_unique<MemberExpr>(loc, std::move(expr), mem);
            }
        }
        else if (match(TokenType::LBracket)) {
            SourceLoc loc = previous().loc;
            std::unique_ptr<Expr> idxExpr;

            if (match(TokenType::Range)) {
                std::unique_ptr<Expr> end = nullptr;
                if (!check(TokenType::RBracket)) end = parseExpr();
                idxExpr = std::make_unique<RangeExpr>(loc, nullptr, std::move(end));
            }
            else {
                auto start = parseExpr();
                if (match(TokenType::Range)) {
                    std::unique_ptr<Expr> end = nullptr;
                    if (!check(TokenType::RBracket)) end = parseExpr();
                    idxExpr = std::make_unique<RangeExpr>(loc, std::move(start), std::move(end));
                } else {
                    idxExpr = std::move(start);
                }
            }
            consume(TokenType::RBracket, "Expected ']'");

            expr = std::make_unique<IndexExpr>(loc, std::move(expr), std::move(idxExpr));
        }
        else if (check(TokenType::LParen)) {
            SourceLoc callLoc = peek().loc;
            auto args = parseArgumentList();
            expr = std::make_unique<CallExpr>(callLoc, std::move(expr), std::move(args));
        }
        else {
            break;
        }
    }

    return expr;
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
// Handles: Primitives, Tuples, Containers (vec<T>), Generics (Box<T>), Shapes
// =============================================================================
Type Parser::parseType() {
    // 1. Tuple Type OR Grouping: (i32, f32) vs (i32)
    if (check(TokenType::LParen)) {
        consume(TokenType::LParen, "Expected '('");
        
        // Handle Unit: () -> Void
        if (check(TokenType::RParen)) {
            consume(TokenType::RParen, "Expected ')'");
            return Type{Type::Void};
        }

        Type first = parseType();

        // If comma follows, it is a Tuple: (i32, f32) or (i32,)
        if (match(TokenType::Comma)) {
            Type t;
            t.kind = Type::Tuple;
            
            // [FIX] Use subtypes for Tuple elements (matches AST.h)
            t.subtypes.push_back(first);

            do {
                // Handle trailing comma: (i32,)
                if (check(TokenType::RParen)) break;
                
                t.subtypes.push_back(parseType());
            } while (match(TokenType::Comma));

            consume(TokenType::RParen, "Expect ')' after tuple type");
            return t;
        }

        // No comma? It's just a grouping: (i32) -> i32
        consume(TokenType::RParen, "Expect ')' after parenthesized type");
        return first;
    }

    Type t;
    t.kind = Type::Void; 
    
    // 2. Primitives
    if (match(TokenType::KwU8))       t.kind = Type::U8;
    else if (match(TokenType::KwU16)) t.kind = Type::U16;
    else if (match(TokenType::KwU32)) t.kind = Type::U32;
    else if (match(TokenType::KwU64)) t.kind = Type::U64;
    else if (match(TokenType::KwI8))  t.kind = Type::I8;
    else if (match(TokenType::KwI16)) t.kind = Type::I16;
    else if (match(TokenType::KwI32)) t.kind = Type::I32;
    else if (match(TokenType::KwI64)) t.kind = Type::I64;
    else if (match(TokenType::KwF32)) t.kind = Type::F32;
    else if (match(TokenType::KwF64)) t.kind = Type::F64;
    else if (match(TokenType::KwBool)) t.kind = Type::Bool;
    else if (match(TokenType::KwStr))  t.kind = Type::Str;
    else if (match(TokenType::KwVoid)) t.kind = Type::Void;
    
    // 3. Built-in Container Keywords
    else if (match(TokenType::KwVec))    t.kind = Type::Vec;
    else if (match(TokenType::KwSlice))  t.kind = Type::Slice;
    else if (match(TokenType::KwTensor)) t.kind = Type::Tensor;
    
    // 4. User Types & User Generics
    else if (match(TokenType::Identifier)) {
        std::string name = previous().text.str();
        
        if (match(TokenType::Less)) {
            t.kind = Type::Generic;
            t.schemaName = name;
            do {
                t.genericArgs.push_back(parseType());
            } while (match(TokenType::Comma));
            consume(TokenType::Greater, "Expect '>' after generic arguments");
        } 
        else {
            t.kind = Type::Schema;
            t.schemaName = name;
        }
    }
    
    // 5. Generics for Built-in Containers (vec<T>)
    if (t.isContainer()) {
        consume(TokenType::Less, "Expected '<'");
        Type inner = parseType();
        
        // [FIX] Store element type in genericArgs[0] (Unified System)
        t.genericArgs.push_back(inner);
        
        consume(TokenType::Greater, "Expected '>'");
    }
    
    // 6. Tensor Shape Syntax: T[2, 2] -> Tensor<T> with shape
    //    Allows writing "f32[2, 2]" instead of "tensor<f32>"
    if (check(TokenType::LBracket)) {
        // Wrap current type 't' as the element of a new Tensor type
        if (t.kind != Type::Tensor) {
             Type element = t;
             t = Type(); // Reset
             t.kind = Type::Tensor;
             
             // [FIX] Unified Generic Storage
             t.genericArgs.push_back(element);
        }

        while (match(TokenType::LBracket)) {
             if (check(TokenType::Integer)) {
                 t.shape.push_back(advance().text.str());
             } else {
                 t.shape.push_back(consume(TokenType::Identifier, "Expected size identifier").text.str());
             }
             consume(TokenType::RBracket, "Expected ']'");
        }
    }
    
    // 7. Memory Space: @gpu:0
    if (match(TokenType::At)) {
        if (match(TokenType::KwGpu)) {
            t.space.kind = Space::GPU;
            if (match(TokenType::Colon)) {
                 std::string devId = consume(TokenType::Integer, "Expected device ID").text.str();
                 try { t.space.deviceId = std::stoi(devId); } catch(...) {}
            }
        } else if (match(TokenType::KwRam)) {
            t.space.kind = Space::RAM;
        }
    }

    return t;
}

} // namespace arklang