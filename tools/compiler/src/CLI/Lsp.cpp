// tools/compiler/CLI/Lsp.cpp
#include "ark/compiler/CLI/Subcommands.hpp"

#include <CLI/CLI.hpp>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include "ark/compiler/Frontend/Lexer.hpp"
#include "ark/compiler/Frontend/Parser.hpp"
#include "ark/compiler/Frontend/AST.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace ark::cli {
namespace {

constexpr int kJsonRpcParseError = -32700;
constexpr int kJsonRpcInvalidRequest = -32600;
constexpr int kJsonRpcMethodNotFound = -32601;
constexpr int kLspServerNotInitialized = -32002;

// =============================================================================
// Transport
// =============================================================================
class LspTransport {
public:
    enum class ReadResult {
        Ok,
        Eof,
        ProtocolError
    };

    LspTransport(std::istream& inStream, llvm::raw_ostream& outStream)
        : in(inStream), out(outStream) {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    }

    ReadResult readMessage(std::string& outPayload) {
        std::string line;
        std::size_t contentLength = 0;
        bool sawAnyHeader = false;
        bool sawContentLength = false;

        for (;;) {
            if (!std::getline(in, line)) {
                if (!sawAnyHeader && in.eof()) return ReadResult::Eof;
                return ReadResult::ProtocolError;
            }

            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) break;

            sawAnyHeader = true;

            const std::size_t colon = line.find(':');
            if (colon == std::string::npos) continue;

            std::string_view name = trimAscii(std::string_view(line).substr(0, colon));
            std::string_view value = trimAscii(std::string_view(line).substr(colon + 1));

            if (asciiIEquals(name, "Content-Length")) {
                std::size_t parsed = 0;
                if (!parseContentLength(value, parsed) || parsed == 0) return ReadResult::ProtocolError;
                contentLength = parsed;
                sawContentLength = true;
            }
        }

        if (!sawContentLength) return ReadResult::ProtocolError;

        outPayload.resize(contentLength);
        in.read(outPayload.data(), static_cast<std::streamsize>(contentLength));
        if (!in) return ReadResult::ProtocolError;

        return ReadResult::Ok;
    }

    void replySuccess(const llvm::json::Value* id, llvm::json::Value result) {
        if (!id) return;

        llvm::json::Object response;
        response["jsonrpc"] = "2.0";
        response["id"] = *id;
        response["result"] = std::move(result);
        writeFramedPayload(llvm::json::Value(std::move(response)));
    }

    void replyNullResult(const llvm::json::Value* id) {
        if (!id) return;

        llvm::json::Object response;
        response["jsonrpc"] = "2.0";
        response["id"] = *id;
        response["result"] = nullptr;
        writeFramedPayload(llvm::json::Value(std::move(response)));
    }

    void replyError(const llvm::json::Value* id, int code, llvm::StringRef message) {
        llvm::json::Object err;
        err["code"] = code;
        err["message"] = message;

        llvm::json::Object response;
        response["jsonrpc"] = "2.0";
        if (id) response["id"] = *id;
        else response["id"] = nullptr;
        response["error"] = std::move(err);

        writeFramedPayload(llvm::json::Value(std::move(response)));
    }

    void sendNotification(llvm::StringRef method, llvm::json::Value params) {
        llvm::json::Object notification;
        notification["jsonrpc"] = "2.0";
        notification["method"] = method;
        notification["params"] = std::move(params);
        writeFramedPayload(llvm::json::Value(std::move(notification)));
    }

private:
    std::istream& in;
    llvm::raw_ostream& out;

    void writeFramedPayload(const llvm::json::Value& payload) {
        std::string body;
        llvm::raw_string_ostream os(body);
        os << payload;
        os.flush();

        out << "Content-Length: " << body.size() << "\r\n\r\n" << body;
        out.flush();
    }

    static std::string_view trimAscii(std::string_view s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
        return s;
    }

    static bool asciiIEquals(std::string_view a, std::string_view b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const auto ca = static_cast<unsigned char>(a[i]);
            const auto cb = static_cast<unsigned char>(b[i]);
            if (std::tolower(ca) != std::tolower(cb)) return false;
        }
        return true;
    }

    static bool parseContentLength(std::string_view s, std::size_t& outValue) {
        s = trimAscii(s);
        if (s.empty()) return false;

        unsigned long long v = 0;
        auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
        if (ec != std::errc{} || ptr != s.data() + s.size()) return false;
        if (v > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) return false;

        outValue = static_cast<std::size_t>(v);
        return true;
    }
};

// =============================================================================
// Cursor/Text Helpers
// =============================================================================
static bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static std::optional<std::size_t> lineColToOffset(const std::string& text, int targetLine, int targetCol) {
    if (targetLine < 0 || targetCol < 0) return std::nullopt;

    int line = 0;
    std::size_t lineStart = 0;

    for (std::size_t i = 0; i < text.size() && line < targetLine; ++i) {
        if (text[i] == '\n') {
            ++line;
            lineStart = i + 1;
        }
    }

    if (line != targetLine) return std::nullopt;

    const std::size_t off = lineStart + static_cast<std::size_t>(targetCol);
    if (off > text.size()) return std::nullopt;
    return off;
}

static std::pair<int, int> offsetToLineCol(const std::string& text, std::size_t offset) {
    if (offset > text.size()) offset = text.size();

    int line = 0;
    int col = 0;
    for (std::size_t i = 0; i < offset; ++i) {
        if (text[i] == '\n') {
            ++line;
            col = 0;
        } else {
            ++col;
        }
    }

    return {line, col};
}

static std::optional<std::string> identifierBeforeDot(const std::string& text, int line, int col) {
    auto offOpt = lineColToOffset(text, line, col);
    if (!offOpt) return std::nullopt;

    const std::size_t off = *offOpt;
    if (off == 0 || off > text.size()) return std::nullopt;
    if (text[off - 1] != '.') return std::nullopt;

    std::size_t i = off - 1;
    if (i == 0) return std::nullopt;
    --i;

    if (!isIdentChar(text[i])) return std::nullopt;

    const std::size_t end = i + 1;
    while (i > 0 && isIdentChar(text[i - 1])) --i;

    return text.substr(i, end - i);
}

static std::optional<std::string> identifierAtCursor(const std::string& text, int line, int col) {
    auto offOpt = lineColToOffset(text, line, col);
    if (!offOpt) return std::nullopt;

    std::size_t probe = *offOpt;
    if (probe > text.size()) return std::nullopt;

    auto tryProbe = [&](std::size_t p) -> std::optional<std::string> {
        if (text.empty()) return std::nullopt;

        if (p >= text.size()) {
            if (p == 0) return std::nullopt;
            p -= 1;
        }

        if (!isIdentChar(text[p])) {
            if (p == 0 || !isIdentChar(text[p - 1])) return std::nullopt;
            p -= 1;
        }

        std::size_t end = p + 1;
        std::size_t start = p;
        while (start > 0 && isIdentChar(text[start - 1])) --start;

        if (start == end) return std::nullopt;
        return text.substr(start, end - start);
    };

    if (auto id = tryProbe(probe)) return id;
    if (probe > 0) return tryProbe(probe - 1);
    return std::nullopt;
}

struct QualifiedMemberRef {
    std::string receiver;
    std::string member;
};

static std::optional<QualifiedMemberRef> qualifiedMemberAtCursor(const std::string& text, int line, int col) {
    auto offOpt = lineColToOffset(text, line, col);
    if (!offOpt) return std::nullopt;

    std::size_t probe = *offOpt;
    if (probe > text.size()) return std::nullopt;

    auto tryProbe = [&](std::size_t p) -> std::optional<QualifiedMemberRef> {
        if (text.empty()) return std::nullopt;

        if (p >= text.size()) {
            if (p == 0) return std::nullopt;
            p -= 1;
        }

        if (!isIdentChar(text[p])) {
            if (p == 0 || !isIdentChar(text[p - 1])) return std::nullopt;
            p -= 1;
        }

        std::size_t memberEnd = p + 1;
        std::size_t memberStart = p;
        while (memberStart > 0 && isIdentChar(text[memberStart - 1])) --memberStart;

        if (memberStart == 0 || text[memberStart - 1] != '.') return std::nullopt;

        const std::size_t recvEnd = memberStart - 1;
        std::size_t recvStart = recvEnd;
        while (recvStart > 0 && isIdentChar(text[recvStart - 1])) --recvStart;

        if (recvStart == recvEnd) return std::nullopt;

        QualifiedMemberRef q;
        q.receiver = text.substr(recvStart, recvEnd - recvStart);
        q.member = text.substr(memberStart, memberEnd - memberStart);
        return q;
    };

    if (auto q = tryProbe(probe)) return q;
    if (probe > 0) return tryProbe(probe - 1);
    return std::nullopt;
}


struct IdentifierSpan {
    std::string name;
    std::size_t startOffset = 0;
    std::size_t endOffset = 0; // exclusive
};

struct QualifiedMemberSpan {
    std::string receiver;
    std::string member;
    std::size_t receiverStartOffset = 0;
    std::size_t receiverEndOffset = 0; // exclusive
    std::size_t memberStartOffset = 0;
    std::size_t memberEndOffset = 0;   // exclusive
};

static std::optional<IdentifierSpan> identifierSpanAtCursor(const std::string& text, int line, int col) {
    auto offOpt = lineColToOffset(text, line, col);
    if (!offOpt) return std::nullopt;

    std::size_t probe = *offOpt;
    if (probe > text.size()) return std::nullopt;

    auto tryProbe = [&](std::size_t p) -> std::optional<IdentifierSpan> {
        if (text.empty()) return std::nullopt;

        if (p >= text.size()) {
            if (p == 0) return std::nullopt;
            p -= 1;
        }

        if (!isIdentChar(text[p])) {
            if (p == 0 || !isIdentChar(text[p - 1])) return std::nullopt;
            p -= 1;
        }

        std::size_t end = p + 1;
        std::size_t start = p;
        while (start > 0 && isIdentChar(text[start - 1])) --start;

        if (start == end) return std::nullopt;

        IdentifierSpan s;
        s.name = text.substr(start, end - start);
        s.startOffset = start;
        s.endOffset = end;
        return s;
    };

    if (auto id = tryProbe(probe)) return id;
    if (probe > 0) return tryProbe(probe - 1);
    return std::nullopt;
}

static std::optional<QualifiedMemberSpan> qualifiedMemberSpanAtCursor(const std::string& text, int line, int col) {
    auto offOpt = lineColToOffset(text, line, col);
    if (!offOpt) return std::nullopt;

    std::size_t probe = *offOpt;
    if (probe > text.size()) return std::nullopt;

    auto tryProbe = [&](std::size_t p) -> std::optional<QualifiedMemberSpan> {
        if (text.empty()) return std::nullopt;

        if (p >= text.size()) {
            if (p == 0) return std::nullopt;
            p -= 1;
        }

        if (!isIdentChar(text[p])) {
            if (p == 0 || !isIdentChar(text[p - 1])) return std::nullopt;
            p -= 1;
        }

        std::size_t memberEnd = p + 1;
        std::size_t memberStart = p;
        while (memberStart > 0 && isIdentChar(text[memberStart - 1])) --memberStart;

        if (memberStart == 0 || text[memberStart - 1] != '.') return std::nullopt;

        const std::size_t recvEnd = memberStart - 1;
        std::size_t recvStart = recvEnd;
        while (recvStart > 0 && isIdentChar(text[recvStart - 1])) --recvStart;

        if (recvStart == recvEnd) return std::nullopt;

        QualifiedMemberSpan s;
        s.receiver = text.substr(recvStart, recvEnd - recvStart);
        s.member = text.substr(memberStart, memberEnd - memberStart);
        s.receiverStartOffset = recvStart;
        s.receiverEndOffset = recvEnd;
        s.memberStartOffset = memberStart;
        s.memberEndOffset = memberEnd;
        return s;
    };

    if (auto q = tryProbe(probe)) return q;
    if (probe > 0) return tryProbe(probe - 1);
    return std::nullopt;
}

// =============================================================================
// URI / Path Helpers
// =============================================================================
static int hexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static std::string percentDecode(std::string_view s) {
    std::string out;
    out.reserve(s.size());

    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const int hi = hexValue(s[i + 1]);
            const int lo = hexValue(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(s[i]);
    }

    return out;
}

static std::optional<std::string> filePathFromUri(std::string_view uri) {
    static constexpr std::string_view kFileScheme = "file://";
    if (uri.size() < kFileScheme.size() || uri.substr(0, kFileScheme.size()) != kFileScheme) {
        return std::nullopt;
    }

    std::string path = percentDecode(uri.substr(kFileScheme.size()));

#if defined(_WIN32)
    if (!path.empty() && path.front() == '/' && path.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(path[1])) && path[2] == ':') {
        path.erase(path.begin());
    }
#endif

    return path;
}

static std::string fileUriFromPath(std::string_view path) {
    std::string out = "file://";

#if defined(_WIN32)
    if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':') {
        out.push_back('/');
    }
#endif

    auto isUnreserved = [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/';
    };

    static constexpr char hex[] = "0123456789ABCDEF";
    for (unsigned char c : std::string(path)) {
        if (isUnreserved(c)) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hex[(c >> 4) & 0xF]);
            out.push_back(hex[c & 0xF]);
        }
    }

    return out;
}

static std::string dirnameOfPath(std::string_view p) {
    llvm::SmallString<512> tmp(p);
    return std::string(llvm::sys::path::parent_path(tmp).str());
}

static std::string joinAndNormalizePath(std::string_view a, std::string_view b) {
    llvm::StringRef bRef(b.data(), b.size());
    llvm::SmallString<512> out;

    if (llvm::sys::path::is_absolute(bRef)) {
        out = llvm::SmallString<512>(bRef);
    } else {
        out = llvm::SmallString<512>(llvm::StringRef(a.data(), a.size()));
        llvm::sys::path::append(out, bRef);
    }

    llvm::sys::path::remove_dots(out, true);
    return std::string(out.str());
}

// =============================================================================
// AST-backed Symbol Indexing
// =============================================================================
struct DefinitionHit {
    std::string path;
    int line = 0;
    int col = 0;
    int length = 1;
};

struct ImportedModuleSymbol {
    enum class Kind {
        Function,
        Schema
    };

    Kind kind = Kind::Function;
    std::string name;
    std::string detail;
    DefinitionHit def;
};

struct ImportedModuleInfo {
    std::string importPath;
    std::string resolvedPath;
    std::vector<ImportedModuleSymbol> symbols;
    std::unordered_map<std::string, DefinitionHit> defsByName;
};

static DefinitionHit makeDefinitionHit(const arklang::SourceLoc& loc,
                                       const std::string& fallbackPath,
                                       const std::string& symbolName) {
    DefinitionHit hit;
    hit.path = !loc.file.empty() ? loc.file : fallbackPath;
    hit.line = std::max(0, loc.line - 1);
    hit.col = std::max(0, loc.col - 1);
    hit.length = std::max(1, static_cast<int>(symbolName.size()));
    return hit;
}

static std::string functionDetailFromAst(const arklang::Function& fn) {
    std::string detail = "fn " + fn.name + "(";
    for (std::size_t i = 0; i < fn.args.size(); ++i) {
        if (i) detail += ", ";
        detail += fn.args[i].first;
        detail += ": ";
        detail += fn.args[i].second.toString();
    }
    detail += ") -> ";
    detail += fn.returnType.toString();
    return detail;
}

static std::string schemaDetailFromAst(const arklang::SchemaDecl& schema) {
    std::string detail = "schema " + schema.name;
    if (schema.kind == arklang::SchemaDecl::Enum) detail += " (enum)";
    if (schema.isSingleton) detail += " [singleton]";
    return detail;
}

static void appendModuleExportsFromAst(const arklang::Module& mod,
                                       const std::string& fallbackPath,
                                       std::vector<ImportedModuleSymbol>& outSymbols,
                                       std::unordered_map<std::string, DefinitionHit>& outDefsByName) {
    outSymbols.clear();
    outDefsByName.clear();

    outSymbols.reserve(mod.functions.size() + mod.schemas.size());

    for (const auto& fnPtr : mod.functions) {
        if (!fnPtr) continue;

        ImportedModuleSymbol sym;
        sym.kind = ImportedModuleSymbol::Kind::Function;
        sym.name = fnPtr->name;
        sym.detail = functionDetailFromAst(*fnPtr);
        sym.def = makeDefinitionHit(fnPtr->loc, fallbackPath, fnPtr->name);

        outDefsByName.emplace(sym.name, sym.def);
        outSymbols.push_back(std::move(sym));
    }

    for (const auto& schemaPtr : mod.schemas) {
        if (!schemaPtr) continue;

        ImportedModuleSymbol sym;
        sym.kind = ImportedModuleSymbol::Kind::Schema;
        sym.name = schemaPtr->name;
        sym.detail = schemaDetailFromAst(*schemaPtr);
        sym.def = makeDefinitionHit(schemaPtr->loc, fallbackPath, schemaPtr->name);

        outDefsByName.emplace(sym.name, sym.def);
        outSymbols.push_back(std::move(sym));
    }

    std::sort(outSymbols.begin(), outSymbols.end(), [](const ImportedModuleSymbol& a, const ImportedModuleSymbol& b) {
        if (a.name != b.name) return a.name < b.name;
        return static_cast<int>(a.kind) < static_cast<int>(b.kind);
    });
}

static std::unique_ptr<arklang::Module> parseModuleFromFileForIndex(const std::string& path) {
    auto mb = llvm::MemoryBuffer::getFile(path);
    if (!mb) return nullptr;

    const llvm::StringRef buf = (*mb)->getBuffer();
    std::string source(buf.data(), buf.size());

    arklang::Lexer lexer(std::move(source), path);
    arklang::TokenStream tokens = lexer.tokenize();
    if (!tokens.errors.empty()) return nullptr;

    arklang::Parser parser(std::move(tokens));
    auto mod = parser.parseModule();
    if (parser.hasErrors()) return nullptr;

    return mod;
}

// =============================================================================
// Heuristic local typing for dot completions
// =============================================================================
enum class BuiltinTypeKind {
    Unknown,
    Vector,
    String
};

struct SymbolInfo {
    BuiltinTypeKind type = BuiltinTypeKind::Unknown;
    std::string detail;
};

// =============================================================================
// Document Store
// =============================================================================
struct DocumentState {
    std::string text;
    std::string parseFilename;
    std::unique_ptr<arklang::Module> ast;
    std::vector<std::string> errors;

    std::unordered_map<std::string, SymbolInfo> symbols; // local variable heuristics
    std::unordered_map<std::string, ImportedModuleInfo> importAliases; // AST-backed
    std::unordered_map<std::string, DefinitionHit> topLevelDefs; // AST-backed
    std::unordered_map<std::string, std::string> topLevelHover; // AST-backed hover text
};


static void rebuildHeuristicSymbolIndex(DocumentState& state) {
    state.symbols.clear();

    static const std::regex typedVecDecl(
        R"(\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(Vec|Vector)\s*(?:<[^>]+>)?)",
        std::regex::icase);

    static const std::regex vecCtorDecl(
        R"(\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(Vec|Vector)\s*(?:::<[^>]+>)?\s*::\s*(new|with_capacity)\s*\()",
        std::regex::icase);

    static const std::regex arrayLiteralDecl(
        R"(\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[)",
        std::regex::icase);

    static const std::regex typedStringDecl(
        R"(\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*String\b)",
        std::regex::icase);

    static const std::regex stringCtorDecl(
        R"(\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*String\s*::\s*(new|from)\s*\()",
        std::regex::icase);

    auto scan = [&](const std::regex& re, BuiltinTypeKind kind, const char* detail) {
        for (std::sregex_iterator it(state.text.begin(), state.text.end(), re), end; it != end; ++it) {
            const std::string name = (*it)[1].str();
            state.symbols[name] = SymbolInfo{kind, detail};
        }
    };

    scan(typedVecDecl, BuiltinTypeKind::Vector, "Vector");
    scan(vecCtorDecl, BuiltinTypeKind::Vector, "Vector");
    scan(arrayLiteralDecl, BuiltinTypeKind::Vector, "Vector");
    scan(typedStringDecl, BuiltinTypeKind::String, "String");
    scan(stringCtorDecl, BuiltinTypeKind::String, "String");
}

static void rebuildTopLevelDefIndexFromAst(DocumentState& state) {
    state.topLevelDefs.clear();
    if (!state.ast) return;

    for (const auto& fnPtr : state.ast->functions) {
        if (!fnPtr) continue;
        state.topLevelDefs.emplace(fnPtr->name, makeDefinitionHit(fnPtr->loc, state.parseFilename, fnPtr->name));
    }

    for (const auto& schemaPtr : state.ast->schemas) {
        if (!schemaPtr) continue;
        state.topLevelDefs.emplace(schemaPtr->name, makeDefinitionHit(schemaPtr->loc, state.parseFilename, schemaPtr->name));
    }
}

static void rebuildTopLevelHoverIndexFromAst(DocumentState& state) {
    state.topLevelHover.clear();
    if (!state.ast) return;

    for (const auto& fnPtr : state.ast->functions) {
        if (!fnPtr) continue;
        state.topLevelHover[fnPtr->name] = functionDetailFromAst(*fnPtr);
    }

    for (const auto& schemaPtr : state.ast->schemas) {
        if (!schemaPtr) continue;
        state.topLevelHover[schemaPtr->name] = schemaDetailFromAst(*schemaPtr);
    }
}

static void rebuildImportAliasIndexFromAst(DocumentState& state, const std::string& docUri) {
    state.importAliases.clear();
    if (!state.ast) return;

    const auto docPathOpt = filePathFromUri(docUri);
    const std::string baseDir = docPathOpt ? dirnameOfPath(*docPathOpt) : dirnameOfPath(state.parseFilename);

    for (const auto& importPtr : state.ast->imports) {
        if (!importPtr) continue;
        if (importPtr->alias.empty()) continue;

        ImportedModuleInfo info;
        info.importPath = importPtr->path;
        info.resolvedPath = joinAndNormalizePath(baseDir, importPtr->path);

        if (importPtr->importedModule) {
            appendModuleExportsFromAst(*importPtr->importedModule, info.resolvedPath, info.symbols, info.defsByName);
        } else if (auto mod = parseModuleFromFileForIndex(info.resolvedPath)) {
            appendModuleExportsFromAst(*mod, info.resolvedPath, info.symbols, info.defsByName);
        }

        state.importAliases[importPtr->alias] = std::move(info);
    }
}

class DocumentStore {
public:
    void openDocument(const std::string& uri, const std::string& text) {
        updateDocument(uri, text);
        llvm::errs() << "[ARK LSP] Opened document: " << uri << "\n";
    }

    void updateDocument(const std::string& uri, const std::string& text) {
        DocumentState state;
        state.text = text;
        state.parseFilename = filePathFromUri(uri).value_or(uri);

        arklang::Lexer lexer(text, state.parseFilename);
        arklang::TokenStream tokens = lexer.tokenize();

        if (!tokens.errors.empty()) {
            state.errors = tokens.errors;
        } else {
            arklang::Parser parser(std::move(tokens));
            state.ast = parser.parseModule();
            state.errors = parser.getErrors();
        }

        rebuildHeuristicSymbolIndex(state);

        if (state.ast) {
            rebuildTopLevelDefIndexFromAst(state);
            rebuildTopLevelHoverIndexFromAst(state);
            rebuildImportAliasIndexFromAst(state, uri);
        } else {
            auto prev = documents.find(uri);
            if (prev != documents.end()) {
                state.topLevelDefs = prev->second.topLevelDefs;
                state.topLevelHover = prev->second.topLevelHover;
                state.importAliases = prev->second.importAliases;
            }
        }

        auto [it, _] = documents.insert_or_assign(uri, std::move(state));
        llvm::errs() << "[ARK LSP] Compiled document: " << uri
                    << " (" << it->second.errors.size() << " errors, "
                    << it->second.symbols.size() << " local symbols, "
                    << it->second.topLevelDefs.size() << " top-level defs, "
                    << it->second.importAliases.size() << " imports)\n";
    }

    void closeDocument(const std::string& uri) {
        documents.erase(uri);
        llvm::errs() << "[ARK LSP] Closed document: " << uri << "\n";
    }

    const DocumentState* getDocument(const std::string& uri) const {
        auto it = documents.find(uri);
        return (it == documents.end()) ? nullptr : &it->second;
    }

private:
    std::unordered_map<std::string, DocumentState> documents;
};

// =============================================================================
// LSP Server
// =============================================================================
class LspServer {
public:
    explicit LspServer(LspTransport& transportRef) : transport(transportRef) {}

    void run() {
        for (;;) {
            std::string payload;
            const auto readResult = transport.readMessage(payload);

            if (readResult == LspTransport::ReadResult::Eof) {
                llvm::errs() << "[ARK LSP] EOF on stdin. Exiting.\n";
                return;
            }

            if (readResult == LspTransport::ReadResult::ProtocolError) {
                llvm::errs() << "[ARK LSP] Protocol framing error\n";
                continue;
            }

            auto parsed = llvm::json::parse(payload);
            if (!parsed) {
                llvm::consumeError(parsed.takeError());
                transport.replyError(nullptr, kJsonRpcParseError, "Parse error");
                continue;
            }

            llvm::json::Object* req = parsed->getAsObject();
            if (!req) {
                transport.replyError(nullptr, kJsonRpcInvalidRequest, "Invalid Request");
                continue;
            }

            handleMessage(*req);
            if (shouldExit) return;
        }
    }

private:
    enum class State {
        PreInitialize,
        Running,
        ShutdownRequested
    };

    LspTransport& transport;
    DocumentStore documents;
    State state = State::PreInitialize;
    bool shouldExit = false;
    bool clientSnippetSupport = false;

    void addSnippetCompletion(
        llvm::json::Array& items,
        std::string label,
        std::string filterText,
        std::string detail,
        std::string snippetInsertText,
        std::string plainInsertText,
        std::string documentation = {},
        int kind = 15) {

        llvm::json::Object item;
        item["label"] = std::move(label);
        item["kind"] = kind;
        item["detail"] = std::move(detail);

        if (!filterText.empty()) item["filterText"] = std::move(filterText);

        if (clientSnippetSupport) {
            item["insertText"] = std::move(snippetInsertText);
            item["insertTextFormat"] = 2;
        } else {
            item["insertText"] = std::move(plainInsertText);
        }

        if (!documentation.empty()) item["documentation"] = std::move(documentation);

        items.push_back(std::move(item));
    }
    static bool detectClientSnippetSupport(const llvm::json::Object* params) {
        if (!params) return false;

        const auto* capabilities = params->getObject("capabilities");
        if (!capabilities) return false;

        const auto* textDocument = capabilities->getObject("textDocument");
        if (!textDocument) return false;

        const auto* completion = textDocument->getObject("completion");
        if (!completion) return false;

        const auto* completionItem = completion->getObject("completionItem");
        if (!completionItem) return false;

        if (auto v = completionItem->getBoolean("snippetSupport")) return *v;
        return false;
    }

    static bool extractLineCol(const std::string& err, int& lineOut, int& colOut) {
        static const std::regex re1(R"(line\s+(\d+).*?col(?:umn)?\s+(\d+))", std::regex::icase);
        static const std::regex re2(R"((\d+):(\d+))");

        std::smatch m;
        if (std::regex_search(err, m, re1) && m.size() >= 3) {
            lineOut = std::max(0, std::stoi(m[1].str()) - 1);
            colOut = std::max(0, std::stoi(m[2].str()) - 1);
            return true;
        }
        if (std::regex_search(err, m, re2) && m.size() >= 3) {
            lineOut = std::max(0, std::stoi(m[1].str()) - 1);
            colOut = std::max(0, std::stoi(m[2].str()) - 1);
            return true;
        }
        return false;
    }

    static void addCompletionItem(
        llvm::json::Array& items,
        std::string label,
        int kind,
        std::string detail,
        std::string insertText = {},
        std::string documentation = {}) {

        llvm::json::Object item;
        item["label"] = std::move(label);
        item["kind"] = kind;
        item["detail"] = std::move(detail);

        if (!insertText.empty()) {
            item["insertText"] = std::move(insertText);
        }

        if (!documentation.empty()) {
            item["documentation"] = std::move(documentation);
        }

        items.push_back(std::move(item));
    }

    static void addVectorCompletions(llvm::json::Array& items) {
        addCompletionItem(items, "len", 2, "fn len() -> Int", "len()", "Returns number of elements.");
        addCompletionItem(items, "is_empty", 2, "fn is_empty() -> Bool", "is_empty()", "Returns true if vector has no elements.");
        addCompletionItem(items, "push", 2, "fn push(value: T) -> Void", "push()", "Appends one element.");
        addCompletionItem(items, "pop", 2, "fn pop() -> Option<T>", "pop()", "Removes and returns the last element.");
        addCompletionItem(items, "clear", 2, "fn clear() -> Void", "clear()", "Removes all elements.");
        addCompletionItem(items, "get", 2, "fn get(index: Int) -> Option<T>", "get()", "Gets element by index.");
    }

    static void addStringCompletions(llvm::json::Array& items) {
        addCompletionItem(items, "len", 2, "fn len() -> Int", "len()", "Returns string length.");
        addCompletionItem(items, "is_empty", 2, "fn is_empty() -> Bool", "is_empty()", "Returns true if empty.");
        addCompletionItem(items, "contains", 2, "fn contains(s: String) -> Bool", "contains()", "Substring check.");
        addCompletionItem(items, "starts_with", 2, "fn starts_with(prefix: String) -> Bool", "starts_with()", "Prefix check.");
    }

    static void addImportedModuleCompletions(llvm::json::Array& items, const ImportedModuleInfo& mod) {
        for (const auto& sym : mod.symbols) {
            if (sym.kind == ImportedModuleSymbol::Kind::Function) {
                addCompletionItem(
                    items,
                    sym.name,
                    3,
                    sym.detail,
                    sym.name + "()",
                    "Imported function from " + mod.importPath);
            } else {
                addCompletionItem(
                    items,
                    sym.name,
                    22,
                    sym.detail,
                    "",
                    "Imported schema from " + mod.importPath);
            }
        }
    }


    void publishDiagnostics(llvm::StringRef uri) {
        llvm::json::Array diagnostics;

        const auto* doc = documents.getDocument(uri.str());
        if (doc) {
            for (const auto& errStr : doc->errors) {
                int line = 0;
                int col = 0;
                (void)extractLineCol(errStr, line, col);

                llvm::json::Object start;
                start["line"] = line;
                start["character"] = col;

                llvm::json::Object end;
                end["line"] = line;
                end["character"] = col + 1;

                llvm::json::Object range;
                range["start"] = std::move(start);
                range["end"] = std::move(end);

                llvm::json::Object diag;
                diag["range"] = std::move(range);
                diag["severity"] = 1;
                diag["source"] = "arkc";
                diag["message"] = errStr;

                diagnostics.push_back(std::move(diag));
            }
        }

        llvm::json::Object params;
        params["uri"] = uri;
        params["diagnostics"] = std::move(diagnostics);
        transport.sendNotification("textDocument/publishDiagnostics", std::move(params));
    }

    void handleMessage(const llvm::json::Object& req) {
        const llvm::json::Value* id = req.get("id");
        const bool isRequest = (id != nullptr);
        std::optional<llvm::StringRef> method = req.getString("method");

        if (!method) {
            if (isRequest) transport.replyError(id, kJsonRpcInvalidRequest, "Invalid Request");
            return;
        }

        if (*method == "exit") {
            shouldExit = true;
            return;
        }

        if (state == State::PreInitialize && *method != "initialize") {
            if (isRequest) transport.replyError(id, kLspServerNotInitialized, "Server not initialized");
            return;
        }

        if (state == State::ShutdownRequested) {
            if (isRequest) transport.replyError(id, kJsonRpcInvalidRequest, "Server is shutting down");
            return;
        }

        if (*method == "$/cancelRequest" || *method == "$/setTrace" || *method == "$/logTrace") {
            return;
        }

        const llvm::json::Object* params = req.getObject("params");

        if (*method == "initialize") {
            handleInitialize(id, params);
            return;
        }

        if (*method == "initialized") {
            llvm::errs() << "[ARK LSP] Client is ready.\n";
            return;
        }

        if (*method == "shutdown") {
            state = State::ShutdownRequested;
            if (isRequest) transport.replyNullResult(id);
            return;
        }

        if (*method == "textDocument/didOpen" && params) {
            handleDidOpen(*params);
            return;
        }

        if (*method == "textDocument/didChange" && params) {
            handleDidChange(*params);
            return;
        }

        if (*method == "textDocument/didClose" && params) {
            handleDidClose(*params);
            return;
        }

        if (*method == "textDocument/completion" && params) {
            handleCompletion(*params, id);
            return;
        }

        if (*method == "textDocument/definition" && params) {
            handleDefinition(*params, id);
            return;
        }

        if (*method == "textDocument/hover") {
            if (params) handleHover(*params, id);
            else if (isRequest) transport.replyNullResult(id);
            return;
        }

        if (isRequest) transport.replyError(id, kJsonRpcMethodNotFound, "Method not found");
    }

    void handleInitialize(const llvm::json::Value* id, const llvm::json::Object* params) {
        clientSnippetSupport = detectClientSnippetSupport(params);

        llvm::json::Object sync;
        sync["openClose"] = true;
        sync["change"] = 1;

        llvm::json::Object completionProvider;
        completionProvider["resolveProvider"] = false;
        llvm::json::Array triggers;
        triggers.push_back(".");
        triggers.push_back(":");
        completionProvider["triggerCharacters"] = std::move(triggers);

        llvm::json::Object capabilities;
        capabilities["textDocumentSync"] = std::move(sync);
        capabilities["completionProvider"] = std::move(completionProvider);
        capabilities["hoverProvider"] = true;
        capabilities["definitionProvider"] = true;

        llvm::json::Object serverInfo;
        serverInfo["name"] = "arknet-lsp";
        serverInfo["version"] = "0.3.1";

        llvm::json::Object result;
        result["capabilities"] = std::move(capabilities);
        result["serverInfo"] = std::move(serverInfo);

        transport.replySuccess(id, std::move(result));
        state = State::Running;
        llvm::errs() << "[ARK LSP] Initialized connection with client. snippets="
                     << (clientSnippetSupport ? "on" : "off") << "\n";
    }


    void handleDidOpen(const llvm::json::Object& params) {
        const auto* doc = params.getObject("textDocument");
        if (!doc) return;

        auto uri = doc->getString("uri");
        auto text = doc->getString("text");
        if (!uri || !text) return;

        documents.openDocument(uri->str(), text->str());
        publishDiagnostics(uri->str());
    }

    void handleDidChange(const llvm::json::Object& params) {
        const auto* doc = params.getObject("textDocument");
        const auto* changes = params.getArray("contentChanges");
        if (!doc || !changes || changes->empty()) return;

        auto uri = doc->getString("uri");
        if (!uri) return;

        const auto* firstChange = (*changes)[0].getAsObject();
        if (!firstChange) return;

        auto text = firstChange->getString("text");
        if (!text) return;

        documents.updateDocument(uri->str(), text->str());
        publishDiagnostics(uri->str());
    }

    void handleDidClose(const llvm::json::Object& params) {
        const auto* doc = params.getObject("textDocument");
        if (!doc) return;

        auto uri = doc->getString("uri");
        if (!uri) return;

        documents.closeDocument(uri->str());

        llvm::json::Object outParams;
        outParams["uri"] = *uri;
        outParams["diagnostics"] = llvm::json::Array();
        transport.sendNotification("textDocument/publishDiagnostics", std::move(outParams));
    }

    void handleCompletion(const llvm::json::Object& params, const llvm::json::Value* id) {
        if (!id) return;

        llvm::json::Array items;

        const auto* docParam = params.getObject("textDocument");
        const auto* pos = params.getObject("position");
        if (!docParam || !pos) {
            llvm::json::Object list;
            list["isIncomplete"] = false;
            list["items"] = std::move(items);
            transport.replySuccess(id, std::move(list));
            return;
        }

        auto uri = docParam->getString("uri");
        auto lineOpt = pos->getInteger("line");
        auto charOpt = pos->getInteger("character");

        if (!uri || !lineOpt || !charOpt || *lineOpt < 0 || *charOpt < 0) {
            llvm::json::Object list;
            list["isIncomplete"] = false;
            list["items"] = std::move(items);
            transport.replySuccess(id, std::move(list));
            return;
        }

        const auto* docState = documents.getDocument(uri->str());
        if (!docState) {
            llvm::json::Object list;
            list["isIncomplete"] = false;
            list["items"] = std::move(items);
            transport.replySuccess(id, std::move(list));
            return;
        }

        const int targetLine = static_cast<int>(*lineOpt);
        const int targetCol = static_cast<int>(*charOpt);

        const auto qSpan = qualifiedMemberSpanAtCursor(docState->text, targetLine, targetCol);

        std::optional<std::string> recvIdent;
        if (qSpan) recvIdent = qSpan->receiver;
        else recvIdent = identifierBeforeDot(docState->text, targetLine, targetCol);

        auto makeRangeFromOffsets = [&](std::size_t startOff, std::size_t endOff) {
            auto [sl, sc] = offsetToLineCol(docState->text, startOff);
            auto [el, ec] = offsetToLineCol(docState->text, endOff);

            llvm::json::Object start;
            start["line"] = sl;
            start["character"] = sc;

            llvm::json::Object end;
            end["line"] = el;
            end["character"] = ec;

            llvm::json::Object range;
            range["start"] = std::move(start);
            range["end"] = std::move(end);
            return range;
        };

        auto addDotCompletion = [&](std::string label,
                                    int kind,
                                    std::string detail,
                                    std::string newText,
                                    std::string documentation = {}) {
            llvm::json::Object item;
            item["label"] = label;
            item["kind"] = kind;
            item["detail"] = detail;

            if (qSpan) {
                llvm::json::Object textEdit;
                textEdit["range"] = makeRangeFromOffsets(qSpan->memberStartOffset, qSpan->memberEndOffset);
                textEdit["newText"] = std::move(newText);
                item["textEdit"] = std::move(textEdit);
            } else {
                item["insertText"] = std::move(newText);
            }

            if (!documentation.empty()) item["documentation"] = std::move(documentation);

            items.push_back(std::move(item));
        };

        if (recvIdent) {
            auto importIt = docState->importAliases.find(*recvIdent);
            if (importIt != docState->importAliases.end()) {
                for (const auto& sym : importIt->second.symbols) {
                    const std::string documentation =
                        ((sym.kind == ImportedModuleSymbol::Kind::Function) ? "Imported function from " : "Imported schema from ")
                        + importIt->second.importPath;

                    if (sym.kind == ImportedModuleSymbol::Kind::Function) {
                        addDotCompletion(
                            std::string(sym.name),
                            3,
                            std::string(sym.detail),
                            std::string(sym.name) + "()",
                            documentation);
                    } else {
                        addDotCompletion(
                            std::string(sym.name),
                            22,
                            std::string(sym.detail),
                            std::string(sym.name),
                            documentation);
                    }
                }
            } else {
                auto it = docState->symbols.find(*recvIdent);
                if (it != docState->symbols.end()) {
                    switch (it->second.type) {
                        case BuiltinTypeKind::Vector:
                            addDotCompletion("len", 2, "fn len() -> Int", "len()", "Returns number of elements.");
                            addDotCompletion("is_empty", 2, "fn is_empty() -> Bool", "is_empty()", "Returns true if vector has no elements.");
                            addDotCompletion("push", 2, "fn push(value: T) -> Void", "push()", "Appends one element.");
                            addDotCompletion("pop", 2, "fn pop() -> Option<T>", "pop()", "Removes and returns the last element.");
                            addDotCompletion("clear", 2, "fn clear() -> Void", "clear()", "Removes all elements.");
                            addDotCompletion("get", 2, "fn get(index: Int) -> Option<T>", "get()", "Gets element by index.");
                            break;

                        case BuiltinTypeKind::String:
                            addDotCompletion("len", 2, "fn len() -> Int", "len()", "Returns string length.");
                            addDotCompletion("is_empty", 2, "fn is_empty() -> Bool", "is_empty()", "Returns true if empty.");
                            addDotCompletion("contains", 2, "fn contains(s: String) -> Bool", "contains()", "Substring check.");
                            addDotCompletion("starts_with", 2, "fn starts_with(prefix: String) -> Bool", "starts_with()", "Prefix check.");
                            break;

                        case BuiltinTypeKind::Unknown:
                        default:
                            addDotCompletion("to_string", 2, "fn to_string() -> String", "to_string()");
                            addDotCompletion("debug", 2, "fn debug() -> Void", "debug()");
                            break;
                    }
                } else {
                    addDotCompletion("to_string", 2, "fn to_string() -> String", "to_string()");
                    addDotCompletion("debug", 2, "fn debug() -> Void", "debug()");
                }
            }
        } else {
            addSnippetCompletion(
                items,
                "fn[host]",
                "fn",
                "Host function",
                "fn[host] ${1:name}(${2}) -> ${3:i32}${4: !IO} {\n\t$0\n}",
                "fn[host] name() -> i32 {\n\t\n}",
                "Host function template");

            addSnippetCompletion(
                items,
                "fn[cpu]",
                "fn",
                "CPU function",
                "fn[cpu] ${1:name}(${2}) -> ${3:i32} {\n\t$0\n}",
                "fn[cpu] name() -> i32 {\n\t\n}",
                "CPU function template");

            addSnippetCompletion(
                items,
                "fn[gpu]",
                "fn",
                "GPU kernel function",
                "fn[gpu] ${1:name}(${2:A: tensor<f32>, B: tensor<f32>, C: tensor<f32>}) {\n\tpar ${3:i} in ${4:C} {\n\t\t$0\n\t}\n}",
                "fn[gpu] name(A: tensor<f32>, B: tensor<f32>, C: tensor<f32>) {\n\tpar i in C {\n\t\t\n\t}\n}",
                "GPU kernel template");

            addSnippetCompletion(
                items,
                "schema",
                "schema",
                "Record schema",
                "schema ${1:Name} {\n\t${2:field}: ${3:i32}\n}",
                "schema Name {\n\tfield: i32\n}",
                "Record schema template");

            addSnippetCompletion(
                items,
                "singleton schema",
                "singleton",
                "Singleton schema",
                "singleton schema ${1:Config} {\n\t${2:key}: ${3:i32} = ${4:0}\n}",
                "singleton schema Config {\n\tkey: i32 = 0\n}",
                "Singleton schema template");

            addSnippetCompletion(
                items,
                "let",
                "let",
                "Variable declaration",
                "let ${1:name} = ${0:value};",
                "let name = value;",
                "Let statement");

            addSnippetCompletion(
                items,
                "let (typed)",
                "let",
                "Typed variable declaration",
                "let ${1:name}: ${2:i32} = ${0:value};",
                "let name: i32 = value;",
                "Typed let statement");

            addSnippetCompletion(
                items,
                "import",
                "import",
                "Import module with alias",
                "import \"${1:path/to/module.ark}\" as ${2:m};",
                "import \"path/to/module.ark\" as m;",
                "Import statement");

            addSnippetCompletion(
                items,
                "for",
                "for",
                "For range loop",
                "for ${1:i} in ${2:0}..${3:n} {\n\t$0\n}",
                "for i in 0..n {\n\t\n}",
                "For loop");

            addSnippetCompletion(
                items,
                "if",
                "if",
                "If statement",
                "if (${1:cond}) {\n\t$0\n}",
                "if (cond) {\n\t\n}",
                "If statement");

            addSnippetCompletion(
                items,
                "par",
                "par",
                "Parallel loop",
                "par ${1:i} in ${2:C} {\n\t$0\n}",
                "par i in C {\n\t\n}",
                "Parallel loop");

            addSnippetCompletion(
                items,
                "match",
                "match",
                "Match statement",
                "match ${1:value} {\n\tcase ${2:Pattern} => {\n\t\t$0\n\t}\n}",
                "match value {\n\tcase Pattern => {\n\t\t\n\t}\n}",
                "Match statement");

            addSnippetCompletion(
                items,
                "await",
                "await",
                "Await async token",
                "await ${1:t1};",
                "await t1;",
                "Await token");
        }

        llvm::json::Object completionList;
        completionList["isIncomplete"] = false;
        completionList["items"] = std::move(items);
        transport.replySuccess(id, std::move(completionList));
    }


    void handleHover(const llvm::json::Object& params, const llvm::json::Value* id) {
        if (!id) return;

        const auto* docParam = params.getObject("textDocument");
        const auto* pos = params.getObject("position");
        if (!docParam || !pos) {
            transport.replyNullResult(id);
            return;
        }

        auto uri = docParam->getString("uri");
        auto lineOpt = pos->getInteger("line");
        auto charOpt = pos->getInteger("character");
        if (!uri || !lineOpt || !charOpt || *lineOpt < 0 || *charOpt < 0) {
            transport.replyNullResult(id);
            return;
        }

        const auto* docState = documents.getDocument(uri->str());
        if (!docState) {
            transport.replyNullResult(id);
            return;
        }

        const int line = static_cast<int>(*lineOpt);
        const int col = static_cast<int>(*charOpt);

        auto replyHover = [&](const std::string& markdownText,
                             std::optional<std::pair<std::size_t, std::size_t>> spanOffsets = std::nullopt) {
            llvm::json::Object markup;
            markup["kind"] = "markdown";
            markup["value"] = "```ark\n" + markdownText + "\n```";

            llvm::json::Object hover;
            hover["contents"] = std::move(markup);

            if (spanOffsets) {
                auto [startLine, startChar] = offsetToLineCol(docState->text, spanOffsets->first);
                auto [endLine, endChar] = offsetToLineCol(docState->text, spanOffsets->second);

                llvm::json::Object start;
                start["line"] = startLine;
                start["character"] = startChar;

                llvm::json::Object end;
                end["line"] = endLine;
                end["character"] = endChar;

                llvm::json::Object range;
                range["start"] = std::move(start);
                range["end"] = std::move(end);
                hover["range"] = std::move(range);
            }

            transport.replySuccess(id, std::move(hover));
        };

        // 1) Imported qualified member hover: m.add / m.Vector
        if (const auto q = qualifiedMemberSpanAtCursor(docState->text, line, col)) {
            const auto importIt = docState->importAliases.find(q->receiver);
            if (importIt != docState->importAliases.end()) {
                for (const auto& sym : importIt->second.symbols) {
                    if (sym.name == q->member) {
                        replyHover(sym.detail, std::make_pair(q->memberStartOffset, q->memberEndOffset));
                        return;
                    }
                }
            }
        }

        // 2) Identifier hover (local heuristic, top-level AST, alias)
        if (const auto ident = identifierSpanAtCursor(docState->text, line, col)) {
            if (auto symIt = docState->symbols.find(ident->name); symIt != docState->symbols.end()) {
                replyHover(symIt->second.detail, std::make_pair(ident->startOffset, ident->endOffset));
                return;
            }

            if (auto topIt = docState->topLevelHover.find(ident->name); topIt != docState->topLevelHover.end()) {
                replyHover(topIt->second, std::make_pair(ident->startOffset, ident->endOffset));
                return;
            }

            if (auto aliasIt = docState->importAliases.find(ident->name); aliasIt != docState->importAliases.end()) {
                replyHover("import alias " + ident->name + " -> " + aliasIt->second.importPath,
                           std::make_pair(ident->startOffset, ident->endOffset));
                return;
            }
        }

        transport.replyNullResult(id);
    }

    void handleDefinition(const llvm::json::Object& params, const llvm::json::Value* id) {
        if (!id) return;

        const auto* docParam = params.getObject("textDocument");
        const auto* pos = params.getObject("position");
        if (!docParam || !pos) {
            transport.replyNullResult(id);
            return;
        }

        auto uri = docParam->getString("uri");
        auto lineOpt = pos->getInteger("line");
        auto charOpt = pos->getInteger("character");
        if (!uri || !lineOpt || !charOpt || *lineOpt < 0 || *charOpt < 0) {
            transport.replyNullResult(id);
            return;
        }

        const auto* docState = documents.getDocument(uri->str());
        if (!docState) {
            transport.replyNullResult(id);
            return;
        }

        const int line = static_cast<int>(*lineOpt);
        const int col = static_cast<int>(*charOpt);

        auto replyLocation = [&](const DefinitionHit& hit) {
            llvm::json::Object start;
            start["line"] = hit.line;
            start["character"] = hit.col;

            llvm::json::Object end;
            end["line"] = hit.line;
            end["character"] = hit.col + std::max(1, hit.length);

            llvm::json::Object range;
            range["start"] = std::move(start);
            range["end"] = std::move(end);

            llvm::json::Object location;
            location["uri"] = fileUriFromPath(hit.path);
            location["range"] = std::move(range);

            transport.replySuccess(id, std::move(location));
        };

        if (const auto q = qualifiedMemberAtCursor(docState->text, line, col)) {
            const auto importIt = docState->importAliases.find(q->receiver);
            if (importIt != docState->importAliases.end()) {
                const auto defIt = importIt->second.defsByName.find(q->member);
                if (defIt != importIt->second.defsByName.end()) {
                    replyLocation(defIt->second);
                    return;
                }
            }
        }

        if (const auto ident = identifierAtCursor(docState->text, line, col)) {
            const auto aliasIt = docState->importAliases.find(*ident);
            if (aliasIt != docState->importAliases.end()) {
                DefinitionHit hit;
                hit.path = aliasIt->second.resolvedPath;
                hit.line = 0;
                hit.col = 0;
                hit.length = 1;
                replyLocation(hit);
                return;
            }

            const auto defIt = docState->topLevelDefs.find(*ident);
            if (defIt != docState->topLevelDefs.end()) {
                replyLocation(defIt->second);
                return;
            }
        }

        transport.replyNullResult(id);
    }
};

} // namespace

void setupLspCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("lsp", "Start the Arknet Language Server Protocol (LSP) process");
    sub->callback([]() {
        llvm::errs() << "[ARK LSP] Starting language server...\n";
        LspTransport transport(std::cin, llvm::outs());
        LspServer server(transport);
        server.run();
    });
}

} // namespace ark::cli