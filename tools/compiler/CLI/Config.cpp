// tools/compiler/CLI/Config.cpp
#include "Config.h"
#include "Subcommands.h"
#include "ark/crypto/Vault.h"

#include <CLI/CLI.hpp>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <toml++/toml.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

namespace ark::cli {
namespace {

// =============================================================================
// Output & Failure Helpers
// =============================================================================
void printOut(const std::string& s) {
    llvm::outs() << s;
}

void note(const std::string& s) {
    llvm::outs() << s << "\n";
}

void warn(const std::string& s) {
    llvm::errs() << "[warn] " << s << "\n";
}

[[noreturn]] void fail(const std::string& s) {
    llvm::errs() << "[error] " << s << "\n";
    std::exit(1);
}

// =============================================================================
// Small String Helpers
// =============================================================================
std::string toString(llvm::StringRef s) {
    return std::string(s.data(), s.size());
}

std::vector<std::string> splitKey(const std::string& key) {
    std::vector<std::string> parts;
    std::string cur;

    for (char c : key) {
        if (c == '.') {
            if (cur.empty()) {
                fail("Invalid config key: empty segment in '" + key + "'");
            }

            parts.push_back(cur);
            cur.clear();
            continue;
        }

        cur.push_back(c);
    }

    if (cur.empty()) {
        fail("Invalid config key: empty segment in '" + key + "'");
    }

    parts.push_back(cur);
    return parts;
}

// =============================================================================
// File Write Helpers
// =============================================================================
static void ensureParentDirExists(const std::string& path) {
    llvm::SmallString<256> p(path);
    const llvm::StringRef parent = llvm::sys::path::parent_path(p);
    if (parent.empty()) {
        return;
    }

    const std::string parentStr = std::string(parent);
    const std::error_code ec = llvm::sys::fs::create_directories(parent);
    if (ec) {
        fail("Failed to create parent directory '" + parentStr + "': " + ec.message());
    }
}

void writeTextFileAtomicOrFail(const std::string& path, llvm::StringRef content) {
    ensureParentDirExists(path);

    const std::string tmpPath = path + ".tmp";

    {
        std::error_code ec;
        llvm::raw_fd_ostream os(tmpPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            fail("Failed to open temp file '" + tmpPath + "': " + ec.message());
        }

        os << content;
        os.flush();

        if (os.has_error()) {
            os.clear_error();
            (void)llvm::sys::fs::remove(tmpPath);
            fail("Failed to write temp file '" + tmpPath + "'");
        }
    }

    const std::error_code ec = llvm::sys::fs::rename(tmpPath, path);
    if (ec) {
        (void)llvm::sys::fs::remove(tmpPath);
        fail("Failed to replace '" + path + "': " + ec.message());
    }
}

void saveTomlAtomicOrFail(const std::string& path, const toml::table& tbl) {
    std::ostringstream ss;
    ss << tbl << "\n";
    writeTextFileAtomicOrFail(path, ss.str());
}

// =============================================================================
// TOML Parsing Helpers
// =============================================================================
std::optional<toml::table> parseTomlIfExists(const std::string& path, bool warnOnParseError = true) {
    if (!llvm::sys::fs::exists(path)) {
        return std::nullopt;
    }

    auto parsed = toml::parse_file(path);
    if (!parsed) {
        if (warnOnParseError) {
            std::ostringstream oss;
            oss << parsed.error();
            warn("Failed to parse " + path + ": " + oss.str());
        }
        return std::nullopt;
    }

    return std::move(parsed).table();
}

toml::table parseTomlOrFail(const std::string& path) {
    auto parsed = toml::parse_file(path);
    if (!parsed) {
        std::ostringstream oss;
        oss << parsed.error();
        fail("Failed to parse " + path + ": " + oss.str());
    }

    return std::move(parsed).table();
}

// =============================================================================
// TOML Value Helpers
// =============================================================================
std::optional<std::string> scalarNodeToString(const toml::node& n) {
    if (auto* s = n.as_string()) {
        return s->get();
    }

    if (auto* i = n.as_integer()) {
        return std::to_string(i->get());
    }

    if (auto* b = n.as_boolean()) {
        return b->get() ? "true" : "false";
    }

    if (auto* f = n.as_floating_point()) {
        return std::to_string(f->get());
    }

    return std::nullopt;
}

void flattenToml(
    const toml::table& tbl,
    const std::string& prefix,
    std::vector<std::pair<std::string, std::string>>& out
) {
    for (auto&& [k, v] : tbl) {
        const std::string key = prefix.empty()
            ? std::string(k.str())
            : (prefix + "." + std::string(k.str()));

        if (auto* t = v.as_table()) {
            flattenToml(*t, key, out);
            continue;
        }

        if (auto val = scalarNodeToString(v)) {
            out.emplace_back(key, *val);
        } else {
            out.emplace_back(key, "[complex value]");
        }
    }
}

toml::table* ensurePathTables(
    toml::table& root,
    const std::vector<std::string>& parts,
    std::size_t countForTables
) {
    toml::table* cur = &root;

    for (std::size_t i = 0; i < countForTables; ++i) {
        const std::string& seg = parts[i];

        auto* existing = cur->get(seg);
        if (!existing || !existing->is_table()) {
            cur->insert_or_assign(seg, toml::table{});
        }

        cur = cur->get_as<toml::table>(seg);
        if (!cur) {
            fail("Failed to create config table path segment '" + seg + "'");
        }
    }

    return cur;
}

toml::table& ensureTableKey(toml::table& root, const std::string& key) {
    auto* n = root.get(key);
    if (!n || !n->is_table()) {
        root.insert_or_assign(key, toml::table{});
    }

    auto* t = root.get_as<toml::table>(key);
    if (!t) {
        fail("Failed to create table '" + key + "'");
    }

    return *t;
}

toml::table* getSecretRecordTable(toml::table& root, const std::string& section, const std::string& key) {
    auto* sec = root.get_as<toml::table>(section);
    if (!sec) {
        return nullptr;
    }
    return sec->get_as<toml::table>(key);
}

const toml::table* getSecretRecordTable(const toml::table& root, const std::string& section, const std::string& key) {
    auto* sec = root.get_as<toml::table>(section);
    if (!sec) {
        return nullptr;
    }
    return sec->get_as<toml::table>(key);
}

} // namespace

// =============================================================================
// Cross-Platform Secure Password Prompt (with '*' echo)
// =============================================================================
std::string promptForPassword(const std::string& promptText) {
    llvm::outs() << promptText << ": ";
    llvm::outs().flush();

    std::string password;

#if defined(_WIN32)
    for (;;) {
        const int raw = _getch();
        if (raw == '\r') {
            break;
        }

        if (raw == 3) {
            llvm::outs() << "\n";
            fail("Interrupted.");
        }

        if (raw == '\b') {
            if (!password.empty()) {
                password.pop_back();
                llvm::outs() << "\b \b";
                llvm::outs().flush();
            }
            continue;
        }

        if (raw == 0 || raw == 224) {
            (void)_getch();
            continue;
        }

        password.push_back(static_cast<char>(raw));
        llvm::outs() << '*';
        llvm::outs().flush();
    }

    llvm::outs() << "\n";
#else
    struct termios oldt {};
    if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
        std::string fallback;
        std::getline(std::cin, fallback);
        return fallback;
    }

    struct TermRestore {
        termios saved {};
        bool active = false;

        ~TermRestore() {
            if (active) {
                tcsetattr(STDIN_FILENO, TCSANOW, &saved);
            }
        }
    } restore;

    restore.saved = oldt;
    restore.active = true;

    struct termios newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);

    if (tcsetattr(STDIN_FILENO, TCSANOW, &newt) != 0) {
        restore.active = false;
        std::string fallback;
        std::getline(std::cin, fallback);
        return fallback;
    }

    char ch = '\0';
    while (read(STDIN_FILENO, &ch, 1) == 1) {
        if (ch == '\n' || ch == '\r') {
            break;
        }

        if (ch == 3) {
            llvm::outs() << "\n";
            fail("Interrupted.");
        }

        if (ch == 127 || ch == '\b') {
            if (!password.empty()) {
                password.pop_back();
                llvm::outs() << "\b \b";
                llvm::outs().flush();
            }
            continue;
        }

        password.push_back(ch);
        llvm::outs() << '*';
        llvm::outs().flush();
    }

    llvm::outs() << "\n";
#endif

    return password;
}

// =============================================================================
// Directory & Path Resolution
// =============================================================================
std::string GlobalConfig::getArknetDir() {
    llvm::SmallString<256> arkDir;

    auto envOrEmpty = [](const char* key) -> std::string {
        if (const char* v = std::getenv(key)) {
            if (*v) {
                return std::string(v);
            }
        }
        return {};
    };

    // 1) Explicit override wins
    if (std::string v = envOrEmpty("ARKNET_HOME"); !v.empty()) {
        arkDir = v;
    }
#if !defined(_WIN32)
    // 2) XDG config base (Linux/macOS)
    else if (std::string xdg = envOrEmpty("XDG_CONFIG_HOME"); !xdg.empty()) {
        arkDir = xdg;
        llvm::sys::path::append(arkDir, "arknet");
    }
#endif
    // 3) Standard home directory (LLVM)
    else {
        llvm::SmallString<256> homeDir;
        if (!llvm::sys::path::home_directory(homeDir) && !homeDir.empty()) {
            arkDir = homeDir;
            llvm::sys::path::append(arkDir, ".arknet");
        } else {
#if defined(_WIN32)
            if (std::string profile = envOrEmpty("USERPROFILE"); !profile.empty()) {
                arkDir = profile;
                llvm::sys::path::append(arkDir, ".arknet");
            } else {
                llvm::SmallString<256> cwd;
                if (llvm::sys::fs::current_path(cwd)) {
                    fail("Unable to determine home directory or current directory.");
                }

                warn("Home directory unavailable; using local ./.arknet");
                arkDir = cwd;
                llvm::sys::path::append(arkDir, ".arknet");
            }
#else
            if (std::string home = envOrEmpty("HOME"); !home.empty()) {
                arkDir = home;
                llvm::sys::path::append(arkDir, ".arknet");
            } else {
                llvm::SmallString<256> cwd;
                if (llvm::sys::fs::current_path(cwd)) {
                    fail("Unable to determine home directory or current directory.");
                }

                warn("Home directory unavailable; using local ./.arknet");
                arkDir = cwd;
                llvm::sys::path::append(arkDir, ".arknet");
            }
#endif
        }
    }

    const std::error_code ec = llvm::sys::fs::create_directories(arkDir);
    if (ec) {
        fail("Failed to create Arknet config directory '" + std::string(arkDir.str()) + "': " + ec.message());
    }

    const std::error_code permEc = llvm::sys::fs::setPermissions(arkDir, llvm::sys::fs::owner_all);
    if (permEc) {
        warn("Failed to set permissions on Arknet config directory: " + permEc.message());
    }

    return std::string(arkDir.str());
}

std::string GlobalConfig::getConfigFilePath() {
    llvm::SmallString<256> path(getArknetDir());
    llvm::sys::path::append(path, "config.toml");
    return std::string(path.str());
}

std::string GlobalConfig::getCredentialsFilePath() {
    llvm::SmallString<256> path(getArknetDir());
    llvm::sys::path::append(path, "credentials.toml");
    return std::string(path.str());
}

// =============================================================================
// Standard Config (config.toml)
// =============================================================================
std::optional<std::string> GlobalConfig::get(const std::string& key) {
    if (key.empty()) {
        return std::nullopt;
    }

    const std::string path = getConfigFilePath();
    auto tblOpt = parseTomlIfExists(path);
    if (!tblOpt) {
        return std::nullopt;
    }

    const auto parts = splitKey(key);

    toml::node* cur = nullptr;
    if (parts.size() == 1) {
        auto* core = tblOpt->get_as<toml::table>("core");
        if (!core) {
            return std::nullopt;
        }

        cur = core->get(parts[0]);
        if (!cur) {
            return std::nullopt;
        }
    } else {
        cur = &*tblOpt;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            auto* t = cur->as_table();
            if (!t) {
                return std::nullopt;
            }

            cur = t->get(parts[i]);
            if (!cur) {
                return std::nullopt;
            }
        }
    }

    if (auto val = scalarNodeToString(*cur)) {
        return val;
    }

    return std::nullopt;
}

void GlobalConfig::set(const std::string& key, const std::string& value) {
    if (key.empty()) {
        fail("Config key cannot be empty.");
    }

    const std::string path = getConfigFilePath();

    toml::table tbl;
    if (auto tblOpt = parseTomlIfExists(path)) {
        tbl = std::move(*tblOpt);
    }

    const auto parts = splitKey(key);
    if (parts.empty()) {
        fail("Config key cannot be empty.");
    }

    toml::table* parent = nullptr;
    std::string param;

    if (parts.size() == 1) {
        if (!tbl.contains("core") || !tbl["core"].is_table()) {
            tbl.insert_or_assign("core", toml::table{});
        }

        parent = tbl["core"].as_table();
        param = parts[0];
    } else {
        parent = ensurePathTables(tbl, parts, parts.size() - 1);
        param = parts.back();
    }

    if (!parent) {
        fail("Failed to resolve target config table for key '" + key + "'");
    }

    parent->insert_or_assign(param, value);
    saveTomlAtomicOrFail(path, tbl);
}

void GlobalConfig::list() {
    const std::string path = getConfigFilePath();
    if (!llvm::sys::fs::exists(path)) {
        llvm::outs() << "No global configuration found.\n";
        return;
    }

    auto tblOpt = parseTomlIfExists(path);
    if (!tblOpt) {
        fail("Failed to parse config: " + path);
    }

    std::vector<std::pair<std::string, std::string>> flat;
    flattenToml(*tblOpt, "", flat);

    std::sort(flat.begin(), flat.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (const auto& [k, v] : flat) {
        llvm::outs() << k << " = " << v << "\n";
    }
}

// =============================================================================
// Secure Credentials (credentials.toml)
// =============================================================================
bool GlobalConfig::hasEncryptedSecret(const std::string& section, const std::string& key) {
    const std::string path = getCredentialsFilePath();
    if (!llvm::sys::fs::exists(path)) {
        return false;
    }

    auto tblOpt = parseTomlIfExists(path, false);
    if (!tblOpt) {
        return true;
    }

    if (const auto* rec = getSecretRecordTable(*tblOpt, section, key)) {
        auto* ct = rec->get_as<std::string>("ciphertext");
        return ct && !ct->get().empty();
    }

    // Legacy compatibility: [auth] ciphertext=...
    if (section == "auth" && key == "provider_token") {
        if (auto* auth = tblOpt->get_as<toml::table>("auth")) {
            if (auto* ct = auth->get_as<std::string>("ciphertext")) {
                return !ct->get().empty();
            }
        }
    }

    return false;
}

void GlobalConfig::setEncryptedSecret(
    const std::string& section,
    const std::string& key,
    const std::string& value,
    const std::string& masterPassword
) {
    if (section.empty()) {
        fail("Secret section cannot be empty.");
    }

    if (key.empty()) {
        fail("Secret key cannot be empty.");
    }

    auto encryptedB64 = ark::crypto::Vault::encrypt(value, masterPassword);
    if (!encryptedB64) {
        fail("Cryptographic engine failed to encrypt secret.");
    }

    const std::string path = getCredentialsFilePath();

    toml::table tbl;
    if (auto tblOpt = parseTomlIfExists(path, false)) {
        tbl = std::move(*tblOpt);
    }

    toml::table& sec = ensureTableKey(tbl, section);
    sec.insert_or_assign(key, toml::table{
        {"algorithm", "argon2id-xsalsa20-poly1305"},
        {"ciphertext", *encryptedB64}
    });

    saveTomlAtomicOrFail(path, tbl);

    const std::error_code permEc = llvm::sys::fs::setPermissions(
        path,
        llvm::sys::fs::owner_read | llvm::sys::fs::owner_write
    );

    if (permEc) {
        warn("Failed to tighten permissions on credentials file: " + permEc.message());
    }
}

std::optional<std::string> GlobalConfig::getDecryptedSecret(
    const std::string& section,
    const std::string& key,
    const std::string& masterPassword
) {
    if (section.empty() || key.empty()) {
        return std::nullopt;
    }

    const std::string path = getCredentialsFilePath();
    if (!llvm::sys::fs::exists(path)) {
        return std::nullopt;
    }

    auto tblOpt = parseTomlIfExists(path, false);
    if (!tblOpt) {
        return std::nullopt;
    }

    if (const auto* rec = getSecretRecordTable(*tblOpt, section, key)) {
        if (auto* ct = rec->get_as<std::string>("ciphertext")) {
            return ark::crypto::Vault::decrypt(ct->get(), masterPassword);
        }
    }

    // Legacy compatibility: [auth] ciphertext=...
    if (section == "auth" && key == "provider_token") {
        if (auto* auth = tblOpt->get_as<toml::table>("auth")) {
            if (auto* ct = auth->get_as<std::string>("ciphertext")) {
                return ark::crypto::Vault::decrypt(ct->get(), masterPassword);
            }
        }
    }

    return std::nullopt;
}

bool GlobalConfig::hasToken() {
    return hasEncryptedSecret("auth", "provider_token");
}

void GlobalConfig::setEncryptedToken(const std::string& token, const std::string& masterPassword) {
    setEncryptedSecret("auth", "provider_token", token, masterPassword);
}

std::optional<std::string> GlobalConfig::getDecryptedToken(const std::string& masterPassword) {
    return getDecryptedSecret("auth", "provider_token", masterPassword);
}

// =============================================================================
// CLI Handlers
// =============================================================================
struct ConfigOptions {
    std::string key;
    std::string value;
    bool list = false;
    bool login = false;
    bool status = false;
};

void setupConfigCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("config", "Manage global Arknet configuration and credentials");

    auto opts = std::make_shared<ConfigOptions>();

    sub->add_flag("-l,--list", opts->list, "List all configurations");
    sub->add_flag("--login", opts->login, "Authenticate and securely store a provider token");
    sub->add_flag("--status", opts->status, "Check authentication vault status");

    sub->add_option("key", opts->key, "Config key (e.g., user.name)");
    sub->add_option("value", opts->value, "Config value to set");

    sub->callback([sub, opts]() {
        const int modeCount =
            (opts->list ? 1 : 0) +
            (opts->login ? 1 : 0) +
            (opts->status ? 1 : 0);

        if (modeCount > 1) {
            fail("Use only one of --list, --login, or --status at a time.");
        }

        if (!opts->login && !opts->status && !opts->list && opts->key.empty()) {
            printOut(sub->help());
            std::exit(1);
        }

        if (opts->login) {
            if (GlobalConfig::hasToken()) {
                warn("A provider token is already stored in your credential vault.");
                printOut("Are you sure you want to overwrite it? [y/N]: ");

                std::string resp;
                std::getline(std::cin, resp);
                if (resp.empty() || (resp[0] != 'y' && resp[0] != 'Y')) {
                    note("Login aborted. Your existing token remains active.");
                    return;
                }
            }

            printOut("Enter your Arknet Provider Token: ");
            std::string token;
            std::getline(std::cin, token);

            if (token.empty()) {
                fail("Token cannot be empty. Aborting.");
            }

            std::string pwd1 = promptForPassword("Set a master password to encrypt this token");
            std::string pwd2 = promptForPassword("Confirm master password");

            if (pwd1.empty()) {
                fail("Master password cannot be empty.");
            }

            if (pwd1 != pwd2) {
                fail("Passwords do not match. Aborting.");
            }

            GlobalConfig::setEncryptedToken(token, pwd1);
            note("Token encrypted and saved securely to ~/.arknet/credentials.toml");
            return;
        }

        if (opts->status) {
            if (GlobalConfig::hasToken()) {
                note("Authenticated. Credential vault is present.");
                std::string pwd = promptForPassword("Enter master password to test decryption");
                auto token = GlobalConfig::getDecryptedToken(pwd);
                if (token && !token->empty()) {
                    note("Decryption successful.");
                } else {
                    fail("Decryption failed. Invalid password or corrupted vault.");
                }
            } else {
                note("Not logged in. Run `arknet config --login` to authenticate.");
            }
            return;
        }

        if (opts->list) {
            GlobalConfig::list();
            return;
        }

        if (opts->key.empty()) {
            printOut(sub->help());
            std::exit(1);
        }

        if (opts->value.empty()) {
            if (auto val = GlobalConfig::get(opts->key)) {
                printOut(*val + "\n");
            }
            return;
        }

        GlobalConfig::set(opts->key, opts->value);
        note("Set " + opts->key + " successfully.");
    });
}

} // namespace ark::cli