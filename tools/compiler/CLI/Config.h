// tools/compiler/CLI/Config.h
#pragma once

#include <optional>
#include <string>

namespace ark::cli {

// Helper for securely reading passwords from the terminal
std::string promptForPassword(const std::string& promptText);

class GlobalConfig {
public:
    // =========================================================================
    // Directory & Path Resolution
    // =========================================================================
    static std::string getArknetDir();
    static std::string getConfigFilePath();
    static std::string getCredentialsFilePath();

    // =========================================================================
    // Standard Config (config.toml)
    // =========================================================================
    static std::optional<std::string> get(const std::string& key);
    static void set(const std::string& key, const std::string& value);
    static void list();

    // =========================================================================
    // Secure Credentials (credentials.toml)
    // =========================================================================
    static bool hasToken();
    static void setEncryptedToken(const std::string& token, const std::string& masterPassword);
    static std::optional<std::string> getDecryptedToken(const std::string& masterPassword);

    // Generic vault secrets (used by capsule sealing key, provider creds, etc.)
    static bool hasEncryptedSecret(const std::string& section, const std::string& key);
    static void setEncryptedSecret(const std::string& section,
                                   const std::string& key,
                                   const std::string& value,
                                   const std::string& masterPassword);
    static std::optional<std::string> getDecryptedSecret(const std::string& section,
                                                         const std::string& key,
                                                         const std::string& masterPassword);
};

} // namespace ark::cli