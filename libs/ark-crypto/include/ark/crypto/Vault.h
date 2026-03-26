// libs/ark-crypto/include/ark/crypto/Vault.h
#pragma once

#include <string>
#include <optional>

namespace ark::crypto {

class Vault {
public:
    /**
     * @brief Encrypts a plaintext string using a master password.
     * Generates a random salt (Argon2id) and nonce (XSalsa20-Poly1305).
     * @return Base64 encoded string containing [salt + nonce + ciphertext], or nullopt on failure.
     */
    static std::optional<std::string> encrypt(const std::string& plaintext, const std::string& password);

    /**
     * @brief Decrypts a Base64 encoded payload using a master password.
     * @return The plaintext string, or nullopt if decryption/authentication fails.
     */
    static std::optional<std::string> decrypt(const std::string& base64Payload, const std::string& password);
};

} // namespace ark::crypto