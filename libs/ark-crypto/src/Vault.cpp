// libs/ark-crypto/Vault.cpp
#include "ark/crypto/Vault.h"
#include <sodium.h>
#include <vector>
#include <cstring>

namespace ark::crypto {

// Helper to ensure libsodium is initialized safely exactly once
static bool initSodium() {
    static bool initialized = false;
    static bool success = false;
    if (!initialized) {
        success = (sodium_init() >= 0);
        initialized = true;
    }
    return success;
}

std::optional<std::string> Vault::encrypt(const std::string& plaintext, const std::string& password) {
    if (!initSodium()) return std::nullopt;

    // 1. Generate random Salt for Argon2id
    unsigned char salt[crypto_pwhash_SALTBYTES];
    randombytes_buf(salt, sizeof(salt));

    // 2. Derive Key using Argon2id
    unsigned char key[crypto_secretbox_KEYBYTES];
    if (crypto_pwhash(key, sizeof(key),
                      password.c_str(), password.length(),
                      salt,
                      crypto_pwhash_OPSLIMIT_INTERACTIVE,
                      crypto_pwhash_MEMLIMIT_INTERACTIVE,
                      crypto_pwhash_ALG_ARGON2ID13) != 0) {
        return std::nullopt; // Usually implies out of memory
    }

    // 3. Generate random Nonce for XSalsa20
    unsigned char nonce[crypto_secretbox_NONCEBYTES];
    randombytes_buf(nonce, sizeof(nonce));

    // 4. Encrypt the plaintext (includes Poly1305 MAC)
    std::vector<unsigned char> ciphertext(plaintext.length() + crypto_secretbox_MACBYTES);
    crypto_secretbox_easy(ciphertext.data(),
                          reinterpret_cast<const unsigned char*>(plaintext.c_str()),
                          plaintext.length(),
                          nonce,
                          key);

    // Securely wipe the derived key from memory
    sodium_memzero(key, sizeof(key));

    // 5. Pack payload: [Salt (16)] + [Nonce (24)] + [Ciphertext + MAC]
    std::vector<unsigned char> packed;
    packed.reserve(sizeof(salt) + sizeof(nonce) + ciphertext.size());
    packed.insert(packed.end(), salt, salt + sizeof(salt));
    packed.insert(packed.end(), nonce, nonce + sizeof(nonce));
    packed.insert(packed.end(), ciphertext.begin(), ciphertext.end());

    // 6. Base64 Encode
    size_t b64_maxlen = sodium_base64_ENCODED_LEN(packed.size(), sodium_base64_VARIANT_ORIGINAL);
    std::string b64_out(b64_maxlen, '\0');
    sodium_bin2base64(b64_out.data(), b64_maxlen,
                      packed.data(), packed.size(),
                      sodium_base64_VARIANT_ORIGINAL);

    // Remove trailing nulls added by libsodium's C-string encoding
    b64_out.resize(strlen(b64_out.c_str()));

    return b64_out;
}

std::optional<std::string> Vault::decrypt(const std::string& base64Payload, const std::string& password) {
    if (!initSodium()) return std::nullopt;

    // 1. Base64 Decode
    size_t bin_maxlen = base64Payload.length() * 3 / 4 + 1;
    std::vector<unsigned char> packed(bin_maxlen);
    size_t packed_len = 0;

    if (sodium_base642bin(packed.data(), bin_maxlen,
                          base64Payload.c_str(), base64Payload.length(),
                          nullptr, &packed_len,
                          nullptr, sodium_base64_VARIANT_ORIGINAL) != 0) {
        return std::nullopt; // Invalid Base64
    }
    packed.resize(packed_len);

    // 2. Validate Minimum Length (Salt + Nonce + MAC)
    const size_t min_len = crypto_pwhash_SALTBYTES + crypto_secretbox_NONCEBYTES + crypto_secretbox_MACBYTES;
    if (packed.size() < min_len) return std::nullopt;

    // 3. Unpack components
    const unsigned char* salt = packed.data();
    const unsigned char* nonce = salt + crypto_pwhash_SALTBYTES;
    const unsigned char* ciphertext = nonce + crypto_secretbox_NONCEBYTES;
    size_t ciphertext_len = packed.size() - crypto_pwhash_SALTBYTES - crypto_secretbox_NONCEBYTES;

    // 4. Derive Key using the extracted Salt
    unsigned char key[crypto_secretbox_KEYBYTES];
    if (crypto_pwhash(key, sizeof(key),
                      password.c_str(), password.length(),
                      salt,
                      crypto_pwhash_OPSLIMIT_INTERACTIVE,
                      crypto_pwhash_MEMLIMIT_INTERACTIVE,
                      crypto_pwhash_ALG_ARGON2ID13) != 0) {
        return std::nullopt;
    }

    // 5. Decrypt and Authenticate
    std::string plaintext(ciphertext_len - crypto_secretbox_MACBYTES, '\0');
    if (crypto_secretbox_open_easy(reinterpret_cast<unsigned char*>(plaintext.data()),
                                   ciphertext,
                                   ciphertext_len,
                                   nonce,
                                   key) != 0) {
        sodium_memzero(key, sizeof(key));
        return std::nullopt; // Forgery detected or wrong password!
    }

    // Securely wipe key
    sodium_memzero(key, sizeof(key));
    
    return plaintext;
}

} // namespace ark::crypto