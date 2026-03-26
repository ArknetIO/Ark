#pragma once
#include "Core.h"
#include <optional>

namespace ark::crypto {

using X25519SecretKey = FixedSecret<BOX_SECKEY_BYTES>;
using SecretBoxKey    = FixedSecret<SECRETBOX_KEY_BYTES>;

struct X25519Keypair {
    std::vector<uint8_t> pub; 
    X25519SecretKey sec;
};

// Factories
X25519Keypair box_keygen();

// -----------------------------------------------------------------------------
// Asymmetric (Sealed Box) - Anonymous Sender
// -----------------------------------------------------------------------------
// Encrypts a message such that only the recipient can read it.
// Sender identity is NOT integrity protected (anonymous).
[[nodiscard]]
std::vector<uint8_t> seal_encrypt(std::span<const uint8_t> msg, std::span<const uint8_t> recipient_pub);

[[nodiscard]]
std::optional<SecureBytes> seal_decrypt(std::span<const uint8_t> cipher, const X25519Keypair& keys);

// -----------------------------------------------------------------------------
// Symmetric (Secret Box) - XChaCha20-Poly1305
// -----------------------------------------------------------------------------
// Uses a random nonce (prepended to output) to encrypt data.
[[nodiscard]]
std::vector<uint8_t> secretbox_encrypt(std::span<const uint8_t> msg, const SecretBoxKey& key);

[[nodiscard]]
std::optional<SecureBytes> secretbox_decrypt(std::span<const uint8_t> packed, const SecretBoxKey& key);

} // namespace ark::crypto