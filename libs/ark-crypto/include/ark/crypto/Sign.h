#pragma once
#include "Core.h"
#include <vector>
#include <span>
#include <optional>

namespace ark::crypto {

// =========================================================
// 1. Algorithm Selection
// =========================================================
enum class SignAlgo {
    Ed25519 // High-speed, high-security, small keys (32 bytes)
};

// =========================================================
// 2. Types
// =========================================================
// Standard Ed25519 Secret Keys are 64 bytes (32-byte Seed + 32-byte PubKey)
using SigningSecretKey = FixedSecret<SIGN_SECKEY_BYTES>;

struct SigningKeypair {
    std::vector<uint8_t> pub; 
    SigningSecretKey sec;     
};

// =========================================================
// 3. The Unified Signature Provider
// =========================================================
class Signature {
public:
    // -------------------------------------------------------------------------
    // Key Generation
    // -------------------------------------------------------------------------
    
    // Generate a fresh random keypair
    [[nodiscard]] 
    static SigningKeypair keygen(SignAlgo algo = SignAlgo::Ed25519);

    // Generate deterministic keypair from a seed
    [[nodiscard]] 
    static SigningKeypair keygen_from_seed(std::span<const uint8_t> seed, SignAlgo algo = SignAlgo::Ed25519);

    // -------------------------------------------------------------------------
    // Key Derivation Utilities
    // -------------------------------------------------------------------------

    // Derive Public Key from Private Key.
    // Supports both:
    // - 32-byte Seed (derives the public key)
    // - 64-byte Full Secret Key (extracts the public key)
    [[nodiscard]]
    static std::vector<uint8_t> get_public_key(
        std::span<const uint8_t> private_key, 
        SignAlgo algo = SignAlgo::Ed25519
    );

    // Derive Full Private Key (64 bytes) from Seed (32 bytes).
    [[nodiscard]]
    static std::vector<uint8_t> get_private_key(
        std::span<const uint8_t> seed, 
        SignAlgo algo = SignAlgo::Ed25519
    );

    // -------------------------------------------------------------------------
    // Operations
    // -------------------------------------------------------------------------

    // Sign a message (Detached Mode)
    // Returns the signature bytes (64 bytes for Ed25519)
    [[nodiscard]] 
    static std::vector<uint8_t> sign(
        std::span<const uint8_t> message, 
        const SigningSecretKey& secret_key,
        SignAlgo algo = SignAlgo::Ed25519
    );

    // Overload for raw bytes vector/span
    [[nodiscard]] 
    static std::vector<uint8_t> sign(
        std::span<const uint8_t> message, 
        std::span<const uint8_t> secret_key,
        SignAlgo algo = SignAlgo::Ed25519
    );

    // Verify a signature (Detached Mode)
    // Returns true if valid, false otherwise.
    [[nodiscard]] 
    static bool verify(
        std::span<const uint8_t> signature, 
        std::span<const uint8_t> message, 
        std::span<const uint8_t> public_key,
        SignAlgo algo = SignAlgo::Ed25519
    );
};

} // namespace ark::crypto