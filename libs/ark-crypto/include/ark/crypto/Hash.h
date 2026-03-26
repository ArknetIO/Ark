#pragma once
#include "Core.h"
#include <vector>
#include <string>
#include <string_view>
#include <span>
#include <memory>

namespace ark::crypto {

// =========================================================
// 1. Algorithm Selection
// =========================================================
enum class HashAlgo {
    SHA256,     // Standard NIST (32 bytes)
    SHA512,     // Standard NIST (64 bytes)
    BLAKE2B,    // High-Performance (Variable 16-64 bytes)
    BLAKE3      // [NEW] Next-Gen Performance (32 bytes default, Extendable)
};

// =========================================================
// 2. The Unified Hasher
// =========================================================
class Hasher {
public:
    // -------------------------------------------------------------------------
    // One-Shot API (Stateless)
    // -------------------------------------------------------------------------
    
    // Compute hash of data using specific algorithm.
    // - BLAKE2B: defaults to 32 bytes.
    // - BLAKE3: defaults to 32 bytes.
    [[nodiscard]] 
    static std::vector<uint8_t> compute(HashAlgo algo, std::span<const uint8_t> data);

    [[nodiscard]] 
    static std::string compute_hex(HashAlgo algo, std::span<const uint8_t> data);

    // BLAKE2b Specifics: Custom length and Keyed Hashing (MAC)
    [[nodiscard]]
    static std::vector<uint8_t> compute_blake2b(
        std::span<const uint8_t> data, 
        size_t out_len = 32,
        std::span<const uint8_t> key = {} 
    );

    // [NEW] BLAKE3 Specifics: Keyed Hashing (MAC) & Derive Key
    // BLAKE3 supports 32-byte keys.
    [[nodiscard]]
    static std::vector<uint8_t> compute_blake3(
        std::span<const uint8_t> data,
        size_t out_len = 32,
        std::span<const uint8_t> key = {} 
    );

    // -------------------------------------------------------------------------
    // Streaming API (Stateful)
    // -------------------------------------------------------------------------
    
    // Construct a hasher. 
    // key: Optional. Used for BLAKE2B / BLAKE3 (Keyed mode). Ignored for SHA.
    // out_len: Optional. Used for BLAKE2B / BLAKE3. Ignored for SHA.
    explicit Hasher(HashAlgo algo, std::span<const uint8_t> key = {}, size_t out_len = 0);
    ~Hasher();

    // No Copy (Opaque state)
    Hasher(const Hasher&) = delete;
    Hasher& operator=(const Hasher&) = delete;

    // Move Supported
    Hasher(Hasher&&) noexcept;
    Hasher& operator=(Hasher&&) noexcept;

    // Ingest Data
    void update(std::span<const uint8_t> data);
    void update(std::string_view str);

    // Finalize: Returns bytes and RESETS state for reuse.
    [[nodiscard]] std::vector<uint8_t> finalize();
    
    // Finalize: Returns hex string and RESETS state.
    [[nodiscard]] std::string finalize_hex();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

// =========================================================
// 3. Password Hashing (Argon2id)
// =========================================================
struct Password {
    // Hashes password using Argon2id v1.3
    // Returns a formatted string: $argon2id$v=19$m=...,t=...,p=...
    [[nodiscard]] static std::string hash(std::string_view password);
    
    // Verifies a password against a hash string
    [[nodiscard]] static bool verify(std::string_view hash, std::string_view password);
};

} // namespace ark::crypto