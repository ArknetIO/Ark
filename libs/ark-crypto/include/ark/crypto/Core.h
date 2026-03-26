#pragma once
#include <vector>
#include <string>
#include <span>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ark::crypto {

// Forward declare the factory struct so the friend declaration binds correctly
struct SecretKeyFactory; 

// =========================================================
// Constants
// =========================================================
constexpr size_t HASH_SHA256_BYTES    = 32;
constexpr size_t HASH_BLAKE2B_BYTES   = 32;

constexpr size_t SIGN_PUBKEY_BYTES    = 32;
constexpr size_t SIGN_SECKEY_BYTES    = 64;
constexpr size_t SIGN_SIG_BYTES       = 64;
constexpr size_t SIGN_SEED_BYTES      = 32;

constexpr size_t BOX_PUBKEY_BYTES     = 32;
constexpr size_t BOX_SECKEY_BYTES     = 32;
constexpr size_t BOX_MAC_BYTES        = 16;
constexpr size_t SEAL_BYTES           = 48;

constexpr size_t SECRETBOX_KEY_BYTES  = 32;
constexpr size_t SECRETBOX_NONCE_BYTES= 24;
constexpr size_t SECRETBOX_MAC_BYTES  = 16;

// =========================================================
// Secure Memory (Guard Pages + Anti-Swap)
// =========================================================
class SecureBytes {
public:
    SecureBytes() : _ptr(nullptr), _size(0) {}
    explicit SecureBytes(size_t size);
    SecureBytes(std::span<const uint8_t> data);
    ~SecureBytes();

    // No Copy
    SecureBytes(const SecureBytes&) = delete;
    SecureBytes& operator=(const SecureBytes&) = delete;

    // Move Supported
    SecureBytes(SecureBytes&& other) noexcept;
    SecureBytes& operator=(SecureBytes&& other) noexcept;

    [[nodiscard]] uint8_t* data() { return _ptr; }
    [[nodiscard]] const uint8_t* data() const { return _ptr; }
    [[nodiscard]] size_t size() const { return _size; }
    [[nodiscard]] std::span<const uint8_t> span() const { return {_ptr, _size}; }
    [[nodiscard]] std::string to_hex() const;

private:
    uint8_t* _ptr;
    size_t _size;
};

// =========================================================
// Fixed-Size Secret (Strong Typed Key wrapper)
// =========================================================
template <size_t N>
class FixedSecret {
public:
    FixedSecret() : _b(N) {}
    explicit FixedSecret(std::span<const uint8_t> s) : _b(s) {}

    [[nodiscard]] const uint8_t* data() const { return _b.data(); }
    [[nodiscard]] size_t size() const { return N; }
    [[nodiscard]] std::span<const uint8_t> span() const { return _b.span(); }

private:
    // [FIX] Friend declaration must match the forward decl exactly
    friend struct SecretKeyFactory;
    
    uint8_t* data_mut() { return _b.data(); }

    SecureBytes _b;
};

// =========================================================
// Utilities
// =========================================================
void init(); 
void wipe(void* ptr, size_t size);

[[nodiscard]] SecureBytes random_bytes(size_t size);
void random_fill(void* ptr, size_t size);

[[nodiscard]] std::string to_hex(std::span<const uint8_t> data);
[[nodiscard]] std::vector<uint8_t> from_hex(std::string_view hex);
[[nodiscard]] std::string to_b64url(std::span<const uint8_t> data);
[[nodiscard]] std::vector<uint8_t> from_b64url(std::string_view b64);

} // namespace ark::crypto