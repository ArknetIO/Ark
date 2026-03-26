#include <ark/crypto/Hash.h>
#include <sodium.h>
#include <stdexcept>
#include <cstring>
#include <variant>

// [NEW] Include the local BLAKE3 implementation
#include "blake3.h"

namespace ark::crypto {

// =========================================================
// PIMPL: Opaque State
// =========================================================
struct Hasher::Impl {
    HashAlgo algo;
    
    // Configuration for BLAKE2b/BLAKE3 (needed for reset)
    size_t out_len_config; // Shared for variable length output
    std::vector<uint8_t> key_storage;

    // Union-like state storage
    union State {
        crypto_hash_sha256_state sha256;
        crypto_hash_sha512_state sha512;
        crypto_generichash_state blake2b;
        blake3_hasher            blake3; // [NEW]
    } state;

    Impl(HashAlgo a, std::span<const uint8_t> key, size_t out_len) 
        : algo(a) 
    {
        // Parameter Validation
        if (algo == HashAlgo::BLAKE2B) {
            out_len_config = (out_len == 0) ? 32 : out_len;
            
            if (out_len_config < crypto_generichash_BYTES_MIN || 
                out_len_config > crypto_generichash_BYTES_MAX) {
                throw std::invalid_argument("Invalid BLAKE2b output length");
            }

            if (!key.empty()) {
                if (key.size() < crypto_generichash_KEYBYTES_MIN || 
                    key.size() > crypto_generichash_KEYBYTES_MAX) {
                    throw std::invalid_argument("Invalid BLAKE2b key length");
                }
                key_storage.assign(key.begin(), key.end());
            }
        } 
        else if (algo == HashAlgo::BLAKE3) { // [NEW] BLAKE3 Init
            out_len_config = (out_len == 0) ? 32 : out_len;
            
            if (!key.empty()) {
                if (key.size() != BLAKE3_KEY_LEN) {
                    throw std::invalid_argument("BLAKE3 keyed mode requires exactly 32 bytes");
                }
                key_storage.assign(key.begin(), key.end());
            }
        }

        reset();
    }

    void reset() {
        switch (algo) {
            case HashAlgo::SHA256:
                crypto_hash_sha256_init(&state.sha256);
                break;
            case HashAlgo::SHA512:
                crypto_hash_sha512_init(&state.sha512);
                break;
            case HashAlgo::BLAKE2B:
                crypto_generichash_init(&state.blake2b, 
                    key_storage.empty() ? nullptr : key_storage.data(), 
                    key_storage.size(), 
                    out_len_config);
                break;
            case HashAlgo::BLAKE3: // [NEW]
                if (!key_storage.empty()) {
                    blake3_hasher_init_keyed(&state.blake3, key_storage.data());
                } else {
                    blake3_hasher_init(&state.blake3);
                }
                break;
        }
    }
};

// =========================================================
// Lifecycle
// =========================================================

Hasher::Hasher(HashAlgo algo, std::span<const uint8_t> key, size_t out_len) 
    : _impl(std::make_unique<Impl>(algo, key, out_len)) 
{}

Hasher::~Hasher() = default;

Hasher::Hasher(Hasher&&) noexcept = default;
Hasher& Hasher::operator=(Hasher&&) noexcept = default;

// =========================================================
// Streaming API
// =========================================================

void Hasher::update(std::span<const uint8_t> data) {
    switch (_impl->algo) {
        case HashAlgo::SHA256:
            crypto_hash_sha256_update(&_impl->state.sha256, data.data(), data.size());
            break;
        case HashAlgo::SHA512:
            crypto_hash_sha512_update(&_impl->state.sha512, data.data(), data.size());
            break;
        case HashAlgo::BLAKE2B:
            crypto_generichash_update(&_impl->state.blake2b, data.data(), data.size());
            break;
        case HashAlgo::BLAKE3: // [NEW]
            blake3_hasher_update(&_impl->state.blake3, data.data(), data.size());
            break;
    }
}

void Hasher::update(std::string_view str) {
    update(std::span<const uint8_t>{reinterpret_cast<const uint8_t*>(str.data()), str.size()});
}

std::vector<uint8_t> Hasher::finalize() {
    std::vector<uint8_t> out;
    
    switch (_impl->algo) {
        case HashAlgo::SHA256:
            out.resize(crypto_hash_sha256_BYTES);
            crypto_hash_sha256_final(&_impl->state.sha256, out.data());
            break;
        case HashAlgo::SHA512:
            out.resize(crypto_hash_sha512_BYTES);
            crypto_hash_sha512_final(&_impl->state.sha512, out.data());
            break;
        case HashAlgo::BLAKE2B:
            out.resize(_impl->out_len_config);
            crypto_generichash_final(&_impl->state.blake2b, out.data(), out.size());
            break;
        case HashAlgo::BLAKE3: // [NEW]
            out.resize(_impl->out_len_config);
            blake3_hasher_finalize(&_impl->state.blake3, out.data(), out.size());
            break;
    }
    
    _impl->reset();
    return out;
}

std::string Hasher::finalize_hex() {
    return to_hex(finalize());
}

// =========================================================
// One-Shot API
// =========================================================

std::vector<uint8_t> Hasher::compute(HashAlgo algo, std::span<const uint8_t> data) {
    std::vector<uint8_t> out;
    switch (algo) {
        case HashAlgo::SHA256:
            out.resize(crypto_hash_sha256_BYTES);
            crypto_hash_sha256(out.data(), data.data(), data.size());
            break;
        case HashAlgo::SHA512:
            out.resize(crypto_hash_sha512_BYTES);
            crypto_hash_sha512(out.data(), data.data(), data.size());
            break;
        case HashAlgo::BLAKE2B:
            out.resize(32); // Default
            crypto_generichash(out.data(), 32, data.data(), data.size(), nullptr, 0);
            break;
        case HashAlgo::BLAKE3: // [NEW]
            out.resize(32); // Default
            {
                blake3_hasher hasher;
                blake3_hasher_init(&hasher);
                blake3_hasher_update(&hasher, data.data(), data.size());
                blake3_hasher_finalize(&hasher, out.data(), 32);
            }
            break;
    }
    return out;
}

std::string Hasher::compute_hex(HashAlgo algo, std::span<const uint8_t> data) {
    return to_hex(compute(algo, data));
}

std::vector<uint8_t> Hasher::compute_blake2b(std::span<const uint8_t> data, size_t out_len, std::span<const uint8_t> key) {
    if (out_len < crypto_generichash_BYTES_MIN || out_len > crypto_generichash_BYTES_MAX) {
        throw std::invalid_argument("Invalid BLAKE2b length");
    }
    std::vector<uint8_t> out(out_len);
    crypto_generichash(out.data(), out_len, 
                       data.data(), data.size(), 
                       key.empty() ? nullptr : key.data(), key.size());
    return out;
}

// [NEW] BLAKE3 Specific Helper
std::vector<uint8_t> Hasher::compute_blake3(std::span<const uint8_t> data, size_t out_len, std::span<const uint8_t> key) {
    std::vector<uint8_t> out(out_len);
    blake3_hasher hasher;
    
    if (!key.empty()) {
        if (key.size() != BLAKE3_KEY_LEN) {
            throw std::invalid_argument("BLAKE3 keyed mode requires exactly 32 bytes");
        }
        blake3_hasher_init_keyed(&hasher, key.data());
    } else {
        blake3_hasher_init(&hasher);
    }
    
    blake3_hasher_update(&hasher, data.data(), data.size());
    blake3_hasher_finalize(&hasher, out.data(), out_len);
    return out;
}

// =========================================================
// Password API
// =========================================================

std::string Password::hash(std::string_view password) {
    init(); // Ensure sodium is up
    std::string out(crypto_pwhash_STRBYTES, '\0');
    if (crypto_pwhash_str(out.data(), password.data(), password.size(), 
                          crypto_pwhash_OPSLIMIT_MODERATE, 
                          crypto_pwhash_MEMLIMIT_MODERATE) != 0) {
        throw std::runtime_error("Argon2 hashing failed");
    }
    out.resize(std::strlen(out.c_str()));
    return out;
}

bool Password::verify(std::string_view hash, std::string_view password) {
    if (hash.empty()) return false;
    std::string safe_hash(hash); 
    return crypto_pwhash_str_verify(safe_hash.c_str(), password.data(), password.size()) == 0;
}

} // namespace ark::crypto