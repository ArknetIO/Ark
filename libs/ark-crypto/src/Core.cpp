#include <ark/crypto/Core.h>
#include <sodium.h>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <cctype>

namespace ark::crypto {

static std::once_flag g_init_flag;

void init() {
    std::call_once(g_init_flag, []{
        if (sodium_init() < 0) {
            throw std::runtime_error("FATAL: sodium_init failed. CPU unsupported?");
        }
    });
}

// =========================================================
// SecureBytes: Nuclear-Grade Memory Hygiene
// =========================================================

SecureBytes::SecureBytes(size_t size) : _ptr(nullptr), _size(size) {
    if (size > 0) {
        // 1. Guarded Allocation (Canary + Guard Page)
        _ptr = static_cast<uint8_t*>(sodium_malloc(size));
        if (!_ptr) throw std::bad_alloc();
        
        // 2. Hard Lock (Anti-Swap)
        // Force OS to keep pages in RAM. Abort if environment is insecure.
        if (sodium_mlock(_ptr, _size) != 0) {
            sodium_free(_ptr);
            throw std::runtime_error("FATAL: sodium_mlock failed. Check 'ulimit -l'. Secrets unsafe on disk.");
        }
    }
}

SecureBytes::SecureBytes(std::span<const uint8_t> data) : SecureBytes(data.size()) {
    if (_size > 0) {
        std::memcpy(_ptr, data.data(), _size);
        // Note: mlock is already active from constructor
    }
}

SecureBytes::~SecureBytes() {
    if (_ptr) {
        // 3. Unlock & Wipe
        // sodium_munlock overwrites with zeros before unlocking.
        if (sodium_munlock(_ptr, _size) != 0) {
            // Destructors cannot throw. Terminate to prevent leakage.
            std::cerr << "FATAL: sodium_munlock failed. Memory state corrupted." << std::endl;
            std::terminate(); 
        }
        sodium_free(_ptr); 
    }
}

SecureBytes::SecureBytes(SecureBytes&& other) noexcept 
    : _ptr(other._ptr), _size(other._size) {
    other._ptr = nullptr;
    other._size = 0;
}

SecureBytes& SecureBytes::operator=(SecureBytes&& other) noexcept {
    if (this != &other) {
        // Wipe current
        if (_ptr) {
            if (sodium_munlock(_ptr, _size) != 0) {
                std::cerr << "FATAL: sodium_munlock failed during move." << std::endl;
                std::terminate();
            }
            sodium_free(_ptr);
        }
        // Steal
        _ptr = other._ptr;
        _size = other._size;
        // Neuter source
        other._ptr = nullptr;
        other._size = 0;
    }
    return *this;
}

std::string SecureBytes::to_hex() const {
    return ark::crypto::to_hex(span());
}

// =========================================================
// Utilities
// =========================================================

void wipe(void* ptr, size_t size) {
    if (ptr && size > 0) sodium_memzero(ptr, size);
}

SecureBytes random_bytes(size_t size) {
    SecureBytes b(size); // mlocked
    randombytes_buf(b.data(), size);
    return b;
}

void random_fill(void* ptr, size_t size) {
    randombytes_buf(ptr, size);
}

// =========================================================
// Encodings
// =========================================================

std::string to_hex(std::span<const uint8_t> data) {
    if (data.empty()) return "";
    std::string s(data.size() * 2 + 1, '\0');
    if (sodium_bin2hex(s.data(), s.size(), data.data(), data.size())) {
        s.resize(data.size() * 2);
        return s;
    }
    throw std::runtime_error("Hex encoding failed");
}

std::vector<uint8_t> from_hex(std::string_view hex) {
    if (hex.empty()) return {};
    if (hex.size() % 2 != 0) throw std::runtime_error("Hex length must be even");

    std::vector<uint8_t> out(hex.size() / 2);
    size_t bin_len = 0;
    const char* end = nullptr;

    if (sodium_hex2bin(out.data(), out.size(), hex.data(), hex.size(), 
                       nullptr, &bin_len, &end) != 0) {
         throw std::runtime_error("Invalid hex char");
    }
    if (bin_len != out.size() || end != hex.data() + hex.size()) {
         throw std::runtime_error("Malformed hex string");
    }
    return out;
}

std::string to_b64url(std::span<const uint8_t> data) {
    if (data.empty()) return "";
    size_t max = sodium_base64_ENCODED_LEN(data.size(), sodium_base64_VARIANT_URLSAFE_NO_PADDING);
    std::string out(max, '\0');
    if (sodium_bin2base64(out.data(), max, data.data(), data.size(), sodium_base64_VARIANT_URLSAFE_NO_PADDING)) {
        out.resize(std::strlen(out.c_str()));
        return out;
    }
    return "";
}

std::vector<uint8_t> from_b64url(std::string_view b64) {
    if (b64.empty()) return {};
    
    // Strict whitespace rejection
    for (unsigned char c : b64) {
        if (std::isspace(c) || std::iscntrl(c)) 
             throw std::runtime_error("Invalid b64: contains whitespace/control");
    }

    std::vector<uint8_t> out(b64.size());
    size_t bin_len = 0;
    
    if (sodium_base642bin(out.data(), out.size(), b64.data(), b64.size(), 
                          nullptr, &bin_len, nullptr, 
                          sodium_base64_VARIANT_URLSAFE_NO_PADDING) != 0) {
        throw std::runtime_error("Invalid base64 string");
    }
    out.resize(bin_len);
    return out;
}

// Instantiate FixedSecret templates to satisfy linker
template class FixedSecret<64>;
template class FixedSecret<32>;

} // namespace ark::crypto