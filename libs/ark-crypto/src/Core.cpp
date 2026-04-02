#include <ark/crypto/Core.h>

#include <sodium.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ark::crypto {

namespace {
std::once_flag g_init_flag;
}

void init() {
    std::call_once(g_init_flag, [] {
        if (sodium_init() < 0) {
            throw std::runtime_error("FATAL: sodium_init failed. CPU unsupported?");
        }
    });
}

SecureBytes::SecureBytes(size_t size) : _ptr(nullptr), _size(size) {
    if (_size == 0) {
        return;
    }

    _ptr = static_cast<uint8_t*>(sodium_malloc(_size));
    if (_ptr == nullptr) {
        throw std::bad_alloc();
    }

    if (sodium_mlock(_ptr, _size) != 0) {
        sodium_free(_ptr);
        _ptr = nullptr;
        _size = 0;
        throw std::runtime_error("FATAL: sodium_mlock failed. Check 'ulimit -l'. Secrets unsafe on disk.");
    }
}

SecureBytes::SecureBytes(std::span<const uint8_t> data) : SecureBytes(data.size()) {
    if (_size != 0) {
        std::memcpy(_ptr, data.data(), _size);
    }
}

SecureBytes::~SecureBytes() {
    if (_ptr == nullptr) {
        return;
    }

    if (sodium_munlock(_ptr, _size) != 0) {
        std::cerr << "FATAL: sodium_munlock failed. Memory state corrupted." << std::endl;
        std::terminate();
    }

    sodium_free(_ptr);
}

SecureBytes::SecureBytes(SecureBytes&& other) noexcept
    : _ptr(other._ptr), _size(other._size) {
    other._ptr = nullptr;
    other._size = 0;
}

SecureBytes& SecureBytes::operator=(SecureBytes&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (_ptr != nullptr) {
        if (sodium_munlock(_ptr, _size) != 0) {
            std::cerr << "FATAL: sodium_munlock failed during move." << std::endl;
            std::terminate();
        }
        sodium_free(_ptr);
    }

    _ptr = other._ptr;
    _size = other._size;

    other._ptr = nullptr;
    other._size = 0;

    return *this;
}

std::string SecureBytes::to_hex() const {
    return ark::crypto::to_hex(span());
}

void wipe(void* ptr, size_t size) {
    if (ptr != nullptr && size != 0) {
        sodium_memzero(ptr, size);
    }
}

SecureBytes random_bytes(size_t size) {
    SecureBytes out(size);
    if (size != 0) {
        randombytes_buf(out.data(), size);
    }
    return out;
}

void random_fill(void* ptr, size_t size) {
    if (ptr != nullptr && size != 0) {
        randombytes_buf(ptr, size);
    }
}

std::string to_hex(std::span<const uint8_t> data) {
    if (data.empty()) {
        return {};
    }

    std::string out(data.size() * 2U + 1U, '\0');
    if (sodium_bin2hex(out.data(), out.size(), data.data(), data.size()) == nullptr) {
        throw std::runtime_error("Hex encoding failed");
    }

    out.resize(data.size() * 2U);
    return out;
}

std::vector<uint8_t> from_hex(std::string_view hex) {
    if (hex.empty()) {
        return {};
    }

    if ((hex.size() % 2U) != 0U) {
        throw std::runtime_error("Hex length must be even");
    }

    std::vector<uint8_t> out(hex.size() / 2U);
    size_t bin_len = 0;
    const char* end = nullptr;

    if (sodium_hex2bin(
            out.data(),
            out.size(),
            hex.data(),
            hex.size(),
            nullptr,
            &bin_len,
            &end) != 0) {
        throw std::runtime_error("Invalid hex char");
    }

    if (bin_len != out.size() || end != (hex.data() + hex.size())) {
        throw std::runtime_error("Malformed hex string");
    }

    return out;
}

std::string to_b64url(std::span<const uint8_t> data) {
    if (data.empty()) {
        return {};
    }

    const size_t encoded_len =
        sodium_base64_ENCODED_LEN(data.size(), sodium_base64_VARIANT_URLSAFE_NO_PADDING);

    std::string out(encoded_len, '\0');
    if (sodium_bin2base64(
            out.data(),
            out.size(),
            data.data(),
            data.size(),
            sodium_base64_VARIANT_URLSAFE_NO_PADDING) == nullptr) {
        throw std::runtime_error("Base64 encoding failed");
    }

    out.resize(std::strlen(out.c_str()));
    return out;
}

std::vector<uint8_t> from_b64url(std::string_view b64) {
    if (b64.empty()) {
        return {};
    }

    for (const char c : b64) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc) != 0 || std::iscntrl(uc) != 0) {
            throw std::runtime_error("Invalid b64: contains whitespace/control");
        }
    }

    std::vector<uint8_t> out(b64.size());
    size_t bin_len = 0;

    if (sodium_base642bin(
            out.data(),
            out.size(),
            b64.data(),
            b64.size(),
            nullptr,
            &bin_len,
            nullptr,
            sodium_base64_VARIANT_URLSAFE_NO_PADDING) != 0) {
        throw std::runtime_error("Invalid base64 string");
    }

    out.resize(bin_len);
    return out;
}

template class FixedSecret<64>;
template class FixedSecret<32>;

} // namespace ark::crypto