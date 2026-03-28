#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <ark/capsule/Capsule.h>
#include <ark/capsule/Platform.h>
#include <ark/crypto/Hash.h>

namespace Crypto = ark::crypto;
using namespace ark::capsule;

#if defined(_MSC_VER)
  #pragma section(".arkstub", read)
  #define ARK_STUB_KEEP __declspec(allocate(".arkstub"))
#else
  #define ARK_STUB_KEEP __attribute__((used, section(".arkstub")))
#endif

#if !defined(_WIN32)
extern "C" char** environ;
#endif

extern "C" {
ARK_STUB_KEEP const uint8_t ARK_STUB_ROOT_PUBKEY[32] = {
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55
};
}

static bool is_key_patched() {
    static constexpr uint8_t pattern[8] = {
        0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55
    };

    return std::memcmp(ARK_STUB_ROOT_PUBKEY, pattern, sizeof(pattern)) != 0;
}

static char** current_envp() noexcept {
#if defined(_WIN32)
    char*** penv = __p__environ();
    return penv ? *penv : nullptr;
#else
    return environ;
#endif
}

static bool fits_streamoff(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max());
}

static bool fits_streamsize(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max());
}

static bool fits_size_t(uint64_t value) {
    return value <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

namespace ark::runner {

static bool read_exact(
    std::ifstream& file,
    std::vector<uint8_t>& buffer,
    uint64_t count,
    uint64_t offset
) {
    if (!fits_size_t(count) || !fits_streamsize(count) || !fits_streamoff(offset)) {
        return false;
    }

    file.clear();
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file) {
        return false;
    }

    if (count == 0) {
        buffer.clear();
        return true;
    }

    buffer.resize(static_cast<size_t>(count));
    file.read(
        reinterpret_cast<char*>(buffer.data()),
        static_cast<std::streamsize>(count)
    );

    return static_cast<uint64_t>(file.gcount()) == count;
}

int stub_entry(int argc, char** argv, char** envp) {
    if (!is_key_patched()) {
        std::cerr << "🚨 FATAL: Ark Capsule Stub is unsealed (No Root Key).\n";
        std::cerr << "   This binary was not processed by 'arkc'. Execution denied.\n";
        return 1;
    }

    const std::string self_path = ark::capsule::platform::get_self_exe_path();
    if (self_path.empty()) {
        std::cerr << "Error: Cannot determine executable path.\n";
        return 1;
    }

    std::ifstream file(self_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Cannot open self.\n";
        return 1;
    }

    const std::streamoff end_pos = file.tellg();
    if (end_pos < 0) {
        std::cerr << "Error: Failed to determine executable size.\n";
        return 1;
    }

    const uint64_t file_size = static_cast<uint64_t>(end_pos);
    if (file_size < ark::capsule::FOOTER_SIZE) {
        std::cerr << "Integrity Error: File is smaller than capsule footer.\n";
        return 1;
    }

    std::vector<uint8_t> footer_bytes;
    if (!read_exact(
            file,
            footer_bytes,
            static_cast<uint64_t>(ark::capsule::FOOTER_SIZE),
            file_size - static_cast<uint64_t>(ark::capsule::FOOTER_SIZE))) {
        std::cerr << "Error: Failed to read capsule footer.\n";
        return 1;
    }

    auto footer_opt = Codec::deserialize_footer(footer_bytes);
    if (!footer_opt) {
        std::cerr << "Integrity Error: Invalid capsule footer.\n";
        return 1;
    }

    const Footer& footer = *footer_opt;
    if (!Codec::validate_layout(footer, file_size)) {
        std::cerr << "Integrity Error: Invalid capsule layout.\n";
        return 1;
    }

    const uint64_t footer_start = file_size - static_cast<uint64_t>(ark::capsule::FOOTER_SIZE);
    if (footer.bundle_offset > footer_start || footer.payload_offset > footer.bundle_offset) {
        std::cerr << "Integrity Error: Invalid capsule offsets.\n";
        return 1;
    }

    const uint64_t capsule_size = footer_start - footer.bundle_offset;
    std::vector<uint8_t> capsule_bytes;
    if (!read_exact(file, capsule_bytes, capsule_size, footer.bundle_offset)) {
        std::cerr << "Error: Failed to read security capsule.\n";
        return 1;
    }

    std::vector<uint8_t> trusted_key(sizeof(ARK_STUB_ROOT_PUBKEY));
    std::memcpy(trusted_key.data(), ARK_STUB_ROOT_PUBKEY, trusted_key.size());

    auto plan_hash_opt = Codec::verify_capsule_v1(capsule_bytes, trusted_key);
    if (!plan_hash_opt) {
        std::cerr << "🚨 SECURITY ALERT: Capsule signature verification failed!\n";
        return 1;
    }

    const auto& expected_hash = *plan_hash_opt;

    const uint64_t payload_len = footer.bundle_offset - footer.payload_offset;
    std::vector<uint8_t> payload;
    if (!read_exact(file, payload, payload_len, footer.payload_offset)) {
        std::cerr << "Error: Failed to read payload.\n";
        return 1;
    }

    const auto calc_hash = Crypto::Hasher::compute(Crypto::HashAlgo::BLAKE3, payload);
    if (calc_hash.size() != expected_hash.size() ||
        std::memcmp(calc_hash.data(), expected_hash.data(), expected_hash.size()) != 0) {
        std::cerr << "🚨 SECURITY ALERT: Payload hash mismatch!\n";
        return 1;
    }

    char** effective_envp = envp ? envp : current_envp();
    ark::capsule::platform::exec_payload(payload, argc, argv, effective_envp);
    return 0;
}

} // namespace ark::runner

#ifdef ARK_STUB_STANDALONE
int main(int argc, char** argv) {
    return ark::runner::stub_entry(argc, argv, current_envp());
}
#endif