#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

// Platform & Crypto Libraries
#include <ark/crypto/Hash.h>      // Checking Payload Integrity
#include <ark/capsule/Capsule.h>  // ABI & Codec
#include <ark/capsule/Platform.h> // Exec & Path

namespace Crypto = ark::crypto;
using namespace ark::capsule;

// -----------------------------------------------------------------------------
// PRODUCTION KEY PLACEHOLDER (The "Seal")
// -----------------------------------------------------------------------------
// We initialize this with a specific MAGIC pattern (32 bytes).
// The Compiler (arkc) searches for this pattern in the binary and overwrites
// it with the actual Root Public Key during the "Sealing" phase.
// -----------------------------------------------------------------------------
extern "C" {
    // Keep this symbol visible so we can find it easily if needed, 
    // though distinct pattern matching is safer against strip.
    volatile const uint8_t ARK_STUB_ROOT_PUBKEY[32] __attribute__((used)) = {
        0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
        0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
        0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
        0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55
    };
}

// Helper for Key Verification: Check if we are still running with the placeholder
static bool is_key_patched() {
    // If the first 8 bytes are still the pattern, we define it as unpatched.
    // (Real Ed25519 keys are statistically impossible to match this pattern)
    const uint8_t pattern[] = {0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55};
    return std::memcmp((const void*)ARK_STUB_ROOT_PUBKEY, pattern, 8) != 0;
}

namespace ark::runner {

// -----------------------------------------------------------------------------
// Helpers (Internal)
// -----------------------------------------------------------------------------
static bool read_exact(std::ifstream& f, std::vector<uint8_t>& buf, size_t count, uint64_t offset) {
    f.seekg(offset);
    if (f.fail()) return false;
    buf.resize(count);
    f.read((char*)buf.data(), count);
    return f.good() && (size_t)f.gcount() == count;
}

// -----------------------------------------------------------------------------
// Stub Entry Logic
// -----------------------------------------------------------------------------
// This function contains the logic to verify and run a capsule.
// -----------------------------------------------------------------------------
int stub_entry(int argc, char** argv, char** envp) {
    // 1. Security Sanity Check
    if (!is_key_patched()) {
        std::cerr << "🚨 FATAL: Ark Capsule Stub is unsealed (No Root Key).\n";
        std::cerr << "   This binary was not processed by 'arkc'. Execution denied.\n";
        return 1;
    }

    std::string selfPath = ark::capsule::platform::get_self_exe_path();
    if (selfPath.empty()) {
        std::cerr << "Error: Cannot determine executable path.\n";
        return 1;
    }
    
    std::ifstream file(selfPath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Cannot open self.\n";
        return 1;
    }

    uint64_t fileSize = file.tellg();
    if (fileSize < ark::capsule::FOOTER_SIZE) return 1;

    // 2. Read Footer
    std::vector<uint8_t> footerBytes(ark::capsule::FOOTER_SIZE);
    file.seekg(fileSize - ark::capsule::FOOTER_SIZE);
    file.read((char*)footerBytes.data(), ark::capsule::FOOTER_SIZE);
    if (!file || (size_t)file.gcount() != ark::capsule::FOOTER_SIZE) return 1;

    auto footerOpt = Codec::deserialize_footer(footerBytes);
    if (!footerOpt) return 1;
    const Footer& footer = *footerOpt;

    // 3. Validate Layout (Physical)
    if (!Codec::validate_layout(footer, fileSize)) {
        std::cerr << "Integrity Error: Invalid Capsule Layout.\n";
        return 1;
    }

    // 4. Read Capsule Blob (ABI)
    uint64_t footerStart = fileSize - ark::capsule::FOOTER_SIZE;
    // [FIX] Changed from capsule_offset to bundle_offset
    uint64_t capsuleSize = footerStart - footer.bundle_offset;
    
    std::vector<uint8_t> capsuleBytes;
    // [FIX] Changed from capsule_offset to bundle_offset
    if (!read_exact(file, capsuleBytes, capsuleSize, footer.bundle_offset)) {
        std::cerr << "Error: Failed to read security capsule.\n";
        return 1;
    }

    // 5. Verify ABI & Signature
    // We cast the volatile pointer to const uint8_t* for the verifier
    std::vector<uint8_t> trustedKey(32);
    std::memcpy(trustedKey.data(), (const void*)ARK_STUB_ROOT_PUBKEY, 32);

    auto planHashOpt = Codec::verify_capsule_v1(capsuleBytes, trustedKey);
    
    if (!planHashOpt) {
        std::cerr << "🚨 SECURITY ALERT: Capsule signature verification failed!\n";
        return 1;
    }
    const auto& expectedHash = *planHashOpt;

    // 6. Read Payload
    // [FIX] Changed from capsule_offset to bundle_offset
    uint64_t payloadLen = footer.bundle_offset - footer.payload_offset;
    std::vector<uint8_t> payload;
    if (!read_exact(file, payload, payloadLen, footer.payload_offset)) {
        std::cerr << "Error: Failed to read payload.\n";
        return 1;
    }

    // 7. Verify Payload Binding
    // Re-hash the payload content to ensure it matches what the signed capsule expects
    auto calcHash = Crypto::Hasher::compute(Crypto::HashAlgo::BLAKE3, payload);
    if (std::memcmp(calcHash.data(), expectedHash.data(), 32) != 0) {
        std::cerr << "🚨 SECURITY ALERT: Payload hash mismatch!\n";
        return 1;
    }

    // 8. Execute
    // Replaces process image
    ark::capsule::platform::exec_payload(payload, argc, argv, envp);

    return 0; // Unreachable
}

} // namespace ark::runner

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------
#ifdef ARK_STUB_STANDALONE
int main(int argc, char** argv, char** envp) {
    return ark::runner::stub_entry(argc, argv, envp);
}
#endif