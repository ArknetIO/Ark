#include "Integration/CapsuleBackend.h"

// New Protocol Includes
#include <ark/capsule/Capsule.h>
#include <ark/crypto.h>

// System
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <iterator>

namespace ark::compiler::integration {

using namespace ark::capsule;
namespace Crypto = ark::crypto;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
// The placeholder pattern inside 'ark-stub' that we must overwrite.
static const uint8_t STUB_KEY_PLACEHOLDER[32] = {
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);
    if (!f.read((char*)buf.data(), size)) return {};
    return buf;
}

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------
// CapsuleBackend.cpp
bool CapsuleBackend::CreateSelfExecutingCapsule(
    arklang::hud::Hud& hud,
    const std::string& sourceDir,
    const std::string& stubPath,
    const std::string& payloadPath,
    const std::string& outPath,
    const std::vector<uint8_t>& privKey,
    uint64_t* outTotalSize
) {
    (void)sourceDir;

    auto stubBytes = readFile(stubPath);
    if (stubBytes.empty()) {
        hud.error("Failed to read Stub: " + stubPath);
        return false;
    }

    auto payloadBytes = readFile(payloadPath);
    if (payloadBytes.empty()) {
        hud.error("Failed to read Payload: " + payloadPath);
        return false;
    }

    auto pubKey = Crypto::Signature::get_public_key(privKey, Crypto::SignAlgo::Ed25519);
    if (pubKey.size() != 32) {
        hud.error("Invalid key generation.");
        return false;
    }

    auto it = std::search(
        stubBytes.begin(), stubBytes.end(),
        std::begin(STUB_KEY_PLACEHOLDER), std::end(STUB_KEY_PLACEHOLDER)
    );
    if (it == stubBytes.end()) {
        hud.error("Broken Stub: Could not find Key Placeholder pattern.");
        return false;
    }
    std::memcpy(&(*it), pubKey.data(), 32);

    auto pHashVec = Crypto::Hasher::compute(Crypto::HashAlgo::BLAKE3, payloadBytes);
    if (pHashVec.size() != 32) {
        hud.error("Internal Error: PlanHash size mismatch.");
        return false;
    }

    std::array<uint8_t, 32> planHash;
    std::copy(pHashVec.begin(), pHashVec.end(), planHash.begin());

    auto capsuleBytes = Codec::create_capsule_v1(planHash, privKey);

    const uint64_t offsetStub = 0;
    const uint64_t offsetPayload = stubBytes.size();
    const uint64_t offsetBundle = offsetPayload + payloadBytes.size();
    const uint64_t totalSize = offsetBundle + capsuleBytes.size() + FOOTER_SIZE;

    Footer footer;
    std::memcpy(footer.magic.data(), FOOTER_MAGIC, 8);
    footer.version = 1;
    footer.payload_offset = offsetPayload;
    footer.bundle_offset = offsetBundle;
    footer.total_size = totalSize;

    auto footerBytes = Codec::serialize_footer(footer);

    if (!Codec::validate_layout(footer, totalSize)) {
        hud.error("Internal Error: Generated capsule layout is invalid.");
        return false;
    }

    std::ofstream out(outPath, std::ios::binary);
    if (!out) {
        hud.error("Cannot write output: " + outPath);
        return false;
    }

    out.write((char*)stubBytes.data(), stubBytes.size());
    out.write((char*)payloadBytes.data(), payloadBytes.size());
    out.write((char*)capsuleBytes.data(), capsuleBytes.size());
    out.write((char*)footerBytes.data(), footerBytes.size());
    out.close();

    if (outTotalSize) *outTotalSize = totalSize;
    return true;
}


} // namespace ark::compiler::integration