#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <optional>
#include <span>

// [CRITICAL] Integrate the normative ABI
#include "ark/abi/capsule_v1.h"

namespace ark::capsule {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
// Magic: "ARK_EXEC" (8 bytes) - Distinguishes the self-exec wrapper from the inner capsule
static constexpr uint8_t FOOTER_MAGIC[8] = {
    0x41, 0x52, 0x4B, 0x5F, 0x45, 0x58, 0x45, 0x43
};

static constexpr size_t FOOTER_SIZE = 36;

// -----------------------------------------------------------------------------
// The Discovery Footer
// -----------------------------------------------------------------------------
struct Footer {
    std::array<uint8_t, 8> magic; // "ARK_EXEC"
    uint32_t version;             // Wrapper Version (1)
    
    // [FIX] Renamed to bundle_offset to match Codec/Backend usage
    uint64_t bundle_offset;       // Where the abi::capsule_v1 blob starts
    
    uint64_t payload_offset;      // Where the raw payload starts
    uint64_t total_size;          // File integrity check
};

// -----------------------------------------------------------------------------
// The Codec (Logic Layer)
// -----------------------------------------------------------------------------
class Codec {
public:
    // Footer Serialization (Explicit LE)
    static std::vector<uint8_t> serialize_footer(const Footer& f);
    static std::optional<Footer> deserialize_footer(std::span<const uint8_t> bytes);

    // -------------------------------------------------------------------------
    // ABI Helpers
    // -------------------------------------------------------------------------
    
    // [NEW] Builder: Create a normative V1 Capsule Blob (Header + TLVs + Sig)
    static std::vector<uint8_t> create_capsule_v1(
        const std::array<uint8_t, 32>& planHash,
        const std::vector<uint8_t>& signingKey
    );

    // Verifier: Validates V1 Capsule blob against ABI rules
    // Returns the extracted PLANHASH if valid.
    static std::optional<std::array<uint8_t, 32>> verify_capsule_v1(
        std::span<const uint8_t> capsuleBytes,
        std::span<const uint8_t> trustedPubKey
    );

    // -------------------------------------------------------------------------
    // Layout Validation
    // -------------------------------------------------------------------------
    static bool validate_layout(const Footer& f, uint64_t fileSize);
};

} // namespace ark::capsule