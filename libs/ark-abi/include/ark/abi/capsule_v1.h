#pragma once
#include <cstdint>

namespace ark::abi::capsule_v1 {

// =============================================================================
// CAPSULE V1 CONSTANTS (NORMATIVE)
// =============================================================================
static constexpr uint32_t ABI_REVISION = 1;

// Magic: "ARKC"
static constexpr uint8_t  kMagic_bytes[4] = { 0x41, 0x52, 0x4B, 0x43 }; // 'A''R''K''C'
// Little-endian u32 encoding of kMagic_bytes (for wire emission convenience only)
static constexpr uint32_t kMagic_u32_le    = 0x434B5241;

static constexpr uint16_t kVersion         = 1;

// =============================================================================
// HEADER LAYOUT (Fixed 16 Bytes)
// =============================================================================
// [0..3]   Magic (4)
// [4..5]   Version (2)
// [6..7]   Reserved (2) - Must be 0
// [8..11]  Flags (4)
// [12..15] TotalLength (4) - Includes Header + Payload TLVs + Signature TLV
static constexpr uint32_t OFF_MAGIC     = 0;
static constexpr uint32_t OFF_VERSION   = 4;
static constexpr uint32_t OFF_RESERVED  = 6;
static constexpr uint32_t OFF_FLAGS     = 8;
static constexpr uint32_t OFF_TOTAL_LEN = 12;

static constexpr uint32_t HEADER_SIZE   = 16;

// Flags (Bitmask)
static constexpr uint32_t FLAG_HAS_SYMTAB        = 1u << 0;
static constexpr uint32_t FLAG_HAS_DEBUG_SIDECAR = 1u << 1;
static constexpr uint32_t FLAG_DETERMINISTIC_IO  = 1u << 2;

// =============================================================================
// TLV LAYOUT (NORMATIVE)
// =============================================================================
// TLV header: Tag(u16 LE) + Len(u16 LE)
static constexpr uint32_t TLV_HDR_SIZE = 4;

// TLV TAGS (u16) - Order is strictly enforced by Builder and validated by Provider.
static constexpr uint16_t TAG_PLANHASH     = 0x0001; // Required
static constexpr uint16_t TAG_COMPILER_ID  = 0x0002; // Required
static constexpr uint16_t TAG_RUNTIME_REQS = 0x0003; // Required
static constexpr uint16_t TAG_SYMTAB_HASH  = 0x0004; // Optional
static constexpr uint16_t TAG_SIGNATURE    = 0xFFFF; // Required, MUST be last

// Required payload sizes
static constexpr uint16_t PLANHASH_LEN     = 32;
static constexpr uint16_t SYMTAB_HASH_LEN  = 32;
static constexpr uint16_t SIG_LEN          = 64;

// Signature TLV total size: hdr(4) + payload(64)
static constexpr uint32_t SIG_TLV_SIZE = TLV_HDR_SIZE + SIG_LEN;

// =============================================================================
// SIGNATURE CONTRACT (NORMATIVE)
// =============================================================================
// Signed region is ALWAYS [0 .. TotalLength - SIG_TLV_SIZE).
// Includes the Header (with finalized TotalLength) and all TLVs EXCEPT the Signature TLV.
// The Signature TLV MUST be appended as the final TLV and must exactly terminate the buffer.

// Returns false if total length cannot possibly be valid.
inline constexpr bool signed_length(uint32_t total_capsule_len, uint32_t& out_signed_len) noexcept {
    if (total_capsule_len < HEADER_SIZE + SIG_TLV_SIZE) return false;
    out_signed_len = total_capsule_len - SIG_TLV_SIZE;
    return true;
}

// Canonical “is the buffer length sane at all”
inline constexpr bool is_total_len_sane(uint32_t total_capsule_len) noexcept {
    uint32_t s = 0;
    return signed_length(total_capsule_len, s);
}

// =============================================================================
// ORDERING LAW (NORMATIVE)
// =============================================================================
// Payload TLVs must be in strictly increasing tag order.
// TAG_SIGNATURE must be last.
inline constexpr bool is_payload_tag_allowed(uint16_t tag) noexcept {
    return (tag == TAG_PLANHASH) ||
           (tag == TAG_COMPILER_ID) ||
           (tag == TAG_RUNTIME_REQS) ||
           (tag == TAG_SYMTAB_HASH);
}

inline constexpr bool is_tag_order_valid(uint16_t prev_tag, uint16_t next_tag) noexcept {
    // Strictly increasing among payload tags; signature handled separately.
    return next_tag > prev_tag;
}

} // namespace ark::abi::capsule_v1