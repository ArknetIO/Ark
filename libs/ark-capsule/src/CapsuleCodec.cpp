#include "ark/capsule/Capsule.h"
#include <ark/crypto.h>
#include <cstring>
#include <vector>
#include <algorithm>

namespace ark::capsule {

namespace abi = ark::abi::capsule_v1;

// -----------------------------------------------------------------------------
// LE Helpers
// -----------------------------------------------------------------------------
static void push_bytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    const auto* s = static_cast<const uint8_t*>(p);
    b.insert(b.end(), s, s + n);
}
static void push_u16_le(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
}
static void push_u32_le(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
    b.push_back(static_cast<uint8_t>(v >> 16));
    b.push_back(static_cast<uint8_t>(v >> 24));
}
static void push_u64_le(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
static bool read_bytes(std::span<const uint8_t>& s, void* out, size_t n) {
    if (s.size() < n) return false;
    std::memcpy(out, s.data(), n);
    s = s.subspan(n);
    return true;
}
static bool read_u16_le(std::span<const uint8_t>& s, uint16_t& out) {
    if (s.size() < 2) return false;
    out = static_cast<uint16_t>(s[0]) | (static_cast<uint16_t>(s[1]) << 8);
    s = s.subspan(2);
    return true;
}
static bool read_u32_le(std::span<const uint8_t>& s, uint32_t& out) {
    if (s.size() < 4) return false;
    out = static_cast<uint32_t>(s[0]) | (static_cast<uint32_t>(s[1]) << 8) |
          (static_cast<uint32_t>(s[2]) << 16) | (static_cast<uint32_t>(s[3]) << 24);
    s = s.subspan(4);
    return true;
}
static bool read_u64_le(std::span<const uint8_t>& s, uint64_t& out) {
    if (s.size() < 8) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= (static_cast<uint64_t>(s[i]) << (8 * i));
    s = s.subspan(8);
    return true;
}

// -----------------------------------------------------------------------------
// Footer Serialization
// -----------------------------------------------------------------------------
std::vector<uint8_t> Codec::serialize_footer(const Footer& f) {
    std::vector<uint8_t> buf;
    buf.reserve(FOOTER_SIZE);
    push_bytes(buf, f.magic.data(), 8);
    push_u32_le(buf, f.version);
    // [FIX] Use bundle_offset to match header
    push_u64_le(buf, f.bundle_offset); 
    push_u64_le(buf, f.payload_offset);
    push_u64_le(buf, f.total_size);
    return buf;
}

std::optional<Footer> Codec::deserialize_footer(std::span<const uint8_t> bytes) {
    if (bytes.size() != FOOTER_SIZE) return std::nullopt;
    Footer f;
    if (!read_bytes(bytes, f.magic.data(), 8)) return std::nullopt;
    if (!read_u32_le(bytes, f.version)) return std::nullopt;
    // [FIX] Use bundle_offset to match header
    if (!read_u64_le(bytes, f.bundle_offset)) return std::nullopt;
    if (!read_u64_le(bytes, f.payload_offset)) return std::nullopt;
    if (!read_u64_le(bytes, f.total_size)) return std::nullopt;
    return f;
}

// -----------------------------------------------------------------------------
// ABI Creation (Builder)
// -----------------------------------------------------------------------------
std::vector<uint8_t> Codec::create_capsule_v1(
    const std::array<uint8_t, 32>& planHash,
    const std::vector<uint8_t>& signingKey
) {
    std::vector<uint8_t> blob;
    blob.reserve(256); // Preallocate reasonable size

    // 1. Header
    // Magic (4 bytes)
    push_bytes(blob, abi::kMagic_bytes, 4);
    // Version (2 bytes)
    push_u16_le(blob, abi::kVersion);
    // Reserved (2 bytes)
    push_u16_le(blob, 0);
    // Flags (4 bytes)
    push_u32_le(blob, 0); // No flags for now
    // Total Length Placeholder (4 bytes) - Offset 12
    size_t lenOffset = blob.size();
    push_u32_le(blob, 0); 

    // 2. TLV: PlanHash
    push_u16_le(blob, abi::TAG_PLANHASH);
    push_u16_le(blob, abi::PLANHASH_LEN);
    push_bytes(blob, planHash.data(), 32);

    // 3. Signature (Must be Last)
    // We first compute the "Signed Region" (Header + TLVs so far)
    // Total Length = current size + Sig TLV size
    uint32_t totalLen = static_cast<uint32_t>(blob.size() + abi::SIG_TLV_SIZE);
    
    // Backpatch Total Length
    blob[lenOffset] = static_cast<uint8_t>(totalLen);
    blob[lenOffset+1] = static_cast<uint8_t>(totalLen >> 8);
    blob[lenOffset+2] = static_cast<uint8_t>(totalLen >> 16);
    blob[lenOffset+3] = static_cast<uint8_t>(totalLen >> 24);

    // Sign the region [0..end]
    auto sig = ark::crypto::Signature::sign(blob, signingKey, ark::crypto::SignAlgo::Ed25519);

    // Append Signature TLV
    push_u16_le(blob, abi::TAG_SIGNATURE);
    push_u16_le(blob, abi::SIG_LEN);
    push_bytes(blob, sig.data(), 64);

    return blob;
}

// -----------------------------------------------------------------------------
// ABI Verification Logic (Verifier)
// -----------------------------------------------------------------------------
std::optional<std::array<uint8_t, 32>> Codec::verify_capsule_v1(
    std::span<const uint8_t> blob,
    std::span<const uint8_t> trustedPubKey
) {
    // 1. Header Check
    if (blob.size() < abi::HEADER_SIZE) return std::nullopt;
    
    // Check Magic
    if (std::memcmp(blob.data(), abi::kMagic_bytes, 4) != 0) return std::nullopt;

    // Check Version (Offset 4)
    uint16_t ver = static_cast<uint16_t>(blob[4]) | (static_cast<uint16_t>(blob[5]) << 8);
    if (ver != abi::kVersion) return std::nullopt;

    // 2. Identify Signed Region
    uint32_t totalLen = 
        static_cast<uint32_t>(blob[12]) | (static_cast<uint32_t>(blob[13]) << 8) |
        (static_cast<uint32_t>(blob[14]) << 16) | (static_cast<uint32_t>(blob[15]) << 24);

    if (totalLen != blob.size()) return std::nullopt;

    uint32_t signedLen = 0;
    if (!abi::signed_length(totalLen, signedLen)) return std::nullopt;

    // 3. Parse TLVs to find PlanHash and Signature
    std::span<const uint8_t> cursor = blob.subspan(abi::HEADER_SIZE);
    std::optional<std::array<uint8_t, 32>> planHash;
    std::span<const uint8_t> signatureBytes;

    uint16_t prevTag = 0;

    while (!cursor.empty()) {
        uint16_t tag, len;
        if (!read_u16_le(cursor, tag)) return std::nullopt;
        if (!read_u16_le(cursor, len)) return std::nullopt;

        if (cursor.size() < len) return std::nullopt;
        std::span<const uint8_t> value = cursor.first(len);
        cursor = cursor.subspan(len);

        // Check Ordering
        if (tag == abi::TAG_SIGNATURE) {
            // Must be last
            if (!cursor.empty()) return std::nullopt; 
            if (len != abi::SIG_LEN) return std::nullopt;
            signatureBytes = value;
            break;
        }

        if (!abi::is_tag_order_valid(prevTag, tag)) return std::nullopt;
        prevTag = tag;

        // Extract Data
        if (tag == abi::TAG_PLANHASH) {
            if (len != abi::PLANHASH_LEN) return std::nullopt;
            std::array<uint8_t, 32> ph;
            std::memcpy(ph.data(), value.data(), 32);
            planHash = ph;
        }
    }

    if (!planHash.has_value()) return std::nullopt;
    if (signatureBytes.empty()) return std::nullopt;

    // 4. Verify Signature
    bool ok = ark::crypto::Signature::verify(
        {signatureBytes.data(), signatureBytes.size()},
        {blob.data(), signedLen},
        {trustedPubKey.data(), trustedPubKey.size()},
        ark::crypto::SignAlgo::Ed25519
    );

    if (!ok) return std::nullopt;

    return planHash;
}

// -----------------------------------------------------------------------------
// Layout Validation
// -----------------------------------------------------------------------------
bool Codec::validate_layout(const Footer& f, uint64_t fileSize) {
    if (f.total_size != fileSize) return false;
    if (std::memcmp(f.magic.data(), FOOTER_MAGIC, 8) != 0) return false;
    if (f.version != 1) return false; 

    const uint64_t footerStart = fileSize - FOOTER_SIZE;

    // [FIX] Use bundle_offset
    if (f.bundle_offset > footerStart) return false;

    // Payload
    if (f.payload_offset >= f.bundle_offset) return false;
    if (f.payload_offset == 0) return false;

    // Payload Length
    const uint64_t payloadLen = f.bundle_offset - f.payload_offset;
    if (payloadLen < 1) return false;
    if (payloadLen > 512ULL * 1024 * 1024) return false;

    return true;
}

} // namespace