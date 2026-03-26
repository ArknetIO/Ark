#pragma once
#include <cstdint>
#include <string>

namespace ark::diagnostics {

static constexpr uint32_t DIAG_REVISION = 7;

// -----------------------------------------------------------------------------
// STATUS CODES
// -----------------------------------------------------------------------------
enum class VerifyStatus {
    Ok = 0,
    InvalidMagic, InvalidVersion, InvalidReserved, InvalidFlags,
    InvalidTotalLength, SignatureMismatch, CryptoError, ReqsTooLarge,
    CorruptTLV, MissingRequiredTags, TagOrderViolation, UnknownTag,
    SigTagInPayload, TrailingGarbage, InternalError
};

// -----------------------------------------------------------------------------
// DIAGNOSTIC RULES (Stable IDs)
// -----------------------------------------------------------------------------
enum class DiagRule : uint16_t {
    None = 0,
    MagicMismatch = 1,
    VersionMismatch = 2,
    ReservedNonZero = 3,
    FlagsUnknownBits = 4,
    TotalLenMismatch = 5,
    SignedLenInvalid = 6,
    SigTagMissing = 7,
    SigLenMismatch = 8,
    SigVerifyFailed = 9,
    PayloadTagOrder = 10,
    UnknownTag = 11,
    MissingRequired = 12,
    ReqsTooLarge = 13,
    SigTagInPayload = 14,
    CorruptStructure = 15,
    TrailingGarbage = 16,
    SigTLVNotExhausted = 17
};

// -----------------------------------------------------------------------------
// CONTEXT
// -----------------------------------------------------------------------------
enum class VerifyStage : uint8_t {
    None = 0, Header = 1, Length = 2, Signature = 3, PayloadTLV = 4
};

enum class DiagKind : uint8_t {
    None = 0, U16 = 1, U32 = 2
};

struct VerifyDiag {
    uint32_t revision = DIAG_REVISION;
    VerifyStage stage = VerifyStage::None;
    DiagRule rule = DiagRule::None;
    DiagKind kind = DiagKind::None;
    
    uint32_t offset = 0;
    
    uint32_t expected_u32 = 0;
    uint32_t actual_u32 = 0;
    uint16_t expected_u16 = 0;
    uint16_t actual_u16 = 0;
    
    uint16_t tlv_tag = 0;
    uint16_t tlv_len = 0;
    uint32_t missing_mask = 0; 
    
    uint32_t total_len = 0;
    uint32_t signed_len = 0;
    uint16_t last_tag = 0;
};

// -----------------------------------------------------------------------------
// FORMATTERS & HELPERS
// -----------------------------------------------------------------------------
const char* to_string(VerifyStatus s) noexcept;
const char* to_string(VerifyStage s) noexcept;
const char* to_string(DiagRule r) noexcept;

const char* rule_help(DiagRule r) noexcept; // Description
const char* rule_hint(DiagRule r) noexcept; // Action

// Utility for safe JSON generation
std::string json_escape(std::string_view s);

} // namespace ark::diagnostics