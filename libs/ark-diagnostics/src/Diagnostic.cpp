#include <ark/diagnostics/Diagnostic.h>
#include <sstream>
#include <iomanip>

namespace ark::diagnostics {

std::string json_escape(std::string_view s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

const char* to_string(VerifyStatus s) noexcept {
    switch(s) {
        case VerifyStatus::Ok: return "Ok";
        case VerifyStatus::InvalidMagic: return "InvalidMagic";
        case VerifyStatus::InvalidVersion: return "InvalidVersion";
        case VerifyStatus::InvalidReserved: return "InvalidReserved";
        case VerifyStatus::InvalidFlags: return "InvalidFlags";
        case VerifyStatus::InvalidTotalLength: return "InvalidTotalLength";
        case VerifyStatus::SignatureMismatch: return "SignatureMismatch";
        case VerifyStatus::CryptoError: return "CryptoError";
        case VerifyStatus::ReqsTooLarge: return "ReqsTooLarge";
        case VerifyStatus::CorruptTLV: return "CorruptTLV";
        case VerifyStatus::MissingRequiredTags: return "MissingRequiredTags";
        case VerifyStatus::TagOrderViolation: return "TagOrderViolation";
        case VerifyStatus::UnknownTag: return "UnknownTag";
        case VerifyStatus::SigTagInPayload: return "SigTagInPayload";
        case VerifyStatus::TrailingGarbage: return "TrailingGarbage";
        case VerifyStatus::InternalError: return "InternalError";
    }
    return "UnknownStatus";
}

const char* to_string(VerifyStage s) noexcept {
    switch(s) {
        case VerifyStage::None: return "None";
        case VerifyStage::Header: return "Header";
        case VerifyStage::Length: return "Length";
        case VerifyStage::Signature: return "Signature";
        case VerifyStage::PayloadTLV: return "PayloadTLV";
    }
    return "UnknownStage";
}

const char* to_string(DiagRule r) noexcept {
    switch(r) {
        case DiagRule::None: return "None";
        case DiagRule::MagicMismatch: return "MagicMismatch";
        case DiagRule::VersionMismatch: return "VersionMismatch";
        case DiagRule::ReservedNonZero: return "ReservedNonZero";
        case DiagRule::FlagsUnknownBits: return "FlagsUnknownBits";
        case DiagRule::TotalLenMismatch: return "TotalLenMismatch";
        case DiagRule::SignedLenInvalid: return "SignedLenInvalid";
        case DiagRule::SigTagMissing: return "SigTagMissing";
        case DiagRule::SigLenMismatch: return "SigLenMismatch";
        case DiagRule::SigVerifyFailed: return "SigVerifyFailed";
        case DiagRule::PayloadTagOrder: return "PayloadTagOrder";
        case DiagRule::UnknownTag: return "UnknownTag";
        case DiagRule::MissingRequired: return "MissingRequired";
        case DiagRule::ReqsTooLarge: return "ReqsTooLarge";
        case DiagRule::SigTagInPayload: return "SigTagInPayload";
        case DiagRule::CorruptStructure: return "CorruptStructure";
        case DiagRule::TrailingGarbage: return "TrailingGarbage";
        case DiagRule::SigTLVNotExhausted: return "SigTLVNotExhausted";
    }
    return "UnknownRule";
}

const char* rule_help(DiagRule r) noexcept {
    switch(r) {
        case DiagRule::None: return "No error.";
        case DiagRule::MagicMismatch: return "Capsule magic bytes do not match 'ARKC'.";
        case DiagRule::VersionMismatch: return "Capsule version is not supported.";
        case DiagRule::ReservedNonZero: return "Reserved fields must be zero.";
        case DiagRule::FlagsUnknownBits: return "Unknown flags detected in header.";
        case DiagRule::TotalLenMismatch: return "Total length field does not match buffer size.";
        case DiagRule::SignedLenInvalid: return "Calculated signed length is invalid.";
        case DiagRule::SigTagMissing: return "Signature TLV missing or misplaced.";
        case DiagRule::SigLenMismatch: return "Signature TLV length incorrect (must be 64).";
        case DiagRule::SigVerifyFailed: return "Ed25519 signature verification failed.";
        case DiagRule::PayloadTagOrder: return "TLVs must be in strictly increasing tag order.";
        case DiagRule::UnknownTag: return "Unknown TLV tag detected in payload.";
        case DiagRule::MissingRequired: return "Required TLVs missing.";
        case DiagRule::ReqsTooLarge: return "Runtime requirements blob exceeds limit.";
        case DiagRule::SigTagInPayload: return "Signature tag found inside payload region.";
        case DiagRule::CorruptStructure: return "TLV structure is malformed or truncated.";
        case DiagRule::TrailingGarbage: return "Data exists after the Signature TLV.";
        case DiagRule::SigTLVNotExhausted: return "Signature TLV contains extra bytes.";
    }
    return "Unknown rule violation.";
}

const char* rule_hint(DiagRule r) noexcept {
    switch(r) {
        case DiagRule::None: return "";
        case DiagRule::MagicMismatch: return "Ensure file is a valid ARK capsule (arkc build).";
        case DiagRule::VersionMismatch: return "Update your runtime provider or recompile.";
        case DiagRule::ReservedNonZero: return "Builder error: zero-init failure.";
        case DiagRule::FlagsUnknownBits: return "Check compiler flags or provider version.";
        case DiagRule::TotalLenMismatch: return "File truncated or corrupted during transfer.";
        case DiagRule::SignedLenInvalid: return "File corruption or builder logic error.";
        case DiagRule::SigTagMissing: return "Capsule not signed or layout invalid.";
        case DiagRule::SigLenMismatch: return "Signature format invalid.";
        case DiagRule::SigVerifyFailed: return "Check compiler public key or file integrity.";
        case DiagRule::PayloadTagOrder: return "Builder error: sort tags before sealing.";
        case DiagRule::UnknownTag: return "Provider is running strict V1 validation.";
        case DiagRule::MissingRequired: return "Rebuild the artifact.";
        case DiagRule::ReqsTooLarge: return "Reduce complexity of runtime requirements.";
        case DiagRule::SigTagInPayload: return "Builder error: signature injected early.";
        case DiagRule::CorruptStructure: return "Binary truncation detected.";
        case DiagRule::TrailingGarbage: return "File transfer corrupted?";
        case DiagRule::SigTLVNotExhausted: return "Signature blob malformed.";
    }
    return "Contact support.";
}

} // namespace ark::diagnostics