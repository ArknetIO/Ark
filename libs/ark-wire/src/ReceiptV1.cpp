#include <ark/wire/ReceiptV1.h>
#include <ark/wire/BufWriter.h> // Must include definition, not just fwd decl

namespace ark::wire {

// Constants for Validation
static constexpr size_t LEN_JOB_ID      = 16;
static constexpr size_t LEN_NONCE       = 32;
static constexpr size_t LEN_HASH        = 32;
static constexpr size_t LEN_SIGNATURE   = 64;

bool ReceiptV1View::is_valid() const {
    if (job_id.size() != LEN_JOB_ID) return false;
    if (nonce.size() != LEN_NONCE) return false;
    if (build_id.size() != LEN_HASH) return false;
    if (provider_id.size() != LEN_HASH) return false;
    
    if (stdout_hash.size() != LEN_HASH) return false;
    if (stderr_hash.size() != LEN_HASH) return false;
    if (artifact_hash.size() != LEN_HASH) return false;
    
    if (signature.size() != LEN_SIGNATURE) return false;
    
    return true;
}

bool write_receipt_v1(BufWriter& w, const ReceiptV1View& v) {
    // 0. Pre-flight Validation
    // Ensure we don't write a partial corrupted receipt.
    if (!v.is_valid()) return false;

    // 1. Header: Magic "ARKR" (Ark Receipt)
    const uint8_t magic[] = {'A', 'R', 'K', 'R'};
    if (!w.write_bytes(magic, 4)) return false;

    // 2. Versioning
    if (!w.write_u8(1)) return false; // Version
    if (!w.write_u8(0)) return false; // Reserved

    // 3. Outcome
    if (!w.write_u8(v.status)) return false;
    if (!w.write_u8(v.exit_code)) return false;

    // 4. Identity Blocks
    // Note: We use write_bytes explicitly to enforce exact size writing
    // defined by the view validation, avoiding generic span write overhead.
    if (!w.write_bytes(v.job_id.data(), LEN_JOB_ID)) return false;
    if (!w.write_bytes(v.nonce.data(), LEN_NONCE)) return false;
    if (!w.write_bytes(v.build_id.data(), LEN_HASH)) return false;
    if (!w.write_bytes(v.provider_id.data(), LEN_HASH)) return false;

    // 5. Timings (u64 Little Endian)
    // BufWriter now natively supports write_u64 with robust bounds checking.
    if (!w.write_u64(v.start_time_ns)) return false;
    if (!w.write_u64(v.duration_ns)) return false;
    if (!w.write_u64(v.slo_target_ns)) return false;
    if (!w.write_u64(v.slo_measured_ns)) return false;

    // 6. Commitments
    if (!w.write_bytes(v.stdout_hash.data(), LEN_HASH)) return false;
    if (!w.write_bytes(v.stderr_hash.data(), LEN_HASH)) return false;
    if (!w.write_bytes(v.artifact_hash.data(), LEN_HASH)) return false;

    // 7. Signature (Final Seal)
    if (!w.write_bytes(v.signature.data(), LEN_SIGNATURE)) return false;

    return true;
}

} // namespace ark::wire