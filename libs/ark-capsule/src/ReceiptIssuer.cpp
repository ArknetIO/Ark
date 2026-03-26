#include <ark/capsule/ReceiptIssuer.h>
#include <ark/wire/ReceiptV1.h>
#include <ark/wire/BufWriter.h>
#include <ark/abi/receipt_v1.h>
#include <ark/crypto.h>

#include <cstring>
#include <vector>

namespace ark::compiler::builder {

using namespace ark::abi::receipt_v1;

std::vector<uint8_t> ReceiptIssuer::Issue(
    const LoadedCapsule& capsule,
    const ExecutionResult& res,
    const std::vector<uint8_t>& priv_key,
    const std::vector<uint8_t>& job_id,
    const std::vector<uint8_t>& nonce
) {
    // 1. Hard Invariants
    if (job_id.size() != 16) return {};
    if (nonce.size() != 32) return {};
    if (!priv_key.empty() && priv_key.size() != 64) return {};

    // 2. Prepare View
    ark::wire::ReceiptV1View v{};

    // Identity
    v.job_id   = job_id;
    v.nonce    = nonce;
    v.build_id = capsule.plan_hash;

    // Provider ID (Placeholder or derived)
    uint8_t provider_id_bytes[32] = {0};
    v.provider_id = std::span<const uint8_t>(provider_id_bytes, 32);

    // Outcome
    v.status    = res.status;
    v.exit_code = res.exit_code;

    // Metrics
    v.start_time_ns   = res.start_time_ns;
    v.duration_ns     = res.duration_ns;
    v.slo_target_ns   = 0;
    v.slo_measured_ns = 0;

    // Commitments
    v.stdout_hash   = res.stdout_hash;
    v.stderr_hash   = res.stderr_hash;
    v.artifact_hash = res.artifact_hash;

    // Signature (Zeroed for serialization phase)
    uint8_t zero_sig[64] = {0};
    v.signature = std::span<const uint8_t>(zero_sig, 64);

    // 3. Serialize
    std::vector<uint8_t> buffer(kSizeBytes);
    ark::wire::BufWriter w(buffer.data(), buffer.size());

    if (!ark::wire::write_receipt_v1(w, v)) {
        return {}; // Buffer too small or view invalid
    }

    // Verify exact size match
    if (w.size() != kSizeBytes) return {};

    // 4. Sign
    if (!priv_key.empty()) {
        // Wrap raw bytes into the robust key type
        ark::crypto::SigningSecretKey secret_key(priv_key);

        // Sign the prefix: [Header ... Commitments] (excludes the 64-byte signature field at end)
        std::span<const uint8_t> message_to_sign(buffer.data(), OFF_SIGNATURE);

        // Compute Signature
        std::vector<uint8_t> sig = ark::crypto::Signature::sign(
            message_to_sign, 
            secret_key
        );
        
        // Write Signature into the buffer at the end
        if (sig.size() == 64) {
            std::memcpy(buffer.data() + OFF_SIGNATURE, sig.data(), 64);
        } else {
            return {}; // Should never happen with Ed25519
        }
    }

    return buffer;
}

} // namespace ark::compiler::builder