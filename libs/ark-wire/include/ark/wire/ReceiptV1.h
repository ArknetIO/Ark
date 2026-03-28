#pragma once
#include <span>
#include <cstdint>
#include <cstddef>

namespace ark::wire {

class BufWriter; 

/// \brief A zero-copy view of a V1 Receipt.
/// Used for both serialization (writing) and deserialization (reading).
struct ReceiptV1View {
    // -------------------------------------------------------------------------
    // 1. Identity
    // -------------------------------------------------------------------------
    std::span<const uint8_t> job_id;      // 16 bytes (UUID)
    std::span<const uint8_t> nonce;       // 32 bytes (Random)
    std::span<const uint8_t> build_id;    // 32 bytes (PlanHash)
    std::span<const uint8_t> provider_id; // 32 bytes (Public Key)

    // -------------------------------------------------------------------------
    // 2. Outcome
    // -------------------------------------------------------------------------
    uint8_t status;    // 0=Pending, 1=Success, 2=Failed
    uint8_t exit_code; // Process exit code

    // -------------------------------------------------------------------------
    // 3. Metrics (Nanoseconds)
    // -------------------------------------------------------------------------
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint64_t slo_target_ns;
    uint64_t slo_measured_ns;

    // -------------------------------------------------------------------------
    // 4. Commitments (Merkle Roots / Hashes)
    // -------------------------------------------------------------------------
    std::span<const uint8_t> stdout_hash;   // 32 bytes (BLAKE3)
    std::span<const uint8_t> stderr_hash;   // 32 bytes (BLAKE3)
    std::span<const uint8_t> artifact_hash; // 32 bytes (BLAKE3)

    // -------------------------------------------------------------------------
    // 5. Authentication
    // -------------------------------------------------------------------------
    std::span<const uint8_t> signature;     // 64 bytes (Ed25519)

    /// \brief Validate field sizes. Returns true if all spans have correct lengths.
    [[nodiscard]] bool is_valid() const;
};

/// \brief Serializes a Receipt V1 view into the writer.
/// \return true on success, false if buffer is too small or view is invalid.
[[nodiscard]] bool write_receipt_v1(BufWriter& w, const ReceiptV1View& v);

} // namespace ark::wire