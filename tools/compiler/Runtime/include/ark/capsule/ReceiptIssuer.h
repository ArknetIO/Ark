#pragma once
#include <vector>
#include <cstdint>
#include <array>
#include <ark/capsule/HashTypes.h>

namespace ark::compiler::builder {

// -----------------------------------------------------------------------------
// Input Structures
// -----------------------------------------------------------------------------

struct LoadedCapsule {
    Hash32 plan_hash; // [FIX] No qualifier needed
    // ...
};

struct ExecutionResult {
    uint8_t status;
    uint8_t exit_code;
    
    uint64_t start_time_ns;
    uint64_t duration_ns;
    
    Hash32 stdout_hash;   // [FIX] No qualifier needed
    Hash32 stderr_hash;   // [FIX] No qualifier needed
    Hash32 artifact_hash; // [FIX] No qualifier needed
};

// -----------------------------------------------------------------------------
// RECEIPT ISSUER
// -----------------------------------------------------------------------------
class ReceiptIssuer {
public:
    static std::vector<uint8_t> Issue(
        const LoadedCapsule& capsule,
        const ExecutionResult& res,
        const std::vector<uint8_t>& priv_key,
        const std::vector<uint8_t>& job_id,
        const std::vector<uint8_t>& nonce
    );
};

} // namespace ark::compiler::builder