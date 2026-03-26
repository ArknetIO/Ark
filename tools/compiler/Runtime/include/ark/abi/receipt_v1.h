#pragma once
#include <cstddef>
#include <cstdint>

namespace ark::abi::receipt_v1 {

    // Fixed size of a V1 Receipt
    // Header(6) + Status(2) + IDs(112) + Timings(32) + Hashes(96) + Sig(64)
    constexpr size_t kSizeBytes = 312;
    
    // Offset where the signature begins (Total - 64)
    // Used for signing the prefix [0..OFF_SIGNATURE)
    constexpr size_t OFF_SIGNATURE = 248;

} // namespace ark::abi::receipt_v1