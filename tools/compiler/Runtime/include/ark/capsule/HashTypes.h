#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <algorithm> // For std::lexicographical_compare

namespace ark::compiler::builder {

// -----------------------------------------------------------------------------
// RAW HASH (32 Bytes)
// -----------------------------------------------------------------------------
// Used for ContentHash, NodeHash, and PlanHash.
// BLAKE3-256 output.
using Hash32 = std::array<uint8_t, 32>;

// Strict lexicographical comparison for deterministic sorting in PlanHash.
inline bool operator<(const Hash32& a, const Hash32& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

// -----------------------------------------------------------------------------
// DEPENDENCY GRAPH NODE
// -----------------------------------------------------------------------------
struct DependencyNode {
    // PRECONDITION: Must be canonicalized bytes (UTF-8, forward slashes).
    // NOT std::string to prevent implicit conversions/encoding assumptions.
    std::vector<uint8_t> rel_path;

    // The BLAKE3 hash of the file content.
    Hash32 content_hash;

    // List of child node hashes.
    // INVARIANT: Builder may populate this unsorted.
    // The PlanHasher is responsible for sorting a local copy before hashing.
    std::vector<Hash32> child_hashes;
};

// -----------------------------------------------------------------------------
// ERROR MODEL
// -----------------------------------------------------------------------------
enum class BuilderStatus {
    Ok = 0,
    InvalidPath,      // Path not canonical or contains forbidden sequences (e.g. "..")
    InputTooLarge,    // TLV length exceeds 65535 bytes
    CryptoError,      // Signing key invalid or signature computation failed
    OutputError       // Write failed or file size overflow (u32 limit)
};

} // namespace ark::compiler::builder