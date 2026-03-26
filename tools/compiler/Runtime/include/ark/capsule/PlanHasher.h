#pragma once
#include "HashTypes.h"
#include <vector>

namespace ark::compiler::builder {

class PlanHasher {
public:
    // -------------------------------------------------------------------------
    // COMPUTE PLAN HASH (V1)
    // -------------------------------------------------------------------------
    // Calculates the deterministic build identity (Top Level).
    //
    // Algorithm:
    // 1. Tag inputs (0x01=ID, 0x02=Reqs, 0x03=RootHash).
    // 2. Stream into BLAKE3-256 hasher.
    //
    // Safety:
    // - Validates canonical paths in the root node.
    // - Does NOT enforce 64KB TLV limits (that is an ArtifactSealer policy).
    static BuilderStatus Compute(
        const std::vector<uint8_t>& compiler_id,
        const std::vector<uint8_t>& runtime_reqs,
        const DependencyNode& root_node,
        Hash32& out_hash
    );

    // -------------------------------------------------------------------------
    // COMPUTE NODE HASH (V1)
    // -------------------------------------------------------------------------
    // Calculates the recursive hash of a single dependency node.
    // Exposed so the BuildEngine can calculate hashes for leaf nodes during
    // graph construction.
    //
    // Algorithm:
    // 1. Tag 0x10.
    // 2. Stream PathLen + Path (Canonical).
    // 3. Stream ContentHash.
    // 4. Stream ChildCount.
    // 5. Sort ChildHashes (Lexicographically) -> Determinism.
    // 6. Stream Sorted ChildHashes.
    static Hash32 HashNode(const DependencyNode& node);
};

} // namespace ark::compiler::builder