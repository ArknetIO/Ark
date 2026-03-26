#include <ark/capsule/PlanHasher.h>
#include <ark/crypto.h>
#include <algorithm> // std::sort
#include <vector>
#include <cstring>   // std::memcpy

namespace ark::compiler::builder {

// -----------------------------------------------------------------------------
// CANONICAL SERIALIZATION HELPERS (Normative)
// -----------------------------------------------------------------------------
static void write_u8(ark::crypto::Hasher& h, uint8_t v) {
    h.update({&v, 1});
}

static void write_u32(ark::crypto::Hasher& h, uint32_t v) {
    uint8_t buf[4];
    buf[0] = v & 0xFF;
    buf[1] = (v >> 8) & 0xFF;
    buf[2] = (v >> 16) & 0xFF;
    buf[3] = (v >> 24) & 0xFF;
    h.update({buf, 4});
}

static void write_bytes(ark::crypto::Hasher& h, const std::vector<uint8_t>& b) {
    h.update({b.data(), b.size()});
}

// -----------------------------------------------------------------------------
// NODE HASH LOGIC
// -----------------------------------------------------------------------------
Hash32 PlanHasher::HashNode(const DependencyNode& node) {
    ark::crypto::Hasher h(ark::crypto::HashAlgo::BLAKE2B);

    // 1. Tag (0x10)
    write_u8(h, 0x10);

    // 2. Path (Len + Bytes)
    write_u32(h, static_cast<uint32_t>(node.rel_path.size()));
    write_bytes(h, node.rel_path);

    // 3. Content Hash (Raw 32 bytes)
    h.update({node.content_hash.data(), 32});

    // 4. Children (Sort -> Count -> Hashes)
    std::vector<Hash32> sorted_children = node.child_hashes;
    std::sort(sorted_children.begin(), sorted_children.end()); 

    write_u32(h, static_cast<uint32_t>(sorted_children.size()));
    
    for (const auto& child_h : sorted_children) {
        h.update({child_h.data(), 32});
    }

    std::vector<uint8_t> res = h.finalize();
    Hash32 out;
    if (res.size() >= 32) {
        std::copy_n(res.begin(), 32, out.begin());
    } else {
        out.fill(0);
    }
    return out;
}

// -----------------------------------------------------------------------------
// PLAN HASH LOGIC
// -----------------------------------------------------------------------------
BuilderStatus PlanHasher::Compute(
    const std::vector<uint8_t>& compiler_id,
    const std::vector<uint8_t>& reqs,
    const DependencyNode& root_node,
    Hash32& out_hash
) {
    Hash32 root_hash = HashNode(root_node);

    ark::crypto::Hasher h(ark::crypto::HashAlgo::BLAKE2B);

    // Tag 0x01: CompilerID
    write_u8(h, 0x01);
    write_u32(h, static_cast<uint32_t>(compiler_id.size()));
    write_bytes(h, compiler_id);

    // Tag 0x02: RuntimeReqs
    write_u8(h, 0x02);
    write_u32(h, static_cast<uint32_t>(reqs.size()));
    write_bytes(h, reqs);

    // Tag 0x03: Root Node Hash
    write_u8(h, 0x03);
    write_u32(h, 32); 
    h.update({root_hash.data(), 32});

    std::vector<uint8_t> res = h.finalize();
    if (res.size() >= 32) {
        std::copy_n(res.begin(), 32, out_hash.begin());
    } else {
        // [FIX] Use defined error code
        return BuilderStatus::CryptoError; 
    }

    return BuilderStatus::Ok;
}

} // namespace ark::compiler::builder