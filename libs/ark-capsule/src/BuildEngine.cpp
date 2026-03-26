#include <ark/capsule/BuildEngine.h>
#include <ark/capsule/Scanner.h>
#include <ark/capsule/PlanHasher.h>
#include <ark/capsule/ArtifactSealer.h>
#include <ark/crypto.h> 

#include <iostream>
#include <fstream>
#include <algorithm> // std::copy_n

namespace ark::compiler::builder {

// -----------------------------------------------------------------------------
// Helper: Compute NodeHash for a single Leaf (File)
// -----------------------------------------------------------------------------
// This replicates the PlanHasher logic for a single node to establish its identity.
static Hash32 ComputeLeafHash(const SourceFile& sf) {
    // 1. Content Hash (BLAKE2b)
    Hash32 content_hash;
    std::vector<uint8_t> digest = ark::crypto::Hasher::compute(
        ark::crypto::HashAlgo::BLAKE2B, 
        {sf.content.data(), sf.content.size()}
    );
    
    // Safety check for digest size (BLAKE2b default is 32)
    if (digest.size() >= 32) {
        std::copy_n(digest.begin(), 32, content_hash.begin());
    } else {
        content_hash.fill(0); 
    }

    // 2. Construct Leaf Node
    DependencyNode leaf;
    leaf.rel_path = sf.rel_path_canonical;
    leaf.content_hash = content_hash;
    // Leaf has no children
    
    // 3. Hash the Node Structure
    // Relies on PlanHasher exposing a static HashNode method.
    return PlanHasher::HashNode(leaf);
}

// -----------------------------------------------------------------------------
// Build Engine Execution
// -----------------------------------------------------------------------------
int BuildEngine::Run(const BuildContext& ctx) {
    // 1. Scan Source Directory
    // -------------------------------------------------------------------------
    std::cout << "[BuildEngine] Scanning " << ctx.source_root << "...\n";
    auto files = Scanner::Scan(ctx.source_root);
    
    if (files.empty()) {
        std::cerr << "[BuildEngine] Error: No source files found in " << ctx.source_root << "\n";
        return 1;
    }

    // 2. Build Dependency Graph (V1: Flat Graph)
    // -------------------------------------------------------------------------
    // Root -> [File1, File2, File3...]
    DependencyNode root;
    root.rel_path = std::vector<uint8_t>{'.', '/'}; // Root path "."
    root.content_hash.fill(0); // Root content hash is zero for directories
    
    // Calculate and attach child hashes
    for (const auto& f : files) {
        Hash32 child_h = ComputeLeafHash(f);
        root.child_hashes.push_back(child_h);
    }
    
    // Note: PlanHasher::HashNode will sort these internally for determinism.

    // 3. Compute PlanHash (The Build Identity)
    // -------------------------------------------------------------------------
    std::vector<uint8_t> cid = {'a','r','k','c'};
    std::vector<uint8_t> reqs = {0x00}; // Default: No specific runtime requirements
    Hash32 plan_hash;
    
    auto status = PlanHasher::Compute(cid, reqs, root, plan_hash);
    if (status != BuilderStatus::Ok) {
        std::cerr << "[BuildEngine] Plan Hashing Failed. Code: " << (int)status << "\n";
        return 1;
    }

    // 4. Seal Artifact (The Capsule Header)
    // -------------------------------------------------------------------------
    // In the Platinum Architecture, the "Capsule" is the signed metadata header.
    // The actual compiled object code (Payload) is appended AFTER this header.
    // For this build engine pass, we generate the valid header.
    
    Hash32 symtab = {0};

    // Use a zero-key (Unsigned/Dev Build) if context doesn't provide one.
    // In a production Driver, the key comes from the KeyStore.
    std::vector<uint8_t> key(64, 0); 
    
    SealedArtifact artifact;
    auto seal_status = ArtifactSealer::Seal(plan_hash, cid, reqs, symtab, key, artifact);
    
    if (seal_status != BuilderStatus::Ok) {
        std::cerr << "[BuildEngine] Sealing Failed. Code: " << (int)seal_status << "\n";
        return 1;
    }

    // 5. Write Output
    // -------------------------------------------------------------------------
    std::ofstream out(ctx.output_path, std::ios::binary);
    if (!out) {
        std::cerr << "[BuildEngine] Error creating output file: " << ctx.output_path << "\n";
        return 1;
    }

    // Write the Signed Capsule Header
    out.write(reinterpret_cast<const char*>(artifact.capsule_bytes.data()), 
              artifact.capsule_bytes.size());
    
    // [NOTE] In a full compiler, we would write the Object Code / Binary here.
    // For this engine (which validates the Identity System), the Header is the output.
    
    std::cout << "[BuildEngine] Success. Artifact written to: " << ctx.output_path << "\n";
    return 0;
}

} // namespace ark::compiler::builder