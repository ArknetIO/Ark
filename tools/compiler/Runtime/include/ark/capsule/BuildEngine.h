#pragma once
#include <string>

namespace ark::compiler::builder {

// -----------------------------------------------------------------------------
// BUILD CONTEXT
// -----------------------------------------------------------------------------
struct BuildContext {
    std::string source_root;  // Directory to scan
    std::string output_path;  // Destination for the Capsule
    
    // Future expansion:
    // bool optimize;
    // bool debug_symbols;
};

// -----------------------------------------------------------------------------
// BUILD ENGINE
// -----------------------------------------------------------------------------
class BuildEngine {
public:
    // -------------------------------------------------------------------------
    // RUN PIPELINE
    // -------------------------------------------------------------------------
    // Orchestrates the "Scan -> Hash -> Seal" workflow.
    //
    // Workflow:
    // 1. Scanner: Crawl source_root, load files, canonicalize paths.
    // 2. PlanHasher: Construct Merkle tree, compute PlanHash.
    // 3. ArtifactSealer: Assemble Capsule V1, sign, and write to disk.
    //
    // Returns:
    // 0 on Success
    // Non-zero on failure (prints errors to stderr)
    static int Run(const BuildContext& ctx);
};

} // namespace ark::compiler::builder