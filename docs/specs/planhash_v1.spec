SPECIFICATION: Arknet Build Plan Hash v1
STATUS: LOCKED
VERSION: 1.1.0

1. DEFINITION
   The PlanHash is the Merkle Root of the execution semantics and dependency closure.

2. INPUTS (Canonical Order - Serialization)
   The hash is computed over a virtual struct serialized in this EXACT order:

   A. Toolchain Identity
      - Compiler Version String
      - Linker/LTO Mode (Canonical Enum)

   B. Execution Semantics
      - RuntimeRequirements Bytes (Copy of Capsule Tag 0x0003)
      - OptimizationProfile (Canonical Enum)
      - LanguageABI (Canonical Version ID)

   C. Dependency Graph (Merkle)
      - Root Node = Project Source Root
      - Path Rules:
        - Separator: '/'
        - No '..' segments. No absolute paths.
        - Symlinks: Must resolve inside the Root. Resolving outside = Build Failure.
      
      - Node Hash = BLAKE3( 
          CanonicalRelativePath || 
          ContentHash32 ||         <-- UPDATED: Hash of content, not bytes
          SortedDependencyHashes32[] 
        )
      - ContentHash32 = BLAKE3(File Bytes)

3. ALGORITHM
   - Algo: BLAKE3
   - Encoding: Binary concatenation of length-prefixed fields.