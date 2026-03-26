SPECIFICATION: Arknet Execution Receipt v1
STATUS: LOCKED
VERSION: 1.1.0

1. PURPOSE
   Proves that a specific Provider executed a specific BuildID under a specific Nonce.

2. SCHEMA
   [Header]
     Magic:        0x41 0x52 0x4B 0x52 "ARKR"
     Version:      1

   [Fields - Fixed Order]
     1. JobID (16b):        UUIDv7.
     2. Nonce (32b):        Supervisor Challenge (Anti-Replay).
     3. BuildID (32b):      Matches Capsule PlanHash.
     4. ProviderID (32b):   BLAKE3(Ed25519PubKeyBytes).
     
     5. SLO Manifest (Var): Target + Measured Latency.
     
     6. Execution Outcome:
        - Status (u8):      { success=1, fail=2, timeout=3, killed=4 }
        - ExitCode (s32).
     
     7. Output Commitments:
        - StdoutHash (32b): BLAKE3(stdout).
        - StderrHash (32b): BLAKE3(stderr) or Zero.
        - ArtifactHash (32b): Merkle Root of outputs or Zero.

     8. Timing: StartTime (u64) + Duration (u64).

   [Signature]
     - Algo: Ed25519
     - Input: Header || Fields (Raw Canonical Bytes).
     - Key: Provider's Private Key.

3. VALIDATION RULES
   - Verify(ProviderPubKey, Signature, Header || Fields) MUST pass.
   - Nonce MUST match Supervisor record.
   - BuildID MUST match Capsule PlanHash.