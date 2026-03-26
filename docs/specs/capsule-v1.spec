SPECIFICATION: Arknet Capsule Format v1
STATUS: LOCKED
VERSION: 1.1.0

1. CARRIER FORMAT
   - ELF Targets:
     - Section Name: .note.ark.provenance
     - Section Type: SHT_NOTE
     - Segment:      PT_NOTE
     - Flags:        SHF_ALLOC
     - Constraint:   Must survive `strip --strip-unneeded` and `objcopy`.
   - Non-ELF Targets:
     - Sidecar File: ${BINARY_NAME}.arkp (Ark Package)

2. ENCODING (Canonical Bytes)
   - Endianness: Little Endian (LE)
   - Alignment:  4-byte aligned
   - Padding:    Zero-filled, only to alignment boundaries.

3. SCHEMA
   [Header]
     Magic:        4 bytes (0x41 0x52 0x4B 0x43) "ARKC"
     Version:      2 bytes (0x00 0x01)
     Flags:        4 bytes (Bitmask)
                   - Bit 0: HasSymTab
                   - Bit 1: HasDebugSidecar
                   - Bit 2: DeterministicIO
                   - Bits 3-31: Reserved (0)
     TotalLen:     4 bytes (Size of Header + Sum of all TLVs including Signature)

   [TLV Records]
     Format: | Tag (2b) | Len (2b) | Value (Len bytes) |

     Tags:
     0x0001 PlanHash (32b):   BLAKE3 hash of the Canonical Build Plan.
     0x0002 CompilerID (Var): Canonical Toolchain String.
     0x0003 RuntimeReqs (Var): Deterministic Requirements Blob.
     0x0004 SymTabHash (32b): BLAKE3 hash of the .ark.symtab section.
     
     0xFFFF Signature (64b):  Ed25519 signature.

4. TLV CANONICALIZATION RULES
   - Tag 0x0001 (PlanHash) MUST be the first TLV.
   - Tags 0x0001..0x0004 MUST NOT appear more than once.
   - Tag 0xFFFF (Signature) MUST be the final TLV.
   - Unknown tags are allowed ONLY if ordered strictly after known tags and before Signature.
   - Parser MUST reject if `CalculatedLength != Header.TotalLen`.

5. RUNTIME REQUIREMENTS BLOB (Tag 0x0003)
   - Structure (Packed, LE):
     - ReqVersion (u16):   1
     - DeviceClass (u8):   { cpu=1, gpu=2, fpga=3, other=255 }
     - IRFamily (u8):      { ark_gpu_ir=1, spirv=2, wasm=3, other=255 }
     - FeatureCount (u16) + Features[] (ID + MinVer)
     - LimitCount (u16) + Limits[] (ID + Value)
   - Rule: NO vendor strings.

6. SIGNING ENVELOPE
   - Message = Header || All TLVs (excluding Signature TLV).
   - Signature = Ed25519(PrivKey, Message).
   - The Signature bytes are wrapped in Tag 0xFFFF and appended.