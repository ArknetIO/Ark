# docs/specs/vectors/capsule_v1_golden.md

# Capsule V1 Golden Vector


**Format:** Binary (Little Endian)

**Signature Scheme:** Ed25519

**Wire Encoding:** TLV with `u16_le tag`, `u16_le len`, then `bytes[len]`


## 1. Minimal Valid Capsule


**Components**

1. Header (16 bytes)

2. PlanHash TLV (Tag `0x0001`)

3. CompilerID TLV (Tag `0x0002`)

4. RuntimeReqs TLV (Tag `0x0003`)

5. Signature TLV (Tag `0xFFFF`)


### 1.1 Header (16 bytes)

Header fields (little endian):

- Magic: 4 bytes: `"ARKC"` (`41 52 4B 43`)

- Version: `u16_le` (example: `01 00`)

- Reserved: `u16_le` (must be `00 00`)

- Flags: `u32_le` (example: `00 00 00 00`)

- TotalLen: `u32_le` (`134` = `86 00 00 00`)


### 1.2 Total Length Calculation

- Header: 16

- PlanHash TLV: 4 + 32 = 36

- CompilerID TLV: 4 + 4 = 8

- RuntimeReqs TLV: 4 + 2 = 6

- Signature TLV: 4 + 64 = 68

- **Total:** 16 + 36 + 8 + 6 + 68 = **134 bytes**


### 1.3 Hex Stream (134 bytes)

Notes:

- `[32-byte PlanHash]` is the PlanHash from `planhash_v1_golden.md`

- `[64-byte Signature]` is the Ed25519 signature over the **signed region** (§1.4)


~~~text

41 52 4B 43 01 00 00 00 00 00 00 00 86 00 00 00

01 00 20 00 [32-byte PlanHash]

02 00 04 00 61 72 6B 63

03 00 02 00 AA BB

FF FF 40 00 [64-byte Signature]

~~~


## 1.4 Signed Region (Normative)


Let:

- `SIG_TLV_SIZE = 4 + 64 = 68`

- `total_len = 134`


Then:

- `signed_len = total_len - SIG_TLV_SIZE = 134 - 68 = 66`

- **Signed byte range:** `[0, signed_len)` = bytes `[0, 66)`


**The signature covers:**

- Header (16 bytes)

- PlanHash TLV (36 bytes)

- CompilerID TLV (8 bytes)

- RuntimeReqs TLV (6 bytes)


**The signature explicitly excludes:**

- The Signature TLV itself (`tag=0xFFFF`, `len=64`, 68 bytes total)


## 1.5 TLV Ordering (Payload Region)

In the payload region (after header, before signature TLV):

- TLVs MUST be in strictly increasing tag order

- `0xFFFF` MUST NOT appear in the payload (it is reserved for the signature TLV at the end)

