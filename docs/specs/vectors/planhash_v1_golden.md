# docs/specs/vectors/planhash_v1_golden.md

# PlanHash V1 Golden Vector


**Algorithm:** BLAKE3-256

**Endianness:** Little Endian (LE)

**String Encoding:** UTF-8 bytes (no null terminators)

**Recursion:** Bottom-Up (leaves first)


## 1. Canonical Hashing Format (Normative)


PlanHash inputs are serialized using a **Canonical Hashing Format** that is **distinct** from the Capsule wire TLV format.


### 1.1 Primitive Encodings

- `u8`: 1 byte

- `u16_le`: 2 bytes, little endian

- `u32_le`: 4 bytes, little endian

- `bytes[N]`: N raw bytes


### 1.2 Canonical Field Pattern

For top-level PlanHash fields:

- **Tag:** `u8`

- **Len:** `u32_le`

- **Value:** `bytes[Len]`


For node hashing:

- **NodeTag:** `u8`

- **PathLen:** `u32_le`

- **Path:** `bytes[PathLen]`

- **ContentHash:** `bytes[32]`

- **ChildCount:** `u32_le`

- **ChildHash[i]:** `bytes[32]` repeated `ChildCount` times


## 2. Node Hash (Level 0)


**Inputs**

- Path: `"main"` (`6D 61 69 6E`)

- ContentHash: 32 bytes of `0x11`

- Children: 1 child hash, 32 bytes of `0x22`


**Serialization Layout (77 bytes)**

1. NodeTag: `0x10`

2. PathLen: `4` (`04 00 00 00`)

3. Path: `"main"` (`6D 61 69 6E`)

4. ContentHash: `11` × 32

5. ChildCount: `1` (`01 00 00 00`)

6. ChildHash[0]: `22` × 32


**Hex Stream (77 bytes)**

~~~text

10 04 00 00 00 6D 61 69 6E

11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11

11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11

01 00 00 00

22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22

22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22

~~~


**Normative NodeHash (BLAKE3-256, 32 bytes)**

~~~text

38 c2 90 23 02 19 46 39 94 35 01 65 97 5d 40 01

15 01 63 35 96 95 62 72 37 07 76 96 61 13 70 7e

~~~


## 3. Plan Hash (Top Level)


**Inputs**

- CompilerID: `"arkc"` (`61 72 6B 63`)

- Reqs: `AA BB`

- RootNodeHash: NodeHash from §2


**Serialization Layout (53 bytes)**

1. Tag `0x01`, Len `4`, Value `"arkc"`

2. Tag `0x02`, Len `2`, Value `AA BB`

3. Tag `0x03`, Len `32`, Value RootNodeHash


**Hex Stream (53 bytes)**

~~~text

01 04 00 00 00 61 72 6B 63

02 02 00 00 00 AA BB

03 20 00 00 00

38 c2 90 23 02 19 46 39 94 35 01 65 97 5d 40 01

15 01 63 35 96 95 62 72 37 07 76 96 61 13 70 7e

~~~


**Normative PlanHash (BLAKE3-256, 32 bytes)**

~~~text

f9 32 45 4e 99 92 c6 40 30 61 24 47 99 79 70 95

69 77 93 d2 58 e7 27 53 06 89 e4 72 25 12 f4 92

~~~ 