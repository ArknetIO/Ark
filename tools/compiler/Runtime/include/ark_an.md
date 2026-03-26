mrkj — here’s the **frozen, language-facing spec** for **`ark-an` v1** (the `!AN` capability + `AN.*` namespace), plus the **ABI contract** the runtime/provider must honor. No implementation, only the contract we can lock and build against.

---

# ArkNet Asset Namespace (`ark-an`) v1 Spec

## 0) What `ark-an` is

`ark-an` is the **standard library + intrinsic set** that lets ArkLang programs **name, load, publish, and create immutable artifacts** on ArkNet **without paths** and without the runner ever touching filesystem traversal.

* **Artifacts** are immutable bytes/tensors, addressed by content hash (**static refs**).
* **Pointers** are mutable names that resolve to a static ref (**dynamic refs**).
* **All side effects** (publishing pointers, uploading artifacts) are **journaled intents** and only applied by the supervisor after successful execution.

---

## 1) Capability Model

### Capability

All operations in this spec require:

* `!AN` — ArkNet capability (remote assets, publication, metering)

If a program does not declare `!AN`, the compiler must reject any call into `AN.*`.

### Why

This keeps “remote data movement + money burn” in the same category as `!NET`, but scoped to ArkNet primitives. It also cleanly separates local FS (`!FS`) from ArkNet assets (`!AN`).

---

## 2) Reference Model

### Canonical textual forms (the only forms `ark-an` guarantees)

* **DynamicRef** (mutable pointer):
  `an:<path>`
* **StaticRef** (immutable CAS):
  `an:@<sha256hex>`

Examples:

* `an:mrkj/models/llama/latest`
* `an:@e3b0c44298fc1c149afbf4c8996fb924...`

### Normalization rules (must be enforced by `AN.*` constructors)

* `<sha256hex>` is **lowercase hex**, length **64**.
* `an:` prefix is canonical (accepting `arknet:` is optional compatibility, but must normalize to `an:`).
* Dynamic `<path>`:

  * UTF-8
  * segments separated by `/`
  * first segment **must not** start with `@` (reserved for static form)
  * no empty segments (`//`)
  * no `.` or `..` segments (reject)

### Types (nominal, not aliases)

These must be distinct and non-interchangeable:

* `AN.StaticRef`
* `AN.DynamicRef`
* `AN.Ref` (sum type: StaticRef | DynamicRef)

---

## 3) Core Namespace: `AN`

### 3.1 Constructors (no IO, no burn)

These only parse/validate/normalize.

* `AN.static(hex: str) -> AN.StaticRef`
* `AN.dynamic(path: str) -> AN.DynamicRef`
* `AN.parse(s: str) -> AN.Ref`
  Parses `an:@...` or `an:...` and returns the correct type.

### 3.2 Load (read-only materialization)

* `AN.load(r: AN.Ref) -> AN.Handle`

**Semantics**

* If `r` is DynamicRef: resolution happens **before runner execution** and is **pinned** for the job duration (determinism).
* Returned `Handle` is **read-only** and **immutable**.
* `AN.load` is allowed to fail before execution if policy/budget cannot be satisfied.

**Handle type**
`AN.Handle` is an opaque handle representing a read-only view of bytes/tensor content.

Minimum required methods:

* `h.len() -> u64`
* `h.bytes() -> BytesView` (read-only view)
* Optional (library convenience): `h.tensor(dtype, shape) -> TensorView` (read-only view)

### 3.3 Store (create new static artifact)

* `AN.store(x: Bytes|Tensor|Blob) -> AN.StaticRef`

**Semantics**

* `AN.store` does not mutate global state directly from within the runner.
* It records an intent “produce artifact bytes X”, producing a StaticRef (the digest of the content).
* The supervisor is responsible for actual CAS persistence and for proving the digest matches.

### 3.4 Save (publish: dynamic pointer update)

* `AN.save(dst: AN.DynamicRef, v: AN.StaticRef) -> void`

**Semantics**

* `AN.save` **never** updates the registry from inside the runner.
* It records a **publication intent**: `(dst -> v)`.
* This intent is applied **only if**:

  1. job exit status is success (see §6)
  2. metering/burn constraints are satisfied
  3. registry authorization allows writing `dst`

---

## 4) Determinism + Snapshot Contract

### Snapshot rule (hard)

For any job execution:

* Every DynamicRef used by `AN.load` must be resolved to a StaticRef **once**, up front.
* That mapping is immutable during the job.

This prevents “mid-run pointer drift” and makes results reproducible.

---

## 5) Metering Contract (Security + “0 file size limit”)

### No protocol file size limit

Refs may represent arbitrarily large data (TB+). The *system* enforces safety by metering and policy.

### What is metered (minimum)

The ArkNet runtime must meter:

* bytes fetched from network / peers
* bytes read into GPU-visible memory / staging
* CAS operations count (manifest/chunk fetch)
* publication ops (save)
* optional: cache occupancy impact (eviction pressure)

### Budget controls (language-facing)

These are not “AI-specific”; they’re execution safety.

* `AN.burn_usd() -> f64`
  Returns burn consumed so far (observed).
* `AN.assert_burn(max_usd: f64) -> void`
  Fail-fast if burn exceeds cap (records a controlled failure).

**Default rule**: if a job has a configured burn cap, runtime must enforce it even if program never calls `AN.assert_burn`.

---

## 6) Completion + Publication Rules

### Success definition (v1)

A job is **successful** if:

* runner exits with code `0` and was not killed for policy (timeout, burn cap, sandbox violation)

Only successful jobs may apply publication intents.

### Publication application order

If multiple `AN.save()` intents exist:

* apply in program order
* if two saves target the same DynamicRef, the last one wins
* failures must be reported distinctly (compute success can still happen if publication fails)

---

## 7) Error Model (language-level)

`AN.*` operations may raise failures with **stable error codes** (not strings).

Minimum codes:

* `AN_ERR_INVALID_REF`
* `AN_ERR_UNAUTHORIZED`
* `AN_ERR_NOT_FOUND`
* `AN_ERR_BUDGET_EXCEEDED`
* `AN_ERR_POLICY_REFUSED` (provider policy: too large, no cache, etc.)
* `AN_ERR_IO`
* `AN_ERR_INTERNAL`

Language behavior:

* uncaught `AN_ERR_*` terminates execution with non-zero code
* termination due to `AN.assert_burn` is treated as a controlled failure (distinct status)

---

# 8) Frozen ABI (Runtime ↔ Supervisor ↔ Runner)

This is the part we “freeze” so `tools/provider` can implement it later with FD-passing and journaling.

## 8.1 Control Channel

* Transport: `AF_UNIX`, `SOCK_SEQPACKET`
* Supervisor creates a socketpair and passes **one end** to runner (fixed FD or env var).
* All asset data access is via **capabilities (FDs)** passed with `SCM_RIGHTS`.

## 8.2 Message framing (stable header)

All messages are:

* 1 header + payload bytes
* optional SCM_RIGHTS FD list

**Header fields (fixed)**

* `magic` = `"ARKAN1\0"` (8 bytes)
* `version` = `1` (u16)
* `op` (u16)
* `flags` (u32)
* `req_id` (u64)
* `payload_len` (u32)
* `reserved` (u32)

## 8.3 Operations (opcodes)

Minimum v1 opcodes:

**Runner → Supervisor**

* `OP_LOAD_REF`
  Payload: `Ref` (normalized string or structured form)
  Response: `OP_LOAD_OK` + 1 FD (read-only artifact)
* `OP_STORE_BEGIN`
  Announces intent to store produced output (metadata only)
* `OP_STORE_CHUNK`
  Streams bytes out (or passes FD) into supervisor-controlled staging
* `OP_STORE_END`
  Finalizes digest; response returns `StaticRef`
* `OP_SAVE_INTENT`
  Payload: `(DynamicRef, StaticRef)` recorded in job journal
* `OP_METER_READ`
  Response: burn counters
* `OP_ASSERT_BURN`
  Payload: max_usd; response indicates pass/fail

**Supervisor → Runner**

* `OP_FAIL`
  Payload: error code + optional short detail
* `OP_OK` / specific OK responses

## 8.4 Ref wire format (freeze it)

Ref must be structured, not “just a string”, to avoid future ambiguity:

* `ref_kind` (u8): 1=static, 2=dynamic
* `hash` (32 bytes) for static
* `path_len + path_bytes` for dynamic (utf-8)

Canonicalization happens before anything reaches the runner.

## 8.5 Artifact delivery rule (freeze it)

* `OP_LOAD_OK` must include:

  * 1 FD that is readable, seekable, mmap-able (read-only)
  * payload includes `len` (u64) and `static_hash` (32 bytes)

Runner never receives paths.

---

# 9) Simple user-facing surface (ArkLang)

You get the minimal, natural primitives:

* `AN.load(ref)` → handle
* `AN.store(data)` → `an:@...`
* `AN.save(dynamic, static)` → records intent
* `AN.burn_usd()` / `AN.assert_burn(max)`

And refs are simple:

* dynamic: `an:...`
* static: `an:@...`

Also provide the sugar you asked for:

* `AN.static("...")`, `AN.dynamic("...")`
* optional parsing: `AN.parse("an:...")`

---

## One-liner improvement

Add `AN.pin(ref) -> AN.StaticRef` as a pure “resolve + snapshot” primitive so the compiler/runtime can force determinism early without changing user code structure.
