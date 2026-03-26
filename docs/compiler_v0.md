## The minimal “Compiler v0” milestone (2 weeks)

If you do nothing else, do this:

**Week A**

* Canonical compiler CLI contract (`--emit`, `--out`, `--target`, `--gpu`)
* Deterministic temp dirs + artifact naming
* External tool runner: captures stdout/stderr, shows once, tagged per step

**Week B**

* Import/module resolution finalized
* Runtime ABI pinned + versioned
* 10 E2E tests: parsing errors, type errors, linker error, run error, gpu missing, gpu ok

After that: start registry.

---

## The key decision that keeps you moving fast

Treat GPU as:

* **experimental feature flag** (`--gpu`) for v0 registry period
* still supported, still demo-worthy, but doesn’t block shipping core ecosystem

One-line improvement: add a `arkc --selftest` command that runs your full “no silent failures + e2e compile/run” suite in under 60 seconds.



This is the official migration checklist based on the features found in your legacy `CodeGen.cpp`.

Our goal is to port these into `GenMIR.cpp`, adapting them to emit **ArkMIR operations** (`ark.slot`, `ark.store`, `ark.index`) instead of raw LLVM instructions.

### **Phase 1: Infrastructure & Scope**

These are the foundational changes needed to make `GenMIR` aware of the project structure.

* [ ] **Function Attributes:** Port logic to tag `main` with `llvm.emit_c_interface` and handle GPU domains.
* [ ] **Recursive Module Handling:** Ensure `lowerModule` recursively compiles imports so types are available.
* [ ] **Import Injection:** Populate the symbol table with imported modules (e.g., `import std.io as io`) so `io.print` works.
* [ ] **Capability Injection:** Add the "Magic Symbols" for `FS`, `NET`, and `IO` to the symbol table if the function declares effects.

### **Phase 2: Complex Types (The "Struct" System)**

Moving beyond primitives (`i32`) to compound data types.

* [ ] **Struct Definition:** Port `registerSchema` to lower `SchemaDecl` into MLIR Struct types.
* [ ] **Generic Monomorphization:** Port the logic to instantiate `Box<T>` into `Box_i32` on demand.
* [ ] **Enum/Variant Layouts:** Port the logic for Tagged Unions (Enums with payloads).
* [ ] **Struct Initialization:** Implement `ExprKind::SchemaExpr` to allocate and fill structs using ArkMIR slots.
* [ ] **Member Access (`.`):** Implement `ExprKind::MemberAccess` to resolve fields (`p.x`), enum variants (`Color.Red`), and module exports.

### **Phase 3: Collections & Memory**

Handling arrays, vectors, and heap allocation via the runtime.

* [ ] **Heap Allocation (`alloc`):** Port `ExprKind::Alloc` to call the runtime allocator.
* [ ] **Array Literals:** Implement `[1, 2, 3]` creation (stack allocation + initialization).
* [ ] **Indexing (`[]`):** Implement `ExprKind::Index` for Arrays and Slices using `ark.index`.
* [ ] **String Operations:** Port string comparison (`==`) and concatenation (`+`) logic.

### **Phase 4: Advanced Control Flow**

Migrating loops and pattern matching.

* [ ] **Pattern Matching:** Port `ExprKind::Match` (Switch statements for Enums).
* [ ] **While Loops:** Port `ExprKind::While`.
* [ ] **Range Loops (`for`):** Port `ExprKind::For` (integer ranges).
* [ ] **Iterator Loops (`iter`):** Port `ExprKind::Iter` (looping over vectors/arrays).
* [ ] **Parallel Loops:** Port `genParLoop` (if supported in ArkMIR).

### **Phase 5: Runtime Intrinsics**

The special handling for system calls.

* [ ] **System Capabilities:** Port the dispatch logic for `FS.*`, `NET.*`, and `IO.*` methods.
* [ ] **Async/Await:** Port `Launch` and `Await` primitives (if not deferred to a later milestone).

---

### **Immediate Next Step**

We have already stabilized the pipeline. The most logical next step is **Phase 1: Infrastructure**, specifically **Function Attributes** and **Imports**, as these are often blockers for running anything complex.

Shall we start with **Function Attributes** (C-interface compatibility) to check that off the list?


Here’s a **12-week plan** (milestones) + a **daily cadence** that hits growth without diluting product quality. No fluff, everything ships.

## Weekly milestones (12 weeks)

### Week 1 — Foundations (no shortcuts)

* Define **Ark Package spec v0** (manifest, semver, targets, features, checksums)
* Define **Registry API v0** (publish, yank, resolve, download, auth)
* Define **Build Artifact format v0** (SBOM/provenance hooks, hash rules)
* Ship: `ark pkg init`, `ark pkg build`, `ark pkg lock` (local-only)

### Week 2 — Local registry + reproducibility

* Implement **local registry** (filesystem backend)
* Deterministic resolves + lockfile enforcement
* Ship: `ark pkg add`, `ark pkg resolve`, `ark pkg publish --local`

### Week 3 — Hosted registry MVP (paid wedge begins)

* HTTP registry service (minimal): users/orgs/tokens, publish, fetch, index
* Upload/download with content-addressed blobs
* Ship: `ark login`, `ark publish`, `ark add foo@1.2.3`

### Week 4 — Trust + safety (product quality gate)

* Package signing v0 (publisher key), transparency log v0 (append-only)
* Malware/basic policy scanning hooks (pluggable)
* Ship: “verified publisher” badge mechanism

### Week 5 — Build cache MVP

* Remote cache protocol: key = (source hash + toolchain + target + flags)
* Local cache + remote cache client
* Ship: `ark build --cache=on` speedup demo

### Week 6 — CI + DX “feels premium”

* GitHub Actions template + `ark ci` helper
* Better errors: show captured tool stderr as **HUD blocks**
* Ship: `ark doctor` (detect CUDA/HIP toolchain, registry auth, cache)

### Week 7 — GPU/CPU mixed program “killer demo”

* 1 flagship example: host pipeline + GPU kernels + fallback path
* Add “capability probing”: run CPU if GPU missing
* Ship: benchmark + video + blog-quality docs

### Week 8 — Ecosystem ignition

* Seed **20 core packages** (stdlib-ish): math, tensor, io, cli, http, json
* Strict quality bar: tests, docs, examples
* Ship: `ark add ark/http` etc.

### Week 9 — IDE baseline

* LSP MVP: go-to-def, hover, diagnostics, formatting stub
* Ship: VSCode extension minimal

### Week 10 — Teams features (money)

* Orgs, private packages, RBAC, audit log
* Billing stub + usage metering (downloads/storage)
* Ship: “Team” plan exists, first paying users

### Week 11 — Hardening + release process

* Toolchain versioning policy, release channels (stable/nightly)
* Crash reporting opt-in, telemetry opt-in, privacy docs
* Ship: v0.1 stable of pkg+registry

### Week 12 — Growth loop

* Docs site + onboarding flow
* “Publish your first package” funnel
* Ship: monthly newsletter + community cadence

---

## Daily plan (repeatable cadence)

**Every day has 4 blocks. No exceptions.**

### Block A — Product (3–4h)

* Implement 1 small shippable slice
* Must include tests + docs stub
* Merge behind feature flag if risky

### Block B — Quality (1–2h)

* Run full suite + fix regressions
* Perf check on 1 representative project
* “No silent failures” audit (especially HUD + tool invocations)

### Block C — Ecosystem (1–2h)

* Improve 1 package or add 1 example
* Write 1 page of docs (short, runnable)
* Close 1 issue that blocks onboarding

### Block D — Growth (45–90m)

* Publish 1 artifact: demo clip / update post / changelog
* Talk to 1 developer (DM, Discord, email) and log feedback
* Add 1 “activation” improvement (onboarding friction)

**Daily exit criteria (non-negotiable)**

* ✅ One PR merged (or queued) with tests
* ✅ One user-facing improvement shipped (docs/example/tooling)
* ✅ One growth touch (public update or user convo)
* ✅ Build remains reproducible / no product compromises

---

## Guardrails so growth never compromises product

* “**No test, no merge**” for registry/pkg/cache core
* “**No breaking manifest changes without migration**”
* “**No silent tool failures**” — always capture stdout/stderr into HUD blocks
* “**One flagship project** must always build on main”

One-line improvement: pick **one** killer vertical (e.g., GPU compute + backend services) and make every weekly milestone land as measurable progress toward that demo.


