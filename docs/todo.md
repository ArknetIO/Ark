**Rating (where ArkLang is today): 6/10 toward “complete language”.** Front-end + IR laws are strong; you’re missing the pieces that make it *run*, *scale*, and *not rot*.

## Next, in order (brutally practical)

### 1) Seal the core AST + typing

* **Let/Return** as real nodes (or an explicit desugar pass).
* **Name resolution + scopes** (block scopes, shadowing rules).
* **Typecheck**: scalar/tensor element kind, shape, memspace, domain/effect annotations.
* **Indexing**: `x[i]` node + rules (index type, bounds policy, memspace legality).

### 2) Finish the IR contract so middle-end can’t guess

* Make `ark.launch` canonical: `(dest, args..., deps...) -> (dest', token)` and keep `deps` as `!ark.token` operands.
* Define `ark.map` contract: `ark.map %ub { ... } -> tensor<ub x T>` (you basically have this now).
* Add `ark.load/ark.store` or `ark.extract/ark.insert` to support indexing and buffer lowering.
* Add function attributes for **effects** (not just domain) so legality survives beyond the frontend.

### 3) A real lowering pipeline to execution

**Minimal CPU-only “it runs” path:**

* `ark.map` → `scf.for/scf.parallel` (still tensor or bufferized memref form).
* Bufferization (one-shot) or commit to memref-first IR.
* Lower to LLVM dialect.
* JIT with `mlir::ExecutionEngine`.

**Mock GPU semantics (still CPU):**

* Lower `ark.launch/await` to explicit sequencing:

  * easiest: inline as a call + treat token as an SSA chain (deps enforced by operands)
  * better: lower to MLIR `async` dialect (`async.execute/async.await`) then to runtime.

### 4) Runtime + ABI

* Define the ABI for tensors/memrefs (`memref` descriptor vs custom).
* Implement allocation/deallocation + printing/debug hooks.
* Decide host entrypoint: `main() -> i32` becomes a C ABI symbol.

### 5) Modules, imports, and build story

* `import` / module path resolution.
* Separate compilation (emit MLIR, link MLIR modules, or emit object files).
* A stable CLI: `arkc file.ark -O2 --emit=mlir|obj|exe`.

### 6) Standard library + diagnostics

* Minimal `core` (types, intrinsics, maybe `println` gated by `!{io}`).
* Great errors: domain/effect/hazard diagnostics with source ranges.

### 7) Optimization + verification

* Canonicalization patterns for Ark ops.
* Verification rules in the dialect (so illegal IR can’t exist).
* IR tests (FileCheck) for every law and every lowering stage.

Suggestion: pick one “hello world that actually executes” target (CPU-only) and drive everything from that pipeline—execution pressure instantly reveals what’s missing and prevents building ornamental features.
