# ArkLang: The AI-Native Compiler


**schemas-first systems language with zero-cost reflection and execution domains ([host]/[gpu]) that make AI + systems code live in one coherent model.**

## 🚀 Key Features (v1.0)

* **First-Class AI Primitives:**
* **`alloc<tensor[N]f32>`**: Native tensor types that lower to ranked MLIR tensors.
* **`par(i in 0..N)`**: Parallel map loops for high-throughput data processing.


* **Asynchronous Hazard Management:**
* **`<-` (Launch Operator)**: Dispatches kernels asynchronously, returning a **Token**.
* **`await`**: Explicit synchronization points based on token dependencies.
* **Automatic Hazard Tracking**: The compiler tracks write-after-write (WAW) and read-after-write (RAW) hazards and automatically injects dependencies into kernel launches.


* **MLIR-Based Backend:**
* Lowers Ark source code directly into the **Ark IR** (a custom MLIR Dialect).
* Leverages LLVM's mature optimization pipeline.



## 🛠️ Architecture

ArkLang is built on top of **LLVM 18/19** and **MLIR**.

1. **Lexer/Parser**: Custom C++ frontend that builds an AST.
2. **Ark Ops (TableGen)**: Defines the custom IR (`ark.launch`, `ark.alloc`, `ark.map`).
3. **Code Generation**: Traverses the AST and emits MLIR SSA (Static Single Assignment) form.
4. **Driver**: The `arkc` binary that orchestrates the pipeline.

## 📦 Build Instructions

### Prerequisites

* C++17 Compiler (GCC/Clang)
* CMake 3.20+
* LLVM/MLIR 18 or 19 (Built from source)

### Building

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure (Point to your LLVM install)
cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-install/lib/cmake/llvm

# 3. Compile
cmake --build . --target arkc

```

## 💻 Usage

The compiler reads `.ark` source files and outputs MLIR code.

```bash
# Compile to MLIR (printed to stdout)
./build/bin/arkc tests/monolith.ark -o -

# Compile to a file
./build/bin/arkc input.ark -o output.mlir

```

## 📝 Example Code

Here is the **Monolith Test** (`tests/monolith.ark`) which demonstrates the core capabilities:

```rust
// Define a GPU kernel
fn[gpu] add_kernel(A: f32, B: f32) -> f32 {
    return A + B;
}

fn[host] main() -> i32 {
    // 1. Native Tensor Allocation
    let A = alloc<f32[1024]> @gpu:0;
    let B = alloc<f32[1024]> @gpu:0;
    let C = alloc<f32[1024]> @gpu:0;

    // 2. Async Kernel Launch
    // "Launch add_kernel reading A, B and writing to C"
    // Returns a token 't1' for synchronization.
    C <- add_kernel(A, B) as t1;

    // 3. Synchronization
    // Wait for the token before reading C on the host
    await t1;

    // 4. Parallel Processing
    let res = par(i in 0..1024) {
        i // Parallel map body
    };

    return 0;
}

```

## 🔮 Roadmap

* **Phase 1 (Complete):** Frontend & IR Definition.
* **Phase 2:** Middle-End Optimization Passes (Parallel loop tiling, memory promotion).
* **Phase 3:** Lowering to LLVM IR (Conversion to `gpu.launch` and `memref`).
* **Phase 4:** Runtime Library (CUDA/Vulkan backend integration).

---
