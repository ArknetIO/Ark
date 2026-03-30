<div align="center">

<img src="https://arknet.io/logo.svg" alt="Arknet logo" width="140" />

# ArkLang
### The AI-native compiler and runtime toolchain for Arknet

**Schemas-first systems programming, execution domains, native async orchestration, and GPU-aware lowering in one coherent compiler stack.**

</div>

---

## What ArkLang is

ArkLang is the compiler and runtime layer behind Arknet.

It is built for programs that need:
- explicit host and GPU execution domains
- native async launch / await semantics
- compiler-managed hazard tracking
- MLIR / LLVM lowering for serious optimization and backend extensibility

ArkLang is not just a frontend. It is a full pipeline:
source → MIR → LLVM dialect → LLVM IR → native binary → optional sealed capsule.

The primary developer-facing tool is **`arknet`**.

`arkc` still exists as the lower-level compiler driver, but **new onboarding and normal workflows should start with `arknet`**.

---

## Why this repository exists

This repository contains:

- the Ark frontend
- Ark MIR generation
- custom MLIR dialects and lowering passes
- the compiler drivers
- the platform runtime
- GPU lowering and fatbin integration
- the self-executing capsule stub

---

## Core capabilities

### Execution domains
ArkLang supports explicit execution domains such as:

```ark
fn[gpu] add_kernel(A: tensor<f32>, B: tensor<f32>, C: tensor<f32>) {
    par i in C {
        C[i] = A[i] + B[i];
    }
}
```

and:

```ark
fn[host] main() -> i32 !IO {
    return 0;
}
```

### Async orchestration
ArkLang makes async launch and synchronization explicit:

```ark
C <- add_kernel(A, B, C) as t1;
await t1;
```

### MLIR / LLVM backend
ArkLang lowers through:
- Ark MIR
- LLVM dialect MLIR
- LLVM IR
- native executable output

### Cross-platform runtime
The compiler and runtime are being brought up across:
- Windows
- Linux
- macOS

---

## Build targets

These source-build instructions build:

- core libraries
- runtime
- compiler stack
- **`arknet`**
- `arkc`
- `ark-stub`

These instructions intentionally do **not** build backend services by default.

---

## Version requirements

- **CMake 3.20+**
- **C++20-capable toolchain**
- **Git**
- **LLVM + MLIR with the targets Ark needs**
- **libsodium**

For the current compiler stack in this repository, the practical requirement is:

- **LLVM + MLIR built with `Native;NVPTX;AMDGPU`**

---

## Repository layout

```text
tools/compiler/          Compiler drivers, pipeline, CLI, runtime integration
tools/compiler/Runtime/  Runtime sources used by produced binaries
libs/ark-ir/             Ark MLIR dialects and lowering
libs/ark-abi/            ABI headers copied into Runtime/include
libs/ark-wire/           Wire / protocol headers copied into Runtime/include
libs/ark-capsule/        Capsule packaging / sealing support
libs/ark-crypto/         Crypto and verification support
tests/                   Compiler and runtime test inputs
```

---

# Build From Source

## Windows

### 1. Install prerequisites

Install:

- **Visual Studio 2022** with **Desktop development with C++**
- **CMake**
- **Git**

Use:

- **x64 Native Tools PowerShell for VS 2022**

Install Git and CMake:

```powershell
winget install Kitware.CMake
winget install Git.Git
```

---

### 2. Install vcpkg and libsodium

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\dev\vcpkg
cd C:\dev\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install libsodium:x64-windows
```

Verify:

```powershell
Test-Path C:\dev\vcpkg\installed\x64-windows\include\sodium.h
Test-Path C:\dev\vcpkg\installed\x64-windows\lib\libsodium.lib
Test-Path C:\dev\vcpkg\installed\x64-windows\debug\lib\libsodium.lib
```

All three commands must print `True`.

---

### 3. Clone LLVM monorepo

```powershell
git clone https://github.com/llvm/llvm-project.git C:\dev\llvm-project
```

---

### 4. Configure LLVM + MLIR

Use dedicated build and install directories:

- build: `C:\dev\llvm-build-release`
- install: `C:\dev\llvm-install-release`

```powershell
cmake -S C:\dev\llvm-project\llvm -B C:\dev\llvm-build-release `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DLLVM_ENABLE_PROJECTS=mlir `
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" `
  -DLLVM_ENABLE_ASSERTIONS=ON `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_EXAMPLES=OFF `
  -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF `
  -DCMAKE_INSTALL_PREFIX=C:\dev\llvm-install-release
```

---

### 5. Build LLVM + MLIR

```powershell
cmake --build C:\dev\llvm-build-release --config Release -- /m
```

If the build is interrupted, **do not delete** `C:\dev\llvm-build-release`. Run the same command again and continue incrementally.

---

### 6. Install LLVM + MLIR

```powershell
cmake --install C:\dev\llvm-build-release --config Release --prefix C:\dev\llvm-install-release
```

---

### 7. Verify LLVM + MLIR install

```powershell
Test-Path C:\dev\llvm-install-release\lib\cmake\llvm\LLVMConfig.cmake
Test-Path C:\dev\llvm-install-release\lib\cmake\mlir\MLIRConfig.cmake
Test-Path C:\dev\llvm-install-release\lib\LLVMOrcJIT.lib
Test-Path C:\dev\llvm-install-release\lib\LLVMAMDGPUCodeGen.lib
Test-Path C:\dev\llvm-install-release\lib\MLIRVectorToLLVM.lib
Test-Path C:\dev\llvm-install-release\lib\MLIRIndexDialect.lib
Test-Path C:\dev\llvm-install-release\lib\MLIRROCDLDialect.lib
```

All commands must print `True`.

---

### 8. Clone Ark

```powershell
git clone https://github.com/ArknetIO/Ark.git C:\dev\Ark
cd C:\dev\Ark
```

---

### 9. Configure Ark

```powershell
cmake -S . -B build-release `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DARK_BUILD_TOOLS=ON `
  -DARK_BUILD_SERVICES=OFF `
  -DLLVM_DIR=C:\dev\llvm-install-release\lib\cmake\llvm `
  -DMLIR_DIR=C:\dev\llvm-install-release\lib\cmake\mlir
```

---

### 10. Build Ark

```powershell
cmake --build build-release --config Release --target arknet arkc ark-stub -- /m
```

---

### 11. Run the main tool

The main target is **`arknet`**.

```powershell
.\build-release\tools\compiler\Release\arknet.exe run .\tests\args.ark
```

You can also use the lower-level compiler directly:

```powershell
.\build-release\tools\compiler\Release\arkc.exe .\tests\args.ark
```

---

### 12. Core libraries only

```powershell
cmake -S . -B build-core-release `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DARK_BUILD_TOOLS=OFF `
  -DARK_BUILD_SERVICES=OFF

cmake --build build-core-release --config Release -- /m
```

---

## Linux

### 1. Install prerequisites

Ubuntu / Debian example:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  clang \
  lld \
  cmake \
  ninja-build \
  git \
  python3 \
  python3-pip \
  pkg-config \
  libsodium-dev \
  zlib1g-dev
```

---

### 2. Build and install LLVM + MLIR

```bash
git clone https://github.com/llvm/llvm-project.git "$HOME/llvm-project"

cmake -S "$HOME/llvm-project/llvm" -B "$HOME/llvm-build-release" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/llvm-install-release" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON

cmake --build "$HOME/llvm-build-release" --parallel
cmake --install "$HOME/llvm-build-release"
```

Verify:

```bash
test -f "$HOME/llvm-install-release/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$HOME/llvm-install-release/lib/cmake/mlir/MLIRConfig.cmake"
```

---

### 3. Clone Ark

```bash
git clone https://github.com/ArknetIO/Ark.git "$HOME/Ark"
cd "$HOME/Ark"
```

---

### 4. Configure Ark

```bash
cmake -S . -B build-release -G Ninja \
  -DARK_BUILD_TOOLS=ON \
  -DARK_BUILD_SERVICES=OFF \
  -DLLVM_DIR="$HOME/llvm-install-release/lib/cmake/llvm" \
  -DMLIR_DIR="$HOME/llvm-install-release/lib/cmake/mlir"
```

---

### 5. Build Ark

```bash
cmake --build build-release --target arknet arkc ark-stub --parallel
```

---

### 6. Run the main tool

```bash
./build-release/tools/compiler/arknet run ./tests/args.ark
```

---

### 7. Core libraries only

```bash
cmake -S . -B build-core-release -G Ninja \
  -DARK_BUILD_TOOLS=OFF \
  -DARK_BUILD_SERVICES=OFF

cmake --build build-core-release --parallel
```

---

## macOS

### 1. Install prerequisites

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Install dependencies:

```bash
brew install cmake ninja git pkg-config libsodium
```

---

### 2. Build and install LLVM + MLIR

```bash
git clone https://github.com/llvm/llvm-project.git "$HOME/llvm-project"

cmake -S "$HOME/llvm-project/llvm" -B "$HOME/llvm-build-release" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/llvm-install-release"

cmake --build "$HOME/llvm-build-release" --parallel
cmake --install "$HOME/llvm-build-release"
```

Verify:

```bash
test -f "$HOME/llvm-install-release/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$HOME/llvm-install-release/lib/cmake/mlir/MLIRConfig.cmake"
```

---

### 3. Clone Ark

```bash
git clone https://github.com/ArknetIO/Ark.git "$HOME/Ark"
cd "$HOME/Ark"
```

---

### 4. Configure Ark

```bash
cmake -S . -B build-release -G Ninja \
  -DARK_BUILD_TOOLS=ON \
  -DARK_BUILD_SERVICES=OFF \
  -DLLVM_DIR="$HOME/llvm-install-release/lib/cmake/llvm" \
  -DMLIR_DIR="$HOME/llvm-install-release/lib/cmake/mlir"
```

---

### 5. Build Ark

```bash
cmake --build build-release --target arknet arkc ark-stub --parallel
```

---

### 6. Run the main tool

```bash
./build-release/tools/compiler/arknet run ./tests/args.ark
```

---

### 7. Core libraries only

```bash
cmake -S . -B build-core-release -G Ninja \
  -DARK_BUILD_TOOLS=OFF \
  -DARK_BUILD_SERVICES=OFF

cmake --build build-core-release --parallel
```

---

# Using the compiler

## Primary tool: `arknet`

Normal compiler workflow should use `arknet`.

Examples:

```bash
arknet run tests/args.ark
```

```bash
arknet run tests/monolith.ark
```

If you need the exact executable path from the build tree:

### Windows

```powershell
.\build-release\tools\compiler\Release\arknet.exe run .\tests\args.ark
```

### Linux / macOS

```bash
./build-release/tools/compiler/arknet run ./tests/args.ark
```

---

## Low-level driver: `arkc`

`arkc` is still useful for direct pipeline work, staging, and lower-level debugging.

Examples:

```bash
arkc tests/args.ark
```

```bash
arkc --emit-mlir tests/args.ark -o args.mlir
```

```bash
arkc --emit-llvm-dialect tests/args.ark -o args.llvm.mlir
```

```bash
arkc --emit-llvm-ir tests/args.ark -o args.ll
```

---

# Example

```ark
fn[gpu] add_kernel(A: tensor<f32>, B: tensor<f32>, C: tensor<f32>) {
    par i in C {
        C[i] = A[i] + B[i];
    }
}

fn[host] main() -> i32 !IO {
    let size = 1024;

    let A = allocof<f32>(size) @gpu:0;
    let B = allocof<f32>(size) @gpu:0;
    let C = allocof<f32>(size) @gpu:0;

    print "Launching GPU kernel...";
    C <- add_kernel(A, B, C) as t1;
    await t1;

    return 0;
}
```

---

# Troubleshooting

## `MLIRConfig.cmake` not found

Your LLVM install does not include MLIR, or `MLIR_DIR` points at the wrong place.

Required files:

```text
.../lib/cmake/llvm/LLVMConfig.cmake
.../lib/cmake/mlir/MLIRConfig.cmake
```

---

## `unofficial-sodium` or libsodium not found on Windows

You either did not install libsodium for the active vcpkg triplet, or you configured without the vcpkg toolchain file.

Use:

```powershell
-DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake
```

and install:

```powershell
.\vcpkg.exe install libsodium:x64-windows
```

---

## LLVM build fails because `llvm-tblgen.exe` or `mlir-tblgen.exe` is blocked

Your Windows application-control policy is blocking LLVM’s generated build tools.

Either:
- allow those executables in policy
- switch the device to audit mode temporarily
- or build LLVM on a machine without that policy and copy the install tree

---

## `LLVMOrcJIT.lib`, `LLVMAMDGPUCodeGen.lib`, or MLIR conversion libs are missing

Your LLVM/MLIR build is incomplete for the features Ark is trying to use.

Rebuild LLVM + MLIR with:

```text
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU"
-DLLVM_ENABLE_PROJECTS=mlir
```

and point Ark at the **install tree**, not the raw build tree.

---

## Windows selects the wrong Clang toolchain

If Windows falls back to vcpkg’s bundled Clang, runtime compilation can fail against the MSVC STL.

Always point Ark at the LLVM install tree produced above. The configured tools should resolve from:

```text
C:\dev\llvm-install-release\bin
```

---

## `Runtime not found at: tools/compiler/Runtime`

The compiler is being run from a layout where it cannot resolve the runtime tree.

Run from the repo root, or set:

```text
ARK_RUNTIME_DIR=<path-to-tools/compiler/Runtime>
```

---

## Services fail to configure

The default source-build instructions disable services on purpose:

```text
-DARK_BUILD_SERVICES=OFF
```

---

# Development notes

- `arknet` is the main entry point
- `arkc` is the low-level pipeline driver
- `ark-stub` is the self-executing capsule stub
- the runtime headers are copied into `tools/compiler/Runtime/include`
- Windows builds should use the LLVM **install tree**
- once `llvm-build-release` exists, reuse it instead of deleting it between attempts

---

# Roadmap

- frontend and IR definition
- middle-end ownership and lowering passes
- LLVM lowering and backend integration
- runtime stabilization across host and GPU targets
- packaging and capsule workflows

---

## License

See the repository license files.
