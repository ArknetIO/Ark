# ArkLang: The AI-Native Compiler

**Schemas-first systems language with zero-cost reflection and execution domains (`[host]` / `[gpu]`) that let AI and systems code live in one coherent model.**

## Key Features

- **First-class AI primitives**
  - `alloc<tensor[N]f32>` for native tensor allocation
  - `par(i in 0..N)` for parallel map-style loops

- **Asynchronous hazard management**
  - `<-` launches work asynchronously and returns a token
  - `await` makes synchronization explicit
  - the compiler tracks RAW/WAW hazards and injects dependencies automatically

- **MLIR-based backend**
  - lowers Ark source into **Ark IR** (custom MLIR dialect)
  - builds on LLVM’s optimization pipeline

## Architecture

ArkLang is built on top of **LLVM 18/19** and **MLIR**.

1. **Lexer / Parser**  
   Custom C++ frontend that builds the AST.

2. **Ark Ops (TableGen)**  
   Defines the custom IR, including operations such as `ark.launch`, `ark.alloc`, and `ark.map`.

3. **Code Generation**  
   Traverses the AST and emits MLIR SSA form.

4. **Driver**  
   The `arkc` binary orchestrates the pipeline.

---

## Build From Source

These instructions build:

- core libraries
- compiler stack
- `arkc`

These instructions **do not** build backend services. Services require extra dependencies and are intentionally left off in the default fresh-install path.

---

## Version Requirements

- **CMake 3.20+**
- **LLVM + MLIR 18 or 19**
- **C++20-capable compiler**
- **Git**

---

## Windows (Fresh Install)

### 1. Install prerequisites

Install:

- **Visual Studio 2022** with **Desktop development with C++**
- **CMake**
- **Git**

Open:

- **x64 Native Tools PowerShell for VS 2022**

Install CMake and Git with `winget`:

```powershell
winget install Kitware.CMake
winget install Git.Git
````

### 2. Install vcpkg and libsodium

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\dev\vcpkg
cd C:\dev\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install libsodium:x64-windows
```

### 3. Build and install LLVM + MLIR

```powershell
git clone https://github.com/llvm/llvm-project.git C:\dev\llvm-project

cmake -S C:\dev\llvm-project\llvm -B C:\dev\llvm-build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DLLVM_ENABLE_PROJECTS=mlir `
  -DLLVM_TARGETS_TO_BUILD=Native `
  -DLLVM_ENABLE_ASSERTIONS=ON `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX=C:\dev\llvm-install

cmake --build C:\dev\llvm-build --config Release --target INSTALL -- /m
```

Verify:

```powershell
Test-Path C:\dev\llvm-install\lib\cmake\llvm\LLVMConfig.cmake
Test-Path C:\dev\llvm-install\lib\cmake\mlir\MLIRConfig.cmake
```

Both commands must print `True`.

### 4. Clone Ark

```powershell
git clone https://github.com/ArknetIO/Ark.git C:\dev\Ark
cd C:\dev\Ark
```

### 5. Configure Ark

```powershell
Remove-Item -Recurse -Force .\build -ErrorAction Ignore

cmake -S . -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DARK_BUILD_TOOLS=ON `
  -DARK_BUILD_SERVICES=OFF `
  -DLLVM_DIR=C:\dev\llvm-install\lib\cmake\llvm `
  -DMLIR_DIR=C:\dev\llvm-install\lib\cmake\mlir
```

### 6. Build Ark

```powershell
cmake --build build --config Release -- /m
```

### 7. Core libraries only

```powershell
Remove-Item -Recurse -Force .\build -ErrorAction Ignore

cmake -S . -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DARK_BUILD_TOOLS=OFF `
  -DARK_BUILD_SERVICES=OFF

cmake --build build --config Release -- /m
```

---

## Linux (Fresh Install, Ubuntu/Debian)

### 1. Install prerequisites

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

### 2. Build and install LLVM + MLIR

```bash
git clone https://github.com/llvm/llvm-project.git "$HOME/llvm-project"

cmake -S "$HOME/llvm-project/llvm" -B "$HOME/llvm-build" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/llvm-install" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON

cmake --build "$HOME/llvm-build" --parallel
cmake --install "$HOME/llvm-build"
```

Verify:

```bash
test -f "$HOME/llvm-install/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$HOME/llvm-install/lib/cmake/mlir/MLIRConfig.cmake"
```

### 3. Clone Ark

```bash
git clone https://github.com/ArknetIO/Ark.git "$HOME/Ark"
cd "$HOME/Ark"
```

### 4. Configure Ark

```bash
rm -rf build

cmake -S . -B build -G Ninja \
  -DARK_BUILD_TOOLS=ON \
  -DARK_BUILD_SERVICES=OFF \
  -DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm" \
  -DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
```

### 5. Build Ark

```bash
cmake --build build --parallel
```

### 6. Core libraries only

```bash
rm -rf build

cmake -S . -B build -G Ninja \
  -DARK_BUILD_TOOLS=OFF \
  -DARK_BUILD_SERVICES=OFF

cmake --build build --parallel
```

---

## macOS (Fresh Install)

### 1. Install prerequisites

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Install Homebrew if needed, then install dependencies:

```bash
brew install cmake ninja git pkg-config libsodium
```

### 2. Build and install LLVM + MLIR

```bash
git clone https://github.com/llvm/llvm-project.git "$HOME/llvm-project"

cmake -S "$HOME/llvm-project/llvm" -B "$HOME/llvm-build" -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/llvm-install"

cmake --build "$HOME/llvm-build" --parallel
cmake --install "$HOME/llvm-build"
```

Verify:

```bash
test -f "$HOME/llvm-install/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$HOME/llvm-install/lib/cmake/mlir/MLIRConfig.cmake"
```

### 3. Clone Ark

```bash
git clone https://github.com/ArknetIO/Ark.git "$HOME/Ark"
cd "$HOME/Ark"
```

### 4. Configure Ark

```bash
rm -rf build

cmake -S . -B build -G Ninja \
  -DARK_BUILD_TOOLS=ON \
  -DARK_BUILD_SERVICES=OFF \
  -DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm" \
  -DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
```

### 5. Build Ark

```bash
cmake --build build --parallel
```

### 6. Core libraries only

```bash
rm -rf build

cmake -S . -B build -G Ninja \
  -DARK_BUILD_TOOLS=OFF \
  -DARK_BUILD_SERVICES=OFF

cmake --build build --parallel
```

---

## Usage

Build output paths depend on generator and platform, but the compiler binary is `arkc`.

Typical usage:

```bash
# print MLIR to stdout
arkc tests/monolith.ark -o -

# write MLIR to a file
arkc input.ark -o output.mlir
```

---

## Example

```ark
fn[gpu] add_kernel(A: f32, B: f32) -> f32 {
    return A + B;
}

fn[host] main() -> i32 {
    let A = alloc<f32[1024]> @gpu:0;
    let B = alloc<f32[1024]> @gpu:0;
    let C = alloc<f32[1024]> @gpu:0;

    C <- add_kernel(A, B) as t1;

    await t1;

    let res = par(i in 0..1024) {
        i
    };

    return 0;
}
```

---

## Common Failures

### `MLIRConfig.cmake` not found

Your LLVM install does not include MLIR, or `MLIR_DIR` is wrong.

Required files:

```text
.../lib/cmake/llvm/LLVMConfig.cmake
.../lib/cmake/mlir/MLIRConfig.cmake
```

### `unofficial-sodium` not found on Windows

You configured without the vcpkg toolchain file or did not install `libsodium:x64-windows`.

Use:

```powershell
-DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake
```

### `cmake` is not recognized

Install CMake, reopen your shell, and try again.

### Services fail to configure

The default source-build instructions disable services on purpose:

```text
-DARK_BUILD_SERVICES=OFF
```

---

## Roadmap

* **Phase 1:** Frontend and IR definition
* **Phase 2:** Middle-end optimization passes
* **Phase 3:** Lowering to LLVM IR
* **Phase 4:** Runtime library and backend integration

[1]: https://mlir.llvm.org/getting_started/?utm_source=chatgpt.com "Getting Started - MLIR"
