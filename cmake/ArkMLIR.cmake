# cmake/ArkMLIR.cmake

message(STATUS "🔍 Locating LLVM and MLIR...")

# 1. Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# 2. Append their CMake directories to our module path so we can use their macros
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# 3. Include necessary LLVM/MLIR macros (TableGen, AddLLVM, AddMLIR)
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# 4. Expose the include directories globally to the compiler targets
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

# 5. Apply LLVM definitions (e.g., -D__STDC_LIMIT_MACROS, etc.)
add_definitions(${LLVM_DEFINITIONS})

message(STATUS "✅ Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "✅ Found MLIR ${MLIR_PACKAGE_VERSION} at ${MLIR_DIR}")