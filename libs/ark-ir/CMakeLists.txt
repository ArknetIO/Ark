cmake_minimum_required(VERSION 3.20)
project(ark-ir)

# 1. Initialize MLIR/LLVM Context
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# 2. Enter the Sub-Directories (Your files)
add_subdirectory(IR)
add_subdirectory(Conversion)