cmake_minimum_required(VERSION 3.20)

# =============================================================================
# ark-ir can be built:
#   1. As part of the Ark monorepo
#   2. Standalone for dialect/conversion development
# Do not reset the parent project when included from the root build.
# =============================================================================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ark-ir LANGUAGES C CXX)
endif()

# =============================================================================
# 1. Initialize LLVM / MLIR only if not already provided by the parent build
# =============================================================================
if(NOT LLVM_FOUND)
  find_package(LLVM REQUIRED CONFIG)
endif()

if(NOT MLIR_FOUND)
  find_package(MLIR REQUIRED CONFIG)
endif()

# Keep the module path additive and stable.
if(DEFINED MLIR_CMAKE_DIR AND NOT MLIR_CMAKE_DIR STREQUAL "")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
endif()

if(DEFINED LLVM_CMAKE_DIR AND NOT LLVM_CMAKE_DIR STREQUAL "")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

# =============================================================================
# 2. Load MLIR / LLVM CMake helpers
# Only include them if the commands are not already available.
# =============================================================================
if(NOT COMMAND mlir_tablegen)
  include(TableGen)
endif()

if(NOT COMMAND add_llvm_library)
  include(AddLLVM)
endif()

if(NOT COMMAND add_mlir_dialect)
  include(AddMLIR)
endif()

# =============================================================================
# 3. Global include / definitions for this subtree
# Keep this subtree self-contained so IR/ and Conversion/ can use MLIR/LLVM
# headers without each child repeating the same setup.
# =============================================================================
include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

if(LLVM_DEFINITIONS)
  add_definitions(${LLVM_DEFINITIONS})
endif()

# =============================================================================
# 4. Enter ark-ir subdirectories
# IR      -> dialects, ops, types, tablegen products
# Conversion -> lowering / translation libraries
# =============================================================================
add_subdirectory(IR)
add_subdirectory(Conversion)