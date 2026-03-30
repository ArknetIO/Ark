# =========================================================
# Ark IR Dialect Definition
# =========================================================
# This file must work both:
#   1. inside the Ark monorepo build
#   2. in a standalone ark-ir build
#
# Use current-list-relative paths so this subtree does not depend on the
# top-level source layout being the active CMake source dir.

# =========================================================
# 1. Setup Source Paths
# =========================================================
# Current directory:
#   libs/ark-ir/IR
# Public headers:
#   libs/ark-ir/include
set(ARK_IR_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(ARK_IR_INCLUDE_DIR "${ARK_IR_ROOT_DIR}/include")
set(ARK_IR_TD_DIR "${ARK_IR_INCLUDE_DIR}/ark/IR")

# Allow this directory to see its own source headers
include_directories(${ARK_IR_INCLUDE_DIR})

# Allow this directory to see its own generated .inc files
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# =========================================================
# 2. TableGen: Generate .inc Files
# =========================================================
# Dialect declarations / definitions
set(LLVM_TARGET_DEFINITIONS "${ARK_IR_TD_DIR}/ArkMirDialect.td")
mlir_tablegen(ArkMirDialect.h.inc   -gen-dialect-decls)
mlir_tablegen(ArkMirDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ArkMirDialectIncGen)

# Type declarations / definitions
set(LLVM_TARGET_DEFINITIONS "${ARK_IR_TD_DIR}/ArkMirTypes.td")
mlir_tablegen(ArkMirTypes.h.inc   -gen-typedef-decls)
mlir_tablegen(ArkMirTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(ArkMirTypesIncGen)

# Operation declarations / definitions
set(LLVM_TARGET_DEFINITIONS "${ARK_IR_TD_DIR}/ArkMirOps.td")
mlir_tablegen(ArkMirOps.h.inc   -gen-op-decls)
mlir_tablegen(ArkMirOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(ArkMirOpsIncGen)

# =========================================================
# 3. Build the Library
# =========================================================
add_mlir_library(ArkMirDialect
  ArkMirDialect.cpp
  ArkMirOps.cpp

  DEPENDS
    ArkMirDialectIncGen
    ArkMirTypesIncGen
    ArkMirOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRBytecodeOpInterface
)

# =========================================================
# 4. Export Include Paths
# =========================================================
# This makes the paths available to downstream users such as arkc / arknet.
target_include_directories(ArkMirDialect PUBLIC
  # A. Source headers (.h / .td-adjacent includes)
  $<BUILD_INTERFACE:${ARK_IR_INCLUDE_DIR}>

  # B. Generated headers (.inc)
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# =========================================================
# 5. Keep generated files visible to IDEs / downstream targets
# =========================================================
set_target_properties(ArkMirDialect PROPERTIES
  FOLDER "ark-ir"
)