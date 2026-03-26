# =========================================================
# Ark IR Dialect Definition
# =========================================================

# 1. Setup Source Paths
# This points to: ~/ark/libs/ark-ir/include
set(ARK_IR_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/ark-ir/include")

# Allow this directory to see its own source headers
include_directories(${ARK_IR_INCLUDE_DIR})
# Allow this directory to see its own generated .inc files
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# =========================================================
# 2. TableGen: Generate .inc Files
# =========================================================
# The .td files live in ../include/ark/IR/
set(LLVM_TARGET_DEFINITIONS ${ARK_IR_INCLUDE_DIR}/ark/IR/ArkMirDialect.td)
mlir_tablegen(ArkMirDialect.h.inc     -gen-dialect-decls)
mlir_tablegen(ArkMirDialect.cpp.inc   -gen-dialect-defs)
add_public_tablegen_target(ArkMirDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${ARK_IR_INCLUDE_DIR}/ark/IR/ArkMirTypes.td)
mlir_tablegen(ArkMirTypes.h.inc    -gen-typedef-decls)
mlir_tablegen(ArkMirTypes.cpp.inc  -gen-typedef-defs)
add_public_tablegen_target(ArkMirTypesIncGen)

set(LLVM_TARGET_DEFINITIONS ${ARK_IR_INCLUDE_DIR}/ark/IR/ArkMirOps.td)
mlir_tablegen(ArkMirOps.h.inc        -gen-op-decls)
mlir_tablegen(ArkMirOps.cpp.inc      -gen-op-defs)
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

  LINK_LIBS
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRBytecodeOpInterface
)

# =========================================================
# 4. [CRITICAL] Export Include Paths
# =========================================================
# This makes the paths available to 'arkc' automatically.
target_include_directories(ArkMirDialect PUBLIC
  # A. The Source Headers (.h)
  ${ARK_IR_INCLUDE_DIR}
  
  # B. The Generated Headers (.inc)
  # This tells arkc: "Look in build/libs/ark-ir/IR/ for includes"
  ${CMAKE_CURRENT_BINARY_DIR}
)