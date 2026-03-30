# =========================================================
# Ark IR -> LLVM Conversion Library
# =========================================================
# This file must work both:
#   1. inside the Ark monorepo build
#   2. in a standalone ark-ir build
#
# Keep paths relative to this subtree instead of relying on CMAKE_SOURCE_DIR.

# =========================================================
# 1. Setup Source / Generated Include Paths
# =========================================================
# Current directory:
#   libs/ark-ir/Conversion
# Public headers:
#   libs/ark-ir/include
# Generated IR .inc headers:
#   <build>/libs/ark-ir/IR
set(ARK_IR_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(ARK_IR_INCLUDE_DIR "${ARK_IR_ROOT_DIR}/include")

get_filename_component(
  ARK_IR_BINARY_INCLUDE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/../IR"
  ABSOLUTE
)

# =========================================================
# 2. Build the Conversion Library
# =========================================================
# NOTE:
# - DEPENDS is for generated/tablegen targets, not for linked libraries.
# - ArkMirDialect must be linked, not listed under DEPENDS.
add_mlir_library(ArkToLLVM
  ArkToLLVM/ArkToLLVM.cpp

  LINK_LIBS PUBLIC
    ArkMirDialect
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRFuncToLLVM
    MLIRControlFlowToLLVM
    MLIRArithToLLVM
    MLIRPass
    MLIRTransforms
)

# =========================================================
# 3. Export Include Paths
# =========================================================
# A. Source headers (.h)
# B. Generated IR headers (.inc) from the sibling IR build dir
target_include_directories(ArkToLLVM PUBLIC
  $<BUILD_INTERFACE:${ARK_IR_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${ARK_IR_BINARY_INCLUDE_DIR}>
)

# =========================================================
# 4. IDE Organization
# =========================================================
set_target_properties(ArkToLLVM PROPERTIES
  FOLDER "ark-ir"
)