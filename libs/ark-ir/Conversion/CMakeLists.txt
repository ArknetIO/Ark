# lib/ark/Conversion/CMakeLists.txt

add_mlir_library(ArkToLLVM
  ArkToLLVM/ArkToLLVM.cpp 

  DEPENDS
  ArkMirDialect

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

target_include_directories(ArkToLLVM PRIVATE
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_SOURCE_DIR}/include
)