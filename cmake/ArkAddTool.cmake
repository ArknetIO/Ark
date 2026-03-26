# arklang/cmake/ArkAddTool.cmake
function(add_ark_tool name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/arkc
  )
  llvm_update_compile_flags(${name})
endfunction()
