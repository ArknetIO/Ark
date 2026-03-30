# arklang/cmake/ArkAddTool.cmake
function(add_ark_tool name)
  set(_ark_root "${PROJECT_SOURCE_DIR}")

  add_executable(${name} ${ARGN})

  target_include_directories(${name} PRIVATE
    "${_ark_root}/include"
    "${_ark_root}/tools/compiler/include"
  )

  llvm_update_compile_flags(${name})
endfunction()