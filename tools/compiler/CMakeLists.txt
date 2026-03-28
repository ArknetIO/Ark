cmake_minimum_required(VERSION 3.20)
# Inherit the root project() from the top-level CMakeLists.txt.

include(GNUInstallDirs)
include(FetchContent)

# =========================================================
# 1. Fetch CLI11 & tomlplusplus
# =========================================================
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG        v2.4.1
)

FetchContent_Declare(
  tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(CLI11)
FetchContent_MakeAvailable(tomlplusplus)

# =========================================================
# 2. LLVM / MLIR feature detection
# =========================================================
#
# Release must only compile and link features whose artifacts actually exist.
# This file therefore computes concrete feature booleans and uses them to
# control:
#   - source selection
#   - link libraries
#   - compile definitions consumed by source code
#

function(ark_has_imported_artifact_release out_var target_name)
  set(_ok FALSE)

  if(TARGET ${target_name})
    foreach(_prop
      IMPORTED_IMPLIB_RELEASE
      IMPORTED_LOCATION_RELEASE
      IMPORTED_IMPLIB
      IMPORTED_LOCATION
    )
      get_target_property(_path ${target_name} ${_prop})
      if(_path AND EXISTS "${_path}")
        set(_ok TRUE)
        break()
      endif()
    endforeach()
  endif()

  set(${out_var} ${_ok} PARENT_SCOPE)
endfunction()

function(ark_append_target_if_release list_var target_name)
  if(TARGET ${target_name})
    ark_has_imported_artifact_release(_ark_has_release ${target_name})
    if(_ark_has_release)
      list(APPEND ${list_var} ${target_name})
      set(${list_var} "${${list_var}}" PARENT_SCOPE)
    else()
      message(STATUS "[Ark] Release artifact missing: skipping ${target_name}")
    endif()
  else()
    message(STATUS "[Ark] Package target missing: skipping ${target_name}")
  endif()
endfunction()

function(ark_llvm_release_lib_exists out_var lib_basename)
  set(_ok FALSE)
  set(_dirs "")

  if(DEFINED LLVM_DIR AND NOT LLVM_DIR STREQUAL "")
    get_filename_component(_llvm_cmake_dir "${LLVM_DIR}" ABSOLUTE)
    get_filename_component(_llvm_root_lib_dir "${_llvm_cmake_dir}/../.." ABSOLUTE)
    list(APPEND _dirs
      "${_llvm_root_lib_dir}"
      "${_llvm_root_lib_dir}/Release/lib"
      "${_llvm_root_lib_dir}/../Release/lib"
      "${_llvm_root_lib_dir}/../../Release/lib"
    )
  endif()

  if(DEFINED LLVM_LIBRARY_DIRS)
    list(APPEND _dirs ${LLVM_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES _dirs)

  foreach(_dir IN LISTS _dirs)
    if(EXISTS "${_dir}/${lib_basename}${CMAKE_STATIC_LIBRARY_SUFFIX}" OR
       EXISTS "${_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_basename}${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(_ok TRUE)
      break()
    endif()
  endforeach()

  set(${out_var} ${_ok} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------
# 2.a LLVM base surface
# ---------------------------------------------------------
set(LLVM_LINK_COMPONENTS
  Support
  Core
  OrcJIT
  ExecutionEngine
  Target
  native
  nativecodegen
  MC
  AsmParser
  BitWriter
  BitReader
  Linker
  Analysis
  TransformUtils
)

if(ARK_ENABLE_NVPTX)
  list(APPEND LLVM_LINK_COMPONENTS NVPTX)
endif()

if(ARK_ENABLE_AMDGPU)
  list(APPEND LLVM_LINK_COMPONENTS AMDGPU)
endif()

list(REMOVE_DUPLICATES LLVM_LINK_COMPONENTS)
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

# ---------------------------------------------------------
# 2.b Optional LLVM host JIT
# ---------------------------------------------------------
ark_has_imported_artifact_release(ARK_HAVE_LLVM_ORCJIT LLVMOrcJIT)
ark_has_imported_artifact_release(ARK_HAVE_LLVM_EXECUTIONENGINE LLVMExecutionEngine)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_EXECUTIONENGINE MLIRExecutionEngine)

set(ARK_ENABLE_HOST_JIT OFF)
if(ARK_HAVE_LLVM_ORCJIT AND ARK_HAVE_LLVM_EXECUTIONENGINE AND ARK_HAVE_MLIR_EXECUTIONENGINE)
  set(ARK_ENABLE_HOST_JIT ON)
  list(APPEND LLVM_LINK_COMPONENTS
    OrcJIT
    ExecutionEngine
  )
  message(STATUS "[Ark][JIT] Host JIT enabled")
else()
  message(STATUS "[Ark][JIT] Host JIT disabled")
endif()

# ---------------------------------------------------------
# 2.c Optional LLVM GPU backends
# ---------------------------------------------------------
set(ARK_HAVE_LLVM_NVPTX_BACKEND OFF)
if(ARK_ENABLE_NVPTX)
  list(FIND LLVM_TARGETS_TO_BUILD "NVPTX" _ark_has_nvptx_metadata)
  if(_ark_has_nvptx_metadata GREATER -1)
    ark_llvm_release_lib_exists(_ark_has_nvptx_codegen LLVMNVPTXCodeGen)
    if(_ark_has_nvptx_codegen)
      set(ARK_HAVE_LLVM_NVPTX_BACKEND ON)
      list(APPEND LLVM_LINK_COMPONENTS NVPTX)
      message(STATUS "[Ark][GPU] NVPTX backend enabled")
    else()
      message(STATUS "[Ark][GPU] NVPTX backend disabled: missing Release LLVMNVPTXCodeGen")
    endif()
  else()
    message(STATUS "[Ark][GPU] NVPTX backend disabled: not present in LLVM_TARGETS_TO_BUILD")
  endif()
endif()

set(ARK_HAVE_LLVM_AMDGPU_BACKEND OFF)
if(ARK_ENABLE_AMDGPU)
  list(FIND LLVM_TARGETS_TO_BUILD "AMDGPU" _ark_has_amdgpu_metadata)
  if(_ark_has_amdgpu_metadata GREATER -1)
    ark_llvm_release_lib_exists(_ark_has_amdgpu_codegen LLVMAMDGPUCodeGen)
    if(_ark_has_amdgpu_codegen)
      set(ARK_HAVE_LLVM_AMDGPU_BACKEND ON)
      list(APPEND LLVM_LINK_COMPONENTS AMDGPU)
      message(STATUS "[Ark][GPU] AMDGPU backend enabled")
    else()
      message(STATUS "[Ark][GPU] AMDGPU backend disabled: missing Release LLVMAMDGPUCodeGen")
    endif()
  else()
    message(STATUS "[Ark][GPU] AMDGPU backend disabled: not present in LLVM_TARGETS_TO_BUILD")
  endif()
endif()

list(REMOVE_DUPLICATES LLVM_LINK_COMPONENTS)
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

# ---------------------------------------------------------
# 2.d Optional MLIR conversion / GPU surface
# ---------------------------------------------------------
ark_has_imported_artifact_release(ARK_HAVE_MLIR_VECTOR_TO_LLVM               MLIRVectorToLLVM)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_GPU_TRANSFORMS               MLIRGPUTransforms)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_GPU_PIPELINES                MLIRGPUPipelines)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_GPU_TO_NVVM_TRANSFORMS       MLIRGPUToNVVMTransforms)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_GPU_TO_ROCDL_TRANSFORMS      MLIRGPUToROCDLTransforms)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_GPU_TO_LLVMIR_TRANSLATION    MLIRGPUToLLVMIRTranslation)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_NVVM_TO_LLVMIR_TRANSLATION   MLIRNVVMToLLVMIRTranslation)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_ROCDL_TO_LLVMIR_TRANSLATION  MLIRROCDLToLLVMIRTranslation)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_NVVMDIALECT                  MLIRNVVMDialect)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_ROCDLDIALECT                 MLIRROCDLDialect)
ark_has_imported_artifact_release(ARK_HAVE_MLIR_NVVM_TO_LLVM                 MLIRNVVMToLLVM)

set(ARK_ENABLE_NVPTX_GPU_PIPELINE OFF)
if(ARK_HAVE_LLVM_NVPTX_BACKEND AND
   ARK_HAVE_MLIR_VECTOR_TO_LLVM AND
   ARK_HAVE_MLIR_GPU_TRANSFORMS AND
   ARK_HAVE_MLIR_GPU_PIPELINES AND
   ARK_HAVE_MLIR_GPU_TO_NVVM_TRANSFORMS AND
   ARK_HAVE_MLIR_GPU_TO_LLVMIR_TRANSLATION AND
   ARK_HAVE_MLIR_NVVM_TO_LLVMIR_TRANSLATION AND
   ARK_HAVE_MLIR_NVVMDIALECT AND
   ARK_HAVE_MLIR_NVVM_TO_LLVM)
  set(ARK_ENABLE_NVPTX_GPU_PIPELINE ON)
endif()

set(ARK_ENABLE_AMDGPU_GPU_PIPELINE OFF)
if(ARK_HAVE_LLVM_AMDGPU_BACKEND AND
   ARK_HAVE_MLIR_VECTOR_TO_LLVM AND
   ARK_HAVE_MLIR_GPU_TRANSFORMS AND
   ARK_HAVE_MLIR_GPU_PIPELINES AND
   ARK_HAVE_MLIR_GPU_TO_ROCDL_TRANSFORMS AND
   ARK_HAVE_MLIR_GPU_TO_LLVMIR_TRANSLATION AND
   ARK_HAVE_MLIR_ROCDL_TO_LLVMIR_TRANSLATION AND
   ARK_HAVE_MLIR_ROCDLDIALECT)
  set(ARK_ENABLE_AMDGPU_GPU_PIPELINE ON)
endif()

set(ARK_ENABLE_GPU_LOWERING OFF)
if(ARK_ENABLE_NVPTX_GPU_PIPELINE OR ARK_ENABLE_AMDGPU_GPU_PIPELINE)
  set(ARK_ENABLE_GPU_LOWERING ON)
endif()

if(ARK_ENABLE_GPU_LOWERING)
  message(STATUS "[Ark][GPU] GPU lowering enabled")
else()
  message(STATUS "[Ark][GPU] GPU lowering disabled")
endif()

# =========================================================
# 3. Runtime (Core + Platform Split)
# =========================================================
set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")

set(RUNTIME_COMMON_SOURCES
  "${RUNTIME_DIR}/core/async.cpp"
  "${RUNTIME_DIR}/core/memory.cpp"
  "${RUNTIME_DIR}/core/string.cpp"
  "${RUNTIME_DIR}/core/print.cpp"
  "${RUNTIME_DIR}/core/panic.cpp"
  "${RUNTIME_DIR}/core/hash.cpp"
  "${RUNTIME_DIR}/core/vector.cpp"
  "${RUNTIME_DIR}/net/remote.cpp"
  "${RUNTIME_DIR}/core/gpu_hal.cpp"
)

set(RUNTIME_PLATFORM_SOURCES)
set(ARK_PLATFORM_LINK_LIBS)
set(ARK_PLATFORM_DL_LIBS)

if(WIN32)
  message(STATUS "[Ark][Runtime] Building Windows runtime sources")
  list(APPEND RUNTIME_PLATFORM_SOURCES
    "${RUNTIME_DIR}/fs/fs_win.cpp"
    "${RUNTIME_DIR}/sys/sys_win.cpp"
    "${RUNTIME_DIR}/net/socket_win.cpp"
  )
  list(APPEND ARK_PLATFORM_LINK_LIBS
    ws2_32
  )
else()
  message(STATUS "[Ark][Runtime] Building POSIX runtime sources")
  list(APPEND RUNTIME_PLATFORM_SOURCES
    "${RUNTIME_DIR}/fs/fs_posix.cpp"
    "${RUNTIME_DIR}/sys/sys_posix.cpp"
    "${RUNTIME_DIR}/net/socket_posix.cpp"
  )

  if(UNIX AND NOT APPLE)
    set(ARK_PLATFORM_DL_LIBS ${CMAKE_DL_LIBS})
  endif()
endif()

set(RUNTIME_CORE_SOURCES
  ${RUNTIME_COMMON_SOURCES}
  ${RUNTIME_PLATFORM_SOURCES}
)

# =========================================================
# 4. Config Header
# =========================================================
configure_file(
  "${RUNTIME_DIR}/include/ark_config.h.in"
  "${CMAKE_BINARY_DIR}/include/ark_config.h"
)

# =========================================================
# 5. GenMIR Reorganization
# =========================================================
set(ARK_GENMIR_SOURCES
  Frontend/GenMIR/GenMIR.Utils.cpp
  Frontend/GenMIR.cpp
)

# =========================================================
# 6. Shared Compiler Sources
# =========================================================
set(ARK_COMPILER_SHARED_SOURCES
  hud.cpp

  # Modular Pipeline
  Pipeline/Compiler.cpp
  Pipeline/Linker.cpp
  Pipeline/LinkerGpuFatbin.cpp
  Pipeline/JIT.cpp
  Integration/CapsuleBackend.cpp

  # Frontend
  Frontend/Lexer.cpp
  Frontend/Parser.cpp
  ${ARK_GENMIR_SOURCES}
  Frontend/MirBuilder.cpp
  Frontend/BuiltinNamespaces.cpp
  Frontend/Intrinsics.cpp

  # Middle-End
  Analysis/OwnershipVerifier.cpp
  Analysis/OwnershipUtils.cpp
  Transforms/DropInsertion.cpp

  # GPU / ABI passes
  Transforms/GpuLowering.cpp
  Transforms/GpuKernelOutlining.cpp
  Transforms/RuntimeABIVerifier.cpp

  # Runtime Manager
  Frontend/RuntimeManager.cpp
  Frontend/ModuleRegistry.cpp
  SymbolRegistry.cpp

  # Runtime Source Injection
  ${RUNTIME_CORE_SOURCES}
)

# Host JIT must always provide a TU for JIT::Run. Use a disabled TU when the
# real host-JIT implementation is unavailable.
if(ARK_ENABLE_HOST_JIT)
  list(APPEND ARK_COMPILER_SHARED_SOURCES
    Pipeline/JIT.cpp
  )
else()
  list(APPEND ARK_COMPILER_SHARED_SOURCES
    Pipeline/JIT.Disabled.cpp
  )
endif()

# GPU lowering sources are only compiled when the full Release GPU conversion
# surface exists.
if(ARK_ENABLE_GPU_LOWERING)
  list(APPEND ARK_COMPILER_SHARED_SOURCES
    Pipeline/LinkerGpuFatbin.cpp
    Transforms/GpuLowering.cpp
    Transforms/GpuKernelOutlining.cpp
    Transforms/RuntimeABIVerifier.cpp
  )
endif()

# =========================================================
# 7. Build Targets (arkc & arknet)
# =========================================================
add_ark_tool(arkc
  main.cpp
  ${ARK_COMPILER_SHARED_SOURCES}
)

add_ark_tool(arknet
  arknet.cpp
  CLI/Init.cpp
  CLI/Compile.cpp
  CLI/Registry.cpp
  CLI/Lsp.cpp
  CLI/Provider.cpp
  CLI/Config.cpp
  CLI/Doctor.cpp
  CLI/Workspace.cpp
  ${ARK_COMPILER_SHARED_SOURCES}
)

# =========================================================
# 8. Common Target Configuration
# =========================================================
function(ark_fix_msvc_cpp_mode target_name)
  if(NOT MSVC)
    return()
  endif()

  get_target_property(_opts ${target_name} COMPILE_OPTIONS)
  if(NOT _opts)
    set(_opts "")
  endif()

  set(_clean_opts ${_opts})
  list(REMOVE_ITEM _clean_opts
    /EHs
    /EHsc
    /EHc
    /EHs-
    /EHc-
    /GR
    /GR-
  )

  get_target_property(_flags ${target_name} COMPILE_FLAGS)
  if(NOT _flags)
    set(_flags "")
  endif()

  foreach(_flag /EHs /EHsc /EHc /EHs- /EHc- /GR /GR-)
    string(REPLACE "${_flag}" "" _flags "${_flags}")
  endforeach()

  string(REGEX REPLACE "[ \t]+" " " _flags "${_flags}")
  string(STRIP "${_flags}" _flags)

  set_target_properties(${target_name} PROPERTIES
    COMPILE_OPTIONS "${_clean_opts}"
    COMPILE_FLAGS   "${_flags}"
  )

  target_compile_options(${target_name} PRIVATE /EHsc /GR)
endfunction()

# ---------------------------------------------------------
# 8.a Common MLIR link surface
# ---------------------------------------------------------
set(ARK_COMMON_MLIR_LIBS)

foreach(_mlir_target
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRUBToLLVM
  MLIRComplexToLLVM
  MLIRFuncDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRTensorDialect
)
  ark_append_target_if_release(ARK_COMMON_MLIR_LIBS ${_mlir_target})
endforeach()

# GPU MLIR surface is appended only if GPU lowering is actually enabled.
set(ARK_GPU_MLIR_LIBS)
if(ARK_ENABLE_GPU_LOWERING)
  foreach(_mlir_target
    MLIRVectorToLLVM
    MLIRGPUDialect
    MLIRNVVMDialect
    MLIRROCDLDialect
    MLIRNVVMToLLVM
    MLIRGPUTransforms
    MLIRGPUPipelines
    MLIRGPUToLLVMIRTranslation
    MLIRFuncToLLVM
    MLIRControlFlowToLLVM
    MLIRArithToLLVM
    MLIRLLVMCommonConversion
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRMemRefToLLVM
    MLIRIndexToLLVM
  )
    ark_append_target_if_release(ARK_GPU_MLIR_LIBS ${_mlir_target})
  endforeach()

  if(ARK_ENABLE_NVPTX_GPU_PIPELINE)
    foreach(_mlir_target
      MLIRGPUToNVVMTransforms
      MLIRNVVMToLLVMIRTranslation
    )
      ark_append_target_if_release(ARK_GPU_MLIR_LIBS ${_mlir_target})
    endforeach()
  endif()

  if(ARK_ENABLE_AMDGPU_GPU_PIPELINE)
    foreach(_mlir_target
      MLIRGPUToROCDLTransforms
      MLIRROCDLToLLVMIRTranslation
    )
      ark_append_target_if_release(ARK_GPU_MLIR_LIBS ${_mlir_target})
    endforeach()
  endif()
endif()

# Host-JIT MLIR surface.
set(ARK_HOST_JIT_MLIR_LIBS)
if(ARK_ENABLE_HOST_JIT)
  ark_append_target_if_release(ARK_HOST_JIT_MLIR_LIBS MLIRExecutionEngine)
endif()

foreach(TOOL_TARGET arkc arknet)

  add_dependencies(${TOOL_TARGET}
    ArkMirDialectIncGen
    ArkMirOpsIncGen
    ArkMirTypesIncGen
    PrepareRuntimeHeaders
  )

  target_include_directories(${TOOL_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/ark-ir/include
    ${CMAKE_SOURCE_DIR}/libs/ark-abi/include
    ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include
    ${CMAKE_SOURCE_DIR}/libs/ark-crypto/include
    ${CMAKE_BINARY_DIR}/libs/ark-ir/IR
    ${CMAKE_BINARY_DIR}/include
    ${RUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${TOOL_TARGET} PRIVATE
    ark-capsule
    ark-diagnostics
    ark-abi
    ark-wire
    Ark::Crypto

    ArkMirDialect
    ArkToLLVM

    ${llvm_libs}

    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
    MLIRROCDLToLLVMIRTranslation

    MLIRCallInterfaces
    MLIRAnalysis
    MLIRDataLayoutInterfaces

    MLIRFuncDialect
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRLLVMDialect
    MLIRVectorDialect
    MLIRTensorDialect
    MLIRGPUDialect
    MLIRNVVMDialect
    MLIRROCDLDialect
    MLIRIndexDialect

    MLIRFuncToLLVM
    MLIRControlFlowToLLVM
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRIndexToLLVM
    MLIRVectorToLLVM
    MLIRLLVMCommonConversion
    MLIRNVVMToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRUBToLLVM
    MLIRComplexToLLVM

    MLIRGPUTransforms
    MLIRGPUPipelines
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms

    ${ARK_PLATFORM_LINK_LIBS}
    ${ARK_PLATFORM_DL_LIBS}
  )

  if(ARK_ENABLE_HOST_JIT)
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_HOST_JIT=1)
  else()
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_HOST_JIT=0)
  endif()

  if(ARK_ENABLE_GPU_LOWERING)
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_GPU_LOWERING=1)
  else()
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_GPU_LOWERING=0)
  endif()

  if(ARK_ENABLE_NVPTX_GPU_PIPELINE)
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_NVPTX_GPU_PIPELINE=1)
  else()
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_NVPTX_GPU_PIPELINE=0)
  endif()

  if(ARK_ENABLE_AMDGPU_GPU_PIPELINE)
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_AMDGPU_GPU_PIPELINE=1)
  else()
    target_compile_definitions(${TOOL_TARGET} PRIVATE ARK_ENABLE_AMDGPU_GPU_PIPELINE=0)
  endif()

  if(UNIX AND NOT APPLE)
    target_link_options(${TOOL_TARGET} PRIVATE "-rdynamic")
  elseif(APPLE)
    target_link_options(${TOOL_TARGET} PRIVATE "-Wl,-export_dynamic")
  endif()

  set_target_properties(${TOOL_TARGET} PROPERTIES ENABLE_EXPORTS 1)

  ark_fix_msvc_cpp_mode(${TOOL_TARGET})

endforeach()

target_link_libraries(arknet PRIVATE CLI11::CLI11 tomlplusplus::tomlplusplus)

# =========================================================
# 9. Build the Capsule Stub (ark-stub)
# =========================================================
add_executable(ark-stub Runner/Stub.cpp)
target_compile_definitions(ark-stub PRIVATE ARK_STUB_STANDALONE)

target_include_directories(ark-stub PRIVATE
  ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include
  ${CMAKE_SOURCE_DIR}/libs/ark-crypto/include
)

target_link_libraries(ark-stub PRIVATE
  ark-capsule
  Ark::Crypto
)

if(MSVC)
  target_compile_options(ark-stub PRIVATE
    $<$<CONFIG:Release>:/O1>
    $<$<CONFIG:RelWithDebInfo>:/O1>
    $<$<CONFIG:MinSizeRel>:/O1>
  )
else()
  target_compile_options(ark-stub PRIVATE
    $<$<CONFIG:Release>:-Os>
    $<$<CONFIG:RelWithDebInfo>:-Os>
    $<$<CONFIG:MinSizeRel>:-Os>
  )

  if(UNIX AND NOT APPLE)
    target_link_options(ark-stub PRIVATE
      $<$<CONFIG:Release>:-Wl,--strip-all>
      $<$<CONFIG:RelWithDebInfo>:-Wl,--strip-all>
      $<$<CONFIG:MinSizeRel>:-Wl,--strip-all>
    )
  endif()
endif()

# =========================================================
# 10. Installation
# =========================================================
install(TARGETS arkc arknet ark-stub DESTINATION bin)

# =========================================================
# 11. Runtime Headers Preparation
# =========================================================
add_custom_target(PrepareRuntimeHeaders ALL
  COMMENT "Copying Protocol & ABI headers to Runtime/include..."
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/libs/ark-abi/include/
    ${RUNTIME_DIR}/include/
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/libs/ark-wire/include/
    ${RUNTIME_DIR}/include/
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include/
    ${RUNTIME_DIR}/include/
)

# =========================================================
# 12. GPU Plugins (SHARED)
# =========================================================
add_subdirectory(Runtime/gpu)