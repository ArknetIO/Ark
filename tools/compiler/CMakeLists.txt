include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
cmake_minimum_required(VERSION 3.20)

include(GNUInstallDirs)
include(FetchContent)

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

function(ark_has_imported_artifact out_var target_name)
  set(_ok FALSE)

  if(TARGET ${target_name})
    foreach(_prop
      IMPORTED_IMPLIB_RELEASE
      IMPORTED_LOCATION_RELEASE
      IMPORTED_IMPLIB_RELWITHDEBINFO
      IMPORTED_LOCATION_RELWITHDEBINFO
      IMPORTED_IMPLIB_MINSIZEREL
      IMPORTED_LOCATION_MINSIZEREL
      IMPORTED_IMPLIB_NOCONFIG
      IMPORTED_LOCATION_NOCONFIG
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

function(ark_require_imported_target list_var target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "[Ark] Required package target missing: ${target_name}")
  endif()

  ark_has_imported_artifact(_ark_has_artifact ${target_name})
  if(NOT _ark_has_artifact)
    message(FATAL_ERROR "[Ark] Required imported target has no usable artifact: ${target_name}")
  endif()

  list(APPEND ${list_var} ${target_name})
  set(${list_var} "${${list_var}}" PARENT_SCOPE)
endfunction()

function(ark_append_imported_target_if_available list_var target_name)
  if(NOT TARGET ${target_name})
    message(STATUS "[Ark] Optional package target missing: ${target_name}")
    set(${list_var} "${${list_var}}" PARENT_SCOPE)
    return()
  endif()

  ark_has_imported_artifact(_ark_has_artifact ${target_name})
  if(NOT _ark_has_artifact)
    message(STATUS "[Ark] Optional imported target has no usable artifact: ${target_name}")
    set(${list_var} "${${list_var}}" PARENT_SCOPE)
    return()
  endif()

  list(APPEND ${list_var} ${target_name})
  set(${list_var} "${${list_var}}" PARENT_SCOPE)
endfunction()

function(ark_llvm_lib_exists out_var lib_basename)
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

set(LLVM_LINK_COMPONENTS
  Support
  Core
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

ark_has_imported_artifact(ARK_HAVE_LLVM_ORCJIT LLVMOrcJIT)
ark_has_imported_artifact(ARK_HAVE_LLVM_EXECUTIONENGINE LLVMExecutionEngine)
ark_has_imported_artifact(ARK_HAVE_MLIR_EXECUTIONENGINE MLIRExecutionEngine)

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

set(ARK_HAVE_LLVM_NVPTX_BACKEND OFF)
if(ARK_ENABLE_NVPTX)
  list(FIND LLVM_TARGETS_TO_BUILD "NVPTX" _ark_has_nvptx_metadata)
  if(_ark_has_nvptx_metadata GREATER -1)
    ark_llvm_lib_exists(_ark_has_nvptx_codegen LLVMNVPTXCodeGen)
    if(_ark_has_nvptx_codegen)
      set(ARK_HAVE_LLVM_NVPTX_BACKEND ON)
      list(APPEND LLVM_LINK_COMPONENTS NVPTX)
      message(STATUS "[Ark][GPU] NVPTX backend enabled")
    else()
      message(FATAL_ERROR "[Ark][GPU] NVPTX requested but Release LLVMNVPTXCodeGen was not found")
    endif()
  else()
    message(FATAL_ERROR "[Ark][GPU] NVPTX requested but is absent from LLVM_TARGETS_TO_BUILD")
  endif()
endif()

set(ARK_HAVE_LLVM_AMDGPU_BACKEND OFF)
if(ARK_ENABLE_AMDGPU)
  list(FIND LLVM_TARGETS_TO_BUILD "AMDGPU" _ark_has_amdgpu_metadata)
  if(_ark_has_amdgpu_metadata GREATER -1)
    ark_llvm_lib_exists(_ark_has_amdgpu_codegen LLVMAMDGPUCodeGen)
    if(_ark_has_amdgpu_codegen)
      set(ARK_HAVE_LLVM_AMDGPU_BACKEND ON)
      list(APPEND LLVM_LINK_COMPONENTS AMDGPU)
      message(STATUS "[Ark][GPU] AMDGPU backend enabled")
    else()
      message(FATAL_ERROR "[Ark][GPU] AMDGPU requested but Release LLVMAMDGPUCodeGen was not found")
    endif()
  else()
    message(FATAL_ERROR "[Ark][GPU] AMDGPU requested but is absent from LLVM_TARGETS_TO_BUILD")
  endif()
endif()

list(REMOVE_DUPLICATES LLVM_LINK_COMPONENTS)
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

set(ARK_REQUIRED_MLIR_LIBS)

foreach(_mlir_target
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRTransforms
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
  MLIRIndexDialect
  MLIRUBDialect
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRROCDLDialect

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
)
  ark_require_imported_target(ARK_REQUIRED_MLIR_LIBS ${_mlir_target})
endforeach()

set(ARK_ENABLE_NVPTX_GPU_PIPELINE ${ARK_HAVE_LLVM_NVPTX_BACKEND})
set(ARK_ENABLE_AMDGPU_GPU_PIPELINE ${ARK_HAVE_LLVM_AMDGPU_BACKEND})

set(ARK_ENABLE_GPU_LOWERING OFF)
if(ARK_ENABLE_NVPTX_GPU_PIPELINE OR ARK_ENABLE_AMDGPU_GPU_PIPELINE)
  set(ARK_ENABLE_GPU_LOWERING ON)
endif()

if(ARK_ENABLE_GPU_LOWERING)
  message(STATUS "[Ark][GPU] GPU lowering enabled")
else()
  message(STATUS "[Ark][GPU] GPU lowering disabled")
endif()

set(ARK_HOST_JIT_MLIR_LIBS)
if(ARK_ENABLE_HOST_JIT)
  ark_require_imported_target(ARK_HOST_JIT_MLIR_LIBS MLIRExecutionEngine)
endif()

set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")

set(RUNTIME_COMMON_SOURCES
  "${RUNTIME_DIR}/core/async.cpp"
  "${RUNTIME_DIR}/core/memory.cpp"
  "${RUNTIME_DIR}/core/string.cpp"
  "${RUNTIME_DIR}/core/print.cpp"
  "${RUNTIME_DIR}/core/panic.cpp"
  "${RUNTIME_DIR}/core/hash.cpp"
  "${RUNTIME_DIR}/core/vector.cpp"
  "${RUNTIME_DIR}/core/gpu_hal.cpp"
  "${RUNTIME_DIR}/net/remote.cpp"
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

set(ARK_RUNTIME_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tools/compiler")

configure_file(
  "${RUNTIME_DIR}/include/ark_config.h.in"
  "${CMAKE_BINARY_DIR}/include/ark_config.h"
)

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

function(ark_configure_runtime_target target_name)
  add_dependencies(${target_name} PrepareRuntimeHeaders)

  target_include_directories(${target_name} PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${RUNTIME_DIR}/include
    ${CMAKE_SOURCE_DIR}/libs/ark-abi/include
    ${CMAKE_SOURCE_DIR}/libs/ark-wire/include
    ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include
  )

  target_link_libraries(${target_name} PRIVATE
    ${ARK_PLATFORM_LINK_LIBS}
    ${ARK_PLATFORM_DL_LIBS}
  )

  target_compile_features(${target_name} PRIVATE cxx_std_20)

  if(WIN32)
    target_compile_definitions(${target_name} PRIVATE
      NOMINMAX
      WIN32_LEAN_AND_MEAN
      _CRT_SECURE_NO_DEPRECATE
      _CRT_SECURE_NO_WARNINGS
      _CRT_NONSTDC_NO_DEPRECATE
      _CRT_NONSTDC_NO_WARNINGS
      _SCL_SECURE_NO_DEPRECATE
      _SCL_SECURE_NO_WARNINGS
      UNICODE
      _UNICODE
      __STDC_CONSTANT_MACROS
      __STDC_FORMAT_MACROS
      __STDC_LIMIT_MACROS
    )
  endif()

  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${ARK_RUNTIME_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${ARK_RUNTIME_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${ARK_RUNTIME_OUTPUT_DIR}"
    POSITION_INDEPENDENT_CODE ON
  )
endfunction()

add_library(ark-runtime-static STATIC
  ${RUNTIME_CORE_SOURCES}
)
set_target_properties(ark-runtime-static PROPERTIES
  OUTPUT_NAME "ark_runtime_static"
)

add_library(ark-runtime-shared SHARED
  ${RUNTIME_CORE_SOURCES}
)
set_target_properties(ark-runtime-shared PROPERTIES
  OUTPUT_NAME "ark_runtime"
)

if(WIN32)
  set_target_properties(ark-runtime-shared PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
endif()

ark_configure_runtime_target(ark-runtime-static)
ark_configure_runtime_target(ark-runtime-shared)

set(ARK_GENMIR_SOURCES
  src/Frontend/GenMIR/GenMIR.Runtime.cpp
  src/Frontend/GenMIR/GenMIR.Types.cpp
  src/Frontend/GenMIR/GenMIR.GPU.cpp
  src/Frontend/GenMIR/GenMIR.Utils.cpp
  src/Frontend/GenMIR/GenMIR.Schema.cpp
  src/Frontend/GenMIR/GenMIR.Intrinsics.cpp
  src/Frontend/GenMIR.cpp
)

set(ARK_COMPILER_SHARED_SOURCES
  src/Support/Hud.cpp

  src/Pipeline/Compiler.cpp
  src/Pipeline/Linker.cpp
  src/Pipeline/LinkerGpuFatbin.cpp
  src/Integration/CapsuleBackend.cpp

  src/Frontend/Lexer.cpp
  src/Frontend/Parser.cpp
  ${ARK_GENMIR_SOURCES}
  src/Frontend/MirBuilder.cpp
  src/Frontend/BuiltinNamespaces.cpp
  src/Frontend/Intrinsics.cpp

  src/Analysis/OwnershipVerifier.cpp
  src/Analysis/OwnershipUtils.cpp
  src/Transforms/DropInsertion.cpp

  src/Transforms/GpuLowering.cpp
  src/Transforms/GpuKernelOutlining.cpp
  src/Transforms/RuntimeABIVerifier.cpp

  src/Frontend/RuntimeManager.cpp
  src/Frontend/ModuleRegistry.cpp
  src/Frontend/SymbolRegistry.cpp
)

if(ARK_ENABLE_HOST_JIT)
  list(APPEND ARK_COMPILER_SHARED_SOURCES
    src/Pipeline/JIT.cpp
  )
else()
  list(APPEND ARK_COMPILER_SHARED_SOURCES
    src/Pipeline/JIT.Disabled.cpp
  )
endif()

function(ark_strip_cpp_mode_flags target_name)
  get_target_property(_opts ${target_name} COMPILE_OPTIONS)
  if(NOT _opts)
    set(_opts "")
  endif()

  set(_clean_opts ${_opts})
  list(REMOVE_ITEM _clean_opts
    -fno-exceptions
    -fexceptions
    -fno-rtti
    -frtti
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

  foreach(_flag
    -fno-exceptions
    -fexceptions
    -fno-rtti
    -frtti
    /EHs
    /EHsc
    /EHc
    /EHs-
    /EHc-
    /GR
    /GR-
  )
    string(REPLACE "${_flag}" "" _flags "${_flags}")
  endforeach()

  string(REGEX REPLACE "[ \t]+" " " _flags "${_flags}")
  string(STRIP "${_flags}" _flags)

  set_target_properties(${target_name} PROPERTIES
    COMPILE_OPTIONS "${_clean_opts}"
    COMPILE_FLAGS   "${_flags}"
  )
endfunction()

function(ark_enforce_cpp_mode target_name)
  ark_strip_cpp_mode_flags(${target_name})

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /EHsc /GR)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target_name} PRIVATE -fexceptions -frtti)
  endif()
endfunction()

if(DEFINED LLVM_DIR AND NOT LLVM_DIR STREQUAL "")
  get_filename_component(ARK_DEFAULT_LLVM_BIN_DIR "${LLVM_DIR}/../../../bin" ABSOLUTE)
else()
  set(ARK_DEFAULT_LLVM_BIN_DIR "")
endif()

set(ARKNET_CLI_SOURCES
  src/CLI/Main.cpp
  src/CLI/Init.cpp
  src/CLI/Common.cpp
  src/CLI/Compile.cpp
  src/CLI/Registry.cpp
  src/CLI/Lsp.cpp
  src/CLI/Provider.cpp
  src/CLI/Config.cpp
  src/CLI/Doctor.cpp
  src/CLI/Workspace.cpp
)

add_ark_tool(arknet
  ${ARKNET_CLI_SOURCES}
  ${ARK_COMPILER_SHARED_SOURCES}
)

foreach(TOOL_TARGET arknet)

  add_dependencies(${TOOL_TARGET}
    ArkMirDialectIncGen
    ArkMirOpsIncGen
    ArkMirTypesIncGen
    PrepareRuntimeHeaders
    ark-runtime-static
    ark-runtime-shared
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
    ark-runtime-static

    ArkMirDialect
    ArkToLLVM

    ${llvm_libs}
    ${ARK_REQUIRED_MLIR_LIBS}
    ${ARK_HOST_JIT_MLIR_LIBS}

    ${ARK_PLATFORM_LINK_LIBS}
    ${ARK_PLATFORM_DL_LIBS}
  )

  if(WIN32)
    target_compile_definitions(${TOOL_TARGET} PRIVATE
      NOMINMAX
      WIN32_LEAN_AND_MEAN
    )
  endif()

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

  if(ARK_DEFAULT_LLVM_BIN_DIR)
    target_compile_definitions(${TOOL_TARGET} PRIVATE
      ARK_DEFAULT_LLVM_BIN_DIR="${ARK_DEFAULT_LLVM_BIN_DIR}"
    )
  endif()

  target_compile_definitions(${TOOL_TARGET} PRIVATE
    ARK_RUNTIME_SOURCE_DIR="${RUNTIME_DIR}"
    ARK_BUILD_DIR="${CMAKE_BINARY_DIR}"
    ARK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )

  if(UNIX AND NOT APPLE)
    target_link_options(${TOOL_TARGET} PRIVATE "-rdynamic")
  elseif(APPLE)
    target_link_options(${TOOL_TARGET} PRIVATE "-Wl,-export_dynamic")
  endif()

  set_target_properties(${TOOL_TARGET} PROPERTIES ENABLE_EXPORTS 1)

  ark_enforce_cpp_mode(${TOOL_TARGET})

  get_target_property(_ark_tool_opts ${TOOL_TARGET} COMPILE_OPTIONS)
  message(STATUS "[Ark][${TOOL_TARGET}] COMPILE_OPTIONS=${_ark_tool_opts}")

endforeach()

target_link_libraries(arknet PRIVATE
  CLI11::CLI11
  tomlplusplus::tomlplusplus
)

add_executable(ark-stub src/Runner/Stub.cpp)
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
  target_compile_definitions(ark-stub PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
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

install(TARGETS
  arknet
  ark-stub
  ark-runtime-static
  ark-runtime-shared
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/libs/ark-abi/include/
  ${CMAKE_SOURCE_DIR}/libs/ark-wire/include/
  ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include/
  ${RUNTIME_DIR}/include/
  DESTINATION include
)

install(FILES
  ${CMAKE_BINARY_DIR}/include/ark_config.h
  DESTINATION include
)

add_subdirectory(Runtime/gpu)