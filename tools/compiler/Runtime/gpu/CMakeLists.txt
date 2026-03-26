cmake_minimum_required(VERSION 3.20)

project(ArkRuntimeGpuPlugins LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Stage location expected by your Makefile:
#   cp -f $(BUILD_COMP)/lib/libark_*_backend.* $(BIN_DIR)/lib
set(ARK_GPU_PLUGIN_OUTDIR "${CMAKE_BINARY_DIR}/lib" CACHE PATH "GPU plugin output dir")

function(ark_set_plugin_output tgt)
  set_target_properties(${tgt} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${ARK_GPU_PLUGIN_OUTDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${ARK_GPU_PLUGIN_OUTDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${ARK_GPU_PLUGIN_OUTDIR}"
  )
endfunction()

# Common includes: Runtime/include contains gpu_backend.h and friends.
set(ARK_RUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

# ==============================================================================
# CUDA plugin
# ==============================================================================
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  message(STATUS "[Ark][GPU] CUDA found: building libark_cuda_backend")

  add_library(ark_cuda_backend SHARED cuda_backend.cpp)

  target_compile_definitions(ark_cuda_backend PRIVATE
    ARK_BACKEND_CUDA
  )

  target_include_directories(ark_cuda_backend PRIVATE
    "${ARK_RUNTIME_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/libs/ark-abi/include"
  )

  target_link_libraries(ark_cuda_backend PRIVATE
    CUDA::cuda_driver
  )

  target_compile_features(ark_cuda_backend PRIVATE cxx_std_20)

  # Ensure filename is libark_cuda_backend.so / .dylib / .dll
  set_target_properties(ark_cuda_backend PROPERTIES OUTPUT_NAME "ark_cuda_backend")
  ark_set_plugin_output(ark_cuda_backend)

else()
  message(STATUS "[Ark][GPU] CUDA not found: skipping libark_cuda_backend")
endif()

# ==============================================================================
# HIP plugin
# ==============================================================================
if(EXISTS "/opt/rocm/hip")
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

find_package(hip QUIET)

if(hip_FOUND)
  message(STATUS "[Ark][GPU] HIP found: building libark_hip_backend")

  add_library(ark_hip_backend SHARED hip_backend.cpp)

  target_compile_definitions(ark_hip_backend PRIVATE
    ARK_BACKEND_HIP
  )

  target_include_directories(ark_hip_backend PRIVATE
    "${ARK_RUNTIME_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/libs/ark-abi/include"
  )

  target_link_libraries(ark_hip_backend PRIVATE
    hip::host
  )

  target_compile_features(ark_hip_backend PRIVATE cxx_std_20)

  set_target_properties(ark_hip_backend PROPERTIES OUTPUT_NAME "ark_hip_backend")
  ark_set_plugin_output(ark_hip_backend)

else()
  message(STATUS "[Ark][GPU] HIP not found: skipping libark_hip_backend")
endif()

# ==============================================================================
# Metal plugin (Apple only)
# ==============================================================================
if(APPLE)
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)

  if(METAL_LIB AND FOUNDATION_LIB)
    message(STATUS "[Ark][GPU] Metal found: building libark_metal_backend")

    add_library(ark_metal_backend SHARED metal_backend.mm)

    target_compile_definitions(ark_metal_backend PRIVATE
      ARK_BACKEND_METAL
    )

    target_include_directories(ark_metal_backend PRIVATE
      "${ARK_RUNTIME_INCLUDE_DIR}"
      "${CMAKE_SOURCE_DIR}/libs/ark-abi/include"
    )

    target_link_libraries(ark_metal_backend PRIVATE
      ${METAL_LIB}
      ${FOUNDATION_LIB}
    )

    set_target_properties(ark_metal_backend PROPERTIES OUTPUT_NAME "ark_metal_backend")
    ark_set_plugin_output(ark_metal_backend)

  else()
    message(STATUS "[Ark][GPU] Metal not found: skipping libark_metal_backend")
  endif()
endif()
