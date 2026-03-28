cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

project(arknet VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(ArkAddTool)

option(ARK_BUILD_TOOLS "Build Compiler Stack (LLVM/MLIR)" ON)
option(ARK_BUILD_SERVICES "Build Backend Stack (Drogon/Postgres)" ON)

option(ARK_ENABLE_GPU "Enable GPU codegen support" ON)
option(ARK_ENABLE_AMDGPU "Enable AMDGPU backend support" ON)
option(ARK_ENABLE_NVPTX "Enable NVPTX backend support" ON)
option(ARK_ENABLE_SPIRV "Enable SPIR-V oriented paths" ON)

set(ARK_REQUESTED_GPU_BACKENDS)
set(ARK_GPU_BACKENDS)

if(ARK_ENABLE_GPU)
    if(ARK_ENABLE_AMDGPU)
        list(APPEND ARK_REQUESTED_GPU_BACKENDS AMDGPU)
    endif()

    if(ARK_ENABLE_NVPTX)
        list(APPEND ARK_REQUESTED_GPU_BACKENDS NVPTX)
    endif()

    if(ARK_ENABLE_SPIRV)
        list(APPEND ARK_REQUESTED_GPU_BACKENDS SPIRV)
    endif()
else()
    if(ARK_ENABLE_AMDGPU OR ARK_ENABLE_NVPTX OR ARK_ENABLE_SPIRV)
        message(STATUS "ARK_ENABLE_GPU=OFF, ignoring individual GPU backend options")
    endif()
endif()

set(ARK_GPU_BACKENDS ${ARK_REQUESTED_GPU_BACKENDS} CACHE INTERNAL "Requested Ark GPU backends")
set(ARK_REQUESTED_GPU_BACKENDS ${ARK_REQUESTED_GPU_BACKENDS} CACHE INTERNAL "Requested Ark GPU backends")
message(STATUS "Ark requested GPU backends: ${ARK_REQUESTED_GPU_BACKENDS}")

add_subdirectory(libs/ark-abi)
add_subdirectory(libs/ark-crypto)
add_subdirectory(libs/ark-wire)
add_subdirectory(libs/ark-diagnostics)
add_subdirectory(libs/ark-capsule)

if(ARK_BUILD_TOOLS)
    message(STATUS "🏗️  Configuring Compiler Toolchain...")

    if(NOT DEFINED LLVM_DIR OR LLVM_DIR STREQUAL "")
        message(FATAL_ERROR "LLVM_DIR is required when ARK_BUILD_TOOLS=ON")
    endif()

    if(NOT DEFINED MLIR_DIR OR MLIR_DIR STREQUAL "")
        message(FATAL_ERROR "MLIR_DIR is required when ARK_BUILD_TOOLS=ON")
    endif()

    include(ArkMLIR)

    add_subdirectory(libs/ark-ir)
    add_subdirectory(tools/compiler)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/provider/CMakeLists.txt")
        add_subdirectory(tools/provider)
    endif()
endif()

if(ARK_BUILD_SERVICES)
    message(STATUS "☁️  Configuring Backend Services...")

    add_subdirectory(services/registry)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/services/website/CMakeLists.txt")
        add_subdirectory(services/website)
    endif()
endif()