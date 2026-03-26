cmake_minimum_required(VERSION 3.20)
project(arknet VERSION 1.0.0 LANGUAGES C CXX)

# =========================================================
# 1. Global Settings
# =========================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Load our Custom Modules (The "cmake/" folder)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Load Helper Functions (add_ark_tool)
include(ArkAddTool)

# Control Flags
option(ARK_BUILD_TOOLS "Build Compiler Stack (LLVM/MLIR)" ON)
option(ARK_BUILD_SERVICES "Build Backend Stack (Drogon/Postgres)" ON)

# =========================================================
# 2. Core Libraries (The Foundation)
# =========================================================
# These have NO dependencies on LLVM or Drogon.
# ORDER MATTERS: Primitives first, then consumers.

add_subdirectory(libs/ark-abi)
add_subdirectory(libs/ark-crypto)

# [FIX] Added missing core libraries
add_subdirectory(libs/ark-wire)        # Serializers
add_subdirectory(libs/ark-diagnostics) # Error Types
add_subdirectory(libs/ark-capsule)     # The Builder SDK (Depends on wire/diag/crypto)

# =========================================================
# 3. Compiler Stack (Tools)
# =========================================================
if(ARK_BUILD_TOOLS)
    message(STATUS "🏗️  Configuring Compiler Toolchain...")

    # [CLEANUP] We use the module 'cmake/ArkMLIR.cmake' to handle 
    # all the find_package(LLVM) and include(TableGen) boilerplate.
    include(ArkMLIR)

    # 1. The Dialects (Moved from lib/ark -> libs/ark-ir)
    # This provides the 'ArkIR' and 'ArkConversion' libraries.
    add_subdirectory(libs/ark-ir)

    # 2. The CLI Tool
    # Ensure your folder is named 'tools/compiler' (or 'tools/arkc')
    add_subdirectory(tools/compiler)
    
    # [OPTIONAL] Provider Tool
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/provider/CMakeLists.txt")
        add_subdirectory(tools/provider)
    endif()
endif()

# =========================================================
# 4. Service Stack (Backend)
# =========================================================
if(ARK_BUILD_SERVICES)
    message(STATUS "☁️  Configuring Backend Services...")

    # 1. The Registry (Business Logic)
    add_subdirectory(services/registry)

    # 2. The Website (Frontend - Optional if C++ backend)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/services/website/CMakeLists.txt")
        add_subdirectory(services/website)
    endif()
endif()