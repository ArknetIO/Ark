cmake_minimum_required(VERSION 3.20)

# -----------------------------------------------------------------------------
# 1. Platform Detection
# -----------------------------------------------------------------------------
set(PLATFORM_IMPL "")

if(APPLE)
    message(STATUS "ArkCapsule: Building for macOS (Platform_Apple.cpp)")
    set(PLATFORM_IMPL "src/Platform_Apple.cpp")
elseif(UNIX)
    message(STATUS "ArkCapsule: Building for Linux (Platform_Linux.cpp)")
    set(PLATFORM_IMPL "src/Platform_Linux.cpp")
elseif(WIN32)
    message(STATUS "ArkCapsule: Building for Windows (Platform_Windows.cpp)")
    set(PLATFORM_IMPL "src/Platform_Windows.cpp")
else()
    message(FATAL_ERROR "ArkCapsule: Unsupported target platform")
endif()

# -----------------------------------------------------------------------------
# 2. Define the Library
# -----------------------------------------------------------------------------
add_library(ark-capsule STATIC
    # Core Logic
    src/CapsuleCodec.cpp
    
    # Platform-Specific Logic (exec_payload, self_path)
    ${PLATFORM_IMPL}
    
    # [LEGACY/UNUSED]
    # src/Scanner.cpp
    # src/PlanHasher.cpp
    # src/ArtifactSealer.cpp
    # src/BuildEngine.cpp
    # src/ReceiptIssuer.cpp
)

# -----------------------------------------------------------------------------
# 3. Dependencies
# -----------------------------------------------------------------------------
# [FIX] Link PUBLIC so consumers get the include paths and definitions
target_link_libraries(ark-capsule PUBLIC 
    ark-abi 
    ark-wire 
    Ark::Crypto 
    ark-diagnostics
)

target_include_directories(ark-capsule PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -----------------------------------------------------------------------------
# 4. Compiler Settings
# -----------------------------------------------------------------------------
target_compile_features(ark-capsule PUBLIC cxx_std_20)

set_target_properties(ark-capsule PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(ark-capsule PRIVATE /W4 /WX)
else()
    target_compile_options(ark-capsule PRIVATE -Wall -Wextra -Wpedantic)
endif()