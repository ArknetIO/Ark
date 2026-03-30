cmake_minimum_required(VERSION 3.20)

# =========================================================
# ark-abi
# Shared ABI / protocol headers used across Ark libraries and runtime.
# Header-only target with stable alias export.
# =========================================================

if(TARGET ark-abi)
  return()
endif()

add_library(ark-abi INTERFACE)
add_library(Ark::ABI ALIAS ark-abi)

target_include_directories(ark-abi
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(ark-abi
  INTERFACE
    cxx_std_20
)