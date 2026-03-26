cmake_minimum_required(VERSION 3.20)

if(TARGET Ark::Crypto)
  return()
endif()

if(TARGET ark-crypto)
  add_library(Ark::Crypto ALIAS ark-crypto)
  return()
endif()

if(NOT CMAKE_C_COMPILER_LOADED)
  enable_language(C)
endif()

if(NOT CMAKE_CXX_COMPILER_LOADED)
  enable_language(CXX)
endif()

add_library(ark-crypto STATIC
  src/Core.cpp
  src/Hash.cpp
  src/Sign.cpp
  src/Box.cpp
  src/blake3.c
  src/Vault.cpp
)

add_library(Ark::Crypto ALIAS ark-crypto)

set_source_files_properties(
  src/blake3.c
  PROPERTIES
    LANGUAGE C
)

target_include_directories(ark-crypto
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ark-crypto
  PUBLIC
    cxx_std_20
)

set_target_properties(ark-crypto PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
)

if(MSVC)
  target_compile_options(ark-crypto
    PRIVATE
      /W4
      /WX
  )
else()
  target_compile_options(ark-crypto
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wshadow
      -Wformat=2
      -Wconversion
  )
endif()

if(NOT TARGET PkgConfig::Sodium)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(Sodium REQUIRED IMPORTED_TARGET libsodium)
endif()

target_link_libraries(ark-crypto
  PUBLIC
    PkgConfig::Sodium
)