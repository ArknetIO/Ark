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
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
)

if(NOT WIN32)
  set_target_properties(ark-crypto PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
endif()

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

if(WIN32)
  set(ARK_CRYPTO_SODIUM_TARGET "")

  # Preferred path: use the package target if vcpkg exports it cleanly.
  find_package(unofficial-sodium CONFIG QUIET)
  if(TARGET unofficial-sodium::sodium)
    set(ARK_CRYPTO_SODIUM_TARGET unofficial-sodium::sodium)
  endif()

  # Robust fallback: import libsodium directly from the active vcpkg triplet.
  if(NOT ARK_CRYPTO_SODIUM_TARGET)
    set(_ark_vcpkg_root "")
    if(DEFINED VCPKG_INSTALLED_DIR AND NOT VCPKG_INSTALLED_DIR STREQUAL "")
      set(_ark_vcpkg_root "${VCPKG_INSTALLED_DIR}")
    elseif(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE STREQUAL "")
      get_filename_component(_ark_vcpkg_scripts_dir "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
      get_filename_component(_ark_vcpkg_root "${_ark_vcpkg_scripts_dir}/.." ABSOLUTE)
      set(_ark_vcpkg_root "${_ark_vcpkg_root}/installed")
    endif()

    set(_ark_triplet "x64-windows")
    if(DEFINED VCPKG_TARGET_TRIPLET AND NOT VCPKG_TARGET_TRIPLET STREQUAL "")
      set(_ark_triplet "${VCPKG_TARGET_TRIPLET}")
    endif()

    set(_ark_sodium_include_dir "${_ark_vcpkg_root}/${_ark_triplet}/include")
    set(_ark_sodium_release_lib   "")
    set(_ark_sodium_debug_lib     "")

    foreach(_name libsodium sodium)
      if(NOT _ark_sodium_release_lib AND EXISTS "${_ark_vcpkg_root}/${_ark_triplet}/lib/${_name}.lib")
        set(_ark_sodium_release_lib "${_ark_vcpkg_root}/${_ark_triplet}/lib/${_name}.lib")
      endif()

      if(NOT _ark_sodium_debug_lib AND EXISTS "${_ark_vcpkg_root}/${_ark_triplet}/debug/lib/${_name}.lib")
        set(_ark_sodium_debug_lib "${_ark_vcpkg_root}/${_ark_triplet}/debug/lib/${_name}.lib")
      endif()
    endforeach()

    if(
      EXISTS "${_ark_sodium_include_dir}/sodium.h" AND
      (_ark_sodium_release_lib OR _ark_sodium_debug_lib)
    )
      add_library(ark_crypto_sodium UNKNOWN IMPORTED)

      set_target_properties(ark_crypto_sodium PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ark_sodium_include_dir}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      )

      if(_ark_sodium_release_lib)
        set_property(TARGET ark_crypto_sodium APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
        set_target_properties(ark_crypto_sodium PROPERTIES
          IMPORTED_LOCATION_RELEASE "${_ark_sodium_release_lib}"
        )
      endif()

      if(_ark_sodium_debug_lib)
        set_property(TARGET ark_crypto_sodium APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
        set_target_properties(ark_crypto_sodium PROPERTIES
          IMPORTED_LOCATION_DEBUG "${_ark_sodium_debug_lib}"
        )
      endif()

      set(ARK_CRYPTO_SODIUM_TARGET ark_crypto_sodium)
    endif()
  endif()

  if(NOT ARK_CRYPTO_SODIUM_TARGET)
    message(FATAL_ERROR
      "libsodium was not found on Windows.\n"
      "Expected either unofficial-sodium::sodium or direct vcpkg artifacts.\n"
      "Checked:\n"
      "  root    = ${_ark_vcpkg_root}\n"
      "  triplet = ${_ark_triplet}\n"
      "  header  = ${_ark_sodium_include_dir}/sodium.h\n"
      "  release = ${_ark_sodium_release_lib}\n"
      "  debug   = ${_ark_sodium_debug_lib}\n"
    )
  endif()

  target_link_libraries(ark-crypto
    PUBLIC
      ${ARK_CRYPTO_SODIUM_TARGET}
  )
else()
  if(NOT TARGET PkgConfig::Sodium)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Sodium REQUIRED IMPORTED_TARGET libsodium)
  endif()

  target_link_libraries(ark-crypto
    PUBLIC
      PkgConfig::Sodium
  )
endif()