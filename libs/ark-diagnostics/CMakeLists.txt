cmake_minimum_required(VERSION 3.20)

# =========================================================
# ark-diagnostics
# Diagnostic model / rule mapping library
# This subdirectory must work cleanly when included from the root build.
# =========================================================

if(TARGET ark-diagnostics)
  return()
endif()

add_library(ark-diagnostics STATIC
  src/Diagnostic.cpp
)

add_library(Ark::Diagnostics ALIAS ark-diagnostics)

target_include_directories(ark-diagnostics
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(ark-diagnostics
  PUBLIC
    cxx_std_20
)

if(NOT WIN32)
  set_target_properties(ark-diagnostics PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
endif()

# ------------------------------------------------------------------------------
# SAFETY GATES: ENFORCE EXHAUSTIVENESS
# ------------------------------------------------------------------------------
# We force the compiler to error if a switch statement misses an enum value.
# This guarantees that adding a new DiagRule requires updating all mappers.

if(MSVC)
  target_compile_options(ark-diagnostics
    PRIVATE
      /W4
      /WX
      /we4062
  )
else()
  target_compile_options(ark-diagnostics
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wswitch
      -Wswitch-enum
  )
endif()