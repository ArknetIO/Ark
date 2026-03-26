cmake_minimum_required(VERSION 3.20)

add_library(ark-diagnostics STATIC
    src/Diagnostic.cpp
)

target_include_directories(ark-diagnostics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------------------------
# SAFETY GATES: ENFORCE EXHAUSTIVENESS
# ------------------------------------------------------------------------------
# We force the compiler to error if a switch statement misses an enum value.
# This guarantees that adding a new DiagRule requires updating all mappers.

if(MSVC)
    target_compile_options(ark-diagnostics PRIVATE /we4062) # Error on unhandled enum
else()
    target_compile_options(ark-diagnostics PRIVATE -Werror -Wswitch -Wswitch-enum)
endif()