cmake_minimum_required(VERSION 3.20)

if(TARGET ark-wire)
  return()
endif()

add_library(ark-wire STATIC
  src/ReceiptV1.cpp
)

add_library(Ark::Wire ALIAS ark-wire)

target_include_directories(ark-wire
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(ark-wire
  PUBLIC
    cxx_std_20
)

if(NOT WIN32)
  set_target_properties(ark-wire PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
endif()

if(MSVC)
  target_compile_options(ark-wire
    PRIVATE
      /W4
      /WX
  )
else()
  target_compile_options(ark-wire
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
  )
endif()