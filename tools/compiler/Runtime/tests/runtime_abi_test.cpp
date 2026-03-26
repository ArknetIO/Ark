// tools/arkc/Runtime/tests/runtime_abi_test.cpp
#include <ark_protocol.h>
#include <cstddef>
#include <cassert>

// Compile-time checks (Redundant with fs_posix but good for CI)
static_assert(sizeof(ArkStr) == 16, "ArkStr size");
static_assert(sizeof(ArkIoError) == 24, "ArkIoError size"); 
static_assert(offsetof(ArkStr, ptr) == 0, "ArkStr.ptr offset");
static_assert(offsetof(ArkStr, len) == 8, "ArkStr.len offset");

int main() {
    // Runtime smoke test could go here
    return 0;
}