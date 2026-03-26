// tools/arkc/abi_asserts.cc
#include "ark/abi.h"
#include <type_traits>

static_assert(sizeof(ArkStr) == 16);
static_assert(alignof(ArkStr) == 8);

static_assert(sizeof(ArkVec) == 24);
static_assert(alignof(ArkVec) == 8);

static_assert(sizeof(ArkSlice) == 16);
static_assert(alignof(ArkSlice) == 8);

static_assert(sizeof(ArkUnionHeader) == 8);
static_assert(alignof(ArkUnionHeader) == 4);

static_assert(sizeof(ArkTensor) == 48);
static_assert(alignof(ArkTensor) == 8);
