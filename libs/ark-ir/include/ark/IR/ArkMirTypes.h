#pragma once

// 1. Source Include (Keeps full path because it exists in libs/ark-ir/include/ark/IR)
#include "ark/IR/ArkMirDialect.h"

#include "mlir/IR/Types.h"

// 2. Generated Include (Drop path because it lives in build/libs/ark-ir/IR)
#define GET_TYPEDEF_CLASSES
#include "ArkMirTypes.h.inc"  // <--- FIXED
#undef GET_TYPEDEF_CLASSES