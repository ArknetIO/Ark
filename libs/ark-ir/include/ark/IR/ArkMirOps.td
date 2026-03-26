#ifndef ARK_MIR_OPS_TD
#define ARK_MIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"

// Types + dialect live here.
// Do NOT redefine "def ArkMir_Dialect" in this file.
include "ark/IR/ArkMirTypes.td"

// =============================================================================
// Constraints
// =============================================================================

// Local replacement for "AnySignlessIntegerOrIndex".
def ArkIndex : AnyTypeOf<[AnyInteger, Index], "integer or index">;

// =============================================================================
// Operations
// =============================================================================

class ArkMir_Op<string mnemonic, list<Trait> traits = []>
  : Op<ArkMir_Dialect, mnemonic, traits>;

// =============================================================================
// Place/State Ops
// =============================================================================

// Slot allocates storage for a place (host stack-like abstraction), returns a fresh state token.
// Runtime-effectful.
def SlotOp : ArkMir_Op<"slot", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocates a slot and produces initial state token";
  let arguments = (ins I64Attr:$id);
  let results   = (outs ArkPlaceType:$place, ArkStateType:$state);

  let assemblyFormat = "$id attr-dict `:` type($place) `,` type($state)";

  let builders = [
    OpBuilder<(ins "::mlir::Type":$pointeeType, "int64_t":$idVal), [{
      auto *ctx = $_builder.getContext();
      auto placeTy = ::arklang::mir::PlaceType::get(ctx, idVal, pointeeType);
      auto stateTy = ::arklang::mir::StateType::get(ctx, idVal);
      build($_builder, $_state, placeTy, stateTy, idVal);
    }]>
  ];
}

// Store/Read/MoveOut/Drop are runtime-effectful. We model them with MemoryEffects to help analyses.

def StoreOp : ArkMir_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Stores value and threads state token";
  let arguments = (ins ArkPlaceType:$place, AnyType:$value, ArkStateType:$stateIn);
  let results   = (outs ArkStateType:$stateOut);

  let assemblyFormat =
    "$value `into` $place `(` $stateIn `)` attr-dict `:` type($value) `,` type($place) `,` type($stateIn) `->` type($stateOut)";
  let hasVerifier = 1;
}

def ReadOp : ArkMir_Op<"read", [MemoryEffects<[MemRead]>]> {
  let summary = "Reads value and threads state token";
  let arguments = (ins ArkPlaceType:$place, ArkStateType:$stateIn);
  let results   = (outs AnyType:$value, ArkStateType:$stateOut);

  let assemblyFormat =
    "$place `(` $stateIn `)` attr-dict `:` type($place) `,` type($stateIn) `->` type($value) `,` type($stateOut)";
  let hasVerifier = 1;
}

def MoveOutOp : ArkMir_Op<"move_out", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Moves value and threads state token";
  let arguments = (ins ArkPlaceType:$place, ArkStateType:$stateIn);
  let results   = (outs AnyType:$value, ArkStateType:$stateOut);

  let assemblyFormat =
    "$place `(` $stateIn `)` attr-dict `:` type($place) `,` type($stateIn) `->` type($value) `,` type($stateOut)";
  let hasVerifier = 1;
}

def DropOp : ArkMir_Op<"drop", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroys value and threads state token";
  let arguments = (ins ArkPlaceType:$place, ArkStateType:$stateIn);
  let results   = (outs ArkStateType:$stateOut);

  let assemblyFormat =
    "$place `(` $stateIn `)` attr-dict `:` type($place) `,` type($stateIn) `->` type($stateOut)";
  let hasVerifier = 1;
}

def IndexOp : ArkMir_Op<"index"> {
  let summary = "Projects an element sub-place (preserves ID)";
  let arguments = (ins ArkPlaceType:$base, ArkIndex:$indexVal);
  let results   = (outs ArkPlaceType:$elem);

  let assemblyFormat =
    "$base `[` $indexVal `]` attr-dict `:` type($base) `,` type($indexVal) `->` type($elem)";
  let hasVerifier = 1;
}

// =============================================================================
// Memory Ops
// =============================================================================

// ark.alloc models device/host buffer allocation.
// - The result memref type encodes memory space (e.g. memref<?xf32, 1> for GPU).
// - Dynamic sizes match memref.alloc semantics: one Index per dynamic dimension.
// - Optional "ark.device_id" (i32) selects physical device when memory space indicates GPU.
// This op is intended to replace memref.alloc during Ark->ArkMIR lowering.
def AllocOp : ArkMir_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocates heap or device memory";
  let description = [{
    Allocates a buffer whose element type/shape/memory-space are encoded in the result memref type.

    Examples:
      %0 = ark.alloc(%n) : memref<?xf32>
      %1 = ark.alloc(%n) {ark.device_id = 0 : i32} : memref<?xf32, 1>
      %2 = ark.alloc(%m, %n) : memref<?x?xf16>
  }];

  // One Index operand per dynamic dimension.
  let arguments = (ins Variadic<Index>:$sizes);
  let results   = (outs AnyMemRef:$result);

  let assemblyFormat = "`(` $sizes `)` attr-dict `:` type($result)";

  let hasVerifier = 1;
}

// =============================================================================
// Async / Kernel Ops
// =============================================================================

def LaunchOp : ArkMir_Op<"launch"> {
  let summary = "Launches a kernel function asynchronously";
  let description = [{
    Dispatches a function to the accelerator. Returns a token (i64 handle) that can be awaited.
  }];

  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results   = (outs I64:$token);

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

def AwaitOp : ArkMir_Op<"await"> {
  let summary = "Blocks until the associated token signals completion";
  let arguments = (ins I64:$token);
  let results = (outs);

  let assemblyFormat = "$token attr-dict";
}

#endif // ARK_MIR_OPS_TD
