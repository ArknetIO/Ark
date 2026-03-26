#ifndef ARK_MIR_TYPES_TD
#define ARK_MIR_TYPES_TD

include "ark/IR/ArkMirDialect.td"
include "mlir/IR/AttrTypeBase.td"
// [NEW] Required for MemRefElementTypeInterface
include "mlir/IR/BuiltinTypeInterfaces.td" 

// Base class for all MIR types
class ArkMir_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ArkMir_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// =============================================================================
// 1. Core Safety Types (The "Borrow Checker" Types)
// =============================================================================

// Places and States are meta-types, not storage types, so they usually don't need MemRef interface.
def ArkPlaceType : ArkMir_Type<"Place", "place"> {
  let summary = "A tracked memory location (!ark.mir.place<id, type>)";
  
  // This line automatically generates getId() and getPointee()
  let parameters = (ins "int64_t":$id, "::mlir::Type":$pointee);
  
  let assemblyFormat = "$id `,` $pointee";
}

def ArkStateType : ArkMir_Type<"State", "state"> {
  let summary = "An ownership state token (!ark.mir.state<id>)";
  let parameters = (ins "int64_t":$id);
  let assemblyFormat = "$id";
}

// =============================================================================
// 2. Data Layout Types (Moved from Legacy to MIR)
// =============================================================================

// [FIX] Added MemRefElementTypeInterface to allow storage in memref<?x!ark.vec<...>>

def ArkVecType : ArkMir_Type<"Vec", "vec", [MemRefElementTypeInterface]> {
  let summary = "A dynamic vector container";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def ArkSliceType : ArkMir_Type<"Slice", "slice", [MemRefElementTypeInterface]> {
  let summary = "A view into a contiguous memory region";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def ArkStructType : ArkMir_Type<"Struct", "struct", [MemRefElementTypeInterface]> {
  let summary = "A user-defined named structure";
  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "$name";
}

#endif // ARK_MIR_TYPES_TD