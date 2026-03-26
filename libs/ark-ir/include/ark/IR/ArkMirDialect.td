#ifndef ARK_MIR_DIALECT_TD
#define ARK_MIR_DIALECT_TD

include "mlir/IR/OpBase.td"

// This matches the "-dialect=ark_mir" flag in CMake
def ArkMir_Dialect : Dialect {
  let name = "ark_mir";
  let cppNamespace = "::arklang::mir";
  let summary = "Ark MIR (Memory Safe Elite Path)";
  let useDefaultTypePrinterParser = 1;
}

#endif // ARK_MIR_DIALECT_TD