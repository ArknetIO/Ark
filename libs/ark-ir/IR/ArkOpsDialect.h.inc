/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArkOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace arklang::ark {

/// The ArkLang IR
/// Ark IR models:
///     - Async kernel launches (Tokens)
///     - Memory spaces (attributes)
///     - Parallel tensor maps
///     - Vector operations
class ArkDialect : public ::mlir::Dialect {
  explicit ArkDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ArkDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("ark");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
};

} // namespace arklang::ark
MLIR_DECLARE_EXPLICIT_TYPE_ID(::arklang::ark::ArkDialect)
