/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace arklang::ark {

/// Opaque handle representing an async operation
class TokenType;
/// Dynamic growable vector
class VecType;
class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ark.token";
  static constexpr ::llvm::StringLiteral dialectName = "ark";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token"};
  }

};
namespace detail {

struct VecTypeStorage;

} // namespace detail
class VecType : public ::mlir::Type::TypeBase<VecType, ::mlir::Type, detail::VecTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ark.vec";
  static constexpr ::llvm::StringLiteral dialectName = "ark";
  static VecType get(::mlir::MLIRContext *context, mlir::Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"vec"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  mlir::Type getElementType() const;
};

} // namespace arklang::ark
MLIR_DECLARE_EXPLICIT_TYPE_ID(::arklang::ark::TokenType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::arklang::ark::VecType)

#endif // GET_TYPEDEF_CLASSES

