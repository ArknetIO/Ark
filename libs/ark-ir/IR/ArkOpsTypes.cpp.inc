/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::arklang::ark::TokenType,
::arklang::ark::VecType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::arklang::ark::TokenType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::arklang::ark::TokenType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::arklang::ark::VecType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::arklang::ark::VecType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::arklang::ark::TokenType>([&](auto t) {
      printer << ::arklang::ark::TokenType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::arklang::ark::VecType>([&](auto t) {
      printer << ::arklang::ark::VecType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace arklang::ark {


} // namespace arklang::ark
MLIR_DEFINE_EXPLICIT_TYPE_ID(::arklang::ark::TokenType)
namespace arklang::ark {

namespace detail {

struct VecTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type>;
  VecTypeStorage(mlir::Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static VecTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<VecTypeStorage>()) VecTypeStorage(std::move(elementType));
  }

  mlir::Type elementType;
};

} // namespace detail
VecType VecType::get(::mlir::MLIRContext *context, mlir::Type elementType) {
  return Base::get(context, std::move(elementType));
}

::mlir::Type VecType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<mlir::Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Ark_VecType parameter 'elementType' which is to be a `mlir::Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return VecType::get(odsParser.getContext(),
      mlir::Type((*_result_elementType)));
}

void VecType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

mlir::Type VecType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace arklang::ark
MLIR_DEFINE_EXPLICIT_TYPE_ID(::arklang::ark::VecType)
namespace arklang::ark {


/// Parse a type registered to this dialect.
::mlir::Type ArkDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void ArkDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace arklang::ark

#endif // GET_TYPEDEF_CLASSES

