/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::arm_sve::ScalableVectorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


static ::mlir::OptionalParseResult generatedTypeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic,
                                      ::mlir::Type &value) {
  if (mnemonic == ::mlir::arm_sve::ScalableVectorType::getMnemonic()) { 
    value = ::mlir::arm_sve::ScalableVectorType::parse(context, parser);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedTypePrinter(
                         ::mlir::Type def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)
    .Case<::mlir::arm_sve::ScalableVectorType>([&](::mlir::arm_sve::ScalableVectorType t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Type) { return ::mlir::failure(); });
}

namespace mlir {
namespace arm_sve {

namespace detail {
  struct ScalableVectorTypeStorage : public ::mlir::TypeStorage {
    ScalableVectorTypeStorage (::llvm::ArrayRef<int64_t> shape, Type elementType)
      : shape(shape), elementType(elementType) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    if (!(shape == std::get<0>(key)))
      return false;
    if (!(elementType == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static ScalableVectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto shape = std::get<0>(key);
      auto elementType = std::get<1>(key);
      shape = allocator.copyInto(shape);

      return new (allocator.allocate<ScalableVectorTypeStorage>())
          ScalableVectorTypeStorage(shape, elementType);
    }
      ::llvm::ArrayRef<int64_t> shape;
      Type elementType;
  };
} // namespace detail
ScalableVectorType ScalableVectorType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType) {
  return Base::get(context, shape, elementType);
}
::llvm::ArrayRef<int64_t> ScalableVectorType::getShape() const { return getImpl()->shape; }
Type ScalableVectorType::getElementType() const { return getImpl()->elementType; }
void ScalableVectorType::print(::mlir::DialectAsmPrinter &printer) const {

    printer << "vector<";
    for (int64_t dim : getShape())
      printer << dim << 'x';
    printer << getElementType() << '>';
  
}
::mlir::Type ScalableVectorType::parse(::mlir::MLIRContext *context, ::mlir::DialectAsmParser &parser) {

    VectorType vector;
    if (parser.parseType(vector))
      return Type();
    return get(context, vector.getShape(), vector.getElementType());
  
}
} // namespace arm_sve
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

