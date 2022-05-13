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
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
namespace arm_sve {
  class ScalableVectorType;

  namespace detail {
    struct ScalableVectorTypeStorage;
  } // end namespace detail
  class ScalableVectorType : public ::mlir::Type::TypeBase<ScalableVectorType, ::mlir::Type,
                                         detail::ScalableVectorTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumElements() const {
      assert(hasStaticShape() &&
             "cannot get element count of dynamic shaped type");
      ArrayRef<int64_t> shape = getShape();
      int64_t num = 1;
      for (auto dim : shape)
        num *= dim;
      return num;
    }
  
    static ScalableVectorType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("vector");
    }

    static ::mlir::Type parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser);
    void print(::mlir::DialectAsmPrinter &printer) const;
    ::llvm::ArrayRef<int64_t> getShape() const;
    Type getElementType() const;
  };
} // namespace arm_sve
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

