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
namespace pdl {
  class AttributeType;
  class OperationType;
  class RangeType;
  class TypeType;
  class ValueType;

  class AttributeType : public ::mlir::Type::TypeBase<AttributeType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("attribute");
    }
  };

  class OperationType : public ::mlir::Type::TypeBase<OperationType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("operation");
    }
  };

  namespace detail {
    struct RangeTypeStorage;
  } // end namespace detail
  class RangeType : public ::mlir::Type::TypeBase<RangeType, ::mlir::pdl::PDLType,
                                         detail::RangeTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static RangeType get(Type elementType);
    static RangeType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type elementType);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("range");
    }

    static ::mlir::Type parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser);
    void print(::mlir::DialectAsmPrinter &printer) const;
    Type getElementType() const;
  };

  class TypeType : public ::mlir::Type::TypeBase<TypeType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("type");
    }
  };

  class ValueType : public ::mlir::Type::TypeBase<ValueType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("value");
    }
  };
} // namespace pdl
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

