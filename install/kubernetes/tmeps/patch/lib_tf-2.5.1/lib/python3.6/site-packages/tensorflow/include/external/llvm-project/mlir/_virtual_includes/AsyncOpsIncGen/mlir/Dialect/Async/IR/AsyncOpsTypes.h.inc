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
namespace async {
  class CoroHandleType;
  class CoroIdType;
  class CoroStateType;
  class GroupType;
  class TokenType;
  class ValueType;

  class CoroHandleType : public ::mlir::Type::TypeBase<CoroHandleType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("coro.handle");
    }
  };

  class CoroIdType : public ::mlir::Type::TypeBase<CoroIdType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("coro.id");
    }
  };

  class CoroStateType : public ::mlir::Type::TypeBase<CoroStateType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("coro.state");
    }
  };

  class GroupType : public ::mlir::Type::TypeBase<GroupType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("group");
    }
  };

  class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("token");
    }
  };

  namespace detail {
    struct ValueTypeStorage;
  } // end namespace detail
  class ValueType : public ::mlir::Type::TypeBase<ValueType, ::mlir::Type,
                                         detail::ValueTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static ValueType get(Type valueType);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("value");
    }

    static ::mlir::Type parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser);
    void print(::mlir::DialectAsmPrinter &printer) const;
    Type getValueType() const;
  };
} // namespace async
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

