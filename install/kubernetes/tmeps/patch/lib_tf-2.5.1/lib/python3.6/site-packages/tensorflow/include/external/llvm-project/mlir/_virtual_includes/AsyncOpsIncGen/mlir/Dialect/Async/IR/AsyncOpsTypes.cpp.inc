/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::async::CoroHandleType,
::mlir::async::CoroIdType,
::mlir::async::CoroStateType,
::mlir::async::GroupType,
::mlir::async::TokenType,
::mlir::async::ValueType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


static ::mlir::OptionalParseResult generatedTypeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic,
                                      ::mlir::Type &value) {
  if (mnemonic == ::mlir::async::CoroHandleType::getMnemonic()) { 
    value = ::mlir::async::CoroHandleType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::CoroIdType::getMnemonic()) { 
    value = ::mlir::async::CoroIdType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::CoroStateType::getMnemonic()) { 
    value = ::mlir::async::CoroStateType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::GroupType::getMnemonic()) { 
    value = ::mlir::async::GroupType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::TokenType::getMnemonic()) { 
    value = ::mlir::async::TokenType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::ValueType::getMnemonic()) { 
    value = ::mlir::async::ValueType::parse(context, parser);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedTypePrinter(
                         ::mlir::Type def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)
    .Case<::mlir::async::CoroHandleType>([&](::mlir::async::CoroHandleType t) {
      printer << ::mlir::async::CoroHandleType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::CoroIdType>([&](::mlir::async::CoroIdType t) {
      printer << ::mlir::async::CoroIdType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::CoroStateType>([&](::mlir::async::CoroStateType t) {
      printer << ::mlir::async::CoroStateType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::GroupType>([&](::mlir::async::GroupType t) {
      printer << ::mlir::async::GroupType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::TokenType>([&](::mlir::async::TokenType t) {
      printer << ::mlir::async::TokenType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::ValueType>([&](::mlir::async::ValueType t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Type) { return ::mlir::failure(); });
}

namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
namespace mlir {
namespace async {

namespace detail {
  struct ValueTypeStorage : public ::mlir::TypeStorage {
    ValueTypeStorage (Type valueType)
      : valueType(valueType) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Type>;
  bool operator==(const KeyTy &key) const {
    if (!(valueType == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static ValueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto valueType = std::get<0>(key);

      return new (allocator.allocate<ValueTypeStorage>())
          ValueTypeStorage(valueType);
    }
      Type valueType;
  };
} // namespace detail
ValueType ValueType::get(Type valueType) {
  
      return Base::get(valueType.getContext(), valueType);
    ;
}
Type ValueType::getValueType() const { return getImpl()->valueType; }
} // namespace async
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

