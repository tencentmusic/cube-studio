/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::LLVM::FMFAttr,
::mlir::LLVM::LoopOptionsAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic, ::mlir::Type type,
                                      ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::LLVM::FMFAttr::getMnemonic()) { 
    value = ::mlir::LLVM::FMFAttr::parse(context, parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::LLVM::LoopOptionsAttr::getMnemonic()) { 
    value = ::mlir::LLVM::LoopOptionsAttr::parse(context, parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedAttributePrinter(
                         ::mlir::Attribute def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)
    .Case<::mlir::LLVM::FMFAttr>([&](::mlir::LLVM::FMFAttr t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LoopOptionsAttr>([&](::mlir::LLVM::LoopOptionsAttr t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Attribute) { return ::mlir::failure(); });
}

namespace mlir {
namespace LLVM {

namespace detail {
  struct FMFAttrStorage : public ::mlir::AttributeStorage {
    FMFAttrStorage (FastmathFlags flags)
      : ::mlir::AttributeStorage(), flags(flags) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<FastmathFlags>;
  bool operator==(const KeyTy &key) const {
    if (!(flags == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static FMFAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto flags = std::get<0>(key);

      return new (allocator.allocate<FMFAttrStorage>())
          FMFAttrStorage(flags);
    }
      FastmathFlags flags;
  };
} // namespace detail
FMFAttr FMFAttr::get(::mlir::MLIRContext *context, FastmathFlags flags) {
  return Base::get(context, flags);
}
FastmathFlags FMFAttr::getFlags() const { return getImpl()->flags; }
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

namespace detail {
  struct LoopOptionsAttrStorage : public ::mlir::AttributeStorage {
    LoopOptionsAttrStorage (::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> options)
      : ::mlir::AttributeStorage(), options(options) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>>;
  bool operator==(const KeyTy &key) const {
    if (!(options == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static LoopOptionsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto options = std::get<0>(key);
      options = allocator.copyInto(options);

      return new (allocator.allocate<LoopOptionsAttrStorage>())
          LoopOptionsAttrStorage(options);
    }
      ::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> options;
  };
} // namespace detail
::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> LoopOptionsAttr::getOptions() const { return getImpl()->options; }
} // namespace LLVM
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

