/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
namespace LLVM {
  class FMFAttr;
  class LoopOptionsAttr;

  namespace detail {
    struct FMFAttrStorage;
  } // end namespace detail
  class FMFAttr : public ::mlir::Attribute::AttrBase<FMFAttr, ::mlir::Attribute,
                                         detail::FMFAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static FMFAttr get(::mlir::MLIRContext *context, FastmathFlags flags);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("fastmath");
    }

    static ::mlir::Attribute parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    FastmathFlags getFlags() const;
  };

  namespace detail {
    struct LoopOptionsAttrStorage;
  } // end namespace detail
  class LoopOptionsAttr : public ::mlir::Attribute::AttrBase<LoopOptionsAttr, ::mlir::Attribute,
                                         detail::LoopOptionsAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;


    using OptionValuePair = std::pair<LoopOptionCase, int64_t>;
    using OptionsArray = ArrayRef<std::pair<LoopOptionCase, int64_t>>;
    Optional<bool> disableUnroll();
    Optional<bool> disableLICM();
    Optional<int64_t> interleaveCount();
  
    static LoopOptionsAttr get(::mlir::MLIRContext *context, ArrayRef<std::pair<LoopOptionCase, int64_t>> sortedOptions);
    static LoopOptionsAttr get(::mlir::MLIRContext *context, LoopOptionsAttrBuilder & optionBuilders);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("loopopts");
    }

    static ::mlir::Attribute parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    ::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> getOptions() const;
  };
} // namespace LLVM
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

