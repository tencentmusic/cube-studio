/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DataLayoutOpInterface;
namespace detail {
struct DataLayoutOpInterfaceInterfaceTraits {
  struct Concept {
    DataLayoutSpecInterface (*getDataLayoutSpec)(::mlir::Operation *);
    unsigned (*getTypeSize)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypeABIAlignment)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypePreferredAlignment)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getDataLayoutSpec, getTypeSize, getTypeABIAlignment, getTypePreferredAlignment} {}

    static inline DataLayoutSpecInterface getDataLayoutSpec(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  };
};
} // end namespace detail
class DataLayoutOpInterface : public ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct DataLayoutOpInterfaceTrait : public ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultTypeSize(type, dataLayout, params);
    }
    static unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
    }
    static unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout,
                                                            params);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::detail::verifyDataLayoutOp(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public DataLayoutOpInterfaceTrait<ConcreteOp> {};
  DataLayoutSpecInterface getDataLayoutSpec();
  unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
};
template<typename ConcreteOp>
DataLayoutSpecInterface detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDataLayoutSpec(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDataLayoutSpec();
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeSize(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeABIAlignment(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypePreferredAlignment(type, dataLayout, params);
}
} // namespace mlir
