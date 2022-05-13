/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DerivedAttributeOpInterface;
namespace detail {
struct DerivedAttributeOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*isDerivedAttribute)(StringRef);
    DictionaryAttr (*materializeDerivedAttributes)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{isDerivedAttribute, materializeDerivedAttributes} {}

    static inline bool isDerivedAttribute(StringRef name);
    static inline DictionaryAttr materializeDerivedAttributes(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class DerivedAttributeOpInterface : public ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct DerivedAttributeOpInterfaceTrait : public ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public DerivedAttributeOpInterfaceTrait<ConcreteOp> {};
  bool isDerivedAttribute(StringRef name);
  DictionaryAttr materializeDerivedAttributes();
};
template<typename ConcreteOp>
bool detail::DerivedAttributeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDerivedAttribute(StringRef name) {
  return ConcreteOp::isDerivedAttribute(name);
}
template<typename ConcreteOp>
DictionaryAttr detail::DerivedAttributeOpInterfaceInterfaceTraits::Model<ConcreteOp>::materializeDerivedAttributes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).materializeDerivedAttributes();
}
} // namespace mlir
