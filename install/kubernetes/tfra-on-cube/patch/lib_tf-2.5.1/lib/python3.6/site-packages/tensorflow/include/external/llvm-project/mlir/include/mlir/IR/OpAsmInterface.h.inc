/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class OpAsmOpInterface;
namespace detail {
struct OpAsmOpInterfaceInterfaceTraits {
  struct Concept {
    void (*getAsmResultNames)(::mlir::Operation *, ::mlir::OpAsmSetValueNameFn);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getAsmResultNames} {}

    static inline void getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
  };
};
} // end namespace detail
class OpAsmOpInterface : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct OpAsmOpInterfaceTrait : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public OpAsmOpInterfaceTrait<ConcreteOp> {};
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
};
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmResultNames(setNameFn);
}
} // namespace mlir
