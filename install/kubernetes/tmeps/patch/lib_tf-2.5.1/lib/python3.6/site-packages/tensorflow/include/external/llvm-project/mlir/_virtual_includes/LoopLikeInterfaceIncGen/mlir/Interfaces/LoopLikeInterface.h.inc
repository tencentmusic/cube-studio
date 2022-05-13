/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class LoopLikeOpInterface;
namespace detail {
struct LoopLikeOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*isDefinedOutsideOfLoop)(::mlir::Operation *, Value );
    Region &(*getLoopBody)(::mlir::Operation *);
    LogicalResult (*moveOutOfLoop)(::mlir::Operation *, ArrayRef<Operation *>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{isDefinedOutsideOfLoop, getLoopBody, moveOutOfLoop} {}

    static inline bool isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, Value  value);
    static inline Region &getLoopBody(::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ArrayRef<Operation *> ops);
  };
};
} // end namespace detail
class LoopLikeOpInterface : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct LoopLikeOpInterfaceTrait : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public LoopLikeOpInterfaceTrait<ConcreteOp> {};
  bool isDefinedOutsideOfLoop(Value  value);
  Region &getLoopBody();
  LogicalResult moveOutOfLoop(ArrayRef<Operation *> ops);
};
template<typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, Value  value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDefinedOutsideOfLoop(value);
}
template<typename ConcreteOp>
Region &detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopBody(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopBody();
}
template<typename ConcreteOp>
LogicalResult detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ArrayRef<Operation *> ops) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).moveOutOfLoop(ops);
}
} // namespace mlir
