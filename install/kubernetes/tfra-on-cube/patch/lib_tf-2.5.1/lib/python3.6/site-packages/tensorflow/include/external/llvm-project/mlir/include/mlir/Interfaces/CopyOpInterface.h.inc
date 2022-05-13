/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class CopyOpInterface;
namespace detail {
struct CopyOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Value (*getSource)(::mlir::Operation *);
    ::mlir::Value (*getTarget)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getSource, getTarget} {}

    static inline ::mlir::Value getSource(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getTarget(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class CopyOpInterface : public ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct CopyOpInterfaceTrait : public ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public CopyOpInterfaceTrait<ConcreteOp> {};
  ::mlir::Value getSource();
  ::mlir::Value getTarget();
};
template<typename ConcreteOp>
::mlir::Value detail::CopyOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSource(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSource();
}
template<typename ConcreteOp>
::mlir::Value detail::CopyOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTarget(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTarget();
}
} // namespace mlir
