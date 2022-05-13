/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
class AsyncOpInterface;
namespace detail {
struct AsyncOpInterfaceInterfaceTraits {
  struct Concept {
    OperandRange (*getAsyncDependencies)(::mlir::Operation *);
    void (*addAsyncDependency)(::mlir::Operation *, Value);
    OpResult (*getAsyncToken)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getAsyncDependencies, addAsyncDependency, getAsyncToken} {}

    static inline OperandRange getAsyncDependencies(::mlir::Operation *tablegen_opaque_val);
    static inline void addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token);
    static inline OpResult getAsyncToken(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class AsyncOpInterface : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AsyncOpInterfaceTrait : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    OperandRange getAsyncDependencies() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
    }
    void addAsyncDependency(Value token) {
      ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
    }
    OpResult getAsyncToken() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AsyncOpInterfaceTrait<ConcreteOp> {};
  OperandRange getAsyncDependencies();
  void addAsyncDependency(Value token);
  OpResult getAsyncToken();
};
template<typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncDependencies();
}
template<typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addAsyncDependency(token);
}
template<typename ConcreteOp>
OpResult detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsyncToken(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncToken();
}
} // namespace gpu
} // namespace mlir
