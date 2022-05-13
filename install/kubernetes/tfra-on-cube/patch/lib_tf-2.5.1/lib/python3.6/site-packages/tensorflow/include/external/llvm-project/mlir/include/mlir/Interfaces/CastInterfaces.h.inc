/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class CastOpInterface;
namespace detail {
struct CastOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*areCastCompatible)(mlir::TypeRange, mlir::TypeRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{areCastCompatible} {}

    static inline bool areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs);
  };
};
} // end namespace detail
class CastOpInterface : public ::mlir::OpInterface<CastOpInterface, detail::CastOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CastOpInterface, detail::CastOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct CastOpInterfaceTrait : public ::mlir::OpInterface<CastOpInterface, detail::CastOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return impl::verifyCastInterfaceOp(op, ConcreteOp::areCastCompatible);
    }

    /// Attempt to fold the given cast operation.
    static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
      return impl::foldCastInterfaceOp(op, operands, results);
    }
  
  };
  template <typename ConcreteOp>
  struct Trait : public CastOpInterfaceTrait<ConcreteOp> {};
  bool areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs);
};
template<typename ConcreteOp>
bool detail::CastOpInterfaceInterfaceTraits::Model<ConcreteOp>::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
  return ConcreteOp::areCastCompatible(inputs, outputs);
}
} // namespace mlir
