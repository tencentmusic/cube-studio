/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class InferFusibilityOpInterface;
namespace detail {
struct InferFusibilityOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*isFusibleWithOperand)(::mlir::Operation *);
    bool (*isFusibleWithConsumer)(::mlir::Operation *);
    bool (*inferInputsShapeEquality)(::mlir::Operation *, int, int);
    bool (*inferOutputsShapeEquality)(::mlir::Operation *, int, int);
    bool (*inferInputOutputShapeEquality)(::mlir::Operation *, int, int);
    llvm::Optional<Value> (*inferEffectiveWorkloadShape)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{isFusibleWithOperand, isFusibleWithConsumer, inferInputsShapeEquality, inferOutputsShapeEquality, inferInputOutputShapeEquality, inferEffectiveWorkloadShape} {}

    static inline bool isFusibleWithOperand(::mlir::Operation *tablegen_opaque_val);
    static inline bool isFusibleWithConsumer(::mlir::Operation *tablegen_opaque_val);
    static inline bool inferInputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs);
    static inline bool inferOutputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs);
    static inline bool inferInputOutputShapeEquality(::mlir::Operation *tablegen_opaque_val, int input, int output);
    static inline llvm::Optional<Value> inferEffectiveWorkloadShape(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class InferFusibilityOpInterface : public ::mlir::OpInterface<InferFusibilityOpInterface, detail::InferFusibilityOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferFusibilityOpInterface, detail::InferFusibilityOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct InferFusibilityOpInterfaceTrait : public ::mlir::OpInterface<InferFusibilityOpInterface, detail::InferFusibilityOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    bool isFusibleWithOperand() {
      /// Returns whether this op can be fused with its operands
        return true;
    }
    bool isFusibleWithConsumer() {
      /// Return whether this op can be fused with its consumers
        return true;
    }
    bool inferInputsShapeEquality(int lhs, int rhs) {
      /// Return whether two inputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return inferShapeEquality(op->getOperand(lhs), op->getOperand(rhs));
    }
    bool inferOutputsShapeEquality(int lhs, int rhs) {
      /// Return whether two outputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return inferShapeEquality(op->getResult(lhs), op->getResult(rhs));
    }
    bool inferInputOutputShapeEquality(int input, int output) {
      /// Return whether the input and the output have the same shape.
        Operation *op = this->getOperation();
        assert(input >= 0 && output >= 0);
        return inferShapeEquality(op->getOperand(input), op->getResult(output));
    }
    llvm::Optional<Value> inferEffectiveWorkloadShape() {
      /// Return effective workload size if possible, otherwise None.
        return {};
    }
  };
  template <typename ConcreteOp>
  struct Trait : public InferFusibilityOpInterfaceTrait<ConcreteOp> {};
  bool isFusibleWithOperand();
  bool isFusibleWithConsumer();
  bool inferInputsShapeEquality(int lhs, int rhs);
  bool inferOutputsShapeEquality(int lhs, int rhs);
  bool inferInputOutputShapeEquality(int input, int output);
  llvm::Optional<Value> inferEffectiveWorkloadShape();

    // Returns whether the given values have the same static shape.
    static bool inferShapeEquality(Value first, Value second) {
      // If both lhs and rhs have static shapes, check them directly.
      auto first_ty = first.getType().dyn_cast<RankedTensorType>();
      auto second_ty = second.getType().dyn_cast<RankedTensorType>();
      if (!first_ty || !first_ty.hasStaticShape() ||
          !second_ty || !second_ty.hasStaticShape() ||
          first_ty.getRank() != second_ty.getRank()) {
        return false;
      }
      return first_ty.getShape() == second_ty.getShape();
    }
  
};
template<typename ConcreteOp>
bool detail::InferFusibilityOpInterfaceInterfaceTraits::Model<ConcreteOp>::isFusibleWithOperand(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isFusibleWithOperand();
}
template<typename ConcreteOp>
bool detail::InferFusibilityOpInterfaceInterfaceTraits::Model<ConcreteOp>::isFusibleWithConsumer(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isFusibleWithConsumer();
}
template<typename ConcreteOp>
bool detail::InferFusibilityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferInputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferInputsShapeEquality(lhs, rhs);
}
template<typename ConcreteOp>
bool detail::InferFusibilityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferOutputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferOutputsShapeEquality(lhs, rhs);
}
template<typename ConcreteOp>
bool detail::InferFusibilityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferInputOutputShapeEquality(::mlir::Operation *tablegen_opaque_val, int input, int output) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferInputOutputShapeEquality(input, output);
}
template<typename ConcreteOp>
llvm::Optional<Value> detail::InferFusibilityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferEffectiveWorkloadShape(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferEffectiveWorkloadShape();
}
