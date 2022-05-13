/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class InferShapedTypeOpInterface;
namespace detail {
struct InferShapedTypeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*inferReturnTypeComponents)(::mlir::MLIRContext*, ::mlir::Optional<::mlir::Location>, ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange, ::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>&);
    ::mlir::LogicalResult (*reifyReturnTypeShapes)(::mlir::Operation *, ::mlir::OpBuilder&, ::mlir::SmallVectorImpl<::mlir::Value>&);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{inferReturnTypeComponents, reifyReturnTypeShapes} {}

    static inline ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
    static inline ::mlir::LogicalResult reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::SmallVectorImpl<::mlir::Value>& reifiedReturnShapes);
  };
};
} // end namespace detail
class InferShapedTypeOpInterface : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct InferShapedTypeOpInterfaceTrait : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::LogicalResult reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::SmallVectorImpl<::mlir::Value>& reifiedReturnShapes) {
      return ::mlir::failure();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public InferShapedTypeOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
  ::mlir::LogicalResult reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::SmallVectorImpl<::mlir::Value>& reifiedReturnShapes);
};
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
  return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::SmallVectorImpl<::mlir::Value>& reifiedReturnShapes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).reifyReturnTypeShapes(builder, reifiedReturnShapes);
}
} // namespace mlir
namespace mlir {
class InferTypeOpInterface;
namespace detail {
struct InferTypeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*inferReturnTypes)(::mlir::MLIRContext *, ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type>&);
    bool (*isCompatibleReturnTypes)(::mlir::TypeRange, ::mlir::TypeRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{inferReturnTypes, isCompatibleReturnTypes} {}

    static inline ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
    static inline bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
};
} // end namespace detail
class InferTypeOpInterface : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct InferTypeOpInterfaceTrait : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
      /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyInferredResultTypes(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public InferTypeOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
};
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return ConcreteOp::inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
}
template<typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
}
} // namespace mlir
