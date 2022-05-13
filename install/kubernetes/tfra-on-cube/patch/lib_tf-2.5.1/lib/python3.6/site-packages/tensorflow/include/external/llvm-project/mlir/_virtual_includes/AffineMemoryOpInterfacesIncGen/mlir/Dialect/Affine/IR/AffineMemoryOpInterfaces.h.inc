/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class AffineMapAccessInterface;
namespace detail {
struct AffineMapAccessInterfaceInterfaceTraits {
  struct Concept {
    NamedAttribute (*getAffineMapAttrForMemRef)(::mlir::Operation *, Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getAffineMapAttrForMemRef} {}

    static inline NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref);
  };
};
} // end namespace detail
class AffineMapAccessInterface : public ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AffineMapAccessInterfaceTrait : public ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    NamedAttribute getAffineMapAttrForMemRef(Value memref) {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        assert(memref == op.getMemRef() &&
               "Expected memref argument to match memref operand");
        return {Identifier::get(op.getMapAttrName(), op.getContext()),
                op.getAffineMapAttr()};
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AffineMapAccessInterfaceTrait<ConcreteOp> {};
  NamedAttribute getAffineMapAttrForMemRef(Value memref);
};
template<typename ConcreteOp>
NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttrForMemRef(memref);
}
class AffineReadOpInterface;
namespace detail {
struct AffineReadOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*getMemRef)(::mlir::Operation *);
    MemRefType (*getMemRefType)(::mlir::Operation *);
    Operation::operand_range (*getMapOperands)(::mlir::Operation *);
    AffineMap (*getAffineMap)(::mlir::Operation *);
    Value (*getValue)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValue} {}

    static inline Value getMemRef(::mlir::Operation *tablegen_opaque_val);
    static inline MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val);
    static inline Value getValue(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class AffineReadOpInterface : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AffineReadOpInterfaceTrait : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value getMemRef() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
    }
    MemRefType getMemRefType() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
    }
    Operation::operand_range getMapOperands() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 1);
    }
    AffineMap getAffineMap() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
    }
    Value getValue() {
      return cast<ConcreteOp>(this->getOperation());
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AffineReadOpInterfaceTrait<ConcreteOp> {};
  Value getMemRef();
  MemRefType getMemRefType();
  Operation::operand_range getMapOperands();
  AffineMap getAffineMap();
  Value getValue();
};
template<typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRef(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
}
template<typename ConcreteOp>
MemRefType detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRefType(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
}
template<typename ConcreteOp>
Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
}
template<typename ConcreteOp>
AffineMap detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMap(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
}
template<typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValue(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValue();
}
class AffineWriteOpInterface;
namespace detail {
struct AffineWriteOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*getMemRef)(::mlir::Operation *);
    MemRefType (*getMemRefType)(::mlir::Operation *);
    Operation::operand_range (*getMapOperands)(::mlir::Operation *);
    AffineMap (*getAffineMap)(::mlir::Operation *);
    Value (*getValueToStore)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValueToStore} {}

    static inline Value getMemRef(::mlir::Operation *tablegen_opaque_val);
    static inline MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val);
    static inline Value getValueToStore(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class AffineWriteOpInterface : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AffineWriteOpInterfaceTrait : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value getMemRef() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
    }
    MemRefType getMemRefType() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
    }
    Operation::operand_range getMapOperands() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 2);
    }
    AffineMap getAffineMap() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
    }
    Value getValueToStore() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getStoredValOperandIndex());
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AffineWriteOpInterfaceTrait<ConcreteOp> {};
  Value getMemRef();
  MemRefType getMemRefType();
  Operation::operand_range getMapOperands();
  AffineMap getAffineMap();
  Value getValueToStore();
};
template<typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRef(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
}
template<typename ConcreteOp>
MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRefType(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
}
template<typename ConcreteOp>
Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
}
template<typename ConcreteOp>
AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMap(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
}
template<typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValueToStore(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValueToStore();
}
