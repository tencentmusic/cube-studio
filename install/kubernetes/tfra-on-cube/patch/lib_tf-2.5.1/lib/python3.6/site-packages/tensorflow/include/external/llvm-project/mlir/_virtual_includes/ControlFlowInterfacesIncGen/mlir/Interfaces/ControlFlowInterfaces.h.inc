/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class BranchOpInterface;
namespace detail {
struct BranchOpInterfaceInterfaceTraits {
  struct Concept {
    Optional<MutableOperandRange> (*getMutableSuccessorOperands)(::mlir::Operation *, unsigned);
    Optional<OperandRange> (*getSuccessorOperands)(::mlir::Operation *, unsigned);
    Optional<BlockArgument> (*getSuccessorBlockArgument)(::mlir::Operation *, unsigned);
    Block *(*getSuccessorForOperands)(::mlir::Operation *, ArrayRef<Attribute>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getMutableSuccessorOperands, getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands} {}

    static inline Optional<MutableOperandRange> getMutableSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline Optional<OperandRange> getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline Optional<BlockArgument> getSuccessorBlockArgument(::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline Block *getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ArrayRef<Attribute> operands);
  };
};
} // end namespace detail
class BranchOpInterface : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct BranchOpInterfaceTrait : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Optional<OperandRange> getSuccessorOperands(unsigned index) {
      ConcreteOp *op = static_cast<ConcreteOp *>(this);
        auto operands = op->getMutableSuccessorOperands(index);
        return operands ? Optional<OperandRange>(*operands) : llvm::None;
    }
    Block *getSuccessorForOperands(ArrayRef<Attribute> operands) {
      return nullptr;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      auto concreteOp = cast<ConcreteOpType>(op);
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      Optional<OperandRange> operands = concreteOp.getSuccessorOperands(i);
      if (failed(detail::verifyBranchSuccessorOperands(op, i, operands)))
        return failure();
    }
    return success();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public BranchOpInterfaceTrait<ConcreteOp> {};
  Optional<MutableOperandRange> getMutableSuccessorOperands(unsigned index);
  Optional<OperandRange> getSuccessorOperands(unsigned index);
  Optional<BlockArgument> getSuccessorBlockArgument(unsigned operandIndex);
  Block *getSuccessorForOperands(ArrayRef<Attribute> operands);
};
template<typename ConcreteOp>
Optional<MutableOperandRange> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMutableSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
}
template<typename ConcreteOp>
Optional<OperandRange> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
}
template<typename ConcreteOp>
Optional<BlockArgument> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorBlockArgument(::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  Operation *opaqueOp = (llvm::cast<ConcreteOp>(tablegen_opaque_val));
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (Optional<BlockArgument> arg = detail::getBranchSuccessorArgument(
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return llvm::None;
}
template<typename ConcreteOp>
Block *detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ArrayRef<Attribute> operands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorForOperands(operands);
}
} // namespace mlir
namespace mlir {
class RegionBranchOpInterface;
namespace detail {
struct RegionBranchOpInterfaceInterfaceTraits {
  struct Concept {
    OperandRange (*getSuccessorEntryOperands)(::mlir::Operation *, unsigned);
    void (*getSuccessorRegions)(::mlir::Operation *, Optional<unsigned>, ArrayRef<Attribute>, SmallVectorImpl<RegionSuccessor> &);
    void (*getNumRegionInvocations)(::mlir::Operation *, ArrayRef<Attribute>, SmallVectorImpl<int64_t> &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getSuccessorEntryOperands, getSuccessorRegions, getNumRegionInvocations} {}

    static inline OperandRange getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline void getSuccessorRegions(::mlir::Operation *tablegen_opaque_val, Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions);
    static inline void getNumRegionInvocations(::mlir::Operation *tablegen_opaque_val, ArrayRef<Attribute> operands, SmallVectorImpl<int64_t> & countPerRegion);
  };
};
} // end namespace detail
class RegionBranchOpInterface : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct RegionBranchOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    OperandRange getSuccessorEntryOperands(unsigned index) {
      auto operandEnd = this->getOperation()->operand_end();
        return OperandRange(operandEnd, operandEnd);
    }
    void getNumRegionInvocations(ArrayRef<Attribute> operands, SmallVectorImpl<int64_t> & countPerRegion) {
      unsigned numRegions = this->getOperation()->getNumRegions();
        assert(countPerRegion.empty());
        countPerRegion.resize(numRegions, kUnknownNumRegionInvocations);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      static_assert(!ConcreteOpType::template hasTrait<OpTrait::ZeroRegion>(),
                  "expected operation to have non-zero regions");
    return success();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public RegionBranchOpInterfaceTrait<ConcreteOp> {};
  OperandRange getSuccessorEntryOperands(unsigned index);
  void getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions);
  void getNumRegionInvocations(ArrayRef<Attribute> operands, SmallVectorImpl<int64_t> & countPerRegion);

    /// Convenience helper in case none of the operands is known.
    void getSuccessorRegions(Optional<unsigned> index,
                             SmallVectorImpl<RegionSuccessor> &regions) {
       SmallVector<Attribute, 2> nullAttrs(getOperation()->getNumOperands());
       getSuccessorRegions(index, nullAttrs, regions);
    }

    /// Verify types along control flow edges described by this interface.
    static LogicalResult verifyTypes(Operation *op) {
      return detail::verifyTypesAlongControlFlowEdges(op);
    }
  
};
template<typename ConcreteOp>
OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorEntryOperands(index);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorRegions(::mlir::Operation *tablegen_opaque_val, Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(index, operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumRegionInvocations(::mlir::Operation *tablegen_opaque_val, ArrayRef<Attribute> operands, SmallVectorImpl<int64_t> & countPerRegion) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumRegionInvocations(operands, countPerRegion);
}
} // namespace mlir
