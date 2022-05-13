/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {
class ContractionOpInterface;
namespace detail {
struct ContractionOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*isRowMajorMatmul)(::mlir::Operation *);
    bool (*isColumnMajorMatmul)(::mlir::Operation *);
    bool (*isRowMajorBatchMatmul)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{isRowMajorMatmul, isColumnMajorMatmul, isRowMajorBatchMatmul} {}

    static inline bool isRowMajorMatmul(::mlir::Operation *tablegen_opaque_val);
    static inline bool isColumnMajorMatmul(::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorBatchMatmul(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class ContractionOpInterface : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ContractionOpInterfaceTrait : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyContractionInterface(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public ContractionOpInterfaceTrait<ConcreteOp> {};
  bool isRowMajorMatmul();
  bool isColumnMajorMatmul();
  bool isRowMajorBatchMatmul();
};
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorMatmul(::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isColumnMajorMatmul(::mlir::Operation *tablegen_opaque_val) {
  return mlir::isColumnMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorBatchMatmul(::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorBatchMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps());
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class LinalgOp;
namespace detail {
struct LinalgOpInterfaceTraits {
  struct Concept {
    unsigned (*getNumPayloadInductionVariables)(::mlir::Operation *);
    unsigned (*getNumParallelLoops)(::mlir::Operation *);
    void (*getParallelDims)(::mlir::Operation *, SmallVectorImpl<AffineExpr> &);
    unsigned (*getNumReductionLoops)(::mlir::Operation *);
    void (*getReductionDims)(::mlir::Operation *, SmallVectorImpl<AffineExpr> &);
    unsigned (*getNumWindowLoops)(::mlir::Operation *);
    void (*getWindowDims)(::mlir::Operation *, SmallVectorImpl<AffineExpr> &);
    unsigned (*getNumLoops)(::mlir::Operation *);
    bool (*hasSingleReductionLoop)(::mlir::Operation *);
    ValueRange (*inputs)(::mlir::Operation *);
    unsigned (*getNumInputs)(::mlir::Operation *);
    ValueRange (*outputs)(::mlir::Operation *);
    unsigned (*getNumOutputs)(::mlir::Operation *);
    Value (*getInput)(::mlir::Operation *, unsigned);
    ShapedType (*getInputShapedType)(::mlir::Operation *, unsigned);
    Operation::operand_range (*getInputs)(::mlir::Operation *);
    MutableArrayRef<OpOperand> (*getInputOpOperands)(::mlir::Operation *);
    SmallVector<Value, 4> (*getInputBuffers)(::mlir::Operation *);
    unsigned (*getNumInputBuffers)(::mlir::Operation *);
    Value (*getInputBuffer)(::mlir::Operation *, unsigned);
    SmallVector<OpOperand*, 4> (*getInputBuffersOpOperands)(::mlir::Operation *);
    SmallVector<Value, 4> (*getInputTensors)(::mlir::Operation *);
    SmallVector<OpOperand*, 4> (*getInputTensorsOpOperands)(::mlir::Operation *);
    SmallVector<MemRefType, 4> (*getInputBufferTypes)(::mlir::Operation *);
    SmallVector<RankedTensorType, 4> (*getInputTensorTypes)(::mlir::Operation *);
    Value (*getOutput)(::mlir::Operation *, unsigned);
    ShapedType (*getOutputShapedType)(::mlir::Operation *, unsigned);
    Operation::operand_range (*getOutputs)(::mlir::Operation *);
    MutableArrayRef<OpOperand> (*getOutputOpOperands)(::mlir::Operation *);
    SmallVector<Value, 4> (*getOutputBuffers)(::mlir::Operation *);
    Value (*getOutputBuffer)(::mlir::Operation *, unsigned);
    SmallVector<OpOperand*, 4> (*getOutputBuffersOpOperands)(::mlir::Operation *);
    unsigned (*getNumOutputBuffers)(::mlir::Operation *);
    SmallVector<Value, 4> (*getOutputTensors)(::mlir::Operation *);
    SmallVector<OpOperand*, 4> (*getOutputTensorsOpOperands)(::mlir::Operation *);
    unsigned (*getNumOutputTensors)(::mlir::Operation *);
    SmallVector<MemRefType, 4> (*getOutputBufferTypes)(::mlir::Operation *);
    SmallVector<RankedTensorType, 4> (*getOutputTensorTypes)(::mlir::Operation *);
    bool (*payloadUsesValueFromOpOperand)(::mlir::Operation *, OpOperand *);
    bool (*payloadUsesValueFromInputOperandIndex)(::mlir::Operation *, unsigned);
    bool (*payloadUsesValueFromOutputOperandIndex)(::mlir::Operation *, unsigned);
    bool (*isInputTensor)(::mlir::Operation *, OpOperand *);
    bool (*isOutputTensor)(::mlir::Operation *, OpOperand *);
    bool (*isInitTensor)(::mlir::Operation *, OpOperand *);
    bool (*isIndexOfInitTensor)(::mlir::Operation *, unsigned);
    SmallVector<Value, 4> (*getInitTensors)(::mlir::Operation *);
    unsigned (*getNumInitTensors)(::mlir::Operation *);
    unsigned (*getNumShapedOperands)(::mlir::Operation *);
    Value (*getShapedOperand)(::mlir::Operation *, unsigned);
    Operation::operand_range (*getShapedOperands)(::mlir::Operation *);
    MutableArrayRef<OpOperand> (*getShapedOpOperands)(::mlir::Operation *);
    OpOperand&(*getShapedOpOperand)(::mlir::Operation *, unsigned);
    SmallVector<ShapedType, 4> (*getShapedOperandTypes)(::mlir::Operation *);
    ShapedType (*getShapedType)(::mlir::Operation *, unsigned);
    ArrayAttr (*iterator_types)(::mlir::Operation *);
    bool (*hasDynamicIndexingMaps)(::mlir::Operation *);
    LogicalResult (*verifyIndexingMapRequiredAttributes)(::mlir::Operation *);
    ArrayAttr (*indexing_maps)(::mlir::Operation *);
    SmallVector<AffineMap, 4> (*getIndexingMaps)(::mlir::Operation *);
    AffineMap (*getIndexingMap)(::mlir::Operation *, unsigned);
    AffineMap (*getInputIndexingMap)(::mlir::Operation *, unsigned);
    AffineMap (*getOutputIndexingMap)(::mlir::Operation *, unsigned);
    bool (*hasBufferSemantics)(::mlir::Operation *);
    bool (*hasTensorSemantics)(::mlir::Operation *);
    bool (*hasSparseSemantics)(::mlir::Operation *);
    std::string (*getLibraryCallName)(::mlir::Operation *);
    AffineMap (*getLoopsToShapesMap)(::mlir::Operation *);
    AffineMap (*getShapesToLoopsMap)(::mlir::Operation *);
    Optional<unsigned> (*getOperandDimPositionInLoopsToShapeMap)(::mlir::Operation *, unsigned, unsigned);
    Optional<unsigned> (*getInputValueDimPositionInLoopsToShapeMap)(::mlir::Operation *, unsigned, unsigned);
    Optional<unsigned> (*getResultValueDimPositionInLoopsToShapeMap)(::mlir::Operation *, unsigned, unsigned);
    SmallVector<int64_t, 8> (*getStaticShape)(::mlir::Operation *);
    Optional<SmallVector<int64_t, 4>> (*getStaticLoopRanges)(::mlir::Operation *);
    Operation *(*clone)(::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange);
    Operation *(*cloneWithMapper)(::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange, BlockAndValueMapping &);
    std::function<void(Block &, ValueRange)> (*getRegionBuilder)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getNumPayloadInductionVariables, getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumWindowLoops, getWindowDims, getNumLoops, hasSingleReductionLoop, inputs, getNumInputs, outputs, getNumOutputs, getInput, getInputShapedType, getInputs, getInputOpOperands, getInputBuffers, getNumInputBuffers, getInputBuffer, getInputBuffersOpOperands, getInputTensors, getInputTensorsOpOperands, getInputBufferTypes, getInputTensorTypes, getOutput, getOutputShapedType, getOutputs, getOutputOpOperands, getOutputBuffers, getOutputBuffer, getOutputBuffersOpOperands, getNumOutputBuffers, getOutputTensors, getOutputTensorsOpOperands, getNumOutputTensors, getOutputBufferTypes, getOutputTensorTypes, payloadUsesValueFromOpOperand, payloadUsesValueFromInputOperandIndex, payloadUsesValueFromOutputOperandIndex, isInputTensor, isOutputTensor, isInitTensor, isIndexOfInitTensor, getInitTensors, getNumInitTensors, getNumShapedOperands, getShapedOperand, getShapedOperands, getShapedOpOperands, getShapedOpOperand, getShapedOperandTypes, getShapedType, iterator_types, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, indexing_maps, getIndexingMaps, getIndexingMap, getInputIndexingMap, getOutputIndexingMap, hasBufferSemantics, hasTensorSemantics, hasSparseSemantics, getLibraryCallName, getLoopsToShapesMap, getShapesToLoopsMap, getOperandDimPositionInLoopsToShapeMap, getInputValueDimPositionInLoopsToShapeMap, getResultValueDimPositionInLoopsToShapeMap, getStaticShape, getStaticLoopRanges, clone, cloneWithMapper, getRegionBuilder} {}

    static inline unsigned getNumPayloadInductionVariables(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumParallelLoops(::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res);
    static inline unsigned getNumReductionLoops(::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res);
    static inline unsigned getNumWindowLoops(::mlir::Operation *tablegen_opaque_val);
    static inline void getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res);
    static inline unsigned getNumLoops(::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val);
    static inline ValueRange inputs(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumInputs(::mlir::Operation *tablegen_opaque_val);
    static inline ValueRange outputs(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumOutputs(::mlir::Operation *tablegen_opaque_val);
    static inline Value getInput(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline ShapedType getInputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline Operation::operand_range getInputs(::mlir::Operation *tablegen_opaque_val);
    static inline MutableArrayRef<OpOperand> getInputOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Value, 4> getInputBuffers(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumInputBuffers(::mlir::Operation *tablegen_opaque_val);
    static inline Value getInputBuffer(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline SmallVector<OpOperand*, 4> getInputBuffersOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Value, 4> getInputTensors(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<OpOperand*, 4> getInputTensorsOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<MemRefType, 4> getInputBufferTypes(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<RankedTensorType, 4> getInputTensorTypes(::mlir::Operation *tablegen_opaque_val);
    static inline Value getOutput(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline ShapedType getOutputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline Operation::operand_range getOutputs(::mlir::Operation *tablegen_opaque_val);
    static inline MutableArrayRef<OpOperand> getOutputOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Value, 4> getOutputBuffers(::mlir::Operation *tablegen_opaque_val);
    static inline Value getOutputBuffer(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline SmallVector<OpOperand*, 4> getOutputBuffersOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumOutputBuffers(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Value, 4> getOutputTensors(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<OpOperand*, 4> getOutputTensorsOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumOutputTensors(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<MemRefType, 4> getOutputBufferTypes(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<RankedTensorType, 4> getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val);
    static inline bool payloadUsesValueFromOpOperand(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool payloadUsesValueFromInputOperandIndex(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline bool payloadUsesValueFromOutputOperandIndex(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline bool isInputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isOutputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isIndexOfInitTensor(::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline SmallVector<Value, 4> getInitTensors(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumInitTensors(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getNumShapedOperands(::mlir::Operation *tablegen_opaque_val);
    static inline Value getShapedOperand(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline Operation::operand_range getShapedOperands(::mlir::Operation *tablegen_opaque_val);
    static inline MutableArrayRef<OpOperand> getShapedOpOperands(::mlir::Operation *tablegen_opaque_val);
    static inline OpOperand&getShapedOpOperand(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline SmallVector<ShapedType, 4> getShapedOperandTypes(::mlir::Operation *tablegen_opaque_val);
    static inline ShapedType getShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline ArrayAttr iterator_types(::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr indexing_maps(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap, 4> getIndexingMaps(::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline AffineMap getInputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline AffineMap getOutputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i);
    static inline bool hasBufferSemantics(::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSparseSemantics(::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val);
    static inline Optional<unsigned> getOperandDimPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val, unsigned operandIdx, unsigned dim);
    static inline Optional<unsigned> getInputValueDimPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val, unsigned inputIdx, unsigned dim);
    static inline Optional<unsigned> getResultValueDimPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val, unsigned resultIdx, unsigned dim);
    static inline SmallVector<int64_t, 8> getStaticShape(::mlir::Operation *tablegen_opaque_val);
    static inline Optional<SmallVector<int64_t, 4>> getStaticLoopRanges(::mlir::Operation *tablegen_opaque_val);
    static inline Operation *clone(::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline Operation *cloneWithMapper(::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm);
    static inline std::function<void(Block &, ValueRange)> getRegionBuilder();
  };
};
} // end namespace detail
class LinalgOp : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct LinalgOpTrait : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::Trait<ConcreteOp> {
    unsigned getNumParallelLoops() {
      return getNumIterators(getParallelIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getParallelDims(SmallVectorImpl<AffineExpr> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getParallelIteratorTypeName(), res);
    }
    unsigned getNumReductionLoops() {
      return getNumIterators(getReductionIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getReductionDims(SmallVectorImpl<AffineExpr> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getReductionIteratorTypeName(), res);
    }
    unsigned getNumWindowLoops() {
      return getNumIterators(getWindowIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getWindowDims(SmallVectorImpl<AffineExpr> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)).getOperation(), getWindowIteratorTypeName(), res);
    }
    unsigned getNumLoops() {
      return getNumIterators((*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    bool hasSingleReductionLoop() {
      auto iters = (*static_cast<ConcreteOp *>(this)).iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
    }
    unsigned getNumInputs() {
      return (*static_cast<ConcreteOp *>(this)).inputs().size();
    }
    unsigned getNumOutputs() {
      return (*static_cast<ConcreteOp *>(this)).outputs().size();
    }
    Value getInput(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumInputs());
        return this->getOperation()->getOperand(i);
    }
    ShapedType getInputShapedType(unsigned i) {
      return getInput(i).getType().template cast<ShapedType>();
    }
    Operation::operand_range getInputs() {
      auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + (*static_cast<ConcreteOp *>(this)).getNumInputs()};
    }
    MutableArrayRef<OpOperand> getInputOpOperands() {
      return this->getOperation()->getOpOperands().take_front(getNumInputs());
    }
    SmallVector<Value, 4> getInputBuffers() {
      return llvm::to_vector<4>(llvm::make_filter_range(
          getInputs(), [](Value in){ return in.getType().template isa<MemRefType>(); }));
    }
    unsigned getNumInputBuffers() {
      return (*static_cast<ConcreteOp *>(this)).getInputBuffers().size();
    }
    Value getInputBuffer(unsigned index) {
      assert(index < getNumInputBuffers());
        return getInputBuffers()[index];
    }
    SmallVector<OpOperand*, 4> getInputBuffersOpOperands() {
      SmallVector<OpOperand*, 4> res;
        res.reserve(getNumInputs());
        for (OpOperand &o : getInputOpOperands())
          if (o.get().getType().isa<MemRefType>())
            res.push_back(&o);
        return res;
    }
    SmallVector<Value, 4> getInputTensors() {
      return llvm::to_vector<4>(llvm::make_filter_range(
          getInputs(),
          [](Value in){ return in.getType().template isa<RankedTensorType>(); }));
    }
    SmallVector<OpOperand*, 4> getInputTensorsOpOperands() {
      SmallVector<OpOperand*, 4> res;
        res.reserve(getNumInputs());
        for (OpOperand &o : getInputOpOperands())
          if (o.get().getType().isa<RankedTensorType>())
            res.push_back(&o);
        return res;
    }
    SmallVector<MemRefType, 4> getInputBufferTypes() {
      return llvm::to_vector<4>(
          llvm::map_range(
            llvm::make_filter_range(
              ValueRange(getInputs()).getTypes(),
              [](Type in){ return in.isa<MemRefType>(); }),
            [](Type in){ return in.cast<MemRefType>(); }));
    }
    SmallVector<RankedTensorType, 4> getInputTensorTypes() {
      return llvm::to_vector<4>(
          llvm::map_range(
            llvm::make_filter_range(
              ValueRange(getInputs()).getTypes(),
              [](Type in){ return in.isa<RankedTensorType>(); }),
            [](Type in){ return in.cast<RankedTensorType>(); }));
    }
    Value getOutput(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumOutputs());
        return this->getOperation()->getOperand(i + (*static_cast<ConcreteOp *>(this)).getNumInputs());
    }
    ShapedType getOutputShapedType(unsigned i) {
      return getOutput(i).getType().template cast<ShapedType>();
    }
    Operation::operand_range getOutputs() {
      auto start =
          this->getOperation()->getOperands().begin() + (*static_cast<ConcreteOp *>(this)).getNumInputs();
        return {start, start + (*static_cast<ConcreteOp *>(this)).getNumOutputs()};
    }
    MutableArrayRef<OpOperand> getOutputOpOperands() {
      return this->getOperation()->getOpOperands().slice(
          getNumInputs(), getNumOutputs());
    }
    SmallVector<Value, 4> getOutputBuffers() {
      return llvm::to_vector<4>(llvm::make_filter_range(
          getOutputs(), [](Value in){ return in.getType().template isa<MemRefType>(); }));
    }
    Value getOutputBuffer(unsigned index) {
      assert(index < getNumOutputBuffers());
        return getOutputBuffers()[index];
    }
    SmallVector<OpOperand*, 4> getOutputBuffersOpOperands() {
      SmallVector<OpOperand*, 4> res;
        res.reserve(getNumOutputs());
        for (OpOperand &o : getOutputOpOperands())
          if (o.get().getType().isa<MemRefType>())
            res.push_back(&o);
        return res;
    }
    unsigned getNumOutputBuffers() {
      return (*static_cast<ConcreteOp *>(this)).getOutputBuffers().size();
    }
    SmallVector<Value, 4> getOutputTensors() {
      return llvm::to_vector<4>(llvm::make_filter_range(
          getOutputs(),
          [](Value in){ return in.getType().template isa<RankedTensorType>(); }));
    }
    SmallVector<OpOperand*, 4> getOutputTensorsOpOperands() {
      SmallVector<OpOperand*, 4> res;
        res.reserve(getNumOutputs());
        for (OpOperand &o : getOutputOpOperands())
          if (o.get().getType().isa<RankedTensorType>())
            res.push_back(&o);
        return res;
    }
    unsigned getNumOutputTensors() {
      return (*static_cast<ConcreteOp *>(this)).getOutputTensors().size();
    }
    SmallVector<MemRefType, 4> getOutputBufferTypes() {
      return llvm::to_vector<4>(
          llvm::map_range(
            llvm::make_filter_range(
              ValueRange(getOutputs()).getTypes(),
              [](Type in){ return in.isa<MemRefType>(); }),
            [](Type in){ return in.cast<MemRefType>(); }));
    }
    SmallVector<RankedTensorType, 4> getOutputTensorTypes() {
      return llvm::to_vector<4>(
          llvm::map_range(
            llvm::make_filter_range(
              ValueRange(getOutputs()).getTypes(),
              [](Type in){ return in.isa<RankedTensorType>(); }),
            [](Type in){ return in.cast<RankedTensorType>(); }));
    }
    bool payloadUsesValueFromOpOperand(OpOperand * opOperand) {
      unsigned bbArgNumber =
          (*static_cast<ConcreteOp *>(this)).getNumPayloadInductionVariables() + opOperand->getOperandNumber();
        // Safeguard against the named linalg ops that are manually defined and
        // that only support buffer semantics: we should not be there.
        // Such ops have an empty regionBuilder and are not constructed with a
        // region for now. In the future they are slated to disappear.
        assert(this->getOperation()->getNumRegions() == 1 && "unexpected "
               "missing region (calling `payloadUsesValueFromOpOperand` on "
               "manually defined named Linalg op?)");
        Block &block = this->getOperation()->getRegion(0).front();
        // Init tensors have uses.
        return !block.getArgument(bbArgNumber).use_empty();
    }
    bool payloadUsesValueFromInputOperandIndex(unsigned index) {
      return payloadUsesValueFromOpOperand(&getInputOpOperands()[index]);
    }
    bool payloadUsesValueFromOutputOperandIndex(unsigned index) {
      return payloadUsesValueFromOpOperand(&getOutputOpOperands()[index]);
    }
    bool isInputTensor(OpOperand * opOperand) {
      if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return true;
        return false;
    }
    bool isOutputTensor(OpOperand * opOperand) {
      if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return true;
        return false;
    }
    bool isInitTensor(OpOperand * opOperand) {
      if (!(*static_cast<ConcreteOp *>(this)).isOutputTensor(opOperand))
          return false;
        return payloadUsesValueFromOpOperand(opOperand);
    }
    bool isIndexOfInitTensor(unsigned index) {
      assert(index < getNumOutputs());
        return isInitTensor(
          &this->getOperation()->getOpOperands()[(*static_cast<ConcreteOp *>(this)).getNumInputs() + index]);
    }
    SmallVector<Value, 4> getInitTensors() {
      auto start =
          this->getOperation()->getOpOperands().begin() + (*static_cast<ConcreteOp *>(this)).getNumInputs();
        return llvm::to_vector<4>(
          llvm::map_range(
            llvm::make_filter_range(
              llvm::make_range(start, start + (*static_cast<ConcreteOp *>(this)).getNumOutputs()),
              [&](OpOperand &opOperand) {
                return (*static_cast<ConcreteOp *>(this)).isInitTensor(&opOperand);
              }),
              [&](OpOperand &opOperand) {
                return opOperand.get();
              }));
    }
    unsigned getNumInitTensors() {
      return getInitTensors().size();
    }
    unsigned getNumShapedOperands() {
      return (*static_cast<ConcreteOp *>(this)).getNumInputs() + (*static_cast<ConcreteOp *>(this)).getNumOutputs();
    }
    Value getShapedOperand(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumShapedOperands());
        return this->getOperation()->getOperand(i);
    }
    Operation::operand_range getShapedOperands() {
      auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + getNumShapedOperands()};
    }
    MutableArrayRef<OpOperand> getShapedOpOperands() {
      return this->getOperation()->getOpOperands().take_front(
          getNumShapedOperands());
    }
    OpOperand&getShapedOpOperand(unsigned i) {
      return *(this->getShapedOpOperands().begin() + i);
    }
    SmallVector<ShapedType, 4> getShapedOperandTypes() {
      return llvm::to_vector<4>(
          llvm::map_range(
            getShapedOperands(),
            [](Value v) { return v.getType().cast<ShapedType>(); }));
    }
    ShapedType getShapedType(unsigned i) {
      return (*static_cast<ConcreteOp *>(this)).getShapedOperand(i).getType().template cast<ShapedType>();
    }
    ArrayAttr iterator_types() {
      return (*static_cast<ConcreteOp *>(this)).iterator_types();
    }
    bool hasDynamicIndexingMaps() {
      return false;
    }
    LogicalResult verifyIndexingMapRequiredAttributes() {
      return success();
    }
    SmallVector<AffineMap, 4> getIndexingMaps() {
      return llvm::to_vector<4>(
          (*static_cast<ConcreteOp *>(this)).indexing_maps().template getAsValueRange<AffineMapAttr>());
    }
    AffineMap getIndexingMap(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumShapedOperands());
        return getIndexingMaps()[i];
    }
    AffineMap getInputIndexingMap(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumInputs());
        return getIndexingMaps()[i];
    }
    AffineMap getOutputIndexingMap(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumOutputs());
        return getIndexingMaps()[i + (*static_cast<ConcreteOp *>(this)).getNumInputs()];
    }
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(getShapedOperands(), [](Value v) {
            return v.getType().template isa<MemRefType>(); });
    }
    bool hasTensorSemantics() {
      return llvm::all_of(getShapedOperands(), [](Value v) {
          return v.getType().template isa<RankedTensorType>();
        });
    }
    bool hasSparseSemantics() {
      return (*static_cast<ConcreteOp *>(this))->getAttr(getSparseAttrName()).
          template dyn_cast_or_null<ArrayAttr>() != nullptr;
    }
    std::string getLibraryCallName() {
      return (*static_cast<ConcreteOp *>(this)).getLibraryCallName();
    }
    AffineMap getLoopsToShapesMap() {
      auto r = (*static_cast<ConcreteOp *>(this)).indexing_maps().template getAsRange<AffineMapAttr>();
        auto maps = llvm::to_vector<8>(
            llvm::map_range(r, [](AffineMapAttr a) { return a.getValue(); }));
        return concatAffineMaps(maps);
    }
    AffineMap getShapesToLoopsMap() {
      return inversePermutation(getLoopsToShapesMap());
    }
    Optional<unsigned> getOperandDimPositionInLoopsToShapeMap(unsigned operandIdx, unsigned dim) {
      unsigned pos = 0;
        for (auto type : llvm::enumerate(getShapedOperandTypes())) {
          if (type.index() == operandIdx) return pos + dim;
          pos += type.value().getRank();
        }
        return {};
    }
    Optional<unsigned> getInputValueDimPositionInLoopsToShapeMap(unsigned inputIdx, unsigned dim) {
      if (inputIdx >= getNumInputs()) return {};
        return getOperandDimPositionInLoopsToShapeMap(inputIdx, dim);
    }
    Optional<unsigned> getResultValueDimPositionInLoopsToShapeMap(unsigned resultIdx, unsigned dim) {
      if (resultIdx >= getNumOutputs()) return {};
        return getOperandDimPositionInLoopsToShapeMap(
            getNumInputs() + resultIdx, dim);
    }
    SmallVector<int64_t, 8> getStaticShape() {
      SmallVector<int64_t, 8> res;
        for (Value v : getShapedOperands()) {
          auto shape = v.getType().cast<ShapedType>().getShape();
          res.append(shape.begin(), shape.end());
        }
        return res;
    }
    Optional<SmallVector<int64_t, 4>> getStaticLoopRanges() {
      SmallVector<int64_t, 8> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        if (!invertedMap)
          return {};
        return invertedMap.compose(viewSizes);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyStructuredOpInterface(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public LinalgOpTrait<ConcreteOp> {};
  unsigned getNumPayloadInductionVariables();
  unsigned getNumParallelLoops();
  void getParallelDims(SmallVectorImpl<AffineExpr> & res);
  unsigned getNumReductionLoops();
  void getReductionDims(SmallVectorImpl<AffineExpr> & res);
  unsigned getNumWindowLoops();
  void getWindowDims(SmallVectorImpl<AffineExpr> & res);
  unsigned getNumLoops();
  bool hasSingleReductionLoop();
  ValueRange inputs();
  unsigned getNumInputs();
  ValueRange outputs();
  unsigned getNumOutputs();
  Value getInput(unsigned i);
  ShapedType getInputShapedType(unsigned i);
  Operation::operand_range getInputs();
  MutableArrayRef<OpOperand> getInputOpOperands();
  SmallVector<Value, 4> getInputBuffers();
  unsigned getNumInputBuffers();
  Value getInputBuffer(unsigned index);
  SmallVector<OpOperand*, 4> getInputBuffersOpOperands();
  SmallVector<Value, 4> getInputTensors();
  SmallVector<OpOperand*, 4> getInputTensorsOpOperands();
  SmallVector<MemRefType, 4> getInputBufferTypes();
  SmallVector<RankedTensorType, 4> getInputTensorTypes();
  Value getOutput(unsigned i);
  ShapedType getOutputShapedType(unsigned i);
  Operation::operand_range getOutputs();
  MutableArrayRef<OpOperand> getOutputOpOperands();
  SmallVector<Value, 4> getOutputBuffers();
  Value getOutputBuffer(unsigned index);
  SmallVector<OpOperand*, 4> getOutputBuffersOpOperands();
  unsigned getNumOutputBuffers();
  SmallVector<Value, 4> getOutputTensors();
  SmallVector<OpOperand*, 4> getOutputTensorsOpOperands();
  unsigned getNumOutputTensors();
  SmallVector<MemRefType, 4> getOutputBufferTypes();
  SmallVector<RankedTensorType, 4> getOutputTensorTypes();
  bool payloadUsesValueFromOpOperand(OpOperand * opOperand);
  bool payloadUsesValueFromInputOperandIndex(unsigned index);
  bool payloadUsesValueFromOutputOperandIndex(unsigned index);
  bool isInputTensor(OpOperand * opOperand);
  bool isOutputTensor(OpOperand * opOperand);
  bool isInitTensor(OpOperand * opOperand);
  bool isIndexOfInitTensor(unsigned index);
  SmallVector<Value, 4> getInitTensors();
  unsigned getNumInitTensors();
  unsigned getNumShapedOperands();
  Value getShapedOperand(unsigned i);
  Operation::operand_range getShapedOperands();
  MutableArrayRef<OpOperand> getShapedOpOperands();
  OpOperand&getShapedOpOperand(unsigned i);
  SmallVector<ShapedType, 4> getShapedOperandTypes();
  ShapedType getShapedType(unsigned i);
  ArrayAttr iterator_types();
  bool hasDynamicIndexingMaps();
  LogicalResult verifyIndexingMapRequiredAttributes();
  ArrayAttr indexing_maps();
  SmallVector<AffineMap, 4> getIndexingMaps();
  AffineMap getIndexingMap(unsigned i);
  AffineMap getInputIndexingMap(unsigned i);
  AffineMap getOutputIndexingMap(unsigned i);
  bool hasBufferSemantics();
  bool hasTensorSemantics();
  bool hasSparseSemantics();
  std::string getLibraryCallName();
  AffineMap getLoopsToShapesMap();
  AffineMap getShapesToLoopsMap();
  Optional<unsigned> getOperandDimPositionInLoopsToShapeMap(unsigned operandIdx, unsigned dim);
  Optional<unsigned> getInputValueDimPositionInLoopsToShapeMap(unsigned inputIdx, unsigned dim);
  Optional<unsigned> getResultValueDimPositionInLoopsToShapeMap(unsigned resultIdx, unsigned dim);
  SmallVector<int64_t, 8> getStaticShape();
  Optional<SmallVector<int64_t, 4>> getStaticLoopRanges();
  Operation *clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
  Operation *cloneWithMapper(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm);
  std::function<void(Block &, ValueRange)> getRegionBuilder();

    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<Value, 4> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Returns all the operands past the inputs, output_buffers and
    /// init_tensors operands. Asserts that these operands are value types to
    /// allow transformations like tiling to just use the values when cloning
    /// `linalgOp`.
    Operation::operand_range getAssumedNonShapedOperands() {
      Operation::operand_range res{
        getOperation()->getOperands().begin() + getNumShapedOperands(),
        getOperation()->getOperands().end()};
      for (Type t : TypeRange{res}) {
        (void)t;
        assert((t.isSignlessIntOrIndexOrFloat() || t.template isa<VectorType>())
               &&"expected scalar or vector type");
      }
      return res;
    }

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    Optional<Value> inferResultDimFromInputShapes
      (OpBuilder &b, Location loc, unsigned resultIdx, unsigned im);

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = (*this)->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  
};
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumPayloadInductionVariables(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumPayloadInductionVariables();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumParallelLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getParallelDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumReductionLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumWindowLoops(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumWindowLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getWindowDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumLoops(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumLoops();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasSingleReductionLoop();
}
template<typename ConcreteOp>
ValueRange detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::inputs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inputs();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumInputs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs();
}
template<typename ConcreteOp>
ValueRange detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::outputs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).outputs();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumOutputs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOutputs();
}
template<typename ConcreteOp>
Value detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInput(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInput(i);
}
template<typename ConcreteOp>
ShapedType detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputShapedType(i);
}
template<typename ConcreteOp>
Operation::operand_range detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputs();
}
template<typename ConcreteOp>
MutableArrayRef<OpOperand> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputOpOperands();
}
template<typename ConcreteOp>
SmallVector<Value, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputBuffers(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputBuffers();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumInputBuffers(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputBuffers();
}
template<typename ConcreteOp>
Value detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputBuffer(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputBuffer(index);
}
template<typename ConcreteOp>
SmallVector<OpOperand*, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputBuffersOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputBuffersOpOperands();
}
template<typename ConcreteOp>
SmallVector<Value, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputTensors(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputTensors();
}
template<typename ConcreteOp>
SmallVector<OpOperand*, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputTensorsOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputTensorsOpOperands();
}
template<typename ConcreteOp>
SmallVector<MemRefType, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputBufferTypes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputBufferTypes();
}
template<typename ConcreteOp>
SmallVector<RankedTensorType, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputTensorTypes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputTensorTypes();
}
template<typename ConcreteOp>
Value detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutput(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutput(i);
}
template<typename ConcreteOp>
ShapedType detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputShapedType(i);
}
template<typename ConcreteOp>
Operation::operand_range detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputs();
}
template<typename ConcreteOp>
MutableArrayRef<OpOperand> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputOpOperands();
}
template<typename ConcreteOp>
SmallVector<Value, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputBuffers(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBuffers();
}
template<typename ConcreteOp>
Value detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputBuffer(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBuffer(index);
}
template<typename ConcreteOp>
SmallVector<OpOperand*, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputBuffersOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBuffersOpOperands();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumOutputBuffers(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOutputBuffers();
}
template<typename ConcreteOp>
SmallVector<Value, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputTensors(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensors();
}
template<typename ConcreteOp>
SmallVector<OpOperand*, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputTensorsOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorsOpOperands();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumOutputTensors(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOutputTensors();
}
template<typename ConcreteOp>
SmallVector<MemRefType, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputBufferTypes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBufferTypes();
}
template<typename ConcreteOp>
SmallVector<RankedTensorType, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorTypes();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::payloadUsesValueFromOpOperand(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).payloadUsesValueFromOpOperand(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::payloadUsesValueFromInputOperandIndex(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).payloadUsesValueFromInputOperandIndex(index);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::payloadUsesValueFromOutputOperandIndex(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).payloadUsesValueFromOutputOperandIndex(index);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isInputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isInputTensor(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isOutputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOutputTensor(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isInitTensor(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isIndexOfInitTensor(::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isIndexOfInitTensor(index);
}
template<typename ConcreteOp>
SmallVector<Value, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInitTensors(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInitTensors();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumInitTensors(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInitTensors();
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumShapedOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumShapedOperands();
}
template<typename ConcreteOp>
Value detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapedOperand(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedOperand(i);
}
template<typename ConcreteOp>
Operation::operand_range detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapedOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedOperands();
}
template<typename ConcreteOp>
MutableArrayRef<OpOperand> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapedOpOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedOpOperands();
}
template<typename ConcreteOp>
OpOperand&detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapedOpOperand(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedOpOperand(i);
}
template<typename ConcreteOp>
SmallVector<ShapedType, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapedOperandTypes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedOperandTypes();
}
template<typename ConcreteOp>
ShapedType detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedType(i);
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::iterator_types(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasDynamicIndexingMaps();
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyIndexingMapRequiredAttributes();
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::indexing_maps(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps();
}
template<typename ConcreteOp>
SmallVector<AffineMap, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMaps(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps();
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMap(i);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputIndexingMap(i);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputIndexingMap(i);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasBufferSemantics();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasTensorSemantics();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasSparseSemantics(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasSparseSemantics();
}
template<typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getLibraryCallName(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopsToShapesMap();
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapesToLoopsMap();
}
template<typename ConcreteOp>
Optional<unsigned> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOperandDimPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val, unsigned operandIdx, unsigned dim) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperandDimPositionInLoopsToShapeMap(operandIdx, dim);
}
template<typename ConcreteOp>
Optional<unsigned> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputValueDimPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val, unsigned inputIdx, unsigned dim) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputValueDimPositionInLoopsToShapeMap(inputIdx, dim);
}
template<typename ConcreteOp>
Optional<unsigned> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getResultValueDimPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val, unsigned resultIdx, unsigned dim) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultValueDimPositionInLoopsToShapeMap(resultIdx, dim);
}
template<typename ConcreteOp>
SmallVector<int64_t, 8> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getStaticShape(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticShape();
}
template<typename ConcreteOp>
Optional<SmallVector<int64_t, 4>> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getStaticLoopRanges(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticLoopRanges();
}
template<typename ConcreteOp>
Operation *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::clone(::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  BlockAndValueMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (Region &r : (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.createOperation(state);
}
template<typename ConcreteOp>
Operation *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::cloneWithMapper(::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm) {
  OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (Region &r : (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.createOperation(state);
}
template<typename ConcreteOp>
std::function<void(Block &, ValueRange)> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionBuilder() {
  return ConcreteOp::getRegionBuilder();
}
} // namespace linalg
} // namespace mlir
