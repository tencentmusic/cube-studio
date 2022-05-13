/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool mlir::linalg::ContractionOpInterface::isRowMajorMatmul() {
      return getImpl()->isRowMajorMatmul(getOperation());
  }
bool mlir::linalg::ContractionOpInterface::isColumnMajorMatmul() {
      return getImpl()->isColumnMajorMatmul(getOperation());
  }
bool mlir::linalg::ContractionOpInterface::isRowMajorBatchMatmul() {
      return getImpl()->isRowMajorBatchMatmul(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumPayloadInductionVariables() {
      return getImpl()->getNumPayloadInductionVariables(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumParallelLoops() {
      return getImpl()->getNumParallelLoops(getOperation());
  }
void mlir::linalg::LinalgOp::getParallelDims(SmallVectorImpl<AffineExpr> & res) {
      return getImpl()->getParallelDims(getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumReductionLoops() {
      return getImpl()->getNumReductionLoops(getOperation());
  }
void mlir::linalg::LinalgOp::getReductionDims(SmallVectorImpl<AffineExpr> & res) {
      return getImpl()->getReductionDims(getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumWindowLoops() {
      return getImpl()->getNumWindowLoops(getOperation());
  }
void mlir::linalg::LinalgOp::getWindowDims(SmallVectorImpl<AffineExpr> & res) {
      return getImpl()->getWindowDims(getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumLoops() {
      return getImpl()->getNumLoops(getOperation());
  }
bool mlir::linalg::LinalgOp::hasSingleReductionLoop() {
      return getImpl()->hasSingleReductionLoop(getOperation());
  }
ValueRange mlir::linalg::LinalgOp::inputs() {
      return getImpl()->inputs(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumInputs() {
      return getImpl()->getNumInputs(getOperation());
  }
ValueRange mlir::linalg::LinalgOp::outputs() {
      return getImpl()->outputs(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumOutputs() {
      return getImpl()->getNumOutputs(getOperation());
  }
Value mlir::linalg::LinalgOp::getInput(unsigned i) {
      return getImpl()->getInput(getOperation(), i);
  }
ShapedType mlir::linalg::LinalgOp::getInputShapedType(unsigned i) {
      return getImpl()->getInputShapedType(getOperation(), i);
  }
Operation::operand_range mlir::linalg::LinalgOp::getInputs() {
      return getImpl()->getInputs(getOperation());
  }
MutableArrayRef<OpOperand> mlir::linalg::LinalgOp::getInputOpOperands() {
      return getImpl()->getInputOpOperands(getOperation());
  }
SmallVector<Value, 4> mlir::linalg::LinalgOp::getInputBuffers() {
      return getImpl()->getInputBuffers(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumInputBuffers() {
      return getImpl()->getNumInputBuffers(getOperation());
  }
Value mlir::linalg::LinalgOp::getInputBuffer(unsigned index) {
      return getImpl()->getInputBuffer(getOperation(), index);
  }
SmallVector<OpOperand*, 4> mlir::linalg::LinalgOp::getInputBuffersOpOperands() {
      return getImpl()->getInputBuffersOpOperands(getOperation());
  }
SmallVector<Value, 4> mlir::linalg::LinalgOp::getInputTensors() {
      return getImpl()->getInputTensors(getOperation());
  }
SmallVector<OpOperand*, 4> mlir::linalg::LinalgOp::getInputTensorsOpOperands() {
      return getImpl()->getInputTensorsOpOperands(getOperation());
  }
SmallVector<MemRefType, 4> mlir::linalg::LinalgOp::getInputBufferTypes() {
      return getImpl()->getInputBufferTypes(getOperation());
  }
SmallVector<RankedTensorType, 4> mlir::linalg::LinalgOp::getInputTensorTypes() {
      return getImpl()->getInputTensorTypes(getOperation());
  }
Value mlir::linalg::LinalgOp::getOutput(unsigned i) {
      return getImpl()->getOutput(getOperation(), i);
  }
ShapedType mlir::linalg::LinalgOp::getOutputShapedType(unsigned i) {
      return getImpl()->getOutputShapedType(getOperation(), i);
  }
Operation::operand_range mlir::linalg::LinalgOp::getOutputs() {
      return getImpl()->getOutputs(getOperation());
  }
MutableArrayRef<OpOperand> mlir::linalg::LinalgOp::getOutputOpOperands() {
      return getImpl()->getOutputOpOperands(getOperation());
  }
SmallVector<Value, 4> mlir::linalg::LinalgOp::getOutputBuffers() {
      return getImpl()->getOutputBuffers(getOperation());
  }
Value mlir::linalg::LinalgOp::getOutputBuffer(unsigned index) {
      return getImpl()->getOutputBuffer(getOperation(), index);
  }
SmallVector<OpOperand*, 4> mlir::linalg::LinalgOp::getOutputBuffersOpOperands() {
      return getImpl()->getOutputBuffersOpOperands(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumOutputBuffers() {
      return getImpl()->getNumOutputBuffers(getOperation());
  }
SmallVector<Value, 4> mlir::linalg::LinalgOp::getOutputTensors() {
      return getImpl()->getOutputTensors(getOperation());
  }
SmallVector<OpOperand*, 4> mlir::linalg::LinalgOp::getOutputTensorsOpOperands() {
      return getImpl()->getOutputTensorsOpOperands(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumOutputTensors() {
      return getImpl()->getNumOutputTensors(getOperation());
  }
SmallVector<MemRefType, 4> mlir::linalg::LinalgOp::getOutputBufferTypes() {
      return getImpl()->getOutputBufferTypes(getOperation());
  }
SmallVector<RankedTensorType, 4> mlir::linalg::LinalgOp::getOutputTensorTypes() {
      return getImpl()->getOutputTensorTypes(getOperation());
  }
bool mlir::linalg::LinalgOp::payloadUsesValueFromOpOperand(OpOperand * opOperand) {
      return getImpl()->payloadUsesValueFromOpOperand(getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::payloadUsesValueFromInputOperandIndex(unsigned index) {
      return getImpl()->payloadUsesValueFromInputOperandIndex(getOperation(), index);
  }
bool mlir::linalg::LinalgOp::payloadUsesValueFromOutputOperandIndex(unsigned index) {
      return getImpl()->payloadUsesValueFromOutputOperandIndex(getOperation(), index);
  }
bool mlir::linalg::LinalgOp::isInputTensor(OpOperand * opOperand) {
      return getImpl()->isInputTensor(getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isOutputTensor(OpOperand * opOperand) {
      return getImpl()->isOutputTensor(getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isInitTensor(OpOperand * opOperand) {
      return getImpl()->isInitTensor(getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isIndexOfInitTensor(unsigned index) {
      return getImpl()->isIndexOfInitTensor(getOperation(), index);
  }
SmallVector<Value, 4> mlir::linalg::LinalgOp::getInitTensors() {
      return getImpl()->getInitTensors(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumInitTensors() {
      return getImpl()->getNumInitTensors(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumShapedOperands() {
      return getImpl()->getNumShapedOperands(getOperation());
  }
Value mlir::linalg::LinalgOp::getShapedOperand(unsigned i) {
      return getImpl()->getShapedOperand(getOperation(), i);
  }
Operation::operand_range mlir::linalg::LinalgOp::getShapedOperands() {
      return getImpl()->getShapedOperands(getOperation());
  }
MutableArrayRef<OpOperand> mlir::linalg::LinalgOp::getShapedOpOperands() {
      return getImpl()->getShapedOpOperands(getOperation());
  }
OpOperand&mlir::linalg::LinalgOp::getShapedOpOperand(unsigned i) {
      return getImpl()->getShapedOpOperand(getOperation(), i);
  }
SmallVector<ShapedType, 4> mlir::linalg::LinalgOp::getShapedOperandTypes() {
      return getImpl()->getShapedOperandTypes(getOperation());
  }
ShapedType mlir::linalg::LinalgOp::getShapedType(unsigned i) {
      return getImpl()->getShapedType(getOperation(), i);
  }
ArrayAttr mlir::linalg::LinalgOp::iterator_types() {
      return getImpl()->iterator_types(getOperation());
  }
bool mlir::linalg::LinalgOp::hasDynamicIndexingMaps() {
      return getImpl()->hasDynamicIndexingMaps(getOperation());
  }
LogicalResult mlir::linalg::LinalgOp::verifyIndexingMapRequiredAttributes() {
      return getImpl()->verifyIndexingMapRequiredAttributes(getOperation());
  }
ArrayAttr mlir::linalg::LinalgOp::indexing_maps() {
      return getImpl()->indexing_maps(getOperation());
  }
SmallVector<AffineMap, 4> mlir::linalg::LinalgOp::getIndexingMaps() {
      return getImpl()->getIndexingMaps(getOperation());
  }
AffineMap mlir::linalg::LinalgOp::getIndexingMap(unsigned i) {
      return getImpl()->getIndexingMap(getOperation(), i);
  }
AffineMap mlir::linalg::LinalgOp::getInputIndexingMap(unsigned i) {
      return getImpl()->getInputIndexingMap(getOperation(), i);
  }
AffineMap mlir::linalg::LinalgOp::getOutputIndexingMap(unsigned i) {
      return getImpl()->getOutputIndexingMap(getOperation(), i);
  }
bool mlir::linalg::LinalgOp::hasBufferSemantics() {
      return getImpl()->hasBufferSemantics(getOperation());
  }
bool mlir::linalg::LinalgOp::hasTensorSemantics() {
      return getImpl()->hasTensorSemantics(getOperation());
  }
bool mlir::linalg::LinalgOp::hasSparseSemantics() {
      return getImpl()->hasSparseSemantics(getOperation());
  }
std::string mlir::linalg::LinalgOp::getLibraryCallName() {
      return getImpl()->getLibraryCallName(getOperation());
  }
AffineMap mlir::linalg::LinalgOp::getLoopsToShapesMap() {
      return getImpl()->getLoopsToShapesMap(getOperation());
  }
AffineMap mlir::linalg::LinalgOp::getShapesToLoopsMap() {
      return getImpl()->getShapesToLoopsMap(getOperation());
  }
Optional<unsigned> mlir::linalg::LinalgOp::getOperandDimPositionInLoopsToShapeMap(unsigned operandIdx, unsigned dim) {
      return getImpl()->getOperandDimPositionInLoopsToShapeMap(getOperation(), operandIdx, dim);
  }
Optional<unsigned> mlir::linalg::LinalgOp::getInputValueDimPositionInLoopsToShapeMap(unsigned inputIdx, unsigned dim) {
      return getImpl()->getInputValueDimPositionInLoopsToShapeMap(getOperation(), inputIdx, dim);
  }
Optional<unsigned> mlir::linalg::LinalgOp::getResultValueDimPositionInLoopsToShapeMap(unsigned resultIdx, unsigned dim) {
      return getImpl()->getResultValueDimPositionInLoopsToShapeMap(getOperation(), resultIdx, dim);
  }
SmallVector<int64_t, 8> mlir::linalg::LinalgOp::getStaticShape() {
      return getImpl()->getStaticShape(getOperation());
  }
Optional<SmallVector<int64_t, 4>> mlir::linalg::LinalgOp::getStaticLoopRanges() {
      return getImpl()->getStaticLoopRanges(getOperation());
  }
Operation *mlir::linalg::LinalgOp::clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
      return getImpl()->clone(getOperation(), b, loc, resultTypes, operands);
  }
Operation *mlir::linalg::LinalgOp::cloneWithMapper(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm) {
      return getImpl()->cloneWithMapper(getOperation(), b, loc, resultTypes, operands, bvm);
  }
std::function<void(Block &, ValueRange)> mlir::linalg::LinalgOp::getRegionBuilder() {
      return getImpl()->getRegionBuilder();
  }
