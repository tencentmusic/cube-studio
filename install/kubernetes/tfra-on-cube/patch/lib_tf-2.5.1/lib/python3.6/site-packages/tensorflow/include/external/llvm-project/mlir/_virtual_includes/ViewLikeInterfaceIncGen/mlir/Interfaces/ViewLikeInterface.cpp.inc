/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

unsigned mlir::OffsetSizeAndStrideOpInterface::getOffsetSizeAndStrideStartOperandIndex() {
      return getImpl()->getOffsetSizeAndStrideStartOperandIndex(getOperation());
  }
std::array<unsigned, 3> mlir::OffsetSizeAndStrideOpInterface::getArrayAttrMaxRanks() {
      return getImpl()->getArrayAttrMaxRanks(getOperation());
  }
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::offsets() {
      return getImpl()->offsets(getOperation());
  }
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::sizes() {
      return getImpl()->sizes(getOperation());
  }
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::strides() {
      return getImpl()->strides(getOperation());
  }
::mlir::ArrayAttr mlir::OffsetSizeAndStrideOpInterface::static_offsets() {
      return getImpl()->static_offsets(getOperation());
  }
::mlir::ArrayAttr mlir::OffsetSizeAndStrideOpInterface::static_sizes() {
      return getImpl()->static_sizes(getOperation());
  }
::mlir::ArrayAttr mlir::OffsetSizeAndStrideOpInterface::static_strides() {
      return getImpl()->static_strides(getOperation());
  }
::mlir::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedOffsets() {
      return getImpl()->getMixedOffsets(getOperation());
  }
::mlir::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedSizes() {
      return getImpl()->getMixedSizes(getOperation());
  }
::mlir::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedStrides() {
      return getImpl()->getMixedStrides(getOperation());
  }
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicOffset(unsigned idx) {
      return getImpl()->isDynamicOffset(getOperation(), idx);
  }
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicSize(unsigned idx) {
      return getImpl()->isDynamicSize(getOperation(), idx);
  }
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicStride(unsigned idx) {
      return getImpl()->isDynamicStride(getOperation(), idx);
  }
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticOffset(unsigned idx) {
      return getImpl()->getStaticOffset(getOperation(), idx);
  }
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticSize(unsigned idx) {
      return getImpl()->getStaticSize(getOperation(), idx);
  }
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticStride(unsigned idx) {
      return getImpl()->getStaticStride(getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicOffset(unsigned idx) {
      return getImpl()->getIndexOfDynamicOffset(getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicSize(unsigned idx) {
      return getImpl()->getIndexOfDynamicSize(getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicStride(unsigned idx) {
      return getImpl()->getIndexOfDynamicStride(getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getNumDynamicEntriesUpToIdx(::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
      return getImpl()->getNumDynamicEntriesUpToIdx(getOperation(), attr, isDynamic, idx);
  }
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicOffset(unsigned idx) {
      return getImpl()->getDynamicOffset(getOperation(), idx);
  }
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicSize(unsigned idx) {
      return getImpl()->getDynamicSize(getOperation(), idx);
  }
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicStride(unsigned idx) {
      return getImpl()->getDynamicStride(getOperation(), idx);
  }
::mlir::Value mlir::ViewLikeOpInterface::getViewSource() {
      return getImpl()->getViewSource(getOperation());
  }
