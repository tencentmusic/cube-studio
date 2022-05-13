/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

StringRef mlir::VectorTransferOpInterface::getMaskedAttrName() {
      return getImpl()->getMaskedAttrName();
  }
StringRef mlir::VectorTransferOpInterface::getPermutationMapAttrName() {
      return getImpl()->getPermutationMapAttrName();
  }
bool mlir::VectorTransferOpInterface::isMaskedDim(unsigned dim) {
      return getImpl()->isMaskedDim(getOperation(), dim);
  }
Value mlir::VectorTransferOpInterface::source() {
      return getImpl()->source(getOperation());
  }
Value mlir::VectorTransferOpInterface::vector() {
      return getImpl()->vector(getOperation());
  }
ValueRange mlir::VectorTransferOpInterface::indices() {
      return getImpl()->indices(getOperation());
  }
AffineMap mlir::VectorTransferOpInterface::permutation_map() {
      return getImpl()->permutation_map(getOperation());
  }
Optional<ArrayAttr> mlir::VectorTransferOpInterface::masked() {
      return getImpl()->masked(getOperation());
  }
ShapedType mlir::VectorTransferOpInterface::getShapedType() {
      return getImpl()->getShapedType(getOperation());
  }
VectorType mlir::VectorTransferOpInterface::getVectorType() {
      return getImpl()->getVectorType(getOperation());
  }
unsigned mlir::VectorTransferOpInterface::getTransferRank() {
      return getImpl()->getTransferRank(getOperation());
  }
unsigned mlir::VectorTransferOpInterface::getLeadingShapedRank() {
      return getImpl()->getLeadingShapedRank(getOperation());
  }
bool mlir::VectorTransferOpInterface::hasMaskedDim() {
      return getImpl()->hasMaskedDim(getOperation());
  }
void mlir::VectorTransferOpInterface::zipResultAndIndexing(llvm::function_ref<void(int64_t, int64_t)> fun) {
      return getImpl()->zipResultAndIndexing(getOperation(), fun);
  }
Optional<SmallVector<int64_t, 4>> mlir::VectorUnrollOpInterface::getShapeForUnroll() {
      return getImpl()->getShapeForUnroll(getOperation());
  }
