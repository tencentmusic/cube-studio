/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

Optional<MutableOperandRange> mlir::BranchOpInterface::getMutableSuccessorOperands(unsigned index) {
      return getImpl()->getMutableSuccessorOperands(getOperation(), index);
  }
Optional<OperandRange> mlir::BranchOpInterface::getSuccessorOperands(unsigned index) {
      return getImpl()->getSuccessorOperands(getOperation(), index);
  }
Optional<BlockArgument> mlir::BranchOpInterface::getSuccessorBlockArgument(unsigned operandIndex) {
      return getImpl()->getSuccessorBlockArgument(getOperation(), operandIndex);
  }
Block *mlir::BranchOpInterface::getSuccessorForOperands(ArrayRef<Attribute> operands) {
      return getImpl()->getSuccessorForOperands(getOperation(), operands);
  }
OperandRange mlir::RegionBranchOpInterface::getSuccessorEntryOperands(unsigned index) {
      return getImpl()->getSuccessorEntryOperands(getOperation(), index);
  }
void mlir::RegionBranchOpInterface::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions) {
      return getImpl()->getSuccessorRegions(getOperation(), index, operands, regions);
  }
void mlir::RegionBranchOpInterface::getNumRegionInvocations(ArrayRef<Attribute> operands, SmallVectorImpl<int64_t> & countPerRegion) {
      return getImpl()->getNumRegionInvocations(getOperation(), operands, countPerRegion);
  }
