/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool mlir::LoopLikeOpInterface::isDefinedOutsideOfLoop(Value  value) {
      return getImpl()->isDefinedOutsideOfLoop(getOperation(), value);
  }
Region &mlir::LoopLikeOpInterface::getLoopBody() {
      return getImpl()->getLoopBody(getOperation());
  }
LogicalResult mlir::LoopLikeOpInterface::moveOutOfLoop(ArrayRef<Operation *> ops) {
      return getImpl()->moveOutOfLoop(getOperation(), ops);
  }
