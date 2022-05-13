/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

NamedAttribute AffineMapAccessInterface::getAffineMapAttrForMemRef(Value memref) {
      return getImpl()->getAffineMapAttrForMemRef(getOperation(), memref);
  }
Value AffineReadOpInterface::getMemRef() {
      return getImpl()->getMemRef(getOperation());
  }
MemRefType AffineReadOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getOperation());
  }
Operation::operand_range AffineReadOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getOperation());
  }
AffineMap AffineReadOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getOperation());
  }
Value AffineReadOpInterface::getValue() {
      return getImpl()->getValue(getOperation());
  }
Value AffineWriteOpInterface::getMemRef() {
      return getImpl()->getMemRef(getOperation());
  }
MemRefType AffineWriteOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getOperation());
  }
Operation::operand_range AffineWriteOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getOperation());
  }
AffineMap AffineWriteOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getOperation());
  }
Value AffineWriteOpInterface::getValueToStore() {
      return getImpl()->getValueToStore(getOperation());
  }
