/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::Value mlir::CopyOpInterface::getSource() {
      return getImpl()->getSource(getOperation());
  }
::mlir::Value mlir::CopyOpInterface::getTarget() {
      return getImpl()->getTarget(getOperation());
  }
