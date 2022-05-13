/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

CallInterfaceCallable mlir::CallOpInterface::getCallableForCallee() {
      return getImpl()->getCallableForCallee(getOperation());
  }
Operation::operand_range mlir::CallOpInterface::getArgOperands() {
      return getImpl()->getArgOperands(getOperation());
  }
Region *mlir::CallableOpInterface::getCallableRegion() {
      return getImpl()->getCallableRegion(getOperation());
  }
ArrayRef<Type> mlir::CallableOpInterface::getCallableResults() {
      return getImpl()->getCallableResults(getOperation());
  }
