/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

OperandRange mlir::gpu::AsyncOpInterface::getAsyncDependencies() {
      return getImpl()->getAsyncDependencies(getOperation());
  }
void mlir::gpu::AsyncOpInterface::addAsyncDependency(Value token) {
      return getImpl()->addAsyncDependency(getOperation(), token);
  }
OpResult mlir::gpu::AsyncOpInterface::getAsyncToken() {
      return getImpl()->getAsyncToken(getOperation());
  }
