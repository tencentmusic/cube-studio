/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Availability Interface Definitions                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::spirv::Version QueryMinVersionInterface::getMinVersion() {
  return getImpl()->getMinVersion(getOperation());
}
::mlir::spirv::Version QueryMaxVersionInterface::getMaxVersion() {
  return getImpl()->getMaxVersion(getOperation());
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> QueryExtensionInterface::getExtensions() {
  return getImpl()->getExtensions(getOperation());
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> QueryCapabilityInterface::getCapabilities() {
  return getImpl()->getCapabilities(getOperation());
}
