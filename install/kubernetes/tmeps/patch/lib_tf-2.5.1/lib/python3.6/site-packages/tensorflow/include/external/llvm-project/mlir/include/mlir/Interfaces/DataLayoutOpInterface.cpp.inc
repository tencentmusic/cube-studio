/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

DataLayoutSpecInterface mlir::DataLayoutOpInterface::getDataLayoutSpec() {
      return getImpl()->getDataLayoutSpec(getOperation());
  }
unsigned mlir::DataLayoutOpInterface::getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeSize(type, dataLayout, params);
  }
unsigned mlir::DataLayoutOpInterface::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeABIAlignment(type, dataLayout, params);
  }
unsigned mlir::DataLayoutOpInterface::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypePreferredAlignment(type, dataLayout, params);
  }
