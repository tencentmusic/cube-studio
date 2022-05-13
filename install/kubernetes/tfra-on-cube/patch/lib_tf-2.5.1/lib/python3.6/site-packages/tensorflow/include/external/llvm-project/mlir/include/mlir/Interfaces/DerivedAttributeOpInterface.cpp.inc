/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool mlir::DerivedAttributeOpInterface::isDerivedAttribute(StringRef name) {
      return getImpl()->isDerivedAttribute(name);
  }
DictionaryAttr mlir::DerivedAttributeOpInterface::materializeDerivedAttributes() {
      return getImpl()->materializeDerivedAttributes(getOperation());
  }
