/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

unsigned mlir::DataLayoutTypeInterface::getTypeSize(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getTypeSize(*this, dataLayout, params);
  }
unsigned mlir::DataLayoutTypeInterface::getABIAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getABIAlignment(*this, dataLayout, params);
  }
unsigned mlir::DataLayoutTypeInterface::getPreferredAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getPreferredAlignment(*this, dataLayout, params);
  }
bool mlir::DataLayoutTypeInterface::areCompatible(::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const {
      return getImpl()->areCompatible(*this, oldLayout, newLayout);
  }
::mlir::LogicalResult mlir::DataLayoutTypeInterface::verifyEntries(::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const {
      return getImpl()->verifyEntries(*this, entries, loc);
  }
