/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::DataLayoutEntryKey mlir::DataLayoutEntryInterface::getKey() const {
      return getImpl()->getKey(*this);
  }
::mlir::Attribute mlir::DataLayoutEntryInterface::getValue() const {
      return getImpl()->getValue(*this);
  }
::mlir::LogicalResult mlir::DataLayoutEntryInterface::verifyEntry(::mlir::Location loc) const {
      return getImpl()->verifyEntry(*this, loc);
  }
::mlir::DataLayoutSpecInterface mlir::DataLayoutSpecInterface::combineWith(::llvm::ArrayRef<DataLayoutSpecInterface> specs) const {
      return getImpl()->combineWith(*this, specs);
  }
::mlir::DataLayoutEntryListRef mlir::DataLayoutSpecInterface::getEntries() const {
      return getImpl()->getEntries(*this);
  }
::mlir::DataLayoutEntryList mlir::DataLayoutSpecInterface::getSpecForType(::mlir::TypeID type) const {
      return getImpl()->getSpecForType(*this, type);
  }
::mlir::DataLayoutEntryInterface mlir::DataLayoutSpecInterface::getSpecForIdentifier(::mlir::Identifier identifier) const {
      return getImpl()->getSpecForIdentifier(*this, identifier);
  }
::mlir::LogicalResult mlir::DataLayoutSpecInterface::verifySpec(::mlir::Location loc) const {
      return getImpl()->verifySpec(*this, loc);
  }
