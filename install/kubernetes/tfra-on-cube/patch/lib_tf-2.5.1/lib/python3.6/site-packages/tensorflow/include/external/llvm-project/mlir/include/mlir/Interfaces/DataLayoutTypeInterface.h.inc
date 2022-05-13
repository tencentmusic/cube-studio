/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DataLayoutTypeInterface;
namespace detail {
struct DataLayoutTypeInterfaceInterfaceTraits {
  struct Concept {
    unsigned (*getTypeSize)(::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getABIAlignment)(::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getPreferredAlignment)(::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    bool (*areCompatible)(::mlir::Type , ::mlir::DataLayoutEntryListRef, ::mlir::DataLayoutEntryListRef);
    ::mlir::LogicalResult (*verifyEntries)(::mlir::Type , ::mlir::DataLayoutEntryListRef, ::mlir::Location);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    Model() : Concept{getTypeSize, getABIAlignment, getPreferredAlignment, areCompatible, verifyEntries} {}

    static inline unsigned getTypeSize(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getABIAlignment(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getPreferredAlignment(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline bool areCompatible(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout);
    static inline ::mlir::LogicalResult verifyEntries(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc);
  };
};
} // end namespace detail
class DataLayoutTypeInterface : public ::mlir::TypeInterface<DataLayoutTypeInterface, detail::DataLayoutTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<DataLayoutTypeInterface, detail::DataLayoutTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct DataLayoutTypeInterfaceTrait : public ::mlir::TypeInterface<DataLayoutTypeInterface, detail::DataLayoutTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
    bool areCompatible(::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const {
      return true;
    }
    ::mlir::LogicalResult verifyEntries(::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const {
      return ::mlir::success();
    }
  };
  template <typename ConcreteType>
  struct Trait : public DataLayoutTypeInterfaceTrait<ConcreteType> {};
  unsigned getTypeSize(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  unsigned getABIAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  unsigned getPreferredAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  bool areCompatible(::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const;
  ::mlir::LogicalResult verifyEntries(::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const;
};
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getTypeSize(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getTypeSize(dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getABIAlignment(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getABIAlignment(dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getPreferredAlignment(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getPreferredAlignment(dataLayout, params);
}
template<typename ConcreteType>
bool detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::areCompatible(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) {
  return (tablegen_opaque_val.cast<ConcreteType>()).areCompatible(oldLayout, newLayout);
}
template<typename ConcreteType>
::mlir::LogicalResult detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::verifyEntries(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) {
  return (tablegen_opaque_val.cast<ConcreteType>()).verifyEntries(entries, loc);
}
} // namespace mlir
