/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DataLayoutEntryInterface;
namespace detail {
struct DataLayoutEntryInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::DataLayoutEntryKey (*getKey)(::mlir::Attribute );
    ::mlir::Attribute (*getValue)(::mlir::Attribute );
    ::mlir::LogicalResult (*verifyEntry)(::mlir::Attribute , ::mlir::Location);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    Model() : Concept{getKey, getValue, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryKey getKey(::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::Attribute getValue(::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
};
} // end namespace detail
class DataLayoutEntryInterface : public ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct DataLayoutEntryInterfaceTrait : public ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    ::mlir::LogicalResult verifyEntry(::mlir::Location loc) const {
      return ::mlir::success();
    }
  };
  template <typename ConcreteAttr>
  struct Trait : public DataLayoutEntryInterfaceTrait<ConcreteAttr> {};
  ::mlir::DataLayoutEntryKey getKey() const;
  ::mlir::Attribute getValue() const;
  ::mlir::LogicalResult verifyEntry(::mlir::Location loc) const;

    /// Returns `true` if the key of this entry is a type.
    bool isTypeEntry() {
      return getKey().is<::mlir::Type>();
    }
  
};
template<typename ConcreteAttr>
::mlir::DataLayoutEntryKey detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::getKey(::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getKey();
}
template<typename ConcreteAttr>
::mlir::Attribute detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::getValue(::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getValue();
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).verifyEntry(loc);
}
} // namespace mlir
namespace mlir {
class DataLayoutSpecInterface;
namespace detail {
struct DataLayoutSpecInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::DataLayoutSpecInterface (*combineWith)(::mlir::Attribute , ::llvm::ArrayRef<DataLayoutSpecInterface>);
    ::mlir::DataLayoutEntryListRef (*getEntries)(::mlir::Attribute );
    ::mlir::DataLayoutEntryList (*getSpecForType)(::mlir::Attribute , ::mlir::TypeID);
    ::mlir::DataLayoutEntryInterface (*getSpecForIdentifier)(::mlir::Attribute , ::mlir::Identifier);
    ::mlir::LogicalResult (*verifySpec)(::mlir::Attribute , ::mlir::Location);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    Model() : Concept{combineWith, getEntries, getSpecForType, getSpecForIdentifier, verifySpec} {}

    static inline ::mlir::DataLayoutSpecInterface combineWith(::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<DataLayoutSpecInterface> specs);
    static inline ::mlir::DataLayoutEntryListRef getEntries(::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::DataLayoutEntryList getSpecForType(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::Identifier identifier);
    static inline ::mlir::LogicalResult verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
};
} // end namespace detail
class DataLayoutSpecInterface : public ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct DataLayoutSpecInterfaceTrait : public ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    ::mlir::DataLayoutEntryList getSpecForType(::mlir::TypeID type) const {
      return ::mlir::detail::filterEntriesForType((*static_cast<const ConcreteAttr *>(this)).getEntries(), type);
    }
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::Identifier identifier) const {
      return ::mlir::detail::filterEntryForIdentifier((*static_cast<const ConcreteAttr *>(this)).getEntries(),
                                                        identifier);
    }
    ::mlir::LogicalResult verifySpec(::mlir::Location loc) const {
      return ::mlir::detail::verifyDataLayoutSpec((*static_cast<const ConcreteAttr *>(this)), loc);
    }
  };
  template <typename ConcreteAttr>
  struct Trait : public DataLayoutSpecInterfaceTrait<ConcreteAttr> {};
  ::mlir::DataLayoutSpecInterface combineWith(::llvm::ArrayRef<DataLayoutSpecInterface> specs) const;
  ::mlir::DataLayoutEntryListRef getEntries() const;
  ::mlir::DataLayoutEntryList getSpecForType(::mlir::TypeID type) const;
  ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::Identifier identifier) const;
  ::mlir::LogicalResult verifySpec(::mlir::Location loc) const;

    /// Returns a copy of the entries related to the type specified as template
    /// parameter.
    template <typename Ty>
    DataLayoutEntryList getSpecForType() {
      return getSpecForType(TypeID::get<Ty>());
    }

    /// Populates the given maps with lists of entries grouped by the type or
    /// identifier they are associated with. Users are not expected to call this
    /// method directly.
    void bucketEntriesByType(
        ::llvm::DenseMap<::mlir::TypeID, ::mlir::DataLayoutEntryList> &types,
        ::llvm::DenseMap<::mlir::Identifier,
                         ::mlir::DataLayoutEntryInterface> &ids);
  
};
template<typename ConcreteAttr>
::mlir::DataLayoutSpecInterface detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::combineWith(::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<DataLayoutSpecInterface> specs) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).combineWith(specs);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getEntries(::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getEntries();
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForType(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getSpecForType(type);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::Identifier identifier) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getSpecForIdentifier(identifier);
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).verifySpec(loc);
}
} // namespace mlir
