/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::AffineMapAttr,
::mlir::ArrayAttr,
::mlir::DenseIntOrFPElementsAttr,
::mlir::DenseStringElementsAttr,
::mlir::DictionaryAttr,
::mlir::FloatAttr,
::mlir::IntegerAttr,
::mlir::IntegerSetAttr,
::mlir::OpaqueAttr,
::mlir::OpaqueElementsAttr,
::mlir::SparseElementsAttr,
::mlir::StringAttr,
::mlir::SymbolRefAttr,
::mlir::TypeAttr,
::mlir::UnitAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

namespace mlir {

namespace detail {
  struct AffineMapAttrStorage : public ::mlir::AttributeStorage {
    AffineMapAttrStorage (AffineMap value)
      : ::mlir::AttributeStorage(IndexType::get(value.getContext())), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<AffineMap>;
  bool operator==(const KeyTy &key) const {
    if (!(value == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static AffineMapAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto value = std::get<0>(key);

      return new (allocator.allocate<AffineMapAttrStorage>())
          AffineMapAttrStorage(value);
    }
      AffineMap value;
  };
} // namespace detail
AffineMapAttr AffineMapAttr::get(AffineMap value) {
  
      return Base::get(value.getContext(), value);
    ;
}
AffineMap AffineMapAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct ArrayAttrStorage : public ::mlir::AttributeStorage {
    ArrayAttrStorage (::llvm::ArrayRef<Attribute> value)
      : ::mlir::AttributeStorage(), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::ArrayRef<Attribute>>;
  bool operator==(const KeyTy &key) const {
    if (!(value == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static ArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto value = std::get<0>(key);
      value = allocator.copyInto(value);

      return new (allocator.allocate<ArrayAttrStorage>())
          ArrayAttrStorage(value);
    }
      ::llvm::ArrayRef<Attribute> value;
  };
} // namespace detail
ArrayAttr ArrayAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Attribute> value) {
  return Base::get(context, value);
}
::llvm::ArrayRef<Attribute> ArrayAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {
} // namespace mlir
namespace mlir {
DenseStringElementsAttr DenseStringElementsAttr::get(ShapedType type, ArrayRef<StringRef> values) {
  
      return Base::get(type.getContext(), type, values,
                   /* isSplat */(values.size() == 1));
    ;
}
} // namespace mlir
namespace mlir {

namespace detail {
  struct DictionaryAttrStorage : public ::mlir::AttributeStorage {
    DictionaryAttrStorage (::llvm::ArrayRef<NamedAttribute> value)
      : ::mlir::AttributeStorage(), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::ArrayRef<NamedAttribute>>;
  bool operator==(const KeyTy &key) const {
    if (!(value == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static DictionaryAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto value = std::get<0>(key);
      value = allocator.copyInto(value);

      return new (allocator.allocate<DictionaryAttrStorage>())
          DictionaryAttrStorage(value);
    }
      ::llvm::ArrayRef<NamedAttribute> value;
  };
} // namespace detail
::llvm::ArrayRef<NamedAttribute> DictionaryAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct FloatAttrStorage : public ::mlir::AttributeStorage {
    FloatAttrStorage (::mlir::Type type, ::llvm::APFloat value)
      : ::mlir::AttributeStorage(type), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, ::llvm::APFloat>;
  bool operator==(const KeyTy &key) const {
    if (!(getType() == std::get<0>(key)))
      return false;
    if (!(value.bitwiseIsEqual(std::get<1>(key))))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static FloatAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto type = std::get<0>(key);
      auto value = std::get<1>(key);

      return new (allocator.allocate<FloatAttrStorage>())
          FloatAttrStorage(type, value);
    }
      ::llvm::APFloat value;
  };
} // namespace detail
FloatAttr FloatAttr::get(Type type, const APFloat & value) {
  
      return Base::get(type.getContext(), type, value);
    ;
}
FloatAttr FloatAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Type type, const APFloat & value) {
  
      return Base::getChecked(emitErrorFn, type.getContext(), type, value);
    ;
}
FloatAttr FloatAttr::get(Type type, double value) {
  
      if (type.isF64() || !type.isa<FloatType>())
        return Base::get(type.getContext(), type, APFloat(value));

      // This handles, e.g., F16 because there is no APFloat constructor for it.
      bool unused;
      APFloat val(value);
      val.convert(type.cast<FloatType>().getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &unused);
      return Base::get(type.getContext(), type, val);
    ;
}
FloatAttr FloatAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Type type, double value) {
  
      if (type.isF64() || !type.isa<FloatType>())
        return Base::getChecked(emitErrorFn, type.getContext(), type, APFloat(value));

      // This handles, e.g., F16 because there is no APFloat constructor for it.
      bool unused;
      APFloat val(value);
      val.convert(type.cast<FloatType>().getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &unused);
      return Base::getChecked(emitErrorFn, type.getContext(), type, val);
    ;
}
::mlir::Type FloatAttr::getType() const { return getImpl()->getType().cast<::mlir::Type>(); }
::llvm::APFloat FloatAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct IntegerAttrStorage : public ::mlir::AttributeStorage {
    IntegerAttrStorage (::mlir::Type type, APInt value)
      : ::mlir::AttributeStorage(type), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, APInt>;
  bool operator==(const KeyTy &key) const {
    if (!(getType() == std::get<0>(key)))
      return false;
    if (!(value == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static IntegerAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto type = std::get<0>(key);
      auto value = std::get<1>(key);

      return new (allocator.allocate<IntegerAttrStorage>())
          IntegerAttrStorage(type, value);
    }
      APInt value;
  };
} // namespace detail
IntegerAttr IntegerAttr::get(Type type, const APInt & value) {
  
      if (type.isSignlessInteger(1))
        return BoolAttr::get(type.getContext(), value.getBoolValue());
      return Base::get(type.getContext(), type, value);
    ;
}
IntegerAttr IntegerAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Type type, const APInt & value) {
  
      if (type.isSignlessInteger(1))
        return BoolAttr::get(type.getContext(), value.getBoolValue());
      return Base::getChecked(emitErrorFn, type.getContext(), type, value);
    ;
}
IntegerAttr IntegerAttr::get(Type type, int64_t value) {
  
      // `index` has a defined internal storage width.
      if (type.isIndex()) {
        APInt apValue(IndexType::kInternalStorageBitWidth, value);
        return Base::get(type.getContext(), type, apValue);
      }

      IntegerType intTy = type.cast<IntegerType>();
      APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
      return Base::get(type.getContext(), type, apValue);
    ;
}
IntegerAttr IntegerAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Type type, int64_t value) {
  
      // `index` has a defined internal storage width.
      if (type.isIndex()) {
        APInt apValue(IndexType::kInternalStorageBitWidth, value);
        return Base::getChecked(emitErrorFn, type.getContext(), type, apValue);
      }

      IntegerType intTy = type.cast<IntegerType>();
      APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
      return Base::getChecked(emitErrorFn, type.getContext(), type, apValue);
    ;
}
::mlir::Type IntegerAttr::getType() const { return getImpl()->getType().cast<::mlir::Type>(); }
APInt IntegerAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct IntegerSetAttrStorage : public ::mlir::AttributeStorage {
    IntegerSetAttrStorage (IntegerSet value)
      : ::mlir::AttributeStorage(), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<IntegerSet>;
  bool operator==(const KeyTy &key) const {
    if (!(value == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static IntegerSetAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto value = std::get<0>(key);

      return new (allocator.allocate<IntegerSetAttrStorage>())
          IntegerSetAttrStorage(value);
    }
      IntegerSet value;
  };
} // namespace detail
IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  
      return Base::get(value.getContext(), value);
    ;
}
IntegerSet IntegerSetAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct OpaqueAttrStorage : public ::mlir::AttributeStorage {
    OpaqueAttrStorage (Identifier dialectNamespace, ::llvm::StringRef attrData, ::mlir::Type type)
      : ::mlir::AttributeStorage(type), dialectNamespace(dialectNamespace), attrData(attrData) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Identifier, ::llvm::StringRef, ::mlir::Type>;
  bool operator==(const KeyTy &key) const {
    if (!(dialectNamespace == std::get<0>(key)))
      return false;
    if (!(attrData == std::get<1>(key)))
      return false;
    if (!(getType() == std::get<2>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static OpaqueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto dialectNamespace = std::get<0>(key);
      auto attrData = std::get<1>(key);
      auto type = std::get<2>(key);
      attrData = allocator.copyInto(attrData);

      return new (allocator.allocate<OpaqueAttrStorage>())
          OpaqueAttrStorage(dialectNamespace, attrData, type);
    }
      Identifier dialectNamespace;
      ::llvm::StringRef attrData;
  };
} // namespace detail
OpaqueAttr OpaqueAttr::get(Identifier dialect, StringRef attrData, Type type) {
  
      return Base::get(dialect.getContext(), dialect, attrData, type);
    ;
}
OpaqueAttr OpaqueAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Identifier dialect, StringRef attrData, Type type) {
  
      return Base::getChecked(emitErrorFn, dialect.getContext(), dialect, attrData, type);
    ;
}
Identifier OpaqueAttr::getDialectNamespace() const { return getImpl()->dialectNamespace; }
::llvm::StringRef OpaqueAttr::getAttrData() const { return getImpl()->attrData; }
::mlir::Type OpaqueAttr::getType() const { return getImpl()->getType().cast<::mlir::Type>(); }
} // namespace mlir
namespace mlir {

namespace detail {
  struct OpaqueElementsAttrStorage : public ::mlir::AttributeStorage {
    OpaqueElementsAttrStorage (Identifier dialect, ::llvm::StringRef value, ShapedType type)
      : ::mlir::AttributeStorage(type), dialect(dialect), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Identifier, ::llvm::StringRef, ShapedType>;
  bool operator==(const KeyTy &key) const {
    if (!(dialect == std::get<0>(key)))
      return false;
    if (!(value == std::get<1>(key)))
      return false;
    if (!(getType() == std::get<2>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static OpaqueElementsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto dialect = std::get<0>(key);
      auto value = std::get<1>(key);
      auto type = std::get<2>(key);
      value = allocator.copyInto(value);

      return new (allocator.allocate<OpaqueElementsAttrStorage>())
          OpaqueElementsAttrStorage(dialect, value, type);
    }
      Identifier dialect;
      ::llvm::StringRef value;
  };
} // namespace detail
OpaqueElementsAttr OpaqueElementsAttr::get(Identifier dialect, ShapedType type, StringRef value) {
  
      return Base::get(dialect.getContext(), dialect, value, type);
    ;
}
OpaqueElementsAttr OpaqueElementsAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Identifier dialect, ShapedType type, StringRef value) {
  
      return Base::getChecked(emitErrorFn, dialect.getContext(), dialect, value, type);
    ;
}
OpaqueElementsAttr OpaqueElementsAttr::get(Dialect * dialect, ShapedType type, StringRef value) {
  
      MLIRContext *ctxt = dialect->getContext();
      Identifier dialectName = Identifier::get(dialect->getNamespace(), ctxt);
      return Base::get(ctxt, dialectName, value, type);
    ;
}
OpaqueElementsAttr OpaqueElementsAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Dialect * dialect, ShapedType type, StringRef value) {
  
      MLIRContext *ctxt = dialect->getContext();
      Identifier dialectName = Identifier::get(dialect->getNamespace(), ctxt);
      return Base::getChecked(emitErrorFn, ctxt, dialectName, value, type);
    ;
}
Identifier OpaqueElementsAttr::getDialect() const { return getImpl()->dialect; }
::llvm::StringRef OpaqueElementsAttr::getValue() const { return getImpl()->value; }
ShapedType OpaqueElementsAttr::getType() const { return getImpl()->getType().cast<ShapedType>(); }
} // namespace mlir
namespace mlir {

namespace detail {
  struct SparseElementsAttrStorage : public ::mlir::AttributeStorage {
    SparseElementsAttrStorage (ShapedType type, DenseIntElementsAttr indices, DenseElementsAttr values)
      : ::mlir::AttributeStorage(type), indices(indices), values(values) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<ShapedType, DenseIntElementsAttr, DenseElementsAttr>;
  bool operator==(const KeyTy &key) const {
    if (!(getType() == std::get<0>(key)))
      return false;
    if (!(indices == std::get<1>(key)))
      return false;
    if (!(values == std::get<2>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static SparseElementsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto type = std::get<0>(key);
      auto indices = std::get<1>(key);
      auto values = std::get<2>(key);

      return new (allocator.allocate<SparseElementsAttrStorage>())
          SparseElementsAttrStorage(type, indices, values);
    }
      DenseIntElementsAttr indices;
      DenseElementsAttr values;
  };
} // namespace detail
SparseElementsAttr SparseElementsAttr::get(ShapedType type, DenseElementsAttr indices, DenseElementsAttr values) {
  
      assert(indices.getType().getElementType().isInteger(64) &&
             "expected sparse indices to be 64-bit integer values");
      assert((type.isa<RankedTensorType, VectorType>()) &&
             "type must be ranked tensor or vector");
      assert(type.hasStaticShape() && "type must have static shape");
      return Base::get(type.getContext(), type,
                   indices.cast<DenseIntElementsAttr>(), values);
    ;
}
ShapedType SparseElementsAttr::getType() const { return getImpl()->getType().cast<ShapedType>(); }
DenseIntElementsAttr SparseElementsAttr::getIndices() const { return getImpl()->indices; }
DenseElementsAttr SparseElementsAttr::getValues() const { return getImpl()->values; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct StringAttrStorage : public ::mlir::AttributeStorage {
    StringAttrStorage (::llvm::StringRef value, ::mlir::Type type)
      : ::mlir::AttributeStorage(type), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::StringRef, ::mlir::Type>;
  bool operator==(const KeyTy &key) const {
    if (!(value == std::get<0>(key)))
      return false;
    if (!(getType() == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static StringAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto value = std::get<0>(key);
      auto type = std::get<1>(key);
      value = allocator.copyInto(value);

      return new (allocator.allocate<StringAttrStorage>())
          StringAttrStorage(value, type);
    }
      ::llvm::StringRef value;
  };
} // namespace detail
StringAttr StringAttr::get(StringRef bytes, Type type) {
  
      return Base::get(type.getContext(), bytes, type);
    ;
}
StringAttr StringAttr::get(::mlir::MLIRContext *context, StringRef bytes) {
  
      return Base::get(context, bytes, NoneType::get(context));
    ;
}
::llvm::StringRef StringAttr::getValue() const { return getImpl()->value; }
::mlir::Type StringAttr::getType() const { return getImpl()->getType().cast<::mlir::Type>(); }
} // namespace mlir
namespace mlir {

namespace detail {
  struct SymbolRefAttrStorage : public ::mlir::AttributeStorage {
    SymbolRefAttrStorage (::llvm::StringRef rootReference, ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences)
      : ::mlir::AttributeStorage(), rootReference(rootReference), nestedReferences(nestedReferences) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::StringRef, ::llvm::ArrayRef<FlatSymbolRefAttr>>;
  bool operator==(const KeyTy &key) const {
    if (!(rootReference == std::get<0>(key)))
      return false;
    if (!(nestedReferences == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static SymbolRefAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto rootReference = std::get<0>(key);
      auto nestedReferences = std::get<1>(key);
      rootReference = allocator.copyInto(rootReference);
      nestedReferences = allocator.copyInto(nestedReferences);

      return new (allocator.allocate<SymbolRefAttrStorage>())
          SymbolRefAttrStorage(rootReference, nestedReferences);
    }
      ::llvm::StringRef rootReference;
      ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences;
  };
} // namespace detail
SymbolRefAttr SymbolRefAttr::get(::mlir::MLIRContext *context, ::llvm::StringRef rootReference, ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences) {
  return Base::get(context, rootReference, nestedReferences);
}
::llvm::StringRef SymbolRefAttr::getRootReference() const { return getImpl()->rootReference; }
::llvm::ArrayRef<FlatSymbolRefAttr> SymbolRefAttr::getNestedReferences() const { return getImpl()->nestedReferences; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct TypeAttrStorage : public ::mlir::AttributeStorage {
    TypeAttrStorage (Type value)
      : ::mlir::AttributeStorage(), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Type>;
  bool operator==(const KeyTy &key) const {
    if (!(value == std::get<0>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static TypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto value = std::get<0>(key);

      return new (allocator.allocate<TypeAttrStorage>())
          TypeAttrStorage(value);
    }
      Type value;
  };
} // namespace detail
TypeAttr TypeAttr::get(Type type) {
  
      return Base::get(type.getContext(), type);
    ;
}
Type TypeAttr::getValue() const { return getImpl()->value; }
} // namespace mlir
namespace mlir {
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

