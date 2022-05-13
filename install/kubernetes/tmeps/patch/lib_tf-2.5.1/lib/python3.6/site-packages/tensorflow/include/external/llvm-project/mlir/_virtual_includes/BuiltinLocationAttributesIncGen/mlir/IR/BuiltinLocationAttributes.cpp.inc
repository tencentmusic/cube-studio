/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::CallSiteLoc,
::mlir::FileLineColLoc,
::mlir::FusedLoc,
::mlir::NameLoc,
::mlir::OpaqueLoc,
::mlir::UnknownLoc

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

namespace mlir {

namespace detail {
  struct CallSiteLocAttrStorage : public ::mlir::AttributeStorage {
    CallSiteLocAttrStorage (Location callee, Location caller)
      : ::mlir::AttributeStorage(), callee(callee), caller(caller) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Location, Location>;
  bool operator==(const KeyTy &key) const {
    if (!(callee == std::get<0>(key)))
      return false;
    if (!(caller == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static CallSiteLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto callee = std::get<0>(key);
      auto caller = std::get<1>(key);

      return new (allocator.allocate<CallSiteLocAttrStorage>())
          CallSiteLocAttrStorage(callee, caller);
    }
      Location callee;
      Location caller;
  };
} // namespace detail
CallSiteLoc CallSiteLoc::get(Location callee, Location caller) {
  
      return Base::get(callee->getContext(), callee, caller);
    ;
}
Location CallSiteLoc::getCallee() const { return getImpl()->callee; }
Location CallSiteLoc::getCaller() const { return getImpl()->caller; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct FileLineColLocAttrStorage : public ::mlir::AttributeStorage {
    FileLineColLocAttrStorage (Identifier filename, unsigned line, unsigned column)
      : ::mlir::AttributeStorage(), filename(filename), line(line), column(column) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Identifier, unsigned, unsigned>;
  bool operator==(const KeyTy &key) const {
    if (!(filename == std::get<0>(key)))
      return false;
    if (!(line == std::get<1>(key)))
      return false;
    if (!(column == std::get<2>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static FileLineColLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto filename = std::get<0>(key);
      auto line = std::get<1>(key);
      auto column = std::get<2>(key);

      return new (allocator.allocate<FileLineColLocAttrStorage>())
          FileLineColLocAttrStorage(filename, line, column);
    }
      Identifier filename;
      unsigned line;
      unsigned column;
  };
} // namespace detail
FileLineColLoc FileLineColLoc::get(Identifier filename, unsigned line, unsigned column) {
  
      return Base::get(filename.getContext(), filename, line, column);
    ;
}
FileLineColLoc FileLineColLoc::get(::mlir::MLIRContext *context, StringRef filename, unsigned line, unsigned column) {
  
      return Base::get(context,
                   Identifier::get(filename.empty() ? "-" : filename, context),
                   line, column);
    ;
}
Identifier FileLineColLoc::getFilename() const { return getImpl()->filename; }
unsigned FileLineColLoc::getLine() const { return getImpl()->line; }
unsigned FileLineColLoc::getColumn() const { return getImpl()->column; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct FusedLocAttrStorage : public ::mlir::AttributeStorage {
    FusedLocAttrStorage (::llvm::ArrayRef<Location> locations, Attribute metadata)
      : ::mlir::AttributeStorage(), locations(locations), metadata(metadata) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::ArrayRef<Location>, Attribute>;
  bool operator==(const KeyTy &key) const {
    if (!(locations == std::get<0>(key)))
      return false;
    if (!(metadata == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static FusedLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto locations = std::get<0>(key);
      auto metadata = std::get<1>(key);
      locations = allocator.copyInto(locations);

      return new (allocator.allocate<FusedLocAttrStorage>())
          FusedLocAttrStorage(locations, metadata);
    }
      ::llvm::ArrayRef<Location> locations;
      Attribute metadata;
  };
} // namespace detail
FusedLoc FusedLoc::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Location> locations, Attribute metadata) {
  return Base::get(context, locations, metadata);
}
::llvm::ArrayRef<Location> FusedLoc::getLocations() const { return getImpl()->locations; }
Attribute FusedLoc::getMetadata() const { return getImpl()->metadata; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct NameLocAttrStorage : public ::mlir::AttributeStorage {
    NameLocAttrStorage (Identifier name, Location childLoc)
      : ::mlir::AttributeStorage(), name(name), childLoc(childLoc) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Identifier, Location>;
  bool operator==(const KeyTy &key) const {
    if (!(name == std::get<0>(key)))
      return false;
    if (!(childLoc == std::get<1>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static NameLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto name = std::get<0>(key);
      auto childLoc = std::get<1>(key);

      return new (allocator.allocate<NameLocAttrStorage>())
          NameLocAttrStorage(name, childLoc);
    }
      Identifier name;
      Location childLoc;
  };
} // namespace detail
NameLoc NameLoc::get(Identifier name, Location childLoc) {
  
      return Base::get(name.getContext(), name, childLoc);
    ;
}
NameLoc NameLoc::get(Identifier name) {
  
      return Base::get(name.getContext(), name,
                   UnknownLoc::get(name.getContext()));
    ;
}
Identifier NameLoc::getName() const { return getImpl()->name; }
Location NameLoc::getChildLoc() const { return getImpl()->childLoc; }
} // namespace mlir
namespace mlir {

namespace detail {
  struct OpaqueLocAttrStorage : public ::mlir::AttributeStorage {
    OpaqueLocAttrStorage (uintptr_t underlyingLocation, TypeID underlyingTypeID, Location fallbackLocation)
      : ::mlir::AttributeStorage(), underlyingLocation(underlyingLocation), underlyingTypeID(underlyingTypeID), fallbackLocation(fallbackLocation) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<uintptr_t, TypeID, Location>;
  bool operator==(const KeyTy &key) const {
    if (!(underlyingLocation == std::get<0>(key)))
      return false;
    if (!(underlyingTypeID == std::get<1>(key)))
      return false;
    if (!(fallbackLocation == std::get<2>(key)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &key) {
      return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static OpaqueLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &key) {
      auto underlyingLocation = std::get<0>(key);
      auto underlyingTypeID = std::get<1>(key);
      auto fallbackLocation = std::get<2>(key);

      return new (allocator.allocate<OpaqueLocAttrStorage>())
          OpaqueLocAttrStorage(underlyingLocation, underlyingTypeID, fallbackLocation);
    }
      uintptr_t underlyingLocation;
      TypeID underlyingTypeID;
      Location fallbackLocation;
  };
} // namespace detail
OpaqueLoc OpaqueLoc::get(uintptr_t underlyingLocation, TypeID underlyingTypeID, Location fallbackLocation) {
  
      return Base::get(fallbackLocation->getContext(), underlyingLocation,
                   underlyingTypeID, fallbackLocation);
    ;
}
uintptr_t OpaqueLoc::getUnderlyingLocation() const { return getImpl()->underlyingLocation; }
TypeID OpaqueLoc::getUnderlyingTypeID() const { return getImpl()->underlyingTypeID; }
Location OpaqueLoc::getFallbackLocation() const { return getImpl()->fallbackLocation; }
} // namespace mlir
namespace mlir {
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

