/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
  class CallSiteLoc;
  class FileLineColLoc;
  class FusedLoc;
  class NameLoc;
  class OpaqueLoc;
  class UnknownLoc;

  namespace detail {
    struct CallSiteLocAttrStorage;
  } // end namespace detail
  class CallSiteLoc : public ::mlir::Attribute::AttrBase<CallSiteLoc, ::mlir::LocationAttr,
                                         detail::CallSiteLocAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static CallSiteLoc get(Location callee, Location caller);
    static CallSiteLoc get(Location name, ArrayRef<Location> frames);
    Location getCallee() const;
    Location getCaller() const;
  };

  namespace detail {
    struct FileLineColLocAttrStorage;
  } // end namespace detail
  class FileLineColLoc : public ::mlir::Attribute::AttrBase<FileLineColLoc, ::mlir::LocationAttr,
                                         detail::FileLineColLocAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static FileLineColLoc get(Identifier filename, unsigned line, unsigned column);
    static FileLineColLoc get(::mlir::MLIRContext *context, StringRef filename, unsigned line, unsigned column);
    Identifier getFilename() const;
    unsigned getLine() const;
    unsigned getColumn() const;
  };

  namespace detail {
    struct FusedLocAttrStorage;
  } // end namespace detail
  class FusedLoc : public ::mlir::Attribute::AttrBase<FusedLoc, ::mlir::LocationAttr,
                                         detail::FusedLocAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;


    static Location get(ArrayRef<Location> locs, Attribute metadata,
                        MLIRContext *context);
    static Location get(MLIRContext *context, ArrayRef<Location> locs) {
      return get(locs, Attribute(), context);
    }
  
    static FusedLoc get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Location> locations, Attribute metadata);
    ::llvm::ArrayRef<Location> getLocations() const;
    Attribute getMetadata() const;
  };

  namespace detail {
    struct NameLocAttrStorage;
  } // end namespace detail
  class NameLoc : public ::mlir::Attribute::AttrBase<NameLoc, ::mlir::LocationAttr,
                                         detail::NameLocAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static NameLoc get(Identifier name, Location childLoc);
    static NameLoc get(Identifier name);
    Identifier getName() const;
    Location getChildLoc() const;
  };

  namespace detail {
    struct OpaqueLocAttrStorage;
  } // end namespace detail
  class OpaqueLoc : public ::mlir::Attribute::AttrBase<OpaqueLoc, ::mlir::LocationAttr,
                                         detail::OpaqueLocAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;


    /// Returns an instance of opaque location which contains a given pointer to
    /// an object. The corresponding MLIR location is set to UnknownLoc.
    template <typename T>
    static Location get(T underlyingLocation, MLIRContext *context);

    /// Returns an instance of opaque location which contains a given pointer to
    /// an object and an additional MLIR location.
    template <typename T>
    static Location get(T underlyingLocation, Location fallbackLocation) {
      return get(reinterpret_cast<uintptr_t>(underlyingLocation),
                 TypeID::get<T>(), fallbackLocation);
    }

    /// Returns a pointer to some data structure that opaque location stores.
    template <typename T> static T getUnderlyingLocation(Location location) {
      assert(isa<T>(location));
      return reinterpret_cast<T>(
          location.cast<mlir::OpaqueLoc>().getUnderlyingLocation());
    }

    /// Returns a pointer to some data structure that opaque location stores.
    /// Returns nullptr if provided location is not opaque location or if it
    /// contains a pointer of different type.
    template <typename T>
    static T getUnderlyingLocationOrNull(Location location) {
      return isa<T>(location)
                 ? reinterpret_cast<T>(
                       location.cast<mlir::OpaqueLoc>().getUnderlyingLocation())
                 : T(nullptr);
    }

    /// Checks whether provided location is opaque location and contains a
    /// pointer to an object of particular type.
    template <typename T> static bool isa(Location location) {
      auto opaque_loc = location.dyn_cast<OpaqueLoc>();
      return opaque_loc && opaque_loc.getUnderlyingTypeID() == TypeID::get<T>();
    }
  
    static OpaqueLoc get(uintptr_t underlyingLocation, TypeID underlyingTypeID, Location fallbackLocation);
    uintptr_t getUnderlyingLocation() const;
    TypeID getUnderlyingTypeID() const;
    Location getFallbackLocation() const;
  };

  class UnknownLoc : public ::mlir::Attribute::AttrBase<UnknownLoc, ::mlir::LocationAttr, ::mlir::AttributeStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static UnknownLoc get(MLIRContext *context);
  
  };
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

