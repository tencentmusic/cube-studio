/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
  class BFloat16Type;
  class ComplexType;
  class Float128Type;
  class Float16Type;
  class Float32Type;
  class Float64Type;
  class Float80Type;
  class FunctionType;
  class IndexType;
  class IntegerType;
  class MemRefType;
  class NoneType;
  class OpaqueType;
  class RankedTensorType;
  class TupleType;
  class UnrankedMemRefType;
  class UnrankedTensorType;
  class VectorType;

  class BFloat16Type : public ::mlir::Type::TypeBase<BFloat16Type, ::mlir::FloatType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static BFloat16Type get(MLIRContext *context);
  
  };

  namespace detail {
    struct ComplexTypeStorage;
  } // end namespace detail
  class ComplexType : public ::mlir::Type::TypeBase<ComplexType, ::mlir::Type,
                                         detail::ComplexTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static ComplexType get(Type elementType);
    static ComplexType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type elementType);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
    Type getElementType() const;
  };

  class Float128Type : public ::mlir::Type::TypeBase<Float128Type, ::mlir::FloatType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static Float128Type get(MLIRContext *context);
  
  };

  class Float16Type : public ::mlir::Type::TypeBase<Float16Type, ::mlir::FloatType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static Float16Type get(MLIRContext *context);
  
  };

  class Float32Type : public ::mlir::Type::TypeBase<Float32Type, ::mlir::FloatType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static Float32Type get(MLIRContext *context);
  
  };

  class Float64Type : public ::mlir::Type::TypeBase<Float64Type, ::mlir::FloatType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static Float64Type get(MLIRContext *context);
  
  };

  class Float80Type : public ::mlir::Type::TypeBase<Float80Type, ::mlir::FloatType, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static Float80Type get(MLIRContext *context);
  
  };

  namespace detail {
    struct FunctionTypeStorage;
  } // end namespace detail
  class FunctionType : public ::mlir::Type::TypeBase<FunctionType, ::mlir::Type,
                                         detail::FunctionTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }

    /// Returns a new function type without the specified arguments and results.
    FunctionType getWithoutArgsAndResults(ArrayRef<unsigned> argIndices,
                                          ArrayRef<unsigned> resultIndices);
  
    static FunctionType get(::mlir::MLIRContext *context, TypeRange inputs, TypeRange results);
    ArrayRef<Type> getInputs() const;
    ArrayRef<Type> getResults() const;
  };

  class IndexType : public ::mlir::Type::TypeBase<IndexType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static IndexType get(MLIRContext *context);

    /// Storage bit width used for IndexType by internal compiler data
    /// structures.
    static constexpr unsigned kInternalStorageBitWidth = 64;
  
  };

  namespace detail {
    struct IntegerTypeStorage;
  } // end namespace detail
  class IntegerType : public ::mlir::Type::TypeBase<IntegerType, ::mlir::Type,
                                         detail::IntegerTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    /// Signedness semantics.
    enum SignednessSemantics : uint32_t {
      Signless, /// No signedness semantics
      Signed,   /// Signed integer
      Unsigned, /// Unsigned integer
    };

    /// Return true if this is a signless integer type.
    bool isSignless() const { return getSignedness() == Signless; }
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getSignedness() == Signed; }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return getSignedness() == Unsigned; }

    /// Get or create a new IntegerType with the same signedness as `this` and a
    /// bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    IntegerType scaleElementBitwidth(unsigned scale);

    /// Integer representation maximal bitwidth.
    /// Note: This is aligned with the maximum width of llvm::IntegerType.
    static constexpr unsigned kMaxWidth = (1 << 24) - 1;
  
    static IntegerType get(::mlir::MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);
    static IntegerType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, SignednessSemantics signedness);
    unsigned getWidth() const;
    SignednessSemantics getSignedness() const;
  };

  namespace detail {
    struct MemRefTypeStorage;
  } // end namespace detail
  class MemRefType : public ::mlir::Type::TypeBase<MemRefType, BaseMemRefType,
                                         detail::MemRefTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must out-live the builder.
    class Builder;

    /// [deprecated] Returns the memory space in old raw integer representation.
    /// New `Attribute getMemorySpace()` method should be used instead.
    unsigned getMemorySpaceAsInt() const;

    // TODO: merge these two special values in a single one used everywhere.
    // Unfortunately, uses of `-1` have crept deep into the codebase now and are
    // hard to track.
    static int64_t getDynamicStrideOrOffset() {
      return ShapedType::kDynamicStrideOrOffset;
    }
  
    static MemRefType get(ArrayRef<int64_t> shape, Type elementType, ArrayRef<AffineMap> affineMaps = {}, Attribute memorySpace = {});
    static MemRefType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, ArrayRef<AffineMap> affineMaps = {}, Attribute memorySpace = {});
    static MemRefType get(ArrayRef<int64_t> shape, Type elementType, ArrayRef<AffineMap> affineMaps, unsigned memorySpace);
    static MemRefType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, ArrayRef<AffineMap> affineMaps, unsigned memorySpace);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, ::llvm::ArrayRef<AffineMap> affineMaps, Attribute memorySpace);
    ::llvm::ArrayRef<int64_t> getShape() const;
    Type getElementType() const;
    ::llvm::ArrayRef<AffineMap> getAffineMaps() const;
    Attribute getMemorySpace() const;
  };

  class NoneType : public ::mlir::Type::TypeBase<NoneType, ::mlir::Type, ::mlir::TypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static NoneType get(MLIRContext *context);
  
  };

  namespace detail {
    struct OpaqueTypeStorage;
  } // end namespace detail
  class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type,
                                         detail::OpaqueTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static OpaqueType get(Identifier dialectNamespace, StringRef typeData = {});
    static OpaqueType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Identifier dialectNamespace, StringRef typeData = {});

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Identifier dialectNamespace, ::llvm::StringRef typeData);
    Identifier getDialectNamespace() const;
    ::llvm::StringRef getTypeData() const;
  };

  namespace detail {
    struct RankedTensorTypeStorage;
  } // end namespace detail
  class RankedTensorType : public ::mlir::Type::TypeBase<RankedTensorType, TensorType,
                                         detail::RankedTensorTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType);
    static RankedTensorType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType);
    ::llvm::ArrayRef<int64_t> getShape() const;
    Type getElementType() const;
  };

  namespace detail {
    struct TupleTypeStorage;
  } // end namespace detail
  class TupleType : public ::mlir::Type::TypeBase<TupleType, ::mlir::Type,
                                         detail::TupleTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    /// Accumulate the types contained in this tuple and tuples nested within
    /// it. Note that this only flattens nested tuples, not any other container
    /// type, e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is
    /// flattened to (i32, tensor<i32>, f32, i64)
    void getFlattenedTypes(SmallVectorImpl<Type> &types);

    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<Type>::iterator;
    iterator begin() const { return getTypes().begin(); }
    iterator end() const { return getTypes().end(); }

    /// Return the element type at index 'index'.
    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getTypes()[index];
    }
  
    static TupleType get(::mlir::MLIRContext *context, TypeRange elementTypes);
    static TupleType get(::mlir::MLIRContext *context);
    ArrayRef<Type> getTypes() const;
  };

  namespace detail {
    struct UnrankedMemRefTypeStorage;
  } // end namespace detail
  class UnrankedMemRefType : public ::mlir::Type::TypeBase<UnrankedMemRefType, BaseMemRefType,
                                         detail::UnrankedMemRefTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    ArrayRef<int64_t> getShape() const { return llvm::None; }

    /// [deprecated] Returns the memory space in old raw integer representation.
    /// New `Attribute getMemorySpace()` method should be used instead.
    unsigned getMemorySpaceAsInt() const;
  
    static UnrankedMemRefType get(Type elementType, Attribute memorySpace);
    static UnrankedMemRefType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace);
    static UnrankedMemRefType get(Type elementType, unsigned memorySpace);
    static UnrankedMemRefType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned memorySpace);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace);
    Type getElementType() const;
    Attribute getMemorySpace() const;
  };

  namespace detail {
    struct UnrankedTensorTypeStorage;
  } // end namespace detail
  class UnrankedTensorType : public ::mlir::Type::TypeBase<UnrankedTensorType, TensorType,
                                         detail::UnrankedTensorTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    ArrayRef<int64_t> getShape() const { return llvm::None; }
  
    static UnrankedTensorType get(Type elementType);
    static UnrankedTensorType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type elementType);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
    Type getElementType() const;
  };

  namespace detail {
    struct VectorTypeStorage;
  } // end namespace detail
  class VectorType : public ::mlir::Type::TypeBase<VectorType, ShapedType,
                                         detail::VectorTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;


    /// Returns true of the given type can be used as an element of a vector
    /// type. In particular, vectors can consist of integer or float primitives.
    static bool isValidElementType(Type t) {
      return t.isa<IntegerType, FloatType>();
    }

    /// Get or create a new VectorType with the same shape as `this` and an
    /// element type of bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    VectorType scaleElementBitwidth(unsigned scale);
  
    static VectorType get(ArrayRef<int64_t> shape, Type elementType);
    static VectorType getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType);

    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType);
    ::llvm::ArrayRef<int64_t> getShape() const;
    Type getElementType() const;
  };
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

