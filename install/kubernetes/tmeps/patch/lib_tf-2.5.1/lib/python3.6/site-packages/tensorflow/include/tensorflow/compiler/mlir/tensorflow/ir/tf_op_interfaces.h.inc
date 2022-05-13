/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class FoldOperandsTransposeInterface;
namespace detail {
struct FoldOperandsTransposeInterfaceInterfaceTraits {
  struct Concept {
    SmallVector<unsigned, 4> (*GetLayoutDependentArgs)(::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentResults)(::mlir::Operation *);
    LogicalResult (*FoldOperandsPermutation)(::mlir::Operation *, ArrayRef<int64_t>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{GetLayoutDependentArgs, GetLayoutDependentResults, FoldOperandsPermutation} {}

    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult FoldOperandsPermutation(::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation);
  };
};
} // end namespace detail
class FoldOperandsTransposeInterface : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct FoldOperandsTransposeInterfaceTrait : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyFoldOperandsTransposeInterface(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public FoldOperandsTransposeInterfaceTrait<ConcreteOp> {};
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
};
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
}
template<typename ConcreteOp>
LogicalResult detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::FoldOperandsPermutation(::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).FoldOperandsPermutation(permutation);
}
class LayoutSensitiveInterface;
namespace detail {
struct LayoutSensitiveInterfaceInterfaceTraits {
  struct Concept {
    StringRef (*data_format)(::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentArgs)(::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentResults)(::mlir::Operation *);
    StringRef (*GetOptimalLayout)(::mlir::Operation *, const RuntimeDevices&);
    LogicalResult (*UpdateDataFormat)(::mlir::Operation *, StringRef);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{data_format, GetLayoutDependentArgs, GetLayoutDependentResults, GetOptimalLayout, UpdateDataFormat} {}

    static inline StringRef data_format(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val);
    static inline StringRef GetOptimalLayout(::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices);
    static inline LogicalResult UpdateDataFormat(::mlir::Operation *tablegen_opaque_val, StringRef data_format);
  };
};
} // end namespace detail
class LayoutSensitiveInterface : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct LayoutSensitiveInterfaceTrait : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyLayoutSensitiveInterface(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public LayoutSensitiveInterfaceTrait<ConcreteOp> {};
  StringRef data_format();
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  StringRef GetOptimalLayout(const RuntimeDevices& devices);
  LogicalResult UpdateDataFormat(StringRef data_format);
};
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::data_format(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).data_format();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetOptimalLayout(::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetOptimalLayout(devices);
}
template<typename ConcreteOp>
LogicalResult detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::UpdateDataFormat(::mlir::Operation *tablegen_opaque_val, StringRef data_format) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).UpdateDataFormat(data_format);
}
class ResourceHandleAllocatorInterface;
namespace detail {
struct ResourceHandleAllocatorInterfaceInterfaceTraits {
  struct Concept {
    ResourceHandleValueAndId (*GetResourceHandleValueAndId)(::mlir::Operation *, llvm::SmallDenseMap<ResourceHandle, int64_t>&, int64_t&);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{GetResourceHandleValueAndId} {}

    static inline ResourceHandleValueAndId GetResourceHandleValueAndId(::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
  };
};
} // end namespace detail
class ResourceHandleAllocatorInterface : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ResourceHandleAllocatorInterfaceTrait : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ResourceHandleAllocatorInterfaceTrait<ConcreteOp> {};
  ResourceHandleValueAndId GetResourceHandleValueAndId(llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
};
template<typename ConcreteOp>
ResourceHandleValueAndId detail::ResourceHandleAllocatorInterfaceInterfaceTraits::Model<ConcreteOp>::GetResourceHandleValueAndId(::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetResourceHandleValueAndId(resource_handle_id_map, next_id);
}
