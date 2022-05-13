/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class VectorTransferOpInterface;
namespace detail {
struct VectorTransferOpInterfaceInterfaceTraits {
  struct Concept {
    StringRef (*getMaskedAttrName)();
    StringRef (*getPermutationMapAttrName)();
    bool (*isMaskedDim)(::mlir::Operation *, unsigned);
    Value (*source)(::mlir::Operation *);
    Value (*vector)(::mlir::Operation *);
    ValueRange (*indices)(::mlir::Operation *);
    AffineMap (*permutation_map)(::mlir::Operation *);
    Optional<ArrayAttr> (*masked)(::mlir::Operation *);
    ShapedType (*getShapedType)(::mlir::Operation *);
    VectorType (*getVectorType)(::mlir::Operation *);
    unsigned (*getTransferRank)(::mlir::Operation *);
    unsigned (*getLeadingShapedRank)(::mlir::Operation *);
    bool (*hasMaskedDim)(::mlir::Operation *);
    void (*zipResultAndIndexing)(::mlir::Operation *, llvm::function_ref<void(int64_t, int64_t)>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getMaskedAttrName, getPermutationMapAttrName, isMaskedDim, source, vector, indices, permutation_map, masked, getShapedType, getVectorType, getTransferRank, getLeadingShapedRank, hasMaskedDim, zipResultAndIndexing} {}

    static inline StringRef getMaskedAttrName();
    static inline StringRef getPermutationMapAttrName();
    static inline bool isMaskedDim(::mlir::Operation *tablegen_opaque_val, unsigned dim);
    static inline Value source(::mlir::Operation *tablegen_opaque_val);
    static inline Value vector(::mlir::Operation *tablegen_opaque_val);
    static inline ValueRange indices(::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap permutation_map(::mlir::Operation *tablegen_opaque_val);
    static inline Optional<ArrayAttr> masked(::mlir::Operation *tablegen_opaque_val);
    static inline ShapedType getShapedType(::mlir::Operation *tablegen_opaque_val);
    static inline VectorType getVectorType(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getTransferRank(::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getLeadingShapedRank(::mlir::Operation *tablegen_opaque_val);
    static inline bool hasMaskedDim(::mlir::Operation *tablegen_opaque_val);
    static inline void zipResultAndIndexing(::mlir::Operation *tablegen_opaque_val, llvm::function_ref<void(int64_t, int64_t)> fun);
  };
};
} // end namespace detail
class VectorTransferOpInterface : public ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct VectorTransferOpInterfaceTrait : public ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static StringRef getMaskedAttrName() {
      return "masked";
    }
    static StringRef getPermutationMapAttrName() {
      return "permutation_map";
    }
    bool isMaskedDim(unsigned dim) {
      return !(*static_cast<ConcreteOp *>(this)).masked() ||
          (*static_cast<ConcreteOp *>(this)).masked()->template cast<ArrayAttr>()[dim]
                        .template cast<BoolAttr>().getValue();
    }
    ShapedType getShapedType() {
      return (*static_cast<ConcreteOp *>(this)).source().getType().template cast<ShapedType>();
    }
    VectorType getVectorType() {
      return (*static_cast<ConcreteOp *>(this)).vector().getType().template dyn_cast<VectorType>();
    }
    unsigned getTransferRank() {
      return (*static_cast<ConcreteOp *>(this)).permutation_map().getNumResults();
    }
    unsigned getLeadingShapedRank() {
      return (*static_cast<ConcreteOp *>(this)).getShapedType().getRank() - (*static_cast<ConcreteOp *>(this)).getTransferRank();
    }
    bool hasMaskedDim() {
      for (unsigned idx = 0, e = (*static_cast<ConcreteOp *>(this)).getTransferRank(); idx < e; ++idx)
          if ((*static_cast<ConcreteOp *>(this)).isMaskedDim(idx))
            return true;
        return false;
    }
    void zipResultAndIndexing(llvm::function_ref<void(int64_t, int64_t)> fun) {
      for (int64_t resultIdx = 0,
                   indicesIdx = (*static_cast<ConcreteOp *>(this)).getLeadingShapedRank(),
                   eResult = (*static_cast<ConcreteOp *>(this)).getTransferRank();
           resultIdx < eResult;
           ++resultIdx, ++indicesIdx)
        fun(resultIdx, indicesIdx);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public VectorTransferOpInterfaceTrait<ConcreteOp> {};
  StringRef getMaskedAttrName();
  StringRef getPermutationMapAttrName();
  bool isMaskedDim(unsigned dim);
  Value source();
  Value vector();
  ValueRange indices();
  AffineMap permutation_map();
  Optional<ArrayAttr> masked();
  ShapedType getShapedType();
  VectorType getVectorType();
  unsigned getTransferRank();
  unsigned getLeadingShapedRank();
  bool hasMaskedDim();
  void zipResultAndIndexing(llvm::function_ref<void(int64_t, int64_t)> fun);
};
template<typename ConcreteOp>
StringRef detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMaskedAttrName() {
  return ConcreteOp::getMaskedAttrName();
}
template<typename ConcreteOp>
StringRef detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPermutationMapAttrName() {
  return ConcreteOp::getPermutationMapAttrName();
}
template<typename ConcreteOp>
bool detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::isMaskedDim(::mlir::Operation *tablegen_opaque_val, unsigned dim) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isMaskedDim(dim);
}
template<typename ConcreteOp>
Value detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::source(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).source();
}
template<typename ConcreteOp>
Value detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::vector(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).vector();
}
template<typename ConcreteOp>
ValueRange detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::indices(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indices();
}
template<typename ConcreteOp>
AffineMap detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::permutation_map(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).permutation_map();
}
template<typename ConcreteOp>
Optional<ArrayAttr> detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::masked(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).masked();
}
template<typename ConcreteOp>
ShapedType detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getShapedType(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedType();
}
template<typename ConcreteOp>
VectorType detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getVectorType(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVectorType();
}
template<typename ConcreteOp>
unsigned detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTransferRank(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTransferRank();
}
template<typename ConcreteOp>
unsigned detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLeadingShapedRank(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLeadingShapedRank();
}
template<typename ConcreteOp>
bool detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::hasMaskedDim(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasMaskedDim();
}
template<typename ConcreteOp>
void detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::zipResultAndIndexing(::mlir::Operation *tablegen_opaque_val, llvm::function_ref<void(int64_t, int64_t)> fun) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).zipResultAndIndexing(fun);
}
} // namespace mlir
namespace mlir {
class VectorUnrollOpInterface;
namespace detail {
struct VectorUnrollOpInterfaceInterfaceTraits {
  struct Concept {
    Optional<SmallVector<int64_t, 4>> (*getShapeForUnroll)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getShapeForUnroll} {}

    static inline Optional<SmallVector<int64_t, 4>> getShapeForUnroll(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class VectorUnrollOpInterface : public ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct VectorUnrollOpInterfaceTrait : public ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Optional<SmallVector<int64_t, 4>> getShapeForUnroll() {
      assert((*static_cast<ConcreteOp *>(this))->getNumResults() == 1);
        auto vt = (*static_cast<ConcreteOp *>(this)).getResult().getType().
          template dyn_cast<VectorType>();
        if (!vt)
          return None;
        SmallVector<int64_t, 4> res(vt.getShape().begin(), vt.getShape().end());
        return res;
    }
  };
  template <typename ConcreteOp>
  struct Trait : public VectorUnrollOpInterfaceTrait<ConcreteOp> {};
  Optional<SmallVector<int64_t, 4>> getShapeForUnroll();
};
template<typename ConcreteOp>
Optional<SmallVector<int64_t, 4>> detail::VectorUnrollOpInterfaceInterfaceTraits::Model<ConcreteOp>::getShapeForUnroll(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapeForUnroll();
}
} // namespace mlir
