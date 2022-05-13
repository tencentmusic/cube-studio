/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class OffsetSizeAndStrideOpInterface;
namespace detail {
struct OffsetSizeAndStrideOpInterfaceInterfaceTraits {
  struct Concept {
    unsigned (*getOffsetSizeAndStrideStartOperandIndex)(::mlir::Operation *);
    std::array<unsigned, 3> (*getArrayAttrMaxRanks)(::mlir::Operation *);
    ::mlir::OperandRange (*offsets)(::mlir::Operation *);
    ::mlir::OperandRange (*sizes)(::mlir::Operation *);
    ::mlir::OperandRange (*strides)(::mlir::Operation *);
    ::mlir::ArrayAttr (*static_offsets)(::mlir::Operation *);
    ::mlir::ArrayAttr (*static_sizes)(::mlir::Operation *);
    ::mlir::ArrayAttr (*static_strides)(::mlir::Operation *);
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> (*getMixedOffsets)(::mlir::Operation *);
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> (*getMixedSizes)(::mlir::Operation *);
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> (*getMixedStrides)(::mlir::Operation *);
    bool (*isDynamicOffset)(::mlir::Operation *, unsigned);
    bool (*isDynamicSize)(::mlir::Operation *, unsigned);
    bool (*isDynamicStride)(::mlir::Operation *, unsigned);
    int64_t (*getStaticOffset)(::mlir::Operation *, unsigned);
    int64_t (*getStaticSize)(::mlir::Operation *, unsigned);
    int64_t (*getStaticStride)(::mlir::Operation *, unsigned);
    unsigned (*getIndexOfDynamicOffset)(::mlir::Operation *, unsigned);
    unsigned (*getIndexOfDynamicSize)(::mlir::Operation *, unsigned);
    unsigned (*getIndexOfDynamicStride)(::mlir::Operation *, unsigned);
    unsigned (*getNumDynamicEntriesUpToIdx)(::mlir::Operation *, ::mlir::ArrayAttr, ::llvm::function_ref<bool(int64_t)>, unsigned);
    ::mlir::Value (*getDynamicOffset)(::mlir::Operation *, unsigned);
    ::mlir::Value (*getDynamicSize)(::mlir::Operation *, unsigned);
    ::mlir::Value (*getDynamicStride)(::mlir::Operation *, unsigned);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getOffsetSizeAndStrideStartOperandIndex, getArrayAttrMaxRanks, offsets, sizes, strides, static_offsets, static_sizes, static_strides, getMixedOffsets, getMixedSizes, getMixedStrides, isDynamicOffset, isDynamicSize, isDynamicStride, getStaticOffset, getStaticSize, getStaticStride, getIndexOfDynamicOffset, getIndexOfDynamicSize, getIndexOfDynamicStride, getNumDynamicEntriesUpToIdx, getDynamicOffset, getDynamicSize, getDynamicStride} {}

    static inline unsigned getOffsetSizeAndStrideStartOperandIndex(::mlir::Operation *tablegen_opaque_val);
    static inline std::array<unsigned, 3> getArrayAttrMaxRanks(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange offsets(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange sizes(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange strides(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_offsets(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_sizes(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_strides(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes(::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides(::mlir::Operation *tablegen_opaque_val);
    static inline bool isDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticSize(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticStride(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getNumDynamicEntriesUpToIdx(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx);
    static inline ::mlir::Value getDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline ::mlir::Value getDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline ::mlir::Value getDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx);
  };
};
} // end namespace detail
class OffsetSizeAndStrideOpInterface : public ::mlir::OpInterface<OffsetSizeAndStrideOpInterface, detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OffsetSizeAndStrideOpInterface, detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct OffsetSizeAndStrideOpInterfaceTrait : public ::mlir::OpInterface<OffsetSizeAndStrideOpInterface, detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::OperandRange offsets() {
      return (*static_cast<ConcreteOp *>(this)).offsets();
    }
    ::mlir::OperandRange sizes() {
      return (*static_cast<ConcreteOp *>(this)).sizes();
    }
    ::mlir::OperandRange strides() {
      return (*static_cast<ConcreteOp *>(this)).strides();
    }
    ::mlir::ArrayAttr static_offsets() {
      return (*static_cast<ConcreteOp *>(this)).static_offsets();
    }
    ::mlir::ArrayAttr static_sizes() {
      return (*static_cast<ConcreteOp *>(this)).static_sizes();
    }
    ::mlir::ArrayAttr static_strides() {
      return (*static_cast<ConcreteOp *>(this)).static_strides();
    }
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets() {
      ::mlir::SmallVector<::mlir::OpFoldResult, 4> res;
        unsigned numDynamic = 0;
        unsigned count = (*static_cast<ConcreteOp *>(this)).static_offsets().size();
        for (unsigned idx = 0; idx < count; ++idx) {
          if (isDynamicOffset(idx))
            res.push_back((*static_cast<ConcreteOp *>(this)).offsets()[numDynamic++]);
          else
            res.push_back((*static_cast<ConcreteOp *>(this)).static_offsets()[idx]);
        }
        return res;
    }
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      ::mlir::SmallVector<::mlir::OpFoldResult, 4> res;
        unsigned numDynamic = 0;
        unsigned count = (*static_cast<ConcreteOp *>(this)).static_sizes().size();
        for (unsigned idx = 0; idx < count; ++idx) {
          if (isDynamicSize(idx))
            res.push_back((*static_cast<ConcreteOp *>(this)).sizes()[numDynamic++]);
          else
            res.push_back((*static_cast<ConcreteOp *>(this)).static_sizes()[idx]);
        }
        return res;
    }
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides() {
      ::mlir::SmallVector<::mlir::OpFoldResult, 4> res;
        unsigned numDynamic = 0;
        unsigned count = (*static_cast<ConcreteOp *>(this)).static_strides().size();
        for (unsigned idx = 0; idx < count; ++idx) {
          if (isDynamicStride(idx))
            res.push_back((*static_cast<ConcreteOp *>(this)).strides()[numDynamic++]);
          else
            res.push_back((*static_cast<ConcreteOp *>(this)).static_strides()[idx]);
        }
        return res;
    }
    bool isDynamicOffset(unsigned idx) {
      ::llvm::APInt v = *(static_offsets()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
    }
    bool isDynamicSize(unsigned idx) {
      ::llvm::APInt v = *(static_sizes()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamic(v.getSExtValue());
    }
    bool isDynamicStride(unsigned idx) {
      ::llvm::APInt v = *(static_strides()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
    }
    int64_t getStaticOffset(unsigned idx) {
      assert(!(*static_cast<ConcreteOp *>(this)).isDynamicOffset(idx) && "expected static offset");
        ::llvm::APInt v = *(static_offsets().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }
    int64_t getStaticSize(unsigned idx) {
      assert(!(*static_cast<ConcreteOp *>(this)).isDynamicSize(idx) && "expected static size");
        ::llvm::APInt v = *(static_sizes().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }
    int64_t getStaticStride(unsigned idx) {
      assert(!(*static_cast<ConcreteOp *>(this)).isDynamicStride(idx) && "expected static stride");
        ::llvm::APInt v = *(static_strides().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }
    unsigned getIndexOfDynamicOffset(unsigned idx) {
      assert((*static_cast<ConcreteOp *>(this)).isDynamicOffset(idx) && "expected dynamic offset");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_offsets().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return (*static_cast<ConcreteOp *>(this)).getOffsetSizeAndStrideStartOperandIndex() + numDynamic;
    }
    unsigned getIndexOfDynamicSize(unsigned idx) {
      assert((*static_cast<ConcreteOp *>(this)).isDynamicSize(idx) && "expected dynamic size");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_sizes().template cast<::mlir::ArrayAttr>(), ::mlir::ShapedType::isDynamic, idx);
        return (*static_cast<ConcreteOp *>(this)).getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + numDynamic;
    }
    unsigned getIndexOfDynamicStride(unsigned idx) {
      assert((*static_cast<ConcreteOp *>(this)).isDynamicStride(idx) && "expected dynamic stride");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_strides().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return (*static_cast<ConcreteOp *>(this)).getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + sizes().size() + numDynamic;
    }
    unsigned getNumDynamicEntriesUpToIdx(::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
      return std::count_if(
          attr.getValue().begin(), attr.getValue().begin() + idx,
          [&](::mlir::Attribute attr) {
            return isDynamic(attr.cast<::mlir::IntegerAttr>().getInt());
          });
    }
    ::mlir::Value getDynamicOffset(unsigned idx) {
      return (*static_cast<ConcreteOp *>(this)).getOperand(getIndexOfDynamicOffset(idx));
    }
    ::mlir::Value getDynamicSize(unsigned idx) {
      return (*static_cast<ConcreteOp *>(this)).getOperand(getIndexOfDynamicSize(idx));
    }
    ::mlir::Value getDynamicStride(unsigned idx) {
      return (*static_cast<ConcreteOp *>(this)).getOperand(getIndexOfDynamicStride(idx));
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::verify(::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(op));
    }
  };
  template <typename ConcreteOp>
  struct Trait : public OffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {};
  unsigned getOffsetSizeAndStrideStartOperandIndex();
  std::array<unsigned, 3> getArrayAttrMaxRanks();
  ::mlir::OperandRange offsets();
  ::mlir::OperandRange sizes();
  ::mlir::OperandRange strides();
  ::mlir::ArrayAttr static_offsets();
  ::mlir::ArrayAttr static_sizes();
  ::mlir::ArrayAttr static_strides();
  ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets();
  ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes();
  ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides();
  bool isDynamicOffset(unsigned idx);
  bool isDynamicSize(unsigned idx);
  bool isDynamicStride(unsigned idx);
  int64_t getStaticOffset(unsigned idx);
  int64_t getStaticSize(unsigned idx);
  int64_t getStaticStride(unsigned idx);
  unsigned getIndexOfDynamicOffset(unsigned idx);
  unsigned getIndexOfDynamicSize(unsigned idx);
  unsigned getIndexOfDynamicStride(unsigned idx);
  unsigned getNumDynamicEntriesUpToIdx(::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx);
  ::mlir::Value getDynamicOffset(unsigned idx);
  ::mlir::Value getDynamicSize(unsigned idx);
  ::mlir::Value getDynamicStride(unsigned idx);

    static unsigned getOffsetOperandGroupPosition() { return 0; }
    static unsigned getSizeOperandGroupPosition() { return 1; }
    static unsigned getStrideOperandGroupPosition() { return 2; }
    static ::llvm::StringRef getStaticOffsetsAttrName() {
      return "static_offsets";
    }
    static ::llvm::StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }
    static ::llvm::StringRef getStaticStridesAttrName() {
      return "static_strides";
    }
    static ::llvm::ArrayRef<::llvm::StringRef> getSpecialAttrNames() {
      static ::llvm::SmallVector<::llvm::StringRef, 4> names{
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName(),
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticSizesAttrName(),
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticStridesAttrName(),
        ::mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()};
      return names;
   }
  
};
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOffsetSizeAndStrideStartOperandIndex(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsetSizeAndStrideStartOperandIndex();
}
template<typename ConcreteOp>
std::array<unsigned, 3> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArrayAttrMaxRanks(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArrayAttrMaxRanks();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::offsets(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).offsets();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::sizes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).sizes();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::strides(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).strides();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::static_offsets(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).static_offsets();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::static_sizes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).static_sizes();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::static_strides(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).static_strides();
}
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMixedOffsets(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMixedOffsets();
}
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMixedSizes(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMixedSizes();
}
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMixedStrides(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMixedStrides();
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicOffset(idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicSize(idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicStride(idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStaticOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticOffset(idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStaticSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticSize(idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStaticStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticStride(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexOfDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfDynamicOffset(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexOfDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfDynamicSize(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexOfDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfDynamicStride(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumDynamicEntriesUpToIdx(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDynamicEntriesUpToIdx(attr, isDynamic, idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDynamicOffset(idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDynamicSize(idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDynamicStride(idx);
}
} // namespace mlir
namespace mlir {
class ViewLikeOpInterface;
namespace detail {
struct ViewLikeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Value (*getViewSource)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getViewSource} {}

    static inline ::mlir::Value getViewSource(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class ViewLikeOpInterface : public ::mlir::OpInterface<ViewLikeOpInterface, detail::ViewLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ViewLikeOpInterface, detail::ViewLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ViewLikeOpInterfaceTrait : public ::mlir::OpInterface<ViewLikeOpInterface, detail::ViewLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ViewLikeOpInterfaceTrait<ConcreteOp> {};
  ::mlir::Value getViewSource();
};
template<typename ConcreteOp>
::mlir::Value detail::ViewLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getViewSource(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getViewSource();
}
} // namespace mlir
