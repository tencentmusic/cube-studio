/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
// processor for loop mapping
enum class Processor : uint64_t {
  BlockX = 0,
  BlockY = 1,
  BlockZ = 2,
  ThreadX = 3,
  ThreadY = 4,
  ThreadZ = 5,
  Sequential = 6,
};

::llvm::Optional<Processor> symbolizeProcessor(uint64_t);
::llvm::StringRef stringifyProcessor(Processor);
::llvm::Optional<Processor> symbolizeProcessor(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForProcessor() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(Processor enumValue) {
  return stringifyProcessor(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Processor> symbolizeEnum<Processor>(::llvm::StringRef str) {
  return symbolizeProcessor(str);
}

class ProcessorAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Processor;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ProcessorAttr get(::mlir::MLIRContext *context, Processor val);
  Processor getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Processor> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::Processor getEmptyKey() {
    return static_cast<::mlir::gpu::Processor>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Processor getTombstoneKey() {
    return static_cast<::mlir::gpu::Processor>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Processor &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Processor &lhs, const ::mlir::gpu::Processor &rhs) {
    return lhs == rhs;
  }
};
}

