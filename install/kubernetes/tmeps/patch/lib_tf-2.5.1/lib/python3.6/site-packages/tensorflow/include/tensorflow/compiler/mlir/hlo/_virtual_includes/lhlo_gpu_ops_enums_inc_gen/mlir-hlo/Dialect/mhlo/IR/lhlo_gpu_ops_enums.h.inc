/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {
// Activation applied with fused convolution
enum class Activation {
  None,
  Sigmoid,
  Tanh,
  Relu,
  Relu6,
  ReluX,
  BandPass,
};

::llvm::StringRef stringifyActivation(Activation);
::llvm::Optional<Activation> symbolizeActivation(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(Activation enumValue) {
  return stringifyActivation(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Activation> symbolizeEnum<Activation>(::llvm::StringRef str) {
  return symbolizeActivation(str);
}
} // namespace lmhlo_gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::lmhlo_gpu::Activation> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::lmhlo_gpu::Activation>::type>;

  static inline ::mlir::lmhlo_gpu::Activation getEmptyKey() {
    return static_cast<::mlir::lmhlo_gpu::Activation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::lmhlo_gpu::Activation getTombstoneKey() {
    return static_cast<::mlir::lmhlo_gpu::Activation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::lmhlo_gpu::Activation &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::lmhlo_gpu::Activation>::type>(val));
  }

  static bool isEqual(const ::mlir::lmhlo_gpu::Activation &lhs, const ::mlir::lmhlo_gpu::Activation &rhs) {
    return lhs == rhs;
  }
};
}

