/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mhlo {
// Which comparison operation to perform.
enum class ComparisonDirection {
  EQ,
  NE,
  GE,
  GT,
  LE,
  LT,
};

::llvm::StringRef stringifyComparisonDirection(ComparisonDirection);
::llvm::Optional<ComparisonDirection> symbolizeComparisonDirection(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ComparisonDirection enumValue) {
  return stringifyComparisonDirection(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ComparisonDirection> symbolizeEnum<ComparisonDirection>(::llvm::StringRef str) {
  return symbolizeComparisonDirection(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::ComparisonDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::mhlo::ComparisonDirection>::type>;

  static inline ::mlir::mhlo::ComparisonDirection getEmptyKey() {
    return static_cast<::mlir::mhlo::ComparisonDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::ComparisonDirection getTombstoneKey() {
    return static_cast<::mlir::mhlo::ComparisonDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::ComparisonDirection &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::mhlo::ComparisonDirection>::type>(val));
  }

  static bool isEqual(const ::mlir::mhlo::ComparisonDirection &lhs, const ::mlir::mhlo::ComparisonDirection &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// Which comparison type to use.
enum class ComparisonType {
  FLOAT,
  TOTALORDER,
  SIGNED,
  UNSIGNED,
};

::llvm::StringRef stringifyComparisonType(ComparisonType);
::llvm::Optional<ComparisonType> symbolizeComparisonType(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ComparisonType enumValue) {
  return stringifyComparisonType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ComparisonType> symbolizeEnum<ComparisonType>(::llvm::StringRef str) {
  return symbolizeComparisonType(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::ComparisonType> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::mhlo::ComparisonType>::type>;

  static inline ::mlir::mhlo::ComparisonType getEmptyKey() {
    return static_cast<::mlir::mhlo::ComparisonType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::ComparisonType getTombstoneKey() {
    return static_cast<::mlir::mhlo::ComparisonType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::ComparisonType &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::mhlo::ComparisonType>::type>(val));
  }

  static bool isEqual(const ::mlir::mhlo::ComparisonType &lhs, const ::mlir::mhlo::ComparisonType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// Dequantization mode. Only MIN_COMBINED is supported.
enum class DequantizeMode {
  MIN_COMBINED,
};

::llvm::StringRef stringifyDequantizeMode(DequantizeMode);
::llvm::Optional<DequantizeMode> symbolizeDequantizeMode(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(DequantizeMode enumValue) {
  return stringifyDequantizeMode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DequantizeMode> symbolizeEnum<DequantizeMode>(::llvm::StringRef str) {
  return symbolizeDequantizeMode(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::DequantizeMode> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::mhlo::DequantizeMode>::type>;

  static inline ::mlir::mhlo::DequantizeMode getEmptyKey() {
    return static_cast<::mlir::mhlo::DequantizeMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::DequantizeMode getTombstoneKey() {
    return static_cast<::mlir::mhlo::DequantizeMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::DequantizeMode &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::mhlo::DequantizeMode>::type>(val));
  }

  static bool isEqual(const ::mlir::mhlo::DequantizeMode &lhs, const ::mlir::mhlo::DequantizeMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// XLA fast fourier transform type.
enum class FftType {
  FFT,
  IFFT,
  RFFT,
  IRFFT,
};

::llvm::StringRef stringifyFftType(FftType);
::llvm::Optional<FftType> symbolizeFftType(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(FftType enumValue) {
  return stringifyFftType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FftType> symbolizeEnum<FftType>(::llvm::StringRef str) {
  return symbolizeFftType(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::FftType> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::mhlo::FftType>::type>;

  static inline ::mlir::mhlo::FftType getEmptyKey() {
    return static_cast<::mlir::mhlo::FftType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::FftType getTombstoneKey() {
    return static_cast<::mlir::mhlo::FftType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::FftType &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::mhlo::FftType>::type>(val));
  }

  static bool isEqual(const ::mlir::mhlo::FftType &lhs, const ::mlir::mhlo::FftType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// XLA precision for an operand. Has backend specific meaning.
enum class Precision {
  DEFAULT,
  HIGH,
  HIGHEST,
};

::llvm::StringRef stringifyPrecision(Precision);
::llvm::Optional<Precision> symbolizePrecision(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(Precision enumValue) {
  return stringifyPrecision(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Precision> symbolizeEnum<Precision>(::llvm::StringRef str) {
  return symbolizePrecision(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::Precision> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::mhlo::Precision>::type>;

  static inline ::mlir::mhlo::Precision getEmptyKey() {
    return static_cast<::mlir::mhlo::Precision>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::Precision getTombstoneKey() {
    return static_cast<::mlir::mhlo::Precision>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::Precision &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::mhlo::Precision>::type>(val));
  }

  static bool isEqual(const ::mlir::mhlo::Precision &lhs, const ::mlir::mhlo::Precision &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// Transpose options
enum class Transpose {
  TRANSPOSE_INVALID,
  NO_TRANSPOSE,
  TRANSPOSE,
  ADJOINT,
};

::llvm::StringRef stringifyTranspose(Transpose);
::llvm::Optional<Transpose> symbolizeTranspose(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(Transpose enumValue) {
  return stringifyTranspose(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Transpose> symbolizeEnum<Transpose>(::llvm::StringRef str) {
  return symbolizeTranspose(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::Transpose> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::mhlo::Transpose>::type>;

  static inline ::mlir::mhlo::Transpose getEmptyKey() {
    return static_cast<::mlir::mhlo::Transpose>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::Transpose getTombstoneKey() {
    return static_cast<::mlir::mhlo::Transpose>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::Transpose &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::mhlo::Transpose>::type>(val));
  }

  static bool isEqual(const ::mlir::mhlo::Transpose &lhs, const ::mlir::mhlo::Transpose &rhs) {
    return lhs == rhs;
  }
};
}

