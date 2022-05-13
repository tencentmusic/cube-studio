/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
// DefaultValue Clause
enum class ClauseDefaultValue {
  present,
  none,
};

::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue);
::llvm::Optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ClauseDefaultValue enumValue) {
  return stringifyClauseDefaultValue(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseDefaultValue> symbolizeEnum<ClauseDefaultValue>(::llvm::StringRef str) {
  return symbolizeClauseDefaultValue(str);
}
} // namespace acc
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ClauseDefaultValue> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::acc::ClauseDefaultValue>::type>;

  static inline ::mlir::acc::ClauseDefaultValue getEmptyKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ClauseDefaultValue getTombstoneKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ClauseDefaultValue &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::acc::ClauseDefaultValue>::type>(val));
  }

  static bool isEqual(const ::mlir::acc::ClauseDefaultValue &lhs, const ::mlir::acc::ClauseDefaultValue &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace acc {
// built-in reduction operations supported by OpenACC
enum class ReductionOpAttr {
  redop_add,
  redop_mul,
  redop_max,
  redop_min,
  redop_and,
  redop_or,
  redop_xor,
  redop_leqv,
  redop_lneqv,
  redop_land,
  redop_lor,
};

::llvm::StringRef stringifyReductionOpAttr(ReductionOpAttr);
::llvm::Optional<ReductionOpAttr> symbolizeReductionOpAttr(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ReductionOpAttr enumValue) {
  return stringifyReductionOpAttr(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ReductionOpAttr> symbolizeEnum<ReductionOpAttr>(::llvm::StringRef str) {
  return symbolizeReductionOpAttr(str);
}
} // namespace acc
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ReductionOpAttr> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::acc::ReductionOpAttr>::type>;

  static inline ::mlir::acc::ReductionOpAttr getEmptyKey() {
    return static_cast<::mlir::acc::ReductionOpAttr>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ReductionOpAttr getTombstoneKey() {
    return static_cast<::mlir::acc::ReductionOpAttr>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ReductionOpAttr &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::acc::ReductionOpAttr>::type>(val));
  }

  static bool isEqual(const ::mlir::acc::ReductionOpAttr &lhs, const ::mlir::acc::ReductionOpAttr &rhs) {
    return lhs == rhs;
  }
};
}

