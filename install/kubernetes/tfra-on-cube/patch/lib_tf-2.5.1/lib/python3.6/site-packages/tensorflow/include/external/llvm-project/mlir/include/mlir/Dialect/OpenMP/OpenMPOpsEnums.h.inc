/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
// default clause
enum class ClauseDefault {
  defprivate,
  deffirstprivate,
  defshared,
  defnone,
};

::llvm::StringRef stringifyClauseDefault(ClauseDefault);
::llvm::Optional<ClauseDefault> symbolizeClauseDefault(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ClauseDefault enumValue) {
  return stringifyClauseDefault(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseDefault> symbolizeEnum<ClauseDefault>(::llvm::StringRef str) {
  return symbolizeClauseDefault(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseDefault> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::omp::ClauseDefault>::type>;

  static inline ::mlir::omp::ClauseDefault getEmptyKey() {
    return static_cast<::mlir::omp::ClauseDefault>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseDefault getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseDefault>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseDefault &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::omp::ClauseDefault>::type>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseDefault &lhs, const ::mlir::omp::ClauseDefault &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OrderKind Clause
enum class ClauseOrderKind {
};

::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(unsigned);
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind);
::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseOrderKind() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(ClauseOrderKind enumValue) {
  return stringifyClauseOrderKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseOrderKind> symbolizeEnum<ClauseOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::omp::ClauseOrderKind>::type>;

  static inline ::mlir::omp::ClauseOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::omp::ClauseOrderKind>::type>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseOrderKind &lhs, const ::mlir::omp::ClauseOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ProcBindKind Clause
enum class ClauseProcBindKind {
  master,
  close,
  spread,
};

::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind);
::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ClauseProcBindKind enumValue) {
  return stringifyClauseProcBindKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseProcBindKind> symbolizeEnum<ClauseProcBindKind>(::llvm::StringRef str) {
  return symbolizeClauseProcBindKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseProcBindKind> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::omp::ClauseProcBindKind>::type>;

  static inline ::mlir::omp::ClauseProcBindKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseProcBindKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseProcBindKind &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::omp::ClauseProcBindKind>::type>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseProcBindKind &lhs, const ::mlir::omp::ClauseProcBindKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ScheduleKind Clause
enum class ClauseScheduleKind {
  Static,
  Dynamic,
  Guided,
  Auto,
  Runtime,
};

::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind);
::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ClauseScheduleKind enumValue) {
  return stringifyClauseScheduleKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseScheduleKind> symbolizeEnum<ClauseScheduleKind>(::llvm::StringRef str) {
  return symbolizeClauseScheduleKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseScheduleKind> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::omp::ClauseScheduleKind>::type>;

  static inline ::mlir::omp::ClauseScheduleKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseScheduleKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseScheduleKind &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::omp::ClauseScheduleKind>::type>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseScheduleKind &lhs, const ::mlir::omp::ClauseScheduleKind &rhs) {
    return lhs == rhs;
  }
};
}

