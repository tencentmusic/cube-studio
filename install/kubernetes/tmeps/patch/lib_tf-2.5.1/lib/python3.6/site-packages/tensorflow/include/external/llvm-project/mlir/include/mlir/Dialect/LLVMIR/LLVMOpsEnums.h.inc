/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
// ATT (0) or Intel (1) asm dialect
enum class AsmDialect : uint64_t {
  AD_ATT = 0,
  AD_Intel = 1,
};

::llvm::Optional<AsmDialect> symbolizeAsmDialect(uint64_t);
::llvm::StringRef stringifyAsmDialect(AsmDialect);
::llvm::Optional<AsmDialect> symbolizeAsmDialect(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAsmDialect() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(AsmDialect enumValue) {
  return stringifyAsmDialect(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AsmDialect> symbolizeEnum<AsmDialect>(::llvm::StringRef str) {
  return symbolizeAsmDialect(str);
}

class AsmDialectAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AsmDialect;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AsmDialectAttr get(::mlir::MLIRContext *context, AsmDialect val);
  AsmDialect getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AsmDialect> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AsmDialect getEmptyKey() {
    return static_cast<::mlir::LLVM::AsmDialect>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AsmDialect getTombstoneKey() {
    return static_cast<::mlir::LLVM::AsmDialect>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AsmDialect &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AsmDialect &lhs, const ::mlir::LLVM::AsmDialect &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.atomicrmw binary operations
enum class AtomicBinOp : uint64_t {
  xchg = 0,
  add = 1,
  sub = 2,
  _and = 3,
  nand = 4,
  _or = 5,
  _xor = 6,
  max = 7,
  min = 8,
  umax = 9,
  umin = 10,
  fadd = 11,
  fsub = 12,
};

::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t);
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp);
::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicBinOp() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(AtomicBinOp enumValue) {
  return stringifyAtomicBinOp(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AtomicBinOp> symbolizeEnum<AtomicBinOp>(::llvm::StringRef str) {
  return symbolizeAtomicBinOp(str);
}

class AtomicBinOpAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AtomicBinOp;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AtomicBinOpAttr get(::mlir::MLIRContext *context, AtomicBinOp val);
  AtomicBinOp getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AtomicBinOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AtomicBinOp getEmptyKey() {
    return static_cast<::mlir::LLVM::AtomicBinOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AtomicBinOp getTombstoneKey() {
    return static_cast<::mlir::LLVM::AtomicBinOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AtomicBinOp &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AtomicBinOp &lhs, const ::mlir::LLVM::AtomicBinOp &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// Atomic ordering for LLVM's memory model
enum class AtomicOrdering : uint64_t {
  not_atomic = 0,
  unordered = 1,
  monotonic = 2,
  acquire = 4,
  release = 5,
  acq_rel = 6,
  seq_cst = 7,
};

::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t);
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering);
::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicOrdering() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(AtomicOrdering enumValue) {
  return stringifyAtomicOrdering(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AtomicOrdering> symbolizeEnum<AtomicOrdering>(::llvm::StringRef str) {
  return symbolizeAtomicOrdering(str);
}

class AtomicOrderingAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AtomicOrdering;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AtomicOrderingAttr get(::mlir::MLIRContext *context, AtomicOrdering val);
  AtomicOrdering getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AtomicOrdering> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AtomicOrdering getEmptyKey() {
    return static_cast<::mlir::LLVM::AtomicOrdering>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AtomicOrdering getTombstoneKey() {
    return static_cast<::mlir::LLVM::AtomicOrdering>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AtomicOrdering &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AtomicOrdering &lhs, const ::mlir::LLVM::AtomicOrdering &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.fcmp comparison predicate
enum class FCmpPredicate : uint64_t {
  _false = 0,
  oeq = 1,
  ogt = 2,
  oge = 3,
  olt = 4,
  ole = 5,
  one = 6,
  ord = 7,
  ueq = 8,
  ugt = 9,
  uge = 10,
  ult = 11,
  ule = 12,
  une = 13,
  uno = 14,
  _true = 15,
};

::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t);
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate);
::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFCmpPredicate() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(FCmpPredicate enumValue) {
  return stringifyFCmpPredicate(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FCmpPredicate> symbolizeEnum<FCmpPredicate>(::llvm::StringRef str) {
  return symbolizeFCmpPredicate(str);
}

class FCmpPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FCmpPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FCmpPredicateAttr get(::mlir::MLIRContext *context, FCmpPredicate val);
  FCmpPredicate getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::FCmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::FCmpPredicate getEmptyKey() {
    return static_cast<::mlir::LLVM::FCmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::FCmpPredicate getTombstoneKey() {
    return static_cast<::mlir::LLVM::FCmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::FCmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::FCmpPredicate &lhs, const ::mlir::LLVM::FCmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM fastmath flags
enum class FastmathFlags : uint32_t {
  nnan = 1,
  ninf = 2,
  nsz = 4,
  arcp = 8,
  contract = 16,
  afn = 32,
  reassoc = 64,
  fast = 128,
};

::llvm::Optional<FastmathFlags> symbolizeFastmathFlags(uint32_t);
std::string stringifyFastmathFlags(FastmathFlags);
::llvm::Optional<FastmathFlags> symbolizeFastmathFlags(::llvm::StringRef);
inline FastmathFlags operator|(FastmathFlags lhs, FastmathFlags rhs) {
  return static_cast<FastmathFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline FastmathFlags operator&(FastmathFlags lhs, FastmathFlags rhs) {
  return static_cast<FastmathFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(FastmathFlags bits, FastmathFlags bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

inline std::string stringifyEnum(FastmathFlags enumValue) {
  return stringifyFastmathFlags(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FastmathFlags> symbolizeEnum<FastmathFlags>(::llvm::StringRef str) {
  return symbolizeFastmathFlags(str);
}

class FastmathFlagsAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FastmathFlags;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FastmathFlagsAttr get(::mlir::MLIRContext *context, FastmathFlags val);
  FastmathFlags getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::FastmathFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::LLVM::FastmathFlags getEmptyKey() {
    return static_cast<::mlir::LLVM::FastmathFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::FastmathFlags getTombstoneKey() {
    return static_cast<::mlir::LLVM::FastmathFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::FastmathFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::FastmathFlags &lhs, const ::mlir::LLVM::FastmathFlags &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.icmp comparison predicate
enum class ICmpPredicate : uint64_t {
  eq = 0,
  ne = 1,
  slt = 2,
  sle = 3,
  sgt = 4,
  sge = 5,
  ult = 6,
  ule = 7,
  ugt = 8,
  uge = 9,
};

::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t);
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate);
::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForICmpPredicate() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(ICmpPredicate enumValue) {
  return stringifyICmpPredicate(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ICmpPredicate> symbolizeEnum<ICmpPredicate>(::llvm::StringRef str) {
  return symbolizeICmpPredicate(str);
}

class ICmpPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ICmpPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ICmpPredicateAttr get(::mlir::MLIRContext *context, ICmpPredicate val);
  ICmpPredicate getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::ICmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::ICmpPredicate getEmptyKey() {
    return static_cast<::mlir::LLVM::ICmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::ICmpPredicate getTombstoneKey() {
    return static_cast<::mlir::LLVM::ICmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::ICmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::ICmpPredicate &lhs, const ::mlir::LLVM::ICmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM linkage types
enum class Linkage : uint64_t {
  Private = 0,
  Internal = 1,
  AvailableExternally = 2,
  Linkonce = 3,
  Weak = 4,
  Common = 5,
  Appending = 6,
  ExternWeak = 7,
  LinkonceODR = 8,
  WeakODR = 9,
  External = 10,
};

::llvm::Optional<Linkage> symbolizeLinkage(uint64_t);
::llvm::StringRef stringifyLinkage(Linkage);
::llvm::Optional<Linkage> symbolizeLinkage(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLinkage() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(Linkage enumValue) {
  return stringifyLinkage(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Linkage> symbolizeEnum<Linkage>(::llvm::StringRef str) {
  return symbolizeLinkage(str);
}

class LinkageAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Linkage;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LinkageAttr get(::mlir::MLIRContext *context, Linkage val);
  Linkage getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::Linkage> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::Linkage getEmptyKey() {
    return static_cast<::mlir::LLVM::Linkage>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::Linkage getTombstoneKey() {
    return static_cast<::mlir::LLVM::Linkage>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::Linkage &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::Linkage &lhs, const ::mlir::LLVM::Linkage &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM loop option
enum class LoopOptionCase : uint32_t {
  disable_unroll = 1,
  disable_licm = 2,
  interleave_count = 3,
  disable_pipeline = 4,
  pipeline_initiation_interval = 5,
};

::llvm::Optional<LoopOptionCase> symbolizeLoopOptionCase(uint32_t);
::llvm::StringRef stringifyLoopOptionCase(LoopOptionCase);
::llvm::Optional<LoopOptionCase> symbolizeLoopOptionCase(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLoopOptionCase() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(LoopOptionCase enumValue) {
  return stringifyLoopOptionCase(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<LoopOptionCase> symbolizeEnum<LoopOptionCase>(::llvm::StringRef str) {
  return symbolizeLoopOptionCase(str);
}

class LoopOptionCaseAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = LoopOptionCase;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LoopOptionCaseAttr get(::mlir::MLIRContext *context, LoopOptionCase val);
  LoopOptionCase getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::LoopOptionCase> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::LLVM::LoopOptionCase getEmptyKey() {
    return static_cast<::mlir::LLVM::LoopOptionCase>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::LoopOptionCase getTombstoneKey() {
    return static_cast<::mlir::LLVM::LoopOptionCase>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::LoopOptionCase &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::LoopOptionCase &lhs, const ::mlir::LLVM::LoopOptionCase &rhs) {
    return lhs == rhs;
  }
};
}

