/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
// error code
enum class ErrorCode : uint32_t {
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  UNAUTHENTICATED = 16,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
};

::llvm::Optional<ErrorCode> symbolizeErrorCode(uint32_t);
::llvm::StringRef stringifyErrorCode(ErrorCode);
::llvm::Optional<ErrorCode> symbolizeErrorCode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForErrorCode() {
  return 16;
}


inline ::llvm::StringRef stringifyEnum(ErrorCode enumValue) {
  return stringifyErrorCode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ErrorCode> symbolizeEnum<ErrorCode>(::llvm::StringRef str) {
  return symbolizeErrorCode(str);
}

class ErrorCodeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ErrorCode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ErrorCodeAttr get(::mlir::MLIRContext *context, ErrorCode val);
  ErrorCode getValue() const;
};
} // namespace tf_framework
} // namespace kernel_gen
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::kernel_gen::tf_framework::ErrorCode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::kernel_gen::tf_framework::ErrorCode getEmptyKey() {
    return static_cast<::mlir::kernel_gen::tf_framework::ErrorCode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::kernel_gen::tf_framework::ErrorCode getTombstoneKey() {
    return static_cast<::mlir::kernel_gen::tf_framework::ErrorCode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::kernel_gen::tf_framework::ErrorCode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::kernel_gen::tf_framework::ErrorCode &lhs, const ::mlir::kernel_gen::tf_framework::ErrorCode &rhs) {
    return lhs == rhs;
  }
};
}

