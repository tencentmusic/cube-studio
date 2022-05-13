/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue val) {
  switch (val) {
    case ClauseDefaultValue::present: return "present";
    case ClauseDefaultValue::none: return "none";
  }
  return "";
}

::llvm::Optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseDefaultValue>>(str)
      .Case("present", ClauseDefaultValue::present)
      .Case("none", ClauseDefaultValue::none)
      .Default(::llvm::None);
}
} // namespace acc
} // namespace mlir

namespace mlir {
namespace acc {
::llvm::StringRef stringifyReductionOpAttr(ReductionOpAttr val) {
  switch (val) {
    case ReductionOpAttr::redop_add: return "redop_add";
    case ReductionOpAttr::redop_mul: return "redop_mul";
    case ReductionOpAttr::redop_max: return "redop_max";
    case ReductionOpAttr::redop_min: return "redop_min";
    case ReductionOpAttr::redop_and: return "redop_and";
    case ReductionOpAttr::redop_or: return "redop_or";
    case ReductionOpAttr::redop_xor: return "redop_xor";
    case ReductionOpAttr::redop_leqv: return "redop_leqv";
    case ReductionOpAttr::redop_lneqv: return "redop_lneqv";
    case ReductionOpAttr::redop_land: return "redop_land";
    case ReductionOpAttr::redop_lor: return "redop_lor";
  }
  return "";
}

::llvm::Optional<ReductionOpAttr> symbolizeReductionOpAttr(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ReductionOpAttr>>(str)
      .Case("redop_add", ReductionOpAttr::redop_add)
      .Case("redop_mul", ReductionOpAttr::redop_mul)
      .Case("redop_max", ReductionOpAttr::redop_max)
      .Case("redop_min", ReductionOpAttr::redop_min)
      .Case("redop_and", ReductionOpAttr::redop_and)
      .Case("redop_or", ReductionOpAttr::redop_or)
      .Case("redop_xor", ReductionOpAttr::redop_xor)
      .Case("redop_leqv", ReductionOpAttr::redop_leqv)
      .Case("redop_lneqv", ReductionOpAttr::redop_lneqv)
      .Case("redop_land", ReductionOpAttr::redop_land)
      .Case("redop_lor", ReductionOpAttr::redop_lor)
      .Default(::llvm::None);
}
} // namespace acc
} // namespace mlir

