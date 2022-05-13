/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseDefault(ClauseDefault val) {
  switch (val) {
    case ClauseDefault::defprivate: return "defprivate";
    case ClauseDefault::deffirstprivate: return "deffirstprivate";
    case ClauseDefault::defshared: return "defshared";
    case ClauseDefault::defnone: return "defnone";
  }
  return "";
}

::llvm::Optional<ClauseDefault> symbolizeClauseDefault(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseDefault>>(str)
      .Case("defprivate", ClauseDefault::defprivate)
      .Case("deffirstprivate", ClauseDefault::deffirstprivate)
      .Case("defshared", ClauseDefault::defshared)
      .Case("defnone", ClauseDefault::defnone)
      .Default(::llvm::None);
}
} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind val) {
  switch (val) {
  }
  return "";
}

::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseOrderKind>>(str)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(unsigned value) {
  switch (value) {
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind val) {
  switch (val) {
    case ClauseProcBindKind::master: return "master";
    case ClauseProcBindKind::close: return "close";
    case ClauseProcBindKind::spread: return "spread";
  }
  return "";
}

::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseProcBindKind>>(str)
      .Case("master", ClauseProcBindKind::master)
      .Case("close", ClauseProcBindKind::close)
      .Case("spread", ClauseProcBindKind::spread)
      .Default(::llvm::None);
}
} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind val) {
  switch (val) {
    case ClauseScheduleKind::Static: return "Static";
    case ClauseScheduleKind::Dynamic: return "Dynamic";
    case ClauseScheduleKind::Guided: return "Guided";
    case ClauseScheduleKind::Auto: return "Auto";
    case ClauseScheduleKind::Runtime: return "Runtime";
  }
  return "";
}

::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseScheduleKind>>(str)
      .Case("Static", ClauseScheduleKind::Static)
      .Case("Dynamic", ClauseScheduleKind::Dynamic)
      .Case("Guided", ClauseScheduleKind::Guided)
      .Case("Auto", ClauseScheduleKind::Auto)
      .Case("Runtime", ClauseScheduleKind::Runtime)
      .Default(::llvm::None);
}
} // namespace omp
} // namespace mlir

