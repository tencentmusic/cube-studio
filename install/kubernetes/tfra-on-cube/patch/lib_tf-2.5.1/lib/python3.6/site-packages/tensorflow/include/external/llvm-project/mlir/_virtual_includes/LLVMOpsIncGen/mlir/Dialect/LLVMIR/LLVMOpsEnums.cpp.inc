/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAsmDialect(AsmDialect val) {
  switch (val) {
    case AsmDialect::AD_ATT: return "att";
    case AsmDialect::AD_Intel: return "intel";
  }
  return "";
}

::llvm::Optional<AsmDialect> symbolizeAsmDialect(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AsmDialect>>(str)
      .Case("att", AsmDialect::AD_ATT)
      .Case("intel", AsmDialect::AD_Intel)
      .Default(::llvm::None);
}
::llvm::Optional<AsmDialect> symbolizeAsmDialect(uint64_t value) {
  switch (value) {
  case 0: return AsmDialect::AD_ATT;
  case 1: return AsmDialect::AD_Intel;
  default: return ::llvm::None;
  }
}

bool AsmDialectAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)));
}
AsmDialectAttr AsmDialectAttr::get(::mlir::MLIRContext *context, AsmDialect val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AsmDialectAttr>();
}
AsmDialect AsmDialectAttr::getValue() const {
  return static_cast<AsmDialect>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp val) {
  switch (val) {
    case AtomicBinOp::xchg: return "xchg";
    case AtomicBinOp::add: return "add";
    case AtomicBinOp::sub: return "sub";
    case AtomicBinOp::_and: return "_and";
    case AtomicBinOp::nand: return "nand";
    case AtomicBinOp::_or: return "_or";
    case AtomicBinOp::_xor: return "_xor";
    case AtomicBinOp::max: return "max";
    case AtomicBinOp::min: return "min";
    case AtomicBinOp::umax: return "umax";
    case AtomicBinOp::umin: return "umin";
    case AtomicBinOp::fadd: return "fadd";
    case AtomicBinOp::fsub: return "fsub";
  }
  return "";
}

::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicBinOp>>(str)
      .Case("xchg", AtomicBinOp::xchg)
      .Case("add", AtomicBinOp::add)
      .Case("sub", AtomicBinOp::sub)
      .Case("_and", AtomicBinOp::_and)
      .Case("nand", AtomicBinOp::nand)
      .Case("_or", AtomicBinOp::_or)
      .Case("_xor", AtomicBinOp::_xor)
      .Case("max", AtomicBinOp::max)
      .Case("min", AtomicBinOp::min)
      .Case("umax", AtomicBinOp::umax)
      .Case("umin", AtomicBinOp::umin)
      .Case("fadd", AtomicBinOp::fadd)
      .Case("fsub", AtomicBinOp::fsub)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t value) {
  switch (value) {
  case 0: return AtomicBinOp::xchg;
  case 1: return AtomicBinOp::add;
  case 2: return AtomicBinOp::sub;
  case 3: return AtomicBinOp::_and;
  case 4: return AtomicBinOp::nand;
  case 5: return AtomicBinOp::_or;
  case 6: return AtomicBinOp::_xor;
  case 7: return AtomicBinOp::max;
  case 8: return AtomicBinOp::min;
  case 9: return AtomicBinOp::umax;
  case 10: return AtomicBinOp::umin;
  case 11: return AtomicBinOp::fadd;
  case 12: return AtomicBinOp::fsub;
  default: return ::llvm::None;
  }
}

bool AtomicBinOpAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)));
}
AtomicBinOpAttr AtomicBinOpAttr::get(::mlir::MLIRContext *context, AtomicBinOp val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AtomicBinOpAttr>();
}
AtomicBinOp AtomicBinOpAttr::getValue() const {
  return static_cast<AtomicBinOp>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering val) {
  switch (val) {
    case AtomicOrdering::not_atomic: return "not_atomic";
    case AtomicOrdering::unordered: return "unordered";
    case AtomicOrdering::monotonic: return "monotonic";
    case AtomicOrdering::acquire: return "acquire";
    case AtomicOrdering::release: return "release";
    case AtomicOrdering::acq_rel: return "acq_rel";
    case AtomicOrdering::seq_cst: return "seq_cst";
  }
  return "";
}

::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicOrdering>>(str)
      .Case("not_atomic", AtomicOrdering::not_atomic)
      .Case("unordered", AtomicOrdering::unordered)
      .Case("monotonic", AtomicOrdering::monotonic)
      .Case("acquire", AtomicOrdering::acquire)
      .Case("release", AtomicOrdering::release)
      .Case("acq_rel", AtomicOrdering::acq_rel)
      .Case("seq_cst", AtomicOrdering::seq_cst)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t value) {
  switch (value) {
  case 0: return AtomicOrdering::not_atomic;
  case 1: return AtomicOrdering::unordered;
  case 2: return AtomicOrdering::monotonic;
  case 4: return AtomicOrdering::acquire;
  case 5: return AtomicOrdering::release;
  case 6: return AtomicOrdering::acq_rel;
  case 7: return AtomicOrdering::seq_cst;
  default: return ::llvm::None;
  }
}

bool AtomicOrderingAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)));
}
AtomicOrderingAttr AtomicOrderingAttr::get(::mlir::MLIRContext *context, AtomicOrdering val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AtomicOrderingAttr>();
}
AtomicOrdering AtomicOrderingAttr::getValue() const {
  return static_cast<AtomicOrdering>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate val) {
  switch (val) {
    case FCmpPredicate::_false: return "_false";
    case FCmpPredicate::oeq: return "oeq";
    case FCmpPredicate::ogt: return "ogt";
    case FCmpPredicate::oge: return "oge";
    case FCmpPredicate::olt: return "olt";
    case FCmpPredicate::ole: return "ole";
    case FCmpPredicate::one: return "one";
    case FCmpPredicate::ord: return "ord";
    case FCmpPredicate::ueq: return "ueq";
    case FCmpPredicate::ugt: return "ugt";
    case FCmpPredicate::uge: return "uge";
    case FCmpPredicate::ult: return "ult";
    case FCmpPredicate::ule: return "ule";
    case FCmpPredicate::une: return "une";
    case FCmpPredicate::uno: return "uno";
    case FCmpPredicate::_true: return "_true";
  }
  return "";
}

::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<FCmpPredicate>>(str)
      .Case("_false", FCmpPredicate::_false)
      .Case("oeq", FCmpPredicate::oeq)
      .Case("ogt", FCmpPredicate::ogt)
      .Case("oge", FCmpPredicate::oge)
      .Case("olt", FCmpPredicate::olt)
      .Case("ole", FCmpPredicate::ole)
      .Case("one", FCmpPredicate::one)
      .Case("ord", FCmpPredicate::ord)
      .Case("ueq", FCmpPredicate::ueq)
      .Case("ugt", FCmpPredicate::ugt)
      .Case("uge", FCmpPredicate::uge)
      .Case("ult", FCmpPredicate::ult)
      .Case("ule", FCmpPredicate::ule)
      .Case("une", FCmpPredicate::une)
      .Case("uno", FCmpPredicate::uno)
      .Case("_true", FCmpPredicate::_true)
      .Default(::llvm::None);
}
::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return FCmpPredicate::_false;
  case 1: return FCmpPredicate::oeq;
  case 2: return FCmpPredicate::ogt;
  case 3: return FCmpPredicate::oge;
  case 4: return FCmpPredicate::olt;
  case 5: return FCmpPredicate::ole;
  case 6: return FCmpPredicate::one;
  case 7: return FCmpPredicate::ord;
  case 8: return FCmpPredicate::ueq;
  case 9: return FCmpPredicate::ugt;
  case 10: return FCmpPredicate::uge;
  case 11: return FCmpPredicate::ult;
  case 12: return FCmpPredicate::ule;
  case 13: return FCmpPredicate::une;
  case 14: return FCmpPredicate::uno;
  case 15: return FCmpPredicate::_true;
  default: return ::llvm::None;
  }
}

bool FCmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)));
}
FCmpPredicateAttr FCmpPredicateAttr::get(::mlir::MLIRContext *context, FCmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<FCmpPredicateAttr>();
}
FCmpPredicate FCmpPredicateAttr::getValue() const {
  return static_cast<FCmpPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
std::string stringifyFastmathFlags(FastmathFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("nnan"); val &= ~1u; }
  if (2u & val) { strs.push_back("ninf"); val &= ~2u; }
  if (4u & val) { strs.push_back("nsz"); val &= ~4u; }
  if (8u & val) { strs.push_back("arcp"); val &= ~8u; }
  if (16u & val) { strs.push_back("contract"); val &= ~16u; }
  if (32u & val) { strs.push_back("afn"); val &= ~32u; }
  if (64u & val) { strs.push_back("reassoc"); val &= ~64u; }
  if (128u & val) { strs.push_back("fast"); val &= ~128u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<FastmathFlags> symbolizeFastmathFlags(::llvm::StringRef str) {
  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("nnan", 1)
      .Case("ninf", 2)
      .Case("nsz", 4)
      .Case("arcp", 8)
      .Case("contract", 16)
      .Case("afn", 32)
      .Case("reassoc", 64)
      .Case("fast", 128)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<FastmathFlags>(val);
}

::llvm::Optional<FastmathFlags> symbolizeFastmathFlags(uint32_t value) {
  if (value & ~(1u | 2u | 4u | 8u | 16u | 32u | 64u | 128u)) return llvm::None;
  return static_cast<FastmathFlags>(value);
}
bool FastmathFlagsAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(1u|2u|4u|8u|16u|32u|64u|128u)))));
}
FastmathFlagsAttr FastmathFlagsAttr::get(::mlir::MLIRContext *context, FastmathFlags val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<FastmathFlagsAttr>();
}
FastmathFlags FastmathFlagsAttr::getValue() const {
  return static_cast<FastmathFlags>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate val) {
  switch (val) {
    case ICmpPredicate::eq: return "eq";
    case ICmpPredicate::ne: return "ne";
    case ICmpPredicate::slt: return "slt";
    case ICmpPredicate::sle: return "sle";
    case ICmpPredicate::sgt: return "sgt";
    case ICmpPredicate::sge: return "sge";
    case ICmpPredicate::ult: return "ult";
    case ICmpPredicate::ule: return "ule";
    case ICmpPredicate::ugt: return "ugt";
    case ICmpPredicate::uge: return "uge";
  }
  return "";
}

::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ICmpPredicate>>(str)
      .Case("eq", ICmpPredicate::eq)
      .Case("ne", ICmpPredicate::ne)
      .Case("slt", ICmpPredicate::slt)
      .Case("sle", ICmpPredicate::sle)
      .Case("sgt", ICmpPredicate::sgt)
      .Case("sge", ICmpPredicate::sge)
      .Case("ult", ICmpPredicate::ult)
      .Case("ule", ICmpPredicate::ule)
      .Case("ugt", ICmpPredicate::ugt)
      .Case("uge", ICmpPredicate::uge)
      .Default(::llvm::None);
}
::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return ICmpPredicate::eq;
  case 1: return ICmpPredicate::ne;
  case 2: return ICmpPredicate::slt;
  case 3: return ICmpPredicate::sle;
  case 4: return ICmpPredicate::sgt;
  case 5: return ICmpPredicate::sge;
  case 6: return ICmpPredicate::ult;
  case 7: return ICmpPredicate::ule;
  case 8: return ICmpPredicate::ugt;
  case 9: return ICmpPredicate::uge;
  default: return ::llvm::None;
  }
}

bool ICmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)));
}
ICmpPredicateAttr ICmpPredicateAttr::get(::mlir::MLIRContext *context, ICmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<ICmpPredicateAttr>();
}
ICmpPredicate ICmpPredicateAttr::getValue() const {
  return static_cast<ICmpPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyLinkage(Linkage val) {
  switch (val) {
    case Linkage::Private: return "private";
    case Linkage::Internal: return "internal";
    case Linkage::AvailableExternally: return "available_externally";
    case Linkage::Linkonce: return "linkonce";
    case Linkage::Weak: return "weak";
    case Linkage::Common: return "common";
    case Linkage::Appending: return "appending";
    case Linkage::ExternWeak: return "extern_weak";
    case Linkage::LinkonceODR: return "linkonce_odr";
    case Linkage::WeakODR: return "weak_odr";
    case Linkage::External: return "external";
  }
  return "";
}

::llvm::Optional<Linkage> symbolizeLinkage(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Linkage>>(str)
      .Case("private", Linkage::Private)
      .Case("internal", Linkage::Internal)
      .Case("available_externally", Linkage::AvailableExternally)
      .Case("linkonce", Linkage::Linkonce)
      .Case("weak", Linkage::Weak)
      .Case("common", Linkage::Common)
      .Case("appending", Linkage::Appending)
      .Case("extern_weak", Linkage::ExternWeak)
      .Case("linkonce_odr", Linkage::LinkonceODR)
      .Case("weak_odr", Linkage::WeakODR)
      .Case("external", Linkage::External)
      .Default(::llvm::None);
}
::llvm::Optional<Linkage> symbolizeLinkage(uint64_t value) {
  switch (value) {
  case 0: return Linkage::Private;
  case 1: return Linkage::Internal;
  case 2: return Linkage::AvailableExternally;
  case 3: return Linkage::Linkonce;
  case 4: return Linkage::Weak;
  case 5: return Linkage::Common;
  case 6: return Linkage::Appending;
  case 7: return Linkage::ExternWeak;
  case 8: return Linkage::LinkonceODR;
  case 9: return Linkage::WeakODR;
  case 10: return Linkage::External;
  default: return ::llvm::None;
  }
}

bool LinkageAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)));
}
LinkageAttr LinkageAttr::get(::mlir::MLIRContext *context, Linkage val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<LinkageAttr>();
}
Linkage LinkageAttr::getValue() const {
  return static_cast<Linkage>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyLoopOptionCase(LoopOptionCase val) {
  switch (val) {
    case LoopOptionCase::disable_unroll: return "disable_unroll";
    case LoopOptionCase::disable_licm: return "disable_licm";
    case LoopOptionCase::interleave_count: return "interleave_count";
    case LoopOptionCase::disable_pipeline: return "disable_pipeline";
    case LoopOptionCase::pipeline_initiation_interval: return "pipeline_initiation_interval";
  }
  return "";
}

::llvm::Optional<LoopOptionCase> symbolizeLoopOptionCase(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<LoopOptionCase>>(str)
      .Case("disable_unroll", LoopOptionCase::disable_unroll)
      .Case("disable_licm", LoopOptionCase::disable_licm)
      .Case("interleave_count", LoopOptionCase::interleave_count)
      .Case("disable_pipeline", LoopOptionCase::disable_pipeline)
      .Case("pipeline_initiation_interval", LoopOptionCase::pipeline_initiation_interval)
      .Default(::llvm::None);
}
::llvm::Optional<LoopOptionCase> symbolizeLoopOptionCase(uint32_t value) {
  switch (value) {
  case 1: return LoopOptionCase::disable_unroll;
  case 2: return LoopOptionCase::disable_licm;
  case 3: return LoopOptionCase::interleave_count;
  case 4: return LoopOptionCase::disable_pipeline;
  case 5: return LoopOptionCase::pipeline_initiation_interval;
  default: return ::llvm::None;
  }
}

bool LoopOptionCaseAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)));
}
LoopOptionCaseAttr LoopOptionCaseAttr::get(::mlir::MLIRContext *context, LoopOptionCase val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<LoopOptionCaseAttr>();
}
LoopOptionCase LoopOptionCaseAttr::getValue() const {
  return static_cast<LoopOptionCase>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

