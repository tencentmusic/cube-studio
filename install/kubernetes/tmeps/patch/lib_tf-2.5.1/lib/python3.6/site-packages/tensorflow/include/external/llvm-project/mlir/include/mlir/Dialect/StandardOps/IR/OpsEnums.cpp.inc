/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
::llvm::StringRef stringifyAtomicRMWKind(AtomicRMWKind val) {
  switch (val) {
    case AtomicRMWKind::addf: return "addf";
    case AtomicRMWKind::addi: return "addi";
    case AtomicRMWKind::assign: return "assign";
    case AtomicRMWKind::maxf: return "maxf";
    case AtomicRMWKind::maxs: return "maxs";
    case AtomicRMWKind::maxu: return "maxu";
    case AtomicRMWKind::minf: return "minf";
    case AtomicRMWKind::mins: return "mins";
    case AtomicRMWKind::minu: return "minu";
    case AtomicRMWKind::mulf: return "mulf";
    case AtomicRMWKind::muli: return "muli";
  }
  return "";
}

::llvm::Optional<AtomicRMWKind> symbolizeAtomicRMWKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicRMWKind>>(str)
      .Case("addf", AtomicRMWKind::addf)
      .Case("addi", AtomicRMWKind::addi)
      .Case("assign", AtomicRMWKind::assign)
      .Case("maxf", AtomicRMWKind::maxf)
      .Case("maxs", AtomicRMWKind::maxs)
      .Case("maxu", AtomicRMWKind::maxu)
      .Case("minf", AtomicRMWKind::minf)
      .Case("mins", AtomicRMWKind::mins)
      .Case("minu", AtomicRMWKind::minu)
      .Case("mulf", AtomicRMWKind::mulf)
      .Case("muli", AtomicRMWKind::muli)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicRMWKind> symbolizeAtomicRMWKind(uint64_t value) {
  switch (value) {
  case 0: return AtomicRMWKind::addf;
  case 1: return AtomicRMWKind::addi;
  case 2: return AtomicRMWKind::assign;
  case 3: return AtomicRMWKind::maxf;
  case 4: return AtomicRMWKind::maxs;
  case 5: return AtomicRMWKind::maxu;
  case 6: return AtomicRMWKind::minf;
  case 7: return AtomicRMWKind::mins;
  case 8: return AtomicRMWKind::minu;
  case 9: return AtomicRMWKind::mulf;
  case 10: return AtomicRMWKind::muli;
  default: return ::llvm::None;
  }
}

bool AtomicRMWKindAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)));
}
AtomicRMWKindAttr AtomicRMWKindAttr::get(::mlir::MLIRContext *context, AtomicRMWKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AtomicRMWKindAttr>();
}
AtomicRMWKind AtomicRMWKindAttr::getValue() const {
  return static_cast<AtomicRMWKind>(::mlir::IntegerAttr::getInt());
}
} // namespace mlir

namespace mlir {
::llvm::StringRef stringifyCmpFPredicate(CmpFPredicate val) {
  switch (val) {
    case CmpFPredicate::AlwaysFalse: return "false";
    case CmpFPredicate::OEQ: return "oeq";
    case CmpFPredicate::OGT: return "ogt";
    case CmpFPredicate::OGE: return "oge";
    case CmpFPredicate::OLT: return "olt";
    case CmpFPredicate::OLE: return "ole";
    case CmpFPredicate::ONE: return "one";
    case CmpFPredicate::ORD: return "ord";
    case CmpFPredicate::UEQ: return "ueq";
    case CmpFPredicate::UGT: return "ugt";
    case CmpFPredicate::UGE: return "uge";
    case CmpFPredicate::ULT: return "ult";
    case CmpFPredicate::ULE: return "ule";
    case CmpFPredicate::UNE: return "une";
    case CmpFPredicate::UNO: return "uno";
    case CmpFPredicate::AlwaysTrue: return "true";
  }
  return "";
}

::llvm::Optional<CmpFPredicate> symbolizeCmpFPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<CmpFPredicate>>(str)
      .Case("false", CmpFPredicate::AlwaysFalse)
      .Case("oeq", CmpFPredicate::OEQ)
      .Case("ogt", CmpFPredicate::OGT)
      .Case("oge", CmpFPredicate::OGE)
      .Case("olt", CmpFPredicate::OLT)
      .Case("ole", CmpFPredicate::OLE)
      .Case("one", CmpFPredicate::ONE)
      .Case("ord", CmpFPredicate::ORD)
      .Case("ueq", CmpFPredicate::UEQ)
      .Case("ugt", CmpFPredicate::UGT)
      .Case("uge", CmpFPredicate::UGE)
      .Case("ult", CmpFPredicate::ULT)
      .Case("ule", CmpFPredicate::ULE)
      .Case("une", CmpFPredicate::UNE)
      .Case("uno", CmpFPredicate::UNO)
      .Case("true", CmpFPredicate::AlwaysTrue)
      .Default(::llvm::None);
}
::llvm::Optional<CmpFPredicate> symbolizeCmpFPredicate(uint64_t value) {
  switch (value) {
  case 0: return CmpFPredicate::AlwaysFalse;
  case 1: return CmpFPredicate::OEQ;
  case 2: return CmpFPredicate::OGT;
  case 3: return CmpFPredicate::OGE;
  case 4: return CmpFPredicate::OLT;
  case 5: return CmpFPredicate::OLE;
  case 6: return CmpFPredicate::ONE;
  case 7: return CmpFPredicate::ORD;
  case 8: return CmpFPredicate::UEQ;
  case 9: return CmpFPredicate::UGT;
  case 10: return CmpFPredicate::UGE;
  case 11: return CmpFPredicate::ULT;
  case 12: return CmpFPredicate::ULE;
  case 13: return CmpFPredicate::UNE;
  case 14: return CmpFPredicate::UNO;
  case 15: return CmpFPredicate::AlwaysTrue;
  default: return ::llvm::None;
  }
}

bool CmpFPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)));
}
CmpFPredicateAttr CmpFPredicateAttr::get(::mlir::MLIRContext *context, CmpFPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<CmpFPredicateAttr>();
}
CmpFPredicate CmpFPredicateAttr::getValue() const {
  return static_cast<CmpFPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace mlir

namespace mlir {
::llvm::StringRef stringifyCmpIPredicate(CmpIPredicate val) {
  switch (val) {
    case CmpIPredicate::eq: return "eq";
    case CmpIPredicate::ne: return "ne";
    case CmpIPredicate::slt: return "slt";
    case CmpIPredicate::sle: return "sle";
    case CmpIPredicate::sgt: return "sgt";
    case CmpIPredicate::sge: return "sge";
    case CmpIPredicate::ult: return "ult";
    case CmpIPredicate::ule: return "ule";
    case CmpIPredicate::ugt: return "ugt";
    case CmpIPredicate::uge: return "uge";
  }
  return "";
}

::llvm::Optional<CmpIPredicate> symbolizeCmpIPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<CmpIPredicate>>(str)
      .Case("eq", CmpIPredicate::eq)
      .Case("ne", CmpIPredicate::ne)
      .Case("slt", CmpIPredicate::slt)
      .Case("sle", CmpIPredicate::sle)
      .Case("sgt", CmpIPredicate::sgt)
      .Case("sge", CmpIPredicate::sge)
      .Case("ult", CmpIPredicate::ult)
      .Case("ule", CmpIPredicate::ule)
      .Case("ugt", CmpIPredicate::ugt)
      .Case("uge", CmpIPredicate::uge)
      .Default(::llvm::None);
}
::llvm::Optional<CmpIPredicate> symbolizeCmpIPredicate(uint64_t value) {
  switch (value) {
  case 0: return CmpIPredicate::eq;
  case 1: return CmpIPredicate::ne;
  case 2: return CmpIPredicate::slt;
  case 3: return CmpIPredicate::sle;
  case 4: return CmpIPredicate::sgt;
  case 5: return CmpIPredicate::sge;
  case 6: return CmpIPredicate::ult;
  case 7: return CmpIPredicate::ule;
  case 8: return CmpIPredicate::ugt;
  case 9: return CmpIPredicate::uge;
  default: return ::llvm::None;
  }
}

bool CmpIPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)));
}
CmpIPredicateAttr CmpIPredicateAttr::get(::mlir::MLIRContext *context, CmpIPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<CmpIPredicateAttr>();
}
CmpIPredicate CmpIPredicateAttr::getValue() const {
  return static_cast<CmpIPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace mlir

