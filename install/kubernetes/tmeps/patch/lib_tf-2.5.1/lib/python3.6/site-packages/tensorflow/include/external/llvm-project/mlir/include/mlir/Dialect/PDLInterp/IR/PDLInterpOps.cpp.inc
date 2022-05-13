/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl_interp::ApplyConstraintOp,
::mlir::pdl_interp::ApplyRewriteOp,
::mlir::pdl_interp::AreEqualOp,
::mlir::pdl_interp::BranchOp,
::mlir::pdl_interp::CheckAttributeOp,
::mlir::pdl_interp::CheckOperandCountOp,
::mlir::pdl_interp::CheckOperationNameOp,
::mlir::pdl_interp::CheckResultCountOp,
::mlir::pdl_interp::CheckTypeOp,
::mlir::pdl_interp::CheckTypesOp,
::mlir::pdl_interp::CreateAttributeOp,
::mlir::pdl_interp::CreateOperationOp,
::mlir::pdl_interp::CreateTypeOp,
::mlir::pdl_interp::CreateTypesOp,
::mlir::pdl_interp::EraseOp,
::mlir::pdl_interp::FinalizeOp,
::mlir::pdl_interp::GetAttributeOp,
::mlir::pdl_interp::GetAttributeTypeOp,
::mlir::pdl_interp::GetDefiningOpOp,
::mlir::pdl_interp::GetOperandOp,
::mlir::pdl_interp::GetOperandsOp,
::mlir::pdl_interp::GetResultOp,
::mlir::pdl_interp::GetResultsOp,
::mlir::pdl_interp::GetValueTypeOp,
::mlir::pdl_interp::InferredTypesOp,
::mlir::pdl_interp::IsNotNullOp,
::mlir::pdl_interp::RecordMatchOp,
::mlir::pdl_interp::ReplaceOp,
::mlir::pdl_interp::SwitchAttributeOp,
::mlir::pdl_interp::SwitchOperandCountOp,
::mlir::pdl_interp::SwitchOperationNameOp,
::mlir::pdl_interp::SwitchResultCountOp,
::mlir::pdl_interp::SwitchTypeOp,
::mlir::pdl_interp::SwitchTypesOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl_interp {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::PDLType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::PDLType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::AttributeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::TypeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::ValueType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::AttributeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::ValueType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps10(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::ValueType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps11(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps12(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyConstraintOp definitions
//===----------------------------------------------------------------------===//

ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(ApplyConstraintOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ApplyConstraintOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyConstraintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ApplyConstraintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyConstraintOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyConstraintOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyConstraintOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr ApplyConstraintOpAdaptor::constParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ApplyConstraintOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl_interp.apply_constraint' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constParams = odsAttrs.get("constParams");
  if (tblgen_constParams) {
    if (!((tblgen_constParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'constParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ApplyConstraintOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ApplyConstraintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyConstraintOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyConstraintOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyConstraintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *ApplyConstraintOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *ApplyConstraintOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::StringAttr ApplyConstraintOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyConstraintOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr ApplyConstraintOp::constParamsAttr() {
  return (*this)->getAttr("constParams").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyConstraintOp::constParams() {
  auto attr = constParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ApplyConstraintOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void ApplyConstraintOp::constParamsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("constParams", attr);
}

::mlir::Attribute ApplyConstraintOp::removeConstParamsAttr() {
  return (*this)->removeAttr("constParams");
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyConstraintOp::verify() {
  if (failed(ApplyConstraintOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult ApplyConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr constParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(constParamsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constParams", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (constParamsAttr) {
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void ApplyConstraintOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.apply_constraint";
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  if ((*this)->getAttr("constParams")) {
  p << ' ';
  p.printAttributeWithoutType(constParamsAttr());
  }
  p << "(";
  p << args();
  p << ' ' << ":";
  p << ' ';
  p << args().getTypes();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name", "constParams"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyRewriteOp definitions
//===----------------------------------------------------------------------===//

ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(ApplyRewriteOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ApplyRewriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyRewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ApplyRewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyRewriteOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyRewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyRewriteOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr ApplyRewriteOpAdaptor::constParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ApplyRewriteOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl_interp.apply_rewrite' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl_interp.apply_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constParams = odsAttrs.get("constParams");
  if (tblgen_constParams) {
    if (!((tblgen_constParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl_interp.apply_rewrite' op ""attribute 'constParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ApplyRewriteOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ApplyRewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyRewriteOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyRewriteOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ApplyRewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ApplyRewriteOp::results() {
  return getODSResults(0);
}

::mlir::StringAttr ApplyRewriteOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyRewriteOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr ApplyRewriteOp::constParamsAttr() {
  return (*this)->getAttr("constParams").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyRewriteOp::constParams() {
  auto attr = constParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ApplyRewriteOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void ApplyRewriteOp::constParamsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("constParams", attr);
}

::mlir::Attribute ApplyRewriteOp::removeConstParamsAttr() {
  return (*this)->removeAttr("constParams");
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyRewriteOp::verify() {
  if (failed(ApplyRewriteOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ApplyRewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr constParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;
  ::mlir::SmallVector<::mlir::Type, 1> resultsTypes;

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(constParamsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constParams", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (constParamsAttr) {
  }
  if (succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyRewriteOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.apply_rewrite";
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  if ((*this)->getAttr("constParams")) {
  p << ' ';
  p.printAttributeWithoutType(constParamsAttr());
  }
  if (!args().empty()) {
  p << "(";
  p << args();
  p << ' ' << ":";
  p << ' ';
  p << args().getTypes();
  p << ")";
  }
  if (!results().empty()) {
  p << ' ' << ":";
  p << ' ';
  p << results().getTypes();
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name", "constParams"});
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::AreEqualOp definitions
//===----------------------------------------------------------------------===//

AreEqualOpAdaptor::AreEqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AreEqualOpAdaptor::AreEqualOpAdaptor(AreEqualOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AreEqualOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AreEqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AreEqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AreEqualOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AreEqualOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AreEqualOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AreEqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AreEqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AreEqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AreEqualOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AreEqualOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AreEqualOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AreEqualOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AreEqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AreEqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *AreEqualOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *AreEqualOp::falseDest() {
  return (*this)->getSuccessor(1);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AreEqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AreEqualOp::verify() {
  if (failed(AreEqualOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult AreEqualOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.resolveOperands(allOperands, ::llvm::concat<const Type>(::llvm::ArrayRef<::mlir::Type>(lhsTypes), ::llvm::ArrayRef<::mlir::Type>(lhsTypes)), allOperandLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void AreEqualOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.are_equal";
  p << ' ';
  p << getOperation()->getOperands();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void AreEqualOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::BranchOp definitions
//===----------------------------------------------------------------------===//

BranchOpAdaptor::BranchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BranchOpAdaptor::BranchOpAdaptor(BranchOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BranchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BranchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BranchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BranchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BranchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BranchOp::dest() {
  return (*this)->getSuccessor(0);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BranchOp::verify() {
  if (failed(BranchOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.branch";
  p << ' ';
  p << dest();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void BranchOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckAttributeOp definitions
//===----------------------------------------------------------------------===//

CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(CheckAttributeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CheckAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckAttributeOpAdaptor::attribute() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CheckAttributeOpAdaptor::constantValue() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("constantValue").cast<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult CheckAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_constantValue = odsAttrs.get("constantValue");
  if (!tblgen_constantValue) return emitError(loc, "'pdl_interp.check_attribute' op ""requires attribute 'constantValue'");
    if (!((true))) return emitError(loc, "'pdl_interp.check_attribute' op ""attribute 'constantValue' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CheckAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckAttributeOp::attribute() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckAttributeOp::attributeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CheckAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckAttributeOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckAttributeOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::Attribute CheckAttributeOp::constantValueAttr() {
  return (*this)->getAttr("constantValue").template cast<::mlir::Attribute>();
}

::mlir::Attribute CheckAttributeOp::constantValue() {
  auto attr = constantValueAttr();
  return attr;
}

void CheckAttributeOp::constantValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr("constantValue", attr);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.addAttribute("constantValue", constantValue);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.addAttribute("constantValue", constantValue);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckAttributeOp::verify() {
  if (failed(CheckAttributeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CheckAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType attributeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> attributeOperands(attributeRawOperands);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::Attribute constantValueAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseAttribute(constantValueAttr, "constantValue", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void CheckAttributeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.check_attribute";
  p << ' ';
  p << attribute();
  p << ' ' << "is";
  p << ' ';
  p.printAttribute(constantValueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"constantValue"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void CheckAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperandCountOp definitions
//===----------------------------------------------------------------------===//

CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(CheckOperandCountOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CheckOperandCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckOperandCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckOperandCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperandCountOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckOperandCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckOperandCountOpAdaptor::count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::UnitAttr CheckOperandCountOpAdaptor::compareAtLeast() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("compareAtLeast").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult CheckOperandCountOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_count = odsAttrs.get("count");
  if (!tblgen_count) return emitError(loc, "'pdl_interp.check_operand_count' op ""requires attribute 'count'");
    if (!((((tblgen_count.isa<::mlir::IntegerAttr>())) && ((tblgen_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_count.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  {
  auto tblgen_compareAtLeast = odsAttrs.get("compareAtLeast");
  if (tblgen_compareAtLeast) {
    if (!((tblgen_compareAtLeast.isa<::mlir::UnitAttr>()))) return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CheckOperandCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckOperandCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperandCountOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckOperandCountOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CheckOperandCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckOperandCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckOperandCountOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckOperandCountOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::IntegerAttr CheckOperandCountOp::countAttr() {
  return (*this)->getAttr("count").template cast<::mlir::IntegerAttr>();
}

uint32_t CheckOperandCountOp::count() {
  auto attr = countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOp::compareAtLeastAttr() {
  return (*this)->getAttr("compareAtLeast").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckOperandCountOp::compareAtLeast() {
  auto attr = compareAtLeastAttr();
  return attr != nullptr;
}

void CheckOperandCountOp::countAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("count", attr);
}

void CheckOperandCountOp::compareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("compareAtLeast", attr);
}

::mlir::Attribute CheckOperandCountOp::removeCompareAtLeastAttr() {
  return (*this)->removeAttr("compareAtLeast");
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", count);
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", count);
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckOperandCountOp::verify() {
  if (failed(CheckOperandCountOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CheckOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.addAttribute("compareAtLeast", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseAttribute(countAttr, parser.getBuilder().getIntegerType(32), "count", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void CheckOperandCountOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.check_operand_count";
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << "is";
  if ((*this)->getAttr("compareAtLeast")) {
  p << ' ' << "at_least";
  }
  p << ' ';
  p.printAttributeWithoutType(countAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"compareAtLeast", "count"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void CheckOperandCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperationNameOp definitions
//===----------------------------------------------------------------------===//

CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(CheckOperationNameOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CheckOperationNameOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckOperationNameOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckOperationNameOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperationNameOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckOperationNameOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CheckOperationNameOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult CheckOperationNameOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl_interp.check_operation_name' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl_interp.check_operation_name' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CheckOperationNameOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckOperationNameOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperationNameOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckOperationNameOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CheckOperationNameOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckOperationNameOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckOperationNameOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckOperationNameOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::StringAttr CheckOperationNameOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef CheckOperationNameOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

void CheckOperationNameOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckOperationNameOp::verify() {
  if (failed(CheckOperationNameOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CheckOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void CheckOperationNameOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.check_operation_name";
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << "is";
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void CheckOperationNameOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckResultCountOp definitions
//===----------------------------------------------------------------------===//

CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(CheckResultCountOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CheckResultCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckResultCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckResultCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckResultCountOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckResultCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckResultCountOpAdaptor::count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::UnitAttr CheckResultCountOpAdaptor::compareAtLeast() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("compareAtLeast").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult CheckResultCountOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_count = odsAttrs.get("count");
  if (!tblgen_count) return emitError(loc, "'pdl_interp.check_result_count' op ""requires attribute 'count'");
    if (!((((tblgen_count.isa<::mlir::IntegerAttr>())) && ((tblgen_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_count.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  {
  auto tblgen_compareAtLeast = odsAttrs.get("compareAtLeast");
  if (tblgen_compareAtLeast) {
    if (!((tblgen_compareAtLeast.isa<::mlir::UnitAttr>()))) return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CheckResultCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckResultCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckResultCountOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckResultCountOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CheckResultCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckResultCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckResultCountOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckResultCountOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::IntegerAttr CheckResultCountOp::countAttr() {
  return (*this)->getAttr("count").template cast<::mlir::IntegerAttr>();
}

uint32_t CheckResultCountOp::count() {
  auto attr = countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOp::compareAtLeastAttr() {
  return (*this)->getAttr("compareAtLeast").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckResultCountOp::compareAtLeast() {
  auto attr = compareAtLeastAttr();
  return attr != nullptr;
}

void CheckResultCountOp::countAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("count", attr);
}

void CheckResultCountOp::compareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("compareAtLeast", attr);
}

::mlir::Attribute CheckResultCountOp::removeCompareAtLeastAttr() {
  return (*this)->removeAttr("compareAtLeast");
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", count);
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", count);
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(operation);
  odsState.addAttribute("count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute("compareAtLeast", odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckResultCountOp::verify() {
  if (failed(CheckResultCountOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CheckResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.addAttribute("compareAtLeast", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseAttribute(countAttr, parser.getBuilder().getIntegerType(32), "count", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void CheckResultCountOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.check_result_count";
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << "is";
  if ((*this)->getAttr("compareAtLeast")) {
  p << ' ' << "at_least";
  }
  p << ' ';
  p.printAttributeWithoutType(countAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"compareAtLeast", "count"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void CheckResultCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypeOp definitions
//===----------------------------------------------------------------------===//

CheckTypeOpAdaptor::CheckTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CheckTypeOpAdaptor::CheckTypeOpAdaptor(CheckTypeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CheckTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypeOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CheckTypeOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::LogicalResult CheckTypeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_type = odsAttrs.get("type");
  if (!tblgen_type) return emitError(loc, "'pdl_interp.check_type' op ""requires attribute 'type'");
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'pdl_interp.check_type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CheckTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypeOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckTypeOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CheckTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckTypeOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckTypeOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::TypeAttr CheckTypeOp::typeAttr() {
  return (*this)->getAttr("type").template cast<::mlir::TypeAttr>();
}

::mlir::Type CheckTypeOp::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void CheckTypeOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr("type", attr);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute("type", type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute("type", type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute("type", ::mlir::TypeAttr::get(type));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute("type", ::mlir::TypeAttr::get(type));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckTypeOp::verify() {
  if (failed(CheckTypeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CheckTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::TypeAttr typeAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseAttribute(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void CheckTypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.check_type";
  p << ' ';
  p << value();
  p << ' ' << "is";
  p << ' ';
  p.printAttributeWithoutType(typeAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"type"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void CheckTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypesOp definitions
//===----------------------------------------------------------------------===//

CheckTypesOpAdaptor::CheckTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CheckTypesOpAdaptor::CheckTypesOpAdaptor(CheckTypesOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CheckTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypesOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CheckTypesOpAdaptor::types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult CheckTypesOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_types = odsAttrs.get("types");
  if (!tblgen_types) return emitError(loc, "'pdl_interp.check_types' op ""requires attribute 'types'");
    if (!(((tblgen_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })))) return emitError(loc, "'pdl_interp.check_types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CheckTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypesOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckTypesOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CheckTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckTypesOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckTypesOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::ArrayAttr CheckTypesOp::typesAttr() {
  return (*this)->getAttr("types").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CheckTypesOp::types() {
  auto attr = typesAttr();
  return attr;
}

void CheckTypesOp::typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("types", attr);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute("types", types);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute("types", types);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckTypesOp::verify() {
  if (failed(CheckTypesOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CheckTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr typesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("are"))
    return ::mlir::failure();

  if (parser.parseAttribute(typesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "types", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void CheckTypesOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.check_types";
  p << ' ';
  p << value();
  p << ' ' << "are";
  p << ' ';
  p.printAttributeWithoutType(typesAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"types"});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void CheckTypesOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateAttributeOp definitions
//===----------------------------------------------------------------------===//

CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(CreateAttributeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CreateAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CreateAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CreateAttributeOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").cast<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult CreateAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_attribute' op ""requires attribute 'value'");
    if (!((true))) return emitError(loc, "'pdl_interp.create_attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CreateAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateAttributeOp::attribute() {
  return *getODSResults(0).begin();
}

::mlir::Attribute CreateAttributeOp::valueAttr() {
  return (*this)->getAttr("value").template cast<::mlir::Attribute>();
}

::mlir::Attribute CreateAttributeOp::value() {
  auto attr = valueAttr();
  return attr;
}

void CreateAttributeOp::valueAttr(::mlir::Attribute attr) {
  (*this)->setAttr("value", attr);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::AttributeType>(), value);
    
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(attribute);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateAttributeOp::verify() {
  if (failed(CreateAttributeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CreateAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;

  if (parser.parseAttribute(valueAttr, "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateAttributeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.create_attribute";
  p << ' ';
  p.printAttribute(valueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

void CreateAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateOperationOp definitions
//===----------------------------------------------------------------------===//

CreateOperationOpAdaptor::CreateOperationOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CreateOperationOpAdaptor::CreateOperationOpAdaptor(CreateOperationOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CreateOperationOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateOperationOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange CreateOperationOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CreateOperationOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ValueRange CreateOperationOpAdaptor::attributes() {
  return getODSOperands(1);
}

::mlir::ValueRange CreateOperationOpAdaptor::types() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr CreateOperationOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CreateOperationOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr CreateOperationOpAdaptor::attributeNames() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("attributeNames").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult CreateOperationOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_attributeNames = odsAttrs.get("attributeNames");
  if (!tblgen_attributeNames) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'attributeNames'");
    if (!(((tblgen_attributeNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_attributeNames.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::StringAttr>()); })))) return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'attributeNames' failed to satisfy constraint: string array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CreateOperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range CreateOperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateOperationOp::operands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range CreateOperationOp::attributes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CreateOperationOp::types() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CreateOperationOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CreateOperationOp::attributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CreateOperationOp::typesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateOperationOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateOperationOp::operation() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr CreateOperationOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef CreateOperationOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOp::attributeNamesAttr() {
  return (*this)->getAttr("attributeNames").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateOperationOp::attributeNames() {
  auto attr = attributeNamesAttr();
  return attr;
}

void CreateOperationOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void CreateOperationOp::attributeNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("attributeNames", attr);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange types, ValueRange operands, ValueRange attributes, ArrayAttr attributeNames) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types);
    
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type operation, ::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  odsState.addAttribute("name", name);
  odsState.addAttribute("attributeNames", attributeNames);
  odsState.addTypes(operation);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  odsState.addAttribute("name", name);
  odsState.addAttribute("attributeNames", attributeNames);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type operation, ::llvm::StringRef name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  odsState.addAttribute("attributeNames", attributeNames);
  odsState.addTypes(operation);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  odsState.addAttribute("attributeNames", attributeNames);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateOperationOp::verify() {
  if (failed(CreateOperationOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CreateOperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> attributesOperands;
  ::llvm::SMLoc attributesOperandsLoc;
  (void)attributesOperandsLoc;
  ::mlir::ArrayAttr attributeNamesAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> typesOperands;
  ::llvm::SMLoc typesOperandsLoc;
  (void)typesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> typesTypes;

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    attributesOperandsLoc = parser.getCurrentLocation();
    if (parseCreateOperationOpAttributes(parser, attributesOperands, attributeNamesAttr))
      return ::mlir::failure();
    result.addAttribute("attributeNames", attributeNamesAttr);
  }
  if (succeeded(parser.parseOptionalArrow())) {
  if (parser.parseLParen())
    return ::mlir::failure();

  typesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(typesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(typesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attributesOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(typesOperands, typesTypes, typesOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(operandsOperands.size()), static_cast<int32_t>(attributesOperands.size()), static_cast<int32_t>(typesOperands.size())}));
  return ::mlir::success();
}

void CreateOperationOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.create_operation";
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  if (!operands().empty()) {
  p << "(";
  p << operands();
  p << ' ' << ":";
  p << ' ';
  p << operands().getTypes();
  p << ")";
  }
  p << ' ';
  printCreateOperationOpAttributes(p, *this, attributes(), attributeNamesAttr());
  if (!types().empty()) {
  p << ' ' << "->";
  p << ' ' << "(";
  p << types();
  p << ' ' << ":";
  p << ' ';
  p << types().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "name", "attributeNames"});
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypeOp definitions
//===----------------------------------------------------------------------===//

CreateTypeOpAdaptor::CreateTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CreateTypeOpAdaptor::CreateTypeOpAdaptor(CreateTypeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CreateTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CreateTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CreateTypeOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("value").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::LogicalResult CreateTypeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_type' op ""requires attribute 'value'");
    if (!(((tblgen_value.isa<::mlir::TypeAttr>())) && ((tblgen_value.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'pdl_interp.create_type' op ""attribute 'value' failed to satisfy constraint: any type attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CreateTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateTypeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::TypeAttr CreateTypeOp::valueAttr() {
  return (*this)->getAttr("value").template cast<::mlir::TypeAttr>();
}

::mlir::Type CreateTypeOp::value() {
  auto attr = valueAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void CreateTypeOp::valueAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr("value", attr);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeAttr type) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), type);
    
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::TypeAttr value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type value) {
  odsState.addAttribute("value", ::mlir::TypeAttr::get(value));
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type value) {
  odsState.addAttribute("value", ::mlir::TypeAttr::get(value));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTypeOp::verify() {
  if (failed(CreateTypeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CreateTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr valueAttr;

  if (parser.parseAttribute(valueAttr, parser.getBuilder().getType<::mlir::NoneType>(), "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.create_type";
  p << ' ';
  p.printAttributeWithoutType(valueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

void CreateTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypesOp definitions
//===----------------------------------------------------------------------===//

CreateTypesOpAdaptor::CreateTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CreateTypesOpAdaptor::CreateTypesOpAdaptor(CreateTypesOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CreateTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CreateTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CreateTypesOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("value").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult CreateTypesOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_types' op ""requires attribute 'value'");
    if (!(((tblgen_value.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_value.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })))) return emitError(loc, "'pdl_interp.create_types' op ""attribute 'value' failed to satisfy constraint: type array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CreateTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateTypesOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr CreateTypesOp::valueAttr() {
  return (*this)->getAttr("value").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateTypesOp::value() {
  auto attr = valueAttr();
  return attr;
}

void CreateTypesOp::valueAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("value", attr);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ArrayAttr value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(result);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTypesOp::verify() {
  if (failed(CreateTypesOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CreateTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr valueAttr;

  if (parser.parseAttribute(valueAttr, parser.getBuilder().getType<::mlir::NoneType>(), "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypesOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.create_types";
  p << ' ';
  p.printAttributeWithoutType(valueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

void CreateTypesOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::EraseOp definitions
//===----------------------------------------------------------------------===//

EraseOpAdaptor::EraseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

EraseOpAdaptor::EraseOpAdaptor(EraseOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange EraseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> EraseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EraseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EraseOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr EraseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult EraseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> EraseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EraseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EraseOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EraseOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> EraseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EraseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation) {
  odsState.addOperands(operation);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation) {
  odsState.addOperands(operation);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EraseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EraseOp::verify() {
  if (failed(EraseOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult EraseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.erase";
  p << ' ';
  p << operation();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FinalizeOp definitions
//===----------------------------------------------------------------------===//

FinalizeOpAdaptor::FinalizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FinalizeOpAdaptor::FinalizeOpAdaptor(FinalizeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FinalizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FinalizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FinalizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FinalizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FinalizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FinalizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FinalizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FinalizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FinalizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FinalizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FinalizeOp::verify() {
  if (failed(FinalizeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult FinalizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void FinalizeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.finalize";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void FinalizeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeOp definitions
//===----------------------------------------------------------------------===//

GetAttributeOpAdaptor::GetAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetAttributeOpAdaptor::GetAttributeOpAdaptor(GetAttributeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GetAttributeOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult GetAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl_interp.get_attribute' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl_interp.get_attribute' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetAttributeOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeOp::attribute() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr GetAttributeOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef GetAttributeOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

void GetAttributeOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value operation, ::mlir::StringAttr name) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", name);
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::StringAttr name) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value operation, ::llvm::StringRef name) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::llvm::StringRef name) {
  odsState.addOperands(operation);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetAttributeOp::verify() {
  if (failed(GetAttributeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(operationOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_attribute";
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
}

void GetAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeTypeOp definitions
//===----------------------------------------------------------------------===//

GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(GetAttributeTypeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetAttributeTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetAttributeTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetAttributeTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeTypeOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetAttributeTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetAttributeTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetAttributeTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetAttributeTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeTypeOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetAttributeTypeOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetAttributeTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetAttributeTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeTypeOp::result() {
  return *getODSResults(0).begin();
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), value);
    
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetAttributeTypeOp::verify() {
  if (failed(GetAttributeTypeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetAttributeTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeTypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_attribute_type";
  p << ' ' << "of";
  p << ' ';
  p << value();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetAttributeTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetDefiningOpOp definitions
//===----------------------------------------------------------------------===//

GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(GetDefiningOpOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetDefiningOpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetDefiningOpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetDefiningOpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetDefiningOpOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetDefiningOpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetDefiningOpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetDefiningOpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetDefiningOpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetDefiningOpOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetDefiningOpOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetDefiningOpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetDefiningOpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetDefiningOpOp::operation() {
  return *getODSResults(0).begin();
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type operation, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(operation);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetDefiningOpOp::verify() {
  if (failed(GetDefiningOpOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetDefiningOpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetDefiningOpOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_defining_op";
  p << ' ' << "of";
  p << ' ';
  p << value();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetDefiningOpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandOp definitions
//===----------------------------------------------------------------------===//

GetOperandOpAdaptor::GetOperandOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetOperandOpAdaptor::GetOperandOpAdaptor(GetOperandOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetOperandOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetOperandOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetOperandOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetOperandOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GetOperandOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (!tblgen_index) return emitError(loc, "'pdl_interp.get_operand' op ""requires attribute 'index'");
    if (!((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.get_operand' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetOperandOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetOperandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetOperandOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetOperandOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetOperandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandOp::value() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetOperandOp::indexAttr() {
  return (*this)->getAttr("index").template cast<::mlir::IntegerAttr>();
}

uint32_t GetOperandOp::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

void GetOperandOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value operation, ::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", index);
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value operation, uint32_t index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, uint32_t index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetOperandOp::verify() {
  if (failed(GetOperandOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetOperandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;

  if (parser.parseAttribute(indexAttr, parser.getBuilder().getIntegerType(32), "index", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(operationOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_operand";
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetOperandOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandsOp definitions
//===----------------------------------------------------------------------===//

GetOperandsOpAdaptor::GetOperandsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetOperandsOpAdaptor::GetOperandsOpAdaptor(GetOperandsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetOperandsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetOperandsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetOperandsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandsOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetOperandsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandsOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GetOperandsOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (tblgen_index) {
    if (!((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.get_operands' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetOperandsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetOperandsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandsOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetOperandsOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetOperandsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetOperandsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandsOp::value() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetOperandsOp::indexAttr() {
  return (*this)->getAttr("index").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> GetOperandsOp::index() {
  auto attr = indexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void GetOperandsOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

::mlir::Attribute GetOperandsOp::removeIndexAttr() {
  return (*this)->removeAttr("index");
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value operation, Optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, operation,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value operation, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  if (index) {
  odsState.addAttribute("index", index);
  }
  odsState.addTypes(value);
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  if (index) {
  odsState.addAttribute("index", index);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetOperandsOp::verify() {
  if (failed(GetOperandsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetOperandsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32), "index", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandsOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_operands";
  if ((*this)->getAttr("index")) {
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  }
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetOperandsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultOp definitions
//===----------------------------------------------------------------------===//

GetResultOpAdaptor::GetResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetResultOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetResultOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (!tblgen_index) return emitError(loc, "'pdl_interp.get_result' op ""requires attribute 'index'");
    if (!((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.get_result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetResultOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetResultOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOp::value() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetResultOp::indexAttr() {
  return (*this)->getAttr("index").template cast<::mlir::IntegerAttr>();
}

uint32_t GetResultOp::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value operation, ::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", index);
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value operation, uint32_t index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, uint32_t index) {
  odsState.addOperands(operation);
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetResultOp::verify() {
  if (failed(GetResultOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;

  if (parser.parseAttribute(indexAttr, parser.getBuilder().getIntegerType(32), "index", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(operationOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_result";
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetResultOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultsOp definitions
//===----------------------------------------------------------------------===//

GetResultsOpAdaptor::GetResultsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetResultsOpAdaptor::GetResultsOpAdaptor(GetResultsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetResultsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetResultsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetResultsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultsOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetResultsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultsOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GetResultsOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (tblgen_index) {
    if (!((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.get_results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetResultsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultsOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetResultsOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetResultsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultsOp::value() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetResultsOp::indexAttr() {
  return (*this)->getAttr("index").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> GetResultsOp::index() {
  auto attr = indexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void GetResultsOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

::mlir::Attribute GetResultsOp::removeIndexAttr() {
  return (*this)->removeAttr("index");
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value operation, Optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, operation,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value operation) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::ValueType>()), operation,
            IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value operation, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  if (index) {
  odsState.addAttribute("index", index);
  }
  odsState.addTypes(value);
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(operation);
  if (index) {
  odsState.addAttribute("index", index);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetResultsOp::verify() {
  if (failed(GetResultsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetResultsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32), "index", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultsOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_results";
  if ((*this)->getAttr("index")) {
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  }
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetResultsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetValueTypeOp definitions
//===----------------------------------------------------------------------===//

GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(GetValueTypeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetValueTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetValueTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetValueTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetValueTypeOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetValueTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetValueTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetValueTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetValueTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetValueTypeOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetValueTypeOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetValueTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetValueTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetValueTypeOp::result() {
  return *getODSResults(0).begin();
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      Type valType = value.getType();
      Type typeType = odsBuilder.getType<pdl::TypeType>();
      build(odsBuilder, odsState,
            valType.isa<pdl::RangeType>() ? pdl::RangeType::get(typeType)
                                          : typeType,
            value);
    
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetValueTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetValueTypeOp::verify() {
  if (failed(GetValueTypeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps11(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()(getGetValueTypeOpValueType((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `value` type matches arity of `result`");
  return ::mlir::success();
}

::mlir::ParseResult GetValueTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, getGetValueTypeOpValueType(resultTypes[0]), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetValueTypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.get_value_type";
  p << ' ' << "of";
  p << ' ';
  p << value();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetValueTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::InferredTypesOp definitions
//===----------------------------------------------------------------------===//

InferredTypesOpAdaptor::InferredTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InferredTypesOpAdaptor::InferredTypesOpAdaptor(InferredTypesOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InferredTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InferredTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InferredTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr InferredTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InferredTypesOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> InferredTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InferredTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> InferredTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InferredTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InferredTypesOp::type() {
  return *getODSResults(0).begin();
}

void InferredTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::TypeType>()));
    
}

void InferredTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type type) {
  odsState.addTypes(type);
}

void InferredTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InferredTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InferredTypesOp::verify() {
  if (failed(InferredTypesOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult InferredTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void InferredTypesOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.inferred_types";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::IsNotNullOp definitions
//===----------------------------------------------------------------------===//

IsNotNullOpAdaptor::IsNotNullOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

IsNotNullOpAdaptor::IsNotNullOpAdaptor(IsNotNullOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange IsNotNullOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IsNotNullOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IsNotNullOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsNotNullOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IsNotNullOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IsNotNullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> IsNotNullOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IsNotNullOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsNotNullOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IsNotNullOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IsNotNullOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IsNotNullOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *IsNotNullOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *IsNotNullOp::falseDest() {
  return (*this)->getSuccessor(1);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsNotNullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IsNotNullOp::verify() {
  if (failed(IsNotNullOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult IsNotNullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(fullSuccessors);
  return ::mlir::success();
}

void IsNotNullOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.is_not_null";
  p << ' ';
  p << value();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << "->";
  p << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);
}

void IsNotNullOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::RecordMatchOp definitions
//===----------------------------------------------------------------------===//

RecordMatchOpAdaptor::RecordMatchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

RecordMatchOpAdaptor::RecordMatchOpAdaptor(RecordMatchOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange RecordMatchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RecordMatchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange RecordMatchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange RecordMatchOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange RecordMatchOpAdaptor::matchedOps() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr RecordMatchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr RecordMatchOpAdaptor::rewriter() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::SymbolRefAttr attr = odsAttrs.get("rewriter").cast<::mlir::SymbolRefAttr>();
  return attr;
}

::mlir::StringAttr RecordMatchOpAdaptor::rootKind() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("rootKind").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr RecordMatchOpAdaptor::generatedOps() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("generatedOps").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::IntegerAttr RecordMatchOpAdaptor::benefit() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("benefit").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult RecordMatchOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_rewriter = odsAttrs.get("rewriter");
  if (!tblgen_rewriter) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'rewriter'");
    if (!((tblgen_rewriter.isa<::mlir::SymbolRefAttr>()))) return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rewriter' failed to satisfy constraint: symbol reference attribute");
  }
  {
  auto tblgen_rootKind = odsAttrs.get("rootKind");
  if (tblgen_rootKind) {
    if (!((tblgen_rootKind.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_generatedOps = odsAttrs.get("generatedOps");
  if (tblgen_generatedOps) {
    if (!(((tblgen_generatedOps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_generatedOps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::StringAttr>()); })))) return emitError(loc, "'pdl_interp.record_match' op ""attribute 'generatedOps' failed to satisfy constraint: string array attribute");
  }
  }
  {
  auto tblgen_benefit = odsAttrs.get("benefit");
  if (!tblgen_benefit) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'benefit'");
    if (!((((tblgen_benefit.isa<::mlir::IntegerAttr>())) && ((tblgen_benefit.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!tblgen_benefit.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl_interp.record_match' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> RecordMatchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range RecordMatchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range RecordMatchOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range RecordMatchOp::matchedOps() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RecordMatchOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange RecordMatchOp::matchedOpsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RecordMatchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *RecordMatchOp::dest() {
  return (*this)->getSuccessor(0);
}

::mlir::SymbolRefAttr RecordMatchOp::rewriterAttr() {
  return (*this)->getAttr("rewriter").template cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr RecordMatchOp::rewriter() {
  auto attr = rewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOp::rootKindAttr() {
  return (*this)->getAttr("rootKind").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > RecordMatchOp::rootKind() {
  auto attr = rootKindAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr RecordMatchOp::generatedOpsAttr() {
  return (*this)->getAttr("generatedOps").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > RecordMatchOp::generatedOps() {
  auto attr = generatedOpsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr RecordMatchOp::benefitAttr() {
  return (*this)->getAttr("benefit").template cast<::mlir::IntegerAttr>();
}

uint16_t RecordMatchOp::benefit() {
  auto attr = benefitAttr();
  return attr.getValue().getZExtValue();
}

void RecordMatchOp::rewriterAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr("rewriter", attr);
}

void RecordMatchOp::rootKindAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("rootKind", attr);
}

void RecordMatchOp::generatedOpsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("generatedOps", attr);
}

void RecordMatchOp::benefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("benefit", attr);
}

::mlir::Attribute RecordMatchOp::removeRootKindAttr() {
  return (*this)->removeAttr("rootKind");
}

::mlir::Attribute RecordMatchOp::removeGeneratedOpsAttr() {
  return (*this)->removeAttr("generatedOps");
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute("rewriter", rewriter);
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute("generatedOps", generatedOps);
  }
  odsState.addAttribute("benefit", benefit);
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute("rewriter", rewriter);
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute("generatedOps", generatedOps);
  }
  odsState.addAttribute("benefit", benefit);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute("rewriter", rewriter);
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute("generatedOps", generatedOps);
  }
  odsState.addAttribute("benefit", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute("rewriter", rewriter);
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute("generatedOps", generatedOps);
  }
  odsState.addAttribute("benefit", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RecordMatchOp::verify() {
  if (failed(RecordMatchOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps12(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult RecordMatchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr rewriterAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::ArrayAttr generatedOpsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> matchedOpsOperands;
  ::llvm::SMLoc matchedOpsOperandsLoc;
  (void)matchedOpsOperandsLoc;
  ::mlir::StringAttr rootKindAttr;
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseAttribute(rewriterAttr, parser.getBuilder().getType<::mlir::NoneType>(), "rewriter", result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseKeyword("benefit"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(benefitAttr, parser.getBuilder().getIntegerType(16), "benefit", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("generatedOps"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(generatedOpsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "generatedOps", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("loc"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  matchedOpsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(matchedOpsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("root"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(rootKindAttr, parser.getBuilder().getType<::mlir::NoneType>(), "rootKind", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matchedOpsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(inputsOperands.size()), static_cast<int32_t>(matchedOpsOperands.size())}));
  return ::mlir::success();
}

void RecordMatchOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.record_match";
  p << ' ';
  p.printAttributeWithoutType(rewriterAttr());
  if (!inputs().empty()) {
  p << "(";
  p << inputs();
  p << ' ' << ":";
  p << ' ';
  p << inputs().getTypes();
  p << ")";
  }
  p << ' ' << ":";
  p << ' ' << "benefit";
  p << "(";
  p.printAttributeWithoutType(benefitAttr());
  p << ")";
  p << ",";
  if ((*this)->getAttr("generatedOps")) {
  p << ' ' << "generatedOps";
  p << "(";
  p.printAttributeWithoutType(generatedOpsAttr());
  p << ")";
  p << ",";
  }
  p << ' ' << "loc";
  p << "(";
  p << "[";
  p << matchedOps();
  p << "]";
  p << ")";
  if ((*this)->getAttr("rootKind")) {
  p << ",";
  p << ' ' << "root";
  p << "(";
  p.printAttributeWithoutType(rootKindAttr());
  p << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "rewriter", "benefit", "generatedOps", "rootKind"});
  p << ' ' << "->";
  p << ' ';
  p << dest();
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ReplaceOp definitions
//===----------------------------------------------------------------------===//

ReplaceOpAdaptor::ReplaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReplaceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReplaceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ReplaceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReplaceOpAdaptor::replValues() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ReplaceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReplaceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ReplaceOp::replValues() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ReplaceOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ReplaceOp::replValuesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReplaceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReplaceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  odsState.addOperands(replValues);
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  odsState.addOperands(replValues);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReplaceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReplaceOp::verify() {
  if (failed(ReplaceOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> replValuesTypes;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (!replValuesOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(replValuesTypes))
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.replace";
  p << ' ';
  p << operation();
  p << ' ' << "with";
  p << ' ';
  p << "(";
  if (!replValues().empty()) {
  p << replValues();
  p << ' ' << ":";
  p << ' ';
  p << replValues().getTypes();
  }
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchAttributeOp definitions
//===----------------------------------------------------------------------===//

SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(SwitchAttributeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchAttributeOpAdaptor::attribute() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchAttributeOpAdaptor::caseValues() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SwitchAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_caseValues = odsAttrs.get("caseValues");
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_attribute' op ""requires attribute 'caseValues'");
    if (!((tblgen_caseValues.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl_interp.switch_attribute' op ""attribute 'caseValues' failed to satisfy constraint: array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchAttributeOp::attribute() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchAttributeOp::attributeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchAttributeOp::defaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchAttributeOp::cases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchAttributeOp::caseValuesAttr() {
  return (*this)->getAttr("caseValues").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchAttributeOp::caseValues() {
  auto attr = caseValuesAttr();
  return attr;
}

void SwitchAttributeOp::caseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("caseValues", attr);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value attribute, ArrayRef<Attribute> caseValues, Block * defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, attribute, odsBuilder.getArrayAttr(caseValues),
          defaultDest, dests);
  
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchAttributeOp::verify() {
  if (failed(SwitchAttributeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
}

::mlir::ParseResult SwitchAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType attributeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> attributeOperands(attributeRawOperands);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseAttribute(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  return ::mlir::success();
}

void SwitchAttributeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.switch_attribute";
  p << ' ';
  p << attribute();
  p << ' ' << "to";
  p << ' ';
  p.printAttributeWithoutType(caseValuesAttr());
  p << "(";
  ::llvm::interleaveComma(cases(), p);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  p << ' ' << "->";
  p << ' ';
  p << defaultDest();
}

void SwitchAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperandCountOp definitions
//===----------------------------------------------------------------------===//

SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(SwitchOperandCountOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchOperandCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchOperandCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchOperandCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperandCountOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchOperandCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpAdaptor::caseValues() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("caseValues").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult SwitchOperandCountOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_caseValues = odsAttrs.get("caseValues");
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_operand_count' op ""requires attribute 'caseValues'");
    if (!(((tblgen_caseValues.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_caseValues.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) return emitError(loc, "'pdl_interp.switch_operand_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchOperandCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchOperandCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperandCountOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchOperandCountOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchOperandCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOperandCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOperandCountOp::defaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOperandCountOp::cases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::caseValuesAttr() {
  return (*this)->getAttr("caseValues").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::caseValues() {
  auto attr = caseValuesAttr();
  return attr;
}

void SwitchOperandCountOp::caseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("caseValues", attr);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value operation, ArrayRef<int32_t> counts, Block * defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, operation, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(operation);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(operation);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOperandCountOp::verify() {
  if (failed(SwitchOperandCountOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
}

::mlir::ParseResult SwitchOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseAttribute(caseValuesAttr, "caseValues", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  return ::mlir::success();
}

void SwitchOperandCountOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.switch_operand_count";
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << "to";
  p << ' ';
  p.printAttribute(caseValuesAttr());
  p << "(";
  ::llvm::interleaveComma(cases(), p);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  p << ' ' << "->";
  p << ' ';
  p << defaultDest();
}

void SwitchOperandCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperationNameOp definitions
//===----------------------------------------------------------------------===//

SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(SwitchOperationNameOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchOperationNameOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchOperationNameOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchOperationNameOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperationNameOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchOperationNameOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchOperationNameOpAdaptor::caseValues() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SwitchOperationNameOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_caseValues = odsAttrs.get("caseValues");
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_operation_name' op ""requires attribute 'caseValues'");
    if (!(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::StringAttr>()); })))) return emitError(loc, "'pdl_interp.switch_operation_name' op ""attribute 'caseValues' failed to satisfy constraint: string array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchOperationNameOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchOperationNameOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperationNameOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchOperationNameOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchOperationNameOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOperationNameOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOperationNameOp::defaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOperationNameOp::cases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchOperationNameOp::caseValuesAttr() {
  return (*this)->getAttr("caseValues").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchOperationNameOp::caseValues() {
  auto attr = caseValuesAttr();
  return attr;
}

void SwitchOperationNameOp::caseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("caseValues", attr);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value operation, ArrayRef<OperationName> names, Block * defaultDest, BlockRange dests) {
      auto stringNames = llvm::to_vector<8>(llvm::map_range(names,
          [](OperationName name) { return name.getStringRef(); }));
      build(odsBuilder, odsState, operation, odsBuilder.getStrArrayAttr(stringNames),
            defaultDest, dests);
    
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(operation);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(operation);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOperationNameOp::verify() {
  if (failed(SwitchOperationNameOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
}

::mlir::ParseResult SwitchOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseAttribute(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  return ::mlir::success();
}

void SwitchOperationNameOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.switch_operation_name";
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << "to";
  p << ' ';
  p.printAttributeWithoutType(caseValuesAttr());
  p << "(";
  ::llvm::interleaveComma(cases(), p);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  p << ' ' << "->";
  p << ' ';
  p << defaultDest();
}

void SwitchOperationNameOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchResultCountOp definitions
//===----------------------------------------------------------------------===//

SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(SwitchResultCountOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchResultCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchResultCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchResultCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchResultCountOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchResultCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpAdaptor::caseValues() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("caseValues").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult SwitchResultCountOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_caseValues = odsAttrs.get("caseValues");
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_result_count' op ""requires attribute 'caseValues'");
    if (!(((tblgen_caseValues.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_caseValues.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) return emitError(loc, "'pdl_interp.switch_result_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchResultCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchResultCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchResultCountOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchResultCountOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchResultCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchResultCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchResultCountOp::defaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchResultCountOp::cases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::caseValuesAttr() {
  return (*this)->getAttr("caseValues").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::caseValues() {
  auto attr = caseValuesAttr();
  return attr;
}

void SwitchResultCountOp::caseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("caseValues", attr);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value operation, ArrayRef<int32_t> counts, Block * defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, operation, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(operation);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(operation);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchResultCountOp::verify() {
  if (failed(SwitchResultCountOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
}

::mlir::ParseResult SwitchResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseAttribute(caseValuesAttr, "caseValues", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  return ::mlir::success();
}

void SwitchResultCountOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.switch_result_count";
  p << ' ' << "of";
  p << ' ';
  p << operation();
  p << ' ' << "to";
  p << ' ';
  p.printAttribute(caseValuesAttr());
  p << "(";
  ::llvm::interleaveComma(cases(), p);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  p << ' ' << "->";
  p << ' ';
  p << defaultDest();
}

void SwitchResultCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypeOp definitions
//===----------------------------------------------------------------------===//

SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(SwitchTypeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypeOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypeOpAdaptor::caseValues() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SwitchTypeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_caseValues = odsAttrs.get("caseValues");
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_type' op ""requires attribute 'caseValues'");
    if (!(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })))) return emitError(loc, "'pdl_interp.switch_type' op ""attribute 'caseValues' failed to satisfy constraint: type array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypeOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchTypeOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchTypeOp::defaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchTypeOp::cases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchTypeOp::caseValuesAttr() {
  return (*this)->getAttr("caseValues").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchTypeOp::caseValues() {
  auto attr = caseValuesAttr();
  return attr;
}

void SwitchTypeOp::caseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("caseValues", attr);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block * defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchTypeOp::verify() {
  if (failed(SwitchTypeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
}

::mlir::ParseResult SwitchTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseAttribute(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  return ::mlir::success();
}

void SwitchTypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.switch_type";
  p << ' ';
  p << value();
  p << ' ' << "to";
  p << ' ';
  p.printAttributeWithoutType(caseValuesAttr());
  p << "(";
  ::llvm::interleaveComma(cases(), p);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  p << ' ' << "->";
  p << ' ';
  p << defaultDest();
}

void SwitchTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypesOp definitions
//===----------------------------------------------------------------------===//

SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(SwitchTypesOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypesOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypesOpAdaptor::caseValues() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SwitchTypesOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_caseValues = odsAttrs.get("caseValues");
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_types' op ""requires attribute 'caseValues'");
    if (!(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })); })))) return emitError(loc, "'pdl_interp.switch_types' op ""attribute 'caseValues' failed to satisfy constraint: type-array array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypesOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchTypesOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchTypesOp::defaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchTypesOp::cases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchTypesOp::caseValuesAttr() {
  return (*this)->getAttr("caseValues").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchTypesOp::caseValues() {
  auto attr = caseValuesAttr();
  return attr;
}

void SwitchTypesOp::caseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("caseValues", attr);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block * defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute("caseValues", caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchTypesOp::verify() {
  if (failed(SwitchTypesOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
}

::mlir::ParseResult SwitchTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseAttribute(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  return ::mlir::success();
}

void SwitchTypesOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl_interp.switch_types";
  p << ' ';
  p << value();
  p << ' ' << "to";
  p << ' ';
  p.printAttributeWithoutType(caseValuesAttr());
  p << "(";
  ::llvm::interleaveComma(cases(), p);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  p << ' ' << "->";
  p << ' ';
  p << defaultDest();
}

void SwitchTypesOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl_interp
} // namespace mlir

#endif  // GET_OP_CLASSES

