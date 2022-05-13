/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl::ApplyNativeConstraintOp,
::mlir::pdl::ApplyNativeRewriteOp,
::mlir::pdl::AttributeOp,
::mlir::pdl::EraseOp,
::mlir::pdl::OperandOp,
::mlir::pdl::OperandsOp,
::mlir::pdl::OperationOp,
::mlir::pdl::PatternOp,
::mlir::pdl::ReplaceOp,
::mlir::pdl::ResultOp,
::mlir::pdl::ResultsOp,
::mlir::pdl::RewriteEndOp,
::mlir::pdl::RewriteOp,
::mlir::pdl::TypeOp,
::mlir::pdl::TypesOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::PDLType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::TypeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::AttributeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::ValueType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be range of PDL handle for an `mlir::Value` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::ValueType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::AttributeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps10(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps11(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::ValueType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps12(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::pdl::TypeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps13(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ApplyNativeConstraintOp definitions
//===----------------------------------------------------------------------===//

ApplyNativeConstraintOpAdaptor::ApplyNativeConstraintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ApplyNativeConstraintOpAdaptor::ApplyNativeConstraintOpAdaptor(ApplyNativeConstraintOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyNativeConstraintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyNativeConstraintOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyNativeConstraintOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr ApplyNativeConstraintOpAdaptor::constParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ApplyNativeConstraintOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl.apply_native_constraint' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.apply_native_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constParams = odsAttrs.get("constParams");
  if (tblgen_constParams) {
    if (!((tblgen_constParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl.apply_native_constraint' op ""attribute 'constParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ApplyNativeConstraintOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyNativeConstraintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyNativeConstraintOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyNativeConstraintOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyNativeConstraintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyNativeConstraintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr ApplyNativeConstraintOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyNativeConstraintOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr ApplyNativeConstraintOp::constParamsAttr() {
  return (*this)->getAttr("constParams").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyNativeConstraintOp::constParams() {
  auto attr = constParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ApplyNativeConstraintOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void ApplyNativeConstraintOp::constParamsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("constParams", attr);
}

::mlir::Attribute ApplyNativeConstraintOp::removeConstParamsAttr() {
  return (*this)->removeAttr("constParams");
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange args ,  ArrayRef<Attribute> params ) {
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name), args,
            params.empty() ? ArrayAttr() : odsBuilder.getArrayAttr(params));
    
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyNativeConstraintOp::verify() {
  if (failed(ApplyNativeConstraintOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ApplyNativeConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr constParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;

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
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyNativeConstraintOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.apply_native_constraint";
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
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ApplyNativeRewriteOp definitions
//===----------------------------------------------------------------------===//

ApplyNativeRewriteOpAdaptor::ApplyNativeRewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ApplyNativeRewriteOpAdaptor::ApplyNativeRewriteOpAdaptor(ApplyNativeRewriteOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyNativeRewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyNativeRewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyNativeRewriteOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr ApplyNativeRewriteOpAdaptor::constParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ApplyNativeRewriteOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl.apply_native_rewrite' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.apply_native_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constParams = odsAttrs.get("constParams");
  if (tblgen_constParams) {
    if (!((tblgen_constParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl.apply_native_rewrite' op ""attribute 'constParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ApplyNativeRewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyNativeRewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyNativeRewriteOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyNativeRewriteOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyNativeRewriteOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ApplyNativeRewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ApplyNativeRewriteOp::results() {
  return getODSResults(0);
}

::mlir::StringAttr ApplyNativeRewriteOp::nameAttr() {
  return (*this)->getAttr("name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyNativeRewriteOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr ApplyNativeRewriteOp::constParamsAttr() {
  return (*this)->getAttr("constParams").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyNativeRewriteOp::constParams() {
  auto attr = constParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ApplyNativeRewriteOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void ApplyNativeRewriteOp::constParamsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("constParams", attr);
}

::mlir::Attribute ApplyNativeRewriteOp::removeConstParamsAttr() {
  return (*this)->removeAttr("constParams");
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addTypes(results);
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addTypes(results);
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyNativeRewriteOp::verify() {
  if (failed(ApplyNativeRewriteOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult ApplyNativeRewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyNativeRewriteOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.apply_native_rewrite";
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
  p << ' ' << ":";
  p << ' ';
  p << results().getTypes();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name", "constParams"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::AttributeOp definitions
//===----------------------------------------------------------------------===//

AttributeOpAdaptor::AttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AttributeOpAdaptor::AttributeOpAdaptor(AttributeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr AttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute AttributeOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").dyn_cast_or_null<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult AttributeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (tblgen_value) {
    if (!((true))) return emitError(loc, "'pdl.attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AttributeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange AttributeOp::typeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOp::attr() {
  return *getODSResults(0).begin();
}

::mlir::Attribute AttributeOp::valueAttr() {
  return (*this)->getAttr("value").template dyn_cast_or_null<::mlir::Attribute>();
}

::llvm::Optional<::mlir::Attribute> AttributeOp::value() {
  auto attr = valueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

void AttributeOp::valueAttr(::mlir::Attribute attr) {
  (*this)->setAttr("value", attr);
}

::mlir::Attribute AttributeOp::removeValueAttr() {
  return (*this)->removeAttr("value");
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value type ) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), type,
            Attribute());
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute attr) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), Value(), attr);
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attr, /*optional*/::mlir::Value type, /*optional*/::mlir::Attribute value) {
  if (type)
    odsState.addOperands(type);
  if (value) {
  odsState.addAttribute("value", value);
  }
  odsState.addTypes(attr);
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type, /*optional*/::mlir::Attribute value) {
  if (type)
    odsState.addOperands(type);
  if (value) {
  odsState.addAttribute("value", value);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AttributeOp::verify() {
  if (failed(AttributeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult AttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  ::mlir::Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(valueAttr, "value", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (valueAttr) {
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AttributeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.attribute";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  if (type()) {
  p << ' ' << ":";
  p << ' ';
  if (::mlir::Value value = type())
    p << value;
  }
  if ((*this)->getAttr("value")) {
  p << ' ';
  p.printAttribute(valueAttr());
  }
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::EraseOp definitions
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(getOperation(), v.getType(), "operand", index)))
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
  p << "pdl.erase";
  p << ' ';
  p << operation();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperandOp definitions
//===----------------------------------------------------------------------===//

OperandOpAdaptor::OperandOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

OperandOpAdaptor::OperandOpAdaptor(OperandOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange OperandOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OperandOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange OperandOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr OperandOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OperandOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> OperandOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OperandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange OperandOp::typeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> OperandOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OperandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandOp::val() {
  return *getODSResults(0).begin();
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, odsBuilder.getType<ValueType>(), Value());
    
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  odsState.addTypes(val);
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OperandOp::verify() {
  if (failed(OperandOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult OperandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperandOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.operand";
  if (type()) {
  p << ' ' << ":";
  p << ' ';
  if (::mlir::Value value = type())
    p << value;
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperandsOp definitions
//===----------------------------------------------------------------------===//

OperandsOpAdaptor::OperandsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

OperandsOpAdaptor::OperandsOpAdaptor(OperandsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange OperandsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OperandsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange OperandsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandsOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr OperandsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OperandsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> OperandsOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OperandsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandsOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange OperandsOp::typeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> OperandsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OperandsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandsOp::val() {
  return *getODSResults(0).begin();
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, RangeType::get(odsBuilder.getType<ValueType>()),
            Value());
    
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  odsState.addTypes(val);
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperandsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OperandsOp::verify() {
  if (failed(OperandsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps6(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult OperandsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::ValueType>());
  ::mlir::Type odsBuildableType1 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperandsOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.operands";
  if (type()) {
  p << ' ' << ":";
  p << ' ';
  if (::mlir::Value value = type())
    p << value;
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperationOp definitions
//===----------------------------------------------------------------------===//

OperationOpAdaptor::OperationOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

OperationOpAdaptor::OperationOpAdaptor(OperationOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange OperationOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OperationOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange OperationOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange OperationOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ValueRange OperationOpAdaptor::attributes() {
  return getODSOperands(1);
}

::mlir::ValueRange OperationOpAdaptor::types() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr OperationOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr OperationOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr OperationOpAdaptor::attributeNames() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("attributeNames").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult OperationOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    {
  auto tblgen_name = odsAttrs.get("name");
  if (tblgen_name) {
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.operation' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_attributeNames = odsAttrs.get("attributeNames");
  if (!tblgen_attributeNames) return emitError(loc, "'pdl.operation' op ""requires attribute 'attributeNames'");
    if (!(((tblgen_attributeNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_attributeNames.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::StringAttr>()); })))) return emitError(loc, "'pdl.operation' op ""attribute 'attributeNames' failed to satisfy constraint: string array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> OperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range OperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OperationOp::operands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range OperationOp::attributes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range OperationOp::types() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange OperationOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange OperationOp::attributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange OperationOp::typesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> OperationOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OperationOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperationOp::op() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr OperationOp::nameAttr() {
  return (*this)->getAttr("name").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > OperationOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr OperationOp::attributeNamesAttr() {
  return (*this)->getAttr("attributeNames").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr OperationOp::attributeNames() {
  auto attr = attributeNamesAttr();
  return attr;
}

void OperationOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void OperationOp::attributeNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("attributeNames", attr);
}

::mlir::Attribute OperationOp::removeNameAttr() {
  return (*this)->removeAttr("name");
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Optional<StringRef> name ,  ValueRange operandValues ,  ArrayRef<StringRef> attrNames ,  ValueRange attrValues ,  ValueRange resultTypes ) {
      auto nameAttr = name ? StringAttr() : odsBuilder.getStringAttr(*name);
      build(odsBuilder, odsState, odsBuilder.getType<OperationType>(), nameAttr,
            operandValues, attrValues, odsBuilder.getStrArrayAttr(attrNames),
            resultTypes);
    
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type op, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  if (name) {
  odsState.addAttribute("name", name);
  }
  odsState.addAttribute("attributeNames", attributeNames);
  odsState.addTypes(op);
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  if (name) {
  odsState.addAttribute("name", name);
  }
  odsState.addAttribute("attributeNames", attributeNames);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OperationOp::verify() {
  if (failed(OperationOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult OperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (nameAttr) {
  }
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
    if (parseOperationOpAttributes(parser, attributesOperands, attributeNamesAttr))
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

void OperationOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.operation";
  if ((*this)->getAttr("name")) {
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  }
  if (!operands().empty()) {
  p << "(";
  p << operands();
  p << ' ' << ":";
  p << ' ';
  p << operands().getTypes();
  p << ")";
  }
  p << ' ';
  printOperationOpAttributes(p, *this, attributes(), attributeNamesAttr());
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

void OperationOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::PatternOp definitions
//===----------------------------------------------------------------------===//

PatternOpAdaptor::PatternOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PatternOpAdaptor::PatternOpAdaptor(PatternOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PatternOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PatternOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PatternOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr PatternOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr PatternOpAdaptor::rootKind() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("rootKind").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::IntegerAttr PatternOpAdaptor::benefit() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("benefit").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::StringAttr PatternOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange PatternOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PatternOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult PatternOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_rootKind = odsAttrs.get("rootKind");
  if (tblgen_rootKind) {
    if (!((tblgen_rootKind.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.pattern' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_benefit = odsAttrs.get("benefit");
  if (!tblgen_benefit) return emitError(loc, "'pdl.pattern' op ""requires attribute 'benefit'");
    if (!((((tblgen_benefit.isa<::mlir::IntegerAttr>())) && ((tblgen_benefit.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!tblgen_benefit.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl.pattern' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  }
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (tblgen_sym_name) {
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.pattern' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PatternOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PatternOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> PatternOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PatternOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &PatternOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr PatternOp::rootKindAttr() {
  return (*this)->getAttr("rootKind").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > PatternOp::rootKind() {
  auto attr = rootKindAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr PatternOp::benefitAttr() {
  return (*this)->getAttr("benefit").template cast<::mlir::IntegerAttr>();
}

uint16_t PatternOp::benefit() {
  auto attr = benefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr PatternOp::sym_nameAttr() {
  return (*this)->getAttr("sym_name").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > PatternOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void PatternOp::rootKindAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("rootKind", attr);
}

void PatternOp::benefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("benefit", attr);
}

void PatternOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_name", attr);
}

::mlir::Attribute PatternOp::removeRootKindAttr() {
  return (*this)->removeAttr("rootKind");
}

::mlir::Attribute PatternOp::removeSym_nameAttr() {
  return (*this)->removeAttr("sym_name");
}



void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::StringAttr rootKind, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", benefit);
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr rootKind, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", benefit);
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::StringAttr rootKind, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr rootKind, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PatternOp::verify() {
  if (failed(PatternOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult PatternOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::StringAttr rootKindAttr;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  // Parsing an optional symbol name doesn't fail, so no need to check the
  // result.
  (void)parser.parseOptionalSymbolName(sym_nameAttr, "sym_name", result.attributes);
  if (sym_nameAttr) {
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
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void PatternOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.pattern";
  if ((*this)->getAttr("sym_name")) {
  p << ' ';
  p.printSymbolName(sym_nameAttr().getValue());
  }
  p << ' ' << ":";
  p << ' ' << "benefit";
  p << "(";
  p.printAttributeWithoutType(benefitAttr());
  p << ")";
  if ((*this)->getAttr("rootKind")) {
  p << ",";
  p << ' ' << "root";
  p << "(";
  p.printAttributeWithoutType(rootKindAttr());
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name", "benefit", "rootKind"});
  p << ' ';
  p.printRegion(body());
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ReplaceOp definitions
//===----------------------------------------------------------------------===//

ReplaceOpAdaptor::ReplaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReplaceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReplaceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
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

::mlir::Value ReplaceOpAdaptor::replOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ReplaceOpAdaptor::replValues() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr ReplaceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
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

::mlir::Value ReplaceOp::replOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ReplaceOp::replValues() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ReplaceOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ReplaceOp::replOperationMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ReplaceOp::replValuesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ReplaceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReplaceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> replValuesTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> replOperationOperands;
  ::llvm::SMLoc replOperationOperandsLoc;
  (void)replOperationOperandsLoc;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(replValuesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    replOperationOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      replOperationOperands.push_back(operand);
    }
  }
  if (!replOperationOperands.empty()) {
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replOperationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(replOperationOperands.size()), static_cast<int32_t>(replValuesOperands.size())}));
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.replace";
  p << ' ';
  p << operation();
  p << ' ' << "with";
  if (!replValues().empty()) {
  p << "(";
  p << replValues();
  p << ' ' << ":";
  p << ' ';
  p << replValues().getTypes();
  p << ")";
  }
  if (replOperation()) {
  p << ' ';
  if (::mlir::Value value = replOperation())
    p << value;
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ResultOp definitions
//===----------------------------------------------------------------------===//

ResultOpAdaptor::ResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ResultOpAdaptor::ResultOpAdaptor(ResultOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ResultOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultOpAdaptor::parent() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResultOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ResultOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ResultOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (!tblgen_index) return emitError(loc, "'pdl.result' op ""requires attribute 'index'");
    if (!(((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'pdl.result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ResultOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultOp::parent() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResultOp::parentMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultOp::val() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ResultOp::indexAttr() {
  return (*this)->getAttr("index").template cast<::mlir::IntegerAttr>();
}

uint32_t ResultOp::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

void ResultOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, ::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  odsState.addAttribute("index", index);
  odsState.addTypes(val);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, ::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  odsState.addAttribute("index", index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, uint32_t index) {
  odsState.addOperands(parent);
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(val);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, uint32_t index) {
  odsState.addOperands(parent);
  odsState.addAttribute("index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResultOp::verify() {
  if (failed(ResultOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::OperandType parentRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> parentOperands(parentRawOperands);  ::llvm::SMLoc parentOperandsLoc;
  (void)parentOperandsLoc;

  if (parser.parseAttribute(indexAttr, parser.getBuilder().getIntegerType(32), "index", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  parentOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(parentRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(parentOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResultOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.result";
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  p << ' ' << "of";
  p << ' ';
  p << parent();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ResultsOp definitions
//===----------------------------------------------------------------------===//

ResultsOpAdaptor::ResultsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ResultsOpAdaptor::ResultsOpAdaptor(ResultsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ResultsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResultsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResultsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultsOpAdaptor::parent() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResultsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ResultsOpAdaptor::index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ResultsOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_index = odsAttrs.get("index");
  if (tblgen_index) {
    if (!(((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'pdl.results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ResultsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResultsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultsOp::parent() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResultsOp::parentMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ResultsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResultsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultsOp::val() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ResultsOp::indexAttr() {
  return (*this)->getAttr("index").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> ResultsOp::index() {
  auto attr = indexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void ResultsOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("index", attr);
}

::mlir::Attribute ResultsOp::removeIndexAttr() {
  return (*this)->removeAttr("index");
}

void ResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  if (index) {
  odsState.addAttribute("index", index);
  }
  odsState.addTypes(val);
}

void ResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  if (index) {
  odsState.addAttribute("index", index);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResultsOp::verify() {
  if (failed(ResultsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps11(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult ResultsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::OperandType parentRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> parentOperands(parentRawOperands);  ::llvm::SMLoc parentOperandsLoc;
  (void)parentOperandsLoc;
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);

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

  parentOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(parentRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseResultsValueType(parser, indexAttr, valRawTypes[0]))
      return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valTypes);
  if (parser.resolveOperands(parentOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResultsOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.results";
  if ((*this)->getAttr("index")) {
  p << ' ';
  p.printAttributeWithoutType(indexAttr());
  }
  p << ' ' << "of";
  p << ' ';
  p << parent();
  p << ' ';
  printResultsValueType(p, *this, indexAttr(), val().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::RewriteEndOp definitions
//===----------------------------------------------------------------------===//

RewriteEndOpAdaptor::RewriteEndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

RewriteEndOpAdaptor::RewriteEndOpAdaptor(RewriteEndOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange RewriteEndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RewriteEndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RewriteEndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr RewriteEndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult RewriteEndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> RewriteEndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RewriteEndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> RewriteEndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RewriteEndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RewriteEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void RewriteEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RewriteEndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RewriteEndOp::verify() {
  if (failed(RewriteEndOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult RewriteEndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void RewriteEndOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.rewrite_end";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::RewriteOp definitions
//===----------------------------------------------------------------------===//

RewriteOpAdaptor::RewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

RewriteOpAdaptor::RewriteOpAdaptor(RewriteOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange RewriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange RewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOpAdaptor::root() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange RewriteOpAdaptor::externalArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr RewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr RewriteOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr RewriteOpAdaptor::externalConstParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("externalConstParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::RegionRange RewriteOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &RewriteOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult RewriteOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (tblgen_name) {
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_externalConstParams = odsAttrs.get("externalConstParams");
  if (tblgen_externalConstParams) {
    if (!((tblgen_externalConstParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl.rewrite' op ""attribute 'externalConstParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> RewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range RewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOp::root() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range RewriteOp::externalArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RewriteOp::rootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange RewriteOp::externalArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RewriteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &RewriteOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr RewriteOp::nameAttr() {
  return (*this)->getAttr("name").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > RewriteOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr RewriteOp::externalConstParamsAttr() {
  return (*this)->getAttr("externalConstParams").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > RewriteOp::externalConstParams() {
  auto attr = externalConstParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void RewriteOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("name", attr);
}

void RewriteOp::externalConstParamsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("externalConstParams", attr);
}

::mlir::Attribute RewriteOp::removeNameAttr() {
  return (*this)->removeAttr("name");
}

::mlir::Attribute RewriteOp::removeExternalConstParamsAttr() {
  return (*this)->removeAttr("externalConstParams");
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs, /*optional*/::mlir::ArrayAttr externalConstParams) {
  odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  if (name) {
  odsState.addAttribute("name", name);
  }
  if (externalConstParams) {
  odsState.addAttribute("externalConstParams", externalConstParams);
  }
  (void)odsState.addRegion();
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs, /*optional*/::mlir::ArrayAttr externalConstParams) {
  odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  if (name) {
  odsState.addAttribute("name", name);
  }
  if (externalConstParams) {
  odsState.addAttribute("externalConstParams", externalConstParams);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RewriteOp::verify() {
  if (failed(RewriteOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult RewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType rootRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rootOperands(rootRawOperands);  ::llvm::SMLoc rootOperandsLoc;
  (void)rootOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr externalConstParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> externalArgsOperands;
  ::llvm::SMLoc externalArgsOperandsLoc;
  (void)externalArgsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> externalArgsTypes;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  rootOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rootRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("with"))) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(externalConstParamsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "externalConstParams", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (externalConstParamsAttr) {
  }
  if (succeeded(parser.parseOptionalLParen())) {

  externalArgsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(externalArgsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(externalArgsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  }

  {
     auto parseResult = parser.parseOptionalRegion(*bodyRegion);
     if (parseResult.hasValue() && failed(*parseResult))
       return ::mlir::failure();
  }
  if (!bodyRegion->empty()) {
  
  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(rootOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(externalArgsOperands, externalArgsTypes, externalArgsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void RewriteOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.rewrite";
  p << ' ';
  p << root();
  if ((*this)->getAttr("name")) {
  p << ' ' << "with";
  p << ' ';
  p.printAttributeWithoutType(nameAttr());
  if ((*this)->getAttr("externalConstParams")) {
  p << ' ';
  p.printAttributeWithoutType(externalConstParamsAttr());
  }
  if (!externalArgs().empty()) {
  p << "(";
  p << externalArgs();
  p << ' ' << ":";
  p << ' ';
  p << externalArgs().getTypes();
  p << ")";
  }
  }
  if (!body().empty()) {
  p << ' ';

  {
    bool printTerminator = true;
    if (auto *term = body().empty() ? nullptr : body().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    p.printRegion(body(), /*printEntryBlockArgs=*/true,
                  /*printBlockTerminators=*/printTerminator);
  }
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"name", "externalConstParams"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::TypeOp definitions
//===----------------------------------------------------------------------===//

TypeOpAdaptor::TypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TypeOpAdaptor::TypeOpAdaptor(TypeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr TypeOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::mlir::LogicalResult TypeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_type = odsAttrs.get("type");
  if (tblgen_type) {
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'pdl.type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TypeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::TypeAttr TypeOp::typeAttr() {
  return (*this)->getAttr("type").template dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> TypeOp::type() {
  auto attr = typeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

void TypeOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr("type", attr);
}

::mlir::Attribute TypeOp::removeTypeAttr() {
  return (*this)->removeAttr("type");
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::TypeAttr type) {
  if (type) {
  odsState.addAttribute("type", type);
  }
  odsState.addTypes(result);
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::TypeAttr type) {
  if (type) {
  odsState.addAttribute("type", type);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TypeOp::verify() {
  if (failed(TypeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps12(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult TypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr typeAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.type";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"type"});
  if ((*this)->getAttr("type")) {
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(typeAttr());
  }
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::TypesOp definitions
//===----------------------------------------------------------------------===//

TypesOpAdaptor::TypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TypesOpAdaptor::TypesOpAdaptor(TypesOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TypesOpAdaptor::types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("types").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult TypesOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_types = odsAttrs.get("types");
  if (tblgen_types) {
    if (!(((tblgen_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })))) return emitError(loc, "'pdl.types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TypesOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TypesOp::typesAttr() {
  return (*this)->getAttr("types").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TypesOp::types() {
  auto attr = typesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void TypesOp::typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("types", attr);
}

::mlir::Attribute TypesOp::removeTypesAttr() {
  return (*this)->removeAttr("types");
}

void TypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::ArrayAttr types) {
  if (types) {
  odsState.addAttribute("types", types);
  }
  odsState.addTypes(result);
}

void TypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr types) {
  if (types) {
  odsState.addAttribute("types", types);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TypesOp::verify() {
  if (failed(TypesOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps13(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult TypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr typesAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(typesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "types", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypesOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.types";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"types"});
  if ((*this)->getAttr("types")) {
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(typesAttr());
  }
}

} // namespace pdl
} // namespace mlir

#endif  // GET_OP_CLASSES

