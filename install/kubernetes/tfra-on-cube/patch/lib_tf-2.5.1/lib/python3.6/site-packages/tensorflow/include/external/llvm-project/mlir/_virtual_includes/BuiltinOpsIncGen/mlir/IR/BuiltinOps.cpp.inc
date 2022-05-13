/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::FuncOp,
::mlir::ModuleOp,
::mlir::ModuleTerminatorOp,
::mlir::UnrealizedConversionCastOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_BuiltinOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

} // namespace mlir
namespace mlir {

//===----------------------------------------------------------------------===//
// ::mlir::FuncOp definitions
//===----------------------------------------------------------------------===//

FuncOpAdaptor::FuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FuncOpAdaptor::FuncOpAdaptor(FuncOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::TypeAttr FuncOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::StringAttr FuncOpAdaptor::sym_visibility() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_visibility").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange FuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FuncOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'func' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_type = odsAttrs.get("type");
  if (!tblgen_type) return emitError(loc, "'func' op ""requires attribute 'type'");
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'func' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  {
  auto tblgen_sym_visibility = odsAttrs.get("sym_visibility");
  if (tblgen_sym_visibility) {
    if (!((tblgen_sym_visibility.isa<::mlir::StringAttr>()))) return emitError(loc, "'func' op ""attribute 'sym_visibility' failed to satisfy constraint: string attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &FuncOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr FuncOp::sym_nameAttr() {
  return (*this)->getAttr("sym_name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef FuncOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::typeAttr() {
  return (*this)->getAttr("type").template cast<::mlir::TypeAttr>();
}

::mlir::Type FuncOp::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::StringAttr FuncOp::sym_visibilityAttr() {
  return (*this)->getAttr("sym_visibility").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > FuncOp::sym_visibility() {
  auto attr = sym_visibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void FuncOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_name", attr);
}

void FuncOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr("type", attr);
}

void FuncOp::sym_visibilityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_visibility", attr);
}

::mlir::Attribute FuncOp::removeSym_visibilityAttr() {
  return (*this)->removeAttr("sym_visibility");
}



void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute("sym_name", sym_name);
  odsState.addAttribute("type", type);
  if (sym_visibility) {
  odsState.addAttribute("sym_visibility", sym_visibility);
  }
  (void)odsState.addRegion();
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute("sym_name", sym_name);
  odsState.addAttribute("type", type);
  if (sym_visibility) {
  odsState.addAttribute("sym_visibility", sym_visibility);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute("type", ::mlir::TypeAttr::get(type));
  if (sym_visibility) {
  odsState.addAttribute("sym_visibility", sym_visibility);
  }
  (void)odsState.addRegion();
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute("type", ::mlir::TypeAttr::get(type));
  if (sym_visibility) {
  odsState.addAttribute("sym_visibility", sym_visibility);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FuncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseFuncOp(parser, result);
}

void FuncOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(*this, p);
}

::mlir::LogicalResult FuncOp::verify() {
  if (failed(FuncOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace mlir
namespace mlir {

//===----------------------------------------------------------------------===//
// ::mlir::ModuleOp definitions
//===----------------------------------------------------------------------===//

ModuleOpAdaptor::ModuleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ModuleOpAdaptor::ModuleOpAdaptor(ModuleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ModuleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ModuleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ModuleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ModuleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ModuleOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr ModuleOpAdaptor::sym_visibility() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_visibility").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange ModuleOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ModuleOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult ModuleOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (tblgen_sym_name) {
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'module' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_sym_visibility = odsAttrs.get("sym_visibility");
  if (tblgen_sym_visibility) {
    if (!((tblgen_sym_visibility.isa<::mlir::StringAttr>()))) return emitError(loc, "'module' op ""attribute 'sym_visibility' failed to satisfy constraint: string attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ModuleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ModuleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ModuleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ModuleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ModuleOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr ModuleOp::sym_nameAttr() {
  return (*this)->getAttr("sym_name").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > ModuleOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr ModuleOp::sym_visibilityAttr() {
  return (*this)->getAttr("sym_visibility").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > ModuleOp::sym_visibility() {
  auto attr = sym_visibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void ModuleOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_name", attr);
}

void ModuleOp::sym_visibilityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_visibility", attr);
}

::mlir::Attribute ModuleOp::removeSym_nameAttr() {
  return (*this)->removeAttr("sym_name");
}

::mlir::Attribute ModuleOp::removeSym_visibilityAttr() {
  return (*this)->removeAttr("sym_visibility");
}



::mlir::LogicalResult ModuleOp::verify() {
  if (failed(ModuleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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

::mlir::ParseResult ModuleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  // Parsing an optional symbol name doesn't fail, so no need to check the
  // result.
  (void)parser.parseOptionalSymbolName(sym_nameAttr, "sym_name", result.attributes);
  if (sym_nameAttr) {
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();

  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void ModuleOp::print(::mlir::OpAsmPrinter &p) {
  p << "module";
  if ((*this)->getAttr("sym_name")) {
  p << ' ';
  p.printSymbolName(sym_nameAttr().getValue());
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
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

} // namespace mlir
namespace mlir {

//===----------------------------------------------------------------------===//
// ::mlir::ModuleTerminatorOp definitions
//===----------------------------------------------------------------------===//

ModuleTerminatorOpAdaptor::ModuleTerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ModuleTerminatorOpAdaptor::ModuleTerminatorOpAdaptor(ModuleTerminatorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ModuleTerminatorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ModuleTerminatorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ModuleTerminatorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ModuleTerminatorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ModuleTerminatorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ModuleTerminatorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ModuleTerminatorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ModuleTerminatorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ModuleTerminatorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ModuleTerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void ModuleTerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ModuleTerminatorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ModuleTerminatorOp::verify() {
  if (failed(ModuleTerminatorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult ModuleTerminatorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void ModuleTerminatorOp::print(::mlir::OpAsmPrinter &p) {
  p << "module_terminator";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace mlir
namespace mlir {

//===----------------------------------------------------------------------===//
// ::mlir::UnrealizedConversionCastOp definitions
//===----------------------------------------------------------------------===//

UnrealizedConversionCastOpAdaptor::UnrealizedConversionCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UnrealizedConversionCastOpAdaptor::UnrealizedConversionCastOpAdaptor(UnrealizedConversionCastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UnrealizedConversionCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UnrealizedConversionCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange UnrealizedConversionCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange UnrealizedConversionCastOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr UnrealizedConversionCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UnrealizedConversionCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UnrealizedConversionCastOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range UnrealizedConversionCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range UnrealizedConversionCastOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange UnrealizedConversionCastOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UnrealizedConversionCastOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range UnrealizedConversionCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range UnrealizedConversionCastOp::outputs() {
  return getODSResults(0);
}

void UnrealizedConversionCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UnrealizedConversionCastOp::verify() {
  if (failed(UnrealizedConversionCastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BuiltinOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BuiltinOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}





::mlir::ParseResult UnrealizedConversionCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::SmallVector<::mlir::Type, 1> outputsTypes;

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (!inputsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(outputsTypes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(outputsTypes);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UnrealizedConversionCastOp::print(::mlir::OpAsmPrinter &p) {
  p << "unrealized_conversion_cast";
  if (!inputs().empty()) {
  p << ' ';
  p << inputs();
  p << ' ' << ":";
  p << ' ';
  p << inputs().getTypes();
  }
  p << ' ' << "to";
  p << ' ';
  p << outputs().getTypes();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void UnrealizedConversionCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace mlir

#endif  // GET_OP_CLASSES

