/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::quant::ConstFakeQuant,
::mlir::quant::ConstFakeQuantPerAxis,
::mlir::quant::CoupledRefOp,
::mlir::quant::DequantizeCastOp,
::mlir::quant::QuantizeCastOp,
::mlir::quant::QuantizeRegionOp,
::mlir::quant::ReturnOp,
::mlir::quant::StatisticsOp,
::mlir::quant::StatisticsRefOp,
::mlir::quant::StorageCastOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace quant {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isF32())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of 32-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::FloatType>())) || ((type.isa<mlir::quant::QuantizedType>()))) || (((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())))) || (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be primitive/tensor/vector of real valued primitive (float or quantized type), but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ((true)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((((type.isa<::mlir::FloatType>())) || ((type.isa<mlir::quant::QuantizedType>()))) || (((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())))) || (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>()))))) || ((((type.isSignlessInteger())) || ((type.isa<mlir::quant::QuantizedType>()))) || (((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())))) || (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>()))))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be , but got " << type;
  }
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::ConstFakeQuant definitions
//===----------------------------------------------------------------------===//

ConstFakeQuantAdaptor::ConstFakeQuantAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConstFakeQuantAdaptor::ConstFakeQuantAdaptor(ConstFakeQuant&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConstFakeQuantAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstFakeQuantAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstFakeQuantAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantAdaptor::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ConstFakeQuantAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FloatAttr ConstFakeQuantAdaptor::min() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("min").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::FloatAttr ConstFakeQuantAdaptor::max() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("max").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::IntegerAttr ConstFakeQuantAdaptor::num_bits() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("num_bits").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::BoolAttr ConstFakeQuantAdaptor::narrow_range() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("narrow_range").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::BoolAttr ConstFakeQuantAdaptor::is_signed() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("is_signed").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::LogicalResult ConstFakeQuantAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_min = odsAttrs.get("min");
  if (!tblgen_min) return emitError(loc, "'quant.const_fake_quant' op ""requires attribute 'min'");
    if (!(((tblgen_min.isa<::mlir::FloatAttr>())) && ((tblgen_min.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'quant.const_fake_quant' op ""attribute 'min' failed to satisfy constraint: 32-bit float attribute");
  }
  {
  auto tblgen_max = odsAttrs.get("max");
  if (!tblgen_max) return emitError(loc, "'quant.const_fake_quant' op ""requires attribute 'max'");
    if (!(((tblgen_max.isa<::mlir::FloatAttr>())) && ((tblgen_max.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'quant.const_fake_quant' op ""attribute 'max' failed to satisfy constraint: 32-bit float attribute");
  }
  {
  auto tblgen_num_bits = odsAttrs.get("num_bits");
  if (!tblgen_num_bits) return emitError(loc, "'quant.const_fake_quant' op ""requires attribute 'num_bits'");
    if (!(((tblgen_num_bits.isa<::mlir::IntegerAttr>())) && ((tblgen_num_bits.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'quant.const_fake_quant' op ""attribute 'num_bits' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_narrow_range = odsAttrs.get("narrow_range");
  if (tblgen_narrow_range) {
    if (!((tblgen_narrow_range.isa<::mlir::BoolAttr>()))) return emitError(loc, "'quant.const_fake_quant' op ""attribute 'narrow_range' failed to satisfy constraint: bool attribute");
  }
  }
  {
  auto tblgen_is_signed = odsAttrs.get("is_signed");
  if (tblgen_is_signed) {
    if (!((tblgen_is_signed.isa<::mlir::BoolAttr>()))) return emitError(loc, "'quant.const_fake_quant' op ""attribute 'is_signed' failed to satisfy constraint: bool attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConstFakeQuant::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstFakeQuant::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuant::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstFakeQuant::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConstFakeQuant::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstFakeQuant::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuant::outputs() {
  return *getODSResults(0).begin();
}

::mlir::FloatAttr ConstFakeQuant::minAttr() {
  return (*this)->getAttr("min").template cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConstFakeQuant::min() {
  auto attr = minAttr();
  return attr.getValue();
}

::mlir::FloatAttr ConstFakeQuant::maxAttr() {
  return (*this)->getAttr("max").template cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConstFakeQuant::max() {
  auto attr = maxAttr();
  return attr.getValue();
}

::mlir::IntegerAttr ConstFakeQuant::num_bitsAttr() {
  return (*this)->getAttr("num_bits").template cast<::mlir::IntegerAttr>();
}

uint64_t ConstFakeQuant::num_bits() {
  auto attr = num_bitsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConstFakeQuant::narrow_rangeAttr() {
  return (*this)->getAttr("narrow_range").template dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuant::narrow_range() {
  auto attr = narrow_rangeAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr ConstFakeQuant::is_signedAttr() {
  return (*this)->getAttr("is_signed").template dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuant::is_signed() {
  auto attr = is_signedAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void ConstFakeQuant::minAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr("min", attr);
}

void ConstFakeQuant::maxAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr("max", attr);
}

void ConstFakeQuant::num_bitsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("num_bits", attr);
}

void ConstFakeQuant::narrow_rangeAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("narrow_range", attr);
}

void ConstFakeQuant::is_signedAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("is_signed", attr);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::mlir::FloatAttr min, ::mlir::FloatAttr max, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("num_bits", num_bits);
  odsState.addAttribute("narrow_range", narrow_range);
  odsState.addAttribute("is_signed", is_signed);
  odsState.addTypes(outputs);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::mlir::FloatAttr min, ::mlir::FloatAttr max, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("num_bits", num_bits);
  odsState.addAttribute("narrow_range", narrow_range);
  odsState.addAttribute("is_signed", is_signed);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::llvm::APFloat min, ::llvm::APFloat max, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min));
  odsState.addAttribute("max", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max));
  odsState.addAttribute("num_bits", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute("narrow_range", odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute("is_signed", odsBuilder.getBoolAttr(is_signed));
  odsState.addTypes(outputs);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::llvm::APFloat min, ::llvm::APFloat max, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min));
  odsState.addAttribute("max", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max));
  odsState.addAttribute("num_bits", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute("narrow_range", odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute("is_signed", odsBuilder.getBoolAttr(is_signed));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuant::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputs, ::mlir::FloatAttr min, ::mlir::FloatAttr max, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("num_bits", num_bits);
  odsState.addAttribute("narrow_range", narrow_range);
  odsState.addAttribute("is_signed", is_signed);
  odsState.addTypes({inputs.getType()});

}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ConstFakeQuant::verify() {
  if (failed(ConstFakeQuantAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ConstFakeQuant::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::ConstFakeQuantPerAxis definitions
//===----------------------------------------------------------------------===//

ConstFakeQuantPerAxisAdaptor::ConstFakeQuantPerAxisAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConstFakeQuantPerAxisAdaptor::ConstFakeQuantPerAxisAdaptor(ConstFakeQuantPerAxis&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConstFakeQuantPerAxisAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstFakeQuantPerAxisAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstFakeQuantPerAxisAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantPerAxisAdaptor::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ConstFakeQuantPerAxisAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ConstFakeQuantPerAxisAdaptor::min() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("min").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ConstFakeQuantPerAxisAdaptor::max() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("max").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::IntegerAttr ConstFakeQuantPerAxisAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr ConstFakeQuantPerAxisAdaptor::num_bits() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("num_bits").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::BoolAttr ConstFakeQuantPerAxisAdaptor::narrow_range() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("narrow_range").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::BoolAttr ConstFakeQuantPerAxisAdaptor::is_signed() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("is_signed").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::LogicalResult ConstFakeQuantPerAxisAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_min = odsAttrs.get("min");
  if (!tblgen_min) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'min'");
    if (!(((tblgen_min.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_min.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32())); })))) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'min' failed to satisfy constraint: 32-bit float array attribute");
  }
  {
  auto tblgen_max = odsAttrs.get("max");
  if (!tblgen_max) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'max'");
    if (!(((tblgen_max.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_max.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32())); })))) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'max' failed to satisfy constraint: 32-bit float array attribute");
  }
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_num_bits = odsAttrs.get("num_bits");
  if (!tblgen_num_bits) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'num_bits'");
    if (!(((tblgen_num_bits.isa<::mlir::IntegerAttr>())) && ((tblgen_num_bits.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'num_bits' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_narrow_range = odsAttrs.get("narrow_range");
  if (tblgen_narrow_range) {
    if (!((tblgen_narrow_range.isa<::mlir::BoolAttr>()))) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'narrow_range' failed to satisfy constraint: bool attribute");
  }
  }
  {
  auto tblgen_is_signed = odsAttrs.get("is_signed");
  if (tblgen_is_signed) {
    if (!((tblgen_is_signed.isa<::mlir::BoolAttr>()))) return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'is_signed' failed to satisfy constraint: bool attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConstFakeQuantPerAxis::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstFakeQuantPerAxis::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantPerAxis::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstFakeQuantPerAxis::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConstFakeQuantPerAxis::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstFakeQuantPerAxis::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantPerAxis::outputs() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::minAttr() {
  return (*this)->getAttr("min").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::min() {
  auto attr = minAttr();
  return attr;
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::maxAttr() {
  return (*this)->getAttr("max").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::max() {
  auto attr = maxAttr();
  return attr;
}

::mlir::IntegerAttr ConstFakeQuantPerAxis::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ConstFakeQuantPerAxis::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConstFakeQuantPerAxis::num_bitsAttr() {
  return (*this)->getAttr("num_bits").template cast<::mlir::IntegerAttr>();
}

uint64_t ConstFakeQuantPerAxis::num_bits() {
  auto attr = num_bitsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConstFakeQuantPerAxis::narrow_rangeAttr() {
  return (*this)->getAttr("narrow_range").template dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuantPerAxis::narrow_range() {
  auto attr = narrow_rangeAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr ConstFakeQuantPerAxis::is_signedAttr() {
  return (*this)->getAttr("is_signed").template dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuantPerAxis::is_signed() {
  auto attr = is_signedAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void ConstFakeQuantPerAxis::minAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("min", attr);
}

void ConstFakeQuantPerAxis::maxAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("max", attr);
}

void ConstFakeQuantPerAxis::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ConstFakeQuantPerAxis::num_bitsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("num_bits", attr);
}

void ConstFakeQuantPerAxis::narrow_rangeAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("narrow_range", attr);
}

void ConstFakeQuantPerAxis::is_signedAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("is_signed", attr);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, ::mlir::IntegerAttr axis, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("axis", axis);
  odsState.addAttribute("num_bits", num_bits);
  odsState.addAttribute("narrow_range", narrow_range);
  odsState.addAttribute("is_signed", is_signed);
  odsState.addTypes(outputs);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, ::mlir::IntegerAttr axis, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("axis", axis);
  odsState.addAttribute("num_bits", num_bits);
  odsState.addAttribute("narrow_range", narrow_range);
  odsState.addAttribute("is_signed", is_signed);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, uint64_t axis, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addAttribute("num_bits", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute("narrow_range", odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute("is_signed", odsBuilder.getBoolAttr(is_signed));
  odsState.addTypes(outputs);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, uint64_t axis, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addAttribute("num_bits", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute("narrow_range", odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute("is_signed", odsBuilder.getBoolAttr(is_signed));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, ::mlir::IntegerAttr axis, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute("min", min);
  odsState.addAttribute("max", max);
  odsState.addAttribute("axis", axis);
  odsState.addAttribute("num_bits", num_bits);
  odsState.addAttribute("narrow_range", narrow_range);
  odsState.addAttribute("is_signed", is_signed);
  odsState.addTypes({inputs.getType()});

}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ConstFakeQuantPerAxis::verify() {
  if (failed(ConstFakeQuantPerAxisAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ConstFakeQuantPerAxis::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::CoupledRefOp definitions
//===----------------------------------------------------------------------===//

CoupledRefOpAdaptor::CoupledRefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoupledRefOpAdaptor::CoupledRefOpAdaptor(CoupledRefOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoupledRefOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoupledRefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoupledRefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoupledRefOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoupledRefOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CoupledRefOpAdaptor::coupledKey() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("coupledKey").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult CoupledRefOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_coupledKey = odsAttrs.get("coupledKey");
  if (!tblgen_coupledKey) return emitError(loc, "'quant.coupled_ref' op ""requires attribute 'coupledKey'");
    if (!((tblgen_coupledKey.isa<::mlir::StringAttr>()))) return emitError(loc, "'quant.coupled_ref' op ""attribute 'coupledKey' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoupledRefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoupledRefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoupledRefOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoupledRefOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoupledRefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoupledRefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr CoupledRefOp::coupledKeyAttr() {
  return (*this)->getAttr("coupledKey").template cast<::mlir::StringAttr>();
}

::llvm::StringRef CoupledRefOp::coupledKey() {
  auto attr = coupledKeyAttr();
  return attr.getValue();
}

void CoupledRefOp::coupledKeyAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("coupledKey", attr);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::mlir::StringAttr coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("coupledKey", coupledKey);
  odsState.addTypes(resultType0);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("coupledKey", coupledKey);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::llvm::StringRef coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("coupledKey", odsBuilder.getStringAttr(coupledKey));
  odsState.addTypes(resultType0);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("coupledKey", odsBuilder.getStringAttr(coupledKey));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoupledRefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("coupledKey", coupledKey);
  odsState.addTypes({arg.getType()});

}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult CoupledRefOp::verify() {
  if (failed(CoupledRefOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::DequantizeCastOp definitions
//===----------------------------------------------------------------------===//

DequantizeCastOpAdaptor::DequantizeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DequantizeCastOpAdaptor::DequantizeCastOpAdaptor(DequantizeCastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DequantizeCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DequantizeCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DequantizeCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DequantizeCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr DequantizeCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DequantizeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> DequantizeCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DequantizeCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DequantizeCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DequantizeCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DequantizeCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DequantizeCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DequantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(resultType0);
}

void DequantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DequantizeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DequantizeCastOp::verify() {
  if (failed(DequantizeCastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void DequantizeCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::QuantizeCastOp definitions
//===----------------------------------------------------------------------===//

QuantizeCastOpAdaptor::QuantizeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

QuantizeCastOpAdaptor::QuantizeCastOpAdaptor(QuantizeCastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange QuantizeCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizeCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange QuantizeCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value QuantizeCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr QuantizeCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult QuantizeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> QuantizeCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range QuantizeCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value QuantizeCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange QuantizeCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> QuantizeCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range QuantizeCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void QuantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(resultType0);
}

void QuantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QuantizeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QuantizeCastOp::verify() {
  if (failed(QuantizeCastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void QuantizeCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::QuantizeRegionOp definitions
//===----------------------------------------------------------------------===//

QuantizeRegionOpAdaptor::QuantizeRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

QuantizeRegionOpAdaptor::QuantizeRegionOpAdaptor(QuantizeRegionOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange QuantizeRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizeRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange QuantizeRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizeRegionOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr QuantizeRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr QuantizeRegionOpAdaptor::input_specs() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("input_specs").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr QuantizeRegionOpAdaptor::output_specs() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("output_specs").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr QuantizeRegionOpAdaptor::logical_kernel() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("logical_kernel").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange QuantizeRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &QuantizeRegionOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult QuantizeRegionOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_input_specs = odsAttrs.get("input_specs");
  if (!tblgen_input_specs) return emitError(loc, "'quant.region' op ""requires attribute 'input_specs'");
    if (!(((tblgen_input_specs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_input_specs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })))) return emitError(loc, "'quant.region' op ""attribute 'input_specs' failed to satisfy constraint: type array attribute");
  }
  {
  auto tblgen_output_specs = odsAttrs.get("output_specs");
  if (!tblgen_output_specs) return emitError(loc, "'quant.region' op ""requires attribute 'output_specs'");
    if (!(((tblgen_output_specs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_specs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())); })))) return emitError(loc, "'quant.region' op ""attribute 'output_specs' failed to satisfy constraint: type array attribute");
  }
  {
  auto tblgen_logical_kernel = odsAttrs.get("logical_kernel");
  if (!tblgen_logical_kernel) return emitError(loc, "'quant.region' op ""requires attribute 'logical_kernel'");
    if (!((tblgen_logical_kernel.isa<::mlir::StringAttr>()))) return emitError(loc, "'quant.region' op ""attribute 'logical_kernel' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> QuantizeRegionOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range QuantizeRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizeRegionOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange QuantizeRegionOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> QuantizeRegionOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range QuantizeRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range QuantizeRegionOp::outputs() {
  return getODSResults(0);
}

::mlir::Region &QuantizeRegionOp::body() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr QuantizeRegionOp::input_specsAttr() {
  return (*this)->getAttr("input_specs").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr QuantizeRegionOp::input_specs() {
  auto attr = input_specsAttr();
  return attr;
}

::mlir::ArrayAttr QuantizeRegionOp::output_specsAttr() {
  return (*this)->getAttr("output_specs").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr QuantizeRegionOp::output_specs() {
  auto attr = output_specsAttr();
  return attr;
}

::mlir::StringAttr QuantizeRegionOp::logical_kernelAttr() {
  return (*this)->getAttr("logical_kernel").template cast<::mlir::StringAttr>();
}

::llvm::StringRef QuantizeRegionOp::logical_kernel() {
  auto attr = logical_kernelAttr();
  return attr.getValue();
}

void QuantizeRegionOp::input_specsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("input_specs", attr);
}

void QuantizeRegionOp::output_specsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("output_specs", attr);
}

void QuantizeRegionOp::logical_kernelAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("logical_kernel", attr);
}

void QuantizeRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::ArrayAttr input_specs, ::mlir::ArrayAttr output_specs, ::mlir::StringAttr logical_kernel) {
  odsState.addOperands(inputs);
  odsState.addAttribute("input_specs", input_specs);
  odsState.addAttribute("output_specs", output_specs);
  odsState.addAttribute("logical_kernel", logical_kernel);
  (void)odsState.addRegion();
  odsState.addTypes(outputs);
}

void QuantizeRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::ArrayAttr input_specs, ::mlir::ArrayAttr output_specs, ::llvm::StringRef logical_kernel) {
  odsState.addOperands(inputs);
  odsState.addAttribute("input_specs", input_specs);
  odsState.addAttribute("output_specs", output_specs);
  odsState.addAttribute("logical_kernel", odsBuilder.getStringAttr(logical_kernel));
  (void)odsState.addRegion();
  odsState.addTypes(outputs);
}

void QuantizeRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QuantizeRegionOp::verify() {
  if (failed(QuantizeRegionOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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
  return verifyRegionOp(*this);
}

void QuantizeRegionOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReturnOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::results() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ReturnOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::results() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::resultsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (failed(ReturnOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StatisticsOp definitions
//===----------------------------------------------------------------------===//

StatisticsOpAdaptor::StatisticsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StatisticsOpAdaptor::StatisticsOpAdaptor(StatisticsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StatisticsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatisticsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StatisticsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StatisticsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr StatisticsOpAdaptor::layerStats() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("layerStats").cast<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr StatisticsOpAdaptor::axisStats() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("axisStats").dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::mlir::IntegerAttr StatisticsOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult StatisticsOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_layerStats = odsAttrs.get("layerStats");
  if (!tblgen_layerStats) return emitError(loc, "'quant.stats' op ""requires attribute 'layerStats'");
    if (!((tblgen_layerStats.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'quant.stats' op ""attribute 'layerStats' failed to satisfy constraint: constant vector/tensor attribute");
  }
  {
  auto tblgen_axisStats = odsAttrs.get("axisStats");
  if (tblgen_axisStats) {
    if (!((tblgen_axisStats.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'quant.stats' op ""attribute 'axisStats' failed to satisfy constraint: constant vector/tensor attribute");
  }
  }
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (tblgen_axis) {
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'quant.stats' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> StatisticsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StatisticsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StatisticsOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StatisticsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StatisticsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ElementsAttr StatisticsOp::layerStatsAttr() {
  return (*this)->getAttr("layerStats").template cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr StatisticsOp::layerStats() {
  auto attr = layerStatsAttr();
  return attr;
}

::mlir::ElementsAttr StatisticsOp::axisStatsAttr() {
  return (*this)->getAttr("axisStats").template dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > StatisticsOp::axisStats() {
  auto attr = axisStatsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr StatisticsOp::axisAttr() {
  return (*this)->getAttr("axis").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> StatisticsOp::axis() {
  auto attr = axisAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void StatisticsOp::layerStatsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("layerStats", attr);
}

void StatisticsOp::axisStatsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("axisStats", attr);
}

void StatisticsOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

::mlir::Attribute StatisticsOp::removeAxisStatsAttr() {
  return (*this)->removeAttr("axisStats");
}

::mlir::Attribute StatisticsOp::removeAxisAttr() {
  return (*this)->removeAttr("axis");
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::mlir::ElementsAttr layerStats, /*optional*/::mlir::ElementsAttr axisStats, /*optional*/::mlir::IntegerAttr axis) {
  odsState.addOperands(arg);
  odsState.addAttribute("layerStats", layerStats);
  if (axisStats) {
  odsState.addAttribute("axisStats", axisStats);
  }
  if (axis) {
  odsState.addAttribute("axis", axis);
  }
  odsState.addTypes(resultType0);
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::ElementsAttr layerStats, /*optional*/::mlir::ElementsAttr axisStats, /*optional*/::mlir::IntegerAttr axis) {
  odsState.addOperands(arg);
  odsState.addAttribute("layerStats", layerStats);
  if (axisStats) {
  odsState.addAttribute("axisStats", axisStats);
  }
  if (axis) {
  odsState.addAttribute("axis", axis);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatisticsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::ElementsAttr layerStats, /*optional*/::mlir::ElementsAttr axisStats, /*optional*/::mlir::IntegerAttr axis) {
  odsState.addOperands(arg);
  odsState.addAttribute("layerStats", layerStats);
  if (axisStats) {
  odsState.addAttribute("axisStats", axisStats);
  }
  if (axis) {
  odsState.addAttribute("axis", axis);
  }
  odsState.addTypes({arg.getType()});

}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult StatisticsOp::verify() {
  if (failed(StatisticsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  auto tensorArg = arg().getType().dyn_cast<TensorType>();
    if (!tensorArg) return emitOpError("arg needs to be tensor type.");

    // Verify layerStats attribute.
    {
      auto layerStatsType = layerStats().getType();
      if (!layerStatsType.getElementType().isa<FloatType>()) {
        return emitOpError(
            "layerStats must have a floating point element type");
      }
      if (layerStatsType.getRank() != 1 || layerStatsType.getDimSize(0) != 2) {
        return emitOpError("layerStats must have shape [2]");
      }
    }
    // Verify axisStats (optional) attribute.
    if (axisStats()) {
      if (!axis()) return emitOpError("axis must be specified for axisStats");

      auto shape = tensorArg.getShape();
      auto argSliceSize = std::accumulate(std::next(shape.begin(),
        *axis()), shape.end(), 1, std::multiplies<int64_t>());

      auto axisStatsType = axisStats()->getType();
      if (!axisStatsType.getElementType().isa<FloatType>()) {
        return emitOpError("axisStats must have a floating point element type");
      }
      if (axisStatsType.getRank() != 2 ||
          axisStatsType.getDimSize(1) != 2 ||
          axisStatsType.getDimSize(0) != argSliceSize) {
        return emitOpError("axisStats must have shape [N,2] "
                           "where N = the slice size defined by the axis dim");
      }
    }
    return success();
}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StatisticsRefOp definitions
//===----------------------------------------------------------------------===//

StatisticsRefOpAdaptor::StatisticsRefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StatisticsRefOpAdaptor::StatisticsRefOpAdaptor(StatisticsRefOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StatisticsRefOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatisticsRefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StatisticsRefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsRefOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StatisticsRefOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr StatisticsRefOpAdaptor::statsKey() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("statsKey").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult StatisticsRefOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_statsKey = odsAttrs.get("statsKey");
  if (!tblgen_statsKey) return emitError(loc, "'quant.stats_ref' op ""requires attribute 'statsKey'");
    if (!((tblgen_statsKey.isa<::mlir::StringAttr>()))) return emitError(loc, "'quant.stats_ref' op ""attribute 'statsKey' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> StatisticsRefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StatisticsRefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsRefOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StatisticsRefOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StatisticsRefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StatisticsRefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr StatisticsRefOp::statsKeyAttr() {
  return (*this)->getAttr("statsKey").template cast<::mlir::StringAttr>();
}

::llvm::StringRef StatisticsRefOp::statsKey() {
  auto attr = statsKeyAttr();
  return attr.getValue();
}

void StatisticsRefOp::statsKeyAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("statsKey", attr);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::mlir::StringAttr statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("statsKey", statsKey);
  odsState.addTypes(resultType0);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("statsKey", statsKey);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::llvm::StringRef statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("statsKey", odsBuilder.getStringAttr(statsKey));
  odsState.addTypes(resultType0);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("statsKey", odsBuilder.getStringAttr(statsKey));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatisticsRefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute("statsKey", statsKey);
  odsState.addTypes({arg.getType()});

}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult StatisticsRefOp::verify() {
  if (failed(StatisticsRefOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StorageCastOp definitions
//===----------------------------------------------------------------------===//

StorageCastOpAdaptor::StorageCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StorageCastOpAdaptor::StorageCastOpAdaptor(StorageCastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StorageCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StorageCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StorageCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StorageCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StorageCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StorageCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> StorageCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StorageCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StorageCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StorageCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StorageCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StorageCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void StorageCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(resultType0);
}

void StorageCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StorageCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StorageCastOp::verify() {
  if (failed(StorageCastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}



void StorageCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace quant
} // namespace mlir

#endif  // GET_OP_CLASSES

