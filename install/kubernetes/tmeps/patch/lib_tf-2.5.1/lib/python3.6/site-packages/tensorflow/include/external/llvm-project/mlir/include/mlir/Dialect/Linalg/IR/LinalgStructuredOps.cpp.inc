/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::linalg::BatchMatmulI16I16I32Op,
::mlir::linalg::BatchMatmulI32I32I32Op,
::mlir::linalg::BatchMatmulI8I8I32Op,
::mlir::linalg::BatchMatmulOp,
::mlir::linalg::ConvDHWOp,
::mlir::linalg::ConvHWOp,
::mlir::linalg::ConvInputNCDHWFilterDHWCFOp,
::mlir::linalg::ConvInputNCHWFilterHWCFOp,
::mlir::linalg::ConvInputNCWFilterWCFOp,
::mlir::linalg::ConvInputNDHWCFilterDHWCFOp,
::mlir::linalg::ConvInputNHWCFilterHWCFOp,
::mlir::linalg::ConvInputNWCFilterWCFOp,
::mlir::linalg::ConvNCDHWOp,
::mlir::linalg::ConvNCHWOp,
::mlir::linalg::ConvNCWOp,
::mlir::linalg::ConvNDHWCOp,
::mlir::linalg::ConvNHWCOp,
::mlir::linalg::ConvNWCOp,
::mlir::linalg::ConvOp,
::mlir::linalg::ConvWOp,
::mlir::linalg::CopyOp,
::mlir::linalg::DepthwiseConvInputNHWCFilterHWCFOp,
::mlir::linalg::DepthwiseConvInputNHWCFilterHWCOp,
::mlir::linalg::DotI16I16I32Op,
::mlir::linalg::DotI32I32I32Op,
::mlir::linalg::DotI8I8I32Op,
::mlir::linalg::DotOp,
::mlir::linalg::FillOp,
::mlir::linalg::GenericOp,
::mlir::linalg::IndexedGenericOp,
::mlir::linalg::MatmulColumnMajorOp,
::mlir::linalg::MatmulI16I16I32Op,
::mlir::linalg::MatmulI32I32I32Op,
::mlir::linalg::MatmulI8I8I32Op,
::mlir::linalg::MatmulOp,
::mlir::linalg::MatvecI16I16I32Op,
::mlir::linalg::MatvecI32I32I32Op,
::mlir::linalg::MatvecI8I8I32Op,
::mlir::linalg::MatvecOp,
::mlir::linalg::PoolingMaxOp,
::mlir::linalg::PoolingMinOp,
::mlir::linalg::PoolingNHWCMaxOp,
::mlir::linalg::PoolingNHWCMinOp,
::mlir::linalg::PoolingNHWCSumOp,
::mlir::linalg::PoolingSumOp,
::mlir::linalg::VecmatI16I16I32Op,
::mlir::linalg::VecmatI32I32I32Op,
::mlir::linalg::VecmatI8I8I32Op,
::mlir::linalg::VecmatOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linalg {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::ShapedType>())) && ((true)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be shaped of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(type.cast<::mlir::MemRefType>()) )))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be strided memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::ShapedType>())) && ((true)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be shaped of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::FloatType>())) || ((type.isSignlessInteger())) || (((type.isa<::mlir::VectorType>())) && ((true))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be floating-point or signless integer or vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulI16I16I32Op definitions
//===----------------------------------------------------------------------===//

BatchMatmulI16I16I32OpAdaptor::BatchMatmulI16I16I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BatchMatmulI16I16I32OpAdaptor::BatchMatmulI16I16I32OpAdaptor(BatchMatmulI16I16I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BatchMatmulI16I16I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulI16I16I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange BatchMatmulI16I16I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulI16I16I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulI16I16I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatmulI16I16I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatmulI16I16I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatmulI16I16I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatmulI16I16I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> BatchMatmulI16I16I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range BatchMatmulI16I16I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulI16I16I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulI16I16I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulI16I16I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange BatchMatmulI16I16I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> BatchMatmulI16I16I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulI16I16I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulI16I16I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulI16I16I32Op::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<BatchMatmulI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void BatchMatmulI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<BatchMatmulI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void BatchMatmulI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult BatchMatmulI16I16I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulI16I16I32Op>(parser, result/*TODO:, captures*/);
}

void BatchMatmulI16I16I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatmulI16I16I32Op::verify() {
  if (failed(BatchMatmulI16I16I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulI32I32I32Op definitions
//===----------------------------------------------------------------------===//

BatchMatmulI32I32I32OpAdaptor::BatchMatmulI32I32I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BatchMatmulI32I32I32OpAdaptor::BatchMatmulI32I32I32OpAdaptor(BatchMatmulI32I32I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BatchMatmulI32I32I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulI32I32I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange BatchMatmulI32I32I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulI32I32I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulI32I32I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatmulI32I32I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatmulI32I32I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatmulI32I32I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatmulI32I32I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> BatchMatmulI32I32I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range BatchMatmulI32I32I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulI32I32I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulI32I32I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulI32I32I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange BatchMatmulI32I32I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> BatchMatmulI32I32I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulI32I32I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulI32I32I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulI32I32I32Op::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<BatchMatmulI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void BatchMatmulI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<BatchMatmulI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void BatchMatmulI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult BatchMatmulI32I32I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulI32I32I32Op>(parser, result/*TODO:, captures*/);
}

void BatchMatmulI32I32I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatmulI32I32I32Op::verify() {
  if (failed(BatchMatmulI32I32I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulI8I8I32Op definitions
//===----------------------------------------------------------------------===//

BatchMatmulI8I8I32OpAdaptor::BatchMatmulI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BatchMatmulI8I8I32OpAdaptor::BatchMatmulI8I8I32OpAdaptor(BatchMatmulI8I8I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BatchMatmulI8I8I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulI8I8I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange BatchMatmulI8I8I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulI8I8I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulI8I8I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatmulI8I8I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatmulI8I8I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatmulI8I8I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatmulI8I8I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> BatchMatmulI8I8I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range BatchMatmulI8I8I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulI8I8I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulI8I8I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulI8I8I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange BatchMatmulI8I8I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> BatchMatmulI8I8I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulI8I8I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulI8I8I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulI8I8I32Op::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<BatchMatmulI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void BatchMatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<BatchMatmulI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void BatchMatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult BatchMatmulI8I8I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulI8I8I32Op>(parser, result/*TODO:, captures*/);
}

void BatchMatmulI8I8I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatmulI8I8I32Op::verify() {
  if (failed(BatchMatmulI8I8I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp definitions
//===----------------------------------------------------------------------===//

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(BatchMatmulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange BatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> BatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range BatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange BatchMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult BatchMatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result/*TODO:, captures*/);
}

void BatchMatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatmulOp::verify() {
  if (failed(BatchMatmulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvDHWOp definitions
//===----------------------------------------------------------------------===//

ConvDHWOpAdaptor::ConvDHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvDHWOpAdaptor::ConvDHWOpAdaptor(ConvDHWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvDHWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvDHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvDHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvDHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvDHWOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvDHWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvDHWOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvDHWOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvDHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvDHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvDHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvDHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvDHWOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvDHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvDHWOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvDHWOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvDHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvDHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvDHWOp::region() {
  return (*this)->getRegion(0);
}

void ConvDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvDHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvDHWOp>(parser, result/*TODO:, captures*/);
}

void ConvDHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvDHWOp::verify() {
  if (failed(ConvDHWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvHWOp definitions
//===----------------------------------------------------------------------===//

ConvHWOpAdaptor::ConvHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvHWOpAdaptor::ConvHWOpAdaptor(ConvHWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvHWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvHWOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvHWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvHWOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvHWOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvHWOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvHWOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvHWOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvHWOp::region() {
  return (*this)->getRegion(0);
}

void ConvHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvHWOp>(parser, result/*TODO:, captures*/);
}

void ConvHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvHWOp::verify() {
  if (failed(ConvHWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvInputNCDHWFilterDHWCFOp definitions
//===----------------------------------------------------------------------===//

ConvInputNCDHWFilterDHWCFOpAdaptor::ConvInputNCDHWFilterDHWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvInputNCDHWFilterDHWCFOpAdaptor::ConvInputNCDHWFilterDHWCFOpAdaptor(ConvInputNCDHWFilterDHWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvInputNCDHWFilterDHWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvInputNCDHWFilterDHWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvInputNCDHWFilterDHWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvInputNCDHWFilterDHWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvInputNCDHWFilterDHWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvInputNCDHWFilterDHWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr ConvInputNCDHWFilterDHWCFOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNCDHWFilterDHWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange ConvInputNCDHWFilterDHWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvInputNCDHWFilterDHWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvInputNCDHWFilterDHWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_3d_input_ncdhw_filter_dhwcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.conv_3d_input_ncdhw_filter_dhwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_3d_input_ncdhw_filter_dhwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.conv_3d_input_ncdhw_filter_dhwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvInputNCDHWFilterDHWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvInputNCDHWFilterDHWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvInputNCDHWFilterDHWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvInputNCDHWFilterDHWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvInputNCDHWFilterDHWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvInputNCDHWFilterDHWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvInputNCDHWFilterDHWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvInputNCDHWFilterDHWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvInputNCDHWFilterDHWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvInputNCDHWFilterDHWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr ConvInputNCDHWFilterDHWCFOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNCDHWFilterDHWCFOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNCDHWFilterDHWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNCDHWFilterDHWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ConvInputNCDHWFilterDHWCFOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvInputNCDHWFilterDHWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvInputNCDHWFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNCDHWFilterDHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNCDHWFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNCDHWFilterDHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNCDHWFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void ConvInputNCDHWFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNCDHWFilterDHWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult ConvInputNCDHWFilterDHWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvInputNCDHWFilterDHWCFOp>(parser, result/*TODO:, captures*/);
}

void ConvInputNCDHWFilterDHWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvInputNCDHWFilterDHWCFOp::verify() {
  if (failed(ConvInputNCDHWFilterDHWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvInputNCHWFilterHWCFOp definitions
//===----------------------------------------------------------------------===//

ConvInputNCHWFilterHWCFOpAdaptor::ConvInputNCHWFilterHWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvInputNCHWFilterHWCFOpAdaptor::ConvInputNCHWFilterHWCFOpAdaptor(ConvInputNCHWFilterHWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvInputNCHWFilterHWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvInputNCHWFilterHWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvInputNCHWFilterHWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvInputNCHWFilterHWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvInputNCHWFilterHWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvInputNCHWFilterHWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr ConvInputNCHWFilterHWCFOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNCHWFilterHWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange ConvInputNCHWFilterHWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvInputNCHWFilterHWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvInputNCHWFilterHWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_2d_input_nchw_filter_hwcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_input_nchw_filter_hwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_2d_input_nchw_filter_hwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_input_nchw_filter_hwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvInputNCHWFilterHWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvInputNCHWFilterHWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvInputNCHWFilterHWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvInputNCHWFilterHWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvInputNCHWFilterHWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvInputNCHWFilterHWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvInputNCHWFilterHWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvInputNCHWFilterHWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvInputNCHWFilterHWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvInputNCHWFilterHWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr ConvInputNCHWFilterHWCFOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNCHWFilterHWCFOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNCHWFilterHWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNCHWFilterHWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ConvInputNCHWFilterHWCFOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvInputNCHWFilterHWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvInputNCHWFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNCHWFilterHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNCHWFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNCHWFilterHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNCHWFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void ConvInputNCHWFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNCHWFilterHWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult ConvInputNCHWFilterHWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvInputNCHWFilterHWCFOp>(parser, result/*TODO:, captures*/);
}

void ConvInputNCHWFilterHWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvInputNCHWFilterHWCFOp::verify() {
  if (failed(ConvInputNCHWFilterHWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvInputNCWFilterWCFOp definitions
//===----------------------------------------------------------------------===//

ConvInputNCWFilterWCFOpAdaptor::ConvInputNCWFilterWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvInputNCWFilterWCFOpAdaptor::ConvInputNCWFilterWCFOpAdaptor(ConvInputNCWFilterWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvInputNCWFilterWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvInputNCWFilterWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvInputNCWFilterWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvInputNCWFilterWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvInputNCWFilterWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvInputNCWFilterWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr ConvInputNCWFilterWCFOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNCWFilterWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange ConvInputNCWFilterWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvInputNCWFilterWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvInputNCWFilterWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_1d_input_ncw_filter_wcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) return emitError(loc, "'linalg.conv_1d_input_ncw_filter_wcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_1d_input_ncw_filter_wcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) return emitError(loc, "'linalg.conv_1d_input_ncw_filter_wcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvInputNCWFilterWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvInputNCWFilterWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvInputNCWFilterWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvInputNCWFilterWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvInputNCWFilterWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvInputNCWFilterWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvInputNCWFilterWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvInputNCWFilterWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvInputNCWFilterWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvInputNCWFilterWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr ConvInputNCWFilterWCFOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNCWFilterWCFOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNCWFilterWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNCWFilterWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ConvInputNCWFilterWCFOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvInputNCWFilterWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvInputNCWFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNCWFilterWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNCWFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNCWFilterWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNCWFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void ConvInputNCWFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNCWFilterWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult ConvInputNCWFilterWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvInputNCWFilterWCFOp>(parser, result/*TODO:, captures*/);
}

void ConvInputNCWFilterWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvInputNCWFilterWCFOp::verify() {
  if (failed(ConvInputNCWFilterWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvInputNDHWCFilterDHWCFOp definitions
//===----------------------------------------------------------------------===//

ConvInputNDHWCFilterDHWCFOpAdaptor::ConvInputNDHWCFilterDHWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvInputNDHWCFilterDHWCFOpAdaptor::ConvInputNDHWCFilterDHWCFOpAdaptor(ConvInputNDHWCFilterDHWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvInputNDHWCFilterDHWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvInputNDHWCFilterDHWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvInputNDHWCFilterDHWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvInputNDHWCFilterDHWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvInputNDHWCFilterDHWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvInputNDHWCFilterDHWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr ConvInputNDHWCFilterDHWCFOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNDHWCFilterDHWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange ConvInputNDHWCFilterDHWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvInputNDHWCFilterDHWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvInputNDHWCFilterDHWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_3d_input_ndhwc_filter_dhwcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.conv_3d_input_ndhwc_filter_dhwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_3d_input_ndhwc_filter_dhwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.conv_3d_input_ndhwc_filter_dhwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvInputNDHWCFilterDHWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvInputNDHWCFilterDHWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvInputNDHWCFilterDHWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvInputNDHWCFilterDHWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvInputNDHWCFilterDHWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvInputNDHWCFilterDHWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvInputNDHWCFilterDHWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvInputNDHWCFilterDHWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvInputNDHWCFilterDHWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvInputNDHWCFilterDHWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr ConvInputNDHWCFilterDHWCFOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNDHWCFilterDHWCFOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNDHWCFilterDHWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNDHWCFilterDHWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ConvInputNDHWCFilterDHWCFOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvInputNDHWCFilterDHWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvInputNDHWCFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNDHWCFilterDHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNDHWCFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNDHWCFilterDHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNDHWCFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void ConvInputNDHWCFilterDHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNDHWCFilterDHWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult ConvInputNDHWCFilterDHWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvInputNDHWCFilterDHWCFOp>(parser, result/*TODO:, captures*/);
}

void ConvInputNDHWCFilterDHWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvInputNDHWCFilterDHWCFOp::verify() {
  if (failed(ConvInputNDHWCFilterDHWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvInputNHWCFilterHWCFOp definitions
//===----------------------------------------------------------------------===//

ConvInputNHWCFilterHWCFOpAdaptor::ConvInputNHWCFilterHWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvInputNHWCFilterHWCFOpAdaptor::ConvInputNHWCFilterHWCFOpAdaptor(ConvInputNHWCFilterHWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvInputNHWCFilterHWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvInputNHWCFilterHWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvInputNHWCFilterHWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvInputNHWCFilterHWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvInputNHWCFilterHWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvInputNHWCFilterHWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr ConvInputNHWCFilterHWCFOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNHWCFilterHWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange ConvInputNHWCFilterHWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvInputNHWCFilterHWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvInputNHWCFilterHWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_2d_input_nhwc_filter_hwcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_input_nhwc_filter_hwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_2d_input_nhwc_filter_hwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_input_nhwc_filter_hwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvInputNHWCFilterHWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvInputNHWCFilterHWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvInputNHWCFilterHWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvInputNHWCFilterHWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvInputNHWCFilterHWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvInputNHWCFilterHWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvInputNHWCFilterHWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvInputNHWCFilterHWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvInputNHWCFilterHWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvInputNHWCFilterHWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr ConvInputNHWCFilterHWCFOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNHWCFilterHWCFOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNHWCFilterHWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNHWCFilterHWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ConvInputNHWCFilterHWCFOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvInputNHWCFilterHWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNHWCFilterHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNHWCFilterHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void ConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNHWCFilterHWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult ConvInputNHWCFilterHWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvInputNHWCFilterHWCFOp>(parser, result/*TODO:, captures*/);
}

void ConvInputNHWCFilterHWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvInputNHWCFilterHWCFOp::verify() {
  if (failed(ConvInputNHWCFilterHWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvInputNWCFilterWCFOp definitions
//===----------------------------------------------------------------------===//

ConvInputNWCFilterWCFOpAdaptor::ConvInputNWCFilterWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvInputNWCFilterWCFOpAdaptor::ConvInputNWCFilterWCFOpAdaptor(ConvInputNWCFilterWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvInputNWCFilterWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvInputNWCFilterWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvInputNWCFilterWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvInputNWCFilterWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvInputNWCFilterWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvInputNWCFilterWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr ConvInputNWCFilterWCFOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNWCFilterWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange ConvInputNWCFilterWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvInputNWCFilterWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvInputNWCFilterWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_1d_input_nwc_filter_wcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) return emitError(loc, "'linalg.conv_1d_input_nwc_filter_wcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_1d_input_nwc_filter_wcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) return emitError(loc, "'linalg.conv_1d_input_nwc_filter_wcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvInputNWCFilterWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvInputNWCFilterWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvInputNWCFilterWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvInputNWCFilterWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvInputNWCFilterWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvInputNWCFilterWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvInputNWCFilterWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvInputNWCFilterWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvInputNWCFilterWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvInputNWCFilterWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr ConvInputNWCFilterWCFOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNWCFilterWCFOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr ConvInputNWCFilterWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ConvInputNWCFilterWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ConvInputNWCFilterWCFOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvInputNWCFilterWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvInputNWCFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNWCFilterWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNWCFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvInputNWCFilterWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvInputNWCFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void ConvInputNWCFilterWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNWCFilterWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult ConvInputNWCFilterWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvInputNWCFilterWCFOp>(parser, result/*TODO:, captures*/);
}

void ConvInputNWCFilterWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvInputNWCFilterWCFOp::verify() {
  if (failed(ConvInputNWCFilterWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCDHWOp definitions
//===----------------------------------------------------------------------===//

ConvNCDHWOpAdaptor::ConvNCDHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvNCDHWOpAdaptor::ConvNCDHWOpAdaptor(ConvNCDHWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvNCDHWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvNCDHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvNCDHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNCDHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNCDHWOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvNCDHWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvNCDHWOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvNCDHWOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvNCDHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvNCDHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvNCDHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNCDHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNCDHWOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvNCDHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCDHWOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNCDHWOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvNCDHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNCDHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNCDHWOp::region() {
  return (*this)->getRegion(0);
}

void ConvNCDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNCDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNCDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNCDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNCDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNCDHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNCDHWOp>(parser, result/*TODO:, captures*/);
}

void ConvNCDHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNCDHWOp::verify() {
  if (failed(ConvNCDHWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCHWOp definitions
//===----------------------------------------------------------------------===//

ConvNCHWOpAdaptor::ConvNCHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvNCHWOpAdaptor::ConvNCHWOpAdaptor(ConvNCHWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvNCHWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvNCHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvNCHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNCHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNCHWOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvNCHWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvNCHWOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvNCHWOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvNCHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvNCHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvNCHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNCHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNCHWOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvNCHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCHWOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNCHWOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvNCHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNCHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNCHWOp::region() {
  return (*this)->getRegion(0);
}

void ConvNCHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNCHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNCHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNCHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNCHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNCHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNCHWOp>(parser, result/*TODO:, captures*/);
}

void ConvNCHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNCHWOp::verify() {
  if (failed(ConvNCHWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCWOp definitions
//===----------------------------------------------------------------------===//

ConvNCWOpAdaptor::ConvNCWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvNCWOpAdaptor::ConvNCWOpAdaptor(ConvNCWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvNCWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvNCWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvNCWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNCWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNCWOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvNCWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvNCWOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvNCWOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvNCWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvNCWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvNCWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNCWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNCWOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvNCWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCWOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNCWOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvNCWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNCWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNCWOp::region() {
  return (*this)->getRegion(0);
}

void ConvNCWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNCWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNCWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNCWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNCWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNCWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNCWOp>(parser, result/*TODO:, captures*/);
}

void ConvNCWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNCWOp::verify() {
  if (failed(ConvNCWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNDHWCOp definitions
//===----------------------------------------------------------------------===//

ConvNDHWCOpAdaptor::ConvNDHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvNDHWCOpAdaptor::ConvNDHWCOpAdaptor(ConvNDHWCOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvNDHWCOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvNDHWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvNDHWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNDHWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNDHWCOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvNDHWCOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvNDHWCOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvNDHWCOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvNDHWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvNDHWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvNDHWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNDHWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNDHWCOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvNDHWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNDHWCOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNDHWCOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvNDHWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNDHWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNDHWCOp::region() {
  return (*this)->getRegion(0);
}

void ConvNDHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNDHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNDHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNDHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNDHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNDHWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNDHWCOp>(parser, result/*TODO:, captures*/);
}

void ConvNDHWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNDHWCOp::verify() {
  if (failed(ConvNDHWCOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNHWCOp definitions
//===----------------------------------------------------------------------===//

ConvNHWCOpAdaptor::ConvNHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvNHWCOpAdaptor::ConvNHWCOpAdaptor(ConvNHWCOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvNHWCOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvNHWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvNHWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNHWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNHWCOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvNHWCOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvNHWCOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvNHWCOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvNHWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvNHWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvNHWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNHWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNHWCOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvNHWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNHWCOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNHWCOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvNHWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNHWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNHWCOp::region() {
  return (*this)->getRegion(0);
}

void ConvNHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNHWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNHWCOp>(parser, result/*TODO:, captures*/);
}

void ConvNHWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNHWCOp::verify() {
  if (failed(ConvNHWCOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNWCOp definitions
//===----------------------------------------------------------------------===//

ConvNWCOpAdaptor::ConvNWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvNWCOpAdaptor::ConvNWCOpAdaptor(ConvNWCOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvNWCOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvNWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvNWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNWCOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvNWCOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvNWCOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvNWCOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvNWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvNWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvNWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNWCOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvNWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNWCOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNWCOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvNWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNWCOp::region() {
  return (*this)->getRegion(0);
}

void ConvNWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvNWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvNWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNWCOp>(parser, result/*TODO:, captures*/);
}

void ConvNWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNWCOp::verify() {
  if (failed(ConvNWCOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvOp definitions
//===----------------------------------------------------------------------===//

ConvOpAdaptor::ConvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvOpAdaptor::ConvOpAdaptor(ConvOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvOpAdaptor::filter() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvOpAdaptor::input() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ConvOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ConvOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ConvOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult ConvOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.conv' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.conv' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.conv' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvOp::filter() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvOp::input() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ConvOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConvOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ConvOp::stridesAttr() {
  return (*this)->getAttr("strides").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr ConvOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvOp::paddingAttr() {
  return (*this)->getAttr("padding").template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void ConvOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("strides", attr);
}

void ConvOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void ConvOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("padding", attr);
}

::mlir::Attribute ConvOp::removeStridesAttr() {
  return (*this)->removeAttr("strides");
}

::mlir::Attribute ConvOp::removeDilationsAttr() {
  return (*this)->removeAttr("dilations");
}

::mlir::Attribute ConvOp::removePaddingAttr() {
  return (*this)->removeAttr("padding");
}

void ConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(filter);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void ConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(filter);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvOp::verify() {
  if (failed(ConvOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}







::mlir::ParseResult ConvOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConvOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.conv";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvWOp definitions
//===----------------------------------------------------------------------===//

ConvWOpAdaptor::ConvWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvWOpAdaptor::ConvWOpAdaptor(ConvWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ConvWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvWOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ConvWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ConvWOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ConvWOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ConvWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ConvWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ConvWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvWOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ConvWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvWOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvWOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ConvWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvWOp::region() {
  return (*this)->getRegion(0);
}

void ConvWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<ConvWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void ConvWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvWOp>(parser, result/*TODO:, captures*/);
}

void ConvWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvWOp::verify() {
  if (failed(ConvWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp definitions
//===----------------------------------------------------------------------===//

CopyOpAdaptor::CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CopyOpAdaptor::CopyOpAdaptor(CopyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CopyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CopyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CopyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopyOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopyOpAdaptor::output() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CopyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr CopyOpAdaptor::inputPermutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("inputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMapAttr CopyOpAdaptor::outputPermutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("outputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::RegionRange CopyOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &CopyOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult CopyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_inputPermutation = odsAttrs.get("inputPermutation");
  if (tblgen_inputPermutation) {
    if (!((tblgen_inputPermutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'linalg.copy' op ""attribute 'inputPermutation' failed to satisfy constraint: AffineMap attribute");
  }
  }
  {
  auto tblgen_outputPermutation = odsAttrs.get("outputPermutation");
  if (tblgen_outputPermutation) {
    if (!((tblgen_outputPermutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'linalg.copy' op ""attribute 'outputPermutation' failed to satisfy constraint: AffineMap attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CopyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopyOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopyOp::output() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CopyOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CopyOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CopyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &CopyOp::region() {
  return (*this)->getRegion(0);
}

::mlir::AffineMapAttr CopyOp::inputPermutationAttr() {
  return (*this)->getAttr("inputPermutation").template dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::llvm::Optional< ::mlir::AffineMap > CopyOp::inputPermutation() {
  auto attr = inputPermutationAttr();
  return attr ? ::llvm::Optional< ::mlir::AffineMap >(attr.getValue()) : (::llvm::None);
}

::mlir::AffineMapAttr CopyOp::outputPermutationAttr() {
  return (*this)->getAttr("outputPermutation").template dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::llvm::Optional< ::mlir::AffineMap > CopyOp::outputPermutation() {
  auto attr = outputPermutationAttr();
  return attr ? ::llvm::Optional< ::mlir::AffineMap >(attr.getValue()) : (::llvm::None);
}

void CopyOp::inputPermutationAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr("inputPermutation", attr);
}

void CopyOp::outputPermutationAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr("outputPermutation", attr);
}

::mlir::Attribute CopyOp::removeInputPermutationAttr() {
  return (*this)->removeAttr("inputPermutation");
}

::mlir::Attribute CopyOp::removeOutputPermutationAttr() {
  return (*this)->removeAttr("outputPermutation");
}



::mlir::LogicalResult CopyOp::verify() {
  if (failed(CopyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
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
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}







::mlir::ParseResult CopyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::OpAsmParser::OperandType outputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> outputOperands(outputRawOperands);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  {
    if (parseCopyOpRegion(parser, *regionRegion, inputRawTypes[0], inputRawTypes[0]))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void CopyOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.copy";
  p << "(";
  p << input();
  p << ",";
  p << ' ';
  p << output();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
  p << ' ';
  printCopyOpRegion(p, *this, region(), input().getType(), input().getType());
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConvInputNHWCFilterHWCFOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConvInputNHWCFilterHWCFOpAdaptor::DepthwiseConvInputNHWCFilterHWCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConvInputNHWCFilterHWCFOpAdaptor::DepthwiseConvInputNHWCFilterHWCFOpAdaptor(DepthwiseConvInputNHWCFilterHWCFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConvInputNHWCFilterHWCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCFOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCFOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConvInputNHWCFilterHWCFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConvInputNHWCFilterHWCFOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange DepthwiseConvInputNHWCFilterHWCFOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConvInputNHWCFilterHWCFOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConvInputNHWCFilterHWCFOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.depthwise_conv_2d_input_nhwc_filter_hwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv_2d_input_nhwc_filter_hwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> DepthwiseConvInputNHWCFilterHWCFOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DepthwiseConvInputNHWCFilterHWCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConvInputNHWCFilterHWCFOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConvInputNHWCFilterHWCFOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConvInputNHWCFilterHWCFOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DepthwiseConvInputNHWCFilterHWCFOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DepthwiseConvInputNHWCFilterHWCFOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConvInputNHWCFilterHWCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConvInputNHWCFilterHWCFOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConvInputNHWCFilterHWCFOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConvInputNHWCFilterHWCFOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConvInputNHWCFilterHWCFOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void DepthwiseConvInputNHWCFilterHWCFOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void DepthwiseConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DepthwiseConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCFOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DepthwiseConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void DepthwiseConvInputNHWCFilterHWCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCFOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult DepthwiseConvInputNHWCFilterHWCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConvInputNHWCFilterHWCFOp>(parser, result/*TODO:, captures*/);
}

void DepthwiseConvInputNHWCFilterHWCFOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DepthwiseConvInputNHWCFilterHWCFOp::verify() {
  if (failed(DepthwiseConvInputNHWCFilterHWCFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConvInputNHWCFilterHWCOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConvInputNHWCFilterHWCOpAdaptor::DepthwiseConvInputNHWCFilterHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConvInputNHWCFilterHWCOpAdaptor::DepthwiseConvInputNHWCFilterHWCOpAdaptor(DepthwiseConvInputNHWCFilterHWCOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConvInputNHWCFilterHWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConvInputNHWCFilterHWCOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConvInputNHWCFilterHWCOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConvInputNHWCFilterHWCOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange DepthwiseConvInputNHWCFilterHWCOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConvInputNHWCFilterHWCOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConvInputNHWCFilterHWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.depthwise_conv_2d_input_nhwc_filter_hwc' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv_2d_input_nhwc_filter_hwc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> DepthwiseConvInputNHWCFilterHWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DepthwiseConvInputNHWCFilterHWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConvInputNHWCFilterHWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConvInputNHWCFilterHWCOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConvInputNHWCFilterHWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DepthwiseConvInputNHWCFilterHWCOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DepthwiseConvInputNHWCFilterHWCOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConvInputNHWCFilterHWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConvInputNHWCFilterHWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConvInputNHWCFilterHWCOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConvInputNHWCFilterHWCOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConvInputNHWCFilterHWCOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void DepthwiseConvInputNHWCFilterHWCOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void DepthwiseConvInputNHWCFilterHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DepthwiseConvInputNHWCFilterHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DepthwiseConvInputNHWCFilterHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void DepthwiseConvInputNHWCFilterHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult DepthwiseConvInputNHWCFilterHWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConvInputNHWCFilterHWCOp>(parser, result/*TODO:, captures*/);
}

void DepthwiseConvInputNHWCFilterHWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DepthwiseConvInputNHWCFilterHWCOp::verify() {
  if (failed(DepthwiseConvInputNHWCFilterHWCOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotI16I16I32Op definitions
//===----------------------------------------------------------------------===//

DotI16I16I32OpAdaptor::DotI16I16I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DotI16I16I32OpAdaptor::DotI16I16I32OpAdaptor(DotI16I16I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DotI16I16I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotI16I16I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DotI16I16I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotI16I16I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotI16I16I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DotI16I16I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange DotI16I16I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DotI16I16I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DotI16I16I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> DotI16I16I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DotI16I16I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotI16I16I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotI16I16I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotI16I16I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DotI16I16I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DotI16I16I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DotI16I16I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotI16I16I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotI16I16I32Op::region() {
  return (*this)->getRegion(0);
}

void DotI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DotI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DotI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DotI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DotI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult DotI16I16I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DotI16I16I32Op>(parser, result/*TODO:, captures*/);
}

void DotI16I16I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DotI16I16I32Op::verify() {
  if (failed(DotI16I16I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotI32I32I32Op definitions
//===----------------------------------------------------------------------===//

DotI32I32I32OpAdaptor::DotI32I32I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DotI32I32I32OpAdaptor::DotI32I32I32OpAdaptor(DotI32I32I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DotI32I32I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotI32I32I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DotI32I32I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotI32I32I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotI32I32I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DotI32I32I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange DotI32I32I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DotI32I32I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DotI32I32I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> DotI32I32I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DotI32I32I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotI32I32I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotI32I32I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotI32I32I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DotI32I32I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DotI32I32I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DotI32I32I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotI32I32I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotI32I32I32Op::region() {
  return (*this)->getRegion(0);
}

void DotI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DotI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DotI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DotI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DotI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult DotI32I32I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DotI32I32I32Op>(parser, result/*TODO:, captures*/);
}

void DotI32I32I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DotI32I32I32Op::verify() {
  if (failed(DotI32I32I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotI8I8I32Op definitions
//===----------------------------------------------------------------------===//

DotI8I8I32OpAdaptor::DotI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DotI8I8I32OpAdaptor::DotI8I8I32OpAdaptor(DotI8I8I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DotI8I8I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotI8I8I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DotI8I8I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotI8I8I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotI8I8I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DotI8I8I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange DotI8I8I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DotI8I8I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DotI8I8I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> DotI8I8I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DotI8I8I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotI8I8I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotI8I8I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotI8I8I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DotI8I8I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DotI8I8I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DotI8I8I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotI8I8I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotI8I8I32Op::region() {
  return (*this)->getRegion(0);
}

void DotI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DotI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DotI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<DotI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void DotI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult DotI8I8I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DotI8I8I32Op>(parser, result/*TODO:, captures*/);
}

void DotI8I8I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DotI8I8I32Op::verify() {
  if (failed(DotI8I8I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp definitions
//===----------------------------------------------------------------------===//

DotOpAdaptor::DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DotOpAdaptor::DotOpAdaptor(DotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange DotOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DotOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DotOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> DotOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DotOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DotOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotOp::region() {
  return (*this)->getRegion(0);
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult DotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DotOp>(parser, result/*TODO:, captures*/);
}

void DotOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DotOp::verify() {
  if (failed(DotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp definitions
//===----------------------------------------------------------------------===//

FillOpAdaptor::FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FillOpAdaptor::FillOpAdaptor(FillOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FillOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FillOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOpAdaptor::output() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOpAdaptor::value() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FillOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult FillOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FillOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FillOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOp::output() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOp::value() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FillOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FillOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FillOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range FillOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOp::result() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::Region &FillOp::region() {
  return (*this)->getRegion(0);
}



void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value output, ::mlir::Value value) {
  odsState.addOperands(output);
  odsState.addOperands(value);
  (void)odsState.addRegion();
  if (result)
    odsState.addTypes(result);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value value) {
  odsState.addOperands(output);
  odsState.addOperands(value);
  (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

void FillOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FillOp::verify() {
  if (failed(FillOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}







::mlir::ParseResult FillOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType outputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> outputOperands(outputRawOperands);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> resultTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (parser.parseLParen())
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalArrow())) {

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      resultTypes.push_back(optionalType);
    }
  }
  }
  {
    if (parseFillOpRegion(parser, *regionRegion, outputRawTypes[0], valueRawOperands[0]))
      return ::mlir::failure();
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void FillOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.fill";
  p << "(";
  p << output();
  p << ",";
  p << ' ';
  p << value();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  if (result()) {
  p << ' ' << "->";
  p << ' ';
  p << (result() ? ::llvm::ArrayRef<::mlir::Type>(result().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  p << ' ';
  printFillOpRegion(p, *this, region(), output().getType(), value());
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp definitions
//===----------------------------------------------------------------------===//

GenericOpAdaptor::GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GenericOpAdaptor::GenericOpAdaptor(GenericOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GenericOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange GenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange GenericOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GenericOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_maps() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::doc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("doc").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::library_call() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("library_call").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::sparse() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("sparse").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::RegionRange GenericOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GenericOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult GenericOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
  if (!tblgen_indexing_maps) return emitError(loc, "'linalg.generic' op ""requires attribute 'indexing_maps'");
    if (!(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) return emitError(loc, "'linalg.generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
  auto tblgen_iterator_types = odsAttrs.get("iterator_types");
  if (!tblgen_iterator_types) return emitError(loc, "'linalg.generic' op ""requires attribute 'iterator_types'");
    if (!((tblgen_iterator_types.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
  auto tblgen_doc = odsAttrs.get("doc");
  if (tblgen_doc) {
    if (!((tblgen_doc.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_library_call = odsAttrs.get("library_call");
  if (tblgen_library_call) {
    if (!((tblgen_library_call.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_sparse = odsAttrs.get("sparse");
  if (tblgen_sparse) {
    if (!((tblgen_sparse.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'sparse' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range GenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range GenericOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange GenericOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> GenericOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range GenericOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range GenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &GenericOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr GenericOp::indexing_mapsAttr() {
  return (*this)->getAttr("indexing_maps").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::iterator_typesAttr() {
  return (*this)->getAttr("iterator_types").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOp::docAttr() {
  return (*this)->getAttr("doc").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::library_callAttr() {
  return (*this)->getAttr("library_call").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr GenericOp::sparseAttr() {
  return (*this)->getAttr("sparse").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > GenericOp::sparse() {
  auto attr = sparseAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void GenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("indexing_maps", attr);
}

void GenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("iterator_types", attr);
}

void GenericOp::docAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("doc", attr);
}

void GenericOp::library_callAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("library_call", attr);
}

void GenericOp::sparseAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("sparse", attr);
}

::mlir::Attribute GenericOp::removeDocAttr() {
  return (*this)->removeAttr("doc");
}

::mlir::Attribute GenericOp::removeLibrary_callAttr() {
  return (*this)->removeAttr("library_call");
}

::mlir::Attribute GenericOp::removeSparseAttr() {
  return (*this)->removeAttr("sparse");
}









void GenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call, /*optional*/::mlir::ArrayAttr sparse) {
  odsState.addOperands(inputs);
  odsState.addOperands(outputs);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(outputs.size())}));
  odsState.addAttribute("indexing_maps", indexing_maps);
  odsState.addAttribute("iterator_types", iterator_types);
  if (doc) {
  odsState.addAttribute("doc", doc);
  }
  if (library_call) {
  odsState.addAttribute("library_call", library_call);
  }
  if (sparse) {
  odsState.addAttribute("sparse", sparse);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result_tensors);
}

void GenericOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult GenericOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGenericOp(parser, result);
}

void GenericOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult GenericOp::verify() {
  if (failed(GenericOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::IndexedGenericOp definitions
//===----------------------------------------------------------------------===//

IndexedGenericOpAdaptor::IndexedGenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

IndexedGenericOpAdaptor::IndexedGenericOpAdaptor(IndexedGenericOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange IndexedGenericOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IndexedGenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange IndexedGenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange IndexedGenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange IndexedGenericOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr IndexedGenericOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr IndexedGenericOpAdaptor::indexing_maps() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr IndexedGenericOpAdaptor::iterator_types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr IndexedGenericOpAdaptor::doc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("doc").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr IndexedGenericOpAdaptor::library_call() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("library_call").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr IndexedGenericOpAdaptor::sparse() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("sparse").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::RegionRange IndexedGenericOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &IndexedGenericOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult IndexedGenericOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
  if (!tblgen_indexing_maps) return emitError(loc, "'linalg.indexed_generic' op ""requires attribute 'indexing_maps'");
    if (!(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
  auto tblgen_iterator_types = odsAttrs.get("iterator_types");
  if (!tblgen_iterator_types) return emitError(loc, "'linalg.indexed_generic' op ""requires attribute 'iterator_types'");
    if (!((tblgen_iterator_types.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
  auto tblgen_doc = odsAttrs.get("doc");
  if (tblgen_doc) {
    if (!((tblgen_doc.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_library_call = odsAttrs.get("library_call");
  if (tblgen_library_call) {
    if (!((tblgen_library_call.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_sparse = odsAttrs.get("sparse");
  if (tblgen_sparse) {
    if (!((tblgen_sparse.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'sparse' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> IndexedGenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range IndexedGenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range IndexedGenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range IndexedGenericOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange IndexedGenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange IndexedGenericOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> IndexedGenericOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range IndexedGenericOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IndexedGenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &IndexedGenericOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr IndexedGenericOp::indexing_mapsAttr() {
  return (*this)->getAttr("indexing_maps").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr IndexedGenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr IndexedGenericOp::iterator_typesAttr() {
  return (*this)->getAttr("iterator_types").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr IndexedGenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr IndexedGenericOp::docAttr() {
  return (*this)->getAttr("doc").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > IndexedGenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr IndexedGenericOp::library_callAttr() {
  return (*this)->getAttr("library_call").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > IndexedGenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr IndexedGenericOp::sparseAttr() {
  return (*this)->getAttr("sparse").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > IndexedGenericOp::sparse() {
  auto attr = sparseAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void IndexedGenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("indexing_maps", attr);
}

void IndexedGenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("iterator_types", attr);
}

void IndexedGenericOp::docAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("doc", attr);
}

void IndexedGenericOp::library_callAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("library_call", attr);
}

void IndexedGenericOp::sparseAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("sparse", attr);
}

::mlir::Attribute IndexedGenericOp::removeDocAttr() {
  return (*this)->removeAttr("doc");
}

::mlir::Attribute IndexedGenericOp::removeLibrary_callAttr() {
  return (*this)->removeAttr("library_call");
}

::mlir::Attribute IndexedGenericOp::removeSparseAttr() {
  return (*this)->removeAttr("sparse");
}









void IndexedGenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call, /*optional*/::mlir::ArrayAttr sparse) {
  odsState.addOperands(inputs);
  odsState.addOperands(outputs);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(outputs.size())}));
  odsState.addAttribute("indexing_maps", indexing_maps);
  odsState.addAttribute("iterator_types", iterator_types);
  if (doc) {
  odsState.addAttribute("doc", doc);
  }
  if (library_call) {
  odsState.addAttribute("library_call", library_call);
  }
  if (sparse) {
  odsState.addAttribute("sparse", sparse);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result_tensors);
}

void IndexedGenericOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult IndexedGenericOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGenericOp(parser, result);
}

void IndexedGenericOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult IndexedGenericOp::verify() {
  if (failed(IndexedGenericOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulColumnMajorOp definitions
//===----------------------------------------------------------------------===//

MatmulColumnMajorOpAdaptor::MatmulColumnMajorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulColumnMajorOpAdaptor::MatmulColumnMajorOpAdaptor(MatmulColumnMajorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulColumnMajorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulColumnMajorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatmulColumnMajorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulColumnMajorOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulColumnMajorOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulColumnMajorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulColumnMajorOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulColumnMajorOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulColumnMajorOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatmulColumnMajorOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatmulColumnMajorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulColumnMajorOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulColumnMajorOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulColumnMajorOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulColumnMajorOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatmulColumnMajorOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulColumnMajorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulColumnMajorOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulColumnMajorOp::region() {
  return (*this)->getRegion(0);
}

void MatmulColumnMajorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulColumnMajorOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulColumnMajorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulColumnMajorOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulColumnMajorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatmulColumnMajorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulColumnMajorOp>(parser, result/*TODO:, captures*/);
}

void MatmulColumnMajorOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulColumnMajorOp::verify() {
  if (failed(MatmulColumnMajorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulI16I16I32Op definitions
//===----------------------------------------------------------------------===//

MatmulI16I16I32OpAdaptor::MatmulI16I16I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulI16I16I32OpAdaptor::MatmulI16I16I32OpAdaptor(MatmulI16I16I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulI16I16I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulI16I16I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatmulI16I16I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulI16I16I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulI16I16I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulI16I16I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulI16I16I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulI16I16I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulI16I16I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatmulI16I16I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatmulI16I16I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulI16I16I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulI16I16I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulI16I16I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulI16I16I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatmulI16I16I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulI16I16I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulI16I16I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulI16I16I32Op::region() {
  return (*this)->getRegion(0);
}

void MatmulI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatmulI16I16I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulI16I16I32Op>(parser, result/*TODO:, captures*/);
}

void MatmulI16I16I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulI16I16I32Op::verify() {
  if (failed(MatmulI16I16I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulI32I32I32Op definitions
//===----------------------------------------------------------------------===//

MatmulI32I32I32OpAdaptor::MatmulI32I32I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulI32I32I32OpAdaptor::MatmulI32I32I32OpAdaptor(MatmulI32I32I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulI32I32I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulI32I32I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatmulI32I32I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulI32I32I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulI32I32I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulI32I32I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulI32I32I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulI32I32I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulI32I32I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatmulI32I32I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatmulI32I32I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulI32I32I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulI32I32I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulI32I32I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulI32I32I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatmulI32I32I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulI32I32I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulI32I32I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulI32I32I32Op::region() {
  return (*this)->getRegion(0);
}

void MatmulI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatmulI32I32I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulI32I32I32Op>(parser, result/*TODO:, captures*/);
}

void MatmulI32I32I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulI32I32I32Op::verify() {
  if (failed(MatmulI32I32I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulI8I8I32Op definitions
//===----------------------------------------------------------------------===//

MatmulI8I8I32OpAdaptor::MatmulI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulI8I8I32OpAdaptor::MatmulI8I8I32OpAdaptor(MatmulI8I8I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulI8I8I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulI8I8I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulI8I8I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulI8I8I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulI8I8I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatmulI8I8I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatmulI8I8I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulI8I8I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulI8I8I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulI8I8I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulI8I8I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatmulI8I8I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulI8I8I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulI8I8I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulI8I8I32Op::region() {
  return (*this)->getRegion(0);
}

void MatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatmulI8I8I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulI8I8I32Op>(parser, result/*TODO:, captures*/);
}

void MatmulI8I8I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulI8I8I32Op::verify() {
  if (failed(MatmulI8I8I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp definitions
//===----------------------------------------------------------------------===//

MatmulOpAdaptor::MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulOp::region() {
  return (*this)->getRegion(0);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult MatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulOp>(parser, result/*TODO:, captures*/);
}

void MatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulOp::verify() {
  if (failed(MatmulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecI16I16I32Op definitions
//===----------------------------------------------------------------------===//

MatvecI16I16I32OpAdaptor::MatvecI16I16I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatvecI16I16I32OpAdaptor::MatvecI16I16I32OpAdaptor(MatvecI16I16I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatvecI16I16I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecI16I16I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatvecI16I16I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecI16I16I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecI16I16I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatvecI16I16I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatvecI16I16I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatvecI16I16I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatvecI16I16I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatvecI16I16I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatvecI16I16I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecI16I16I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecI16I16I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecI16I16I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatvecI16I16I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatvecI16I16I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatvecI16I16I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecI16I16I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecI16I16I32Op::region() {
  return (*this)->getRegion(0);
}

void MatvecI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatvecI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatvecI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatvecI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatvecI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatvecI16I16I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatvecI16I16I32Op>(parser, result/*TODO:, captures*/);
}

void MatvecI16I16I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatvecI16I16I32Op::verify() {
  if (failed(MatvecI16I16I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecI32I32I32Op definitions
//===----------------------------------------------------------------------===//

MatvecI32I32I32OpAdaptor::MatvecI32I32I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatvecI32I32I32OpAdaptor::MatvecI32I32I32OpAdaptor(MatvecI32I32I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatvecI32I32I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecI32I32I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatvecI32I32I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecI32I32I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecI32I32I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatvecI32I32I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatvecI32I32I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatvecI32I32I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatvecI32I32I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatvecI32I32I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatvecI32I32I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecI32I32I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecI32I32I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecI32I32I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatvecI32I32I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatvecI32I32I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatvecI32I32I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecI32I32I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecI32I32I32Op::region() {
  return (*this)->getRegion(0);
}

void MatvecI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatvecI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatvecI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatvecI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatvecI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatvecI32I32I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatvecI32I32I32Op>(parser, result/*TODO:, captures*/);
}

void MatvecI32I32I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatvecI32I32I32Op::verify() {
  if (failed(MatvecI32I32I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecI8I8I32Op definitions
//===----------------------------------------------------------------------===//

MatvecI8I8I32OpAdaptor::MatvecI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatvecI8I8I32OpAdaptor::MatvecI8I8I32OpAdaptor(MatvecI8I8I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatvecI8I8I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecI8I8I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatvecI8I8I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecI8I8I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecI8I8I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatvecI8I8I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatvecI8I8I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatvecI8I8I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatvecI8I8I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatvecI8I8I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatvecI8I8I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecI8I8I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecI8I8I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecI8I8I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatvecI8I8I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatvecI8I8I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatvecI8I8I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecI8I8I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecI8I8I32Op::region() {
  return (*this)->getRegion(0);
}

void MatvecI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatvecI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatvecI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<MatvecI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void MatvecI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatvecI8I8I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatvecI8I8I32Op>(parser, result/*TODO:, captures*/);
}

void MatvecI8I8I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatvecI8I8I32Op::verify() {
  if (failed(MatvecI8I8I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp definitions
//===----------------------------------------------------------------------===//

MatvecOpAdaptor::MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatvecOpAdaptor::MatvecOpAdaptor(MatvecOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange MatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatvecOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatvecOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatvecOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatvecOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> MatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range MatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatvecOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatvecOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatvecOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecOp::region() {
  return (*this)->getRegion(0);
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult MatvecOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatvecOp>(parser, result/*TODO:, captures*/);
}

void MatvecOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatvecOp::verify() {
  if (failed(MatvecOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingMaxOpAdaptor::PoolingMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingMaxOpAdaptor::PoolingMaxOpAdaptor(PoolingMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMaxOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMaxOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMaxOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr PoolingMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PoolingMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingMaxOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingMaxOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PoolingMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMaxOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMaxOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMaxOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingMaxOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMaxOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMaxOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PoolingMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingMaxOp::stridesAttr() {
  return (*this)->getAttr("strides").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMaxOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingMaxOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingMaxOp::paddingAttr() {
  return (*this)->getAttr("padding").template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingMaxOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingMaxOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("strides", attr);
}

void PoolingMaxOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void PoolingMaxOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("padding", attr);
}

::mlir::Attribute PoolingMaxOp::removeStridesAttr() {
  return (*this)->removeAttr("strides");
}

::mlir::Attribute PoolingMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr("dilations");
}

::mlir::Attribute PoolingMaxOp::removePaddingAttr() {
  return (*this)->removeAttr("padding");
}

void PoolingMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void PoolingMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingMaxOp::verify() {
  if (failed(PoolingMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}







::mlir::ParseResult PoolingMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingMaxOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.pooling_max";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMinOp definitions
//===----------------------------------------------------------------------===//

PoolingMinOpAdaptor::PoolingMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingMinOpAdaptor::PoolingMinOpAdaptor(PoolingMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMinOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMinOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMinOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr PoolingMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PoolingMinOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingMinOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingMinOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingMinOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PoolingMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMinOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMinOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMinOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingMinOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMinOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMinOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PoolingMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingMinOp::stridesAttr() {
  return (*this)->getAttr("strides").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMinOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingMinOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMinOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingMinOp::paddingAttr() {
  return (*this)->getAttr("padding").template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingMinOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingMinOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("strides", attr);
}

void PoolingMinOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void PoolingMinOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("padding", attr);
}

::mlir::Attribute PoolingMinOp::removeStridesAttr() {
  return (*this)->removeAttr("strides");
}

::mlir::Attribute PoolingMinOp::removeDilationsAttr() {
  return (*this)->removeAttr("dilations");
}

::mlir::Attribute PoolingMinOp::removePaddingAttr() {
  return (*this)->removeAttr("padding");
}

void PoolingMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void PoolingMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingMinOp::verify() {
  if (failed(PoolingMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}







::mlir::ParseResult PoolingMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingMinOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.pooling_min";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNHWCMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNHWCMaxOpAdaptor::PoolingNHWCMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNHWCMaxOpAdaptor::PoolingNHWCMaxOpAdaptor(PoolingNHWCMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNHWCMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNHWCMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange PoolingNHWCMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNHWCMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNHWCMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNHWCMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNHWCMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNHWCMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNHWCMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNHWCMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNHWCMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nhwc_max' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nhwc_max' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PoolingNHWCMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range PoolingNHWCMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNHWCMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNHWCMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNHWCMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange PoolingNHWCMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> PoolingNHWCMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNHWCMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNHWCMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNHWCMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNHWCMaxOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNHWCMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNHWCMaxOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNHWCMaxOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void PoolingNHWCMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void PoolingNHWCMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void PoolingNHWCMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<PoolingNHWCMaxOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void PoolingNHWCMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<PoolingNHWCMaxOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void PoolingNHWCMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void PoolingNHWCMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNHWCMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult PoolingNHWCMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNHWCMaxOp>(parser, result/*TODO:, captures*/);
}

void PoolingNHWCMaxOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNHWCMaxOp::verify() {
  if (failed(PoolingNHWCMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNHWCMinOp definitions
//===----------------------------------------------------------------------===//

PoolingNHWCMinOpAdaptor::PoolingNHWCMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNHWCMinOpAdaptor::PoolingNHWCMinOpAdaptor(PoolingNHWCMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNHWCMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNHWCMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange PoolingNHWCMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNHWCMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNHWCMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNHWCMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNHWCMinOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNHWCMinOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNHWCMinOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNHWCMinOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNHWCMinOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nhwc_min' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nhwc_min' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PoolingNHWCMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range PoolingNHWCMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNHWCMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNHWCMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNHWCMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange PoolingNHWCMinOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> PoolingNHWCMinOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNHWCMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNHWCMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNHWCMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNHWCMinOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNHWCMinOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNHWCMinOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNHWCMinOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void PoolingNHWCMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void PoolingNHWCMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void PoolingNHWCMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<PoolingNHWCMinOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void PoolingNHWCMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<PoolingNHWCMinOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void PoolingNHWCMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void PoolingNHWCMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNHWCMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult PoolingNHWCMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNHWCMinOp>(parser, result/*TODO:, captures*/);
}

void PoolingNHWCMinOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNHWCMinOp::verify() {
  if (failed(PoolingNHWCMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNHWCSumOp definitions
//===----------------------------------------------------------------------===//

PoolingNHWCSumOpAdaptor::PoolingNHWCSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNHWCSumOpAdaptor::PoolingNHWCSumOpAdaptor(PoolingNHWCSumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNHWCSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNHWCSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange PoolingNHWCSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNHWCSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNHWCSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNHWCSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNHWCSumOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNHWCSumOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNHWCSumOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNHWCSumOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNHWCSumOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PoolingNHWCSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range PoolingNHWCSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNHWCSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNHWCSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNHWCSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange PoolingNHWCSumOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> PoolingNHWCSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNHWCSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNHWCSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNHWCSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNHWCSumOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNHWCSumOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNHWCSumOp::stridesAttr() {
  return (*this)->getAttr("strides").template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNHWCSumOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void PoolingNHWCSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void PoolingNHWCSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("strides", attr);
}

void PoolingNHWCSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<PoolingNHWCSumOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void PoolingNHWCSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<PoolingNHWCSumOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void PoolingNHWCSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

void PoolingNHWCSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute dilations, Attribute strides) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNHWCSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        odsState.addAttribute("dilations", dilations);
odsState.addAttribute("strides", strides);
      
}

::mlir::ParseResult PoolingNHWCSumOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNHWCSumOp>(parser, result/*TODO:, captures*/);
}

void PoolingNHWCSumOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNHWCSumOp::verify() {
  if (failed(PoolingNHWCSumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingSumOp definitions
//===----------------------------------------------------------------------===//

PoolingSumOpAdaptor::PoolingSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingSumOpAdaptor::PoolingSumOpAdaptor(PoolingSumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingSumOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingSumOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingSumOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr PoolingSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PoolingSumOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingSumOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingSumOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingSumOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PoolingSumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingSumOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingSumOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingSumOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingSumOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingSumOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingSumOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PoolingSumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingSumOp::stridesAttr() {
  return (*this)->getAttr("strides").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingSumOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingSumOp::dilationsAttr() {
  return (*this)->getAttr("dilations").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingSumOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingSumOp::paddingAttr() {
  return (*this)->getAttr("padding").template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingSumOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingSumOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("strides", attr);
}

void PoolingSumOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilations", attr);
}

void PoolingSumOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr("padding", attr);
}

::mlir::Attribute PoolingSumOp::removeStridesAttr() {
  return (*this)->removeAttr("strides");
}

::mlir::Attribute PoolingSumOp::removeDilationsAttr() {
  return (*this)->removeAttr("dilations");
}

::mlir::Attribute PoolingSumOp::removePaddingAttr() {
  return (*this)->removeAttr("padding");
}

void PoolingSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void PoolingSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingSumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingSumOp::verify() {
  if (failed(PoolingSumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}







::mlir::ParseResult PoolingSumOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingSumOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.pooling_sum";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatI16I16I32Op definitions
//===----------------------------------------------------------------------===//

VecmatI16I16I32OpAdaptor::VecmatI16I16I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VecmatI16I16I32OpAdaptor::VecmatI16I16I32OpAdaptor(VecmatI16I16I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VecmatI16I16I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatI16I16I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange VecmatI16I16I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatI16I16I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatI16I16I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr VecmatI16I16I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange VecmatI16I16I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &VecmatI16I16I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult VecmatI16I16I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> VecmatI16I16I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range VecmatI16I16I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatI16I16I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatI16I16I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatI16I16I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange VecmatI16I16I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> VecmatI16I16I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range VecmatI16I16I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatI16I16I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatI16I16I32Op::region() {
  return (*this)->getRegion(0);
}

void VecmatI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<VecmatI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void VecmatI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<VecmatI16I16I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void VecmatI16I16I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult VecmatI16I16I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<VecmatI16I16I32Op>(parser, result/*TODO:, captures*/);
}

void VecmatI16I16I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult VecmatI16I16I32Op::verify() {
  if (failed(VecmatI16I16I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatI32I32I32Op definitions
//===----------------------------------------------------------------------===//

VecmatI32I32I32OpAdaptor::VecmatI32I32I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VecmatI32I32I32OpAdaptor::VecmatI32I32I32OpAdaptor(VecmatI32I32I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VecmatI32I32I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatI32I32I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange VecmatI32I32I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatI32I32I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatI32I32I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr VecmatI32I32I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange VecmatI32I32I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &VecmatI32I32I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult VecmatI32I32I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> VecmatI32I32I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range VecmatI32I32I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatI32I32I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatI32I32I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatI32I32I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange VecmatI32I32I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> VecmatI32I32I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range VecmatI32I32I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatI32I32I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatI32I32I32Op::region() {
  return (*this)->getRegion(0);
}

void VecmatI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<VecmatI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void VecmatI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<VecmatI32I32I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void VecmatI32I32I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult VecmatI32I32I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<VecmatI32I32I32Op>(parser, result/*TODO:, captures*/);
}

void VecmatI32I32I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult VecmatI32I32I32Op::verify() {
  if (failed(VecmatI32I32I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatI8I8I32Op definitions
//===----------------------------------------------------------------------===//

VecmatI8I8I32OpAdaptor::VecmatI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VecmatI8I8I32OpAdaptor::VecmatI8I8I32OpAdaptor(VecmatI8I8I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VecmatI8I8I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatI8I8I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange VecmatI8I8I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatI8I8I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatI8I8I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr VecmatI8I8I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange VecmatI8I8I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &VecmatI8I8I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult VecmatI8I8I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> VecmatI8I8I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range VecmatI8I8I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatI8I8I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatI8I8I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatI8I8I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange VecmatI8I8I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> VecmatI8I8I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range VecmatI8I8I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatI8I8I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatI8I8I32Op::region() {
  return (*this)->getRegion(0);
}

void VecmatI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<VecmatI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void VecmatI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<VecmatI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        
}

void VecmatI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult VecmatI8I8I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<VecmatI8I8I32Op>(parser, result/*TODO:, captures*/);
}

void VecmatI8I8I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult VecmatI8I8I32Op::verify() {
  if (failed(VecmatI8I8I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp definitions
//===----------------------------------------------------------------------===//

VecmatOpAdaptor::VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VecmatOpAdaptor::VecmatOpAdaptor(VecmatOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VecmatOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange VecmatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr VecmatOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange VecmatOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &VecmatOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult VecmatOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> VecmatOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range VecmatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange VecmatOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> VecmatOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range VecmatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatOp::region() {
  return (*this)->getRegion(0);
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult VecmatOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<VecmatOp>(parser, result/*TODO:, captures*/);
}

void VecmatOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult VecmatOp::verify() {
  if (failed(VecmatOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace linalg
} // namespace mlir

#endif  // GET_OP_CLASSES

