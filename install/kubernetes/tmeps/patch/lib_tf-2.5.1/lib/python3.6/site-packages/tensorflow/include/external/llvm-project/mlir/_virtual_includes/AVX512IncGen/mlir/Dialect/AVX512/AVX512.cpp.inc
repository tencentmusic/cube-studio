/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::avx512::MaskCompressIntrOp,
::mlir::avx512::MaskCompressOp,
::mlir::avx512::MaskRndScaleOp,
::mlir::avx512::MaskRndScalePDIntrOp,
::mlir::avx512::MaskRndScalePSIntrOp,
::mlir::avx512::MaskScaleFOp,
::mlir::avx512::MaskScaleFPDIntrOp,
::mlir::avx512::MaskScaleFPSIntrOp,
::mlir::avx512::Vp2IntersectDIntrOp,
::mlir::avx512::Vp2IntersectOp,
::mlir::avx512::Vp2IntersectQIntrOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace avx512 {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5120(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 32-bit float or 32-bit signless integer or 64-bit float or 64-bit signless integer values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5121(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1)))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 1-bit signless integer values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5122(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5123(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 32-bit float or 32-bit signless integer or 64-bit float or 64-bit signless integer values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5124(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5125(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5126(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isSignlessInteger(16))) || ((type.isSignlessInteger(8))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 16-bit signless integer or 8-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5127(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isF64()))) && (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 64-bit float values of length 8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5128(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(8)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 8-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX5129(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isF32()))) && (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 16))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 32-bit float values of length 16, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX51210(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(16)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 16-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX51211(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32)))) && (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 16))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 32-bit signless integer values of length 16, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX51212(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 32-bit signless integer or 64-bit signless integer values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AVX51213(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) && (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be vector of 64-bit signless integer values of length 8, but got " << type;
  }
  return ::mlir::success();
}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskCompressIntrOp definitions
//===----------------------------------------------------------------------===//

MaskCompressIntrOpAdaptor::MaskCompressIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskCompressIntrOpAdaptor::MaskCompressIntrOpAdaptor(MaskCompressIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskCompressIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskCompressIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskCompressIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskCompressIntrOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskCompressIntrOpAdaptor::src() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskCompressIntrOpAdaptor::k() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MaskCompressIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskCompressIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskCompressIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskCompressIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskCompressIntrOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskCompressIntrOp::src() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskCompressIntrOp::k() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MaskCompressIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskCompressIntrOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskCompressIntrOp::kMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskCompressIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskCompressIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskCompressIntrOp::res() {
  return *getODSResults(0).begin();
}

void MaskCompressIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value src, ::mlir::Value k) {
  odsState.addOperands(a);
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addTypes(res);
}

void MaskCompressIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value src, ::mlir::Value k) {
  odsState.addOperands(a);
  odsState.addOperands(src);
  odsState.addOperands(k);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskCompressIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskCompressIntrOp::verify() {
  if (failed(MaskCompressIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5120(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5120(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5121(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {a, src, res} have same type");
  if (!((std::equal_to<>()(VectorType::get({(*this->getODSResults(0).begin()).getType().cast<VectorType>().getShape()[0]}, IntegerType::get((*this->getODSResults(0).begin()).getType().getContext(), 1)), (*this->getODSOperands(2).begin()).getType()))))
    return emitOpError("failed to verify that `k` has the same number of bits as elements in `res`");
  return ::mlir::success();
}

void MaskCompressIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskCompressOp definitions
//===----------------------------------------------------------------------===//

MaskCompressOpAdaptor::MaskCompressOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskCompressOpAdaptor::MaskCompressOpAdaptor(MaskCompressOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskCompressOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskCompressOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange MaskCompressOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskCompressOpAdaptor::k() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskCompressOpAdaptor::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskCompressOpAdaptor::src() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr MaskCompressOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr MaskCompressOpAdaptor::constant_src() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("constant_src").dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::mlir::LogicalResult MaskCompressOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_constant_src = odsAttrs.get("constant_src");
  if (tblgen_constant_src) {
    if (!((tblgen_constant_src.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'avx512.mask.compress' op ""attribute 'constant_src' failed to satisfy constraint: constant vector/tensor attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskCompressOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range MaskCompressOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskCompressOp::k() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskCompressOp::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskCompressOp::src() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange MaskCompressOp::kMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskCompressOp::aMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskCompressOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskCompressOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskCompressOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskCompressOp::dst() {
  return *getODSResults(0).begin();
}

::mlir::ElementsAttr MaskCompressOp::constant_srcAttr() {
  return (*this)->getAttr("constant_src").template dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > MaskCompressOp::constant_src() {
  auto attr = constant_srcAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

void MaskCompressOp::constant_srcAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("constant_src", attr);
}

::mlir::Attribute MaskCompressOp::removeConstant_srcAttr() {
  return (*this)->removeAttr("constant_src");
}

void MaskCompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value k, ::mlir::Value a, /*optional*/::mlir::Value src, /*optional*/::mlir::ElementsAttr constant_src) {
  odsState.addOperands(k);
  odsState.addOperands(a);
  if (src)
    odsState.addOperands(src);
  if (constant_src) {
  odsState.addAttribute("constant_src", constant_src);
  }
  odsState.addTypes(dst);
}

void MaskCompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value k, ::mlir::Value a, /*optional*/::mlir::Value src, /*optional*/::mlir::ElementsAttr constant_src) {
  odsState.addOperands(k);
  odsState.addOperands(a);
  if (src)
    odsState.addOperands(src);
  if (constant_src) {
  odsState.addAttribute("constant_src", constant_src);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskCompressOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskCompressOp::verify() {
  if (failed(MaskCompressOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5121(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5120(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5123(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5120(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {a, dst} have same type");
  if (!((std::equal_to<>()(VectorType::get({(*this->getODSResults(0).begin()).getType().cast<VectorType>().getShape()[0]}, IntegerType::get((*this->getODSResults(0).begin()).getType().getContext(), 1)), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `k` has the same number of bits as elements in `dst`");
  return ::verify(*this);
}

::mlir::ParseResult MaskCompressOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType kRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> kOperands(kRawOperands);  ::llvm::SMLoc kOperandsLoc;
  (void)kOperandsLoc;
  ::mlir::OpAsmParser::OperandType aRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aOperands(aRawOperands);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> srcOperands;
  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> srcTypes;

  kOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(kRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalComma())) {

  {
    srcOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      srcOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalComma())) {

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      srcTypes.push_back(optionalType);
    }
  }
  }
  for (::mlir::Type type : dstTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
      return parser.emitError(parser.getNameLoc()) << "'dst' must be vector of 32-bit float or 32-bit signless integer or 64-bit float or 64-bit signless integer values of length 16/8, but got " << type;
    }
  }
  result.addTypes(dstTypes);
  if (parser.resolveOperands(kOperands, VectorType::get({dstTypes[0].cast<VectorType>().getShape()[0]}, IntegerType::get(dstTypes[0].getContext(), 1)), kOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskCompressOp::print(::mlir::OpAsmPrinter &p) {
  p << "avx512.mask.compress";
  p << ' ';
  p << k();
  p << ",";
  p << ' ';
  p << a();
  if (src()) {
  p << ",";
  p << ' ';
  if (::mlir::Value value = src())
    p << value;
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
  if (src()) {
  p << ",";
  p << ' ';
  p << (src() ? ::llvm::ArrayRef<::mlir::Type>(src().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
}

void MaskCompressOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskRndScaleOp definitions
//===----------------------------------------------------------------------===//

MaskRndScaleOpAdaptor::MaskRndScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskRndScaleOpAdaptor::MaskRndScaleOpAdaptor(MaskRndScaleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskRndScaleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskRndScaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskRndScaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScaleOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr MaskRndScaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskRndScaleOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskRndScaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskRndScaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScaleOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScaleOp::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScaleOp::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScaleOp::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScaleOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskRndScaleOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::kMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::aMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::immMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskRndScaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskRndScaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScaleOp::dst() {
  return *getODSResults(0).begin();
}

void MaskRndScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  odsState.addTypes(dst);
}

void MaskRndScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskRndScaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskRndScaleOp::verify() {
  if (failed(MaskRndScaleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5125(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5126(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5125(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, dst} have same type");
  if (!((std::equal_to<>()(IntegerType::get((*this->getODSResults(0).begin()).getType().getContext(), ((*this->getODSResults(0).begin()).getType().cast<VectorType>().getShape()[0])), (*this->getODSOperands(3).begin()).getType()))))
    return emitOpError("failed to verify that imm has the same number of bits as elements in dst");
  return ::mlir::success();
}

::mlir::ParseResult MaskRndScaleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::OperandType kRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> kOperands(kRawOperands);  ::llvm::SMLoc kOperandsLoc;
  (void)kOperandsLoc;
  ::mlir::OpAsmParser::OperandType aRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aOperands(aRawOperands);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::OperandType immRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> immOperands(immRawOperands);  ::llvm::SMLoc immOperandsLoc;
  (void)immOperandsLoc;
  ::mlir::OpAsmParser::OperandType roundingRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> roundingOperands(roundingRawOperands);  ::llvm::SMLoc roundingOperandsLoc;
  (void)roundingOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  kOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(kRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  immOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(immRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  roundingOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(roundingRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : dstTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
      return parser.emitError(parser.getNameLoc()) << "'dst' must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(dstTypes);
  if (parser.resolveOperands(srcOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(kOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(immOperands, IntegerType::get(dstTypes[0].getContext(), (dstTypes[0].cast<VectorType>().getShape()[0])), immOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(roundingOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskRndScaleOp::print(::mlir::OpAsmPrinter &p) {
  p << "avx512.mask.rndscale";
  p << ' ';
  p << src();
  p << ",";
  p << ' ';
  p << k();
  p << ",";
  p << ' ';
  p << a();
  p << ",";
  p << ' ';
  p << imm();
  p << ",";
  p << ' ';
  p << rounding();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void MaskRndScaleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskRndScalePDIntrOp definitions
//===----------------------------------------------------------------------===//

MaskRndScalePDIntrOpAdaptor::MaskRndScalePDIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskRndScalePDIntrOpAdaptor::MaskRndScalePDIntrOpAdaptor(MaskRndScalePDIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskRndScalePDIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskRndScalePDIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskRndScalePDIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScalePDIntrOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScalePDIntrOpAdaptor::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScalePDIntrOpAdaptor::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScalePDIntrOpAdaptor::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScalePDIntrOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr MaskRndScalePDIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskRndScalePDIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskRndScalePDIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskRndScalePDIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScalePDIntrOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScalePDIntrOp::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScalePDIntrOp::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScalePDIntrOp::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScalePDIntrOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskRndScalePDIntrOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePDIntrOp::kMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePDIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePDIntrOp::immMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePDIntrOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskRndScalePDIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskRndScalePDIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScalePDIntrOp::res() {
  return *getODSResults(0).begin();
}

void MaskRndScalePDIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  odsState.addTypes(res);
}

void MaskRndScalePDIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskRndScalePDIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskRndScalePDIntrOp::verify() {
  if (failed(MaskRndScalePDIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5127(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5125(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5127(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5128(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, res} have same type");
  return ::mlir::success();
}

void MaskRndScalePDIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskRndScalePSIntrOp definitions
//===----------------------------------------------------------------------===//

MaskRndScalePSIntrOpAdaptor::MaskRndScalePSIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskRndScalePSIntrOpAdaptor::MaskRndScalePSIntrOpAdaptor(MaskRndScalePSIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskRndScalePSIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskRndScalePSIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskRndScalePSIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScalePSIntrOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScalePSIntrOpAdaptor::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScalePSIntrOpAdaptor::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScalePSIntrOpAdaptor::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScalePSIntrOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr MaskRndScalePSIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskRndScalePSIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskRndScalePSIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskRndScalePSIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScalePSIntrOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScalePSIntrOp::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScalePSIntrOp::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScalePSIntrOp::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScalePSIntrOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskRndScalePSIntrOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePSIntrOp::kMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePSIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePSIntrOp::immMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScalePSIntrOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskRndScalePSIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskRndScalePSIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScalePSIntrOp::res() {
  return *getODSResults(0).begin();
}

void MaskRndScalePSIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  odsState.addTypes(res);
}

void MaskRndScalePSIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskRndScalePSIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskRndScalePSIntrOp::verify() {
  if (failed(MaskRndScalePSIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5129(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5125(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5129(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51210(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, res} have same type");
  return ::mlir::success();
}

void MaskRndScalePSIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskScaleFOp definitions
//===----------------------------------------------------------------------===//

MaskScaleFOpAdaptor::MaskScaleFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskScaleFOpAdaptor::MaskScaleFOpAdaptor(MaskScaleFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskScaleFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskScaleFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskScaleFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFOpAdaptor::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFOpAdaptor::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFOpAdaptor::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr MaskScaleFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskScaleFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskScaleFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskScaleFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFOp::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFOp::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFOp::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskScaleFOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::aMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::bMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::kMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskScaleFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskScaleFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFOp::dst() {
  return *getODSResults(0).begin();
}

void MaskScaleFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  odsState.addTypes(dst);
}

void MaskScaleFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskScaleFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskScaleFOp::verify() {
  if (failed(MaskScaleFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5126(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5125(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5124(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, b, dst} have same type");
  if (!((std::equal_to<>()(IntegerType::get((*this->getODSResults(0).begin()).getType().getContext(), ((*this->getODSResults(0).begin()).getType().cast<VectorType>().getShape()[0])), (*this->getODSOperands(3).begin()).getType()))))
    return emitOpError("failed to verify that k has the same number of bits as elements in dst");
  return ::mlir::success();
}

::mlir::ParseResult MaskScaleFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::OperandType aRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aOperands(aRawOperands);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::OperandType bRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> bOperands(bRawOperands);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::OperandType kRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> kOperands(kRawOperands);  ::llvm::SMLoc kOperandsLoc;
  (void)kOperandsLoc;
  ::mlir::OpAsmParser::OperandType roundingRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> roundingOperands(roundingRawOperands);  ::llvm::SMLoc roundingOperandsLoc;
  (void)roundingOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  kOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(kRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  roundingOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(roundingRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : dstTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
      return parser.emitError(parser.getNameLoc()) << "'dst' must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(dstTypes);
  if (parser.resolveOperands(srcOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(kOperands, IntegerType::get(dstTypes[0].getContext(), (dstTypes[0].cast<VectorType>().getShape()[0])), kOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(roundingOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskScaleFOp::print(::mlir::OpAsmPrinter &p) {
  p << "avx512.mask.scalef";
  p << ' ';
  p << src();
  p << ",";
  p << ' ';
  p << a();
  p << ",";
  p << ' ';
  p << b();
  p << ",";
  p << ' ';
  p << k();
  p << ",";
  p << ' ';
  p << rounding();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void MaskScaleFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskScaleFPDIntrOp definitions
//===----------------------------------------------------------------------===//

MaskScaleFPDIntrOpAdaptor::MaskScaleFPDIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskScaleFPDIntrOpAdaptor::MaskScaleFPDIntrOpAdaptor(MaskScaleFPDIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskScaleFPDIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskScaleFPDIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskScaleFPDIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFPDIntrOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFPDIntrOpAdaptor::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFPDIntrOpAdaptor::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFPDIntrOpAdaptor::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFPDIntrOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr MaskScaleFPDIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskScaleFPDIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskScaleFPDIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskScaleFPDIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFPDIntrOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFPDIntrOp::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFPDIntrOp::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFPDIntrOp::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFPDIntrOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskScaleFPDIntrOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPDIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPDIntrOp::bMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPDIntrOp::kMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPDIntrOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskScaleFPDIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskScaleFPDIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFPDIntrOp::res() {
  return *getODSResults(0).begin();
}

void MaskScaleFPDIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  odsState.addTypes(res);
}

void MaskScaleFPDIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskScaleFPDIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskScaleFPDIntrOp::verify() {
  if (failed(MaskScaleFPDIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5127(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5127(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5127(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5128(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, b, res} have same type");
  return ::mlir::success();
}

void MaskScaleFPDIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskScaleFPSIntrOp definitions
//===----------------------------------------------------------------------===//

MaskScaleFPSIntrOpAdaptor::MaskScaleFPSIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskScaleFPSIntrOpAdaptor::MaskScaleFPSIntrOpAdaptor(MaskScaleFPSIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskScaleFPSIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskScaleFPSIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskScaleFPSIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFPSIntrOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFPSIntrOpAdaptor::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFPSIntrOpAdaptor::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFPSIntrOpAdaptor::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFPSIntrOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr MaskScaleFPSIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskScaleFPSIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskScaleFPSIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskScaleFPSIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFPSIntrOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFPSIntrOp::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFPSIntrOp::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFPSIntrOp::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFPSIntrOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskScaleFPSIntrOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPSIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPSIntrOp::bMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPSIntrOp::kMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFPSIntrOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskScaleFPSIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskScaleFPSIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFPSIntrOp::res() {
  return *getODSResults(0).begin();
}

void MaskScaleFPSIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  odsState.addTypes(res);
}

void MaskScaleFPSIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskScaleFPSIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskScaleFPSIntrOp::verify() {
  if (failed(MaskScaleFPSIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5129(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5129(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5129(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51210(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, b, res} have same type");
  return ::mlir::success();
}

void MaskScaleFPSIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::Vp2IntersectDIntrOp definitions
//===----------------------------------------------------------------------===//

Vp2IntersectDIntrOpAdaptor::Vp2IntersectDIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Vp2IntersectDIntrOpAdaptor::Vp2IntersectDIntrOpAdaptor(Vp2IntersectDIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Vp2IntersectDIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Vp2IntersectDIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Vp2IntersectDIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectDIntrOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value Vp2IntersectDIntrOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr Vp2IntersectDIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Vp2IntersectDIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Vp2IntersectDIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Vp2IntersectDIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectDIntrOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value Vp2IntersectDIntrOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange Vp2IntersectDIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Vp2IntersectDIntrOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Vp2IntersectDIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Vp2IntersectDIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectDIntrOp::res() {
  return *getODSResults(0).begin();
}

void Vp2IntersectDIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void Vp2IntersectDIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Vp2IntersectDIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Vp2IntersectDIntrOp::verify() {
  if (failed(Vp2IntersectDIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51211(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51211(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void Vp2IntersectDIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::Vp2IntersectOp definitions
//===----------------------------------------------------------------------===//

Vp2IntersectOpAdaptor::Vp2IntersectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Vp2IntersectOpAdaptor::Vp2IntersectOpAdaptor(Vp2IntersectOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Vp2IntersectOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Vp2IntersectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Vp2IntersectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value Vp2IntersectOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr Vp2IntersectOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Vp2IntersectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Vp2IntersectOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "k1");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "k2");
}



std::pair<unsigned, unsigned> Vp2IntersectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Vp2IntersectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value Vp2IntersectOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange Vp2IntersectOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Vp2IntersectOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Vp2IntersectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Vp2IntersectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectOp::k1() {
  return *getODSResults(0).begin();
}

::mlir::Value Vp2IntersectOp::k2() {
  return *getODSResults(1).begin();
}

void Vp2IntersectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type k1, ::mlir::Type k2, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(k1);
  odsState.addTypes(k2);
}

void Vp2IntersectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Vp2IntersectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Vp2IntersectOp::verify() {
  if (failed(Vp2IntersectOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51212(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51212(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5121(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5121(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType()})))))
    return emitOpError("failed to verify that all of {a, b} have same type");
  if (!((std::equal_to<>()(VectorType::get({(*this->getODSOperands(0).begin()).getType().cast<VectorType>().getShape()[0]}, IntegerType::get((*this->getODSOperands(0).begin()).getType().getContext(), 1)), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that k1 has the same number of bits as elements in a");
  if (!((std::equal_to<>()(VectorType::get({(*this->getODSOperands(0).begin()).getType().cast<VectorType>().getShape()[0]}, IntegerType::get((*this->getODSOperands(0).begin()).getType().getContext(), 1)), (*this->getODSResults(1).begin()).getType()))))
    return emitOpError("failed to verify that k2 has the same number of bits as elements in b");
  return ::mlir::success();
}

::mlir::ParseResult Vp2IntersectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType aRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aOperands(aRawOperands);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::OperandType bRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> bOperands(bRawOperands);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::Type aRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> aTypes(aRawTypes);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(aRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : aTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
      return parser.emitError(parser.getNameLoc()) << "'a' must be vector of 32-bit signless integer or 64-bit signless integer values of length 16/8, but got " << type;
    }
  }
  result.addTypes(VectorType::get({aTypes[0].cast<VectorType>().getShape()[0]}, IntegerType::get(aTypes[0].getContext(), 1)));
  result.addTypes(VectorType::get({aTypes[0].cast<VectorType>().getShape()[0]}, IntegerType::get(aTypes[0].getContext(), 1)));
  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, aTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void Vp2IntersectOp::print(::mlir::OpAsmPrinter &p) {
  p << "avx512.vp2intersect";
  p << ' ';
  p << a();
  p << ",";
  p << ' ';
  p << b();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(a().getType());
}

void Vp2IntersectOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::Vp2IntersectQIntrOp definitions
//===----------------------------------------------------------------------===//

Vp2IntersectQIntrOpAdaptor::Vp2IntersectQIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Vp2IntersectQIntrOpAdaptor::Vp2IntersectQIntrOpAdaptor(Vp2IntersectQIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Vp2IntersectQIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Vp2IntersectQIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Vp2IntersectQIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectQIntrOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value Vp2IntersectQIntrOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr Vp2IntersectQIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Vp2IntersectQIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Vp2IntersectQIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Vp2IntersectQIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectQIntrOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value Vp2IntersectQIntrOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange Vp2IntersectQIntrOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Vp2IntersectQIntrOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Vp2IntersectQIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Vp2IntersectQIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Vp2IntersectQIntrOp::res() {
  return *getODSResults(0).begin();
}

void Vp2IntersectQIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void Vp2IntersectQIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Vp2IntersectQIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Vp2IntersectQIntrOp::verify() {
  if (failed(Vp2IntersectQIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51213(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX51213(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AVX5122(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void Vp2IntersectQIntrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir

#endif  // GET_OP_CLASSES

