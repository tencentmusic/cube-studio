/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::acc::DataOp,
::mlir::acc::EnterDataOp,
::mlir::acc::ExitDataOp,
::mlir::acc::InitOp,
::mlir::acc::LoopOp,
::mlir::acc::ParallelOp,
::mlir::acc::ShutdownOp,
::mlir::acc::TerminatorOp,
::mlir::acc::UpdateOp,
::mlir::acc::WaitOp,
::mlir::acc::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace acc {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp definitions
//===----------------------------------------------------------------------===//

DataOpAdaptor::DataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DataOpAdaptor::DataOpAdaptor(DataOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange DataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DataOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange DataOpAdaptor::copyOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange DataOpAdaptor::copyinOperands() {
  return getODSOperands(2);
}

::mlir::ValueRange DataOpAdaptor::copyinReadonlyOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange DataOpAdaptor::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange DataOpAdaptor::copyoutZeroOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange DataOpAdaptor::createOperands() {
  return getODSOperands(6);
}

::mlir::ValueRange DataOpAdaptor::createZeroOperands() {
  return getODSOperands(7);
}

::mlir::ValueRange DataOpAdaptor::noCreateOperands() {
  return getODSOperands(8);
}

::mlir::ValueRange DataOpAdaptor::presentOperands() {
  return getODSOperands(9);
}

::mlir::ValueRange DataOpAdaptor::deviceptrOperands() {
  return getODSOperands(10);
}

::mlir::ValueRange DataOpAdaptor::attachOperands() {
  return getODSOperands(11);
}

::mlir::DictionaryAttr DataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr DataOpAdaptor::defaultAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("defaultAttr").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange DataOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DataOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DataOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 12)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 12 elements, but got ") << numElements;
  }
    {
  auto tblgen_defaultAttr = odsAttrs.get("defaultAttr");
  if (tblgen_defaultAttr) {
    if (!(((tblgen_defaultAttr.isa<::mlir::StringAttr>())) && (((tblgen_defaultAttr.cast<::mlir::StringAttr>().getValue() == "present")) || ((tblgen_defaultAttr.cast<::mlir::StringAttr>().getValue() == "none"))))) return emitError(loc, "'acc.data' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> DataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range DataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DataOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range DataOp::copyOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DataOp::copyinOperands() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range DataOp::copyinReadonlyOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range DataOp::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range DataOp::copyoutZeroOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range DataOp::createOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range DataOp::createZeroOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range DataOp::noCreateOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range DataOp::presentOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range DataOp::deviceptrOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range DataOp::attachOperands() {
  return getODSOperands(11);
}

::mlir::MutableOperandRange DataOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::copyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::copyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::copyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::copyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::copyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::createOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::createZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::noCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::presentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::deviceptrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DataOp::attachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &DataOp::region() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr DataOp::defaultAttrAttr() {
  return (*this)->getAttr("defaultAttr").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > DataOp::defaultAttr() {
  auto attr = defaultAttrAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void DataOp::defaultAttrAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("defaultAttr", attr);
}

::mlir::Attribute DataOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr("defaultAttr");
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::StringAttr defaultAttr) {
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(deviceptrOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(deviceptrOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (defaultAttr) {
  odsState.addAttribute("defaultAttr", defaultAttr);
  }
  (void)odsState.addRegion();
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::StringAttr defaultAttr) {
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(deviceptrOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(deviceptrOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (defaultAttr) {
  odsState.addAttribute("defaultAttr", defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DataOp::verify() {
  if (failed(DataOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup8 = getODSOperands(8);
    for (::mlir::Value v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup9 = getODSOperands(9);
    for (::mlir::Value v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup10 = getODSOperands(10);
    for (::mlir::Value v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup11 = getODSOperands(11);
    for (::mlir::Value v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
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

::mlir::ParseResult DataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyOperandsOperands;
  ::llvm::SMLoc copyOperandsOperandsLoc;
  (void)copyOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyinReadonlyOperandsOperands;
  ::llvm::SMLoc copyinReadonlyOperandsOperandsLoc;
  (void)copyinReadonlyOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyinReadonlyOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyoutZeroOperandsOperands;
  ::llvm::SMLoc copyoutZeroOperandsOperandsLoc;
  (void)copyoutZeroOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyoutZeroOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> noCreateOperandsOperands;
  ::llvm::SMLoc noCreateOperandsOperandsLoc;
  (void)noCreateOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> noCreateOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> presentOperandsOperands;
  ::llvm::SMLoc presentOperandsOperandsLoc;
  (void)presentOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> presentOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceptrOperandsOperands;
  ::llvm::SMLoc deviceptrOperandsOperandsLoc;
  (void)deviceptrOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceptrOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copy"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyin"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyin_readonly"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinReadonlyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinReadonlyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinReadonlyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyout"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyout_zero"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create_zero"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("no_create"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  noCreateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(noCreateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(noCreateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("present"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  presentOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(presentOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(presentOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("deviceptr"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceptrOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceptrOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceptrOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attach"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyOperandsOperands, copyOperandsTypes, copyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinReadonlyOperandsOperands, copyinReadonlyOperandsTypes, copyinReadonlyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutZeroOperandsOperands, copyoutZeroOperandsTypes, copyoutZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(noCreateOperandsOperands, noCreateOperandsTypes, noCreateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(presentOperandsOperands, presentOperandsTypes, presentOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceptrOperandsOperands, deviceptrOperandsTypes, deviceptrOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(copyOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(copyinReadonlyOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(copyoutZeroOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(noCreateOperandsOperands.size()), static_cast<int32_t>(presentOperandsOperands.size()), static_cast<int32_t>(deviceptrOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size())}));
  return ::mlir::success();
}

void DataOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.data";
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  if (!copyOperands().empty()) {
  p << ' ' << "copy";
  p << "(";
  p << copyOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyOperands().getTypes();
  p << ")";
  }
  if (!copyinOperands().empty()) {
  p << ' ' << "copyin";
  p << "(";
  p << copyinOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyinOperands().getTypes();
  p << ")";
  }
  if (!copyinReadonlyOperands().empty()) {
  p << ' ' << "copyin_readonly";
  p << "(";
  p << copyinReadonlyOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyinReadonlyOperands().getTypes();
  p << ")";
  }
  if (!copyoutOperands().empty()) {
  p << ' ' << "copyout";
  p << "(";
  p << copyoutOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyoutOperands().getTypes();
  p << ")";
  }
  if (!copyoutZeroOperands().empty()) {
  p << ' ' << "copyout_zero";
  p << "(";
  p << copyoutZeroOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyoutZeroOperands().getTypes();
  p << ")";
  }
  if (!createOperands().empty()) {
  p << ' ' << "create";
  p << "(";
  p << createOperands();
  p << ' ' << ":";
  p << ' ';
  p << createOperands().getTypes();
  p << ")";
  }
  if (!createZeroOperands().empty()) {
  p << ' ' << "create_zero";
  p << "(";
  p << createZeroOperands();
  p << ' ' << ":";
  p << ' ';
  p << createZeroOperands().getTypes();
  p << ")";
  }
  if (!noCreateOperands().empty()) {
  p << ' ' << "no_create";
  p << "(";
  p << noCreateOperands();
  p << ' ' << ":";
  p << ' ';
  p << noCreateOperands().getTypes();
  p << ")";
  }
  if (!presentOperands().empty()) {
  p << ' ' << "present";
  p << "(";
  p << presentOperands();
  p << ' ' << ":";
  p << ' ';
  p << presentOperands().getTypes();
  p << ")";
  }
  if (!deviceptrOperands().empty()) {
  p << ' ' << "deviceptr";
  p << "(";
  p << deviceptrOperands();
  p << ' ' << ":";
  p << ' ';
  p << deviceptrOperands().getTypes();
  p << ")";
  }
  if (!attachOperands().empty()) {
  p << ' ' << "attach";
  p << "(";
  p << attachOperands();
  p << ' ' << ":";
  p << ' ';
  p << attachOperands().getTypes();
  p << ")";
  }
  p << ' ';
  p.printRegion(region());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp definitions
//===----------------------------------------------------------------------===//

EnterDataOpAdaptor::EnterDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

EnterDataOpAdaptor::EnterDataOpAdaptor(EnterDataOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange EnterDataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> EnterDataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange EnterDataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EnterDataOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange EnterDataOpAdaptor::waitOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange EnterDataOpAdaptor::copyinOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange EnterDataOpAdaptor::createOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange EnterDataOpAdaptor::createZeroOperands() {
  return getODSOperands(6);
}

::mlir::ValueRange EnterDataOpAdaptor::attachOperands() {
  return getODSOperands(7);
}

::mlir::DictionaryAttr EnterDataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr EnterDataOpAdaptor::async() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("async").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr EnterDataOpAdaptor::wait() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("wait").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult EnterDataOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 8)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 8 elements, but got ") << numElements;
  }
    {
  auto tblgen_async = odsAttrs.get("async");
  if (tblgen_async) {
    if (!((tblgen_async.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.enter_data' op ""attribute 'async' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_wait = odsAttrs.get("wait");
  if (tblgen_wait) {
    if (!((tblgen_wait.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.enter_data' op ""attribute 'wait' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> EnterDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range EnterDataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EnterDataOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range EnterDataOp::waitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range EnterDataOp::copyinOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range EnterDataOp::createOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range EnterDataOp::createZeroOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range EnterDataOp::attachOperands() {
  return getODSOperands(7);
}

::mlir::MutableOperandRange EnterDataOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::copyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::createOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::createZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange EnterDataOp::attachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> EnterDataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EnterDataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr EnterDataOp::asyncAttr() {
  return (*this)->getAttr("async").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr EnterDataOp::waitAttr() {
  return (*this)->getAttr("wait").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

void EnterDataOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("async", attr);
}

void EnterDataOp::waitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("wait", attr);
}

::mlir::Attribute EnterDataOp::removeAsyncAttr() {
  return (*this)->removeAttr("async");
}

::mlir::Attribute EnterDataOp::removeWaitAttr() {
  return (*this)->removeAttr("wait");
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  if (wait) {
  odsState.addAttribute("wait", wait);
  }
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  if (wait) {
  odsState.addAttribute("wait", wait);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute("wait", odsBuilder.getUnitAttr());
  }
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute("wait", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EnterDataOp::verify() {
  if (failed(EnterDataOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult EnterDataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyin"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create_zero"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attach"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size())}));
  return ::mlir::success();
}

void EnterDataOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.enter_data";
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  if (asyncOperand()) {
  p << ' ' << "async";
  p << "(";
  if (::mlir::Value value = asyncOperand())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (waitDevnum()) {
  p << ' ' << "wait_devnum";
  p << "(";
  if (::mlir::Value value = waitDevnum())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (!waitOperands().empty()) {
  p << ' ' << "wait";
  p << "(";
  p << waitOperands();
  p << ' ' << ":";
  p << ' ';
  p << waitOperands().getTypes();
  p << ")";
  }
  if (!copyinOperands().empty()) {
  p << ' ' << "copyin";
  p << "(";
  p << copyinOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyinOperands().getTypes();
  p << ")";
  }
  if (!createOperands().empty()) {
  p << ' ' << "create";
  p << "(";
  p << createOperands();
  p << ' ' << ":";
  p << ' ';
  p << createOperands().getTypes();
  p << ")";
  }
  if (!createZeroOperands().empty()) {
  p << ' ' << "create_zero";
  p << "(";
  p << createZeroOperands();
  p << ' ' << ":";
  p << ' ';
  p << createZeroOperands().getTypes();
  p << ")";
  }
  if (!attachOperands().empty()) {
  p << ' ' << "attach";
  p << "(";
  p << attachOperands();
  p << ' ' << ":";
  p << ' ';
  p << attachOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp definitions
//===----------------------------------------------------------------------===//

ExitDataOpAdaptor::ExitDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ExitDataOpAdaptor::ExitDataOpAdaptor(ExitDataOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ExitDataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExitDataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ExitDataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExitDataOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ExitDataOpAdaptor::waitOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange ExitDataOpAdaptor::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange ExitDataOpAdaptor::deleteOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange ExitDataOpAdaptor::detachOperands() {
  return getODSOperands(6);
}

::mlir::DictionaryAttr ExitDataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ExitDataOpAdaptor::async() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("async").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr ExitDataOpAdaptor::wait() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("wait").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr ExitDataOpAdaptor::finalize() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("finalize").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult ExitDataOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 7 elements, but got ") << numElements;
  }
    {
  auto tblgen_async = odsAttrs.get("async");
  if (tblgen_async) {
    if (!((tblgen_async.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.exit_data' op ""attribute 'async' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_wait = odsAttrs.get("wait");
  if (tblgen_wait) {
    if (!((tblgen_wait.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.exit_data' op ""attribute 'wait' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_finalize = odsAttrs.get("finalize");
  if (tblgen_finalize) {
    if (!((tblgen_finalize.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.exit_data' op ""attribute 'finalize' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ExitDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ExitDataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExitDataOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ExitDataOp::waitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ExitDataOp::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range ExitDataOp::deleteOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range ExitDataOp::detachOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange ExitDataOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ExitDataOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ExitDataOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ExitDataOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ExitDataOp::copyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ExitDataOp::deleteOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ExitDataOp::detachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ExitDataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExitDataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr ExitDataOp::asyncAttr() {
  return (*this)->getAttr("async").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOp::waitAttr() {
  return (*this)->getAttr("wait").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOp::finalizeAttr() {
  return (*this)->getAttr("finalize").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::finalize() {
  auto attr = finalizeAttr();
  return attr != nullptr;
}

void ExitDataOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("async", attr);
}

void ExitDataOp::waitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("wait", attr);
}

void ExitDataOp::finalizeAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("finalize", attr);
}

::mlir::Attribute ExitDataOp::removeAsyncAttr() {
  return (*this)->removeAttr("async");
}

::mlir::Attribute ExitDataOp::removeWaitAttr() {
  return (*this)->removeAttr("wait");
}

::mlir::Attribute ExitDataOp::removeFinalizeAttr() {
  return (*this)->removeAttr("finalize");
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  if (wait) {
  odsState.addAttribute("wait", wait);
  }
  if (finalize) {
  odsState.addAttribute("finalize", finalize);
  }
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  if (wait) {
  odsState.addAttribute("wait", wait);
  }
  if (finalize) {
  odsState.addAttribute("finalize", finalize);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute("wait", odsBuilder.getUnitAttr());
  }
  if (finalize) {
  odsState.addAttribute("finalize", odsBuilder.getUnitAttr());
  }
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute("wait", odsBuilder.getUnitAttr());
  }
  if (finalize) {
  odsState.addAttribute("finalize", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExitDataOp::verify() {
  if (failed(ExitDataOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ExitDataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deleteOperandsOperands;
  ::llvm::SMLoc deleteOperandsOperandsLoc;
  (void)deleteOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deleteOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> detachOperandsOperands;
  ::llvm::SMLoc detachOperandsOperandsLoc;
  (void)detachOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> detachOperandsTypes;
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyout"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("delete"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deleteOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deleteOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deleteOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("detach"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  detachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(detachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(detachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deleteOperandsOperands, deleteOperandsTypes, deleteOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(detachOperandsOperands, detachOperandsTypes, detachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(deleteOperandsOperands.size()), static_cast<int32_t>(detachOperandsOperands.size())}));
  return ::mlir::success();
}

void ExitDataOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.exit_data";
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  if (asyncOperand()) {
  p << ' ' << "async";
  p << "(";
  if (::mlir::Value value = asyncOperand())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (waitDevnum()) {
  p << ' ' << "wait_devnum";
  p << "(";
  if (::mlir::Value value = waitDevnum())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (!waitOperands().empty()) {
  p << ' ' << "wait";
  p << "(";
  p << waitOperands();
  p << ' ' << ":";
  p << ' ';
  p << waitOperands().getTypes();
  p << ")";
  }
  if (!copyoutOperands().empty()) {
  p << ' ' << "copyout";
  p << "(";
  p << copyoutOperands();
  p << ' ' << ":";
  p << ' ';
  p << copyoutOperands().getTypes();
  p << ")";
  }
  if (!deleteOperands().empty()) {
  p << ' ' << "delete";
  p << "(";
  p << deleteOperands();
  p << ' ' << ":";
  p << ' ';
  p << deleteOperands().getTypes();
  p << ")";
  }
  if (!detachOperands().empty()) {
  p << ' ' << "detach";
  p << "(";
  p << detachOperands();
  p << ' ' << ":";
  p << ' ';
  p << detachOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp definitions
//===----------------------------------------------------------------------===//

InitOpAdaptor::InitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InitOpAdaptor::InitOpAdaptor(InitOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange InitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InitOpAdaptor::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value InitOpAdaptor::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value InitOpAdaptor::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr InitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InitOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> InitOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range InitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InitOp::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value InitOp::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value InitOp::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange InitOp::deviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange InitOp::deviceNumOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange InitOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> InitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void InitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
}

void InitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InitOp::verify() {
  if (failed(InitOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult InitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceNumOperandOperands;
  ::llvm::SMLoc deviceNumOperandOperandsLoc;
  (void)deviceNumOperandOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceNumOperandTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (succeeded(parser.parseOptionalKeyword("device_type"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device_num"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceNumOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceNumOperandOperands, deviceNumOperandTypes, deviceNumOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(deviceNumOperandOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  return ::mlir::success();
}

void InitOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.init";
  if (!deviceTypeOperands().empty()) {
  p << ' ' << "device_type";
  p << "(";
  p << deviceTypeOperands();
  p << ' ' << ":";
  p << ' ';
  p << deviceTypeOperands().getTypes();
  p << ")";
  }
  if (deviceNumOperand()) {
  p << ' ' << "device_num";
  p << "(";
  if (::mlir::Value value = deviceNumOperand())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (deviceNumOperand() ? ::llvm::ArrayRef<::mlir::Type>(deviceNumOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp definitions
//===----------------------------------------------------------------------===//

LoopOpAdaptor::LoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LoopOpAdaptor::LoopOpAdaptor(LoopOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LoopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange LoopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopOpAdaptor::gangNum() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOpAdaptor::gangStatic() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOpAdaptor::workerNum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOpAdaptor::vectorLength() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange LoopOpAdaptor::tileOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange LoopOpAdaptor::privateOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange LoopOpAdaptor::reductionOperands() {
  return getODSOperands(6);
}

::mlir::DictionaryAttr LoopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopOpAdaptor::collapse() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("collapse").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::UnitAttr LoopOpAdaptor::seq() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("seq").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr LoopOpAdaptor::independent() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("independent").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr LoopOpAdaptor::auto_() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("auto_").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::StringAttr LoopOpAdaptor::reductionOp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("reductionOp").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::IntegerAttr LoopOpAdaptor::exec_mapping() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("exec_mapping").dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

::mlir::RegionRange LoopOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &LoopOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult LoopOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 7 elements, but got ") << numElements;
  }
    {
  auto tblgen_collapse = odsAttrs.get("collapse");
  if (tblgen_collapse) {
    if (!(((tblgen_collapse.isa<::mlir::IntegerAttr>())) && ((tblgen_collapse.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'acc.loop' op ""attribute 'collapse' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  {
  auto tblgen_seq = odsAttrs.get("seq");
  if (tblgen_seq) {
    if (!((tblgen_seq.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.loop' op ""attribute 'seq' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_independent = odsAttrs.get("independent");
  if (tblgen_independent) {
    if (!((tblgen_independent.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.loop' op ""attribute 'independent' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_auto_ = odsAttrs.get("auto_");
  if (tblgen_auto_) {
    if (!((tblgen_auto_.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.loop' op ""attribute 'auto_' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_reductionOp = odsAttrs.get("reductionOp");
  if (tblgen_reductionOp) {
    if (!(((tblgen_reductionOp.isa<::mlir::StringAttr>())) && (((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_add")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_mul")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_max")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_min")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_and")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_or")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_xor")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_leqv")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_lneqv")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_land")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_lor"))))) return emitError(loc, "'acc.loop' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");
  }
  }
  {
  auto tblgen_exec_mapping = odsAttrs.get("exec_mapping");
  if (tblgen_exec_mapping) {
    if (!(((tblgen_exec_mapping.isa<::mlir::IntegerAttr>())) && ((tblgen_exec_mapping.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'acc.loop' op ""attribute 'exec_mapping' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range LoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopOp::gangNum() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOp::gangStatic() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOp::workerNum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOp::vectorLength() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range LoopOp::tileOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range LoopOp::privateOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range LoopOp::reductionOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange LoopOp::gangNumMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange LoopOp::gangStaticMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange LoopOp::workerNumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange LoopOp::vectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange LoopOp::tileOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange LoopOp::privateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange LoopOp::reductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> LoopOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range LoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range LoopOp::results() {
  return getODSResults(0);
}

::mlir::Region &LoopOp::region() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr LoopOp::collapseAttr() {
  return (*this)->getAttr("collapse").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> LoopOp::collapse() {
  auto attr = collapseAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr LoopOp::seqAttr() {
  return (*this)->getAttr("seq").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::seq() {
  auto attr = seqAttr();
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::independentAttr() {
  return (*this)->getAttr("independent").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::independent() {
  auto attr = independentAttr();
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::auto_Attr() {
  return (*this)->getAttr("auto_").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::auto_() {
  auto attr = auto_Attr();
  return attr != nullptr;
}

::mlir::StringAttr LoopOp::reductionOpAttr() {
  return (*this)->getAttr("reductionOp").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > LoopOp::reductionOp() {
  auto attr = reductionOpAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr LoopOp::exec_mappingAttr() {
  return (*this)->getAttr("exec_mapping").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t LoopOp::exec_mapping() {
  auto attr = exec_mappingAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

void LoopOp::collapseAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("collapse", attr);
}

void LoopOp::seqAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("seq", attr);
}

void LoopOp::independentAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("independent", attr);
}

void LoopOp::auto_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr("auto_", attr);
}

void LoopOp::reductionOpAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("reductionOp", attr);
}

void LoopOp::exec_mappingAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("exec_mapping", attr);
}

::mlir::Attribute LoopOp::removeCollapseAttr() {
  return (*this)->removeAttr("collapse");
}

::mlir::Attribute LoopOp::removeSeqAttr() {
  return (*this)->removeAttr("seq");
}

::mlir::Attribute LoopOp::removeIndependentAttr() {
  return (*this)->removeAttr("independent");
}

::mlir::Attribute LoopOp::removeAuto_Attr() {
  return (*this)->removeAttr("auto_");
}

::mlir::Attribute LoopOp::removeReductionOpAttr() {
  return (*this)->removeAttr("reductionOp");
}

void LoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::IntegerAttr exec_mapping) {
  if (gangNum)
    odsState.addOperands(gangNum);
  if (gangStatic)
    odsState.addOperands(gangStatic);
  if (workerNum)
    odsState.addOperands(workerNum);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  odsState.addOperands(tileOperands);
  odsState.addOperands(privateOperands);
  odsState.addOperands(reductionOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(gangNum ? 1 : 0), (gangStatic ? 1 : 0), (workerNum ? 1 : 0), (vectorLength ? 1 : 0), static_cast<int32_t>(tileOperands.size()), static_cast<int32_t>(privateOperands.size()), static_cast<int32_t>(reductionOperands.size())}));
  if (collapse) {
  odsState.addAttribute("collapse", collapse);
  }
  if (seq) {
  odsState.addAttribute("seq", seq);
  }
  if (independent) {
  odsState.addAttribute("independent", independent);
  }
  if (auto_) {
  odsState.addAttribute("auto_", auto_);
  }
  if (reductionOp) {
  odsState.addAttribute("reductionOp", reductionOp);
  }
  odsState.addAttribute("exec_mapping", exec_mapping);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, uint64_t exec_mapping) {
  if (gangNum)
    odsState.addOperands(gangNum);
  if (gangStatic)
    odsState.addOperands(gangStatic);
  if (workerNum)
    odsState.addOperands(workerNum);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  odsState.addOperands(tileOperands);
  odsState.addOperands(privateOperands);
  odsState.addOperands(reductionOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(gangNum ? 1 : 0), (gangStatic ? 1 : 0), (workerNum ? 1 : 0), (vectorLength ? 1 : 0), static_cast<int32_t>(tileOperands.size()), static_cast<int32_t>(privateOperands.size()), static_cast<int32_t>(reductionOperands.size())}));
  if (collapse) {
  odsState.addAttribute("collapse", collapse);
  }
  if (seq) {
  odsState.addAttribute("seq", odsBuilder.getUnitAttr());
  }
  if (independent) {
  odsState.addAttribute("independent", odsBuilder.getUnitAttr());
  }
  if (auto_) {
  odsState.addAttribute("auto_", odsBuilder.getUnitAttr());
  }
  if (reductionOp) {
  odsState.addAttribute("reductionOp", reductionOp);
  }
  odsState.addAttribute("exec_mapping", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), exec_mapping));
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult LoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseLoopOp(parser, result);
}

void LoopOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult LoopOp::verify() {
  if (failed(LoopOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    if (valueGroup3.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup3.size();
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "result", index)))
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
  return ::verifyLoopOp(*this);
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp definitions
//===----------------------------------------------------------------------===//

ParallelOpAdaptor::ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ParallelOpAdaptor::ParallelOpAdaptor(ParallelOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ParallelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ParallelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ParallelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOpAdaptor::async() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ParallelOpAdaptor::waitOperands() {
  return getODSOperands(1);
}

::mlir::Value ParallelOpAdaptor::numGangs() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::numWorkers() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::vectorLength() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::ifCond() {
  auto operands = getODSOperands(5);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::selfCond() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ParallelOpAdaptor::reductionOperands() {
  return getODSOperands(7);
}

::mlir::ValueRange ParallelOpAdaptor::copyOperands() {
  return getODSOperands(8);
}

::mlir::ValueRange ParallelOpAdaptor::copyinOperands() {
  return getODSOperands(9);
}

::mlir::ValueRange ParallelOpAdaptor::copyinReadonlyOperands() {
  return getODSOperands(10);
}

::mlir::ValueRange ParallelOpAdaptor::copyoutOperands() {
  return getODSOperands(11);
}

::mlir::ValueRange ParallelOpAdaptor::copyoutZeroOperands() {
  return getODSOperands(12);
}

::mlir::ValueRange ParallelOpAdaptor::createOperands() {
  return getODSOperands(13);
}

::mlir::ValueRange ParallelOpAdaptor::createZeroOperands() {
  return getODSOperands(14);
}

::mlir::ValueRange ParallelOpAdaptor::noCreateOperands() {
  return getODSOperands(15);
}

::mlir::ValueRange ParallelOpAdaptor::presentOperands() {
  return getODSOperands(16);
}

::mlir::ValueRange ParallelOpAdaptor::devicePtrOperands() {
  return getODSOperands(17);
}

::mlir::ValueRange ParallelOpAdaptor::attachOperands() {
  return getODSOperands(18);
}

::mlir::ValueRange ParallelOpAdaptor::gangPrivateOperands() {
  return getODSOperands(19);
}

::mlir::ValueRange ParallelOpAdaptor::gangFirstPrivateOperands() {
  return getODSOperands(20);
}

::mlir::DictionaryAttr ParallelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ParallelOpAdaptor::asyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("asyncAttr").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr ParallelOpAdaptor::waitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("waitAttr").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr ParallelOpAdaptor::selfAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("selfAttr").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::StringAttr ParallelOpAdaptor::reductionOp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("reductionOp").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr ParallelOpAdaptor::defaultAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("defaultAttr").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange ParallelOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ParallelOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ParallelOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 21)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 21 elements, but got ") << numElements;
  }
    {
  auto tblgen_asyncAttr = odsAttrs.get("asyncAttr");
  if (tblgen_asyncAttr) {
    if (!((tblgen_asyncAttr.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.parallel' op ""attribute 'asyncAttr' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_waitAttr = odsAttrs.get("waitAttr");
  if (tblgen_waitAttr) {
    if (!((tblgen_waitAttr.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.parallel' op ""attribute 'waitAttr' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_selfAttr = odsAttrs.get("selfAttr");
  if (tblgen_selfAttr) {
    if (!((tblgen_selfAttr.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.parallel' op ""attribute 'selfAttr' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_reductionOp = odsAttrs.get("reductionOp");
  if (tblgen_reductionOp) {
    if (!(((tblgen_reductionOp.isa<::mlir::StringAttr>())) && (((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_add")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_mul")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_max")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_min")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_and")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_or")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_xor")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_leqv")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_lneqv")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_land")) || ((tblgen_reductionOp.cast<::mlir::StringAttr>().getValue() == "redop_lor"))))) return emitError(loc, "'acc.parallel' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");
  }
  }
  {
  auto tblgen_defaultAttr = odsAttrs.get("defaultAttr");
  if (tblgen_defaultAttr) {
    if (!(((tblgen_defaultAttr.isa<::mlir::StringAttr>())) && (((tblgen_defaultAttr.cast<::mlir::StringAttr>().getValue() == "present")) || ((tblgen_defaultAttr.cast<::mlir::StringAttr>().getValue() == "none"))))) return emitError(loc, "'acc.parallel' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ParallelOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ParallelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOp::async() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ParallelOp::waitOperands() {
  return getODSOperands(1);
}

::mlir::Value ParallelOp::numGangs() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::numWorkers() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::vectorLength() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::ifCond() {
  auto operands = getODSOperands(5);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::selfCond() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ParallelOp::reductionOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range ParallelOp::copyOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range ParallelOp::copyinOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range ParallelOp::copyinReadonlyOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range ParallelOp::copyoutOperands() {
  return getODSOperands(11);
}

::mlir::Operation::operand_range ParallelOp::copyoutZeroOperands() {
  return getODSOperands(12);
}

::mlir::Operation::operand_range ParallelOp::createOperands() {
  return getODSOperands(13);
}

::mlir::Operation::operand_range ParallelOp::createZeroOperands() {
  return getODSOperands(14);
}

::mlir::Operation::operand_range ParallelOp::noCreateOperands() {
  return getODSOperands(15);
}

::mlir::Operation::operand_range ParallelOp::presentOperands() {
  return getODSOperands(16);
}

::mlir::Operation::operand_range ParallelOp::devicePtrOperands() {
  return getODSOperands(17);
}

::mlir::Operation::operand_range ParallelOp::attachOperands() {
  return getODSOperands(18);
}

::mlir::Operation::operand_range ParallelOp::gangPrivateOperands() {
  return getODSOperands(19);
}

::mlir::Operation::operand_range ParallelOp::gangFirstPrivateOperands() {
  return getODSOperands(20);
}

::mlir::MutableOperandRange ParallelOp::asyncMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::numGangsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::numWorkersMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::vectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::selfCondMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::reductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::copyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::copyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::copyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::copyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::copyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(12);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(12u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::createOperandsMutable() {
  auto range = getODSOperandIndexAndLength(13);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(13u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::createZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(14);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(14u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::noCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(15);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(15u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::presentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(16);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(16u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::devicePtrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(17);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(17u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::attachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(18);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(18u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::gangPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(19);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(19u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ParallelOp::gangFirstPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(20);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(20u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ParallelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ParallelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ParallelOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr ParallelOp::asyncAttrAttr() {
  return (*this)->getAttr("asyncAttr").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::asyncAttr() {
  auto attr = asyncAttrAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOp::waitAttrAttr() {
  return (*this)->getAttr("waitAttr").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::waitAttr() {
  auto attr = waitAttrAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOp::selfAttrAttr() {
  return (*this)->getAttr("selfAttr").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::selfAttr() {
  auto attr = selfAttrAttr();
  return attr != nullptr;
}

::mlir::StringAttr ParallelOp::reductionOpAttr() {
  return (*this)->getAttr("reductionOp").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > ParallelOp::reductionOp() {
  auto attr = reductionOpAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr ParallelOp::defaultAttrAttr() {
  return (*this)->getAttr("defaultAttr").template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > ParallelOp::defaultAttr() {
  auto attr = defaultAttrAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void ParallelOp::asyncAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("asyncAttr", attr);
}

void ParallelOp::waitAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("waitAttr", attr);
}

void ParallelOp::selfAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("selfAttr", attr);
}

void ParallelOp::reductionOpAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("reductionOp", attr);
}

void ParallelOp::defaultAttrAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("defaultAttr", attr);
}

::mlir::Attribute ParallelOp::removeAsyncAttrAttr() {
  return (*this)->removeAttr("asyncAttr");
}

::mlir::Attribute ParallelOp::removeWaitAttrAttr() {
  return (*this)->removeAttr("waitAttr");
}

::mlir::Attribute ParallelOp::removeSelfAttrAttr() {
  return (*this)->removeAttr("selfAttr");
}

::mlir::Attribute ParallelOp::removeReductionOpAttr() {
  return (*this)->removeAttr("reductionOp");
}

::mlir::Attribute ParallelOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr("defaultAttr");
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute("asyncAttr", asyncAttr);
  }
  if (waitAttr) {
  odsState.addAttribute("waitAttr", waitAttr);
  }
  if (selfAttr) {
  odsState.addAttribute("selfAttr", selfAttr);
  }
  if (reductionOp) {
  odsState.addAttribute("reductionOp", reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute("defaultAttr", defaultAttr);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute("asyncAttr", asyncAttr);
  }
  if (waitAttr) {
  odsState.addAttribute("waitAttr", waitAttr);
  }
  if (selfAttr) {
  odsState.addAttribute("selfAttr", selfAttr);
  }
  if (reductionOp) {
  odsState.addAttribute("reductionOp", reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute("defaultAttr", defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute("asyncAttr", odsBuilder.getUnitAttr());
  }
  if (waitAttr) {
  odsState.addAttribute("waitAttr", odsBuilder.getUnitAttr());
  }
  if (selfAttr) {
  odsState.addAttribute("selfAttr", odsBuilder.getUnitAttr());
  }
  if (reductionOp) {
  odsState.addAttribute("reductionOp", reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute("defaultAttr", defaultAttr);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute("asyncAttr", odsBuilder.getUnitAttr());
  }
  if (waitAttr) {
  odsState.addAttribute("waitAttr", odsBuilder.getUnitAttr());
  }
  if (selfAttr) {
  odsState.addAttribute("selfAttr", odsBuilder.getUnitAttr());
  }
  if (reductionOp) {
  odsState.addAttribute("reductionOp", reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute("defaultAttr", defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ParallelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseParallelOp(parser, result);
}

void ParallelOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult ParallelOp::verify() {
  if (failed(ParallelOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    if (valueGroup3.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup3.size();
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    if (valueGroup4.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup4.size();
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    if (valueGroup5.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup5.size();
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    if (valueGroup6.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup6.size();
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup8 = getODSOperands(8);
    for (::mlir::Value v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup9 = getODSOperands(9);
    for (::mlir::Value v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup10 = getODSOperands(10);
    for (::mlir::Value v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup11 = getODSOperands(11);
    for (::mlir::Value v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup12 = getODSOperands(12);
    for (::mlir::Value v : valueGroup12) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup13 = getODSOperands(13);
    for (::mlir::Value v : valueGroup13) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup14 = getODSOperands(14);
    for (::mlir::Value v : valueGroup14) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup15 = getODSOperands(15);
    for (::mlir::Value v : valueGroup15) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup16 = getODSOperands(16);
    for (::mlir::Value v : valueGroup16) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup17 = getODSOperands(17);
    for (::mlir::Value v : valueGroup17) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup18 = getODSOperands(18);
    for (::mlir::Value v : valueGroup18) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup19 = getODSOperands(19);
    for (::mlir::Value v : valueGroup19) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup20 = getODSOperands(20);
    for (::mlir::Value v : valueGroup20) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
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
  return ::mlir::success();
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp definitions
//===----------------------------------------------------------------------===//

ShutdownOpAdaptor::ShutdownOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ShutdownOpAdaptor::ShutdownOpAdaptor(ShutdownOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ShutdownOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShutdownOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange ShutdownOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ShutdownOpAdaptor::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value ShutdownOpAdaptor::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ShutdownOpAdaptor::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr ShutdownOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ShutdownOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    return ::mlir::success();
}



std::pair<unsigned, unsigned> ShutdownOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range ShutdownOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ShutdownOp::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value ShutdownOp::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ShutdownOp::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange ShutdownOp::deviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ShutdownOp::deviceNumOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ShutdownOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ShutdownOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShutdownOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ShutdownOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
}

void ShutdownOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShutdownOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShutdownOp::verify() {
  if (failed(ShutdownOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ShutdownOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceNumOperandOperands;
  ::llvm::SMLoc deviceNumOperandOperandsLoc;
  (void)deviceNumOperandOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceNumOperandTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (succeeded(parser.parseOptionalKeyword("device_type"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device_num"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceNumOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceNumOperandOperands, deviceNumOperandTypes, deviceNumOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(deviceNumOperandOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  return ::mlir::success();
}

void ShutdownOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.shutdown";
  if (!deviceTypeOperands().empty()) {
  p << ' ' << "device_type";
  p << "(";
  p << deviceTypeOperands();
  p << ' ' << ":";
  p << ' ';
  p << deviceTypeOperands().getTypes();
  p << ")";
  }
  if (deviceNumOperand()) {
  p << ' ' << "device_num";
  p << "(";
  if (::mlir::Value value = deviceNumOperand())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (deviceNumOperand() ? ::llvm::ArrayRef<::mlir::Type>(deviceNumOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp definitions
//===----------------------------------------------------------------------===//

TerminatorOpAdaptor::TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TerminatorOpAdaptor::TerminatorOpAdaptor(TerminatorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TerminatorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TerminatorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TerminatorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TerminatorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TerminatorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TerminatorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TerminatorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TerminatorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TerminatorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TerminatorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TerminatorOp::verify() {
  if (failed(TerminatorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult TerminatorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TerminatorOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.terminator";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp definitions
//===----------------------------------------------------------------------===//

UpdateOpAdaptor::UpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UpdateOpAdaptor::UpdateOpAdaptor(UpdateOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UpdateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UpdateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange UpdateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UpdateOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value UpdateOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange UpdateOpAdaptor::waitOperands() {
  return getODSOperands(2);
}

::mlir::ValueRange UpdateOpAdaptor::deviceTypeOperands() {
  return getODSOperands(3);
}

::mlir::Value UpdateOpAdaptor::ifCond() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange UpdateOpAdaptor::hostOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange UpdateOpAdaptor::deviceOperands() {
  return getODSOperands(6);
}

::mlir::DictionaryAttr UpdateOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr UpdateOpAdaptor::async() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("async").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr UpdateOpAdaptor::wait() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("wait").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr UpdateOpAdaptor::ifPresent() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("ifPresent").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult UpdateOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 7 elements, but got ") << numElements;
  }
    {
  auto tblgen_async = odsAttrs.get("async");
  if (tblgen_async) {
    if (!((tblgen_async.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.update' op ""attribute 'async' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_wait = odsAttrs.get("wait");
  if (tblgen_wait) {
    if (!((tblgen_wait.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.update' op ""attribute 'wait' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_ifPresent = odsAttrs.get("ifPresent");
  if (tblgen_ifPresent) {
    if (!((tblgen_ifPresent.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.update' op ""attribute 'ifPresent' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UpdateOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range UpdateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UpdateOp::asyncOperand() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value UpdateOp::waitDevnum() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range UpdateOp::waitOperands() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range UpdateOp::deviceTypeOperands() {
  return getODSOperands(3);
}

::mlir::Value UpdateOp::ifCond() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range UpdateOp::hostOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range UpdateOp::deviceOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange UpdateOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange UpdateOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange UpdateOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange UpdateOp::deviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange UpdateOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange UpdateOp::hostOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange UpdateOp::deviceOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> UpdateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UpdateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr UpdateOp::asyncAttr() {
  return (*this)->getAttr("async").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOp::waitAttr() {
  return (*this)->getAttr("wait").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOp::ifPresentAttr() {
  return (*this)->getAttr("ifPresent").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::ifPresent() {
  auto attr = ifPresentAttr();
  return attr != nullptr;
}

void UpdateOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("async", attr);
}

void UpdateOp::waitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("wait", attr);
}

void UpdateOp::ifPresentAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("ifPresent", attr);
}

::mlir::Attribute UpdateOp::removeAsyncAttr() {
  return (*this)->removeAttr("async");
}

::mlir::Attribute UpdateOp::removeWaitAttr() {
  return (*this)->removeAttr("wait");
}

::mlir::Attribute UpdateOp::removeIfPresentAttr() {
  return (*this)->removeAttr("ifPresent");
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent) {
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), (ifCond ? 1 : 0), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  if (wait) {
  odsState.addAttribute("wait", wait);
  }
  if (ifPresent) {
  odsState.addAttribute("ifPresent", ifPresent);
  }
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent) {
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), (ifCond ? 1 : 0), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  if (wait) {
  odsState.addAttribute("wait", wait);
  }
  if (ifPresent) {
  odsState.addAttribute("ifPresent", ifPresent);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent) {
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), (ifCond ? 1 : 0), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute("wait", odsBuilder.getUnitAttr());
  }
  if (ifPresent) {
  odsState.addAttribute("ifPresent", odsBuilder.getUnitAttr());
  }
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent) {
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({(asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), (ifCond ? 1 : 0), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute("wait", odsBuilder.getUnitAttr());
  }
  if (ifPresent) {
  odsState.addAttribute("ifPresent", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UpdateOp::verify() {
  if (failed(UpdateOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    if (valueGroup4.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup4.size();
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult UpdateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> hostOperandsOperands;
  ::llvm::SMLoc hostOperandsOperandsLoc;
  (void)hostOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> hostOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceOperandsOperands;
  ::llvm::SMLoc deviceOperandsOperandsLoc;
  (void)deviceOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceOperandsTypes;
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device_type"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("host"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  hostOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(hostOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(hostOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(hostOperandsOperands, hostOperandsTypes, hostOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperandsOperands, deviceOperandsTypes, deviceOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(hostOperandsOperands.size()), static_cast<int32_t>(deviceOperandsOperands.size())}));
  return ::mlir::success();
}

void UpdateOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.update";
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  if (asyncOperand()) {
  p << ' ' << "async";
  p << "(";
  if (::mlir::Value value = asyncOperand())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (waitDevnum()) {
  p << ' ' << "wait_devnum";
  p << "(";
  if (::mlir::Value value = waitDevnum())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (!deviceTypeOperands().empty()) {
  p << ' ' << "device_type";
  p << "(";
  p << deviceTypeOperands();
  p << ' ' << ":";
  p << ' ';
  p << deviceTypeOperands().getTypes();
  p << ")";
  }
  if (!waitOperands().empty()) {
  p << ' ' << "wait";
  p << "(";
  p << waitOperands();
  p << ' ' << ":";
  p << ' ';
  p << waitOperands().getTypes();
  p << ")";
  }
  if (!hostOperands().empty()) {
  p << ' ' << "host";
  p << "(";
  p << hostOperands();
  p << ' ' << ":";
  p << ' ';
  p << hostOperands().getTypes();
  p << ")";
  }
  if (!deviceOperands().empty()) {
  p << ' ' << "device";
  p << "(";
  p << deviceOperands();
  p << ' ' << ":";
  p << ' ';
  p << deviceOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp definitions
//===----------------------------------------------------------------------===//

WaitOpAdaptor::WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WaitOpAdaptor::WaitOpAdaptor(WaitOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WaitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WaitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange WaitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WaitOpAdaptor::waitOperands() {
  return getODSOperands(0);
}

::mlir::Value WaitOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOpAdaptor::ifCond() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr WaitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr WaitOpAdaptor::async() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("async").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult WaitOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements, but got ") << numElements;
  }
    {
  auto tblgen_async = odsAttrs.get("async");
  if (tblgen_async) {
    if (!((tblgen_async.isa<::mlir::UnitAttr>()))) return emitError(loc, "'acc.wait' op ""attribute 'async' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> WaitOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range WaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WaitOp::waitOperands() {
  return getODSOperands(0);
}

::mlir::Value WaitOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOp::ifCond() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange WaitOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange WaitOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange WaitOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange WaitOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> WaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr WaitOp::asyncAttr() {
  return (*this)->getAttr("async").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WaitOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

void WaitOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("async", attr);
}

::mlir::Attribute WaitOp::removeAsyncAttr() {
  return (*this)->removeAttr("async");
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute("async", async);
  }
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute("async", async);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute("async", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WaitOp::verify() {
  if (failed(WaitOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    if (valueGroup3.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup3.size();
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult WaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (succeeded(parser.parseOptionalLParen())) {

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  return ::mlir::success();
}

void WaitOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.wait";
  if (!waitOperands().empty()) {
  p << "(";
  p << waitOperands();
  p << ' ' << ":";
  p << ' ';
  p << waitOperands().getTypes();
  p << ")";
  }
  if (asyncOperand()) {
  p << ' ' << "async";
  p << "(";
  if (::mlir::Value value = asyncOperand())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (waitDevnum()) {
  p << ' ' << "wait_devnum";
  p << "(";
  if (::mlir::Value value = waitDevnum())
    p << value;
  p << ' ' << ":";
  p << ' ';
  p << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  p << ")";
  }
  if (ifCond()) {
  p << ' ' << "if";
  p << "(";
  if (::mlir::Value value = ifCond())
    p << value;
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

YieldOpAdaptor::YieldOpAdaptor(YieldOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 /* nothing to do */ 
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verify() {
  if (failed(YieldOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &p) {
  p << "acc.yield";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  p << ' ';
  p << operands();
  p << ' ' << ":";
  p << ' ';
  p << operands().getTypes();
  }
}

} // namespace acc
} // namespace mlir

#endif  // GET_OP_CLASSES

