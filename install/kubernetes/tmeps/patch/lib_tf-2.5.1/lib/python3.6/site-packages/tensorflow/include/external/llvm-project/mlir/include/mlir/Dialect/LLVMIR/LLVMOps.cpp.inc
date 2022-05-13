/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::LLVM::AShrOp,
::mlir::LLVM::AccessGroupMetadataOp,
::mlir::LLVM::AddOp,
::mlir::LLVM::AddrSpaceCastOp,
::mlir::LLVM::AddressOfOp,
::mlir::LLVM::AllocaOp,
::mlir::LLVM::AndOp,
::mlir::LLVM::AssumeOp,
::mlir::LLVM::AtomicCmpXchgOp,
::mlir::LLVM::AtomicRMWOp,
::mlir::LLVM::BitReverseOp,
::mlir::LLVM::BitcastOp,
::mlir::LLVM::BrOp,
::mlir::LLVM::CallOp,
::mlir::LLVM::CondBrOp,
::mlir::LLVM::ConstantOp,
::mlir::LLVM::CopySignOp,
::mlir::LLVM::CoroBeginOp,
::mlir::LLVM::CoroEndOp,
::mlir::LLVM::CoroFreeOp,
::mlir::LLVM::CoroIdOp,
::mlir::LLVM::CoroResumeOp,
::mlir::LLVM::CoroSaveOp,
::mlir::LLVM::CoroSizeOp,
::mlir::LLVM::CoroSuspendOp,
::mlir::LLVM::CosOp,
::mlir::LLVM::CtPopOp,
::mlir::LLVM::DialectCastOp,
::mlir::LLVM::Exp2Op,
::mlir::LLVM::ExpOp,
::mlir::LLVM::ExtractElementOp,
::mlir::LLVM::ExtractValueOp,
::mlir::LLVM::FAbsOp,
::mlir::LLVM::FAddOp,
::mlir::LLVM::FCeilOp,
::mlir::LLVM::FCmpOp,
::mlir::LLVM::FDivOp,
::mlir::LLVM::FFloorOp,
::mlir::LLVM::FMAOp,
::mlir::LLVM::FMulAddOp,
::mlir::LLVM::FMulOp,
::mlir::LLVM::FNegOp,
::mlir::LLVM::FPExtOp,
::mlir::LLVM::FPToSIOp,
::mlir::LLVM::FPToUIOp,
::mlir::LLVM::FPTruncOp,
::mlir::LLVM::FRemOp,
::mlir::LLVM::FSubOp,
::mlir::LLVM::FenceOp,
::mlir::LLVM::FreezeOp,
::mlir::LLVM::GEPOp,
::mlir::LLVM::GetActiveLaneMaskOp,
::mlir::LLVM::GlobalOp,
::mlir::LLVM::ICmpOp,
::mlir::LLVM::InlineAsmOp,
::mlir::LLVM::InsertElementOp,
::mlir::LLVM::InsertValueOp,
::mlir::LLVM::IntToPtrOp,
::mlir::LLVM::InvokeOp,
::mlir::LLVM::LLVMFuncOp,
::mlir::LLVM::LShrOp,
::mlir::LLVM::LandingpadOp,
::mlir::LLVM::LoadOp,
::mlir::LLVM::Log10Op,
::mlir::LLVM::Log2Op,
::mlir::LLVM::LogOp,
::mlir::LLVM::MaskedLoadOp,
::mlir::LLVM::MaskedStoreOp,
::mlir::LLVM::MatrixColumnMajorLoadOp,
::mlir::LLVM::MatrixColumnMajorStoreOp,
::mlir::LLVM::MatrixMultiplyOp,
::mlir::LLVM::MatrixTransposeOp,
::mlir::LLVM::MaxNumOp,
::mlir::LLVM::MaximumOp,
::mlir::LLVM::MemcpyInlineOp,
::mlir::LLVM::MemcpyOp,
::mlir::LLVM::MetadataOp,
::mlir::LLVM::MinNumOp,
::mlir::LLVM::MinimumOp,
::mlir::LLVM::MulOp,
::mlir::LLVM::NullOp,
::mlir::LLVM::OrOp,
::mlir::LLVM::PowOp,
::mlir::LLVM::Prefetch,
::mlir::LLVM::PtrToIntOp,
::mlir::LLVM::ResumeOp,
::mlir::LLVM::ReturnOp,
::mlir::LLVM::SAddWithOverflowOp,
::mlir::LLVM::SDivOp,
::mlir::LLVM::SExtOp,
::mlir::LLVM::SIToFPOp,
::mlir::LLVM::SMaxOp,
::mlir::LLVM::SMinOp,
::mlir::LLVM::SMulWithOverflowOp,
::mlir::LLVM::SRemOp,
::mlir::LLVM::SSubWithOverflowOp,
::mlir::LLVM::SelectOp,
::mlir::LLVM::ShlOp,
::mlir::LLVM::ShuffleVectorOp,
::mlir::LLVM::SinOp,
::mlir::LLVM::SqrtOp,
::mlir::LLVM::StackRestoreOp,
::mlir::LLVM::StackSaveOp,
::mlir::LLVM::StoreOp,
::mlir::LLVM::SubOp,
::mlir::LLVM::SwitchOp,
::mlir::LLVM::TruncOp,
::mlir::LLVM::UAddWithOverflowOp,
::mlir::LLVM::UDivOp,
::mlir::LLVM::UIToFPOp,
::mlir::LLVM::UMulWithOverflowOp,
::mlir::LLVM::URemOp,
::mlir::LLVM::USubWithOverflowOp,
::mlir::LLVM::UndefOp,
::mlir::LLVM::UnreachableOp,
::mlir::LLVM::XOrOp,
::mlir::LLVM::ZExtOp,
::mlir::LLVM::masked_compressstore,
::mlir::LLVM::masked_expandload,
::mlir::LLVM::masked_gather,
::mlir::LLVM::masked_scatter,
::mlir::LLVM::vector_reduce_add,
::mlir::LLVM::vector_reduce_and,
::mlir::LLVM::vector_reduce_fadd,
::mlir::LLVM::vector_reduce_fmax,
::mlir::LLVM::vector_reduce_fmin,
::mlir::LLVM::vector_reduce_fmul,
::mlir::LLVM::vector_reduce_mul,
::mlir::LLVM::vector_reduce_or,
::mlir::LLVM::vector_reduce_smax,
::mlir::LLVM::vector_reduce_smin,
::mlir::LLVM::vector_reduce_umax,
::mlir::LLVM::vector_reduce_umin,
::mlir::LLVM::vector_reduce_xor
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM integer type or LLVM dialect-compatible vector of LLVM integer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::LLVM::LLVMPointerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM integer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMPointerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::IntegerType>())) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMPointerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM pointer to LLVM integer type or LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::LLVM::LLVMPointerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM integer type or LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMStructType>().getBody().size() == 2)) && (((type.cast<::mlir::LLVM::LLVMStructType>().getBody()[0].isa<::mlir::IntegerType>())) || ((type.cast<::mlir::LLVM::LLVMStructType>().getBody()[0].isa<::mlir::LLVM::LLVMPointerType>()))) && ((type.cast<::mlir::LLVM::LLVMStructType>().getBody()[1].isSignlessInteger(1))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be an LLVM struct type with any integer or pointer followed by a single-bit integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((::mlir::LLVM::isCompatibleFloatingPointType(type.cast<::mlir::LLVM::LLVMPointerType>().getElementType()))) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::IntegerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM pointer to floating point LLVM type or LLVM integer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || ((type.isa<::mlir::IntegerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be floating point LLVM type or LLVM integer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps10(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(!((type.isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM non-aggregate type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps11(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps12(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps13(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMTokenType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM token type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps14(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isSignlessInteger(8))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM pointer to 8-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps15(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps16(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps17(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleVectorType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible vector type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps18(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>()))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM aggregate type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps19(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType(type))))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps20(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isSignlessInteger(1))) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isSignlessInteger(1)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps21(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM integer type or LLVM dialect-compatible vector of LLVM integer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps22(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps23(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((::mlir::LLVM::isCompatibleType(type))) && ((!type.isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>())))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be primitive LLVM type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps24(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && ((((::mlir::LLVM::isCompatibleType(type.cast<::mlir::LLVM::LLVMPointerType>().getElementType()))) && ((!type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>()))) && (!(((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().cast<::mlir::LLVM::LLVMStructType>().isOpaque()))))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM pointer to LLVM type with size, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps25(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((::mlir::LLVM::isCompatibleType(type))) && ((!type.isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>()))) && (!(((type.isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMStructType>().isOpaque())))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM type with size, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps26(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AShrOp definitions
//===----------------------------------------------------------------------===//

AShrOpAdaptor::AShrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AShrOpAdaptor::AShrOpAdaptor(AShrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AShrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AShrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AShrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AShrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AShrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AShrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AShrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AShrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOp::res() {
  return *getODSResults(0).begin();
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AShrOp::verify() {
  if (failed(AShrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult AShrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AShrOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.ashr";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void AShrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AccessGroupMetadataOp definitions
//===----------------------------------------------------------------------===//

AccessGroupMetadataOpAdaptor::AccessGroupMetadataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AccessGroupMetadataOpAdaptor::AccessGroupMetadataOpAdaptor(AccessGroupMetadataOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AccessGroupMetadataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AccessGroupMetadataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AccessGroupMetadataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AccessGroupMetadataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AccessGroupMetadataOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult AccessGroupMetadataOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'llvm.access_group' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'llvm.access_group' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AccessGroupMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AccessGroupMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AccessGroupMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AccessGroupMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AccessGroupMetadataOp::sym_nameAttr() {
  return (*this)->getAttr("sym_name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef AccessGroupMetadataOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

void AccessGroupMetadataOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_name", attr);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name) {
  odsState.addAttribute("sym_name", sym_name);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AccessGroupMetadataOp::verify() {
  if (failed(AccessGroupMetadataOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult AccessGroupMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AccessGroupMetadataOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.access_group";
  p << ' ';
  p.printSymbolName(sym_nameAttr().getValue());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddOp definitions
//===----------------------------------------------------------------------===//

AddOpAdaptor::AddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AddOpAdaptor::AddOpAdaptor(AddOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::res() {
  return *getODSResults(0).begin();
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AddOp::verify() {
  if (failed(AddOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AddOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.add";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void AddOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddrSpaceCastOp definitions
//===----------------------------------------------------------------------===//

AddrSpaceCastOpAdaptor::AddrSpaceCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AddrSpaceCastOpAdaptor::AddrSpaceCastOpAdaptor(AddrSpaceCastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AddrSpaceCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AddrSpaceCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddrSpaceCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AddrSpaceCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AddrSpaceCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AddrSpaceCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddrSpaceCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AddrSpaceCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddrSpaceCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddrSpaceCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOp::res() {
  return *getODSResults(0).begin();
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AddrSpaceCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void AddrSpaceCastOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult AddrSpaceCastOp::verify() {
  if (failed(AddrSpaceCastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void AddrSpaceCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddressOfOp definitions
//===----------------------------------------------------------------------===//

AddressOfOpAdaptor::AddressOfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AddressOfOpAdaptor::AddressOfOpAdaptor(AddressOfOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AddressOfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AddressOfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddressOfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AddressOfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr AddressOfOpAdaptor::global_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("global_name").cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult AddressOfOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_global_name = odsAttrs.get("global_name");
  if (!tblgen_global_name) return emitError(loc, "'llvm.mlir.addressof' op ""requires attribute 'global_name'");
    if (!((tblgen_global_name.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'llvm.mlir.addressof' op ""attribute 'global_name' failed to satisfy constraint: flat symbol reference attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AddressOfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddressOfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AddressOfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddressOfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddressOfOp::res() {
  return *getODSResults(0).begin();
}

::mlir::FlatSymbolRefAttr AddressOfOp::global_nameAttr() {
  return (*this)->getAttr("global_name").template cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef AddressOfOp::global_name() {
  auto attr = global_nameAttr();
  return attr.getValue();
}

void AddressOfOp::global_nameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr("global_name", attr);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, GlobalOp global, ArrayRef<NamedAttribute> attrs ) {
      build(odsBuilder, odsState,
            LLVM::LLVMPointerType::get(global.getType(), global.addr_space()),
            global.sym_name());
      odsState.addAttributes(attrs);
    
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFuncOp func, ArrayRef<NamedAttribute> attrs ) {
      build(odsBuilder, odsState,
            LLVM::LLVMPointerType::get(func.getType()), func.getName());
      odsState.addAttributes(attrs);
    
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::FlatSymbolRefAttr global_name) {
  odsState.addAttribute("global_name", global_name);
  odsState.addTypes(res);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr global_name) {
  odsState.addAttribute("global_name", global_name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::llvm::StringRef global_name) {
  odsState.addAttribute("global_name", odsBuilder.getSymbolRefAttr(global_name));
  odsState.addTypes(res);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef global_name) {
  odsState.addAttribute("global_name", odsBuilder.getSymbolRefAttr(global_name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddressOfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddressOfOp::verify() {
  if (failed(AddressOfOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult AddressOfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr global_nameAttr;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  if (parser.parseAttribute(global_nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "global_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void AddressOfOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.mlir.addressof";
  p << ' ';
  p.printAttributeWithoutType(global_nameAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"global_name"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AllocaOp definitions
//===----------------------------------------------------------------------===//

AllocaOpAdaptor::AllocaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AllocaOpAdaptor::AllocaOpAdaptor(AllocaOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AllocaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AllocaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AllocaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocaOpAdaptor::arraySize() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AllocaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AllocaOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult AllocaOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (tblgen_alignment) {
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'llvm.alloca' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AllocaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AllocaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocaOp::arraySize() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AllocaOp::arraySizeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AllocaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocaOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr AllocaOp::alignmentAttr() {
  return (*this)->getAttr("alignment").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> AllocaOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void AllocaOp::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

::mlir::Attribute AllocaOp::removeAlignmentAttr() {
  return (*this)->removeAttr("alignment");
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value arraySize, unsigned alignment) {
      if (alignment == 0)
        return build(odsBuilder, odsState, resultType, arraySize, IntegerAttr());
      build(odsBuilder, odsState, resultType, arraySize,
        odsBuilder.getI64IntegerAttr(alignment));
  
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(arraySize);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  odsState.addTypes(res);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(arraySize);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AllocaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseAllocaOp(parser, result);
}

void AllocaOp::print(::mlir::OpAsmPrinter &p) {
  printAllocaOp(p, *this);
}

::mlir::LogicalResult AllocaOp::verify() {
  if (failed(AllocaOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AndOp definitions
//===----------------------------------------------------------------------===//

AndOpAdaptor::AndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AndOpAdaptor::AndOpAdaptor(AndOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AndOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AndOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AndOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AndOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::res() {
  return *getODSResults(0).begin();
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AndOp::verify() {
  if (failed(AndOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult AndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AndOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.and";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void AndOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AssumeOp definitions
//===----------------------------------------------------------------------===//

AssumeOpAdaptor::AssumeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AssumeOpAdaptor::AssumeOpAdaptor(AssumeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AssumeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AssumeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AssumeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssumeOpAdaptor::cond() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AssumeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AssumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AssumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AssumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssumeOp::cond() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AssumeOp::condMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AssumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AssumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AssumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value cond) {
  odsState.addOperands(cond);
}

void AssumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond) {
  odsState.addOperands(cond);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AssumeOp::verify() {
  if (failed(AssumeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AtomicCmpXchgOp definitions
//===----------------------------------------------------------------------===//

AtomicCmpXchgOpAdaptor::AtomicCmpXchgOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AtomicCmpXchgOpAdaptor::AtomicCmpXchgOpAdaptor(AtomicCmpXchgOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AtomicCmpXchgOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicCmpXchgOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicCmpXchgOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOpAdaptor::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicCmpXchgOpAdaptor::cmp() {
  return *getODSOperands(1).begin();
}

::mlir::Value AtomicCmpXchgOpAdaptor::val() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr AtomicCmpXchgOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOpAdaptor::success_ordering() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::AtomicOrderingAttr attr = odsAttrs.get("success_ordering").cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOpAdaptor::failure_ordering() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::AtomicOrderingAttr attr = odsAttrs.get("failure_ordering").cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LogicalResult AtomicCmpXchgOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_success_ordering = odsAttrs.get("success_ordering");
  if (!tblgen_success_ordering) return emitError(loc, "'llvm.cmpxchg' op ""requires attribute 'success_ordering'");
    if (!((tblgen_success_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>()))) return emitError(loc, "'llvm.cmpxchg' op ""attribute 'success_ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");
  }
  {
  auto tblgen_failure_ordering = odsAttrs.get("failure_ordering");
  if (!tblgen_failure_ordering) return emitError(loc, "'llvm.cmpxchg' op ""requires attribute 'failure_ordering'");
    if (!((tblgen_failure_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>()))) return emitError(loc, "'llvm.cmpxchg' op ""attribute 'failure_ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AtomicCmpXchgOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicCmpXchgOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOp::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicCmpXchgOp::cmp() {
  return *getODSOperands(1).begin();
}

::mlir::Value AtomicCmpXchgOp::val() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange AtomicCmpXchgOp::ptrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AtomicCmpXchgOp::cmpMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AtomicCmpXchgOp::valMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AtomicCmpXchgOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicCmpXchgOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOp::success_orderingAttr() {
  return (*this)->getAttr("success_ordering").template cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOp::success_ordering() {
  auto attr = success_orderingAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOp::failure_orderingAttr() {
  return (*this)->getAttr("failure_ordering").template cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOp::failure_ordering() {
  auto attr = failure_orderingAttr();
  return attr.getValue();
}

void AtomicCmpXchgOp::success_orderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr("success_ordering", attr);
}

void AtomicCmpXchgOp::failure_orderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr("failure_ordering", attr);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute("success_ordering", success_ordering);
  odsState.addAttribute("failure_ordering", failure_ordering);
  odsState.addTypes(res);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute("success_ordering", success_ordering);
  odsState.addAttribute("failure_ordering", failure_ordering);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute("success_ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute("failure_ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  odsState.addTypes(res);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute("success_ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute("failure_ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AtomicCmpXchgOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseAtomicCmpXchgOp(parser, result);
}

void AtomicCmpXchgOp::print(::mlir::OpAsmPrinter &p) {
  printAtomicCmpXchgOp(p, *this);
}

::mlir::LogicalResult AtomicCmpXchgOp::verify() {
  if (failed(AtomicCmpXchgOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AtomicRMWOp definitions
//===----------------------------------------------------------------------===//

AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(AtomicRMWOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AtomicRMWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicRMWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicRMWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOpAdaptor::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicRMWOpAdaptor::val() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AtomicRMWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicBinOpAttr AtomicRMWOpAdaptor::bin_op() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::AtomicBinOpAttr attr = odsAttrs.get("bin_op").cast<::mlir::LLVM::AtomicBinOpAttr>();
  return attr;
}

::mlir::LLVM::AtomicOrderingAttr AtomicRMWOpAdaptor::ordering() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::AtomicOrderingAttr attr = odsAttrs.get("ordering").cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::LogicalResult AtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_bin_op = odsAttrs.get("bin_op");
  if (!tblgen_bin_op) return emitError(loc, "'llvm.atomicrmw' op ""requires attribute 'bin_op'");
    if (!((tblgen_bin_op.isa<::mlir::LLVM::AtomicBinOpAttr>()))) return emitError(loc, "'llvm.atomicrmw' op ""attribute 'bin_op' failed to satisfy constraint: llvm.atomicrmw binary operations");
  }
  {
  auto tblgen_ordering = odsAttrs.get("ordering");
  if (!tblgen_ordering) return emitError(loc, "'llvm.atomicrmw' op ""requires attribute 'ordering'");
    if (!((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>()))) return emitError(loc, "'llvm.atomicrmw' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicRMWOp::val() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AtomicRMWOp::ptrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AtomicRMWOp::valMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::AtomicBinOpAttr AtomicRMWOp::bin_opAttr() {
  return (*this)->getAttr("bin_op").template cast<::mlir::LLVM::AtomicBinOpAttr>();
}

::mlir::LLVM::AtomicBinOp AtomicRMWOp::bin_op() {
  auto attr = bin_opAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicRMWOp::orderingAttr() {
  return (*this)->getAttr("ordering").template cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicRMWOp::ordering() {
  auto attr = orderingAttr();
  return attr.getValue();
}

void AtomicRMWOp::bin_opAttr(::mlir::LLVM::AtomicBinOpAttr attr) {
  (*this)->setAttr("bin_op", attr);
}

void AtomicRMWOp::orderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr("ordering", attr);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute("bin_op", bin_op);
  odsState.addAttribute("ordering", ordering);
  odsState.addTypes(res);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute("bin_op", bin_op);
  odsState.addAttribute("ordering", ordering);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute("bin_op", ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute("ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  odsState.addTypes(res);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute("bin_op", ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute("ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AtomicRMWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseAtomicRMWOp(parser, result);
}

void AtomicRMWOp::print(::mlir::OpAsmPrinter &p) {
  printAtomicRMWOp(p, *this);
}

::mlir::LogicalResult AtomicRMWOp::verify() {
  if (failed(AtomicRMWOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BitReverseOp definitions
//===----------------------------------------------------------------------===//

BitReverseOpAdaptor::BitReverseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BitReverseOpAdaptor::BitReverseOpAdaptor(BitReverseOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BitReverseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitReverseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitReverseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitReverseOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BitReverseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitReverseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BitReverseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitReverseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitReverseOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitReverseOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BitReverseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitReverseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitReverseOp::res() {
  return *getODSResults(0).begin();
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitReverseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult BitReverseOp::verify() {
  if (failed(BitReverseOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void BitReverseOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BitcastOp definitions
//===----------------------------------------------------------------------===//

BitcastOpAdaptor::BitcastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BitcastOpAdaptor::BitcastOpAdaptor(BitcastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BitcastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitcastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitcastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BitcastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitcastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BitcastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitcastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitcastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BitcastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitcastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOp::res() {
  return *getODSResults(0).begin();
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitcastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult BitcastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void BitcastOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult BitcastOp::verify() {
  if (failed(BitcastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void BitcastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BrOp definitions
//===----------------------------------------------------------------------===//

BrOpAdaptor::BrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BrOpAdaptor::BrOpAdaptor(BrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BrOpAdaptor::destOperands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr BrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BrOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range BrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BrOp::destOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange BrOp::destOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BrOp::dest() {
  return (*this)->getSuccessor(0);
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BrOp::verify() {
  if (failed(BrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
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



::mlir::ParseResult BrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> destOperandsOperands;
  ::llvm::SMLoc destOperandsOperandsLoc;
  (void)destOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> destOperandsTypes;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  destOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(destOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(destOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  return ::mlir::success();
}

void BrOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.br";
  p << ' ';
  p << dest();
  if (!destOperands().empty()) {
  p << "(";
  p << destOperands();
  p << ' ' << ":";
  p << ' ';
  p << destOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void BrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CallOp definitions
//===----------------------------------------------------------------------===//

CallOpAdaptor::CallOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CallOpAdaptor::CallOpAdaptor(CallOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CallOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CallOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CallOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CallOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr CallOpAdaptor::callee() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("callee").dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LLVM::FMFAttr CallOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult CallOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_callee = odsAttrs.get("callee");
  if (tblgen_callee) {
    if (!((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'llvm.call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  }
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.call' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CallOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CallOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::FlatSymbolRefAttr CallOp::calleeAttr() {
  return (*this)->getAttr("callee").template dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > CallOp::callee() {
  auto attr = calleeAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LLVM::FMFAttr CallOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags CallOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void CallOp::calleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr("callee", attr);
}

void CallOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

::mlir::Attribute CallOp::removeCalleeAttr() {
  return (*this)->removeAttr("callee");
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFuncOp func, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
      Type resultType = func.getType().getReturnType();
      if (!resultType.isa<LLVM::LLVMVoidType>())
        odsState.addTypes(resultType);
      odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(func));
      odsState.addAttributes(attributes);
      odsState.addOperands(operands);
    
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(odsArg_0);
  if (callee) {
  odsState.addAttribute("callee", callee);
  }
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(odsArg_0);
  if (callee) {
  odsState.addAttribute("callee", callee);
  }
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult CallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseCallOp(parser, result);
}

void CallOp::print(::mlir::OpAsmPrinter &p) {
  printCallOp(p, *this);
}

::mlir::LogicalResult CallOp::verify() {
  if (failed(CallOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CondBrOp definitions
//===----------------------------------------------------------------------===//

CondBrOpAdaptor::CondBrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CondBrOpAdaptor::CondBrOpAdaptor(CondBrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CondBrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CondBrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange CondBrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CondBrOpAdaptor::condition() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CondBrOpAdaptor::trueDestOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange CondBrOpAdaptor::falseDestOperands() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr CondBrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr CondBrOpAdaptor::branch_weights() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("branch_weights").dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::mlir::LogicalResult CondBrOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    {
  auto tblgen_branch_weights = odsAttrs.get("branch_weights");
  if (tblgen_branch_weights) {
    if (!((tblgen_branch_weights.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'llvm.cond_br' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CondBrOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range CondBrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CondBrOp::condition() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CondBrOp::trueDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CondBrOp::falseDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CondBrOp::conditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CondBrOp::trueDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CondBrOp::falseDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> CondBrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CondBrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CondBrOp::trueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CondBrOp::falseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::ElementsAttr CondBrOp::branch_weightsAttr() {
  return (*this)->getAttr("branch_weights").template dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > CondBrOp::branch_weights() {
  auto attr = branch_weightsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

void CondBrOp::branch_weightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("branch_weights", attr);
}

::mlir::Attribute CondBrOp::removeBranch_weightsAttr() {
  return (*this)->removeAttr("branch_weights");
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block * trueDest, ValueRange trueOperands, Block * falseDest, ValueRange falseOperands, Optional<std::pair<uint32_t, uint32_t>> weights ) {
        ElementsAttr weightsAttr;
        if (weights) {
          weightsAttr =
              odsBuilder.getI32VectorAttr({static_cast<int32_t>(weights->first),
                                       static_cast<int32_t>(weights->second)});
        }
        build(odsBuilder, odsState, condition, trueOperands, falseOperands, weightsAttr,
              trueDest, falseDest);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block * trueDest, Block * falseDest, ValueRange falseOperands ) {
      build(odsBuilder, odsState, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  if (branch_weights) {
  odsState.addAttribute("branch_weights", branch_weights);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  if (branch_weights) {
  odsState.addAttribute("branch_weights", branch_weights);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CondBrOp::verify() {
  if (failed(CondBrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
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



::mlir::ParseResult CondBrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType conditionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> conditionOperands(conditionRawOperands);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::ElementsAttr branch_weightsAttr;
  ::mlir::Block *trueDestSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> trueDestOperandsOperands;
  ::llvm::SMLoc trueDestOperandsOperandsLoc;
  (void)trueDestOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> trueDestOperandsTypes;
  ::mlir::Block *falseDestSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> falseDestOperandsOperands;
  ::llvm::SMLoc falseDestOperandsOperandsLoc;
  (void)falseDestOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> falseDestOperandsTypes;

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("weights"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(branch_weightsAttr, "branch_weights", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(trueDestSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(trueDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(falseDestSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(falseDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}));
  return ::mlir::success();
}

void CondBrOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.cond_br";
  p << ' ';
  p << condition();
  if ((*this)->getAttr("branch_weights")) {
  p << ' ' << "weights";
  p << "(";
  p.printAttribute(branch_weightsAttr());
  p << ")";
  }
  p << ",";
  p << ' ';
  p << trueDest();
  if (!trueDestOperands().empty()) {
  p << "(";
  p << trueDestOperands();
  p << ' ' << ":";
  p << ' ';
  p << trueDestOperands().getTypes();
  p << ")";
  }
  p << ",";
  p << ' ';
  p << falseDest();
  if (!falseDestOperands().empty()) {
  p << "(";
  p << falseDestOperands();
  p << ' ' << ":";
  p << ' ';
  p << falseDestOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "branch_weights"});
}

void CondBrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConstantOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstantOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ConstantOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute ConstantOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").cast<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'llvm.mlir.constant' op ""requires attribute 'value'");
    if (!((true))) return emitError(loc, "'llvm.mlir.constant' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantOp::res() {
  return *getODSResults(0).begin();
}

::mlir::Attribute ConstantOp::valueAttr() {
  return (*this)->getAttr("value").template cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::valueAttr(::mlir::Attribute attr) {
  (*this)->setAttr("value", attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(res);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verify() {
  if (failed(ConstantOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(valueAttr, "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void ConstantOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.mlir.constant";
  p << "(";
  p.printAttribute(valueAttr());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void ConstantOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CopySignOp definitions
//===----------------------------------------------------------------------===//

CopySignOpAdaptor::CopySignOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CopySignOpAdaptor::CopySignOpAdaptor(CopySignOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CopySignOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CopySignOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CopySignOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopySignOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CopySignOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CopySignOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CopySignOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopySignOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopySignOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CopySignOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CopySignOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CopySignOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopySignOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOp::res() {
  return *getODSResults(0).begin();
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult CopySignOp::verify() {
  if (failed(CopySignOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void CopySignOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroBeginOp definitions
//===----------------------------------------------------------------------===//

CoroBeginOpAdaptor::CoroBeginOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroBeginOpAdaptor::CoroBeginOpAdaptor(CoroBeginOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroBeginOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroBeginOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroBeginOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroBeginOpAdaptor::token() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroBeginOpAdaptor::mem() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroBeginOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroBeginOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroBeginOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroBeginOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroBeginOp::token() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroBeginOp::mem() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroBeginOp::tokenMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroBeginOp::memMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroBeginOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroBeginOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroBeginOp::res() {
  return *getODSResults(0).begin();
}

void CoroBeginOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value token, ::mlir::Value mem) {
  odsState.addOperands(token);
  odsState.addOperands(mem);
  odsState.addTypes(res);
}

void CoroBeginOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value mem) {
  odsState.addOperands(token);
  odsState.addOperands(mem);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroBeginOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroBeginOp::verify() {
  if (failed(CoroBeginOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroBeginOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tokenRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tokenOperands(tokenRawOperands);  ::llvm::SMLoc tokenOperandsLoc;
  (void)tokenOperandsLoc;
  ::mlir::OpAsmParser::OperandType memRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memOperands(memRawOperands);  ::llvm::SMLoc memOperandsLoc;
  (void)memOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  tokenOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tokenRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  memOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  result.addTypes(resTypes);
  if (parser.resolveOperands(tokenOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroBeginOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.begin";
  p << ' ';
  p << token();
  p << ",";
  p << ' ';
  p << mem();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroEndOp definitions
//===----------------------------------------------------------------------===//

CoroEndOpAdaptor::CoroEndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroEndOpAdaptor::CoroEndOpAdaptor(CoroEndOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroEndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroEndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroEndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroEndOpAdaptor::handle() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroEndOpAdaptor::unwind() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroEndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroEndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroEndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroEndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroEndOp::handle() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroEndOp::unwind() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroEndOp::handleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroEndOp::unwindMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroEndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroEndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroEndOp::res() {
  return *getODSResults(0).begin();
}

void CoroEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value handle, ::mlir::Value unwind) {
  odsState.addOperands(handle);
  odsState.addOperands(unwind);
  odsState.addTypes(res);
}

void CoroEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value handle, ::mlir::Value unwind) {
  odsState.addOperands(handle);
  odsState.addOperands(unwind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroEndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroEndOp::verify() {
  if (failed(CoroEndOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroEndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::OpAsmParser::OperandType unwindRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> unwindOperands(unwindRawOperands);  ::llvm::SMLoc unwindOperandsLoc;
  (void)unwindOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  unwindOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(unwindRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(resTypes);
  if (parser.resolveOperands(handleOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(unwindOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroEndOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.end";
  p << ' ';
  p << handle();
  p << ",";
  p << ' ';
  p << unwind();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroFreeOp definitions
//===----------------------------------------------------------------------===//

CoroFreeOpAdaptor::CoroFreeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroFreeOpAdaptor::CoroFreeOpAdaptor(CoroFreeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroFreeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroFreeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroFreeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroFreeOpAdaptor::id() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroFreeOpAdaptor::handle() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroFreeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroFreeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroFreeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroFreeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroFreeOp::id() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroFreeOp::handle() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroFreeOp::idMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroFreeOp::handleMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroFreeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroFreeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroFreeOp::res() {
  return *getODSResults(0).begin();
}

void CoroFreeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value id, ::mlir::Value handle) {
  odsState.addOperands(id);
  odsState.addOperands(handle);
  odsState.addTypes(res);
}

void CoroFreeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value id, ::mlir::Value handle) {
  odsState.addOperands(id);
  odsState.addOperands(handle);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroFreeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroFreeOp::verify() {
  if (failed(CoroFreeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroFreeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType idRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> idOperands(idRawOperands);  ::llvm::SMLoc idOperandsLoc;
  (void)idOperandsLoc;
  ::mlir::OpAsmParser::OperandType handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  idOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(idRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  result.addTypes(resTypes);
  if (parser.resolveOperands(idOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(handleOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroFreeOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.free";
  p << ' ';
  p << id();
  p << ",";
  p << ' ';
  p << handle();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroIdOp definitions
//===----------------------------------------------------------------------===//

CoroIdOpAdaptor::CoroIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroIdOpAdaptor::CoroIdOpAdaptor(CoroIdOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroIdOpAdaptor::align() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroIdOpAdaptor::promise() {
  return *getODSOperands(1).begin();
}

::mlir::Value CoroIdOpAdaptor::coroaddr() {
  return *getODSOperands(2).begin();
}

::mlir::Value CoroIdOpAdaptor::fnaddrs() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr CoroIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroIdOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroIdOp::align() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroIdOp::promise() {
  return *getODSOperands(1).begin();
}

::mlir::Value CoroIdOp::coroaddr() {
  return *getODSOperands(2).begin();
}

::mlir::Value CoroIdOp::fnaddrs() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange CoroIdOp::alignMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroIdOp::promiseMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroIdOp::coroaddrMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroIdOp::fnaddrsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroIdOp::res() {
  return *getODSResults(0).begin();
}

void CoroIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value align, ::mlir::Value promise, ::mlir::Value coroaddr, ::mlir::Value fnaddrs) {
  odsState.addOperands(align);
  odsState.addOperands(promise);
  odsState.addOperands(coroaddr);
  odsState.addOperands(fnaddrs);
  odsState.addTypes(res);
}

void CoroIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value align, ::mlir::Value promise, ::mlir::Value coroaddr, ::mlir::Value fnaddrs) {
  odsState.addOperands(align);
  odsState.addOperands(promise);
  odsState.addOperands(coroaddr);
  odsState.addOperands(fnaddrs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroIdOp::verify() {
  if (failed(CoroIdOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroIdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType alignRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> alignOperands(alignRawOperands);  ::llvm::SMLoc alignOperandsLoc;
  (void)alignOperandsLoc;
  ::mlir::OpAsmParser::OperandType promiseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> promiseOperands(promiseRawOperands);  ::llvm::SMLoc promiseOperandsLoc;
  (void)promiseOperandsLoc;
  ::mlir::OpAsmParser::OperandType coroaddrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> coroaddrOperands(coroaddrRawOperands);  ::llvm::SMLoc coroaddrOperandsLoc;
  (void)coroaddrOperandsLoc;
  ::mlir::OpAsmParser::OperandType fnaddrsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> fnaddrsOperands(fnaddrsRawOperands);  ::llvm::SMLoc fnaddrsOperandsLoc;
  (void)fnaddrsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  alignOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(alignRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  promiseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(promiseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  coroaddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(coroaddrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  fnaddrsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(fnaddrsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  result.addTypes(resTypes);
  if (parser.resolveOperands(alignOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(promiseOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coroaddrOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(fnaddrsOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroIdOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.id";
  p << ' ';
  p << align();
  p << ",";
  p << ' ';
  p << promise();
  p << ",";
  p << ' ';
  p << coroaddr();
  p << ",";
  p << ' ';
  p << fnaddrs();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroResumeOp definitions
//===----------------------------------------------------------------------===//

CoroResumeOpAdaptor::CoroResumeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroResumeOpAdaptor::CoroResumeOpAdaptor(CoroResumeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroResumeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroResumeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroResumeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroResumeOpAdaptor::handle() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoroResumeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroResumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroResumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroResumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroResumeOp::handle() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoroResumeOp::handleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroResumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroResumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CoroResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value handle) {
  odsState.addOperands(handle);
}

void CoroResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value handle) {
  odsState.addOperands(handle);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroResumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroResumeOp::verify() {
  if (failed(CoroResumeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroResumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  if (parser.resolveOperands(handleOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroResumeOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.resume";
  p << ' ';
  p << handle();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroSaveOp definitions
//===----------------------------------------------------------------------===//

CoroSaveOpAdaptor::CoroSaveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroSaveOpAdaptor::CoroSaveOpAdaptor(CoroSaveOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroSaveOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroSaveOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroSaveOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSaveOpAdaptor::handle() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoroSaveOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroSaveOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroSaveOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroSaveOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSaveOp::handle() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoroSaveOp::handleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroSaveOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroSaveOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSaveOp::res() {
  return *getODSResults(0).begin();
}

void CoroSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value handle) {
  odsState.addOperands(handle);
  odsState.addTypes(res);
}

void CoroSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value handle) {
  odsState.addOperands(handle);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroSaveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroSaveOp::verify() {
  if (failed(CoroSaveOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroSaveOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  result.addTypes(resTypes);
  if (parser.resolveOperands(handleOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroSaveOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.save";
  p << ' ';
  p << handle();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroSizeOp definitions
//===----------------------------------------------------------------------===//

CoroSizeOpAdaptor::CoroSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroSizeOpAdaptor::CoroSizeOpAdaptor(CoroSizeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroSizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroSizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroSizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CoroSizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroSizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroSizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroSizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CoroSizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroSizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSizeOp::res() {
  return *getODSResults(0).begin();
}

void CoroSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void CoroSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroSizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroSizeOp::verify() {
  if (failed(CoroSizeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroSizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void CoroSizeOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.size";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroSuspendOp definitions
//===----------------------------------------------------------------------===//

CoroSuspendOpAdaptor::CoroSuspendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CoroSuspendOpAdaptor::CoroSuspendOpAdaptor(CoroSuspendOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CoroSuspendOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroSuspendOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroSuspendOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSuspendOpAdaptor::save() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroSuspendOpAdaptor::final() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroSuspendOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroSuspendOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CoroSuspendOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroSuspendOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSuspendOp::save() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroSuspendOp::final() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroSuspendOp::saveMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CoroSuspendOp::finalMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CoroSuspendOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroSuspendOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSuspendOp::res() {
  return *getODSResults(0).begin();
}

void CoroSuspendOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value save, ::mlir::Value final) {
  odsState.addOperands(save);
  odsState.addOperands(final);
  odsState.addTypes(res);
}

void CoroSuspendOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value save, ::mlir::Value final) {
  odsState.addOperands(save);
  odsState.addOperands(final);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroSuspendOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroSuspendOp::verify() {
  if (failed(CoroSuspendOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CoroSuspendOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType saveRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> saveOperands(saveRawOperands);  ::llvm::SMLoc saveOperandsLoc;
  (void)saveOperandsLoc;
  ::mlir::OpAsmParser::OperandType finalRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> finalOperands(finalRawOperands);  ::llvm::SMLoc finalOperandsLoc;
  (void)finalOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  saveOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(saveRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  finalOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(finalRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(resTypes);
  if (parser.resolveOperands(saveOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(finalOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroSuspendOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.coro.suspend";
  p << ' ';
  p << save();
  p << ",";
  p << ' ';
  p << final();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CosOp definitions
//===----------------------------------------------------------------------===//

CosOpAdaptor::CosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CosOpAdaptor::CosOpAdaptor(CosOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CosOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CosOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CosOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOp::res() {
  return *getODSResults(0).begin();
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult CosOp::verify() {
  if (failed(CosOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void CosOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CtPopOp definitions
//===----------------------------------------------------------------------===//

CtPopOpAdaptor::CtPopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CtPopOpAdaptor::CtPopOpAdaptor(CtPopOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CtPopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CtPopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CtPopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CtPopOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CtPopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CtPopOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CtPopOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CtPopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CtPopOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CtPopOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CtPopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CtPopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CtPopOp::res() {
  return *getODSResults(0).begin();
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CtPopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult CtPopOp::verify() {
  if (failed(CtPopOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void CtPopOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::DialectCastOp definitions
//===----------------------------------------------------------------------===//

DialectCastOpAdaptor::DialectCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DialectCastOpAdaptor::DialectCastOpAdaptor(DialectCastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DialectCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DialectCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DialectCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DialectCastOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr DialectCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DialectCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> DialectCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DialectCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DialectCastOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DialectCastOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DialectCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DialectCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DialectCastOp::res() {
  return *getODSResults(0).begin();
}

void DialectCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void DialectCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DialectCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DialectCastOp::verify() {
  if (failed(DialectCastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps16(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps16(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult DialectCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inOperands(inRawOperands);  ::llvm::SMLoc inOperandsLoc;
  (void)inOperandsLoc;
  ::mlir::Type inRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inTypes(inRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  inOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(inOperands, inTypes, inOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DialectCastOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.mlir.cast";
  p << ' ';
  p << in();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(in().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void DialectCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Exp2Op definitions
//===----------------------------------------------------------------------===//

Exp2OpAdaptor::Exp2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Exp2OpAdaptor::Exp2OpAdaptor(Exp2Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Exp2OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Exp2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Exp2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2OpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr Exp2OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Exp2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Exp2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Exp2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2Op::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Exp2Op::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Exp2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Exp2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2Op::res() {
  return *getODSResults(0).begin();
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult Exp2Op::verify() {
  if (failed(Exp2OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void Exp2Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExpOp definitions
//===----------------------------------------------------------------------===//

ExpOpAdaptor::ExpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ExpOpAdaptor::ExpOpAdaptor(ExpOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ExpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ExpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExpOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOp::res() {
  return *getODSResults(0).begin();
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ExpOp::verify() {
  if (failed(ExpOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ExpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExtractElementOp definitions
//===----------------------------------------------------------------------===//

ExtractElementOpAdaptor::ExtractElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ExtractElementOpAdaptor::ExtractElementOpAdaptor(ExtractElementOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ExtractElementOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOpAdaptor::position() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ExtractElementOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExtractElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ExtractElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOp::position() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ExtractElementOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ExtractElementOp::positionMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::res() {
  return *getODSResults(0).begin();
}



void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);
  odsState.addTypes(res);
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ExtractElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseExtractElementOp(parser, result);
}

void ExtractElementOp::print(::mlir::OpAsmPrinter &p) {
  printExtractElementOp(p, *this);
}

::mlir::LogicalResult ExtractElementOp::verify() {
  if (failed(ExtractElementOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ExtractElementOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExtractValueOp definitions
//===----------------------------------------------------------------------===//

ExtractValueOpAdaptor::ExtractValueOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ExtractValueOpAdaptor::ExtractValueOpAdaptor(ExtractValueOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ExtractValueOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractValueOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractValueOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOpAdaptor::container() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExtractValueOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExtractValueOpAdaptor::position() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("position").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ExtractValueOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_position = odsAttrs.get("position");
  if (!tblgen_position) return emitError(loc, "'llvm.extractvalue' op ""requires attribute 'position'");
    if (!((tblgen_position.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'llvm.extractvalue' op ""attribute 'position' failed to satisfy constraint: array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ExtractValueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractValueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOp::container() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExtractValueOp::containerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExtractValueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractValueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ExtractValueOp::positionAttr() {
  return (*this)->getAttr("position").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractValueOp::position() {
  auto attr = positionAttr();
  return attr;
}

void ExtractValueOp::positionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("position", attr);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addAttribute("position", position);
  odsState.addTypes(res);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addAttribute("position", position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractValueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ExtractValueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseExtractValueOp(parser, result);
}

void ExtractValueOp::print(::mlir::OpAsmPrinter &p) {
  printExtractValueOp(p, *this);
}

::mlir::LogicalResult ExtractValueOp::verify() {
  if (failed(ExtractValueOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ExtractValueOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FAbsOp definitions
//===----------------------------------------------------------------------===//

FAbsOpAdaptor::FAbsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FAbsOpAdaptor::FAbsOpAdaptor(FAbsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FAbsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FAbsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FAbsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAbsOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FAbsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FAbsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FAbsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FAbsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAbsOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FAbsOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FAbsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FAbsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAbsOp::res() {
  return *getODSResults(0).begin();
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAbsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FAbsOp::verify() {
  if (failed(FAbsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FAbsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FAddOp definitions
//===----------------------------------------------------------------------===//

FAddOpAdaptor::FAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FAddOpAdaptor::FAddOpAdaptor(FAddOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FAddOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FAddOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FAddOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.fadd' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FAddOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FAddOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FAddOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FAddOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FAddOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FAddOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes({lhs.getType()});

}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FAddOp::verify() {
  if (failed(FAddOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FAddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FAddOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.fadd";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void FAddOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FCeilOp definitions
//===----------------------------------------------------------------------===//

FCeilOpAdaptor::FCeilOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FCeilOpAdaptor::FCeilOpAdaptor(FCeilOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FCeilOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FCeilOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FCeilOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCeilOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FCeilOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FCeilOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FCeilOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FCeilOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCeilOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FCeilOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FCeilOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FCeilOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCeilOp::res() {
  return *getODSResults(0).begin();
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCeilOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FCeilOp::verify() {
  if (failed(FCeilOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FCeilOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FCmpOp definitions
//===----------------------------------------------------------------------===//

FCmpOpAdaptor::FCmpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FCmpOpAdaptor::FCmpOpAdaptor(FCmpOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FCmpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FCmpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FCmpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FCmpOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FCmpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FCmpPredicateAttr FCmpOpAdaptor::predicate() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FCmpPredicateAttr attr = odsAttrs.get("predicate").cast<::mlir::LLVM::FCmpPredicateAttr>();
  return attr;
}

::mlir::LLVM::FMFAttr FCmpOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FCmpOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_predicate = odsAttrs.get("predicate");
  if (!tblgen_predicate) return emitError(loc, "'llvm.fcmp' op ""requires attribute 'predicate'");
    if (!((tblgen_predicate.isa<::mlir::LLVM::FCmpPredicateAttr>()))) return emitError(loc, "'llvm.fcmp' op ""attribute 'predicate' failed to satisfy constraint: llvm.fcmp comparison predicate");
  }
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.fcmp' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FCmpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FCmpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FCmpOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FCmpOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FCmpOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FCmpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FCmpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FCmpPredicateAttr FCmpOp::predicateAttr() {
  return (*this)->getAttr("predicate").template cast<::mlir::LLVM::FCmpPredicateAttr>();
}

::mlir::LLVM::FCmpPredicate FCmpOp::predicate() {
  auto attr = predicateAttr();
  return attr.getValue();
}

::mlir::LLVM::FMFAttr FCmpOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FCmpOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FCmpOp::predicateAttr(::mlir::LLVM::FCmpPredicateAttr attr) {
  (*this)->setAttr("predicate", attr);
}

void FCmpOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FCmpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseCmpOp<FCmpPredicate>(parser, result);
}

void FCmpOp::print(::mlir::OpAsmPrinter &p) {
  printFCmpOp(p, *this);
}

::mlir::LogicalResult FCmpOp::verify() {
  if (failed(FCmpOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FCmpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FDivOp definitions
//===----------------------------------------------------------------------===//

FDivOpAdaptor::FDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FDivOpAdaptor::FDivOpAdaptor(FDivOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FDivOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FDivOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FDivOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.fdiv' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FDivOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FDivOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FDivOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FDivOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FDivOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FDivOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes({lhs.getType()});

}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FDivOp::verify() {
  if (failed(FDivOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FDivOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.fdiv";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void FDivOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FFloorOp definitions
//===----------------------------------------------------------------------===//

FFloorOpAdaptor::FFloorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FFloorOpAdaptor::FFloorOpAdaptor(FFloorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FFloorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FFloorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FFloorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFloorOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FFloorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FFloorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FFloorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FFloorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFloorOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FFloorOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FFloorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FFloorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFloorOp::res() {
  return *getODSResults(0).begin();
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FFloorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FFloorOp::verify() {
  if (failed(FFloorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FFloorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMAOp definitions
//===----------------------------------------------------------------------===//

FMAOpAdaptor::FMAOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FMAOpAdaptor::FMAOpAdaptor(FMAOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FMAOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMAOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMAOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOpAdaptor::c() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr FMAOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FMAOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FMAOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMAOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOp::c() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FMAOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FMAOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FMAOp::cMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FMAOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMAOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOp::res() {
  return *getODSResults(0).begin();
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes(res);
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes({a.getType()});

}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FMAOp::verify() {
  if (failed(FMAOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FMAOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMulAddOp definitions
//===----------------------------------------------------------------------===//

FMulAddOpAdaptor::FMulAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FMulAddOpAdaptor::FMulAddOpAdaptor(FMulAddOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FMulAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMulAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMulAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulAddOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulAddOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMulAddOpAdaptor::c() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr FMulAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FMulAddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FMulAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMulAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulAddOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulAddOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMulAddOp::c() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FMulAddOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FMulAddOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FMulAddOp::cMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FMulAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMulAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulAddOp::res() {
  return *getODSResults(0).begin();
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes(res);
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes({a.getType()});

}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FMulAddOp::verify() {
  if (failed(FMulAddOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FMulAddOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMulOp definitions
//===----------------------------------------------------------------------===//

FMulOpAdaptor::FMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FMulOpAdaptor::FMulOpAdaptor(FMulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FMulOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FMulOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.fmul' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FMulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FMulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FMulOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FMulOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FMulOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes({lhs.getType()});

}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FMulOp::verify() {
  if (failed(FMulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FMulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FMulOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.fmul";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void FMulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FNegOp definitions
//===----------------------------------------------------------------------===//

FNegOpAdaptor::FNegOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FNegOpAdaptor::FNegOpAdaptor(FNegOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FNegOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FNegOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FNegOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FNegOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FNegOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FNegOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.fneg' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FNegOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FNegOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FNegOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FNegOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FNegOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FNegOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FNegOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FNegOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value operand, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes({operand.getType()});

}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FNegOp::verify() {
  if (failed(FNegOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FNegOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(operandOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FNegOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.fneg";
  p << ' ';
  p << operand();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void FNegOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPExtOp definitions
//===----------------------------------------------------------------------===//

FPExtOpAdaptor::FPExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FPExtOpAdaptor::FPExtOpAdaptor(FPExtOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FPExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FPExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPExtOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::res() {
  return *getODSResults(0).begin();
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void FPExtOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPExtOp::verify() {
  if (failed(FPExtOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FPExtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPToSIOp definitions
//===----------------------------------------------------------------------===//

FPToSIOpAdaptor::FPToSIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FPToSIOpAdaptor::FPToSIOpAdaptor(FPToSIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FPToSIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPToSIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPToSIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPToSIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPToSIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FPToSIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToSIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPToSIOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPToSIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToSIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::res() {
  return *getODSResults(0).begin();
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToSIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPToSIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void FPToSIOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPToSIOp::verify() {
  if (failed(FPToSIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FPToSIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPToUIOp definitions
//===----------------------------------------------------------------------===//

FPToUIOpAdaptor::FPToUIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FPToUIOpAdaptor::FPToUIOpAdaptor(FPToUIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FPToUIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPToUIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPToUIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPToUIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPToUIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FPToUIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToUIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPToUIOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPToUIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToUIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::res() {
  return *getODSResults(0).begin();
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToUIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPToUIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void FPToUIOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPToUIOp::verify() {
  if (failed(FPToUIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FPToUIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPTruncOp definitions
//===----------------------------------------------------------------------===//

FPTruncOpAdaptor::FPTruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FPTruncOpAdaptor::FPTruncOpAdaptor(FPTruncOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FPTruncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPTruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPTruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPTruncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPTruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FPTruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPTruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPTruncOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPTruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPTruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::res() {
  return *getODSResults(0).begin();
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPTruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPTruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void FPTruncOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPTruncOp::verify() {
  if (failed(FPTruncOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FPTruncOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FRemOp definitions
//===----------------------------------------------------------------------===//

FRemOpAdaptor::FRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FRemOpAdaptor::FRemOpAdaptor(FRemOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FRemOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FRemOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FRemOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.frem' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FRemOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FRemOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FRemOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FRemOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FRemOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FRemOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes({lhs.getType()});

}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FRemOp::verify() {
  if (failed(FRemOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FRemOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.frem";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void FRemOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FSubOp definitions
//===----------------------------------------------------------------------===//

FSubOpAdaptor::FSubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FSubOpAdaptor::FSubOpAdaptor(FSubOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FSubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FSubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FSubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FSubOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FSubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FSubOpAdaptor::fastmathFlags() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::FMFAttr attr = odsAttrs.get("fastmathFlags").dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
  if (!attr)
    attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
  return attr;
}

::mlir::LogicalResult FSubOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_fastmathFlags = odsAttrs.get("fastmathFlags");
  if (tblgen_fastmathFlags) {
    if (!((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>()))) return emitError(loc, "'llvm.fsub' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FSubOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FSubOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FSubOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FSubOp::fastmathFlagsAttr() {
  return (*this)->getAttr("fastmathFlags").template dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FSubOp::fastmathFlags() {
  auto attr = fastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FSubOp::fastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr("fastmathFlags", attr);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes(res);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("fastmathFlags", fastmathFlags);
  odsState.addTypes({lhs.getType()});

}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FSubOp::verify() {
  if (failed(FSubOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FSubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FSubOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.fsub";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void FSubOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FenceOp definitions
//===----------------------------------------------------------------------===//

FenceOpAdaptor::FenceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FenceOpAdaptor::FenceOpAdaptor(FenceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FenceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FenceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FenceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FenceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicOrderingAttr FenceOpAdaptor::ordering() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::AtomicOrderingAttr attr = odsAttrs.get("ordering").cast<::mlir::LLVM::AtomicOrderingAttr>();
  return attr;
}

::mlir::StringAttr FenceOpAdaptor::syncscope() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("syncscope").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult FenceOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_ordering = odsAttrs.get("ordering");
  if (!tblgen_ordering) return emitError(loc, "'llvm.fence' op ""requires attribute 'ordering'");
    if (!((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>()))) return emitError(loc, "'llvm.fence' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");
  }
  {
  auto tblgen_syncscope = odsAttrs.get("syncscope");
  if (!tblgen_syncscope) return emitError(loc, "'llvm.fence' op ""requires attribute 'syncscope'");
    if (!((tblgen_syncscope.isa<::mlir::StringAttr>()))) return emitError(loc, "'llvm.fence' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FenceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FenceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FenceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FenceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LLVM::AtomicOrderingAttr FenceOp::orderingAttr() {
  return (*this)->getAttr("ordering").template cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering FenceOp::ordering() {
  auto attr = orderingAttr();
  return attr.getValue();
}

::mlir::StringAttr FenceOp::syncscopeAttr() {
  return (*this)->getAttr("syncscope").template cast<::mlir::StringAttr>();
}

::llvm::StringRef FenceOp::syncscope() {
  auto attr = syncscopeAttr();
  return attr.getValue();
}

void FenceOp::orderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr("ordering", attr);
}

void FenceOp::syncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("syncscope", attr);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicOrderingAttr ordering, ::mlir::StringAttr syncscope) {
  odsState.addAttribute("ordering", ordering);
  odsState.addAttribute("syncscope", syncscope);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicOrderingAttr ordering, ::mlir::StringAttr syncscope) {
  odsState.addAttribute("ordering", ordering);
  odsState.addAttribute("syncscope", syncscope);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicOrdering ordering, ::llvm::StringRef syncscope) {
  odsState.addAttribute("ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  odsState.addAttribute("syncscope", odsBuilder.getStringAttr(syncscope));
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicOrdering ordering, ::llvm::StringRef syncscope) {
  odsState.addAttribute("ordering", ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  odsState.addAttribute("syncscope", odsBuilder.getStringAttr(syncscope));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FenceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseFenceOp(parser, result);
}

void FenceOp::print(::mlir::OpAsmPrinter &p) {
  printFenceOp(p, *this);
}

::mlir::LogicalResult FenceOp::verify() {
  if (failed(FenceOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FreezeOp definitions
//===----------------------------------------------------------------------===//

FreezeOpAdaptor::FreezeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FreezeOpAdaptor::FreezeOpAdaptor(FreezeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FreezeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FreezeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FreezeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOpAdaptor::val() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FreezeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FreezeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FreezeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FreezeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOp::val() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FreezeOp::valMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FreezeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FreezeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOp::res() {
  return *getODSResults(0).begin();
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val) {
  odsState.addOperands(val);
  odsState.addTypes(res);
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val) {
  odsState.addOperands(val);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FreezeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value val) {
  odsState.addOperands(val);
  odsState.addTypes({val.getType()});

}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FreezeOp::verify() {
  if (failed(FreezeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult FreezeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valOperands(valRawOperands);  ::llvm::SMLoc valOperandsLoc;
  (void)valOperandsLoc;
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);

  valOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(valTypes);
  if (parser.resolveOperands(valOperands, valTypes, valOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FreezeOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.freeze";
  p << ' ';
  p << val();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(val().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GEPOp definitions
//===----------------------------------------------------------------------===//

GEPOpAdaptor::GEPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GEPOpAdaptor::GEPOpAdaptor(GEPOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GEPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GEPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GEPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange GEPOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GEPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GEPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GEPOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GEPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range GEPOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GEPOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GEPOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GEPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GEPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOp::res() {
  return *getODSResults(0).begin();
}

void GEPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void GEPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value base, ::mlir::ValueRange indices) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addTypes(res);
}

void GEPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GEPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GEPOp::verify() {
  if (failed(GEPOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps21(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GEPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::OperandType>(baseOperands, indicesOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GEPOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.getelementptr";
  p << ' ';
  p << base();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void GEPOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GetActiveLaneMaskOp definitions
//===----------------------------------------------------------------------===//

GetActiveLaneMaskOpAdaptor::GetActiveLaneMaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GetActiveLaneMaskOpAdaptor::GetActiveLaneMaskOpAdaptor(GetActiveLaneMaskOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GetActiveLaneMaskOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetActiveLaneMaskOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetActiveLaneMaskOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetActiveLaneMaskOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::Value GetActiveLaneMaskOpAdaptor::n() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr GetActiveLaneMaskOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetActiveLaneMaskOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GetActiveLaneMaskOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetActiveLaneMaskOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetActiveLaneMaskOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Value GetActiveLaneMaskOp::n() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GetActiveLaneMaskOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GetActiveLaneMaskOp::nMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetActiveLaneMaskOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetActiveLaneMaskOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetActiveLaneMaskOp::res() {
  return *getODSResults(0).begin();
}

void GetActiveLaneMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value base, ::mlir::Value n) {
  odsState.addOperands(base);
  odsState.addOperands(n);
  odsState.addTypes(res);
}

void GetActiveLaneMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::Value n) {
  odsState.addOperands(base);
  odsState.addOperands(n);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetActiveLaneMaskOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetActiveLaneMaskOp::verify() {
  if (failed(GetActiveLaneMaskOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetActiveLaneMaskOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::OpAsmParser::OperandType nRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> nOperands(nRawOperands);  ::llvm::SMLoc nOperandsLoc;
  (void)nOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type nRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> nTypes(nRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  nOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(nRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(baseRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(nRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(nOperands, nTypes, nOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetActiveLaneMaskOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.get.active.lane.mask";
  p << ' ';
  p << base();
  p << ",";
  p << ' ';
  p << n();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(base().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(n().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void GetActiveLaneMaskOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalOp definitions
//===----------------------------------------------------------------------===//

GlobalOpAdaptor::GlobalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GlobalOpAdaptor::GlobalOpAdaptor(GlobalOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GlobalOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GlobalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GlobalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr GlobalOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::UnitAttr GlobalOpAdaptor::constant() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("constant").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::StringAttr GlobalOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LLVM::LinkageAttr GlobalOpAdaptor::linkage() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::LinkageAttr attr = odsAttrs.get("linkage").cast<::mlir::LLVM::LinkageAttr>();
  return attr;
}

::mlir::Attribute GlobalOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").dyn_cast_or_null<::mlir::Attribute>();
  return attr;
}

::mlir::IntegerAttr GlobalOpAdaptor::addr_space() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("addr_space").dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(32), 0);
  return attr;
}

::mlir::RegionRange GlobalOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GlobalOpAdaptor::initializer() {
  return *odsRegions[0];
}

::mlir::LogicalResult GlobalOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_type = odsAttrs.get("type");
  if (!tblgen_type) return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'type'");
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'llvm.mlir.global' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  {
  auto tblgen_constant = odsAttrs.get("constant");
  if (tblgen_constant) {
    if (!((tblgen_constant.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.mlir.global' op ""attribute 'constant' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'llvm.mlir.global' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_linkage = odsAttrs.get("linkage");
  if (!tblgen_linkage) return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'linkage'");
    if (!((tblgen_linkage.isa<::mlir::LLVM::LinkageAttr>()))) return emitError(loc, "'llvm.mlir.global' op ""attribute 'linkage' failed to satisfy constraint: LLVM linkage types");
  }
  {
  auto tblgen_value = odsAttrs.get("value");
  if (tblgen_value) {
    if (!((true))) return emitError(loc, "'llvm.mlir.global' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  }
  {
  auto tblgen_addr_space = odsAttrs.get("addr_space");
  if (tblgen_addr_space) {
    if (!((((tblgen_addr_space.isa<::mlir::IntegerAttr>())) && ((tblgen_addr_space.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_addr_space.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'llvm.mlir.global' op ""attribute 'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GlobalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &GlobalOp::initializer() {
  return (*this)->getRegion(0);
}

::mlir::TypeAttr GlobalOp::typeAttr() {
  return (*this)->getAttr("type").template cast<::mlir::TypeAttr>();
}

::mlir::Type GlobalOp::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOp::constantAttr() {
  return (*this)->getAttr("constant").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::constant() {
  auto attr = constantAttr();
  return attr != nullptr;
}

::mlir::StringAttr GlobalOp::sym_nameAttr() {
  return (*this)->getAttr("sym_name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef GlobalOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::LLVM::LinkageAttr GlobalOp::linkageAttr() {
  return (*this)->getAttr("linkage").template cast<::mlir::LLVM::LinkageAttr>();
}

::mlir::LLVM::Linkage GlobalOp::linkage() {
  auto attr = linkageAttr();
  return attr.getValue();
}

::mlir::Attribute GlobalOp::valueAttr() {
  return (*this)->getAttr("value").template dyn_cast_or_null<::mlir::Attribute>();
}

::llvm::Optional<::mlir::Attribute> GlobalOp::value() {
  auto attr = valueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

::mlir::IntegerAttr GlobalOp::addr_spaceAttr() {
  return (*this)->getAttr("addr_space").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint32_t GlobalOp::addr_space() {
  auto attr = addr_spaceAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

void GlobalOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr("type", attr);
}

void GlobalOp::constantAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("constant", attr);
}

void GlobalOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_name", attr);
}

void GlobalOp::linkageAttr(::mlir::LLVM::LinkageAttr attr) {
  (*this)->setAttr("linkage", attr);
}

void GlobalOp::valueAttr(::mlir::Attribute attr) {
  (*this)->setAttr("value", attr);
}

void GlobalOp::addr_spaceAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("addr_space", attr);
}

::mlir::Attribute GlobalOp::removeConstantAttr() {
  return (*this)->removeAttr("constant");
}

::mlir::Attribute GlobalOp::removeValueAttr() {
  return (*this)->removeAttr("value");
}



void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeAttr type, /*optional*/::mlir::UnitAttr constant, ::mlir::StringAttr sym_name, ::mlir::LLVM::LinkageAttr linkage, /*optional*/::mlir::Attribute value, ::mlir::IntegerAttr addr_space) {
  odsState.addAttribute("type", type);
  if (constant) {
  odsState.addAttribute("constant", constant);
  }
  odsState.addAttribute("sym_name", sym_name);
  odsState.addAttribute("linkage", linkage);
  if (value) {
  odsState.addAttribute("value", value);
  }
  odsState.addAttribute("addr_space", addr_space);
  (void)odsState.addRegion();
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr type, /*optional*/::mlir::UnitAttr constant, ::mlir::StringAttr sym_name, ::mlir::LLVM::LinkageAttr linkage, /*optional*/::mlir::Attribute value, ::mlir::IntegerAttr addr_space) {
  odsState.addAttribute("type", type);
  if (constant) {
  odsState.addAttribute("constant", constant);
  }
  odsState.addAttribute("sym_name", sym_name);
  odsState.addAttribute("linkage", linkage);
  if (value) {
  odsState.addAttribute("value", value);
  }
  odsState.addAttribute("addr_space", addr_space);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage, /*optional*/::mlir::Attribute value, uint32_t addr_space) {
  odsState.addAttribute("type", ::mlir::TypeAttr::get(type));
  if (constant) {
  odsState.addAttribute("constant", odsBuilder.getUnitAttr());
  }
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute("linkage", ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), linkage));
  if (value) {
  odsState.addAttribute("value", value);
  }
  odsState.addAttribute("addr_space", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), addr_space));
  (void)odsState.addRegion();
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage, /*optional*/::mlir::Attribute value, uint32_t addr_space) {
  odsState.addAttribute("type", ::mlir::TypeAttr::get(type));
  if (constant) {
  odsState.addAttribute("constant", odsBuilder.getUnitAttr());
  }
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute("linkage", ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), linkage));
  if (value) {
  odsState.addAttribute("value", value);
  }
  odsState.addAttribute("addr_space", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), addr_space));
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult GlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseGlobalOp(parser, result);
}

void GlobalOp::print(::mlir::OpAsmPrinter &p) {
  printGlobalOp(p, *this);
}

::mlir::LogicalResult GlobalOp::verify() {
  if (failed(GlobalOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
        return emitOpError("region #") << index << " ('initializer') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ICmpOp definitions
//===----------------------------------------------------------------------===//

ICmpOpAdaptor::ICmpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ICmpOpAdaptor::ICmpOpAdaptor(ICmpOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ICmpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ICmpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ICmpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ICmpOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ICmpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::ICmpPredicateAttr ICmpOpAdaptor::predicate() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::ICmpPredicateAttr attr = odsAttrs.get("predicate").cast<::mlir::LLVM::ICmpPredicateAttr>();
  return attr;
}

::mlir::LogicalResult ICmpOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_predicate = odsAttrs.get("predicate");
  if (!tblgen_predicate) return emitError(loc, "'llvm.icmp' op ""requires attribute 'predicate'");
    if (!((tblgen_predicate.isa<::mlir::LLVM::ICmpPredicateAttr>()))) return emitError(loc, "'llvm.icmp' op ""attribute 'predicate' failed to satisfy constraint: llvm.icmp comparison predicate");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ICmpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ICmpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ICmpOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ICmpOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ICmpOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ICmpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ICmpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOp::res() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::ICmpPredicateAttr ICmpOp::predicateAttr() {
  return (*this)->getAttr("predicate").template cast<::mlir::LLVM::ICmpPredicateAttr>();
}

::mlir::LLVM::ICmpPredicate ICmpOp::predicate() {
  auto attr = predicateAttr();
  return attr.getValue();
}

void ICmpOp::predicateAttr(::mlir::LLVM::ICmpPredicateAttr attr) {
  (*this)->setAttr("predicate", attr);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ICmpPredicate predicate, Value lhs, Value rhs) {
      build(odsBuilder, odsState, IntegerType::get(lhs.getType().getContext(), 1),
            predicate, lhs, rhs);
    
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addTypes(res);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addTypes(res);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ICmpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseCmpOp<ICmpPredicate>(parser, result);
}

void ICmpOp::print(::mlir::OpAsmPrinter &p) {
  printICmpOp(p, *this);
}

::mlir::LogicalResult ICmpOp::verify() {
  if (failed(ICmpOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ICmpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InlineAsmOp definitions
//===----------------------------------------------------------------------===//

InlineAsmOpAdaptor::InlineAsmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InlineAsmOpAdaptor::InlineAsmOpAdaptor(InlineAsmOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InlineAsmOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InlineAsmOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange InlineAsmOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InlineAsmOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr InlineAsmOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr InlineAsmOpAdaptor::asm_string() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("asm_string").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr InlineAsmOpAdaptor::constraints() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("constraints").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::UnitAttr InlineAsmOpAdaptor::has_side_effects() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("has_side_effects").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr InlineAsmOpAdaptor::is_align_stack() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("is_align_stack").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LLVM::AsmDialectAttr InlineAsmOpAdaptor::asm_dialect() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::AsmDialectAttr attr = odsAttrs.get("asm_dialect").dyn_cast_or_null<::mlir::LLVM::AsmDialectAttr>();
  return attr;
}

::mlir::LogicalResult InlineAsmOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_asm_string = odsAttrs.get("asm_string");
  if (!tblgen_asm_string) return emitError(loc, "'llvm.inline_asm' op ""requires attribute 'asm_string'");
    if (!((tblgen_asm_string.isa<::mlir::StringAttr>()))) return emitError(loc, "'llvm.inline_asm' op ""attribute 'asm_string' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constraints = odsAttrs.get("constraints");
  if (!tblgen_constraints) return emitError(loc, "'llvm.inline_asm' op ""requires attribute 'constraints'");
    if (!((tblgen_constraints.isa<::mlir::StringAttr>()))) return emitError(loc, "'llvm.inline_asm' op ""attribute 'constraints' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_has_side_effects = odsAttrs.get("has_side_effects");
  if (tblgen_has_side_effects) {
    if (!((tblgen_has_side_effects.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.inline_asm' op ""attribute 'has_side_effects' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_is_align_stack = odsAttrs.get("is_align_stack");
  if (tblgen_is_align_stack) {
    if (!((tblgen_is_align_stack.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.inline_asm' op ""attribute 'is_align_stack' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_asm_dialect = odsAttrs.get("asm_dialect");
  if (tblgen_asm_dialect) {
    if (!((tblgen_asm_dialect.isa<::mlir::LLVM::AsmDialectAttr>()))) return emitError(loc, "'llvm.inline_asm' op ""attribute 'asm_dialect' failed to satisfy constraint: ATT (0) or Intel (1) asm dialect");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> InlineAsmOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InlineAsmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InlineAsmOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange InlineAsmOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> InlineAsmOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range InlineAsmOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InlineAsmOp::res() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::StringAttr InlineAsmOp::asm_stringAttr() {
  return (*this)->getAttr("asm_string").template cast<::mlir::StringAttr>();
}

::llvm::StringRef InlineAsmOp::asm_string() {
  auto attr = asm_stringAttr();
  return attr.getValue();
}

::mlir::StringAttr InlineAsmOp::constraintsAttr() {
  return (*this)->getAttr("constraints").template cast<::mlir::StringAttr>();
}

::llvm::StringRef InlineAsmOp::constraints() {
  auto attr = constraintsAttr();
  return attr.getValue();
}

::mlir::UnitAttr InlineAsmOp::has_side_effectsAttr() {
  return (*this)->getAttr("has_side_effects").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool InlineAsmOp::has_side_effects() {
  auto attr = has_side_effectsAttr();
  return attr != nullptr;
}

::mlir::UnitAttr InlineAsmOp::is_align_stackAttr() {
  return (*this)->getAttr("is_align_stack").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool InlineAsmOp::is_align_stack() {
  auto attr = is_align_stackAttr();
  return attr != nullptr;
}

::mlir::LLVM::AsmDialectAttr InlineAsmOp::asm_dialectAttr() {
  return (*this)->getAttr("asm_dialect").template dyn_cast_or_null<::mlir::LLVM::AsmDialectAttr>();
}

::llvm::Optional<::mlir::LLVM::AsmDialect> InlineAsmOp::asm_dialect() {
  auto attr = asm_dialectAttr();
  return attr ? ::llvm::Optional<::mlir::LLVM::AsmDialect>(attr.getValue()) : (::llvm::None);
}

void InlineAsmOp::asm_stringAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("asm_string", attr);
}

void InlineAsmOp::constraintsAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("constraints", attr);
}

void InlineAsmOp::has_side_effectsAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("has_side_effects", attr);
}

void InlineAsmOp::is_align_stackAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("is_align_stack", attr);
}

void InlineAsmOp::asm_dialectAttr(::mlir::LLVM::AsmDialectAttr attr) {
  (*this)->setAttr("asm_dialect", attr);
}

::mlir::Attribute InlineAsmOp::removeHas_side_effectsAttr() {
  return (*this)->removeAttr("has_side_effects");
}

::mlir::Attribute InlineAsmOp::removeIs_align_stackAttr() {
  return (*this)->removeAttr("is_align_stack");
}

::mlir::Attribute InlineAsmOp::removeAsm_dialectAttr() {
  return (*this)->removeAttr("asm_dialect");
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::mlir::StringAttr asm_string, ::mlir::StringAttr constraints, /*optional*/::mlir::UnitAttr has_side_effects, /*optional*/::mlir::UnitAttr is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect) {
  odsState.addOperands(operands);
  odsState.addAttribute("asm_string", asm_string);
  odsState.addAttribute("constraints", constraints);
  if (has_side_effects) {
  odsState.addAttribute("has_side_effects", has_side_effects);
  }
  if (is_align_stack) {
  odsState.addAttribute("is_align_stack", is_align_stack);
  }
  if (asm_dialect) {
  odsState.addAttribute("asm_dialect", asm_dialect);
  }
  if (res)
    odsState.addTypes(res);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::StringAttr asm_string, ::mlir::StringAttr constraints, /*optional*/::mlir::UnitAttr has_side_effects, /*optional*/::mlir::UnitAttr is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect) {
  odsState.addOperands(operands);
  odsState.addAttribute("asm_string", asm_string);
  odsState.addAttribute("constraints", constraints);
  if (has_side_effects) {
  odsState.addAttribute("has_side_effects", has_side_effects);
  }
  if (is_align_stack) {
  odsState.addAttribute("is_align_stack", is_align_stack);
  }
  if (asm_dialect) {
  odsState.addAttribute("asm_dialect", asm_dialect);
  }
  odsState.addTypes(resultTypes);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect) {
  odsState.addOperands(operands);
  odsState.addAttribute("asm_string", odsBuilder.getStringAttr(asm_string));
  odsState.addAttribute("constraints", odsBuilder.getStringAttr(constraints));
  if (has_side_effects) {
  odsState.addAttribute("has_side_effects", odsBuilder.getUnitAttr());
  }
  if (is_align_stack) {
  odsState.addAttribute("is_align_stack", odsBuilder.getUnitAttr());
  }
  if (asm_dialect) {
  odsState.addAttribute("asm_dialect", asm_dialect);
  }
  if (res)
    odsState.addTypes(res);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect) {
  odsState.addOperands(operands);
  odsState.addAttribute("asm_string", odsBuilder.getStringAttr(asm_string));
  odsState.addAttribute("constraints", odsBuilder.getStringAttr(constraints));
  if (has_side_effects) {
  odsState.addAttribute("has_side_effects", odsBuilder.getUnitAttr());
  }
  if (is_align_stack) {
  odsState.addAttribute("is_align_stack", odsBuilder.getUnitAttr());
  }
  if (asm_dialect) {
  odsState.addAttribute("asm_dialect", asm_dialect);
  }
  odsState.addTypes(resultTypes);
}

void InlineAsmOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InlineAsmOp::verify() {
  if (failed(InlineAsmOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps22(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult InlineAsmOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::LLVM::AsmDialectAttr asm_dialectAttr;
  ::mlir::StringAttr asm_stringAttr;
  ::mlir::StringAttr constraintsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  if (succeeded(parser.parseOptionalKeyword("has_side_effects"))) {
    result.addAttribute("has_side_effects", parser.getBuilder().getUnitAttr());
  }
  if (succeeded(parser.parseOptionalKeyword("is_align_stack"))) {
    result.addAttribute("is_align_stack", parser.getBuilder().getUnitAttr());
  }
  if (succeeded(parser.parseOptionalKeyword("asm_dialect"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"att","intel"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "asm_dialect", attrStorage);
      if (parseResult.hasValue()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAsmDialect(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "asm_dialect attribute specification: \"" << attrStr << '"';;

      asm_dialectAttr = ::mlir::LLVM::AsmDialectAttr::get(parser.getBuilder().getContext(), attrOptional.getValue());
      result.addAttribute("asm_dialect", asm_dialectAttr);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseAttribute(asm_stringAttr, parser.getBuilder().getType<::mlir::NoneType>(), "asm_string", result.attributes))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseAttribute(constraintsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constraints", result.attributes))
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InlineAsmOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.inline_asm";
  if ((*this)->getAttr("has_side_effects")) {
  p << ' ' << "has_side_effects";
  }
  if ((*this)->getAttr("is_align_stack")) {
  p << ' ' << "is_align_stack";
  }
  if ((*this)->getAttr("asm_dialect")) {
  p << ' ' << "asm_dialect";
  p << ' ' << "=";
  p << ' ';

  {
    auto caseValue = *asm_dialect();
    auto caseValueStr = stringifyAsmDialect(caseValue);
    p << caseValueStr;
  }
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"has_side_effects", "is_align_stack", "asm_dialect", "asm_string", "constraints"});
  p << ' ';
  p.printAttributeWithoutType(asm_stringAttr());
  p << ",";
  p << ' ';
  p.printAttributeWithoutType(constraintsAttr());
  p << ' ';
  p << getOperation()->getOperands();
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InsertElementOp definitions
//===----------------------------------------------------------------------===//

InsertElementOpAdaptor::InsertElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InsertElementOpAdaptor::InsertElementOpAdaptor(InsertElementOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InsertElementOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InsertElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOpAdaptor::value() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOpAdaptor::position() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr InsertElementOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InsertElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> InsertElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOp::value() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOp::position() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange InsertElementOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange InsertElementOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange InsertElementOp::positionMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> InsertElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::res() {
  return *getODSResults(0).begin();
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);
  odsState.addTypes(res);
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult InsertElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseInsertElementOp(parser, result);
}

void InsertElementOp::print(::mlir::OpAsmPrinter &p) {
  printInsertElementOp(p, *this);
}

::mlir::LogicalResult InsertElementOp::verify() {
  if (failed(InsertElementOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps23(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void InsertElementOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InsertValueOp definitions
//===----------------------------------------------------------------------===//

InsertValueOpAdaptor::InsertValueOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InsertValueOpAdaptor::InsertValueOpAdaptor(InsertValueOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InsertValueOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertValueOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InsertValueOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOpAdaptor::container() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertValueOpAdaptor::value() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr InsertValueOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr InsertValueOpAdaptor::position() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("position").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult InsertValueOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_position = odsAttrs.get("position");
  if (!tblgen_position) return emitError(loc, "'llvm.insertvalue' op ""requires attribute 'position'");
    if (!((tblgen_position.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'llvm.insertvalue' op ""attribute 'position' failed to satisfy constraint: array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> InsertValueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertValueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOp::container() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertValueOp::value() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange InsertValueOp::containerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange InsertValueOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> InsertValueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertValueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InsertValueOp::positionAttr() {
  return (*this)->getAttr("position").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertValueOp::position() {
  auto attr = positionAttr();
  return attr;
}

void InsertValueOp::positionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("position", attr);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value container, Value value, ArrayAttr position) {
      build(odsBuilder, odsState, container.getType(), container, value, position);
    
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::Value value, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute("position", position);
  odsState.addTypes(res);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::Value value, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute("position", position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertValueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult InsertValueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseInsertValueOp(parser, result);
}

void InsertValueOp::print(::mlir::OpAsmPrinter &p) {
  printInsertValueOp(p, *this);
}

::mlir::LogicalResult InsertValueOp::verify() {
  if (failed(InsertValueOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps23(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void InsertValueOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::IntToPtrOp definitions
//===----------------------------------------------------------------------===//

IntToPtrOpAdaptor::IntToPtrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

IntToPtrOpAdaptor::IntToPtrOpAdaptor(IntToPtrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange IntToPtrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IntToPtrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IntToPtrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IntToPtrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IntToPtrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> IntToPtrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IntToPtrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IntToPtrOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IntToPtrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IntToPtrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOp::res() {
  return *getODSResults(0).begin();
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IntToPtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult IntToPtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void IntToPtrOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult IntToPtrOp::verify() {
  if (failed(IntToPtrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void IntToPtrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InvokeOp definitions
//===----------------------------------------------------------------------===//

InvokeOpAdaptor::InvokeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InvokeOpAdaptor::InvokeOpAdaptor(InvokeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InvokeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InvokeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange InvokeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InvokeOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ValueRange InvokeOpAdaptor::normalDestOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange InvokeOpAdaptor::unwindDestOperands() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr InvokeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr InvokeOpAdaptor::callee() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("callee").dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult InvokeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    {
  auto tblgen_callee = odsAttrs.get("callee");
  if (tblgen_callee) {
    if (!((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'llvm.invoke' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> InvokeOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range InvokeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InvokeOp::operands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range InvokeOp::normalDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range InvokeOp::unwindDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InvokeOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange InvokeOp::normalDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange InvokeOp::unwindDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> InvokeOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range InvokeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *InvokeOp::normalDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *InvokeOp::unwindDest() {
  return (*this)->getSuccessor(1);
}

::mlir::FlatSymbolRefAttr InvokeOp::calleeAttr() {
  return (*this)->getAttr("callee").template dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > InvokeOp::callee() {
  auto attr = calleeAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void InvokeOp::calleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr("callee", attr);
}

::mlir::Attribute InvokeOp::removeCalleeAttr() {
  return (*this)->removeAttr("callee");
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange tys, FlatSymbolRefAttr callee, ValueRange ops, Block* normal, ValueRange normalOps, Block* unwind, ValueRange unwindOps) {
      odsState.addAttribute("callee", callee);
      build(odsBuilder, odsState, tys, ops, normal, normalOps, unwind, unwindOps);
    
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange tys, ValueRange ops, Block* normal, ValueRange normalOps, Block* unwind, ValueRange unwindOps) {
      build(odsBuilder, odsState, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange operands, ::mlir::ValueRange normalDestOperands, ::mlir::ValueRange unwindDestOperands, ::mlir::Block *normalDest, ::mlir::Block *unwindDest) {
  odsState.addOperands(operands);
  odsState.addOperands(normalDestOperands);
  odsState.addOperands(unwindDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(normalDestOperands.size()), static_cast<int32_t>(unwindDestOperands.size())}));
  if (callee) {
  odsState.addAttribute("callee", callee);
  }
  odsState.addSuccessors(normalDest);
  odsState.addSuccessors(unwindDest);
  odsState.addTypes(resultType0);
}

void InvokeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult InvokeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseInvokeOp(parser, result);
}

void InvokeOp::print(::mlir::OpAsmPrinter &p) {
  printInvokeOp(p, *this);
}

::mlir::LogicalResult InvokeOp::verify() {
  if (failed(InvokeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}



} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LLVMFuncOp definitions
//===----------------------------------------------------------------------===//

LLVMFuncOpAdaptor::LLVMFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LLVMFuncOpAdaptor::LLVMFuncOpAdaptor(LLVMFuncOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LLVMFuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LLVMFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LLVMFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LLVMFuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::LinkageAttr LLVMFuncOpAdaptor::linkage() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::LLVM::LinkageAttr attr = odsAttrs.get("linkage").dyn_cast_or_null<::mlir::LLVM::LinkageAttr>();
  if (!attr)
    attr = ::mlir::LLVM::LinkageAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), Linkage::External);
  return attr;
}

::mlir::FlatSymbolRefAttr LLVMFuncOpAdaptor::personality() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("personality").dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::ArrayAttr LLVMFuncOpAdaptor::passthrough() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("passthrough").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::RegionRange LLVMFuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &LLVMFuncOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult LLVMFuncOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_linkage = odsAttrs.get("linkage");
  if (tblgen_linkage) {
    if (!((tblgen_linkage.isa<::mlir::LLVM::LinkageAttr>()))) return emitError(loc, "'llvm.func' op ""attribute 'linkage' failed to satisfy constraint: LLVM linkage types");
  }
  }
  {
  auto tblgen_personality = odsAttrs.get("personality");
  if (tblgen_personality) {
    if (!((tblgen_personality.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'llvm.func' op ""attribute 'personality' failed to satisfy constraint: flat symbol reference attribute");
  }
  }
  {
  auto tblgen_passthrough = odsAttrs.get("passthrough");
  if (tblgen_passthrough) {
    if (!((tblgen_passthrough.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'llvm.func' op ""attribute 'passthrough' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LLVMFuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LLVMFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LLVMFuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LLVMFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &LLVMFuncOp::body() {
  return (*this)->getRegion(0);
}

::mlir::LLVM::LinkageAttr LLVMFuncOp::linkageAttr() {
  return (*this)->getAttr("linkage").template dyn_cast_or_null<::mlir::LLVM::LinkageAttr>();
}

::mlir::LLVM::Linkage LLVMFuncOp::linkage() {
  auto attr = linkageAttr();
    if (!attr)
      return ::mlir::LLVM::LinkageAttr::get(::mlir::Builder((*this)->getContext()).getContext(), Linkage::External).getValue();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr LLVMFuncOp::personalityAttr() {
  return (*this)->getAttr("personality").template dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > LLVMFuncOp::personality() {
  auto attr = personalityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr LLVMFuncOp::passthroughAttr() {
  return (*this)->getAttr("passthrough").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LLVMFuncOp::passthrough() {
  auto attr = passthroughAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void LLVMFuncOp::linkageAttr(::mlir::LLVM::LinkageAttr attr) {
  (*this)->setAttr("linkage", attr);
}

void LLVMFuncOp::personalityAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr("personality", attr);
}

void LLVMFuncOp::passthroughAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("passthrough", attr);
}

::mlir::Attribute LLVMFuncOp::removePersonalityAttr() {
  return (*this)->removeAttr("personality");
}

::mlir::Attribute LLVMFuncOp::removePassthroughAttr() {
  return (*this)->removeAttr("passthrough");
}



::mlir::ParseResult LLVMFuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseLLVMFuncOp(parser, result);
}

void LLVMFuncOp::print(::mlir::OpAsmPrinter &p) {
  printLLVMFuncOp(p, *this);
}

::mlir::LogicalResult LLVMFuncOp::verify() {
  if (failed(LLVMFuncOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LShrOp definitions
//===----------------------------------------------------------------------===//

LShrOpAdaptor::LShrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LShrOpAdaptor::LShrOpAdaptor(LShrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LShrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LShrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LShrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value LShrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LShrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value LShrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LShrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LShrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOp::res() {
  return *getODSResults(0).begin();
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult LShrOp::verify() {
  if (failed(LShrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult LShrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LShrOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.lshr";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void LShrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LandingpadOp definitions
//===----------------------------------------------------------------------===//

LandingpadOpAdaptor::LandingpadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LandingpadOpAdaptor::LandingpadOpAdaptor(LandingpadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LandingpadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LandingpadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LandingpadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LandingpadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr LandingpadOpAdaptor::cleanup() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("cleanup").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult LandingpadOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_cleanup = odsAttrs.get("cleanup");
  if (tblgen_cleanup) {
    if (!((tblgen_cleanup.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.landingpad' op ""attribute 'cleanup' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LandingpadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LandingpadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LandingpadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LandingpadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LandingpadOp::res() {
  return *getODSResults(0).begin();
}

::mlir::UnitAttr LandingpadOp::cleanupAttr() {
  return (*this)->getAttr("cleanup").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LandingpadOp::cleanup() {
  auto attr = cleanupAttr();
  return attr != nullptr;
}

void LandingpadOp::cleanupAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("cleanup", attr);
}

::mlir::Attribute LandingpadOp::removeCleanupAttr() {
  return (*this)->removeAttr("cleanup");
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, /*optional*/::mlir::UnitAttr cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute("cleanup", cleanup);
  }
  odsState.addTypes(res);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute("cleanup", cleanup);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, /*optional*/bool cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute("cleanup", odsBuilder.getUnitAttr());
  }
  odsState.addTypes(res);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute("cleanup", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LandingpadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult LandingpadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseLandingpadOp(parser, result);
}

void LandingpadOp::print(::mlir::OpAsmPrinter &p) {
  printLandingpadOp(p, *this);
}

::mlir::LogicalResult LandingpadOp::verify() {
  if (failed(LandingpadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LoadOp definitions
//===----------------------------------------------------------------------===//

LoadOpAdaptor::LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LoadOpAdaptor::LoadOpAdaptor(LoadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOpAdaptor::addr() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr LoadOpAdaptor::access_groups() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("access_groups").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::IntegerAttr LoadOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::UnitAttr LoadOpAdaptor::volatile_() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("volatile_").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr LoadOpAdaptor::nontemporal() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("nontemporal").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_access_groups = odsAttrs.get("access_groups");
  if (tblgen_access_groups) {
    if (!(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::SymbolRefAttr>()); })))) return emitError(loc, "'llvm.load' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");
  }
  }
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (tblgen_alignment) {
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'llvm.load' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  {
  auto tblgen_volatile_ = odsAttrs.get("volatile_");
  if (tblgen_volatile_) {
    if (!((tblgen_volatile_.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.load' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_nontemporal = odsAttrs.get("nontemporal");
  if (tblgen_nontemporal) {
    if (!((tblgen_nontemporal.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.load' op ""attribute 'nontemporal' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::addr() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LoadOp::addrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr LoadOp::access_groupsAttr() {
  return (*this)->getAttr("access_groups").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOp::access_groups() {
  auto attr = access_groupsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr LoadOp::alignmentAttr() {
  return (*this)->getAttr("alignment").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> LoadOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr LoadOp::volatile_Attr() {
  return (*this)->getAttr("volatile_").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::volatile_() {
  auto attr = volatile_Attr();
  return attr != nullptr;
}

::mlir::UnitAttr LoadOp::nontemporalAttr() {
  return (*this)->getAttr("nontemporal").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::nontemporal() {
  auto attr = nontemporalAttr();
  return attr != nullptr;
}

void LoadOp::access_groupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("access_groups", attr);
}

void LoadOp::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

void LoadOp::volatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr("volatile_", attr);
}

void LoadOp::nontemporalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("nontemporal", attr);
}

::mlir::Attribute LoadOp::removeAccess_groupsAttr() {
  return (*this)->removeAttr("access_groups");
}

::mlir::Attribute LoadOp::removeAlignmentAttr() {
  return (*this)->removeAttr("alignment");
}

::mlir::Attribute LoadOp::removeVolatile_Attr() {
  return (*this)->removeAttr("volatile_");
}

::mlir::Attribute LoadOp::removeNontemporalAttr() {
  return (*this)->removeAttr("nontemporal");
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value addr, unsigned alignment ,  bool isVolatile ,  bool isNonTemporal ) {
      auto type = addr.getType().cast<LLVMPointerType>().getElementType();
      build(odsBuilder, odsState, type, addr, alignment, isVolatile, isNonTemporal);
    
}



void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", nontemporal);
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", nontemporal);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", odsBuilder.getUnitAttr());
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseLoadOp(parser, result);
}

void LoadOp::print(::mlir::OpAsmPrinter &p) {
  printLoadOp(p, *this);
}

::mlir::LogicalResult LoadOp::verify() {
  if (failed(LoadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps24(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Log10Op definitions
//===----------------------------------------------------------------------===//

Log10OpAdaptor::Log10OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Log10OpAdaptor::Log10OpAdaptor(Log10Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Log10OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Log10OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Log10OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10OpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr Log10OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Log10OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Log10Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Log10Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10Op::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Log10Op::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Log10Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Log10Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10Op::res() {
  return *getODSResults(0).begin();
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult Log10Op::verify() {
  if (failed(Log10OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void Log10Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Log2Op definitions
//===----------------------------------------------------------------------===//

Log2OpAdaptor::Log2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Log2OpAdaptor::Log2OpAdaptor(Log2Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Log2OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Log2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Log2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2OpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr Log2OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Log2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Log2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Log2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2Op::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Log2Op::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Log2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Log2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2Op::res() {
  return *getODSResults(0).begin();
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult Log2Op::verify() {
  if (failed(Log2OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void Log2Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LogOp definitions
//===----------------------------------------------------------------------===//

LogOpAdaptor::LogOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogOpAdaptor::LogOpAdaptor(LogOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LogOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOp::res() {
  return *getODSResults(0).begin();
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult LogOp::verify() {
  if (failed(LogOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaskedLoadOp definitions
//===----------------------------------------------------------------------===//

MaskedLoadOpAdaptor::MaskedLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskedLoadOpAdaptor::MaskedLoadOpAdaptor(MaskedLoadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskedLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskedLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MaskedLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOpAdaptor::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedLoadOpAdaptor::mask() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange MaskedLoadOpAdaptor::pass_thru() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr MaskedLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MaskedLoadOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MaskedLoadOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (!tblgen_alignment) return emitError(loc, "'llvm.intr.masked.load' op ""requires attribute 'alignment'");
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.masked.load' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskedLoadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range MaskedLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOp::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedLoadOp::mask() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range MaskedLoadOp::pass_thru() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange MaskedLoadOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskedLoadOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskedLoadOp::pass_thruMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskedLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskedLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MaskedLoadOp::alignmentAttr() {
  return (*this)->getAttr("alignment").template cast<::mlir::IntegerAttr>();
}

uint32_t MaskedLoadOp::alignment() {
  auto attr = alignmentAttr();
  return attr.getValue().getZExtValue();
}

void MaskedLoadOp::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", alignment);
  odsState.addTypes(res);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", alignment);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  odsState.addTypes(res);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskedLoadOp::verify() {
  if (failed(MaskedLoadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MaskedLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskedLoadOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.masked.load";
  p << ' ';
  p << getOperation()->getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaskedStoreOp definitions
//===----------------------------------------------------------------------===//

MaskedStoreOpAdaptor::MaskedStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaskedStoreOpAdaptor::MaskedStoreOpAdaptor(MaskedStoreOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaskedStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskedStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskedStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedStoreOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedStoreOpAdaptor::data() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskedStoreOpAdaptor::mask() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MaskedStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MaskedStoreOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MaskedStoreOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (!tblgen_alignment) return emitError(loc, "'llvm.intr.masked.store' op ""requires attribute 'alignment'");
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.masked.store' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaskedStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskedStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedStoreOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedStoreOp::data() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskedStoreOp::mask() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MaskedStoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskedStoreOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskedStoreOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskedStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskedStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr MaskedStoreOp::alignmentAttr() {
  return (*this)->getAttr("alignment").template cast<::mlir::IntegerAttr>();
}

uint32_t MaskedStoreOp::alignment() {
  auto attr = alignmentAttr();
  return attr.getValue().getZExtValue();
}

void MaskedStoreOp::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", alignment);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskedStoreOp::verify() {
  if (failed(MaskedStoreOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult MaskedStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType dataRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> dataOperands(dataRawOperands);  ::llvm::SMLoc dataOperandsLoc;
  (void)dataOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type dataRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dataTypes(dataRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dataOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dataRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  if (parser.parseType(dataRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataOperands, dataTypes, dataOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskedStoreOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.masked.store";
  p << ' ';
  p << value();
  p << ",";
  p << ' ';
  p << data();
  p << ",";
  p << ' ';
  p << mask();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ' ' << "into";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(data().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixColumnMajorLoadOp definitions
//===----------------------------------------------------------------------===//

MatrixColumnMajorLoadOpAdaptor::MatrixColumnMajorLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatrixColumnMajorLoadOpAdaptor::MatrixColumnMajorLoadOpAdaptor(MatrixColumnMajorLoadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatrixColumnMajorLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixColumnMajorLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixColumnMajorLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorLoadOpAdaptor::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorLoadOpAdaptor::stride() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MatrixColumnMajorLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixColumnMajorLoadOpAdaptor::isVolatile() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("isVolatile").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixColumnMajorLoadOpAdaptor::rows() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rows").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixColumnMajorLoadOpAdaptor::columns() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("columns").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MatrixColumnMajorLoadOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_isVolatile = odsAttrs.get("isVolatile");
  if (!tblgen_isVolatile) return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""requires attribute 'isVolatile'");
    if (!(((tblgen_isVolatile.isa<::mlir::IntegerAttr>())) && ((tblgen_isVolatile.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(1))))) return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""attribute 'isVolatile' failed to satisfy constraint: 1-bit signless integer attribute");
  }
  {
  auto tblgen_rows = odsAttrs.get("rows");
  if (!tblgen_rows) return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""requires attribute 'rows'");
    if (!(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_columns = odsAttrs.get("columns");
  if (!tblgen_columns) return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""requires attribute 'columns'");
    if (!(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MatrixColumnMajorLoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixColumnMajorLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorLoadOp::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorLoadOp::stride() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatrixColumnMajorLoadOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MatrixColumnMajorLoadOp::strideMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MatrixColumnMajorLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixColumnMajorLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorLoadOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOp::isVolatileAttr() {
  return (*this)->getAttr("isVolatile").template cast<::mlir::IntegerAttr>();
}

bool MatrixColumnMajorLoadOp::isVolatile() {
  auto attr = isVolatileAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOp::rowsAttr() {
  return (*this)->getAttr("rows").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorLoadOp::rows() {
  auto attr = rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOp::columnsAttr() {
  return (*this)->getAttr("columns").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorLoadOp::columns() {
  auto attr = columnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixColumnMajorLoadOp::isVolatileAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("isVolatile", attr);
}

void MatrixColumnMajorLoadOp::rowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("rows", attr);
}

void MatrixColumnMajorLoadOp::columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("columns", attr);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", isVolatile);
  odsState.addAttribute("rows", rows);
  odsState.addAttribute("columns", columns);
  odsState.addTypes(res);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", isVolatile);
  odsState.addAttribute("rows", rows);
  odsState.addAttribute("columns", columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute("rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute("columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  odsState.addTypes(res);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute("rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute("columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixColumnMajorLoadOp::verify() {
  if (failed(MatrixColumnMajorLoadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MatrixColumnMajorLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType dataRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> dataOperands(dataRawOperands);  ::llvm::SMLoc dataOperandsLoc;
  (void)dataOperandsLoc;
  ::mlir::OpAsmParser::OperandType strideRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> strideOperands(strideRawOperands);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  ::mlir::Type dataRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dataTypes(dataRawTypes);
  ::mlir::Type strideRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> strideTypes(strideRawTypes);

  dataOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dataRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLess())
    return ::mlir::failure();
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseGreater())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("from"))
    return ::mlir::failure();

  if (parser.parseType(dataRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();

  if (parser.parseType(strideRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(dataOperands, dataTypes, dataOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixColumnMajorLoadOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.matrix.column.major.load";
  p << ' ';
  p << data();
  p << ",";
  p << ' ' << "<";
  p << "stride";
  p << ' ' << "=";
  p << ' ';
  p << stride();
  p << ">";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
  p << ' ' << "from";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(data().getType());
  p << ' ' << "stride";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(stride().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixColumnMajorStoreOp definitions
//===----------------------------------------------------------------------===//

MatrixColumnMajorStoreOpAdaptor::MatrixColumnMajorStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatrixColumnMajorStoreOpAdaptor::MatrixColumnMajorStoreOpAdaptor(MatrixColumnMajorStoreOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatrixColumnMajorStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixColumnMajorStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixColumnMajorStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorStoreOpAdaptor::matrix() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorStoreOpAdaptor::data() {
  return *getODSOperands(1).begin();
}

::mlir::Value MatrixColumnMajorStoreOpAdaptor::stride() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MatrixColumnMajorStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixColumnMajorStoreOpAdaptor::isVolatile() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("isVolatile").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixColumnMajorStoreOpAdaptor::rows() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rows").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixColumnMajorStoreOpAdaptor::columns() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("columns").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MatrixColumnMajorStoreOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_isVolatile = odsAttrs.get("isVolatile");
  if (!tblgen_isVolatile) return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""requires attribute 'isVolatile'");
    if (!(((tblgen_isVolatile.isa<::mlir::IntegerAttr>())) && ((tblgen_isVolatile.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(1))))) return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""attribute 'isVolatile' failed to satisfy constraint: 1-bit signless integer attribute");
  }
  {
  auto tblgen_rows = odsAttrs.get("rows");
  if (!tblgen_rows) return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""requires attribute 'rows'");
    if (!(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_columns = odsAttrs.get("columns");
  if (!tblgen_columns) return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""requires attribute 'columns'");
    if (!(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MatrixColumnMajorStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixColumnMajorStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorStoreOp::matrix() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorStoreOp::data() {
  return *getODSOperands(1).begin();
}

::mlir::Value MatrixColumnMajorStoreOp::stride() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MatrixColumnMajorStoreOp::matrixMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MatrixColumnMajorStoreOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MatrixColumnMajorStoreOp::strideMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MatrixColumnMajorStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixColumnMajorStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr MatrixColumnMajorStoreOp::isVolatileAttr() {
  return (*this)->getAttr("isVolatile").template cast<::mlir::IntegerAttr>();
}

bool MatrixColumnMajorStoreOp::isVolatile() {
  auto attr = isVolatileAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorStoreOp::rowsAttr() {
  return (*this)->getAttr("rows").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorStoreOp::rows() {
  auto attr = rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorStoreOp::columnsAttr() {
  return (*this)->getAttr("columns").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorStoreOp::columns() {
  auto attr = columnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixColumnMajorStoreOp::isVolatileAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("isVolatile", attr);
}

void MatrixColumnMajorStoreOp::rowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("rows", attr);
}

void MatrixColumnMajorStoreOp::columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("columns", attr);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", isVolatile);
  odsState.addAttribute("rows", rows);
  odsState.addAttribute("columns", columns);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", isVolatile);
  odsState.addAttribute("rows", rows);
  odsState.addAttribute("columns", columns);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute("rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute("columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute("isVolatile", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute("rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute("columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixColumnMajorStoreOp::verify() {
  if (failed(MatrixColumnMajorStoreOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult MatrixColumnMajorStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType matrixRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> matrixOperands(matrixRawOperands);  ::llvm::SMLoc matrixOperandsLoc;
  (void)matrixOperandsLoc;
  ::mlir::OpAsmParser::OperandType dataRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> dataOperands(dataRawOperands);  ::llvm::SMLoc dataOperandsLoc;
  (void)dataOperandsLoc;
  ::mlir::OpAsmParser::OperandType strideRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> strideOperands(strideRawOperands);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::mlir::Type matrixRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixTypes(matrixRawTypes);
  ::mlir::Type dataRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dataTypes(dataRawTypes);
  ::mlir::Type strideRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> strideTypes(strideRawTypes);

  matrixOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dataOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dataRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLess())
    return ::mlir::failure();
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseGreater())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(matrixRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(dataRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();

  if (parser.parseType(strideRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(matrixOperands, matrixTypes, matrixOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataOperands, dataTypes, dataOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixColumnMajorStoreOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.matrix.column.major.store";
  p << ' ';
  p << matrix();
  p << ",";
  p << ' ';
  p << data();
  p << ",";
  p << ' ' << "<";
  p << "stride";
  p << ' ' << "=";
  p << ' ';
  p << stride();
  p << ">";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(matrix().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(data().getType());
  p << ' ' << "stride";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(stride().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixMultiplyOp definitions
//===----------------------------------------------------------------------===//

MatrixMultiplyOpAdaptor::MatrixMultiplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatrixMultiplyOpAdaptor::MatrixMultiplyOpAdaptor(MatrixMultiplyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatrixMultiplyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixMultiplyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixMultiplyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixMultiplyOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixMultiplyOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MatrixMultiplyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixMultiplyOpAdaptor::lhs_rows() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("lhs_rows").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixMultiplyOpAdaptor::lhs_columns() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("lhs_columns").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixMultiplyOpAdaptor::rhs_columns() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rhs_columns").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MatrixMultiplyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_lhs_rows = odsAttrs.get("lhs_rows");
  if (!tblgen_lhs_rows) return emitError(loc, "'llvm.intr.matrix.multiply' op ""requires attribute 'lhs_rows'");
    if (!(((tblgen_lhs_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_lhs_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.multiply' op ""attribute 'lhs_rows' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_lhs_columns = odsAttrs.get("lhs_columns");
  if (!tblgen_lhs_columns) return emitError(loc, "'llvm.intr.matrix.multiply' op ""requires attribute 'lhs_columns'");
    if (!(((tblgen_lhs_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_lhs_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.multiply' op ""attribute 'lhs_columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_rhs_columns = odsAttrs.get("rhs_columns");
  if (!tblgen_rhs_columns) return emitError(loc, "'llvm.intr.matrix.multiply' op ""requires attribute 'rhs_columns'");
    if (!(((tblgen_rhs_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_rhs_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.multiply' op ""attribute 'rhs_columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MatrixMultiplyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixMultiplyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixMultiplyOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixMultiplyOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatrixMultiplyOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MatrixMultiplyOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MatrixMultiplyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixMultiplyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixMultiplyOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatrixMultiplyOp::lhs_rowsAttr() {
  return (*this)->getAttr("lhs_rows").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixMultiplyOp::lhs_rows() {
  auto attr = lhs_rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixMultiplyOp::lhs_columnsAttr() {
  return (*this)->getAttr("lhs_columns").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixMultiplyOp::lhs_columns() {
  auto attr = lhs_columnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixMultiplyOp::rhs_columnsAttr() {
  return (*this)->getAttr("rhs_columns").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixMultiplyOp::rhs_columns() {
  auto attr = rhs_columnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixMultiplyOp::lhs_rowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("lhs_rows", attr);
}

void MatrixMultiplyOp::lhs_columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("lhs_columns", attr);
}

void MatrixMultiplyOp::rhs_columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("rhs_columns", attr);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("lhs_rows", lhs_rows);
  odsState.addAttribute("lhs_columns", lhs_columns);
  odsState.addAttribute("rhs_columns", rhs_columns);
  odsState.addTypes(res);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("lhs_rows", lhs_rows);
  odsState.addAttribute("lhs_columns", lhs_columns);
  odsState.addAttribute("rhs_columns", rhs_columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("lhs_rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_rows));
  odsState.addAttribute("lhs_columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_columns));
  odsState.addAttribute("rhs_columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rhs_columns));
  odsState.addTypes(res);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("lhs_rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_rows));
  odsState.addAttribute("lhs_columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_columns));
  odsState.addAttribute("rhs_columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rhs_columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixMultiplyOp::verify() {
  if (failed(MatrixMultiplyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MatrixMultiplyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixMultiplyOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.matrix.multiply";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ' << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << ' ' << "->";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixTransposeOp definitions
//===----------------------------------------------------------------------===//

MatrixTransposeOpAdaptor::MatrixTransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatrixTransposeOpAdaptor::MatrixTransposeOpAdaptor(MatrixTransposeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatrixTransposeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixTransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixTransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixTransposeOpAdaptor::matrix() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MatrixTransposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixTransposeOpAdaptor::rows() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rows").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr MatrixTransposeOpAdaptor::columns() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("columns").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MatrixTransposeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_rows = odsAttrs.get("rows");
  if (!tblgen_rows) return emitError(loc, "'llvm.intr.matrix.transpose' op ""requires attribute 'rows'");
    if (!(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.transpose' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_columns = odsAttrs.get("columns");
  if (!tblgen_columns) return emitError(loc, "'llvm.intr.matrix.transpose' op ""requires attribute 'columns'");
    if (!(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.matrix.transpose' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MatrixTransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixTransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixTransposeOp::matrix() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MatrixTransposeOp::matrixMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MatrixTransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixTransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixTransposeOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatrixTransposeOp::rowsAttr() {
  return (*this)->getAttr("rows").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixTransposeOp::rows() {
  auto attr = rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixTransposeOp::columnsAttr() {
  return (*this)->getAttr("columns").template cast<::mlir::IntegerAttr>();
}

uint32_t MatrixTransposeOp::columns() {
  auto attr = columnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixTransposeOp::rowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("rows", attr);
}

void MatrixTransposeOp::columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("columns", attr);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute("rows", rows);
  odsState.addAttribute("columns", columns);
  odsState.addTypes(res);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute("rows", rows);
  odsState.addAttribute("columns", columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute("rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute("columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  odsState.addTypes(res);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute("rows", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute("columns", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixTransposeOp::verify() {
  if (failed(MatrixTransposeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MatrixTransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType matrixRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> matrixOperands(matrixRawOperands);  ::llvm::SMLoc matrixOperandsLoc;
  (void)matrixOperandsLoc;
  ::mlir::Type matrixRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixTypes(matrixRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  matrixOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(matrixRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(matrixOperands, matrixTypes, matrixOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixTransposeOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.matrix.transpose";
  p << ' ';
  p << matrix();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(matrix().getType());
  p << ' ' << "into";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaxNumOp definitions
//===----------------------------------------------------------------------===//

MaxNumOpAdaptor::MaxNumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaxNumOpAdaptor::MaxNumOpAdaptor(MaxNumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaxNumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaxNumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaxNumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxNumOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaxNumOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MaxNumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaxNumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaxNumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaxNumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxNumOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaxNumOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MaxNumOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaxNumOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaxNumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaxNumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxNumOp::res() {
  return *getODSResults(0).begin();
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaxNumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult MaxNumOp::verify() {
  if (failed(MaxNumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MaxNumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaximumOp definitions
//===----------------------------------------------------------------------===//

MaximumOpAdaptor::MaximumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaximumOpAdaptor::MaximumOpAdaptor(MaximumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaximumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaximumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaximumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MaximumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaximumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaximumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaximumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MaximumOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaximumOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaximumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaximumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOp::res() {
  return *getODSResults(0).begin();
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaximumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult MaximumOp::verify() {
  if (failed(MaximumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MaximumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MemcpyInlineOp definitions
//===----------------------------------------------------------------------===//

MemcpyInlineOpAdaptor::MemcpyInlineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MemcpyInlineOpAdaptor::MemcpyInlineOpAdaptor(MemcpyInlineOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MemcpyInlineOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemcpyInlineOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemcpyInlineOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyInlineOpAdaptor::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyInlineOpAdaptor::src() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyInlineOpAdaptor::len() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyInlineOpAdaptor::isVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MemcpyInlineOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemcpyInlineOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MemcpyInlineOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemcpyInlineOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyInlineOp::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyInlineOp::src() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyInlineOp::len() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyInlineOp::isVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MemcpyInlineOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MemcpyInlineOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MemcpyInlineOp::lenMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MemcpyInlineOp::isVolatileMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MemcpyInlineOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemcpyInlineOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemcpyInlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
}

void MemcpyInlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemcpyInlineOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemcpyInlineOp::verify() {
  if (failed(MemcpyInlineOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MemcpyOp definitions
//===----------------------------------------------------------------------===//

MemcpyOpAdaptor::MemcpyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MemcpyOpAdaptor::MemcpyOpAdaptor(MemcpyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MemcpyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemcpyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemcpyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyOpAdaptor::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyOpAdaptor::src() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyOpAdaptor::len() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyOpAdaptor::isVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MemcpyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemcpyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MemcpyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemcpyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyOp::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyOp::src() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyOp::len() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyOp::isVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MemcpyOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MemcpyOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MemcpyOp::lenMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MemcpyOp::isVolatileMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MemcpyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemcpyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemcpyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
}

void MemcpyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemcpyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemcpyOp::verify() {
  if (failed(MemcpyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MetadataOp definitions
//===----------------------------------------------------------------------===//

MetadataOpAdaptor::MetadataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MetadataOpAdaptor::MetadataOpAdaptor(MetadataOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MetadataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MetadataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MetadataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr MetadataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr MetadataOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange MetadataOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MetadataOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult MetadataOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'llvm.metadata' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'llvm.metadata' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> MetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &MetadataOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr MetadataOp::sym_nameAttr() {
  return (*this)->getAttr("sym_name").template cast<::mlir::StringAttr>();
}

::llvm::StringRef MetadataOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

void MetadataOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("sym_name", attr);
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  (void)odsState.addRegion();
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  (void)odsState.addRegion();
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(sym_name));
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MetadataOp::verify() {
  if (failed(MetadataOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
  return ::mlir::success();
}

::mlir::ParseResult MetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void MetadataOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.metadata";
  p << ' ';
  p.printSymbolName(sym_nameAttr().getValue());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
  p << ' ';
  p.printRegion(body());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MinNumOp definitions
//===----------------------------------------------------------------------===//

MinNumOpAdaptor::MinNumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MinNumOpAdaptor::MinNumOpAdaptor(MinNumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MinNumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MinNumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MinNumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinNumOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinNumOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MinNumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MinNumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MinNumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MinNumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinNumOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinNumOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MinNumOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MinNumOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MinNumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MinNumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinNumOp::res() {
  return *getODSResults(0).begin();
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MinNumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult MinNumOp::verify() {
  if (failed(MinNumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MinNumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MinimumOp definitions
//===----------------------------------------------------------------------===//

MinimumOpAdaptor::MinimumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MinimumOpAdaptor::MinimumOpAdaptor(MinimumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MinimumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MinimumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MinimumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MinimumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MinimumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MinimumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MinimumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MinimumOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MinimumOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MinimumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MinimumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOp::res() {
  return *getODSResults(0).begin();
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MinimumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult MinimumOp::verify() {
  if (failed(MinimumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MinimumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MulOp definitions
//===----------------------------------------------------------------------===//

MulOpAdaptor::MulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MulOpAdaptor::MulOpAdaptor(MulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::res() {
  return *getODSResults(0).begin();
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult MulOp::verify() {
  if (failed(MulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MulOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.mul";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void MulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::NullOp definitions
//===----------------------------------------------------------------------===//

NullOpAdaptor::NullOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

NullOpAdaptor::NullOpAdaptor(NullOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange NullOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> NullOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NullOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr NullOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult NullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> NullOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NullOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> NullOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NullOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NullOp::res() {
  return *getODSResults(0).begin();
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NullOp::verify() {
  if (failed(NullOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult NullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void NullOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.mlir.null";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void NullOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::OrOp definitions
//===----------------------------------------------------------------------===//

OrOpAdaptor::OrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

OrOpAdaptor::OrOpAdaptor(OrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange OrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange OrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr OrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> OrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange OrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange OrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> OrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::res() {
  return *getODSResults(0).begin();
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult OrOp::verify() {
  if (failed(OrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult OrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OrOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.or";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void OrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PowOp definitions
//===----------------------------------------------------------------------===//

PowOpAdaptor::PowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PowOpAdaptor::PowOpAdaptor(PowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr PowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PowOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PowOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOp::res() {
  return *getODSResults(0).begin();
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult PowOp::verify() {
  if (failed(PowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void PowOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Prefetch definitions
//===----------------------------------------------------------------------===//

PrefetchAdaptor::PrefetchAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PrefetchAdaptor::PrefetchAdaptor(Prefetch&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PrefetchAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PrefetchAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PrefetchAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrefetchAdaptor::addr() {
  return *getODSOperands(0).begin();
}

::mlir::Value PrefetchAdaptor::rw() {
  return *getODSOperands(1).begin();
}

::mlir::Value PrefetchAdaptor::hint() {
  return *getODSOperands(2).begin();
}

::mlir::Value PrefetchAdaptor::cache() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr PrefetchAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PrefetchAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Prefetch::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Prefetch::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Prefetch::addr() {
  return *getODSOperands(0).begin();
}

::mlir::Value Prefetch::rw() {
  return *getODSOperands(1).begin();
}

::mlir::Value Prefetch::hint() {
  return *getODSOperands(2).begin();
}

::mlir::Value Prefetch::cache() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange Prefetch::addrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Prefetch::rwMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Prefetch::hintMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Prefetch::cacheMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Prefetch::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Prefetch::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Prefetch::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value addr, ::mlir::Value rw, ::mlir::Value hint, ::mlir::Value cache) {
  odsState.addOperands(addr);
  odsState.addOperands(rw);
  odsState.addOperands(hint);
  odsState.addOperands(cache);
}

void Prefetch::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, ::mlir::Value rw, ::mlir::Value hint, ::mlir::Value cache) {
  odsState.addOperands(addr);
  odsState.addOperands(rw);
  odsState.addOperands(hint);
  odsState.addOperands(cache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Prefetch::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Prefetch::verify() {
  if (failed(PrefetchAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PtrToIntOp definitions
//===----------------------------------------------------------------------===//

PtrToIntOpAdaptor::PtrToIntOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PtrToIntOpAdaptor::PtrToIntOpAdaptor(PtrToIntOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PtrToIntOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PtrToIntOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PtrToIntOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PtrToIntOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PtrToIntOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PtrToIntOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PtrToIntOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PtrToIntOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PtrToIntOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PtrToIntOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOp::res() {
  return *getODSResults(0).begin();
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PtrToIntOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult PtrToIntOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void PtrToIntOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult PtrToIntOp::verify() {
  if (failed(PtrToIntOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void PtrToIntOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ResumeOp definitions
//===----------------------------------------------------------------------===//

ResumeOpAdaptor::ResumeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ResumeOpAdaptor::ResumeOpAdaptor(ResumeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ResumeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResumeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResumeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResumeOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResumeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ResumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ResumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResumeOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResumeOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ResumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void ResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResumeOp::verify() {
  if (failed(ResumeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!isa_and_nonnull<LandingpadOp>(value().getDefiningOp()))
      return emitOpError("expects landingpad value as operand");
    // No check for personality of function - landingpad op verifies it.
    return success();
}

::mlir::ParseResult ResumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResumeOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.resume";
  p << ' ';
  p << value();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ReturnOp definitions
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

::mlir::ValueRange ReturnOpAdaptor::args() {
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

::mlir::Operation::operand_range ReturnOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::argsMutable() {
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

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args) {
  odsState.addOperands(args);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseReturnOp(parser, result);
}

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  printReturnOp(p, *this);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (failed(ReturnOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

void ReturnOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SAddWithOverflowOp definitions
//===----------------------------------------------------------------------===//

SAddWithOverflowOpAdaptor::SAddWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SAddWithOverflowOpAdaptor::SAddWithOverflowOpAdaptor(SAddWithOverflowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SAddWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SAddWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SAddWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SAddWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SAddWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SAddWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SAddWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SAddWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SAddWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SAddWithOverflowOp::res() {
  return *getODSResults(0).begin();
}

void SAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void SAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SAddWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SAddWithOverflowOp::verify() {
  if (failed(SAddWithOverflowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SDivOp definitions
//===----------------------------------------------------------------------===//

SDivOpAdaptor::SDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SDivOpAdaptor::SDivOpAdaptor(SDivOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SDivOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SDivOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SDivOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SDivOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOp::res() {
  return *getODSResults(0).begin();
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SDivOp::verify() {
  if (failed(SDivOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult SDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SDivOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.sdiv";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SDivOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SExtOp definitions
//===----------------------------------------------------------------------===//

SExtOpAdaptor::SExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SExtOpAdaptor::SExtOpAdaptor(SExtOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SExtOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOp::res() {
  return *getODSResults(0).begin();
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void SExtOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult SExtOp::verify() {
  if (failed(SExtOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SExtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SIToFPOp definitions
//===----------------------------------------------------------------------===//

SIToFPOpAdaptor::SIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SIToFPOpAdaptor::SIToFPOpAdaptor(SIToFPOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SIToFPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SIToFPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SIToFPOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::res() {
  return *getODSResults(0).begin();
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void SIToFPOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult SIToFPOp::verify() {
  if (failed(SIToFPOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SIToFPOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SMaxOp definitions
//===----------------------------------------------------------------------===//

SMaxOpAdaptor::SMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SMaxOpAdaptor::SMaxOpAdaptor(SMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMaxOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMaxOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SMaxOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMaxOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMaxOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SMaxOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SMaxOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMaxOp::res() {
  return *getODSResults(0).begin();
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SMaxOp::verify() {
  if (failed(SMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SMaxOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SMinOp definitions
//===----------------------------------------------------------------------===//

SMinOpAdaptor::SMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SMinOpAdaptor::SMinOpAdaptor(SMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMinOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMinOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMinOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMinOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SMinOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SMinOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMinOp::res() {
  return *getODSResults(0).begin();
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes({a.getType()});

}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SMinOp::verify() {
  if (failed(SMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SMinOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SMulWithOverflowOp definitions
//===----------------------------------------------------------------------===//

SMulWithOverflowOpAdaptor::SMulWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SMulWithOverflowOpAdaptor::SMulWithOverflowOpAdaptor(SMulWithOverflowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SMulWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SMulWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SMulWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SMulWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SMulWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SMulWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SMulWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SMulWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SMulWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMulWithOverflowOp::res() {
  return *getODSResults(0).begin();
}

void SMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void SMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMulWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SMulWithOverflowOp::verify() {
  if (failed(SMulWithOverflowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SRemOp definitions
//===----------------------------------------------------------------------===//

SRemOpAdaptor::SRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SRemOpAdaptor::SRemOpAdaptor(SRemOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SRemOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SRemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SRemOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SRemOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SRemOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOp::res() {
  return *getODSResults(0).begin();
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SRemOp::verify() {
  if (failed(SRemOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult SRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SRemOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.srem";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SRemOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SSubWithOverflowOp definitions
//===----------------------------------------------------------------------===//

SSubWithOverflowOpAdaptor::SSubWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SSubWithOverflowOpAdaptor::SSubWithOverflowOpAdaptor(SSubWithOverflowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SSubWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SSubWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SSubWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SSubWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SSubWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SSubWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SSubWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SSubWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SSubWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SSubWithOverflowOp::res() {
  return *getODSResults(0).begin();
}

void SSubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void SSubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SSubWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SSubWithOverflowOp::verify() {
  if (failed(SSubWithOverflowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SelectOp definitions
//===----------------------------------------------------------------------===//

SelectOpAdaptor::SelectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SelectOpAdaptor::SelectOpAdaptor(SelectOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SelectOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SelectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SelectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOpAdaptor::condition() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOpAdaptor::trueValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOpAdaptor::falseValue() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr SelectOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SelectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::condition() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOp::trueValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOp::falseValue() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SelectOp::conditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SelectOp::trueValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SelectOp::falseValueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::res() {
  return *getODSResults(0).begin();
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Value lhs, Value rhs) {
      build(odsBuilder, odsState, lhs.getType(), condition, lhs, rhs);
    
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  odsState.addTypes(res);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SelectOp::verify() {
  if (failed(SelectOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {trueValue, falseValue, res} have same type");
  return ::mlir::success();
}

::mlir::ParseResult SelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::Type conditionRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> conditionTypes(conditionRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(conditionRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const Type>(::llvm::ArrayRef<::mlir::Type>(conditionTypes), ::llvm::ArrayRef<::mlir::Type>(resTypes), ::llvm::ArrayRef<::mlir::Type>(resTypes)), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SelectOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.select";
  p << ' ';
  p << getOperation()->getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(condition().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SelectOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ShlOp definitions
//===----------------------------------------------------------------------===//

ShlOpAdaptor::ShlOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ShlOpAdaptor::ShlOpAdaptor(ShlOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ShlOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShlOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShlOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShlOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ShlOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ShlOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ShlOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShlOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShlOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ShlOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ShlOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ShlOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShlOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOp::res() {
  return *getODSResults(0).begin();
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShlOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ShlOp::verify() {
  if (failed(ShlOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ShlOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShlOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.shl";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void ShlOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ShuffleVectorOp definitions
//===----------------------------------------------------------------------===//

ShuffleVectorOpAdaptor::ShuffleVectorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ShuffleVectorOpAdaptor::ShuffleVectorOpAdaptor(ShuffleVectorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ShuffleVectorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShuffleVectorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShuffleVectorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOpAdaptor::v1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleVectorOpAdaptor::v2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ShuffleVectorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ShuffleVectorOpAdaptor::mask() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("mask").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ShuffleVectorOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_mask = odsAttrs.get("mask");
  if (!tblgen_mask) return emitError(loc, "'llvm.shufflevector' op ""requires attribute 'mask'");
    if (!((tblgen_mask.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'llvm.shufflevector' op ""attribute 'mask' failed to satisfy constraint: array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ShuffleVectorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShuffleVectorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOp::v1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleVectorOp::v2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ShuffleVectorOp::v1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ShuffleVectorOp::v2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ShuffleVectorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShuffleVectorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ShuffleVectorOp::maskAttr() {
  return (*this)->getAttr("mask").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ShuffleVectorOp::mask() {
  auto attr = maskAttr();
  return attr;
}

void ShuffleVectorOp::maskAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("mask", attr);
}



void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute("mask", mask);
  odsState.addTypes(res);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute("mask", mask);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ShuffleVectorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseShuffleVectorOp(parser, result);
}

void ShuffleVectorOp::print(::mlir::OpAsmPrinter &p) {
  printShuffleVectorOp(p, *this);
}

::mlir::LogicalResult ShuffleVectorOp::verify() {
  if (failed(ShuffleVectorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  auto type1 = v1().getType();
    auto type2 = v2().getType();
    if (::mlir::LLVM::getVectorElementType(type1) !=
            ::mlir::LLVM::getVectorElementType(type2))
      return emitOpError("expected matching LLVM IR Dialect element types");
    return success();
}

void ShuffleVectorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SinOp definitions
//===----------------------------------------------------------------------===//

SinOpAdaptor::SinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SinOpAdaptor::SinOpAdaptor(SinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SinOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOp::res() {
  return *getODSResults(0).begin();
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SinOp::verify() {
  if (failed(SinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SinOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SqrtOp definitions
//===----------------------------------------------------------------------===//

SqrtOpAdaptor::SqrtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SqrtOpAdaptor::SqrtOpAdaptor(SqrtOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SqrtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SqrtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SqrtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SqrtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SqrtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SqrtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SqrtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SqrtOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SqrtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SqrtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOp::res() {
  return *getODSResults(0).begin();
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes({in.getType()});

}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SqrtOp::verify() {
  if (failed(SqrtOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SqrtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StackRestoreOp definitions
//===----------------------------------------------------------------------===//

StackRestoreOpAdaptor::StackRestoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StackRestoreOpAdaptor::StackRestoreOpAdaptor(StackRestoreOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StackRestoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StackRestoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StackRestoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StackRestoreOpAdaptor::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StackRestoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StackRestoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> StackRestoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StackRestoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StackRestoreOp::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StackRestoreOp::ptrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StackRestoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StackRestoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void StackRestoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr) {
  odsState.addOperands(ptr);
}

void StackRestoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr) {
  odsState.addOperands(ptr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StackRestoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StackRestoreOp::verify() {
  if (failed(StackRestoreOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult StackRestoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8));
  if (parser.resolveOperands(ptrOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StackRestoreOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.stackrestore";
  p << ' ';
  p << ptr();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StackSaveOp definitions
//===----------------------------------------------------------------------===//

StackSaveOpAdaptor::StackSaveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StackSaveOpAdaptor::StackSaveOpAdaptor(StackSaveOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StackSaveOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StackSaveOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StackSaveOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr StackSaveOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StackSaveOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> StackSaveOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StackSaveOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> StackSaveOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StackSaveOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StackSaveOp::res() {
  return *getODSResults(0).begin();
}

void StackSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void StackSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StackSaveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StackSaveOp::verify() {
  if (failed(StackSaveOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult StackSaveOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void StackSaveOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.stacksave";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StoreOp definitions
//===----------------------------------------------------------------------===//

StoreOpAdaptor::StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

StoreOpAdaptor::StoreOpAdaptor(StoreOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange StoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOpAdaptor::addr() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr StoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr StoreOpAdaptor::access_groups() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("access_groups").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::IntegerAttr StoreOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::UnitAttr StoreOpAdaptor::volatile_() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("volatile_").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::UnitAttr StoreOpAdaptor::nontemporal() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("nontemporal").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_access_groups = odsAttrs.get("access_groups");
  if (tblgen_access_groups) {
    if (!(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::SymbolRefAttr>()); })))) return emitError(loc, "'llvm.store' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");
  }
  }
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (tblgen_alignment) {
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'llvm.store' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  {
  auto tblgen_volatile_ = odsAttrs.get("volatile_");
  if (tblgen_volatile_) {
    if (!((tblgen_volatile_.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.store' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_nontemporal = odsAttrs.get("nontemporal");
  if (tblgen_nontemporal) {
    if (!((tblgen_nontemporal.isa<::mlir::UnitAttr>()))) return emitError(loc, "'llvm.store' op ""attribute 'nontemporal' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> StoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOp::addr() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange StoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange StoreOp::addrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr StoreOp::access_groupsAttr() {
  return (*this)->getAttr("access_groups").template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOp::access_groups() {
  auto attr = access_groupsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr StoreOp::alignmentAttr() {
  return (*this)->getAttr("alignment").template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> StoreOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr StoreOp::volatile_Attr() {
  return (*this)->getAttr("volatile_").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool StoreOp::volatile_() {
  auto attr = volatile_Attr();
  return attr != nullptr;
}

::mlir::UnitAttr StoreOp::nontemporalAttr() {
  return (*this)->getAttr("nontemporal").template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool StoreOp::nontemporal() {
  auto attr = nontemporalAttr();
  return attr != nullptr;
}

void StoreOp::access_groupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("access_groups", attr);
}

void StoreOp::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

void StoreOp::volatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr("volatile_", attr);
}

void StoreOp::nontemporalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr("nontemporal", attr);
}

::mlir::Attribute StoreOp::removeAccess_groupsAttr() {
  return (*this)->removeAttr("access_groups");
}

::mlir::Attribute StoreOp::removeAlignmentAttr() {
  return (*this)->removeAttr("alignment");
}

::mlir::Attribute StoreOp::removeVolatile_Attr() {
  return (*this)->removeAttr("volatile_");
}

::mlir::Attribute StoreOp::removeNontemporalAttr() {
  return (*this)->removeAttr("nontemporal");
}



void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", nontemporal);
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", nontemporal);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", odsBuilder.getUnitAttr());
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute("access_groups", access_groups);
  }
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  if (volatile_) {
  odsState.addAttribute("volatile_", odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute("nontemporal", odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseStoreOp(parser, result);
}

void StoreOp::print(::mlir::OpAsmPrinter &p) {
  printStoreOp(p, *this);
}

::mlir::LogicalResult StoreOp::verify() {
  if (failed(StoreOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps25(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps24(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SubOp definitions
//===----------------------------------------------------------------------===//

SubOpAdaptor::SubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubOpAdaptor::SubOpAdaptor(SubOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SubOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOp::res() {
  return *getODSResults(0).begin();
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SubOp::verify() {
  if (failed(SubOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult SubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.sub";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SubOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SwitchOp definitions
//===----------------------------------------------------------------------===//

SwitchOpAdaptor::SwitchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SwitchOpAdaptor::SwitchOpAdaptor(SwitchOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SwitchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::ValueRange SwitchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SwitchOpAdaptor::defaultOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange SwitchOpAdaptor::caseOperands() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr SwitchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr SwitchOpAdaptor::case_values() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("case_values").dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr SwitchOpAdaptor::case_operand_offsets() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("case_operand_offsets").dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr SwitchOpAdaptor::branch_weights() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("branch_weights").dyn_cast_or_null<::mlir::ElementsAttr>();
  return attr;
}

::mlir::LogicalResult SwitchOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    {
  auto tblgen_case_values = odsAttrs.get("case_values");
  if (tblgen_case_values) {
    if (!((tblgen_case_values.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'llvm.switch' op ""attribute 'case_values' failed to satisfy constraint: constant vector/tensor attribute");
  }
  }
  {
  auto tblgen_case_operand_offsets = odsAttrs.get("case_operand_offsets");
  if (tblgen_case_operand_offsets) {
    if (!((tblgen_case_operand_offsets.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'llvm.switch' op ""attribute 'case_operand_offsets' failed to satisfy constraint: constant vector/tensor attribute");
  }
  }
  {
  auto tblgen_branch_weights = odsAttrs.get("branch_weights");
  if (tblgen_branch_weights) {
    if (!((tblgen_branch_weights.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'llvm.switch' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SwitchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  auto sizeAttrValues = sizeAttr.getValues<uint32_t>();
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += *(sizeAttrValues.begin() + i);
  unsigned size = *(sizeAttrValues.begin() + index);
  return {start, size};
}

::mlir::Operation::operand_range SwitchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range SwitchOp::defaultOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SwitchOp::caseOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SwitchOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SwitchOp::defaultOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SwitchOp::caseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> SwitchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOp::defaultDestination() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOp::caseDestinations() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ElementsAttr SwitchOp::case_valuesAttr() {
  return (*this)->getAttr("case_values").template dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOp::case_values() {
  auto attr = case_valuesAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::ElementsAttr SwitchOp::case_operand_offsetsAttr() {
  return (*this)->getAttr("case_operand_offsets").template dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOp::case_operand_offsets() {
  auto attr = case_operand_offsetsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::ElementsAttr SwitchOp::branch_weightsAttr() {
  return (*this)->getAttr("branch_weights").template dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOp::branch_weights() {
  auto attr = branch_weightsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

void SwitchOp::case_valuesAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("case_values", attr);
}

void SwitchOp::case_operand_offsetsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("case_operand_offsets", attr);
}

void SwitchOp::branch_weightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("branch_weights", attr);
}

::mlir::Attribute SwitchOp::removeCase_valuesAttr() {
  return (*this)->removeAttr("case_values");
}

::mlir::Attribute SwitchOp::removeCase_operand_offsetsAttr() {
  return (*this)->removeAttr("case_operand_offsets");
}

::mlir::Attribute SwitchOp::removeBranch_weightsAttr() {
  return (*this)->removeAttr("branch_weights");
}



void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ValueRange defaultOperands, ::mlir::ValueRange caseOperands, /*optional*/::mlir::ElementsAttr case_values, /*optional*/::mlir::ElementsAttr case_operand_offsets, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(value);
  odsState.addOperands(defaultOperands);
  odsState.addOperands(caseOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(caseOperands.size())}));
  if (case_values) {
  odsState.addAttribute("case_values", case_values);
  }
  if (case_operand_offsets) {
  odsState.addAttribute("case_operand_offsets", case_operand_offsets);
  }
  if (branch_weights) {
  odsState.addAttribute("branch_weights", branch_weights);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ValueRange defaultOperands, ::mlir::ValueRange caseOperands, /*optional*/::mlir::ElementsAttr case_values, /*optional*/::mlir::ElementsAttr case_operand_offsets, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(value);
  odsState.addOperands(defaultOperands);
  odsState.addOperands(caseOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(caseOperands.size())}));
  if (case_values) {
  odsState.addAttribute("case_values", case_values);
  }
  if (case_operand_offsets) {
  odsState.addAttribute("case_operand_offsets", case_operand_offsets);
  }
  if (branch_weights) {
  odsState.addAttribute("branch_weights", branch_weights);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOp::verify() {
  if (failed(SwitchOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps26(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps26(getOperation(), v.getType(), "operand", index)))
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
  return ::verify(*this);
}



::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Block *defaultDestinationSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> defaultOperandsOperands;
  ::llvm::SMLoc defaultOperandsOperandsLoc;
  (void)defaultOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> defaultOperandsTypes;
  ::mlir::ElementsAttr case_valuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> caseDestinationsSuccessors;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> caseOperandsOperands;
  ::llvm::SMLoc caseOperandsOperandsLoc;
  (void)caseOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> caseOperandsTypes;
  ::mlir::ElementsAttr case_operand_offsetsAttr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestinationSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  defaultOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(defaultOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(defaultOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    caseOperandsOperandsLoc = parser.getCurrentLocation();
    if (parseSwitchOpCases(parser, case_valuesAttr, caseDestinationsSuccessors, caseOperandsOperands, caseOperandsTypes, case_operand_offsetsAttr))
      return ::mlir::failure();
    if (case_valuesAttr)
      result.addAttribute("case_values", case_valuesAttr);
    if (case_operand_offsetsAttr)
      result.addAttribute("case_operand_offsets", case_operand_offsetsAttr);
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(defaultOperandsOperands, defaultOperandsTypes, defaultOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(caseOperandsOperands, caseOperandsTypes, caseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(defaultDestinationSuccessor);
  result.addSuccessors(caseDestinationsSuccessors);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(defaultOperandsOperands.size()), static_cast<int32_t>(caseOperandsOperands.size())}));
  return ::mlir::success();
}

void SwitchOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.switch";
  p << ' ';
  p << value();
  p << ",";
  p << ' ';
  p << defaultDestination();
  if (!defaultOperands().empty()) {
  p << "(";
  p << defaultOperands();
  p << ' ' << ":";
  p << ' ';
  p << defaultOperands().getTypes();
  p << ")";
  }
  p << ' ' << "[";
  p.printNewline();
  printSwitchOpCases(p, *this, case_valuesAttr(), caseDestinations(), caseOperands(), caseOperands().getTypes(), case_operand_offsetsAttr());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "case_values", "case_operand_offsets"});
}

void SwitchOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::TruncOp definitions
//===----------------------------------------------------------------------===//

TruncOpAdaptor::TruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TruncOpAdaptor::TruncOpAdaptor(TruncOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TruncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TruncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TruncOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOp::res() {
  return *getODSResults(0).begin();
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult TruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void TruncOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult TruncOp::verify() {
  if (failed(TruncOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void TruncOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UAddWithOverflowOp definitions
//===----------------------------------------------------------------------===//

UAddWithOverflowOpAdaptor::UAddWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UAddWithOverflowOpAdaptor::UAddWithOverflowOpAdaptor(UAddWithOverflowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UAddWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UAddWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UAddWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UAddWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UAddWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UAddWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UAddWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UAddWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UAddWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UAddWithOverflowOp::res() {
  return *getODSResults(0).begin();
}

void UAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void UAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UAddWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UAddWithOverflowOp::verify() {
  if (failed(UAddWithOverflowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UDivOp definitions
//===----------------------------------------------------------------------===//

UDivOpAdaptor::UDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UDivOpAdaptor::UDivOpAdaptor(UDivOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UDivOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr UDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UDivOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UDivOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange UDivOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOp::res() {
  return *getODSResults(0).begin();
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult UDivOp::verify() {
  if (failed(UDivOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult UDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UDivOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.udiv";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void UDivOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UIToFPOp definitions
//===----------------------------------------------------------------------===//

UIToFPOpAdaptor::UIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UIToFPOpAdaptor::UIToFPOpAdaptor(UIToFPOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UIToFPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr UIToFPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange UIToFPOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::res() {
  return *getODSResults(0).begin();
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult UIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void UIToFPOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult UIToFPOp::verify() {
  if (failed(UIToFPOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void UIToFPOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UMulWithOverflowOp definitions
//===----------------------------------------------------------------------===//

UMulWithOverflowOpAdaptor::UMulWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UMulWithOverflowOpAdaptor::UMulWithOverflowOpAdaptor(UMulWithOverflowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UMulWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UMulWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UMulWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UMulWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UMulWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UMulWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UMulWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UMulWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UMulWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMulWithOverflowOp::res() {
  return *getODSResults(0).begin();
}

void UMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void UMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UMulWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UMulWithOverflowOp::verify() {
  if (failed(UMulWithOverflowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::URemOp definitions
//===----------------------------------------------------------------------===//

URemOpAdaptor::URemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

URemOpAdaptor::URemOpAdaptor(URemOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange URemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> URemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange URemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value URemOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr URemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult URemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> URemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range URemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value URemOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange URemOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange URemOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> URemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range URemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOp::res() {
  return *getODSResults(0).begin();
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void URemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult URemOp::verify() {
  if (failed(URemOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult URemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void URemOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.urem";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void URemOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::USubWithOverflowOp definitions
//===----------------------------------------------------------------------===//

USubWithOverflowOpAdaptor::USubWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

USubWithOverflowOpAdaptor::USubWithOverflowOpAdaptor(USubWithOverflowOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange USubWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> USubWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange USubWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr USubWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult USubWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> USubWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range USubWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> USubWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range USubWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value USubWithOverflowOp::res() {
  return *getODSResults(0).begin();
}

void USubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void USubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void USubWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult USubWithOverflowOp::verify() {
  if (failed(USubWithOverflowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UndefOp definitions
//===----------------------------------------------------------------------===//

UndefOpAdaptor::UndefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UndefOpAdaptor::UndefOpAdaptor(UndefOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UndefOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UndefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UndefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UndefOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UndefOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UndefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UndefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UndefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UndefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UndefOp::res() {
  return *getODSResults(0).begin();
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UndefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UndefOp::verify() {
  if (failed(UndefOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult UndefOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void UndefOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.mlir.undef";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void UndefOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UnreachableOp definitions
//===----------------------------------------------------------------------===//

UnreachableOpAdaptor::UnreachableOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UnreachableOpAdaptor::UnreachableOpAdaptor(UnreachableOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UnreachableOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UnreachableOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UnreachableOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UnreachableOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UnreachableOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> UnreachableOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnreachableOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UnreachableOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnreachableOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnreachableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void UnreachableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnreachableOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult UnreachableOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return success();
}

void UnreachableOp::print(::mlir::OpAsmPrinter &p) {
  p << getOperationName();
}

::mlir::LogicalResult UnreachableOp::verify() {
  if (failed(UnreachableOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::XOrOp definitions
//===----------------------------------------------------------------------===//

XOrOpAdaptor::XOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

XOrOpAdaptor::XOrOpAdaptor(XOrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange XOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> XOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange XOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value XOrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr XOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult XOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> XOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range XOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value XOrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange XOrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange XOrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> XOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range XOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::res() {
  return *getODSResults(0).begin();
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult XOrOp::verify() {
  if (failed(XOrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult XOrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void XOrOp::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.xor";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p << ' ';
  printLLVMOpAttrs(p, *this, getOperation()->getAttrDictionary());
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void XOrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ZExtOp definitions
//===----------------------------------------------------------------------===//

ZExtOpAdaptor::ZExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ZExtOpAdaptor::ZExtOpAdaptor(ZExtOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ZExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ZExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ZExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ZExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ZExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ZExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ZExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ZExtOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ZExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ZExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOp::res() {
  return *getODSResults(0).begin();
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ZExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ZExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return mlir::impl::parseCastOp(parser, result);
}

void ZExtOp::print(::mlir::OpAsmPrinter &p) {
  mlir::impl::printCastOp(this->getOperation(), p);
}

::mlir::LogicalResult ZExtOp::verify() {
  if (failed(ZExtOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ZExtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_compressstore definitions
//===----------------------------------------------------------------------===//

masked_compressstoreAdaptor::masked_compressstoreAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

masked_compressstoreAdaptor::masked_compressstoreAdaptor(masked_compressstore&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange masked_compressstoreAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_compressstoreAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange masked_compressstoreAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr masked_compressstoreAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult masked_compressstoreAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> masked_compressstore::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range masked_compressstore::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> masked_compressstore::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_compressstore::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void masked_compressstore::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
}

void masked_compressstore::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_compressstore::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_compressstore::verify() {
  if (failed(masked_compressstoreAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_expandload definitions
//===----------------------------------------------------------------------===//

masked_expandloadAdaptor::masked_expandloadAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

masked_expandloadAdaptor::masked_expandloadAdaptor(masked_expandload&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange masked_expandloadAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_expandloadAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange masked_expandloadAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr masked_expandloadAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult masked_expandloadAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> masked_expandload::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range masked_expandload::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> masked_expandload::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_expandload::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_expandload::res() {
  return *getODSResults(0).begin();
}

void masked_expandload::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void masked_expandload::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_expandload::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_expandload::verify() {
  if (failed(masked_expandloadAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_gather definitions
//===----------------------------------------------------------------------===//

masked_gatherAdaptor::masked_gatherAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

masked_gatherAdaptor::masked_gatherAdaptor(masked_gather&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange masked_gatherAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_gatherAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange masked_gatherAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_gatherAdaptor::ptrs() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_gatherAdaptor::mask() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange masked_gatherAdaptor::pass_thru() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr masked_gatherAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr masked_gatherAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult masked_gatherAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (!tblgen_alignment) return emitError(loc, "'llvm.intr.masked.gather' op ""requires attribute 'alignment'");
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.masked.gather' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> masked_gather::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range masked_gather::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_gather::ptrs() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_gather::mask() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range masked_gather::pass_thru() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange masked_gather::ptrsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange masked_gather::maskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange masked_gather::pass_thruMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> masked_gather::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_gather::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_gather::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr masked_gather::alignmentAttr() {
  return (*this)->getAttr("alignment").template cast<::mlir::IntegerAttr>();
}

uint32_t masked_gather::alignment() {
  auto attr = alignmentAttr();
  return attr.getValue().getZExtValue();
}

void masked_gather::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", alignment);
  odsState.addTypes(res);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", alignment);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  odsState.addTypes(res);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_gather::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_gather::verify() {
  if (failed(masked_gatherAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult masked_gather::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void masked_gather::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.masked.gather";
  p << ' ';
  p << getOperation()->getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_scatter definitions
//===----------------------------------------------------------------------===//

masked_scatterAdaptor::masked_scatterAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

masked_scatterAdaptor::masked_scatterAdaptor(masked_scatter&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange masked_scatterAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_scatterAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange masked_scatterAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_scatterAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_scatterAdaptor::ptrs() {
  return *getODSOperands(1).begin();
}

::mlir::Value masked_scatterAdaptor::mask() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr masked_scatterAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr masked_scatterAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult masked_scatterAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (!tblgen_alignment) return emitError(loc, "'llvm.intr.masked.scatter' op ""requires attribute 'alignment'");
    if (!(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'llvm.intr.masked.scatter' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> masked_scatter::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range masked_scatter::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_scatter::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_scatter::ptrs() {
  return *getODSOperands(1).begin();
}

::mlir::Value masked_scatter::mask() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange masked_scatter::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange masked_scatter::ptrsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange masked_scatter::maskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> masked_scatter::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_scatter::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr masked_scatter::alignmentAttr() {
  return (*this)->getAttr("alignment").template cast<::mlir::IntegerAttr>();
}

uint32_t masked_scatter::alignment() {
  auto attr = alignmentAttr();
  return attr.getValue().getZExtValue();
}

void masked_scatter::alignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("alignment", attr);
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes) {
      odsState.addAttribute(namedAttr.first, namedAttr.second);
    }
  
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", alignment);
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_scatter::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_scatter::verify() {
  if (failed(masked_scatterAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult masked_scatter::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType ptrsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> ptrsOperands(ptrsRawOperands);  ::llvm::SMLoc ptrsOperandsLoc;
  (void)ptrsOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type ptrsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> ptrsTypes(ptrsRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  ptrsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  if (parser.parseType(ptrsRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ptrsOperands, ptrsTypes, ptrsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void masked_scatter::print(::mlir::OpAsmPrinter &p) {
  p << "llvm.intr.masked.scatter";
  p << ' ';
  p << value();
  p << ",";
  p << ' ';
  p << ptrs();
  p << ",";
  p << ' ';
  p << mask();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ' ' << "into";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(ptrs().getType());
}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_add definitions
//===----------------------------------------------------------------------===//

vector_reduce_addAdaptor::vector_reduce_addAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_addAdaptor::vector_reduce_addAdaptor(vector_reduce_add&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_addAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_addAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_addAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_addAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_addAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_add::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_add::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_add::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_add::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_add::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_add::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_add::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_add::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_add::verify() {
  if (failed(vector_reduce_addAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_add::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_and definitions
//===----------------------------------------------------------------------===//

vector_reduce_andAdaptor::vector_reduce_andAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_andAdaptor::vector_reduce_andAdaptor(vector_reduce_and&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_andAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_andAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_andAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_andAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_andAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_and::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_and::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_and::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_and::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_and::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_and::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_and::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_and::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_and::verify() {
  if (failed(vector_reduce_andAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_and::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fadd definitions
//===----------------------------------------------------------------------===//

vector_reduce_faddAdaptor::vector_reduce_faddAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_faddAdaptor::vector_reduce_faddAdaptor(vector_reduce_fadd&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_faddAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_faddAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_faddAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_faddAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr vector_reduce_faddAdaptor::reassoc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("reassoc").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::LogicalResult vector_reduce_faddAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_reassoc = odsAttrs.get("reassoc");
  if (tblgen_reassoc) {
    if (!((tblgen_reassoc.isa<::mlir::BoolAttr>()))) return emitError(loc, "'llvm.intr.vector.reduce.fadd' op ""attribute 'reassoc' failed to satisfy constraint: bool attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_fadd::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fadd::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fadd::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fadd::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fadd::res() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr vector_reduce_fadd::reassocAttr() {
  return (*this)->getAttr("reassoc").template dyn_cast_or_null<::mlir::BoolAttr>();
}

bool vector_reduce_fadd::reassoc() {
  auto attr = reassocAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void vector_reduce_fadd::reassocAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("reassoc", attr);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", reassoc);
  odsState.addTypes(res);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", reassoc);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", odsBuilder.getBoolAttr(reassoc));
  odsState.addTypes(res);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", odsBuilder.getBoolAttr(reassoc));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fadd::verify() {
  if (failed(vector_reduce_faddAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_fadd::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fmax definitions
//===----------------------------------------------------------------------===//

vector_reduce_fmaxAdaptor::vector_reduce_fmaxAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_fmaxAdaptor::vector_reduce_fmaxAdaptor(vector_reduce_fmax&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_fmaxAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_fmaxAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_fmaxAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_fmaxAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_fmaxAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_fmax::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fmax::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fmax::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fmax::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fmax::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_fmax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_fmax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmax::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fmax::verify() {
  if (failed(vector_reduce_fmaxAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_fmax::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fmin definitions
//===----------------------------------------------------------------------===//

vector_reduce_fminAdaptor::vector_reduce_fminAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_fminAdaptor::vector_reduce_fminAdaptor(vector_reduce_fmin&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_fminAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_fminAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_fminAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_fminAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_fminAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_fmin::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fmin::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fmin::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fmin::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fmin::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_fmin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_fmin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmin::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fmin::verify() {
  if (failed(vector_reduce_fminAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_fmin::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fmul definitions
//===----------------------------------------------------------------------===//

vector_reduce_fmulAdaptor::vector_reduce_fmulAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_fmulAdaptor::vector_reduce_fmulAdaptor(vector_reduce_fmul&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_fmulAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_fmulAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_fmulAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_fmulAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr vector_reduce_fmulAdaptor::reassoc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("reassoc").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::LogicalResult vector_reduce_fmulAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_reassoc = odsAttrs.get("reassoc");
  if (tblgen_reassoc) {
    if (!((tblgen_reassoc.isa<::mlir::BoolAttr>()))) return emitError(loc, "'llvm.intr.vector.reduce.fmul' op ""attribute 'reassoc' failed to satisfy constraint: bool attribute");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_fmul::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fmul::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fmul::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fmul::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fmul::res() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr vector_reduce_fmul::reassocAttr() {
  return (*this)->getAttr("reassoc").template dyn_cast_or_null<::mlir::BoolAttr>();
}

bool vector_reduce_fmul::reassoc() {
  auto attr = reassocAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void vector_reduce_fmul::reassocAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("reassoc", attr);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", reassoc);
  odsState.addTypes(res);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", reassoc);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", odsBuilder.getBoolAttr(reassoc));
  odsState.addTypes(res);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute("reassoc", odsBuilder.getBoolAttr(reassoc));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fmul::verify() {
  if (failed(vector_reduce_fmulAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_fmul::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_mul definitions
//===----------------------------------------------------------------------===//

vector_reduce_mulAdaptor::vector_reduce_mulAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_mulAdaptor::vector_reduce_mulAdaptor(vector_reduce_mul&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_mulAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_mulAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_mulAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_mulAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_mulAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_mul::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_mul::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_mul::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_mul::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_mul::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_mul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_mul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_mul::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_mul::verify() {
  if (failed(vector_reduce_mulAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_mul::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_or definitions
//===----------------------------------------------------------------------===//

vector_reduce_orAdaptor::vector_reduce_orAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_orAdaptor::vector_reduce_orAdaptor(vector_reduce_or&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_orAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_orAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_orAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_orAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_orAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_or::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_or::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_or::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_or::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_or::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_or::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_or::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_or::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_or::verify() {
  if (failed(vector_reduce_orAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_or::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_smax definitions
//===----------------------------------------------------------------------===//

vector_reduce_smaxAdaptor::vector_reduce_smaxAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_smaxAdaptor::vector_reduce_smaxAdaptor(vector_reduce_smax&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_smaxAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_smaxAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_smaxAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_smaxAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_smaxAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_smax::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_smax::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_smax::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_smax::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_smax::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_smax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_smax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_smax::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_smax::verify() {
  if (failed(vector_reduce_smaxAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_smax::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_smin definitions
//===----------------------------------------------------------------------===//

vector_reduce_sminAdaptor::vector_reduce_sminAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_sminAdaptor::vector_reduce_sminAdaptor(vector_reduce_smin&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_sminAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_sminAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_sminAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_sminAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_sminAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_smin::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_smin::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_smin::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_smin::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_smin::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_smin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_smin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_smin::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_smin::verify() {
  if (failed(vector_reduce_sminAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_smin::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_umax definitions
//===----------------------------------------------------------------------===//

vector_reduce_umaxAdaptor::vector_reduce_umaxAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_umaxAdaptor::vector_reduce_umaxAdaptor(vector_reduce_umax&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_umaxAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_umaxAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_umaxAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_umaxAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_umaxAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_umax::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_umax::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_umax::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_umax::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_umax::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_umax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_umax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_umax::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_umax::verify() {
  if (failed(vector_reduce_umaxAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_umax::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_umin definitions
//===----------------------------------------------------------------------===//

vector_reduce_uminAdaptor::vector_reduce_uminAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_uminAdaptor::vector_reduce_uminAdaptor(vector_reduce_umin&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_uminAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_uminAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_uminAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_uminAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_uminAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_umin::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_umin::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_umin::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_umin::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_umin::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_umin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_umin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_umin::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_umin::verify() {
  if (failed(vector_reduce_uminAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_umin::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_xor definitions
//===----------------------------------------------------------------------===//

vector_reduce_xorAdaptor::vector_reduce_xorAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

vector_reduce_xorAdaptor::vector_reduce_xorAdaptor(vector_reduce_xor&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange vector_reduce_xorAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_xorAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_xorAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_xorAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_xorAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> vector_reduce_xor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_xor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_xor::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_xor::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_xor::res() {
  return *getODSResults(0).begin();
}

void vector_reduce_xor::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_xor::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_xor::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_xor::verify() {
  if (failed(vector_reduce_xorAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void vector_reduce_xor::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace LLVM
} // namespace mlir

#endif  // GET_OP_CLASSES

