/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

mlir::tosa::AbsOp,
mlir::tosa::AddOp,
mlir::tosa::ApplyScaleOp,
mlir::tosa::ArgMaxOp,
mlir::tosa::ArithmeticRightShiftOp,
mlir::tosa::AvgPool2dOp,
mlir::tosa::BitwiseAndOp,
mlir::tosa::BitwiseNotOp,
mlir::tosa::BitwiseOrOp,
mlir::tosa::BitwiseXorOp,
mlir::tosa::CastOp,
mlir::tosa::CeilOp,
mlir::tosa::ClampOp,
mlir::tosa::ClzOp,
mlir::tosa::ConcatOp,
mlir::tosa::ConstOp,
mlir::tosa::Conv2DOp,
mlir::tosa::Conv3DOp,
mlir::tosa::CustomOp,
mlir::tosa::DepthwiseConv2DOp,
mlir::tosa::EqualOp,
mlir::tosa::ExpOp,
mlir::tosa::FloorOp,
mlir::tosa::FullyConnectedOp,
mlir::tosa::GatherOp,
mlir::tosa::GreaterEqualOp,
mlir::tosa::GreaterOp,
mlir::tosa::IdentityNOp,
mlir::tosa::IdentityOp,
mlir::tosa::IfOp,
mlir::tosa::LogOp,
mlir::tosa::LogicalAndOp,
mlir::tosa::LogicalLeftShiftOp,
mlir::tosa::LogicalNotOp,
mlir::tosa::LogicalOrOp,
mlir::tosa::LogicalRightShiftOp,
mlir::tosa::LogicalXorOp,
mlir::tosa::MatMulOp,
mlir::tosa::MaxPool2dOp,
mlir::tosa::MaximumOp,
mlir::tosa::MinimumOp,
mlir::tosa::MulOp,
mlir::tosa::NegateOp,
mlir::tosa::PadOp,
mlir::tosa::PlaceholderOp,
mlir::tosa::PowOp,
mlir::tosa::ReciprocalOp,
mlir::tosa::ReduceAllOp,
mlir::tosa::ReduceAnyOp,
mlir::tosa::ReduceMaxOp,
mlir::tosa::ReduceMinOp,
mlir::tosa::ReduceProdOp,
mlir::tosa::ReduceSumOp,
mlir::tosa::ReluNOp,
mlir::tosa::RescaleOp,
mlir::tosa::ReshapeOp,
mlir::tosa::ResizeOp,
mlir::tosa::ReverseOp,
mlir::tosa::RsqrtOp,
mlir::tosa::SelectOp,
mlir::tosa::SigmoidOp,
mlir::tosa::SliceOp,
mlir::tosa::SubOp,
mlir::tosa::TableOp,
mlir::tosa::TanhOp,
mlir::tosa::TileOp,
mlir::tosa::TransposeConv2DOp,
mlir::tosa::TransposeOp,
mlir::tosa::WhileOp,
mlir::tosa::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tosa {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 0)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 0D/1D/2D/3D/4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isSignlessInteger(32))) || (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32)))) || (((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be signless-integer-32-like, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isSignlessInteger(8))) || (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) || (((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be signless-integer-32-like, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1D/2D/3D/4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 4))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 5))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 5D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16())))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps10(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 2))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 2D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps11(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of 32-bit signless integer or 64-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps12(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 0)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 5)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 6)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 0D/1D/2D/3D/4D/5D/6D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps13(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 0)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 5)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 6)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 0D/1D/2D/3D/4D/5D/6D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps14(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(48))) || ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))))) || ((((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((type.cast<::mlir::ShapedType>().getElementType().isa<mlir::quant::QuantizedType>())) && ((type.cast<::mlir::ShapedType>().getElementType().cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16)))) || (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isBF16()))))) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 5)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 6)))))) {
    return op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1D/2D/3D/4D/5D/6D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::AbsOp definitions
//===----------------------------------------------------------------------===//

AbsOpAdaptor::AbsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AbsOpAdaptor::AbsOpAdaptor(AbsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AbsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AbsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AbsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AbsOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AbsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AbsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AbsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AbsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AbsOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AbsOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AbsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AbsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AbsOp::output() {
  return *getODSResults(0).begin();
}

void AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AbsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AbsOp::verify() {
  if (failed(AbsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void AbsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::AddOp definitions
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

::mlir::Value AddOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOpAdaptor::input2() {
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

::mlir::Value AddOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddOp::input2Mutable() {
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

::mlir::Value AddOp::output() {
  return *getODSResults(0).begin();
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
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

::mlir::LogicalResult AddOp::verify() {
  if (failed(AddOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void AddOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ApplyScaleOp definitions
//===----------------------------------------------------------------------===//

ApplyScaleOpAdaptor::ApplyScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ApplyScaleOpAdaptor::ApplyScaleOpAdaptor(ApplyScaleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ApplyScaleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyScaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ApplyScaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyScaleOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value ApplyScaleOpAdaptor::multiplier() {
  return *getODSOperands(1).begin();
}

::mlir::Value ApplyScaleOpAdaptor::shift() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ApplyScaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr ApplyScaleOpAdaptor::double_round() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("double_round").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::LogicalResult ApplyScaleOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_double_round = odsAttrs.get("double_round");
  if (!tblgen_double_round) return emitError(loc, "'tosa.apply_scale' op ""requires attribute 'double_round'");
    if (!((tblgen_double_round.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tosa.apply_scale' op ""attribute 'double_round' failed to satisfy constraint: bool attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ApplyScaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ApplyScaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyScaleOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value ApplyScaleOp::multiplier() {
  return *getODSOperands(1).begin();
}

::mlir::Value ApplyScaleOp::shift() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ApplyScaleOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ApplyScaleOp::multiplierMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ApplyScaleOp::shiftMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyScaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyScaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyScaleOp::output() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr ApplyScaleOp::double_roundAttr() {
  return (*this)->getAttr("double_round").template cast<::mlir::BoolAttr>();
}

bool ApplyScaleOp::double_round() {
  auto attr = double_roundAttr();
  return attr.getValue();
}

void ApplyScaleOp::double_roundAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("double_round", attr);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, ::mlir::BoolAttr double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute("double_round", double_round);
  odsState.addTypes(output);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, ::mlir::BoolAttr double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute("double_round", double_round);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute("double_round", odsBuilder.getBoolAttr(double_round));
  odsState.addTypes(output);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute("double_round", odsBuilder.getBoolAttr(double_round));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyScaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyScaleOp::verify() {
  if (failed(ApplyScaleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ApplyScaleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ArgMaxOp definitions
//===----------------------------------------------------------------------===//

ArgMaxOpAdaptor::ArgMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ArgMaxOpAdaptor::ArgMaxOpAdaptor(ArgMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ArgMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ArgMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ArgMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArgMaxOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ArgMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ArgMaxOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ArgMaxOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.argmax' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.argmax' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ArgMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ArgMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArgMaxOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ArgMaxOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ArgMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ArgMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArgMaxOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ArgMaxOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ArgMaxOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ArgMaxOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArgMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ArgMaxOp::verify() {
  if (failed(ArgMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ArgMaxOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ArithmeticRightShiftOp definitions
//===----------------------------------------------------------------------===//

ArithmeticRightShiftOpAdaptor::ArithmeticRightShiftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ArithmeticRightShiftOpAdaptor::ArithmeticRightShiftOpAdaptor(ArithmeticRightShiftOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ArithmeticRightShiftOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ArithmeticRightShiftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ArithmeticRightShiftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArithmeticRightShiftOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ArithmeticRightShiftOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ArithmeticRightShiftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr ArithmeticRightShiftOpAdaptor::round() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("round").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::LogicalResult ArithmeticRightShiftOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_round = odsAttrs.get("round");
  if (!tblgen_round) return emitError(loc, "'tosa.arithmetic_right_shift' op ""requires attribute 'round'");
    if (!((tblgen_round.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tosa.arithmetic_right_shift' op ""attribute 'round' failed to satisfy constraint: bool attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ArithmeticRightShiftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ArithmeticRightShiftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArithmeticRightShiftOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ArithmeticRightShiftOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ArithmeticRightShiftOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ArithmeticRightShiftOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ArithmeticRightShiftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ArithmeticRightShiftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArithmeticRightShiftOp::output() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr ArithmeticRightShiftOp::roundAttr() {
  return (*this)->getAttr("round").template cast<::mlir::BoolAttr>();
}

bool ArithmeticRightShiftOp::round() {
  auto attr = roundAttr();
  return attr.getValue();
}

void ArithmeticRightShiftOp::roundAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("round", attr);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::BoolAttr round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("round", round);
  odsState.addTypes(output);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::BoolAttr round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("round", round);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, bool round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("round", odsBuilder.getBoolAttr(round));
  odsState.addTypes(output);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, bool round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("round", odsBuilder.getBoolAttr(round));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ArithmeticRightShiftOp::verify() {
  if (failed(ArithmeticRightShiftOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ArithmeticRightShiftOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::AvgPool2dOp definitions
//===----------------------------------------------------------------------===//

AvgPool2dOpAdaptor::AvgPool2dOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AvgPool2dOpAdaptor::AvgPool2dOpAdaptor(AvgPool2dOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AvgPool2dOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AvgPool2dOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AvgPool2dOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AvgPool2dOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AvgPool2dOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::kernel() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("kernel").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::pad() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("pad").cast<::mlir::ArrayAttr>();
  return attr;
}

mlir::tosa::UnaryOpQuantizationAttr AvgPool2dOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::UnaryOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult AvgPool2dOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_kernel = odsAttrs.get("kernel");
  if (!tblgen_kernel) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'kernel'");
    if (!((((tblgen_kernel.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_kernel.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_kernel.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'kernel' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_pad = odsAttrs.get("pad");
  if (!tblgen_pad) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'pad'");
    if (!((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4)))) return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");
  }
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::UnaryOpQuantizationAttr>()))) return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for UnaryOp quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> AvgPool2dOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AvgPool2dOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AvgPool2dOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AvgPool2dOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AvgPool2dOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AvgPool2dOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AvgPool2dOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr AvgPool2dOp::kernelAttr() {
  return (*this)->getAttr("kernel").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AvgPool2dOp::kernel() {
  auto attr = kernelAttr();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AvgPool2dOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOp::padAttr() {
  return (*this)->getAttr("pad").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AvgPool2dOp::pad() {
  auto attr = padAttr();
  return attr;
}

mlir::tosa::UnaryOpQuantizationAttr AvgPool2dOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::UnaryOpQuantizationAttr> AvgPool2dOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::UnaryOpQuantizationAttr>(attr) : (::llvm::None);
}

void AvgPool2dOp::kernelAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("kernel", attr);
}

void AvgPool2dOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void AvgPool2dOp::padAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("pad", attr);
}

void AvgPool2dOp::quantization_infoAttr(mlir::tosa::UnaryOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute AvgPool2dOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, ArrayAttr kernel, ArrayAttr stride, ArrayAttr pad) {
    buildAvgPool2dOpWithQuantInfo(odsBuilder, odsState, outputType,
                                  input, kernel, stride, pad);
  
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addAttribute("kernel", kernel);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("pad", pad);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addAttribute("kernel", kernel);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("pad", pad);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AvgPool2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AvgPool2dOp::verify() {
  if (failed(AvgPool2dOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void AvgPool2dOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseAndOp definitions
//===----------------------------------------------------------------------===//

BitwiseAndOpAdaptor::BitwiseAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BitwiseAndOpAdaptor::BitwiseAndOpAdaptor(BitwiseAndOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BitwiseAndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitwiseAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseAndOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseAndOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BitwiseAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseAndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BitwiseAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseAndOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseAndOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BitwiseAndOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BitwiseAndOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BitwiseAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseAndOp::output() {
  return *getODSResults(0).begin();
}

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseAndOp::verify() {
  if (failed(BitwiseAndOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void BitwiseAndOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseNotOp definitions
//===----------------------------------------------------------------------===//

BitwiseNotOpAdaptor::BitwiseNotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BitwiseNotOpAdaptor::BitwiseNotOpAdaptor(BitwiseNotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BitwiseNotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitwiseNotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseNotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseNotOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BitwiseNotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseNotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BitwiseNotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseNotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseNotOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitwiseNotOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BitwiseNotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseNotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseNotOp::output() {
  return *getODSResults(0).begin();
}

void BitwiseNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void BitwiseNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseNotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseNotOp::verify() {
  if (failed(BitwiseNotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void BitwiseNotOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseOrOp definitions
//===----------------------------------------------------------------------===//

BitwiseOrOpAdaptor::BitwiseOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BitwiseOrOpAdaptor::BitwiseOrOpAdaptor(BitwiseOrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BitwiseOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitwiseOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseOrOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseOrOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BitwiseOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BitwiseOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseOrOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseOrOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BitwiseOrOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BitwiseOrOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BitwiseOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseOrOp::output() {
  return *getODSResults(0).begin();
}

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseOrOp::verify() {
  if (failed(BitwiseOrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void BitwiseOrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseXorOp definitions
//===----------------------------------------------------------------------===//

BitwiseXorOpAdaptor::BitwiseXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BitwiseXorOpAdaptor::BitwiseXorOpAdaptor(BitwiseXorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BitwiseXorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitwiseXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseXorOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseXorOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BitwiseXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseXorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> BitwiseXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseXorOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseXorOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BitwiseXorOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BitwiseXorOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BitwiseXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseXorOp::output() {
  return *getODSResults(0).begin();
}

void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseXorOp::verify() {
  if (failed(BitwiseXorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void BitwiseXorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CastOpAdaptor::CastOpAdaptor(CastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::output() {
  return *getODSResults(0).begin();
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verify() {
  if (failed(CastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void CastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::CeilOp definitions
//===----------------------------------------------------------------------===//

CeilOpAdaptor::CeilOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CeilOpAdaptor::CeilOpAdaptor(CeilOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CeilOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CeilOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CeilOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CeilOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CeilOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CeilOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CeilOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CeilOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CeilOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CeilOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CeilOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CeilOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CeilOp::output() {
  return *getODSResults(0).begin();
}

void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CeilOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult CeilOp::verify() {
  if (failed(CeilOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void CeilOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ClampOp definitions
//===----------------------------------------------------------------------===//

ClampOpAdaptor::ClampOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ClampOpAdaptor::ClampOpAdaptor(ClampOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ClampOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ClampOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ClampOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClampOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ClampOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ClampOpAdaptor::min_int() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("min_int").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr ClampOpAdaptor::max_int() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("max_int").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::FloatAttr ClampOpAdaptor::min_fp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("min_fp").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::FloatAttr ClampOpAdaptor::max_fp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("max_fp").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::LogicalResult ClampOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_min_int = odsAttrs.get("min_int");
  if (!tblgen_min_int) return emitError(loc, "'tosa.clamp' op ""requires attribute 'min_int'");
    if (!(((tblgen_min_int.isa<::mlir::IntegerAttr>())) && ((tblgen_min_int.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.clamp' op ""attribute 'min_int' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_max_int = odsAttrs.get("max_int");
  if (!tblgen_max_int) return emitError(loc, "'tosa.clamp' op ""requires attribute 'max_int'");
    if (!(((tblgen_max_int.isa<::mlir::IntegerAttr>())) && ((tblgen_max_int.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.clamp' op ""attribute 'max_int' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_min_fp = odsAttrs.get("min_fp");
  if (!tblgen_min_fp) return emitError(loc, "'tosa.clamp' op ""requires attribute 'min_fp'");
    if (!(((tblgen_min_fp.isa<::mlir::FloatAttr>())) && ((tblgen_min_fp.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'tosa.clamp' op ""attribute 'min_fp' failed to satisfy constraint: 32-bit float attribute");
  }
  {
  auto tblgen_max_fp = odsAttrs.get("max_fp");
  if (!tblgen_max_fp) return emitError(loc, "'tosa.clamp' op ""requires attribute 'max_fp'");
    if (!(((tblgen_max_fp.isa<::mlir::FloatAttr>())) && ((tblgen_max_fp.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'tosa.clamp' op ""attribute 'max_fp' failed to satisfy constraint: 32-bit float attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ClampOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ClampOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClampOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ClampOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ClampOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ClampOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClampOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ClampOp::min_intAttr() {
  return (*this)->getAttr("min_int").template cast<::mlir::IntegerAttr>();
}

uint64_t ClampOp::min_int() {
  auto attr = min_intAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ClampOp::max_intAttr() {
  return (*this)->getAttr("max_int").template cast<::mlir::IntegerAttr>();
}

uint64_t ClampOp::max_int() {
  auto attr = max_intAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ClampOp::min_fpAttr() {
  return (*this)->getAttr("min_fp").template cast<::mlir::FloatAttr>();
}

::llvm::APFloat ClampOp::min_fp() {
  auto attr = min_fpAttr();
  return attr.getValue();
}

::mlir::FloatAttr ClampOp::max_fpAttr() {
  return (*this)->getAttr("max_fp").template cast<::mlir::FloatAttr>();
}

::llvm::APFloat ClampOp::max_fp() {
  auto attr = max_fpAttr();
  return attr.getValue();
}

void ClampOp::min_intAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("min_int", attr);
}

void ClampOp::max_intAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("max_int", attr);
}

void ClampOp::min_fpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr("min_fp", attr);
}

void ClampOp::max_fpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr("max_fp", attr);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr min_int, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr min_fp, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("min_int", min_int);
  odsState.addAttribute("max_int", max_int);
  odsState.addAttribute("min_fp", min_fp);
  odsState.addAttribute("max_fp", max_fp);
  odsState.addTypes(output);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr min_int, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr min_fp, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("min_int", min_int);
  odsState.addAttribute("max_int", max_int);
  odsState.addAttribute("min_fp", min_fp);
  odsState.addAttribute("max_fp", max_fp);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min_fp, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("min_int", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), min_int));
  odsState.addAttribute("max_int", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int));
  odsState.addAttribute("min_fp", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp));
  odsState.addAttribute("max_fp", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp));
  odsState.addTypes(output);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min_fp, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("min_int", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), min_int));
  odsState.addAttribute("max_int", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int));
  odsState.addAttribute("min_fp", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp));
  odsState.addAttribute("max_fp", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClampOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr min_int, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr min_fp, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("min_int", min_int);
  odsState.addAttribute("max_int", max_int);
  odsState.addAttribute("min_fp", min_fp);
  odsState.addAttribute("max_fp", max_fp);
  odsState.addTypes({input.getType()});

}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ClampOp::verify() {
  if (failed(ClampOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ClampOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ClzOp definitions
//===----------------------------------------------------------------------===//

ClzOpAdaptor::ClzOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ClzOpAdaptor::ClzOpAdaptor(ClzOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ClzOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ClzOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ClzOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClzOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ClzOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ClzOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ClzOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ClzOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClzOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ClzOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ClzOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ClzOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClzOp::output() {
  return *getODSResults(0).begin();
}

void ClzOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void ClzOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClzOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ClzOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void ClzOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ClzOp::verify() {
  if (failed(ClzOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ClzOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ConcatOp definitions
//===----------------------------------------------------------------------===//

ConcatOpAdaptor::ConcatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConcatOpAdaptor::ConcatOpAdaptor(ConcatOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConcatOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConcatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConcatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConcatOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConcatOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ConcatOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ConcatOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ConcatOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.concat' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.concat' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConcatOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConcatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConcatOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConcatOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ConcatOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConcatOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConcatOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConcatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConcatOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ConcatOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ConcatOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ConcatOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConcatOp::verify() {
  if (failed(ConcatOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ConcatOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ConstOp definitions
//===----------------------------------------------------------------------===//

ConstOpAdaptor::ConstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConstOpAdaptor::ConstOpAdaptor(ConstOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConstOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ConstOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr ConstOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ElementsAttr attr = odsAttrs.get("value").cast<::mlir::ElementsAttr>();
  return attr;
}

::mlir::LogicalResult ConstOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'tosa.const' op ""requires attribute 'value'");
    if (!((tblgen_value.isa<::mlir::ElementsAttr>()))) return emitError(loc, "'tosa.const' op ""attribute 'value' failed to satisfy constraint: constant vector/tensor attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ConstOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ElementsAttr ConstOp::valueAttr() {
  return (*this)->getAttr("value").template cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr ConstOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstOp::valueAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr("value", attr);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ElementsAttr value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(output);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ElementsAttr value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  for (auto attr : attributes) {
    if (attr.first != "value") continue;
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
    odsState.addTypes({attr.second.getType()});
  }
}

::mlir::LogicalResult ConstOp::verify() {
  if (failed(ConstOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}



void ConstOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::Conv2DOp definitions
//===----------------------------------------------------------------------===//

Conv2DOpAdaptor::Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Conv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv2DOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv2DOpAdaptor::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv2DOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr Conv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr Conv2DOpAdaptor::pad() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("pad").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilation").cast<::mlir::ArrayAttr>();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv2DOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::ConvOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult Conv2DOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_pad = odsAttrs.get("pad");
  if (!tblgen_pad) return emitError(loc, "'tosa.conv2d' op ""requires attribute 'pad'");
    if (!((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4)))) return emitError(loc, "'tosa.conv2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.conv2d' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.conv2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_dilation = odsAttrs.get("dilation");
  if (!tblgen_dilation) return emitError(loc, "'tosa.conv2d' op ""requires attribute 'dilation'");
    if (!((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.conv2d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>()))) return emitError(loc, "'tosa.conv2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Conv2DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Conv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv2DOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv2DOp::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv2DOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange Conv2DOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Conv2DOp::weightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Conv2DOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Conv2DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Conv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv2DOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr Conv2DOp::padAttr() {
  return (*this)->getAttr("pad").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv2DOp::pad() {
  auto attr = padAttr();
  return attr;
}

::mlir::ArrayAttr Conv2DOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv2DOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr Conv2DOp::dilationAttr() {
  return (*this)->getAttr("dilation").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv2DOp::dilation() {
  auto attr = dilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv2DOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr> Conv2DOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr>(attr) : (::llvm::None);
}

void Conv2DOp::padAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("pad", attr);
}

void Conv2DOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void Conv2DOp::dilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilation", attr);
}

void Conv2DOp::quantization_infoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute Conv2DOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute("pad", pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute("pad", pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv2DOp::verify() {
  if (failed(Conv2DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return verifyConvOp(*this);
}

void Conv2DOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::Conv3DOp definitions
//===----------------------------------------------------------------------===//

Conv3DOpAdaptor::Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv3DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Conv3DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv3DOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv3DOpAdaptor::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv3DOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr Conv3DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr Conv3DOpAdaptor::pad() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("pad").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilation").cast<::mlir::ArrayAttr>();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv3DOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::ConvOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult Conv3DOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_pad = odsAttrs.get("pad");
  if (!tblgen_pad) return emitError(loc, "'tosa.conv3d' op ""requires attribute 'pad'");
    if (!((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 6)))) return emitError(loc, "'tosa.conv3d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 6 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.conv3d' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'tosa.conv3d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  {
  auto tblgen_dilation = odsAttrs.get("dilation");
  if (!tblgen_dilation) return emitError(loc, "'tosa.conv3d' op ""requires attribute 'dilation'");
    if (!((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 3)))) return emitError(loc, "'tosa.conv3d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");
  }
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>()))) return emitError(loc, "'tosa.conv3d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> Conv3DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Conv3DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv3DOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv3DOp::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv3DOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange Conv3DOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Conv3DOp::weightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Conv3DOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Conv3DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Conv3DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv3DOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr Conv3DOp::padAttr() {
  return (*this)->getAttr("pad").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv3DOp::pad() {
  auto attr = padAttr();
  return attr;
}

::mlir::ArrayAttr Conv3DOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv3DOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr Conv3DOp::dilationAttr() {
  return (*this)->getAttr("dilation").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv3DOp::dilation() {
  auto attr = dilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv3DOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr> Conv3DOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr>(attr) : (::llvm::None);
}

void Conv3DOp::padAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("pad", attr);
}

void Conv3DOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void Conv3DOp::dilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilation", attr);
}

void Conv3DOp::quantization_infoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute Conv3DOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute("pad", pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute("pad", pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv3DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv3DOp::verify() {
  if (failed(Conv3DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return verifyConvOp(*this);
}

void Conv3DOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::CustomOp definitions
//===----------------------------------------------------------------------===//

CustomOpAdaptor::CustomOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CustomOpAdaptor::CustomOpAdaptor(CustomOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CustomOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CustomOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CustomOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CustomOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr CustomOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CustomOpAdaptor::identifier() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("identifier").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult CustomOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_identifier = odsAttrs.get("identifier");
  if (!tblgen_identifier) return emitError(loc, "'tosa.custom' op ""requires attribute 'identifier'");
    if (!((tblgen_identifier.isa<::mlir::StringAttr>()))) return emitError(loc, "'tosa.custom' op ""attribute 'identifier' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> CustomOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CustomOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CustomOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CustomOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CustomOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CustomOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CustomOp::outputs() {
  return getODSResults(0);
}

::mlir::StringAttr CustomOp::identifierAttr() {
  return (*this)->getAttr("identifier").template cast<::mlir::StringAttr>();
}

::llvm::StringRef CustomOp::identifier() {
  auto attr = identifierAttr();
  return attr.getValue();
}

void CustomOp::identifierAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("identifier", attr);
}

void CustomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::StringAttr identifier, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("identifier", identifier);
  odsState.addTypes(outputs);
}

void CustomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::llvm::StringRef identifier, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("identifier", odsBuilder.getStringAttr(identifier));
  odsState.addTypes(outputs);
}

void CustomOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CustomOp::verify() {
  if (failed(CustomOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::DepthwiseConv2DOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DOpAdaptor::DepthwiseConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConv2DOpAdaptor::DepthwiseConv2DOpAdaptor(DepthwiseConv2DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConv2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DepthwiseConv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DepthwiseConv2DOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value DepthwiseConv2DOpAdaptor::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value DepthwiseConv2DOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr DepthwiseConv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::pad() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("pad").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilation").cast<::mlir::ArrayAttr>();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr DepthwiseConv2DOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::ConvOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult DepthwiseConv2DOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_pad = odsAttrs.get("pad");
  if (!tblgen_pad) return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'pad'");
    if (!((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4)))) return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_dilation = odsAttrs.get("dilation");
  if (!tblgen_dilation) return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'dilation'");
    if (!((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>()))) return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> DepthwiseConv2DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DepthwiseConv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DepthwiseConv2DOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value DepthwiseConv2DOp::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value DepthwiseConv2DOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange DepthwiseConv2DOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange DepthwiseConv2DOp::weightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange DepthwiseConv2DOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DepthwiseConv2DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DepthwiseConv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DepthwiseConv2DOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr DepthwiseConv2DOp::padAttr() {
  return (*this)->getAttr("pad").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DepthwiseConv2DOp::pad() {
  auto attr = padAttr();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DepthwiseConv2DOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOp::dilationAttr() {
  return (*this)->getAttr("dilation").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DepthwiseConv2DOp::dilation() {
  auto attr = dilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr DepthwiseConv2DOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr> DepthwiseConv2DOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr>(attr) : (::llvm::None);
}

void DepthwiseConv2DOp::padAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("pad", attr);
}

void DepthwiseConv2DOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void DepthwiseConv2DOp::dilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilation", attr);
}

void DepthwiseConv2DOp::quantization_infoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute DepthwiseConv2DOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute("pad", pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute("pad", pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DepthwiseConv2DOp::verify() {
  if (failed(DepthwiseConv2DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return verifyConvOp(*this);
}

void DepthwiseConv2DOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::EqualOp definitions
//===----------------------------------------------------------------------===//

EqualOpAdaptor::EqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

EqualOpAdaptor::EqualOpAdaptor(EqualOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange EqualOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> EqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value EqualOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr EqualOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult EqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> EqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value EqualOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange EqualOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange EqualOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> EqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOp::output() {
  return *getODSResults(0).begin();
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EqualOp::verify() {
  if (failed(EqualOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void EqualOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ExpOp definitions
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

::mlir::Value ExpOpAdaptor::input1() {
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

::mlir::Value ExpOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExpOp::input1Mutable() {
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

::mlir::Value ExpOp::output() {
  return *getODSResults(0).begin();
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
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

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ExpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::FloorOp definitions
//===----------------------------------------------------------------------===//

FloorOpAdaptor::FloorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FloorOpAdaptor::FloorOpAdaptor(FloorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FloorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FloorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FloorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FloorOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FloorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FloorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FloorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FloorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FloorOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FloorOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FloorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FloorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FloorOp::output() {
  return *getODSResults(0).begin();
}

void FloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void FloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FloorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void FloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult FloorOp::verify() {
  if (failed(FloorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void FloorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::FullyConnectedOp definitions
//===----------------------------------------------------------------------===//

FullyConnectedOpAdaptor::FullyConnectedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FullyConnectedOpAdaptor::FullyConnectedOpAdaptor(FullyConnectedOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FullyConnectedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FullyConnectedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FullyConnectedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FullyConnectedOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value FullyConnectedOpAdaptor::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value FullyConnectedOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr FullyConnectedOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::ConvOpQuantizationAttr FullyConnectedOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::ConvOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult FullyConnectedOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>()))) return emitError(loc, "'tosa.fully_connected' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> FullyConnectedOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FullyConnectedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FullyConnectedOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value FullyConnectedOp::weight() {
  return *getODSOperands(1).begin();
}

::mlir::Value FullyConnectedOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FullyConnectedOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FullyConnectedOp::weightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FullyConnectedOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FullyConnectedOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FullyConnectedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FullyConnectedOp::output() {
  return *getODSResults(0).begin();
}

mlir::tosa::ConvOpQuantizationAttr FullyConnectedOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr> FullyConnectedOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr>(attr) : (::llvm::None);
}

void FullyConnectedOp::quantization_infoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute FullyConnectedOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias) {
    buildFCOpWithQuantInfo(odsBuilder, odsState, outputType,
                           input, weight, bias);
  
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FullyConnectedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FullyConnectedOp::verify() {
  if (failed(FullyConnectedOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return verifyConvOp(*this);
}

void FullyConnectedOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::GatherOp definitions
//===----------------------------------------------------------------------===//

GatherOpAdaptor::GatherOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GatherOpAdaptor::GatherOpAdaptor(GatherOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GatherOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GatherOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GatherOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOpAdaptor::indices() {
  return *getODSOperands(0).begin();
}

::mlir::Value GatherOpAdaptor::values() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr GatherOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GatherOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GatherOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.gather' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'tosa.gather' op ""attribute 'axis' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GatherOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GatherOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOp::indices() {
  return *getODSOperands(0).begin();
}

::mlir::Value GatherOp::values() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GatherOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GatherOp::valuesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GatherOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GatherOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GatherOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint32_t GatherOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void GatherOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value indices, ::mlir::Value values, ::mlir::IntegerAttr axis) {
  odsState.addOperands(indices);
  odsState.addOperands(values);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value indices, ::mlir::Value values, ::mlir::IntegerAttr axis) {
  odsState.addOperands(indices);
  odsState.addOperands(values);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value indices, ::mlir::Value values, uint32_t axis) {
  odsState.addOperands(indices);
  odsState.addOperands(values);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), axis));
  odsState.addTypes(output);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value indices, ::mlir::Value values, uint32_t axis) {
  odsState.addOperands(indices);
  odsState.addOperands(values);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GatherOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GatherOp::verify() {
  if (failed(GatherOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void GatherOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::GreaterEqualOp definitions
//===----------------------------------------------------------------------===//

GreaterEqualOpAdaptor::GreaterEqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GreaterEqualOpAdaptor::GreaterEqualOpAdaptor(GreaterEqualOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GreaterEqualOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GreaterEqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GreaterEqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterEqualOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value GreaterEqualOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr GreaterEqualOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GreaterEqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GreaterEqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GreaterEqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterEqualOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value GreaterEqualOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GreaterEqualOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GreaterEqualOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GreaterEqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GreaterEqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterEqualOp::output() {
  return *getODSResults(0).begin();
}

void GreaterEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void GreaterEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GreaterEqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GreaterEqualOp::verify() {
  if (failed(GreaterEqualOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void GreaterEqualOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::GreaterOp definitions
//===----------------------------------------------------------------------===//

GreaterOpAdaptor::GreaterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GreaterOpAdaptor::GreaterOpAdaptor(GreaterOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GreaterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GreaterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GreaterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value GreaterOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr GreaterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GreaterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> GreaterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GreaterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value GreaterOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GreaterOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GreaterOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GreaterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GreaterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterOp::output() {
  return *getODSResults(0).begin();
}

void GreaterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void GreaterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GreaterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GreaterOp::verify() {
  if (failed(GreaterOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void GreaterOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::IdentityNOp definitions
//===----------------------------------------------------------------------===//

IdentityNOpAdaptor::IdentityNOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

IdentityNOpAdaptor::IdentityNOpAdaptor(IdentityNOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange IdentityNOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IdentityNOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange IdentityNOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange IdentityNOpAdaptor::input1() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr IdentityNOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IdentityNOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> IdentityNOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range IdentityNOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range IdentityNOp::input1() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange IdentityNOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IdentityNOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range IdentityNOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IdentityNOp::output() {
  return getODSResults(0);
}

void IdentityNOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IdentityNOp::verify() {
  if (failed(IdentityNOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void IdentityNOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::IdentityOp definitions
//===----------------------------------------------------------------------===//

IdentityOpAdaptor::IdentityOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

IdentityOpAdaptor::IdentityOpAdaptor(IdentityOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange IdentityOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IdentityOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IdentityOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IdentityOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IdentityOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IdentityOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> IdentityOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IdentityOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IdentityOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IdentityOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IdentityOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IdentityOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IdentityOp::output() {
  return *getODSResults(0).begin();
}

void IdentityOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void IdentityOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IdentityOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IdentityOp::verify() {
  if (failed(IdentityOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void IdentityOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::IfOp definitions
//===----------------------------------------------------------------------===//

IfOpAdaptor::IfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

IfOpAdaptor::IfOpAdaptor(IfOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange IfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange IfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IfOpAdaptor::cond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange IfOpAdaptor::inputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr IfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange IfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &IfOpAdaptor::then_branch() {
  return *odsRegions[0];
}

::mlir::Region &IfOpAdaptor::else_branch() {
  return *odsRegions[1];
}

::mlir::LogicalResult IfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> IfOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range IfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IfOp::cond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range IfOp::inputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange IfOp::condMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange IfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range IfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IfOp::output() {
  return getODSResults(0);
}

::mlir::Region &IfOp::then_branch() {
  return (*this)->getRegion(0);
}

::mlir::Region &IfOp::else_branch() {
  return (*this)->getRegion(1);
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange output, ::mlir::Value cond, ::mlir::ValueRange inputs) {
  odsState.addOperands(cond);
  odsState.addOperands(inputs);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void IfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IfOp::verify() {
  if (failed(IfOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('then_branch') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(1))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('else_branch') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogOp definitions
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

::mlir::Value LogOpAdaptor::input1() {
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

::mlir::Value LogOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogOp::input1Mutable() {
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

::mlir::Value LogOp::output() {
  return *getODSResults(0).begin();
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
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

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalAndOp definitions
//===----------------------------------------------------------------------===//

LogicalAndOpAdaptor::LogicalAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogicalAndOpAdaptor::LogicalAndOpAdaptor(LogicalAndOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogicalAndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogicalAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalAndOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalAndOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LogicalAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalAndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogicalAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalAndOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalAndOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalAndOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LogicalAndOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogicalAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalAndOp::z() {
  return *getODSResults(0).begin();
}

void LogicalAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void LogicalAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalAndOp::verify() {
  if (failed(LogicalAndOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogicalAndOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalLeftShiftOp definitions
//===----------------------------------------------------------------------===//

LogicalLeftShiftOpAdaptor::LogicalLeftShiftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogicalLeftShiftOpAdaptor::LogicalLeftShiftOpAdaptor(LogicalLeftShiftOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogicalLeftShiftOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogicalLeftShiftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalLeftShiftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalLeftShiftOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalLeftShiftOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LogicalLeftShiftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalLeftShiftOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogicalLeftShiftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalLeftShiftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalLeftShiftOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalLeftShiftOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalLeftShiftOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LogicalLeftShiftOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogicalLeftShiftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalLeftShiftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalLeftShiftOp::output() {
  return *getODSResults(0).begin();
}

void LogicalLeftShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void LogicalLeftShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalLeftShiftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalLeftShiftOp::verify() {
  if (failed(LogicalLeftShiftOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogicalLeftShiftOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalNotOp definitions
//===----------------------------------------------------------------------===//

LogicalNotOpAdaptor::LogicalNotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogicalNotOpAdaptor::LogicalNotOpAdaptor(LogicalNotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogicalNotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogicalNotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalNotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalNotOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LogicalNotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalNotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogicalNotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalNotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalNotOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogicalNotOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogicalNotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalNotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalNotOp::output() {
  return *getODSResults(0).begin();
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalNotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult LogicalNotOp::verify() {
  if (failed(LogicalNotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogicalNotOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalOrOp definitions
//===----------------------------------------------------------------------===//

LogicalOrOpAdaptor::LogicalOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogicalOrOpAdaptor::LogicalOrOpAdaptor(LogicalOrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogicalOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogicalOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalOrOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalOrOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LogicalOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogicalOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalOrOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalOrOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalOrOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LogicalOrOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogicalOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalOrOp::z() {
  return *getODSResults(0).begin();
}

void LogicalOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void LogicalOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalOrOp::verify() {
  if (failed(LogicalOrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogicalOrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalRightShiftOp definitions
//===----------------------------------------------------------------------===//

LogicalRightShiftOpAdaptor::LogicalRightShiftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogicalRightShiftOpAdaptor::LogicalRightShiftOpAdaptor(LogicalRightShiftOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogicalRightShiftOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogicalRightShiftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalRightShiftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalRightShiftOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalRightShiftOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LogicalRightShiftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalRightShiftOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogicalRightShiftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalRightShiftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalRightShiftOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalRightShiftOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalRightShiftOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LogicalRightShiftOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogicalRightShiftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalRightShiftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalRightShiftOp::output() {
  return *getODSResults(0).begin();
}

void LogicalRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void LogicalRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalRightShiftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalRightShiftOp::verify() {
  if (failed(LogicalRightShiftOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogicalRightShiftOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalXorOp definitions
//===----------------------------------------------------------------------===//

LogicalXorOpAdaptor::LogicalXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LogicalXorOpAdaptor::LogicalXorOpAdaptor(LogicalXorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LogicalXorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogicalXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalXorOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalXorOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LogicalXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalXorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> LogicalXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalXorOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalXorOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalXorOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LogicalXorOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogicalXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalXorOp::z() {
  return *getODSResults(0).begin();
}

void LogicalXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void LogicalXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalXorOp::verify() {
  if (failed(LogicalXorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void LogicalXorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MatMulOp definitions
//===----------------------------------------------------------------------===//

MatMulOpAdaptor::MatMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatMulOpAdaptor::MatMulOpAdaptor(MatMulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatMulOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatMulOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MatMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::MatMulOpQuantizationAttr MatMulOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::MatMulOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::MatMulOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult MatMulOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::MatMulOpQuantizationAttr>()))) return emitError(loc, "'tosa.matmul' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for MatMulOp quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MatMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatMulOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatMulOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatMulOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MatMulOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MatMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatMulOp::c() {
  return *getODSResults(0).begin();
}

mlir::tosa::MatMulOpQuantizationAttr MatMulOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::MatMulOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::MatMulOpQuantizationAttr> MatMulOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::MatMulOpQuantizationAttr>(attr) : (::llvm::None);
}

void MatMulOp::quantization_infoAttr(mlir::tosa::MatMulOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute MatMulOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value a, Value b) {
    buildMatMulOpWithQuantInfo(odsBuilder, odsState, outputType,
                               a, b);
  
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type c, ::mlir::Value a, ::mlir::Value b, /*optional*/mlir::tosa::MatMulOpQuantizationAttr quantization_info) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(c);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, /*optional*/mlir::tosa::MatMulOpQuantizationAttr quantization_info) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatMulOp::verify() {
  if (failed(MatMulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MatMulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MaxPool2dOp definitions
//===----------------------------------------------------------------------===//

MaxPool2dOpAdaptor::MaxPool2dOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MaxPool2dOpAdaptor::MaxPool2dOpAdaptor(MaxPool2dOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MaxPool2dOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaxPool2dOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaxPool2dOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxPool2dOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MaxPool2dOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::kernel() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("kernel").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::pad() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("pad").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult MaxPool2dOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_kernel = odsAttrs.get("kernel");
  if (!tblgen_kernel) return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'kernel'");
    if (!((((tblgen_kernel.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_kernel.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_kernel.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.max_pool2d' op ""attribute 'kernel' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.max_pool2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_pad = odsAttrs.get("pad");
  if (!tblgen_pad) return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'pad'");
    if (!((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4)))) return emitError(loc, "'tosa.max_pool2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> MaxPool2dOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaxPool2dOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxPool2dOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MaxPool2dOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaxPool2dOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaxPool2dOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxPool2dOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr MaxPool2dOp::kernelAttr() {
  return (*this)->getAttr("kernel").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MaxPool2dOp::kernel() {
  auto attr = kernelAttr();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MaxPool2dOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOp::padAttr() {
  return (*this)->getAttr("pad").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MaxPool2dOp::pad() {
  auto attr = padAttr();
  return attr;
}

void MaxPool2dOp::kernelAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("kernel", attr);
}

void MaxPool2dOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void MaxPool2dOp::padAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("pad", attr);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad) {
  odsState.addOperands(input);
  odsState.addAttribute("kernel", kernel);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("pad", pad);
  odsState.addTypes(output);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad) {
  odsState.addOperands(input);
  odsState.addAttribute("kernel", kernel);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("pad", pad);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaxPool2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaxPool2dOp::verify() {
  if (failed(MaxPool2dOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MaxPool2dOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MaximumOp definitions
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

::mlir::Value MaximumOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOpAdaptor::input2() {
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

::mlir::Value MaximumOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MaximumOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaximumOp::input2Mutable() {
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

::mlir::Value MaximumOp::output() {
  return *getODSResults(0).begin();
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
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

::mlir::LogicalResult MaximumOp::verify() {
  if (failed(MaximumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MaximumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MinimumOp definitions
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

::mlir::Value MinimumOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOpAdaptor::input2() {
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

::mlir::Value MinimumOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MinimumOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MinimumOp::input2Mutable() {
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

::mlir::Value MinimumOp::output() {
  return *getODSResults(0).begin();
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
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

::mlir::LogicalResult MinimumOp::verify() {
  if (failed(MinimumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MinimumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MulOp definitions
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

::mlir::Value MulOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MulOpAdaptor::shift() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("shift").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_shift = odsAttrs.get("shift");
  if (!tblgen_shift) return emitError(loc, "'tosa.mul' op ""requires attribute 'shift'");
    if (!(((tblgen_shift.isa<::mlir::IntegerAttr>())) && ((tblgen_shift.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'tosa.mul' op ""attribute 'shift' failed to satisfy constraint: 32-bit signless integer attribute");
  }
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

::mlir::Value MulOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MulOp::input2Mutable() {
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

::mlir::Value MulOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MulOp::shiftAttr() {
  return (*this)->getAttr("shift").template cast<::mlir::IntegerAttr>();
}

uint32_t MulOp::shift() {
  auto attr = shiftAttr();
  return attr.getValue().getZExtValue();
}

void MulOp::shiftAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("shift", attr);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("shift", shift);
  odsState.addTypes(output);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("shift", shift);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, uint32_t shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("shift", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift));
  odsState.addTypes(output);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, uint32_t shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute("shift", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift));
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

::mlir::LogicalResult MulOp::verify() {
  if (failed(MulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void MulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::NegateOp definitions
//===----------------------------------------------------------------------===//

NegateOpAdaptor::NegateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

NegateOpAdaptor::NegateOpAdaptor(NegateOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange NegateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> NegateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NegateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NegateOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr NegateOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::UnaryOpQuantizationAttr NegateOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::UnaryOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult NegateOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::UnaryOpQuantizationAttr>()))) return emitError(loc, "'tosa.negate' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for UnaryOp quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> NegateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NegateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NegateOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange NegateOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> NegateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NegateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NegateOp::output() {
  return *getODSResults(0).begin();
}

mlir::tosa::UnaryOpQuantizationAttr NegateOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::UnaryOpQuantizationAttr> NegateOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::UnaryOpQuantizationAttr>(attr) : (::llvm::None);
}

void NegateOp::quantization_infoAttr(mlir::tosa::UnaryOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute NegateOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input) {
    buildUnaryOpWithQuantInfo(odsBuilder, odsState, outputType, input);
  
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NegateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes({input1.getType()});

}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult NegateOp::verify() {
  if (failed(NegateOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void NegateOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::PadOp definitions
//===----------------------------------------------------------------------===//

PadOpAdaptor::PadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PadOpAdaptor::PadOpAdaptor(PadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PadOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value PadOpAdaptor::padding() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr PadOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::PadOpQuantizationAttr PadOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::PadOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::PadOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult PadOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::PadOpQuantizationAttr>()))) return emitError(loc, "'tosa.pad' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for PadOp quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PadOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value PadOp::padding() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PadOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PadOp::paddingMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PadOp::output() {
  return *getODSResults(0).begin();
}

mlir::tosa::PadOpQuantizationAttr PadOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::PadOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::PadOpQuantizationAttr> PadOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::PadOpQuantizationAttr>(attr) : (::llvm::None);
}

void PadOp::quantization_infoAttr(mlir::tosa::PadOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute PadOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value paddings) {
    buildPadOpWithQuantInfo(odsBuilder, odsState, outputType,
                            input, paddings);
  
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value padding, /*optional*/mlir::tosa::PadOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  odsState.addOperands(padding);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value padding, /*optional*/mlir::tosa::PadOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  odsState.addOperands(padding);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PadOp::verify() {
  if (failed(PadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void PadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::PlaceholderOp definitions
//===----------------------------------------------------------------------===//

PlaceholderOpAdaptor::PlaceholderOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PlaceholderOpAdaptor::PlaceholderOpAdaptor(PlaceholderOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PlaceholderOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PlaceholderOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PlaceholderOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr PlaceholderOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PlaceholderOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> PlaceholderOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PlaceholderOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> PlaceholderOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PlaceholderOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PlaceholderOp::output() {
  return *getODSResults(0).begin();
}

void PlaceholderOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output) {
  odsState.addTypes(output);
}

void PlaceholderOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PlaceholderOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PlaceholderOp::verify() {
  if (failed(PlaceholderOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void PlaceholderOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::PowOp definitions
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

::mlir::Value PowOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOpAdaptor::input2() {
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

::mlir::Value PowOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PowOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PowOp::input2Mutable() {
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

::mlir::Value PowOp::z() {
  return *getODSResults(0).begin();
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
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

::mlir::LogicalResult PowOp::verify() {
  if (failed(PowOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void PowOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReciprocalOp definitions
//===----------------------------------------------------------------------===//

ReciprocalOpAdaptor::ReciprocalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReciprocalOpAdaptor::ReciprocalOpAdaptor(ReciprocalOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReciprocalOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReciprocalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReciprocalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReciprocalOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReciprocalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReciprocalOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReciprocalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReciprocalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReciprocalOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReciprocalOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReciprocalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReciprocalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReciprocalOp::output() {
  return *getODSResults(0).begin();
}

void ReciprocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void ReciprocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReciprocalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ReciprocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void ReciprocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ReciprocalOp::verify() {
  if (failed(ReciprocalOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReciprocalOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceAllOp definitions
//===----------------------------------------------------------------------===//

ReduceAllOpAdaptor::ReduceAllOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReduceAllOpAdaptor::ReduceAllOpAdaptor(ReduceAllOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReduceAllOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReduceAllOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceAllOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAllOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReduceAllOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceAllOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReduceAllOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_all' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reduce_all' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReduceAllOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceAllOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAllOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceAllOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReduceAllOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceAllOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAllOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceAllOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReduceAllOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceAllOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAllOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceAllOp::verify() {
  if (failed(ReduceAllOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReduceAllOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceAnyOp definitions
//===----------------------------------------------------------------------===//

ReduceAnyOpAdaptor::ReduceAnyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReduceAnyOpAdaptor::ReduceAnyOpAdaptor(ReduceAnyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReduceAnyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReduceAnyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceAnyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAnyOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReduceAnyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceAnyOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReduceAnyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_any' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reduce_any' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReduceAnyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceAnyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAnyOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceAnyOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReduceAnyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceAnyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAnyOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceAnyOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReduceAnyOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceAnyOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAnyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceAnyOp::verify() {
  if (failed(ReduceAnyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReduceAnyOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceMaxOp definitions
//===----------------------------------------------------------------------===//

ReduceMaxOpAdaptor::ReduceMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReduceMaxOpAdaptor::ReduceMaxOpAdaptor(ReduceMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReduceMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReduceMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMaxOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReduceMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceMaxOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReduceMaxOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_max' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reduce_max' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReduceMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMaxOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceMaxOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReduceMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMaxOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceMaxOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReduceMaxOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceMaxOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceMaxOp::verify() {
  if (failed(ReduceMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReduceMaxOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceMinOp definitions
//===----------------------------------------------------------------------===//

ReduceMinOpAdaptor::ReduceMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReduceMinOpAdaptor::ReduceMinOpAdaptor(ReduceMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReduceMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReduceMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMinOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReduceMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceMinOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReduceMinOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_min' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reduce_min' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReduceMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMinOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceMinOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReduceMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMinOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceMinOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReduceMinOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceMinOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceMinOp::verify() {
  if (failed(ReduceMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReduceMinOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceProdOp definitions
//===----------------------------------------------------------------------===//

ReduceProdOpAdaptor::ReduceProdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReduceProdOpAdaptor::ReduceProdOpAdaptor(ReduceProdOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReduceProdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReduceProdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceProdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceProdOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReduceProdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceProdOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReduceProdOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_prod' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reduce_prod' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReduceProdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceProdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceProdOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceProdOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReduceProdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceProdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceProdOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceProdOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReduceProdOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceProdOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceProdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceProdOp::verify() {
  if (failed(ReduceProdOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReduceProdOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceSumOp definitions
//===----------------------------------------------------------------------===//

ReduceSumOpAdaptor::ReduceSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReduceSumOpAdaptor::ReduceSumOpAdaptor(ReduceSumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReduceSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReduceSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceSumOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReduceSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceSumOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReduceSumOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_sum' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reduce_sum' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReduceSumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceSumOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceSumOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReduceSumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceSumOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceSumOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReduceSumOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceSumOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceSumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceSumOp::verify() {
  if (failed(ReduceSumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReduceSumOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReluNOp definitions
//===----------------------------------------------------------------------===//

ReluNOpAdaptor::ReluNOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReluNOpAdaptor::ReluNOpAdaptor(ReluNOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReluNOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReluNOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReluNOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReluNOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReluNOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReluNOpAdaptor::max_int() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("max_int").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::FloatAttr ReluNOpAdaptor::max_fp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("max_fp").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::LogicalResult ReluNOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_max_int = odsAttrs.get("max_int");
  if (!tblgen_max_int) return emitError(loc, "'tosa.reluN' op ""requires attribute 'max_int'");
    if (!(((tblgen_max_int.isa<::mlir::IntegerAttr>())) && ((tblgen_max_int.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reluN' op ""attribute 'max_int' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_max_fp = odsAttrs.get("max_fp");
  if (!tblgen_max_fp) return emitError(loc, "'tosa.reluN' op ""requires attribute 'max_fp'");
    if (!(((tblgen_max_fp.isa<::mlir::FloatAttr>())) && ((tblgen_max_fp.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'tosa.reluN' op ""attribute 'max_fp' failed to satisfy constraint: 32-bit float attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReluNOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReluNOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReluNOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReluNOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReluNOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReluNOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReluNOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReluNOp::max_intAttr() {
  return (*this)->getAttr("max_int").template cast<::mlir::IntegerAttr>();
}

uint64_t ReluNOp::max_int() {
  auto attr = max_intAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ReluNOp::max_fpAttr() {
  return (*this)->getAttr("max_fp").template cast<::mlir::FloatAttr>();
}

::llvm::APFloat ReluNOp::max_fp() {
  auto attr = max_fpAttr();
  return attr.getValue();
}

void ReluNOp::max_intAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("max_int", attr);
}

void ReluNOp::max_fpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr("max_fp", attr);
}

void ReluNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("max_int", max_int);
  odsState.addAttribute("max_fp", max_fp);
  odsState.addTypes(output);
}

void ReluNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("max_int", max_int);
  odsState.addAttribute("max_fp", max_fp);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReluNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t max_int, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("max_int", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int));
  odsState.addAttribute("max_fp", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp));
  odsState.addTypes(output);
}

void ReluNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t max_int, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("max_int", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int));
  odsState.addAttribute("max_fp", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReluNOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ReluNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute("max_int", max_int);
  odsState.addAttribute("max_fp", max_fp);
  odsState.addTypes({input.getType()});

}

void ReluNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult ReluNOp::verify() {
  if (failed(ReluNOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReluNOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::RescaleOp definitions
//===----------------------------------------------------------------------===//

RescaleOpAdaptor::RescaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

RescaleOpAdaptor::RescaleOpAdaptor(RescaleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange RescaleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RescaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RescaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RescaleOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr RescaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr RescaleOpAdaptor::input_zp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("input_zp").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr RescaleOpAdaptor::output_zp() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("output_zp").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr RescaleOpAdaptor::multiplier() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("multiplier").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr RescaleOpAdaptor::shift() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("shift").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::BoolAttr RescaleOpAdaptor::scale32() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("scale32").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::BoolAttr RescaleOpAdaptor::double_round() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("double_round").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::BoolAttr RescaleOpAdaptor::per_channel() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("per_channel").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::LogicalResult RescaleOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_input_zp = odsAttrs.get("input_zp");
  if (!tblgen_input_zp) return emitError(loc, "'tosa.rescale' op ""requires attribute 'input_zp'");
    if (!(((tblgen_input_zp.isa<::mlir::IntegerAttr>())) && ((tblgen_input_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'tosa.rescale' op ""attribute 'input_zp' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_output_zp = odsAttrs.get("output_zp");
  if (!tblgen_output_zp) return emitError(loc, "'tosa.rescale' op ""requires attribute 'output_zp'");
    if (!(((tblgen_output_zp.isa<::mlir::IntegerAttr>())) && ((tblgen_output_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'tosa.rescale' op ""attribute 'output_zp' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_multiplier = odsAttrs.get("multiplier");
  if (!tblgen_multiplier) return emitError(loc, "'tosa.rescale' op ""requires attribute 'multiplier'");
    if (!(((tblgen_multiplier.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_multiplier.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))); })))) return emitError(loc, "'tosa.rescale' op ""attribute 'multiplier' failed to satisfy constraint: 32-bit integer array attribute");
  }
  {
  auto tblgen_shift = odsAttrs.get("shift");
  if (!tblgen_shift) return emitError(loc, "'tosa.rescale' op ""requires attribute 'shift'");
    if (!(((tblgen_shift.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_shift.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))); })))) return emitError(loc, "'tosa.rescale' op ""attribute 'shift' failed to satisfy constraint: 32-bit integer array attribute");
  }
  {
  auto tblgen_scale32 = odsAttrs.get("scale32");
  if (!tblgen_scale32) return emitError(loc, "'tosa.rescale' op ""requires attribute 'scale32'");
    if (!((tblgen_scale32.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tosa.rescale' op ""attribute 'scale32' failed to satisfy constraint: bool attribute");
  }
  {
  auto tblgen_double_round = odsAttrs.get("double_round");
  if (!tblgen_double_round) return emitError(loc, "'tosa.rescale' op ""requires attribute 'double_round'");
    if (!((tblgen_double_round.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tosa.rescale' op ""attribute 'double_round' failed to satisfy constraint: bool attribute");
  }
  {
  auto tblgen_per_channel = odsAttrs.get("per_channel");
  if (!tblgen_per_channel) return emitError(loc, "'tosa.rescale' op ""requires attribute 'per_channel'");
    if (!((tblgen_per_channel.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tosa.rescale' op ""attribute 'per_channel' failed to satisfy constraint: bool attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> RescaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RescaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RescaleOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange RescaleOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RescaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RescaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RescaleOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr RescaleOp::input_zpAttr() {
  return (*this)->getAttr("input_zp").template cast<::mlir::IntegerAttr>();
}

uint32_t RescaleOp::input_zp() {
  auto attr = input_zpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr RescaleOp::output_zpAttr() {
  return (*this)->getAttr("output_zp").template cast<::mlir::IntegerAttr>();
}

uint32_t RescaleOp::output_zp() {
  auto attr = output_zpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr RescaleOp::multiplierAttr() {
  return (*this)->getAttr("multiplier").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr RescaleOp::multiplier() {
  auto attr = multiplierAttr();
  return attr;
}

::mlir::ArrayAttr RescaleOp::shiftAttr() {
  return (*this)->getAttr("shift").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr RescaleOp::shift() {
  auto attr = shiftAttr();
  return attr;
}

::mlir::BoolAttr RescaleOp::scale32Attr() {
  return (*this)->getAttr("scale32").template cast<::mlir::BoolAttr>();
}

bool RescaleOp::scale32() {
  auto attr = scale32Attr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOp::double_roundAttr() {
  return (*this)->getAttr("double_round").template cast<::mlir::BoolAttr>();
}

bool RescaleOp::double_round() {
  auto attr = double_roundAttr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOp::per_channelAttr() {
  return (*this)->getAttr("per_channel").template cast<::mlir::BoolAttr>();
}

bool RescaleOp::per_channel() {
  auto attr = per_channelAttr();
  return attr.getValue();
}

void RescaleOp::input_zpAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("input_zp", attr);
}

void RescaleOp::output_zpAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("output_zp", attr);
}

void RescaleOp::multiplierAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("multiplier", attr);
}

void RescaleOp::shiftAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("shift", attr);
}

void RescaleOp::scale32Attr(::mlir::BoolAttr attr) {
  (*this)->setAttr("scale32", attr);
}

void RescaleOp::double_roundAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("double_round", attr);
}

void RescaleOp::per_channelAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr("per_channel", attr);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute("input_zp", input_zp);
  odsState.addAttribute("output_zp", output_zp);
  odsState.addAttribute("multiplier", multiplier);
  odsState.addAttribute("shift", shift);
  odsState.addAttribute("scale32", scale32);
  odsState.addAttribute("double_round", double_round);
  odsState.addAttribute("per_channel", per_channel);
  odsState.addTypes(output);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute("input_zp", input_zp);
  odsState.addAttribute("output_zp", output_zp);
  odsState.addAttribute("multiplier", multiplier);
  odsState.addAttribute("shift", shift);
  odsState.addAttribute("scale32", scale32);
  odsState.addAttribute("double_round", double_round);
  odsState.addAttribute("per_channel", per_channel);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint32_t input_zp, uint32_t output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, bool scale32, bool double_round, bool per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute("input_zp", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), input_zp));
  odsState.addAttribute("output_zp", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), output_zp));
  odsState.addAttribute("multiplier", multiplier);
  odsState.addAttribute("shift", shift);
  odsState.addAttribute("scale32", odsBuilder.getBoolAttr(scale32));
  odsState.addAttribute("double_round", odsBuilder.getBoolAttr(double_round));
  odsState.addAttribute("per_channel", odsBuilder.getBoolAttr(per_channel));
  odsState.addTypes(output);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t input_zp, uint32_t output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, bool scale32, bool double_round, bool per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute("input_zp", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), input_zp));
  odsState.addAttribute("output_zp", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), output_zp));
  odsState.addAttribute("multiplier", multiplier);
  odsState.addAttribute("shift", shift);
  odsState.addAttribute("scale32", odsBuilder.getBoolAttr(scale32));
  odsState.addAttribute("double_round", odsBuilder.getBoolAttr(double_round));
  odsState.addAttribute("per_channel", odsBuilder.getBoolAttr(per_channel));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RescaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RescaleOp::verify() {
  if (failed(RescaleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void RescaleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReshapeOp definitions
//===----------------------------------------------------------------------===//

ReshapeOpAdaptor::ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReshapeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReshapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReshapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReshapeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ReshapeOpAdaptor::new_shape() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("new_shape").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_new_shape = odsAttrs.get("new_shape");
  if (!tblgen_new_shape) return emitError(loc, "'tosa.reshape' op ""requires attribute 'new_shape'");
    if (!(((tblgen_new_shape.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_new_shape.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tosa.reshape' op ""attribute 'new_shape' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReshapeOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ReshapeOp::new_shapeAttr() {
  return (*this)->getAttr("new_shape").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ReshapeOp::new_shape() {
  auto attr = new_shapeAttr();
  return attr;
}

void ReshapeOp::new_shapeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("new_shape", attr);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.addAttribute("new_shape", new_shape);
  odsState.addTypes(output);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.addAttribute("new_shape", new_shape);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verify() {
  if (failed(ReshapeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReshapeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ResizeOp definitions
//===----------------------------------------------------------------------===//

ResizeOpAdaptor::ResizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ResizeOpAdaptor::ResizeOpAdaptor(ResizeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ResizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResizeOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ResizeOpAdaptor::output_size() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("output_size").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::offset() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offset").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::IntegerAttr ResizeOpAdaptor::shift() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("shift").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::StringAttr ResizeOpAdaptor::mode() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("mode").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult ResizeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_output_size = odsAttrs.get("output_size");
  if (!tblgen_output_size) return emitError(loc, "'tosa.resize' op ""requires attribute 'output_size'");
    if (!((((tblgen_output_size.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_size.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_output_size.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.resize' op ""attribute 'output_size' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.resize' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.resize' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_offset = odsAttrs.get("offset");
  if (!tblgen_offset) return emitError(loc, "'tosa.resize' op ""requires attribute 'offset'");
    if (!((((tblgen_offset.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_offset.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_offset.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.resize' op ""attribute 'offset' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_shift = odsAttrs.get("shift");
  if (!tblgen_shift) return emitError(loc, "'tosa.resize' op ""requires attribute 'shift'");
    if (!(((tblgen_shift.isa<::mlir::IntegerAttr>())) && ((tblgen_shift.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) return emitError(loc, "'tosa.resize' op ""attribute 'shift' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
  auto tblgen_mode = odsAttrs.get("mode");
  if (!tblgen_mode) return emitError(loc, "'tosa.resize' op ""requires attribute 'mode'");
    if (!((tblgen_mode.cast<StringAttr>().getValue() == "BILINEAR"  || tblgen_mode.cast<StringAttr>().getValue() == "NEAREST_NEIGHBOR"))) return emitError(loc, "'tosa.resize' op ""attribute 'mode' failed to satisfy constraint: Supported resize/upsampling strategies");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ResizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResizeOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResizeOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ResizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResizeOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ResizeOp::output_sizeAttr() {
  return (*this)->getAttr("output_size").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ResizeOp::output_size() {
  auto attr = output_sizeAttr();
  return attr;
}

::mlir::ArrayAttr ResizeOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ResizeOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr ResizeOp::offsetAttr() {
  return (*this)->getAttr("offset").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ResizeOp::offset() {
  auto attr = offsetAttr();
  return attr;
}

::mlir::IntegerAttr ResizeOp::shiftAttr() {
  return (*this)->getAttr("shift").template cast<::mlir::IntegerAttr>();
}

uint32_t ResizeOp::shift() {
  auto attr = shiftAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr ResizeOp::modeAttr() {
  return (*this)->getAttr("mode").template cast<::mlir::StringAttr>();
}

::llvm::StringRef ResizeOp::mode() {
  auto attr = modeAttr();
  return attr.getValue();
}

void ResizeOp::output_sizeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("output_size", attr);
}

void ResizeOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void ResizeOp::offsetAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("offset", attr);
}

void ResizeOp::shiftAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("shift", attr);
}

void ResizeOp::modeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr("mode", attr);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr output_size, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr offset, ::mlir::IntegerAttr shift, ::mlir::StringAttr mode) {
  odsState.addOperands(input);
  odsState.addAttribute("output_size", output_size);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("offset", offset);
  odsState.addAttribute("shift", shift);
  odsState.addAttribute("mode", mode);
  odsState.addTypes(output);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr output_size, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr offset, ::mlir::IntegerAttr shift, ::mlir::StringAttr mode) {
  odsState.addOperands(input);
  odsState.addAttribute("output_size", output_size);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("offset", offset);
  odsState.addAttribute("shift", shift);
  odsState.addAttribute("mode", mode);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr output_size, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr offset, uint32_t shift, ::llvm::StringRef mode) {
  odsState.addOperands(input);
  odsState.addAttribute("output_size", output_size);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("offset", offset);
  odsState.addAttribute("shift", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift));
  odsState.addAttribute("mode", odsBuilder.getStringAttr(mode));
  odsState.addTypes(output);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr output_size, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr offset, uint32_t shift, ::llvm::StringRef mode) {
  odsState.addOperands(input);
  odsState.addAttribute("output_size", output_size);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("offset", offset);
  odsState.addAttribute("shift", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift));
  odsState.addAttribute("mode", odsBuilder.getStringAttr(mode));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResizeOp::verify() {
  if (failed(ResizeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ResizeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReverseOp definitions
//===----------------------------------------------------------------------===//

ReverseOpAdaptor::ReverseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReverseOpAdaptor::ReverseOpAdaptor(ReverseOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReverseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReverseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReverseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReverseOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ReverseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReverseOpAdaptor::axis() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("axis").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult ReverseOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_axis = odsAttrs.get("axis");
  if (!tblgen_axis) return emitError(loc, "'tosa.reverse' op ""requires attribute 'axis'");
    if (!(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tosa.reverse' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> ReverseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReverseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReverseOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReverseOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReverseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReverseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReverseOp::output() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReverseOp::axisAttr() {
  return (*this)->getAttr("axis").template cast<::mlir::IntegerAttr>();
}

uint64_t ReverseOp::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

void ReverseOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr("axis", attr);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  odsState.addTypes(output);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute("axis", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReverseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReverseOp::verify() {
  if (failed(ReverseOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void ReverseOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::RsqrtOp definitions
//===----------------------------------------------------------------------===//

RsqrtOpAdaptor::RsqrtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

RsqrtOpAdaptor::RsqrtOpAdaptor(RsqrtOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange RsqrtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RsqrtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RsqrtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RsqrtOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr RsqrtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult RsqrtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> RsqrtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RsqrtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RsqrtOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange RsqrtOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RsqrtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RsqrtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RsqrtOp::output() {
  return *getODSResults(0).begin();
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes({input1.getType()});

}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult RsqrtOp::verify() {
  if (failed(RsqrtOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void RsqrtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SelectOp definitions
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

::mlir::Value SelectOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOpAdaptor::input2() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOpAdaptor::input3() {
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

::mlir::Value SelectOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOp::input3() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SelectOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SelectOp::input2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SelectOp::input3Mutable() {
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

::mlir::Value SelectOp::output() {
  return *getODSResults(0).begin();
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addOperands(input3);
  odsState.addTypes(output);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addOperands(input3);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SelectOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SigmoidOp definitions
//===----------------------------------------------------------------------===//

SigmoidOpAdaptor::SigmoidOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SigmoidOpAdaptor::SigmoidOpAdaptor(SigmoidOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SigmoidOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SigmoidOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SigmoidOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SigmoidOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SigmoidOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SigmoidOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SigmoidOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SigmoidOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SigmoidOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SigmoidOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SigmoidOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SigmoidOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SigmoidOp::output() {
  return *getODSResults(0).begin();
}

void SigmoidOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void SigmoidOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SigmoidOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SigmoidOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes({input.getType()});

}

void SigmoidOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SigmoidOp::verify() {
  if (failed(SigmoidOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SigmoidOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SliceOp definitions
//===----------------------------------------------------------------------===//

SliceOpAdaptor::SliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SliceOpAdaptor::SliceOpAdaptor(SliceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SliceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SliceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SliceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SliceOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SliceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SliceOpAdaptor::start() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("start").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SliceOpAdaptor::size() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("size").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SliceOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_start = odsAttrs.get("start");
  if (!tblgen_start) return emitError(loc, "'tosa.slice' op ""requires attribute 'start'");
    if (!(((tblgen_start.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_start.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tosa.slice' op ""attribute 'start' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_size = odsAttrs.get("size");
  if (!tblgen_size) return emitError(loc, "'tosa.slice' op ""requires attribute 'size'");
    if (!(((tblgen_size.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_size.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tosa.slice' op ""attribute 'size' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> SliceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SliceOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SliceOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SliceOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr SliceOp::startAttr() {
  return (*this)->getAttr("start").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SliceOp::start() {
  auto attr = startAttr();
  return attr;
}

::mlir::ArrayAttr SliceOp::sizeAttr() {
  return (*this)->getAttr("size").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SliceOp::size() {
  auto attr = sizeAttr();
  return attr;
}

void SliceOp::startAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("start", attr);
}

void SliceOp::sizeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("size", attr);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr start, ::mlir::ArrayAttr size) {
  odsState.addOperands(input);
  odsState.addAttribute("start", start);
  odsState.addAttribute("size", size);
  odsState.addTypes(output);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr start, ::mlir::ArrayAttr size) {
  odsState.addOperands(input);
  odsState.addAttribute("start", start);
  odsState.addAttribute("size", size);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SliceOp::verify() {
  if (failed(SliceOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SliceOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SubOp definitions
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

::mlir::Value SubOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOpAdaptor::input2() {
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

::mlir::Value SubOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOp::input2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SubOp::input2Mutable() {
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

::mlir::Value SubOp::output() {
  return *getODSResults(0).begin();
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
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

::mlir::LogicalResult SubOp::verify() {
  if (failed(SubOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void SubOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TableOp definitions
//===----------------------------------------------------------------------===//

TableOpAdaptor::TableOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TableOpAdaptor::TableOpAdaptor(TableOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TableOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TableOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TableOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TableOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value TableOpAdaptor::table() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr TableOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TableOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TableOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TableOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TableOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value TableOp::table() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange TableOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange TableOp::tableMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TableOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TableOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TableOp::output() {
  return *getODSResults(0).begin();
}

void TableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value table) {
  odsState.addOperands(input);
  odsState.addOperands(table);
  odsState.addTypes(output);
}

void TableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value table) {
  odsState.addOperands(input);
  odsState.addOperands(table);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TableOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TableOp::verify() {
  if (failed(TableOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void TableOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TanhOp definitions
//===----------------------------------------------------------------------===//

TanhOpAdaptor::TanhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TanhOpAdaptor::TanhOpAdaptor(TanhOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TanhOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TanhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TanhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanhOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TanhOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TanhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TanhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TanhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanhOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TanhOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TanhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TanhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanhOp::output() {
  return *getODSResults(0).begin();
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes({input.getType()});

}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult TanhOp::verify() {
  if (failed(TanhOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void TanhOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TileOp definitions
//===----------------------------------------------------------------------===//

TileOpAdaptor::TileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TileOpAdaptor::TileOpAdaptor(TileOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TileOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TileOpAdaptor::multiples() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("multiples").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult TileOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_multiples = odsAttrs.get("multiples");
  if (!tblgen_multiples) return emitError(loc, "'tosa.tile' op ""requires attribute 'multiples'");
    if (!(((tblgen_multiples.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_multiples.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tosa.tile' op ""attribute 'multiples' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TileOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TileOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TileOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TileOp::multiplesAttr() {
  return (*this)->getAttr("multiples").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileOp::multiples() {
  auto attr = multiplesAttr();
  return attr;
}

void TileOp::multiplesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("multiples", attr);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::ArrayAttr multiples) {
  odsState.addOperands(input1);
  odsState.addAttribute("multiples", multiples);
  odsState.addTypes(output);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::ArrayAttr multiples) {
  odsState.addOperands(input1);
  odsState.addAttribute("multiples", multiples);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TileOp::verify() {
  if (failed(TileOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void TileOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TransposeConv2DOp definitions
//===----------------------------------------------------------------------===//

TransposeConv2DOpAdaptor::TransposeConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TransposeConv2DOpAdaptor::TransposeConv2DOpAdaptor(TransposeConv2DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TransposeConv2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TransposeConv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeConv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeConv2DOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeConv2DOpAdaptor::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value TransposeConv2DOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr TransposeConv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::out_pad() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("out_pad").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::stride() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("stride").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilation").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::out_shape() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("out_shape").cast<::mlir::ArrayAttr>();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr TransposeConv2DOpAdaptor::quantization_info() {
  assert(odsAttrs && "no attributes when constructing adapter");
  mlir::tosa::ConvOpQuantizationAttr attr = odsAttrs.get("quantization_info").dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::mlir::LogicalResult TransposeConv2DOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_out_pad = odsAttrs.get("out_pad");
  if (!tblgen_out_pad) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'out_pad'");
    if (!((((tblgen_out_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_out_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_out_pad.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'out_pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_stride = odsAttrs.get("stride");
  if (!tblgen_stride) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'stride'");
    if (!((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_dilation = odsAttrs.get("dilation");
  if (!tblgen_dilation) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'dilation'");
    if (!((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 2)))) return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");
  }
  {
  auto tblgen_out_shape = odsAttrs.get("out_shape");
  if (!tblgen_out_shape) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'out_shape'");
    if (!((((tblgen_out_shape.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_out_shape.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))) && ((tblgen_out_shape.cast<::mlir::ArrayAttr>().size() <= 4)))) return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'out_shape' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements");
  }
  {
  auto tblgen_quantization_info = odsAttrs.get("quantization_info");
  if (tblgen_quantization_info) {
    if (!((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>()))) return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  }
  }
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TransposeConv2DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeConv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeConv2DOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeConv2DOp::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value TransposeConv2DOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange TransposeConv2DOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange TransposeConv2DOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange TransposeConv2DOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TransposeConv2DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeConv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeConv2DOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TransposeConv2DOp::out_padAttr() {
  return (*this)->getAttr("out_pad").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::out_pad() {
  auto attr = out_padAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOp::strideAttr() {
  return (*this)->getAttr("stride").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::stride() {
  auto attr = strideAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOp::dilationAttr() {
  return (*this)->getAttr("dilation").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::dilation() {
  auto attr = dilationAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOp::out_shapeAttr() {
  return (*this)->getAttr("out_shape").template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::out_shape() {
  auto attr = out_shapeAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr TransposeConv2DOp::quantization_infoAttr() {
  return (*this)->getAttr("quantization_info").template dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr> TransposeConv2DOp::quantization_info() {
  auto attr = quantization_infoAttr();
  return attr ? ::llvm::Optional<mlir::tosa::ConvOpQuantizationAttr>(attr) : (::llvm::None);
}

void TransposeConv2DOp::out_padAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("out_pad", attr);
}

void TransposeConv2DOp::strideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("stride", attr);
}

void TransposeConv2DOp::dilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("dilation", attr);
}

void TransposeConv2DOp::out_shapeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr("out_shape", attr);
}

void TransposeConv2DOp::quantization_infoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr("quantization_info", attr);
}

::mlir::Attribute TransposeConv2DOp::removeQuantization_infoAttr() {
  return (*this)->removeAttr("quantization_info");
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr outpad, ArrayAttr stride, ArrayAttr dilation, ArrayAttr outputShape) {
    buildTransConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                                  input, weight, bias,
                                  outpad, stride, dilation,
                                  outputShape);
  
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::ArrayAttr out_pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addAttribute("out_pad", out_pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  odsState.addAttribute("out_shape", out_shape);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  odsState.addTypes(output);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::ArrayAttr out_pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addAttribute("out_pad", out_pad);
  odsState.addAttribute("stride", stride);
  odsState.addAttribute("dilation", dilation);
  odsState.addAttribute("out_shape", out_shape);
  if (quantization_info) {
  odsState.addAttribute("quantization_info", quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeConv2DOp::verify() {
  if (failed(TransposeConv2DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void TransposeConv2DOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TransposeOp definitions
//===----------------------------------------------------------------------===//

TransposeOpAdaptor::TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TransposeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOpAdaptor::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeOpAdaptor::perms() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr TransposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOp::input1() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeOp::perms() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange TransposeOp::input1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange TransposeOp::permsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOp::output() {
  return *getODSResults(0).begin();
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value perms) {
  odsState.addOperands(input1);
  odsState.addOperands(perms);
  odsState.addTypes(output);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value perms) {
  odsState.addOperands(input1);
  odsState.addOperands(perms);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeOp::verify() {
  if (failed(TransposeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

void TransposeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::WhileOp definitions
//===----------------------------------------------------------------------===//

WhileOpAdaptor::WhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WhileOpAdaptor::WhileOpAdaptor(WhileOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WhileOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WhileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WhileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WhileOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WhileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange WhileOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &WhileOpAdaptor::cond() {
  return *odsRegions[0];
}

::mlir::Region &WhileOpAdaptor::body() {
  return *odsRegions[1];
}

::mlir::LogicalResult WhileOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}



std::pair<unsigned, unsigned> WhileOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WhileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WhileOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WhileOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> WhileOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range WhileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range WhileOp::output() {
  return getODSResults(0);
}

::mlir::Region &WhileOp::cond() {
  return (*this)->getRegion(0);
}

::mlir::Region &WhileOp::body() {
  return (*this)->getRegion(1);
}

void WhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WhileOp::verify() {
  if (failed(WhileOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('cond') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(1))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::mlir::success();
}







} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::YieldOp definitions
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

::mlir::ValueRange YieldOpAdaptor::inputs() {
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

::mlir::Operation::operand_range YieldOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::inputsMutable() {
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

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void YieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tosa
} // namespace mlir

#endif  // GET_OP_CLASSES

