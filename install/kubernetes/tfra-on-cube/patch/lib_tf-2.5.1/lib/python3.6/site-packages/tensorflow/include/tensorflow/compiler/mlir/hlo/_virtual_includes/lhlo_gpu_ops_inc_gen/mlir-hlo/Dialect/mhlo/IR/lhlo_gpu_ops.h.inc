/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace lmhlo_gpu {
class BatchNormGradOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class BatchNormInferenceOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class BatchNormTrainingOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CholeskyOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardFilterOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedSideInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMMOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMM_BiasOp;
} // namespace lmhlo_gpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormGradOp declarations
//===----------------------------------------------------------------------===//

class BatchNormGradOpAdaptor {
public:
  BatchNormGradOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BatchNormGradOpAdaptor(BatchNormGradOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value grad_output();
  ::mlir::Value grad_operand();
  ::mlir::Value grad_scale();
  ::mlir::Value grad_offset();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::IntegerAttr feature_index();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchNormGradOp : public ::mlir::Op<BatchNormGradOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<8>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchNormGradOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.batch_norm_grad");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value grad_output();
  ::mlir::Value grad_operand();
  ::mlir::Value grad_scale();
  ::mlir::Value grad_offset();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange stddevMutable();
  ::mlir::MutableOperandRange grad_outputMutable();
  ::mlir::MutableOperandRange grad_operandMutable();
  ::mlir::MutableOperandRange grad_scaleMutable();
  ::mlir::MutableOperandRange grad_offsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::IntegerAttr feature_indexAttr();
  uint64_t feature_index();
  void epsilonAttr(::mlir::FloatAttr attr);
  void feature_indexAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormInferenceOp declarations
//===----------------------------------------------------------------------===//

class BatchNormInferenceOpAdaptor {
public:
  BatchNormInferenceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BatchNormInferenceOpAdaptor(BatchNormInferenceOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::IntegerAttr feature_index();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchNormInferenceOp : public ::mlir::Op<BatchNormInferenceOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchNormInferenceOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.batch_norm_inference");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value output();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange stddevMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::IntegerAttr feature_indexAttr();
  uint64_t feature_index();
  void epsilonAttr(::mlir::FloatAttr attr);
  void feature_indexAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormTrainingOp declarations
//===----------------------------------------------------------------------===//

class BatchNormTrainingOpAdaptor {
public:
  BatchNormTrainingOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BatchNormTrainingOpAdaptor(BatchNormTrainingOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value output();
  ::mlir::Value batch_mean();
  ::mlir::Value batch_stddev();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::IntegerAttr feature_index();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchNormTrainingOp : public ::mlir::Op<BatchNormTrainingOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchNormTrainingOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.batch_norm_training");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value output();
  ::mlir::Value batch_mean();
  ::mlir::Value batch_stddev();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange batch_meanMutable();
  ::mlir::MutableOperandRange batch_stddevMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::IntegerAttr feature_indexAttr();
  uint64_t feature_index();
  void epsilonAttr(::mlir::FloatAttr attr);
  void feature_indexAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CholeskyOp declarations
//===----------------------------------------------------------------------===//

class CholeskyOpAdaptor {
public:
  CholeskyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CholeskyOpAdaptor(CholeskyOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::Value info();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr is_lower();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CholeskyOp : public ::mlir::Op<CholeskyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CholeskyOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.cholesky");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::Value info();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  ::mlir::MutableOperandRange infoMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr is_lowerAttr();
  bool is_lower();
  void is_lowerAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardFilterOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardFilterOpAdaptor {
public:
  ConvBackwardFilterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvBackwardFilterOpAdaptor(ConvBackwardFilterOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value d_output();
  ::mlir::Value d_filter();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvBackwardFilterOp : public ::mlir::Op<ConvBackwardFilterOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardFilterOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardfilter");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value d_output();
  ::mlir::Value d_filter();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange d_outputMutable();
  ::mlir::MutableOperandRange d_filterMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardInputOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardInputOpAdaptor {
public:
  ConvBackwardInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvBackwardInputOpAdaptor(ConvBackwardInputOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value d_output();
  ::mlir::Value filter();
  ::mlir::Value d_input();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvBackwardInputOp : public ::mlir::Op<ConvBackwardInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardInputOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardinput");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value d_output();
  ::mlir::Value filter();
  ::mlir::Value d_input();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange d_outputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange d_inputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedOpAdaptor {
public:
  ConvForwardFusedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvForwardFusedOpAdaptor(ConvForwardFusedOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::StringAttr activation_mode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardFusedOp : public ::mlir::Op<ConvForwardFusedOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<5>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedSideInputOpAdaptor {
public:
  ConvForwardFusedSideInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvForwardFusedSideInputOpAdaptor(ConvForwardFusedSideInputOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value side_input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::StringAttr activation_mode();
  ::mlir::FloatAttr side_input_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardFusedSideInputOp : public ::mlir::Op<ConvForwardFusedSideInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedSideInputOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused_with_side_input");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value side_input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange side_inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::FloatAttr side_input_scaleAttr();
  ::llvm::APFloat side_input_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void side_input_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardOpAdaptor {
public:
  ConvForwardOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvForwardOpAdaptor(ConvForwardOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardOp : public ::mlir::Op<ConvForwardOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.conv_forward");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMMOp declarations
//===----------------------------------------------------------------------===//

class GEMMOpAdaptor {
public:
  GEMMOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GEMMOpAdaptor(GEMMOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_real();
  ::mlir::FloatAttr alpha_imag();
  ::mlir::IntegerAttr batch_size();
  ::mlir::IntegerAttr algorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GEMMOp : public ::mlir::Op<GEMMOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMMOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.gemm");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value output();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  void dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbers attr);
  void alpha_realAttr(::mlir::FloatAttr attr);
  void alpha_imagAttr(::mlir::FloatAttr attr);
  void batch_sizeAttr(::mlir::IntegerAttr attr);
  void algorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::IntegerAttr batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::IntegerAttr batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, uint64_t batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, uint64_t batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMM_BiasOp declarations
//===----------------------------------------------------------------------===//

class GEMM_BiasOpAdaptor {
public:
  GEMM_BiasOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GEMM_BiasOpAdaptor(GEMM_BiasOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_real();
  ::mlir::FloatAttr alpha_imag();
  ::mlir::FloatAttr beta();
  ::mlir::IntegerAttr batch_size();
  ::mlir::IntegerAttr algorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GEMM_BiasOp : public ::mlir::Op<GEMM_BiasOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMM_BiasOpAdaptor;
  static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("lmhlo_gpu.gemm_bias");
}
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::FloatAttr betaAttr();
  ::llvm::APFloat beta();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  void dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbers attr);
  void alpha_realAttr(::mlir::FloatAttr attr);
  void alpha_imagAttr(::mlir::FloatAttr attr);
  void betaAttr(::mlir::FloatAttr attr);
  void batch_sizeAttr(::mlir::IntegerAttr attr);
  void algorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, uint64_t batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, uint64_t batch_size, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir

#endif  // GET_OP_CLASSES

