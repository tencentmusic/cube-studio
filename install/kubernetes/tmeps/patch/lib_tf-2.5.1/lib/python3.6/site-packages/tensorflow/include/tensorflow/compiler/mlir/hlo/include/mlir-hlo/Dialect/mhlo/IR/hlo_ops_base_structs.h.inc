/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mhlo {

// two 64-bit integers 'handle' and 'type'
class ChannelHandle : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ChannelHandle get(
      ::mlir::IntegerAttr handle,
      ::mlir::IntegerAttr type,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr handle() const;
  ::mlir::IntegerAttr type() const;
};

} // namespace mlir
} // namespace mhlo
namespace mlir {
namespace mhlo {

// Structure of dimension information for conv op
class ConvDimensionNumbers : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ConvDimensionNumbers get(
      ::mlir::IntegerAttr input_batch_dimension,
      ::mlir::IntegerAttr input_feature_dimension,
      ::mlir::DenseIntElementsAttr input_spatial_dimensions,
      ::mlir::IntegerAttr kernel_input_feature_dimension,
      ::mlir::IntegerAttr kernel_output_feature_dimension,
      ::mlir::DenseIntElementsAttr kernel_spatial_dimensions,
      ::mlir::IntegerAttr output_batch_dimension,
      ::mlir::IntegerAttr output_feature_dimension,
      ::mlir::DenseIntElementsAttr output_spatial_dimensions,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr input_batch_dimension() const;
  ::mlir::IntegerAttr input_feature_dimension() const;
  ::mlir::DenseIntElementsAttr input_spatial_dimensions() const;
  ::mlir::IntegerAttr kernel_input_feature_dimension() const;
  ::mlir::IntegerAttr kernel_output_feature_dimension() const;
  ::mlir::DenseIntElementsAttr kernel_spatial_dimensions() const;
  ::mlir::IntegerAttr output_batch_dimension() const;
  ::mlir::IntegerAttr output_feature_dimension() const;
  ::mlir::DenseIntElementsAttr output_spatial_dimensions() const;
};

} // namespace mlir
} // namespace mhlo
namespace mlir {
namespace mhlo {

// Structure of dimension information for dot product
class DotDimensionNumbers : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static DotDimensionNumbers get(
      ::mlir::DenseIntElementsAttr lhs_batching_dimensions,
      ::mlir::DenseIntElementsAttr rhs_batching_dimensions,
      ::mlir::DenseIntElementsAttr lhs_contracting_dimensions,
      ::mlir::DenseIntElementsAttr rhs_contracting_dimensions,
      ::mlir::MLIRContext* context);

  ::mlir::DenseIntElementsAttr lhs_batching_dimensions() const;
  ::mlir::DenseIntElementsAttr rhs_batching_dimensions() const;
  ::mlir::DenseIntElementsAttr lhs_contracting_dimensions() const;
  ::mlir::DenseIntElementsAttr rhs_contracting_dimensions() const;
};

} // namespace mlir
} // namespace mhlo
namespace mlir {
namespace mhlo {

// Structure of dimension information for gather
class GatherDimensionNumbers : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static GatherDimensionNumbers get(
      ::mlir::DenseIntElementsAttr offset_dims,
      ::mlir::DenseIntElementsAttr collapsed_slice_dims,
      ::mlir::DenseIntElementsAttr start_index_map,
      ::mlir::IntegerAttr index_vector_dim,
      ::mlir::MLIRContext* context);

  ::mlir::DenseIntElementsAttr offset_dims() const;
  ::mlir::DenseIntElementsAttr collapsed_slice_dims() const;
  ::mlir::DenseIntElementsAttr start_index_map() const;
  ::mlir::IntegerAttr index_vector_dim() const;
};

} // namespace mlir
} // namespace mhlo
namespace mlir {
namespace mhlo {

// Structure of dimension information for scatter
class ScatterDimensionNumbers : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ScatterDimensionNumbers get(
      ::mlir::DenseIntElementsAttr update_window_dims,
      ::mlir::DenseIntElementsAttr inserted_window_dims,
      ::mlir::DenseIntElementsAttr scatter_dims_to_operand_dims,
      ::mlir::IntegerAttr index_vector_dim,
      ::mlir::MLIRContext* context);

  ::mlir::DenseIntElementsAttr update_window_dims() const;
  ::mlir::DenseIntElementsAttr inserted_window_dims() const;
  ::mlir::DenseIntElementsAttr scatter_dims_to_operand_dims() const;
  ::mlir::IntegerAttr index_vector_dim() const;
};

} // namespace mlir
} // namespace mhlo
