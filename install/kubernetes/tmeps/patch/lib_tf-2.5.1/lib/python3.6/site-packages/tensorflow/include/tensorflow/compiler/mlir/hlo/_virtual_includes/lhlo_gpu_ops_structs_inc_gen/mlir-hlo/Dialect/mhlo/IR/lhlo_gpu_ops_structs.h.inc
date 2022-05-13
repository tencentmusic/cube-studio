/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {

// GPU Convolution backend configuration
class ConvolutionBackendConfig : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ConvolutionBackendConfig get(
      ::mlir::IntegerAttr algorithm,
      ::mlir::BoolAttr tensor_ops_enabled,
      ::mlir::ArrayAttr operand_0_layout,
      ::mlir::ArrayAttr operand_1_layout,
      ::mlir::ArrayAttr result_layout,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr algorithm() const;
  ::mlir::BoolAttr tensor_ops_enabled() const;
  ::mlir::ArrayAttr operand_0_layout() const;
  ::mlir::ArrayAttr operand_1_layout() const;
  ::mlir::ArrayAttr result_layout() const;
};

} // namespace mlir
} // namespace lmhlo_gpu
