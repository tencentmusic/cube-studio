/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo {

// Custom call operands to target argument mapping info
class CustomCallTargetArgMapping : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static CustomCallTargetArgMapping get(
      ::mlir::IntegerAttr num_args,
      ::mlir::IntegerAttr num_results,
      ::mlir::ArrayAttr args_to_target_args,
      ::mlir::ArrayAttr results_to_target_results,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr num_args() const;
  ::mlir::IntegerAttr num_results() const;
  ::mlir::ArrayAttr args_to_target_args() const;
  ::mlir::ArrayAttr results_to_target_results() const;
};

} // namespace mlir
} // namespace lmhlo
