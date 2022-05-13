/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {

// Attribute for Conv type op quantization information.
class ConvOpQuantizationAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ConvOpQuantizationAttr get(
      ::mlir::IntegerAttr input_zp,
      ::mlir::IntegerAttr weight_zp,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr input_zp() const;
  ::mlir::IntegerAttr weight_zp() const;
};

} // namespace mlir
} // namespace tosa
namespace mlir {
namespace tosa {

// Attribute for MatMulOp quantization information.
class MatMulOpQuantizationAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static MatMulOpQuantizationAttr get(
      ::mlir::IntegerAttr a_zp,
      ::mlir::IntegerAttr b_zp,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr a_zp() const;
  ::mlir::IntegerAttr b_zp() const;
};

} // namespace mlir
} // namespace tosa
namespace mlir {
namespace tosa {

// Attribute for PadOp quantization information.
class PadOpQuantizationAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static PadOpQuantizationAttr get(
      ::mlir::IntegerAttr input_zp,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr input_zp() const;
};

} // namespace mlir
} // namespace tosa
namespace mlir {
namespace tosa {

// Attribute for UnaryOp quantization information.
class UnaryOpQuantizationAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static UnaryOpQuantizationAttr get(
      ::mlir::IntegerAttr input_zp,
      ::mlir::IntegerAttr output_zp,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr input_zp() const;
  ::mlir::IntegerAttr output_zp() const;
};

} // namespace mlir
} // namespace tosa
