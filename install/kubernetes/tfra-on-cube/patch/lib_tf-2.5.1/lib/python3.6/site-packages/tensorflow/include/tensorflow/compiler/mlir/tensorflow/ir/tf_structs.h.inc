/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace TF {

// DictionaryAttr with field(s): 'cc_major', 'cc_minor' (each field having its own constraints)
class GpuDeviceMetadata : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static GpuDeviceMetadata get(
      ::mlir::IntegerAttr cc_major,
      ::mlir::IntegerAttr cc_minor,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr cc_major() const;
  ::mlir::IntegerAttr cc_minor() const;
};

} // namespace mlir
} // namespace TF
