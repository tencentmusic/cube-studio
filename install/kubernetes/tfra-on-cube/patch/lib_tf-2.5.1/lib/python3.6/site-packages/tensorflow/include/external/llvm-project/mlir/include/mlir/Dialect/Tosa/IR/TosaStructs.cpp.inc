/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {
ConvOpQuantizationAttr ConvOpQuantizationAttr::get(
    ::mlir::IntegerAttr input_zp,
    ::mlir::IntegerAttr weight_zp,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields;

  assert(input_zp);
  auto input_zp_id = ::mlir::Identifier::get("input_zp", context);
  fields.emplace_back(input_zp_id, input_zp);

  assert(weight_zp);
  auto weight_zp_id = ::mlir::Identifier::get("weight_zp", context);
  fields.emplace_back(weight_zp_id, weight_zp);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<ConvOpQuantizationAttr>();
}

bool ConvOpQuantizationAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto input_zp = derived.get("input_zp");
  if (!input_zp || !(((input_zp.isa<::mlir::IntegerAttr>())) && ((input_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto weight_zp = derived.get("weight_zp");
  if (!weight_zp || !(((weight_zp.isa<::mlir::IntegerAttr>())) && ((weight_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 2;
}

::mlir::IntegerAttr ConvOpQuantizationAttr::input_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto input_zp = derived.get("input_zp");
  assert(input_zp && "attribute not found.");
  assert(input_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return input_zp.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr ConvOpQuantizationAttr::weight_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto weight_zp = derived.get("weight_zp");
  assert(weight_zp && "attribute not found.");
  assert(weight_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return weight_zp.cast<::mlir::IntegerAttr>();
}
} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {
MatMulOpQuantizationAttr MatMulOpQuantizationAttr::get(
    ::mlir::IntegerAttr a_zp,
    ::mlir::IntegerAttr b_zp,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields;

  assert(a_zp);
  auto a_zp_id = ::mlir::Identifier::get("a_zp", context);
  fields.emplace_back(a_zp_id, a_zp);

  assert(b_zp);
  auto b_zp_id = ::mlir::Identifier::get("b_zp", context);
  fields.emplace_back(b_zp_id, b_zp);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<MatMulOpQuantizationAttr>();
}

bool MatMulOpQuantizationAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto a_zp = derived.get("a_zp");
  if (!a_zp || !(((a_zp.isa<::mlir::IntegerAttr>())) && ((a_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto b_zp = derived.get("b_zp");
  if (!b_zp || !(((b_zp.isa<::mlir::IntegerAttr>())) && ((b_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 2;
}

::mlir::IntegerAttr MatMulOpQuantizationAttr::a_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto a_zp = derived.get("a_zp");
  assert(a_zp && "attribute not found.");
  assert(a_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return a_zp.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr MatMulOpQuantizationAttr::b_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto b_zp = derived.get("b_zp");
  assert(b_zp && "attribute not found.");
  assert(b_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return b_zp.cast<::mlir::IntegerAttr>();
}
} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {
PadOpQuantizationAttr PadOpQuantizationAttr::get(
    ::mlir::IntegerAttr input_zp,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> fields;

  assert(input_zp);
  auto input_zp_id = ::mlir::Identifier::get("input_zp", context);
  fields.emplace_back(input_zp_id, input_zp);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<PadOpQuantizationAttr>();
}

bool PadOpQuantizationAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto input_zp = derived.get("input_zp");
  if (!input_zp || !(((input_zp.isa<::mlir::IntegerAttr>())) && ((input_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 1;
}

::mlir::IntegerAttr PadOpQuantizationAttr::input_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto input_zp = derived.get("input_zp");
  assert(input_zp && "attribute not found.");
  assert(input_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return input_zp.cast<::mlir::IntegerAttr>();
}
} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {
UnaryOpQuantizationAttr UnaryOpQuantizationAttr::get(
    ::mlir::IntegerAttr input_zp,
    ::mlir::IntegerAttr output_zp,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields;

  assert(input_zp);
  auto input_zp_id = ::mlir::Identifier::get("input_zp", context);
  fields.emplace_back(input_zp_id, input_zp);

  assert(output_zp);
  auto output_zp_id = ::mlir::Identifier::get("output_zp", context);
  fields.emplace_back(output_zp_id, output_zp);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<UnaryOpQuantizationAttr>();
}

bool UnaryOpQuantizationAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto input_zp = derived.get("input_zp");
  if (!input_zp || !(((input_zp.isa<::mlir::IntegerAttr>())) && ((input_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto output_zp = derived.get("output_zp");
  if (!output_zp || !(((output_zp.isa<::mlir::IntegerAttr>())) && ((output_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 2;
}

::mlir::IntegerAttr UnaryOpQuantizationAttr::input_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto input_zp = derived.get("input_zp");
  assert(input_zp && "attribute not found.");
  assert(input_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return input_zp.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr UnaryOpQuantizationAttr::output_zp() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto output_zp = derived.get("output_zp");
  assert(output_zp && "attribute not found.");
  assert(output_zp.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return output_zp.cast<::mlir::IntegerAttr>();
}
} // namespace tosa
} // namespace mlir
