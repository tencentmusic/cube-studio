/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace spirv {
CooperativeMatrixPropertiesNVAttr CooperativeMatrixPropertiesNVAttr::get(
    ::mlir::IntegerAttr m_size,
    ::mlir::IntegerAttr n_size,
    ::mlir::IntegerAttr k_size,
    ::mlir::TypeAttr a_type,
    ::mlir::TypeAttr b_type,
    ::mlir::TypeAttr c_type,
    ::mlir::TypeAttr result_type,
    ::mlir::spirv::ScopeAttr scope,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 8> fields;

  assert(m_size);
  auto m_size_id = ::mlir::Identifier::get("m_size", context);
  fields.emplace_back(m_size_id, m_size);

  assert(n_size);
  auto n_size_id = ::mlir::Identifier::get("n_size", context);
  fields.emplace_back(n_size_id, n_size);

  assert(k_size);
  auto k_size_id = ::mlir::Identifier::get("k_size", context);
  fields.emplace_back(k_size_id, k_size);

  assert(a_type);
  auto a_type_id = ::mlir::Identifier::get("a_type", context);
  fields.emplace_back(a_type_id, a_type);

  assert(b_type);
  auto b_type_id = ::mlir::Identifier::get("b_type", context);
  fields.emplace_back(b_type_id, b_type);

  assert(c_type);
  auto c_type_id = ::mlir::Identifier::get("c_type", context);
  fields.emplace_back(c_type_id, c_type);

  assert(result_type);
  auto result_type_id = ::mlir::Identifier::get("result_type", context);
  fields.emplace_back(result_type_id, result_type);

  assert(scope);
  auto scope_id = ::mlir::Identifier::get("scope", context);
  fields.emplace_back(scope_id, scope);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<CooperativeMatrixPropertiesNVAttr>();
}

bool CooperativeMatrixPropertiesNVAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto m_size = derived.get("m_size");
  if (!m_size || !(((m_size.isa<::mlir::IntegerAttr>())) && ((m_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto n_size = derived.get("n_size");
  if (!n_size || !(((n_size.isa<::mlir::IntegerAttr>())) && ((n_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto k_size = derived.get("k_size");
  if (!k_size || !(((k_size.isa<::mlir::IntegerAttr>())) && ((k_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto a_type = derived.get("a_type");
  if (!a_type || !(((a_type.isa<::mlir::TypeAttr>())) && ((a_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return false;

  auto b_type = derived.get("b_type");
  if (!b_type || !(((b_type.isa<::mlir::TypeAttr>())) && ((b_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return false;

  auto c_type = derived.get("c_type");
  if (!c_type || !(((c_type.isa<::mlir::TypeAttr>())) && ((c_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return false;

  auto result_type = derived.get("result_type");
  if (!result_type || !(((result_type.isa<::mlir::TypeAttr>())) && ((result_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return false;

  auto scope = derived.get("scope");
  if (!scope || !((((scope.isa<::mlir::IntegerAttr>())) && ((scope.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((::mlir::spirv::symbolizeScope(scope.cast<IntegerAttr>().getValue().getZExtValue()).hasValue()))))
    return false;

  return derived.size() + num_absent_attrs == 8;
}

::mlir::IntegerAttr CooperativeMatrixPropertiesNVAttr::m_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto m_size = derived.get("m_size");
  assert(m_size && "attribute not found.");
  assert(m_size.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return m_size.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr CooperativeMatrixPropertiesNVAttr::n_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto n_size = derived.get("n_size");
  assert(n_size && "attribute not found.");
  assert(n_size.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return n_size.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr CooperativeMatrixPropertiesNVAttr::k_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto k_size = derived.get("k_size");
  assert(k_size && "attribute not found.");
  assert(k_size.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return k_size.cast<::mlir::IntegerAttr>();
}

::mlir::TypeAttr CooperativeMatrixPropertiesNVAttr::a_type() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto a_type = derived.get("a_type");
  assert(a_type && "attribute not found.");
  assert(a_type.isa<::mlir::TypeAttr>() && "incorrect Attribute type found.");
  return a_type.cast<::mlir::TypeAttr>();
}

::mlir::TypeAttr CooperativeMatrixPropertiesNVAttr::b_type() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto b_type = derived.get("b_type");
  assert(b_type && "attribute not found.");
  assert(b_type.isa<::mlir::TypeAttr>() && "incorrect Attribute type found.");
  return b_type.cast<::mlir::TypeAttr>();
}

::mlir::TypeAttr CooperativeMatrixPropertiesNVAttr::c_type() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto c_type = derived.get("c_type");
  assert(c_type && "attribute not found.");
  assert(c_type.isa<::mlir::TypeAttr>() && "incorrect Attribute type found.");
  return c_type.cast<::mlir::TypeAttr>();
}

::mlir::TypeAttr CooperativeMatrixPropertiesNVAttr::result_type() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto result_type = derived.get("result_type");
  assert(result_type && "attribute not found.");
  assert(result_type.isa<::mlir::TypeAttr>() && "incorrect Attribute type found.");
  return result_type.cast<::mlir::TypeAttr>();
}

::mlir::spirv::ScopeAttr CooperativeMatrixPropertiesNVAttr::scope() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto scope = derived.get("scope");
  assert(scope && "attribute not found.");
  assert(scope.isa<::mlir::spirv::ScopeAttr>() && "incorrect Attribute type found.");
  return scope.cast<::mlir::spirv::ScopeAttr>();
}
} // namespace spirv
} // namespace mlir
namespace mlir {
namespace spirv {
EntryPointABIAttr EntryPointABIAttr::get(
    ::mlir::DenseIntElementsAttr local_size,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> fields;

  assert(local_size);
  auto local_size_id = ::mlir::Identifier::get("local_size", context);
  fields.emplace_back(local_size_id, local_size);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<EntryPointABIAttr>();
}

bool EntryPointABIAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto local_size = derived.get("local_size");
  if (!local_size || !(((local_size.isa<::mlir::DenseIntElementsAttr>())) && ((local_size.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 1;
}

::mlir::DenseIntElementsAttr EntryPointABIAttr::local_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto local_size = derived.get("local_size");
  assert(local_size && "attribute not found.");
  assert(local_size.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return local_size.cast<::mlir::DenseIntElementsAttr>();
}
} // namespace spirv
} // namespace mlir
namespace mlir {
namespace spirv {
ResourceLimitsAttr ResourceLimitsAttr::get(
    ::mlir::IntegerAttr max_compute_shared_memory_size,
    ::mlir::IntegerAttr max_compute_workgroup_invocations,
    ::mlir::DenseIntElementsAttr max_compute_workgroup_size,
    ::mlir::IntegerAttr subgroup_size,
    ::mlir::ArrayAttr cooperative_matrix_properties_nv,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 5> fields;

  if (max_compute_shared_memory_size) {
    auto max_compute_shared_memory_size_id = ::mlir::Identifier::get("max_compute_shared_memory_size", context);
    fields.emplace_back(max_compute_shared_memory_size_id, max_compute_shared_memory_size);
  }

  if (max_compute_workgroup_invocations) {
    auto max_compute_workgroup_invocations_id = ::mlir::Identifier::get("max_compute_workgroup_invocations", context);
    fields.emplace_back(max_compute_workgroup_invocations_id, max_compute_workgroup_invocations);
  }

  if (max_compute_workgroup_size) {
    auto max_compute_workgroup_size_id = ::mlir::Identifier::get("max_compute_workgroup_size", context);
    fields.emplace_back(max_compute_workgroup_size_id, max_compute_workgroup_size);
  }

  if (subgroup_size) {
    auto subgroup_size_id = ::mlir::Identifier::get("subgroup_size", context);
    fields.emplace_back(subgroup_size_id, subgroup_size);
  }

  if (cooperative_matrix_properties_nv) {
    auto cooperative_matrix_properties_nv_id = ::mlir::Identifier::get("cooperative_matrix_properties_nv", context);
    fields.emplace_back(cooperative_matrix_properties_nv_id, cooperative_matrix_properties_nv);
  }

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<ResourceLimitsAttr>();
}

bool ResourceLimitsAttr::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto max_compute_shared_memory_size = derived.get("max_compute_shared_memory_size");
  if (!max_compute_shared_memory_size)
    ++num_absent_attrs;
  else if (!(((max_compute_shared_memory_size.isa<::mlir::IntegerAttr>())) && ((max_compute_shared_memory_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto max_compute_workgroup_invocations = derived.get("max_compute_workgroup_invocations");
  if (!max_compute_workgroup_invocations)
    ++num_absent_attrs;
  else if (!(((max_compute_workgroup_invocations.isa<::mlir::IntegerAttr>())) && ((max_compute_workgroup_invocations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto max_compute_workgroup_size = derived.get("max_compute_workgroup_size");
  if (!max_compute_workgroup_size)
    ++num_absent_attrs;
  else if (!(((max_compute_workgroup_size.isa<::mlir::DenseIntElementsAttr>())) && ((max_compute_workgroup_size.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return false;

  auto subgroup_size = derived.get("subgroup_size");
  if (!subgroup_size)
    ++num_absent_attrs;
  else if (!(((subgroup_size.isa<::mlir::IntegerAttr>())) && ((subgroup_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto cooperative_matrix_properties_nv = derived.get("cooperative_matrix_properties_nv");
  if (!cooperative_matrix_properties_nv)
    ++num_absent_attrs;
  else if (!(((cooperative_matrix_properties_nv.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(cooperative_matrix_properties_nv.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::spirv::CooperativeMatrixPropertiesNVAttr>()); }))))
    return false;

  return derived.size() + num_absent_attrs == 5;
}

::mlir::IntegerAttr ResourceLimitsAttr::max_compute_shared_memory_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto max_compute_shared_memory_size = derived.get("max_compute_shared_memory_size");
  if (!max_compute_shared_memory_size) {
    ::mlir::Builder builder(getContext());
    return builder.getIntegerAttr(builder.getIntegerType(32), 16384);
  }
  assert(max_compute_shared_memory_size.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return max_compute_shared_memory_size.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr ResourceLimitsAttr::max_compute_workgroup_invocations() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto max_compute_workgroup_invocations = derived.get("max_compute_workgroup_invocations");
  if (!max_compute_workgroup_invocations) {
    ::mlir::Builder builder(getContext());
    return builder.getIntegerAttr(builder.getIntegerType(32), 128);
  }
  assert(max_compute_workgroup_invocations.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return max_compute_workgroup_invocations.cast<::mlir::IntegerAttr>();
}

::mlir::DenseIntElementsAttr ResourceLimitsAttr::max_compute_workgroup_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto max_compute_workgroup_size = derived.get("max_compute_workgroup_size");
  if (!max_compute_workgroup_size) {
    ::mlir::Builder builder(getContext());
    return ::mlir::DenseElementsAttr::get(::mlir::RankedTensorType::get({}, builder.getIntegerType(32)), ::llvm::makeArrayRef({128, 128, 64})).cast<::mlir::DenseIntElementsAttr>();
  }
  assert(max_compute_workgroup_size.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return max_compute_workgroup_size.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::IntegerAttr ResourceLimitsAttr::subgroup_size() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto subgroup_size = derived.get("subgroup_size");
  if (!subgroup_size) {
    ::mlir::Builder builder(getContext());
    return builder.getIntegerAttr(builder.getIntegerType(32), 0x7FFFFFFF);
  }
  assert(subgroup_size.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return subgroup_size.cast<::mlir::IntegerAttr>();
}

::mlir::ArrayAttr ResourceLimitsAttr::cooperative_matrix_properties_nv() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto cooperative_matrix_properties_nv = derived.get("cooperative_matrix_properties_nv");
  if (!cooperative_matrix_properties_nv) {
    ::mlir::Builder builder(getContext());
    return builder.getArrayAttr({});
  }
  assert(cooperative_matrix_properties_nv.isa<::mlir::ArrayAttr>() && "incorrect Attribute type found.");
  return cooperative_matrix_properties_nv.cast<::mlir::ArrayAttr>();
}
} // namespace spirv
} // namespace mlir
