/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace spirv {

// DictionaryAttr with field(s): 'm_size', 'n_size', 'k_size', 'a_type', 'b_type', 'c_type', 'result_type', 'scope' (each field having its own constraints)
class CooperativeMatrixPropertiesNVAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static CooperativeMatrixPropertiesNVAttr get(
      ::mlir::IntegerAttr m_size,
      ::mlir::IntegerAttr n_size,
      ::mlir::IntegerAttr k_size,
      ::mlir::TypeAttr a_type,
      ::mlir::TypeAttr b_type,
      ::mlir::TypeAttr c_type,
      ::mlir::TypeAttr result_type,
      ::mlir::spirv::ScopeAttr scope,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr m_size() const;
  ::mlir::IntegerAttr n_size() const;
  ::mlir::IntegerAttr k_size() const;
  ::mlir::TypeAttr a_type() const;
  ::mlir::TypeAttr b_type() const;
  ::mlir::TypeAttr c_type() const;
  ::mlir::TypeAttr result_type() const;
  ::mlir::spirv::ScopeAttr scope() const;
};

} // namespace mlir
} // namespace spirv
namespace mlir {
namespace spirv {

// DictionaryAttr with field(s): 'local_size' (each field having its own constraints)
class EntryPointABIAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static EntryPointABIAttr get(
      ::mlir::DenseIntElementsAttr local_size,
      ::mlir::MLIRContext* context);

  ::mlir::DenseIntElementsAttr local_size() const;
};

} // namespace mlir
} // namespace spirv
namespace mlir {
namespace spirv {

// DictionaryAttr with field(s): 'max_compute_shared_memory_size', 'max_compute_workgroup_invocations', 'max_compute_workgroup_size', 'subgroup_size', 'cooperative_matrix_properties_nv' (each field having its own constraints)
class ResourceLimitsAttr : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ResourceLimitsAttr get(
      ::mlir::IntegerAttr max_compute_shared_memory_size,
      ::mlir::IntegerAttr max_compute_workgroup_invocations,
      ::mlir::DenseIntElementsAttr max_compute_workgroup_size,
      ::mlir::IntegerAttr subgroup_size,
      ::mlir::ArrayAttr cooperative_matrix_properties_nv,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr max_compute_shared_memory_size() const;
  ::mlir::IntegerAttr max_compute_workgroup_invocations() const;
  ::mlir::DenseIntElementsAttr max_compute_workgroup_size() const;
  ::mlir::IntegerAttr subgroup_size() const;
  ::mlir::ArrayAttr cooperative_matrix_properties_nv() const;
};

} // namespace mlir
} // namespace spirv
