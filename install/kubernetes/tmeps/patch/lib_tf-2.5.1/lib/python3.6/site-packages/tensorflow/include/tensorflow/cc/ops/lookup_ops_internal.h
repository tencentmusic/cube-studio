// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup lookup_ops_internal Lookup Ops Internal
/// @{

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class InitializeTableFromDataset {
 public:
  InitializeTableFromDataset(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input table_handle,
                           ::tensorflow::Input dataset);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Removes keys and its associated values from a table.
///
/// The tensor `keys` must of the same type as the keys of the table. Keys not
/// already in the table are silently ignored.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
/// * keys: Any shape.  Keys of the elements to remove.
///
/// Returns:
/// * the created `Operation`
class LookupTableRemove {
 public:
  LookupTableRemove(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  table_handle, ::tensorflow::Input keys);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_
