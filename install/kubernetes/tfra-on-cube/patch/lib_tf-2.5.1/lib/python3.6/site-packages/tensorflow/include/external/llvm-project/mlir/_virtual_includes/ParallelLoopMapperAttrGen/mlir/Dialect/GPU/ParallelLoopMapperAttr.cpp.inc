/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
ParallelLoopDimMapping ParallelLoopDimMapping::get(
    ::mlir::gpu::ProcessorAttr processor,
    ::mlir::AffineMapAttr map,
    ::mlir::AffineMapAttr bound,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 3> fields;

  assert(processor);
  auto processor_id = ::mlir::Identifier::get("processor", context);
  fields.emplace_back(processor_id, processor);

  assert(map);
  auto map_id = ::mlir::Identifier::get("map", context);
  fields.emplace_back(map_id, map);

  assert(bound);
  auto bound_id = ::mlir::Identifier::get("bound", context);
  fields.emplace_back(bound_id, bound);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<ParallelLoopDimMapping>();
}

bool ParallelLoopDimMapping::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto processor = derived.get("processor");
  if (!processor || !((processor.isa<::mlir::gpu::ProcessorAttr>())))
    return false;

  auto map = derived.get("map");
  if (!map || !((map.isa<::mlir::AffineMapAttr>())))
    return false;

  auto bound = derived.get("bound");
  if (!bound || !((bound.isa<::mlir::AffineMapAttr>())))
    return false;

  return derived.size() + num_absent_attrs == 3;
}

::mlir::gpu::ProcessorAttr ParallelLoopDimMapping::processor() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto processor = derived.get("processor");
  assert(processor && "attribute not found.");
  assert(processor.isa<::mlir::gpu::ProcessorAttr>() && "incorrect Attribute type found.");
  return processor.cast<::mlir::gpu::ProcessorAttr>();
}

::mlir::AffineMapAttr ParallelLoopDimMapping::map() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto map = derived.get("map");
  assert(map && "attribute not found.");
  assert(map.isa<::mlir::AffineMapAttr>() && "incorrect Attribute type found.");
  return map.cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMapAttr ParallelLoopDimMapping::bound() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto bound = derived.get("bound");
  assert(bound && "attribute not found.");
  assert(bound.isa<::mlir::AffineMapAttr>() && "incorrect Attribute type found.");
  return bound.cast<::mlir::AffineMapAttr>();
}
} // namespace gpu
} // namespace mlir
