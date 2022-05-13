/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Attribute Utilities                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_DIALECT_SPIRV_IR_ATTR_UTILS_H_
#define MLIR_DIALECT_SPIRV_IR_ATTR_UTILS_H_
template <typename EnumClass> inline constexpr StringRef attributeName();
template <> inline StringRef attributeName<AddressingModel>() {
  static constexpr const char attrName[] = "addressing_model";
  return attrName;
}
template <> inline StringRef attributeName<ImageArrayedInfo>() {
  static constexpr const char attrName[] = "image_arrayed_info";
  return attrName;
}
template <> inline StringRef attributeName<BuiltIn>() {
  static constexpr const char attrName[] = "built_in";
  return attrName;
}
template <> inline StringRef attributeName<Capability>() {
  static constexpr const char attrName[] = "capability";
  return attrName;
}
template <> inline StringRef attributeName<Decoration>() {
  static constexpr const char attrName[] = "decoration";
  return attrName;
}
template <> inline StringRef attributeName<ImageDepthInfo>() {
  static constexpr const char attrName[] = "image_depth_info";
  return attrName;
}
template <> inline StringRef attributeName<DeviceType>() {
  static constexpr const char attrName[] = "device_type";
  return attrName;
}
template <> inline StringRef attributeName<Dim>() {
  static constexpr const char attrName[] = "dim";
  return attrName;
}
template <> inline StringRef attributeName<ExecutionMode>() {
  static constexpr const char attrName[] = "execution_mode";
  return attrName;
}
template <> inline StringRef attributeName<ExecutionModel>() {
  static constexpr const char attrName[] = "execution_model";
  return attrName;
}
template <> inline StringRef attributeName<Extension>() {
  static constexpr const char attrName[] = "extension";
  return attrName;
}
template <> inline StringRef attributeName<FunctionControl>() {
  static constexpr const char attrName[] = "function_control";
  return attrName;
}
template <> inline StringRef attributeName<GroupOperation>() {
  static constexpr const char attrName[] = "group_operation";
  return attrName;
}
template <> inline StringRef attributeName<ImageFormat>() {
  static constexpr const char attrName[] = "image_format";
  return attrName;
}
template <> inline StringRef attributeName<LinkageType>() {
  static constexpr const char attrName[] = "linkage_type";
  return attrName;
}
template <> inline StringRef attributeName<LoopControl>() {
  static constexpr const char attrName[] = "loop_control";
  return attrName;
}
template <> inline StringRef attributeName<MemoryAccess>() {
  static constexpr const char attrName[] = "memory_access";
  return attrName;
}
template <> inline StringRef attributeName<MemoryModel>() {
  static constexpr const char attrName[] = "memory_model";
  return attrName;
}
template <> inline StringRef attributeName<MemorySemantics>() {
  static constexpr const char attrName[] = "memory_semantics";
  return attrName;
}
template <> inline StringRef attributeName<Opcode>() {
  static constexpr const char attrName[] = "opcode";
  return attrName;
}
template <> inline StringRef attributeName<ImageSamplerUseInfo>() {
  static constexpr const char attrName[] = "image_sampler_use_info";
  return attrName;
}
template <> inline StringRef attributeName<ImageSamplingInfo>() {
  static constexpr const char attrName[] = "image_sampling_info";
  return attrName;
}
template <> inline StringRef attributeName<Scope>() {
  static constexpr const char attrName[] = "scope";
  return attrName;
}
template <> inline StringRef attributeName<SelectionControl>() {
  static constexpr const char attrName[] = "selection_control";
  return attrName;
}
template <> inline StringRef attributeName<StorageClass>() {
  static constexpr const char attrName[] = "storage_class";
  return attrName;
}
template <> inline StringRef attributeName<Vendor>() {
  static constexpr const char attrName[] = "vendor";
  return attrName;
}
template <> inline StringRef attributeName<Version>() {
  static constexpr const char attrName[] = "version";
  return attrName;
}
#endif // MLIR_DIALECT_SPIRV_IR_ATTR_UTILS_H
