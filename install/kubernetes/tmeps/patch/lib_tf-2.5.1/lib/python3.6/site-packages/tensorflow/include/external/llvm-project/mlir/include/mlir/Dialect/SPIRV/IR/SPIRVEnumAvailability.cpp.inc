/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Enum Availability Definitions                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(AddressingModel value) {
  switch (value) {
  case AddressingModel::Physical32: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Addresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case AddressingModel::Physical64: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Addresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case AddressingModel::PhysicalStorageBuffer64: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::PhysicalStorageBufferAddresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(AddressingModel value) {
  switch (value) {
  case AddressingModel::PhysicalStorageBuffer64: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_physical_storage_buffer, ::mlir::spirv::Extension::SPV_KHR_physical_storage_buffer}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(BuiltIn value) {
  switch (value) {
  case BuiltIn::Position: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PointSize: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ClipDistance: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ClipDistance}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::CullDistance: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CullDistance}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::VertexId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::InstanceId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PrimitiveId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry, ::mlir::spirv::Capability::RayTracingNV, ::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::InvocationId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry, ::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::Layer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry, ::mlir::spirv::Capability::ShaderLayer, ::mlir::spirv::Capability::ShaderViewportIndexLayerEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ViewportIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MultiViewport, ::mlir::spirv::Capability::ShaderViewportIndex, ::mlir::spirv::Capability::ShaderViewportIndexLayerEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::TessLevelOuter: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::TessLevelInner: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::TessCoord: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PatchVertices: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FragCoord: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PointCoord: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FrontFacing: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SampleId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SampleRateShading}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SamplePosition: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SampleRateShading}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SampleMask: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FragDepth: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::HelperInvocation: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::WorkDim: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::GlobalSize: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::EnqueuedWorkgroupSize: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::GlobalOffset: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::GlobalLinearId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupSize: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniform, ::mlir::spirv::Capability::Kernel, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupMaxSize: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::NumSubgroups: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniform, ::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::NumEnqueuedSubgroups: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniform, ::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupLocalInvocationId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniform, ::mlir::spirv::Capability::Kernel, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::VertexIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::InstanceIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupEqMask: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupGeMask: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupGtMask: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupLeMask: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SubgroupLtMask: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::BaseVertex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::DrawParameters}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::BaseInstance: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::DrawParameters}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::DrawIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::DrawParameters, ::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::DeviceIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::DeviceGroup}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ViewIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MultiView}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FragStencilRefEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StencilExportEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ViewportMaskNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV, ::mlir::spirv::Capability::ShaderViewportMaskNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SecondaryPositionNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderStereoViewNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SecondaryViewportMaskNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderStereoViewNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PositionPerViewNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV, ::mlir::spirv::Capability::PerViewAttributesNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ViewportMaskPerViewNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV, ::mlir::spirv::Capability::PerViewAttributesNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FullyCoveredEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentFullyCoveredEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::TaskCountNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PrimitiveCountNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::PrimitiveIndicesNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ClipDistancePerViewNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::CullDistancePerViewNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::LayerPerViewNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::MeshViewCountNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::MeshViewIndicesNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::BaryCoordNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentBarycentricNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::BaryCoordNoPerspNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentBarycentricNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FragSizeEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentDensityEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::FragInvocationCountEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentDensityEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::LaunchIdNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::LaunchSizeNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::WorldRayOriginNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::WorldRayDirectionNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ObjectRayOriginNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ObjectRayDirectionNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::RayTminNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::RayTmaxNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::InstanceCustomIndexNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::ObjectToWorldNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::WorldToObjectNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::HitTNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::HitKindNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::IncomingRayFlagsNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::WarpsPerSMNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderSMBuiltinsNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SMCountNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderSMBuiltinsNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::WarpIDNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderSMBuiltinsNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case BuiltIn::SMIDNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderSMBuiltinsNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(BuiltIn value) {
  switch (value) {
  case BuiltIn::SubgroupEqMask: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case BuiltIn::SubgroupGeMask: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case BuiltIn::SubgroupGtMask: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case BuiltIn::SubgroupLeMask: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case BuiltIn::SubgroupLtMask: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(BuiltIn value) {
  switch (value) {
  case BuiltIn::BaseVertex: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_draw_parameters}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaseInstance: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_draw_parameters}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::DrawIndex: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_draw_parameters, ::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::DeviceIndex: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_device_group}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ViewIndex: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_multiview}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordNoPerspAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordNoPerspCentroidAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordNoPerspSampleAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordSmoothAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordSmoothCentroidAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordSmoothSampleAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordPullModelAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::FragStencilRefEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_shader_stencil_export}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ViewportMaskNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader, ::mlir::spirv::Extension::SPV_NV_viewport_array2}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::SecondaryPositionNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_stereo_view_rendering}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::SecondaryViewportMaskNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_stereo_view_rendering}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::PositionPerViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NVX_multiview_per_view_attributes, ::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ViewportMaskPerViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NVX_multiview_per_view_attributes, ::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::FullyCoveredEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_fully_covered}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::TaskCountNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::PrimitiveCountNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::PrimitiveIndicesNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ClipDistancePerViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::CullDistancePerViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::LayerPerViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::MeshViewCountNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::MeshViewIndicesNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_fragment_shader_barycentric}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::BaryCoordNoPerspNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_fragment_shader_barycentric}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::FragSizeEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_invocation_density, ::mlir::spirv::Extension::SPV_NV_shading_rate}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::FragInvocationCountEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_invocation_density, ::mlir::spirv::Extension::SPV_NV_shading_rate}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::LaunchIdNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::LaunchSizeNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::WorldRayOriginNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::WorldRayDirectionNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ObjectRayOriginNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ObjectRayDirectionNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::RayTminNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::RayTmaxNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::InstanceCustomIndexNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::ObjectToWorldNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::WorldToObjectNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::HitTNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::HitKindNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::IncomingRayFlagsNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::WarpsPerSMNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_sm_builtins}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::SMCountNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_sm_builtins}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::WarpIDNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_sm_builtins}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case BuiltIn::SMIDNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_sm_builtins}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::mlir::spirv::Version> getMinVersion(Capability value) {
  switch (value) {
  case Capability::GroupNonUniform: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::ShaderLayer: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::ShaderViewportIndex: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::VulkanMemoryModel: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::VulkanMemoryModelDeviceScope: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::NamedBarrier: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case Capability::GroupNonUniformVote: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::GroupNonUniformArithmetic: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::GroupNonUniformBallot: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::GroupNonUniformShuffle: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::GroupNonUniformShuffleRelative: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::GroupNonUniformClustered: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::GroupNonUniformQuad: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  case Capability::UniformTexelBufferArrayDynamicIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::SubgroupDispatch: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case Capability::PipeStorage: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case Capability::ShaderNonUniform: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::RuntimeDescriptorArray: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::StorageTexelBufferArrayDynamicIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::InputAttachmentArrayDynamicIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::UniformBufferArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::SampledImageArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::StorageBufferArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::StorageImageArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::InputAttachmentArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::UniformTexelBufferArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Capability::StorageTexelBufferArrayNonUniformIndexing: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(Capability value) {
  switch (value) {
  case Capability::Groups: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_ballot}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupBallotKHR: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_ballot}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupVoteKHR: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_subgroup_vote}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StorageBuffer16BitAccess: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_16bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StoragePushConstant16: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_16bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StorageInputOutput16: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_16bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::DeviceGroup: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_device_group}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::AtomicStorageOps: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_atomic_counter_ops}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SampleMaskPostDepthCoverage: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_post_depth_coverage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StorageBuffer8BitAccess: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_8bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StoragePushConstant8: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_8bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::DenormPreserve: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::DenormFlushToZero: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SignedZeroInfNanPreserve: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::RoundingModeRTE: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::RoundingModeRTZ: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ImageFootprintNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_image_footprint}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentBarycentricNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_fragment_shader_barycentric}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ComputeDerivativeGroupQuadsNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_compute_shader_derivatives}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::GroupNonUniformPartitionedNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_subgroup_partitioned}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ComputeDerivativeGroupLinearNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_compute_shader_derivatives}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupShuffleINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupBufferBlockIOINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupImageBlockIOINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupImageMediaBlockIOINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_media_block_io}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupAvcMotionEstimationINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_device_side_avc_motion_estimation}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupAvcMotionEstimationIntraINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_device_side_avc_motion_estimation}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SubgroupAvcMotionEstimationChromaINTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_device_side_avc_motion_estimation}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StorageUniform16: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_16bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::UniformAndStorageBuffer8BitAccess: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_8bit_storage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::DrawParameters: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_draw_parameters}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::MultiView: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_multiview}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::VariablePointersStorageBuffer: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_variable_pointers}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::Float16ImageAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_gpu_shader_half_float_fetch}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ImageGatherBiasLodAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_texture_gather_bias_lod}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentMaskAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_fragment_mask}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::StencilExportEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_shader_stencil_export}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ImageReadWriteLodAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_image_load_store_lod}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ShaderClockKHR: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_clock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentFullyCoveredEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_fully_covered}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::MeshShadingNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentDensityEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_invocation_density, ::mlir::spirv::Extension::SPV_NV_shading_rate}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::RayTracingNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::PhysicalStorageBufferAddresses: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_physical_storage_buffer, ::mlir::spirv::Extension::SPV_KHR_physical_storage_buffer}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::CooperativeMatrixNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentShaderSampleInterlockEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentShaderShadingRateInterlockEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ShaderSMBuiltinsNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_sm_builtins}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::FragmentShaderPixelInterlockEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::DemoteToHelperInvocationEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_demote_to_helper_invocation}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::IntegerFunctions2INTEL: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_shader_integer_functions2}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::VariablePointers: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_variable_pointers}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::SampleMaskOverrideCoverageNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_sample_mask_override_coverage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::GeometryShaderPassthroughNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_geometry_shader_passthrough}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::PerViewAttributesNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NVX_multiview_per_view_attributes}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ShaderViewportIndexLayerEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_shader_viewport_index_layer}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ShaderViewportMaskNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_viewport_array2}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Capability::ShaderStereoViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_stereo_view_rendering}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(Decoration value) {
  switch (value) {
  case Decoration::RelaxedPrecision: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::SpecId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Block: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::BufferBlock: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::RowMajor: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::ColMajor: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::ArrayStride: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::MatrixStride: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::GLSLShared: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::GLSLPacked: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::CPacked: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::NoPerspective: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Flat: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Patch: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Centroid: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Sample: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SampleRateShading}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Invariant: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Constant: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Uniform: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::UniformId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::SaturatedConversion: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Stream: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GeometryStreams}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Location: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Component: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Index: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Binding: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::DescriptorSet: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Offset: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::XfbBuffer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::TransformFeedback}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::XfbStride: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::TransformFeedback}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::FuncParamAttr: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::FPFastMathMode: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::LinkageAttributes: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Linkage}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::NoContraction: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::InputAttachmentIndex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::InputAttachment}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::Alignment: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::MaxByteOffset: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Addresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::AlignmentId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::MaxByteOffsetId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Addresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::OverrideCoverageNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SampleMaskOverrideCoverageNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::PassthroughNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GeometryShaderPassthroughNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::ViewportRelativeNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderViewportMaskNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::SecondaryViewportRelativeNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderStereoViewNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::PerPrimitiveNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::PerViewNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::PerTaskNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::PerVertexNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentBarycentricNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::NonUniform: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ShaderNonUniform}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::RestrictPointer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::PhysicalStorageBufferAddresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Decoration::AliasedPointer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::PhysicalStorageBufferAddresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(Decoration value) {
  switch (value) {
  case Decoration::UniformId: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  case Decoration::MaxByteOffset: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case Decoration::AlignmentId: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_2); }
  case Decoration::MaxByteOffsetId: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_2); }
  case Decoration::NonUniform: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case Decoration::CounterBuffer: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  case Decoration::UserSemantic: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMaxVersion(Decoration value) {
  switch (value) {
  case Decoration::BufferBlock: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(Decoration value) {
  switch (value) {
  case Decoration::NoSignedWrap: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_no_integer_wrap_decoration}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::NoUnsignedWrap: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_no_integer_wrap_decoration}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::ExplicitInterpAMD: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_AMD_shader_explicit_vertex_parameter}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::OverrideCoverageNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_sample_mask_override_coverage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::PassthroughNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_geometry_shader_passthrough}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::SecondaryViewportRelativeNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_stereo_view_rendering}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::PerPrimitiveNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::PerViewNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::PerTaskNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::PerVertexNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_fragment_shader_barycentric}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::RestrictPointer: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_physical_storage_buffer, ::mlir::spirv::Extension::SPV_KHR_physical_storage_buffer}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::AliasedPointer: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_physical_storage_buffer, ::mlir::spirv::Extension::SPV_KHR_physical_storage_buffer}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case Decoration::UserTypeGOOGLE: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_GOOGLE_user_type}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(Dim value) {
  switch (value) {
  case Dim::Dim1D: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Image1D, ::mlir::spirv::Capability::Sampled1D}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Dim::Dim2D: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ImageMSArray, ::mlir::spirv::Capability::Kernel, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Dim::Cube: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ImageCubeArray, ::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Dim::Rect: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ImageRect, ::mlir::spirv::Capability::SampledRect}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Dim::Buffer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ImageBuffer, ::mlir::spirv::Capability::SampledBuffer}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case Dim::SubpassData: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::InputAttachment}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(ExecutionMode value) {
  switch (value) {
  case ExecutionMode::Invocations: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SpacingEqual: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SpacingFractionalEven: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SpacingFractionalOdd: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::VertexOrderCw: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::VertexOrderCcw: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::PixelCenterInteger: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OriginUpperLeft: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OriginLowerLeft: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::EarlyFragmentTests: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::PointMode: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::Xfb: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::TransformFeedback}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DepthReplacing: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DepthGreater: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DepthLess: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DepthUnchanged: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::LocalSizeHint: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::InputPoints: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::InputLines: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::InputLinesAdjacency: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::Triangles: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry, ::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::InputTrianglesAdjacency: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::Quads: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::Isolines: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputVertices: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry, ::mlir::spirv::Capability::MeshShadingNV, ::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputPoints: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry, ::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputLineStrip: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputTriangleStrip: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::VecTypeHint: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::ContractionOff: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::Initializer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::Finalizer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SubgroupSize: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupDispatch}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SubgroupsPerWorkgroup: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupDispatch}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SubgroupsPerWorkgroupId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupDispatch}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::LocalSizeHintId: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::PostDepthCoverage: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SampleMaskPostDepthCoverage}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DenormPreserve: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::DenormPreserve}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DenormFlushToZero: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::DenormFlushToZero}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SignedZeroInfNanPreserve: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SignedZeroInfNanPreserve}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::RoundingModeRTE: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RoundingModeRTE}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::RoundingModeRTZ: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RoundingModeRTZ}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::StencilRefReplacingEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StencilExportEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputLinesNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputPrimitivesNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DerivativeGroupQuadsNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ComputeDerivativeGroupQuadsNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::DerivativeGroupLinearNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::ComputeDerivativeGroupLinearNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::OutputTrianglesNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::PixelInterlockOrderedEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentShaderPixelInterlockEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::PixelInterlockUnorderedEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentShaderPixelInterlockEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SampleInterlockOrderedEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentShaderSampleInterlockEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::SampleInterlockUnorderedEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentShaderSampleInterlockEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::ShadingRateInterlockOrderedEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentShaderShadingRateInterlockEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionMode::ShadingRateInterlockUnorderedEXT: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::FragmentShaderShadingRateInterlockEXT}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(ExecutionMode value) {
  switch (value) {
  case ExecutionMode::Initializer: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case ExecutionMode::Finalizer: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case ExecutionMode::SubgroupSize: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case ExecutionMode::SubgroupsPerWorkgroup: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case ExecutionMode::SubgroupsPerWorkgroupId: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_2); }
  case ExecutionMode::LocalSizeId: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_2); }
  case ExecutionMode::LocalSizeHintId: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_2); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(ExecutionMode value) {
  switch (value) {
  case ExecutionMode::PostDepthCoverage: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_post_depth_coverage}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::DenormPreserve: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::DenormFlushToZero: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::SignedZeroInfNanPreserve: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::RoundingModeRTE: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::RoundingModeRTZ: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_float_controls}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::StencilRefReplacingEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_shader_stencil_export}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::OutputLinesNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::OutputPrimitivesNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::DerivativeGroupQuadsNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_compute_shader_derivatives}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::DerivativeGroupLinearNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_compute_shader_derivatives}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::OutputTrianglesNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_mesh_shader}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::PixelInterlockOrderedEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::PixelInterlockUnorderedEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::SampleInterlockOrderedEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::SampleInterlockUnorderedEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::ShadingRateInterlockOrderedEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case ExecutionMode::ShadingRateInterlockUnorderedEXT: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_fragment_shader_interlock}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(ExecutionModel value) {
  switch (value) {
  case ExecutionModel::Vertex: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::TessellationControl: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::TessellationEvaluation: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Tessellation}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::Geometry: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Geometry}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::Fragment: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::GLCompute: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::Kernel: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::TaskNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::MeshNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::MeshShadingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::RayGenerationNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::IntersectionNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::AnyHitNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::ClosestHitNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::MissNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ExecutionModel::CallableNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(GroupOperation value) {
  switch (value) {
  case GroupOperation::Reduce: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case GroupOperation::InclusiveScan: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case GroupOperation::ExclusiveScan: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformBallot, ::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case GroupOperation::ClusteredReduce: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformClustered}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case GroupOperation::PartitionedReduceNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case GroupOperation::PartitionedInclusiveScanNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case GroupOperation::PartitionedExclusiveScanNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(GroupOperation value) {
  switch (value) {
  case GroupOperation::ClusteredReduce: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_3); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(GroupOperation value) {
  switch (value) {
  case GroupOperation::PartitionedReduceNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_subgroup_partitioned}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case GroupOperation::PartitionedInclusiveScanNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_subgroup_partitioned}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case GroupOperation::PartitionedExclusiveScanNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_shader_subgroup_partitioned}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(ImageFormat value) {
  switch (value) {
  case ImageFormat::Rgba32f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba16f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R32f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba8: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba8Snorm: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg32f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg16f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R11fG11fB10f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R16f: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba16: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgb10A2: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg16: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg8: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R16: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R8: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba16Snorm: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg16Snorm: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg8Snorm: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R16Snorm: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R8Snorm: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba32i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba16i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba8i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R32i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg32i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg16i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg8i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R16i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R8i: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba32ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba16ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgba8ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R32ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rgb10a2ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg32ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg16ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::Rg8ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R16ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case ImageFormat::R8ui: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::StorageImageExtendedFormats}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(LinkageType value) {
  switch (value) {
  case LinkageType::Export: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Linkage}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case LinkageType::Import: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Linkage}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::mlir::spirv::Version> getMinVersion(LoopControl value) {
  assert(::llvm::countPopulation(static_cast<uint32_t>(value)) <= 1 && "cannot have more than one bit set");
  switch (value) {
  case LoopControl::DependencyInfinite: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case LoopControl::DependencyLength: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_1); }
  case LoopControl::MinIterations: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  case LoopControl::MaxIterations: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  case LoopControl::IterationMultiple: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  case LoopControl::PeelCount: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  case LoopControl::PartialCount: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_4); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(MemoryAccess value) {
  assert(::llvm::countPopulation(static_cast<uint32_t>(value)) <= 1 && "cannot have more than one bit set");
  switch (value) {
  case MemoryAccess::MakePointerAvailable: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemoryAccess::MakePointerVisible: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemoryAccess::NonPrivatePointer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(MemoryAccess value) {
  assert(::llvm::countPopulation(static_cast<uint32_t>(value)) <= 1 && "cannot have more than one bit set");
  switch (value) {
  case MemoryAccess::MakePointerAvailable: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case MemoryAccess::MakePointerVisible: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case MemoryAccess::NonPrivatePointer: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(MemoryModel value) {
  switch (value) {
  case MemoryModel::Simple: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemoryModel::GLSL450: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemoryModel::OpenCL: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemoryModel::Vulkan: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(MemoryModel value) {
  switch (value) {
  case MemoryModel::Vulkan: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_vulkan_memory_model}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(MemorySemantics value) {
  assert(::llvm::countPopulation(static_cast<uint32_t>(value)) <= 1 && "cannot have more than one bit set");
  switch (value) {
  case MemorySemantics::UniformMemory: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemorySemantics::AtomicCounterMemory: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::AtomicStorage}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemorySemantics::OutputMemory: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemorySemantics::MakeAvailable: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemorySemantics::MakeVisible: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case MemorySemantics::Volatile: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(MemorySemantics value) {
  assert(::llvm::countPopulation(static_cast<uint32_t>(value)) <= 1 && "cannot have more than one bit set");
  switch (value) {
  case MemorySemantics::OutputMemory: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case MemorySemantics::MakeAvailable: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  case MemorySemantics::MakeVisible: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(MemorySemantics value) {
  assert(::llvm::countPopulation(static_cast<uint32_t>(value)) <= 1 && "cannot have more than one bit set");
  switch (value) {
  case MemorySemantics::Volatile: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_vulkan_memory_model}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(Scope value) {
  switch (value) {
  case Scope::QueueFamily: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::VulkanMemoryModel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::mlir::spirv::Version> getMinVersion(Scope value) {
  switch (value) {
  case Scope::QueueFamily: {  return ::mlir::spirv::Version(::mlir::spirv::Version::V_1_5); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Capability>> getCapabilities(StorageClass value) {
  switch (value) {
  case StorageClass::Uniform: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::Output: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::Private: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::Generic: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GenericPointer}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::PushConstant: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::AtomicCounter: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::AtomicStorage}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::StorageBuffer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::CallableDataNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::IncomingCallableDataNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::RayPayloadNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::HitAttributeNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::IncomingRayPayloadNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::ShaderRecordBufferNV: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::RayTracingNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  case StorageClass::PhysicalStorageBuffer: { static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::PhysicalStorageBufferAddresses}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps)); return ::llvm::ArrayRef<::mlir::spirv::Capability>(ref); }
  default: break;
  }
  return llvm::None;
}
llvm::Optional<::llvm::ArrayRef<::mlir::spirv::Extension>> getExtensions(StorageClass value) {
  switch (value) {
  case StorageClass::StorageBuffer: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class, ::mlir::spirv::Extension::SPV_KHR_variable_pointers}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::CallableDataNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::IncomingCallableDataNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::RayPayloadNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::HitAttributeNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::IncomingRayPayloadNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::ShaderRecordBufferNV: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_ray_tracing}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  case StorageClass::PhysicalStorageBuffer: { static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_EXT_physical_storage_buffer, ::mlir::spirv::Extension::SPV_KHR_physical_storage_buffer}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts)); return ::llvm::ArrayRef<::mlir::spirv::Extension>(ref); }
  default: break;
  }
  return llvm::None;
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
} // namespace spirv
} // namespace mlir

