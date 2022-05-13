/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Capability Implication                                              *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

ArrayRef<spirv::Capability> spirv::getDirectImpliedCapabilities(spirv::Capability cap) {
  switch (cap) {
  default: return {};
  case spirv::Capability::Shader: {static const spirv::Capability implies[1] = {spirv::Capability::Matrix}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Vector16: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Float16Buffer: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Int64Atomics: {static const spirv::Capability implies[1] = {spirv::Capability::Int64}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageBasic: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Pipes: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::DeviceEnqueue: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::LiteralSampler: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GenericPointer: {static const spirv::Capability implies[1] = {spirv::Capability::Addresses}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Image1D: {static const spirv::Capability implies[1] = {spirv::Capability::Sampled1D}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageBuffer: {static const spirv::Capability implies[1] = {spirv::Capability::SampledBuffer}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::NamedBarrier: {static const spirv::Capability implies[1] = {spirv::Capability::Kernel}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformVote: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformArithmetic: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformBallot: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformShuffle: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformShuffleRelative: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformClustered: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GroupNonUniformQuad: {static const spirv::Capability implies[1] = {spirv::Capability::GroupNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageUniform16: {static const spirv::Capability implies[1] = {spirv::Capability::StorageBuffer16BitAccess}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::UniformAndStorageBuffer8BitAccess: {static const spirv::Capability implies[1] = {spirv::Capability::StorageBuffer8BitAccess}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::UniformTexelBufferArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::SampledBuffer}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Geometry: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Tessellation: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageReadWrite: {static const spirv::Capability implies[1] = {spirv::Capability::ImageBasic}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageMipmap: {static const spirv::Capability implies[1] = {spirv::Capability::ImageBasic}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::AtomicStorage: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageGatherExtended: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageImageMultisample: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::UniformBufferArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SampledImageArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageBufferArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageImageArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ClipDistance: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::CullDistance: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SampleRateShading: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SampledRect: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::InputAttachment: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SparseResidency: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::MinLod: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SampledCubeArray: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageMSArray: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageImageExtendedFormats: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageQuery: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::DerivativeControl: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::InterpolationFunction: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::TransformFeedback: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageImageReadWithoutFormat: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageImageWriteWithoutFormat: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SubgroupDispatch: {static const spirv::Capability implies[1] = {spirv::Capability::DeviceEnqueue}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::PipeStorage: {static const spirv::Capability implies[1] = {spirv::Capability::Pipes}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::DrawParameters: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::MultiView: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::VariablePointersStorageBuffer: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::Float16ImageAMD: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageGatherBiasLodAMD: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::FragmentMaskAMD: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StencilExportEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageReadWriteLodAMD: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ShaderClockKHR: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::FragmentFullyCoveredEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::MeshShadingNV: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::FragmentDensityEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ShaderNonUniform: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::RuntimeDescriptorArray: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageTexelBufferArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::ImageBuffer}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::RayTracingNV: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::PhysicalStorageBufferAddresses: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::CooperativeMatrixNV: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::FragmentShaderSampleInterlockEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::FragmentShaderShadingRateInterlockEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ShaderSMBuiltinsNV: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::FragmentShaderPixelInterlockEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::DemoteToHelperInvocationEXT: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::IntegerFunctions2INTEL: {static const spirv::Capability implies[1] = {spirv::Capability::Shader}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::TessellationPointSize: {static const spirv::Capability implies[1] = {spirv::Capability::Tessellation}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GeometryPointSize: {static const spirv::Capability implies[1] = {spirv::Capability::Geometry}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageCubeArray: {static const spirv::Capability implies[1] = {spirv::Capability::SampledCubeArray}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ImageRect: {static const spirv::Capability implies[1] = {spirv::Capability::SampledRect}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GeometryStreams: {static const spirv::Capability implies[1] = {spirv::Capability::Geometry}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::MultiViewport: {static const spirv::Capability implies[1] = {spirv::Capability::Geometry}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::VariablePointers: {static const spirv::Capability implies[1] = {spirv::Capability::VariablePointersStorageBuffer}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SampleMaskOverrideCoverageNV: {static const spirv::Capability implies[1] = {spirv::Capability::SampleRateShading}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::GeometryShaderPassthroughNV: {static const spirv::Capability implies[1] = {spirv::Capability::Geometry}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::PerViewAttributesNV: {static const spirv::Capability implies[1] = {spirv::Capability::MultiView}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::InputAttachmentArrayDynamicIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::InputAttachment}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::UniformBufferArrayNonUniformIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::SampledImageArrayNonUniformIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageBufferArrayNonUniformIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::StorageImageArrayNonUniformIndexing: {static const spirv::Capability implies[1] = {spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::InputAttachmentArrayNonUniformIndexing: {static const spirv::Capability implies[2] = {spirv::Capability::InputAttachment, spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 2); }
  case spirv::Capability::UniformTexelBufferArrayNonUniformIndexing: {static const spirv::Capability implies[2] = {spirv::Capability::SampledBuffer, spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 2); }
  case spirv::Capability::StorageTexelBufferArrayNonUniformIndexing: {static const spirv::Capability implies[2] = {spirv::Capability::ImageBuffer, spirv::Capability::ShaderNonUniform}; return ArrayRef<spirv::Capability>(implies, 2); }
  case spirv::Capability::ShaderViewportIndexLayerEXT: {static const spirv::Capability implies[1] = {spirv::Capability::MultiViewport}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ShaderViewportMaskNV: {static const spirv::Capability implies[1] = {spirv::Capability::ShaderViewportIndexLayerEXT}; return ArrayRef<spirv::Capability>(implies, 1); }
  case spirv::Capability::ShaderStereoViewNV: {static const spirv::Capability implies[1] = {spirv::Capability::ShaderViewportMaskNV}; return ArrayRef<spirv::Capability>(implies, 1); }
  }
}
