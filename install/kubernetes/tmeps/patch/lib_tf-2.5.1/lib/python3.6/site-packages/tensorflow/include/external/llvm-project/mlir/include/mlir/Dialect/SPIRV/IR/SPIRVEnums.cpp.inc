/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyAddressingModel(AddressingModel val) {
  switch (val) {
    case AddressingModel::Logical: return "Logical";
    case AddressingModel::Physical32: return "Physical32";
    case AddressingModel::Physical64: return "Physical64";
    case AddressingModel::PhysicalStorageBuffer64: return "PhysicalStorageBuffer64";
  }
  return "";
}

::llvm::Optional<AddressingModel> symbolizeAddressingModel(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AddressingModel>>(str)
      .Case("Logical", AddressingModel::Logical)
      .Case("Physical32", AddressingModel::Physical32)
      .Case("Physical64", AddressingModel::Physical64)
      .Case("PhysicalStorageBuffer64", AddressingModel::PhysicalStorageBuffer64)
      .Default(::llvm::None);
}
::llvm::Optional<AddressingModel> symbolizeAddressingModel(uint32_t value) {
  switch (value) {
  case 0: return AddressingModel::Logical;
  case 1: return AddressingModel::Physical32;
  case 2: return AddressingModel::Physical64;
  case 5348: return AddressingModel::PhysicalStorageBuffer64;
  default: return ::llvm::None;
  }
}

bool AddressingModelAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5348)));
}
AddressingModelAttr AddressingModelAttr::get(::mlir::MLIRContext *context, AddressingModel val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<AddressingModelAttr>();
}
AddressingModel AddressingModelAttr::getValue() const {
  return static_cast<AddressingModel>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyImageArrayedInfo(ImageArrayedInfo val) {
  switch (val) {
    case ImageArrayedInfo::NonArrayed: return "NonArrayed";
    case ImageArrayedInfo::Arrayed: return "Arrayed";
  }
  return "";
}

::llvm::Optional<ImageArrayedInfo> symbolizeImageArrayedInfo(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ImageArrayedInfo>>(str)
      .Case("NonArrayed", ImageArrayedInfo::NonArrayed)
      .Case("Arrayed", ImageArrayedInfo::Arrayed)
      .Default(::llvm::None);
}
::llvm::Optional<ImageArrayedInfo> symbolizeImageArrayedInfo(uint32_t value) {
  switch (value) {
  case 0: return ImageArrayedInfo::NonArrayed;
  case 1: return ImageArrayedInfo::Arrayed;
  default: return ::llvm::None;
  }
}

bool ImageArrayedInfoAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)));
}
ImageArrayedInfoAttr ImageArrayedInfoAttr::get(::mlir::MLIRContext *context, ImageArrayedInfo val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ImageArrayedInfoAttr>();
}
ImageArrayedInfo ImageArrayedInfoAttr::getValue() const {
  return static_cast<ImageArrayedInfo>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyBuiltIn(BuiltIn val) {
  switch (val) {
    case BuiltIn::Position: return "Position";
    case BuiltIn::PointSize: return "PointSize";
    case BuiltIn::ClipDistance: return "ClipDistance";
    case BuiltIn::CullDistance: return "CullDistance";
    case BuiltIn::VertexId: return "VertexId";
    case BuiltIn::InstanceId: return "InstanceId";
    case BuiltIn::PrimitiveId: return "PrimitiveId";
    case BuiltIn::InvocationId: return "InvocationId";
    case BuiltIn::Layer: return "Layer";
    case BuiltIn::ViewportIndex: return "ViewportIndex";
    case BuiltIn::TessLevelOuter: return "TessLevelOuter";
    case BuiltIn::TessLevelInner: return "TessLevelInner";
    case BuiltIn::TessCoord: return "TessCoord";
    case BuiltIn::PatchVertices: return "PatchVertices";
    case BuiltIn::FragCoord: return "FragCoord";
    case BuiltIn::PointCoord: return "PointCoord";
    case BuiltIn::FrontFacing: return "FrontFacing";
    case BuiltIn::SampleId: return "SampleId";
    case BuiltIn::SamplePosition: return "SamplePosition";
    case BuiltIn::SampleMask: return "SampleMask";
    case BuiltIn::FragDepth: return "FragDepth";
    case BuiltIn::HelperInvocation: return "HelperInvocation";
    case BuiltIn::NumWorkgroups: return "NumWorkgroups";
    case BuiltIn::WorkgroupSize: return "WorkgroupSize";
    case BuiltIn::WorkgroupId: return "WorkgroupId";
    case BuiltIn::LocalInvocationId: return "LocalInvocationId";
    case BuiltIn::GlobalInvocationId: return "GlobalInvocationId";
    case BuiltIn::LocalInvocationIndex: return "LocalInvocationIndex";
    case BuiltIn::WorkDim: return "WorkDim";
    case BuiltIn::GlobalSize: return "GlobalSize";
    case BuiltIn::EnqueuedWorkgroupSize: return "EnqueuedWorkgroupSize";
    case BuiltIn::GlobalOffset: return "GlobalOffset";
    case BuiltIn::GlobalLinearId: return "GlobalLinearId";
    case BuiltIn::SubgroupSize: return "SubgroupSize";
    case BuiltIn::SubgroupMaxSize: return "SubgroupMaxSize";
    case BuiltIn::NumSubgroups: return "NumSubgroups";
    case BuiltIn::NumEnqueuedSubgroups: return "NumEnqueuedSubgroups";
    case BuiltIn::SubgroupId: return "SubgroupId";
    case BuiltIn::SubgroupLocalInvocationId: return "SubgroupLocalInvocationId";
    case BuiltIn::VertexIndex: return "VertexIndex";
    case BuiltIn::InstanceIndex: return "InstanceIndex";
    case BuiltIn::SubgroupEqMask: return "SubgroupEqMask";
    case BuiltIn::SubgroupGeMask: return "SubgroupGeMask";
    case BuiltIn::SubgroupGtMask: return "SubgroupGtMask";
    case BuiltIn::SubgroupLeMask: return "SubgroupLeMask";
    case BuiltIn::SubgroupLtMask: return "SubgroupLtMask";
    case BuiltIn::BaseVertex: return "BaseVertex";
    case BuiltIn::BaseInstance: return "BaseInstance";
    case BuiltIn::DrawIndex: return "DrawIndex";
    case BuiltIn::DeviceIndex: return "DeviceIndex";
    case BuiltIn::ViewIndex: return "ViewIndex";
    case BuiltIn::BaryCoordNoPerspAMD: return "BaryCoordNoPerspAMD";
    case BuiltIn::BaryCoordNoPerspCentroidAMD: return "BaryCoordNoPerspCentroidAMD";
    case BuiltIn::BaryCoordNoPerspSampleAMD: return "BaryCoordNoPerspSampleAMD";
    case BuiltIn::BaryCoordSmoothAMD: return "BaryCoordSmoothAMD";
    case BuiltIn::BaryCoordSmoothCentroidAMD: return "BaryCoordSmoothCentroidAMD";
    case BuiltIn::BaryCoordSmoothSampleAMD: return "BaryCoordSmoothSampleAMD";
    case BuiltIn::BaryCoordPullModelAMD: return "BaryCoordPullModelAMD";
    case BuiltIn::FragStencilRefEXT: return "FragStencilRefEXT";
    case BuiltIn::ViewportMaskNV: return "ViewportMaskNV";
    case BuiltIn::SecondaryPositionNV: return "SecondaryPositionNV";
    case BuiltIn::SecondaryViewportMaskNV: return "SecondaryViewportMaskNV";
    case BuiltIn::PositionPerViewNV: return "PositionPerViewNV";
    case BuiltIn::ViewportMaskPerViewNV: return "ViewportMaskPerViewNV";
    case BuiltIn::FullyCoveredEXT: return "FullyCoveredEXT";
    case BuiltIn::TaskCountNV: return "TaskCountNV";
    case BuiltIn::PrimitiveCountNV: return "PrimitiveCountNV";
    case BuiltIn::PrimitiveIndicesNV: return "PrimitiveIndicesNV";
    case BuiltIn::ClipDistancePerViewNV: return "ClipDistancePerViewNV";
    case BuiltIn::CullDistancePerViewNV: return "CullDistancePerViewNV";
    case BuiltIn::LayerPerViewNV: return "LayerPerViewNV";
    case BuiltIn::MeshViewCountNV: return "MeshViewCountNV";
    case BuiltIn::MeshViewIndicesNV: return "MeshViewIndicesNV";
    case BuiltIn::BaryCoordNV: return "BaryCoordNV";
    case BuiltIn::BaryCoordNoPerspNV: return "BaryCoordNoPerspNV";
    case BuiltIn::FragSizeEXT: return "FragSizeEXT";
    case BuiltIn::FragInvocationCountEXT: return "FragInvocationCountEXT";
    case BuiltIn::LaunchIdNV: return "LaunchIdNV";
    case BuiltIn::LaunchSizeNV: return "LaunchSizeNV";
    case BuiltIn::WorldRayOriginNV: return "WorldRayOriginNV";
    case BuiltIn::WorldRayDirectionNV: return "WorldRayDirectionNV";
    case BuiltIn::ObjectRayOriginNV: return "ObjectRayOriginNV";
    case BuiltIn::ObjectRayDirectionNV: return "ObjectRayDirectionNV";
    case BuiltIn::RayTminNV: return "RayTminNV";
    case BuiltIn::RayTmaxNV: return "RayTmaxNV";
    case BuiltIn::InstanceCustomIndexNV: return "InstanceCustomIndexNV";
    case BuiltIn::ObjectToWorldNV: return "ObjectToWorldNV";
    case BuiltIn::WorldToObjectNV: return "WorldToObjectNV";
    case BuiltIn::HitTNV: return "HitTNV";
    case BuiltIn::HitKindNV: return "HitKindNV";
    case BuiltIn::IncomingRayFlagsNV: return "IncomingRayFlagsNV";
    case BuiltIn::WarpsPerSMNV: return "WarpsPerSMNV";
    case BuiltIn::SMCountNV: return "SMCountNV";
    case BuiltIn::WarpIDNV: return "WarpIDNV";
    case BuiltIn::SMIDNV: return "SMIDNV";
  }
  return "";
}

::llvm::Optional<BuiltIn> symbolizeBuiltIn(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<BuiltIn>>(str)
      .Case("Position", BuiltIn::Position)
      .Case("PointSize", BuiltIn::PointSize)
      .Case("ClipDistance", BuiltIn::ClipDistance)
      .Case("CullDistance", BuiltIn::CullDistance)
      .Case("VertexId", BuiltIn::VertexId)
      .Case("InstanceId", BuiltIn::InstanceId)
      .Case("PrimitiveId", BuiltIn::PrimitiveId)
      .Case("InvocationId", BuiltIn::InvocationId)
      .Case("Layer", BuiltIn::Layer)
      .Case("ViewportIndex", BuiltIn::ViewportIndex)
      .Case("TessLevelOuter", BuiltIn::TessLevelOuter)
      .Case("TessLevelInner", BuiltIn::TessLevelInner)
      .Case("TessCoord", BuiltIn::TessCoord)
      .Case("PatchVertices", BuiltIn::PatchVertices)
      .Case("FragCoord", BuiltIn::FragCoord)
      .Case("PointCoord", BuiltIn::PointCoord)
      .Case("FrontFacing", BuiltIn::FrontFacing)
      .Case("SampleId", BuiltIn::SampleId)
      .Case("SamplePosition", BuiltIn::SamplePosition)
      .Case("SampleMask", BuiltIn::SampleMask)
      .Case("FragDepth", BuiltIn::FragDepth)
      .Case("HelperInvocation", BuiltIn::HelperInvocation)
      .Case("NumWorkgroups", BuiltIn::NumWorkgroups)
      .Case("WorkgroupSize", BuiltIn::WorkgroupSize)
      .Case("WorkgroupId", BuiltIn::WorkgroupId)
      .Case("LocalInvocationId", BuiltIn::LocalInvocationId)
      .Case("GlobalInvocationId", BuiltIn::GlobalInvocationId)
      .Case("LocalInvocationIndex", BuiltIn::LocalInvocationIndex)
      .Case("WorkDim", BuiltIn::WorkDim)
      .Case("GlobalSize", BuiltIn::GlobalSize)
      .Case("EnqueuedWorkgroupSize", BuiltIn::EnqueuedWorkgroupSize)
      .Case("GlobalOffset", BuiltIn::GlobalOffset)
      .Case("GlobalLinearId", BuiltIn::GlobalLinearId)
      .Case("SubgroupSize", BuiltIn::SubgroupSize)
      .Case("SubgroupMaxSize", BuiltIn::SubgroupMaxSize)
      .Case("NumSubgroups", BuiltIn::NumSubgroups)
      .Case("NumEnqueuedSubgroups", BuiltIn::NumEnqueuedSubgroups)
      .Case("SubgroupId", BuiltIn::SubgroupId)
      .Case("SubgroupLocalInvocationId", BuiltIn::SubgroupLocalInvocationId)
      .Case("VertexIndex", BuiltIn::VertexIndex)
      .Case("InstanceIndex", BuiltIn::InstanceIndex)
      .Case("SubgroupEqMask", BuiltIn::SubgroupEqMask)
      .Case("SubgroupGeMask", BuiltIn::SubgroupGeMask)
      .Case("SubgroupGtMask", BuiltIn::SubgroupGtMask)
      .Case("SubgroupLeMask", BuiltIn::SubgroupLeMask)
      .Case("SubgroupLtMask", BuiltIn::SubgroupLtMask)
      .Case("BaseVertex", BuiltIn::BaseVertex)
      .Case("BaseInstance", BuiltIn::BaseInstance)
      .Case("DrawIndex", BuiltIn::DrawIndex)
      .Case("DeviceIndex", BuiltIn::DeviceIndex)
      .Case("ViewIndex", BuiltIn::ViewIndex)
      .Case("BaryCoordNoPerspAMD", BuiltIn::BaryCoordNoPerspAMD)
      .Case("BaryCoordNoPerspCentroidAMD", BuiltIn::BaryCoordNoPerspCentroidAMD)
      .Case("BaryCoordNoPerspSampleAMD", BuiltIn::BaryCoordNoPerspSampleAMD)
      .Case("BaryCoordSmoothAMD", BuiltIn::BaryCoordSmoothAMD)
      .Case("BaryCoordSmoothCentroidAMD", BuiltIn::BaryCoordSmoothCentroidAMD)
      .Case("BaryCoordSmoothSampleAMD", BuiltIn::BaryCoordSmoothSampleAMD)
      .Case("BaryCoordPullModelAMD", BuiltIn::BaryCoordPullModelAMD)
      .Case("FragStencilRefEXT", BuiltIn::FragStencilRefEXT)
      .Case("ViewportMaskNV", BuiltIn::ViewportMaskNV)
      .Case("SecondaryPositionNV", BuiltIn::SecondaryPositionNV)
      .Case("SecondaryViewportMaskNV", BuiltIn::SecondaryViewportMaskNV)
      .Case("PositionPerViewNV", BuiltIn::PositionPerViewNV)
      .Case("ViewportMaskPerViewNV", BuiltIn::ViewportMaskPerViewNV)
      .Case("FullyCoveredEXT", BuiltIn::FullyCoveredEXT)
      .Case("TaskCountNV", BuiltIn::TaskCountNV)
      .Case("PrimitiveCountNV", BuiltIn::PrimitiveCountNV)
      .Case("PrimitiveIndicesNV", BuiltIn::PrimitiveIndicesNV)
      .Case("ClipDistancePerViewNV", BuiltIn::ClipDistancePerViewNV)
      .Case("CullDistancePerViewNV", BuiltIn::CullDistancePerViewNV)
      .Case("LayerPerViewNV", BuiltIn::LayerPerViewNV)
      .Case("MeshViewCountNV", BuiltIn::MeshViewCountNV)
      .Case("MeshViewIndicesNV", BuiltIn::MeshViewIndicesNV)
      .Case("BaryCoordNV", BuiltIn::BaryCoordNV)
      .Case("BaryCoordNoPerspNV", BuiltIn::BaryCoordNoPerspNV)
      .Case("FragSizeEXT", BuiltIn::FragSizeEXT)
      .Case("FragInvocationCountEXT", BuiltIn::FragInvocationCountEXT)
      .Case("LaunchIdNV", BuiltIn::LaunchIdNV)
      .Case("LaunchSizeNV", BuiltIn::LaunchSizeNV)
      .Case("WorldRayOriginNV", BuiltIn::WorldRayOriginNV)
      .Case("WorldRayDirectionNV", BuiltIn::WorldRayDirectionNV)
      .Case("ObjectRayOriginNV", BuiltIn::ObjectRayOriginNV)
      .Case("ObjectRayDirectionNV", BuiltIn::ObjectRayDirectionNV)
      .Case("RayTminNV", BuiltIn::RayTminNV)
      .Case("RayTmaxNV", BuiltIn::RayTmaxNV)
      .Case("InstanceCustomIndexNV", BuiltIn::InstanceCustomIndexNV)
      .Case("ObjectToWorldNV", BuiltIn::ObjectToWorldNV)
      .Case("WorldToObjectNV", BuiltIn::WorldToObjectNV)
      .Case("HitTNV", BuiltIn::HitTNV)
      .Case("HitKindNV", BuiltIn::HitKindNV)
      .Case("IncomingRayFlagsNV", BuiltIn::IncomingRayFlagsNV)
      .Case("WarpsPerSMNV", BuiltIn::WarpsPerSMNV)
      .Case("SMCountNV", BuiltIn::SMCountNV)
      .Case("WarpIDNV", BuiltIn::WarpIDNV)
      .Case("SMIDNV", BuiltIn::SMIDNV)
      .Default(::llvm::None);
}
::llvm::Optional<BuiltIn> symbolizeBuiltIn(uint32_t value) {
  switch (value) {
  case 0: return BuiltIn::Position;
  case 1: return BuiltIn::PointSize;
  case 3: return BuiltIn::ClipDistance;
  case 4: return BuiltIn::CullDistance;
  case 5: return BuiltIn::VertexId;
  case 6: return BuiltIn::InstanceId;
  case 7: return BuiltIn::PrimitiveId;
  case 8: return BuiltIn::InvocationId;
  case 9: return BuiltIn::Layer;
  case 10: return BuiltIn::ViewportIndex;
  case 11: return BuiltIn::TessLevelOuter;
  case 12: return BuiltIn::TessLevelInner;
  case 13: return BuiltIn::TessCoord;
  case 14: return BuiltIn::PatchVertices;
  case 15: return BuiltIn::FragCoord;
  case 16: return BuiltIn::PointCoord;
  case 17: return BuiltIn::FrontFacing;
  case 18: return BuiltIn::SampleId;
  case 19: return BuiltIn::SamplePosition;
  case 20: return BuiltIn::SampleMask;
  case 22: return BuiltIn::FragDepth;
  case 23: return BuiltIn::HelperInvocation;
  case 24: return BuiltIn::NumWorkgroups;
  case 25: return BuiltIn::WorkgroupSize;
  case 26: return BuiltIn::WorkgroupId;
  case 27: return BuiltIn::LocalInvocationId;
  case 28: return BuiltIn::GlobalInvocationId;
  case 29: return BuiltIn::LocalInvocationIndex;
  case 30: return BuiltIn::WorkDim;
  case 31: return BuiltIn::GlobalSize;
  case 32: return BuiltIn::EnqueuedWorkgroupSize;
  case 33: return BuiltIn::GlobalOffset;
  case 34: return BuiltIn::GlobalLinearId;
  case 36: return BuiltIn::SubgroupSize;
  case 37: return BuiltIn::SubgroupMaxSize;
  case 38: return BuiltIn::NumSubgroups;
  case 39: return BuiltIn::NumEnqueuedSubgroups;
  case 40: return BuiltIn::SubgroupId;
  case 41: return BuiltIn::SubgroupLocalInvocationId;
  case 42: return BuiltIn::VertexIndex;
  case 43: return BuiltIn::InstanceIndex;
  case 4416: return BuiltIn::SubgroupEqMask;
  case 4417: return BuiltIn::SubgroupGeMask;
  case 4418: return BuiltIn::SubgroupGtMask;
  case 4419: return BuiltIn::SubgroupLeMask;
  case 4420: return BuiltIn::SubgroupLtMask;
  case 4424: return BuiltIn::BaseVertex;
  case 4425: return BuiltIn::BaseInstance;
  case 4426: return BuiltIn::DrawIndex;
  case 4438: return BuiltIn::DeviceIndex;
  case 4440: return BuiltIn::ViewIndex;
  case 4992: return BuiltIn::BaryCoordNoPerspAMD;
  case 4993: return BuiltIn::BaryCoordNoPerspCentroidAMD;
  case 4994: return BuiltIn::BaryCoordNoPerspSampleAMD;
  case 4995: return BuiltIn::BaryCoordSmoothAMD;
  case 4996: return BuiltIn::BaryCoordSmoothCentroidAMD;
  case 4997: return BuiltIn::BaryCoordSmoothSampleAMD;
  case 4998: return BuiltIn::BaryCoordPullModelAMD;
  case 5014: return BuiltIn::FragStencilRefEXT;
  case 5253: return BuiltIn::ViewportMaskNV;
  case 5257: return BuiltIn::SecondaryPositionNV;
  case 5258: return BuiltIn::SecondaryViewportMaskNV;
  case 5261: return BuiltIn::PositionPerViewNV;
  case 5262: return BuiltIn::ViewportMaskPerViewNV;
  case 5264: return BuiltIn::FullyCoveredEXT;
  case 5274: return BuiltIn::TaskCountNV;
  case 5275: return BuiltIn::PrimitiveCountNV;
  case 5276: return BuiltIn::PrimitiveIndicesNV;
  case 5277: return BuiltIn::ClipDistancePerViewNV;
  case 5278: return BuiltIn::CullDistancePerViewNV;
  case 5279: return BuiltIn::LayerPerViewNV;
  case 5280: return BuiltIn::MeshViewCountNV;
  case 5281: return BuiltIn::MeshViewIndicesNV;
  case 5286: return BuiltIn::BaryCoordNV;
  case 5287: return BuiltIn::BaryCoordNoPerspNV;
  case 5292: return BuiltIn::FragSizeEXT;
  case 5293: return BuiltIn::FragInvocationCountEXT;
  case 5319: return BuiltIn::LaunchIdNV;
  case 5320: return BuiltIn::LaunchSizeNV;
  case 5321: return BuiltIn::WorldRayOriginNV;
  case 5322: return BuiltIn::WorldRayDirectionNV;
  case 5323: return BuiltIn::ObjectRayOriginNV;
  case 5324: return BuiltIn::ObjectRayDirectionNV;
  case 5325: return BuiltIn::RayTminNV;
  case 5326: return BuiltIn::RayTmaxNV;
  case 5327: return BuiltIn::InstanceCustomIndexNV;
  case 5330: return BuiltIn::ObjectToWorldNV;
  case 5331: return BuiltIn::WorldToObjectNV;
  case 5332: return BuiltIn::HitTNV;
  case 5333: return BuiltIn::HitKindNV;
  case 5351: return BuiltIn::IncomingRayFlagsNV;
  case 5374: return BuiltIn::WarpsPerSMNV;
  case 5375: return BuiltIn::SMCountNV;
  case 5376: return BuiltIn::WarpIDNV;
  case 5377: return BuiltIn::SMIDNV;
  default: return ::llvm::None;
  }
}

bool BuiltInAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 18)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 22)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 23)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 24)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 25)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 26)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 27)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 28)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 29)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 30)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 31)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 32)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 33)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 34)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 36)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 37)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 38)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 39)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 40)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 41)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 42)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 43)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4416)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4417)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4418)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4419)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4420)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4424)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4425)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4426)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4438)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4440)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4992)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4993)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4994)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4995)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4996)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4997)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4998)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5014)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5253)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5257)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5258)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5261)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5262)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5264)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5274)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5275)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5276)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5277)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5278)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5279)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5280)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5281)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5286)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5287)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5292)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5293)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5319)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5320)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5321)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5322)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5323)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5324)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5325)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5326)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5327)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5330)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5331)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5332)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5333)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5351)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5374)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5375)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5376)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5377)));
}
BuiltInAttr BuiltInAttr::get(::mlir::MLIRContext *context, BuiltIn val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<BuiltInAttr>();
}
BuiltIn BuiltInAttr::getValue() const {
  return static_cast<BuiltIn>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyCapability(Capability val) {
  switch (val) {
    case Capability::Matrix: return "Matrix";
    case Capability::Addresses: return "Addresses";
    case Capability::Linkage: return "Linkage";
    case Capability::Kernel: return "Kernel";
    case Capability::Float16: return "Float16";
    case Capability::Float64: return "Float64";
    case Capability::Int64: return "Int64";
    case Capability::Groups: return "Groups";
    case Capability::Int16: return "Int16";
    case Capability::Int8: return "Int8";
    case Capability::Sampled1D: return "Sampled1D";
    case Capability::SampledBuffer: return "SampledBuffer";
    case Capability::GroupNonUniform: return "GroupNonUniform";
    case Capability::ShaderLayer: return "ShaderLayer";
    case Capability::ShaderViewportIndex: return "ShaderViewportIndex";
    case Capability::SubgroupBallotKHR: return "SubgroupBallotKHR";
    case Capability::SubgroupVoteKHR: return "SubgroupVoteKHR";
    case Capability::StorageBuffer16BitAccess: return "StorageBuffer16BitAccess";
    case Capability::StoragePushConstant16: return "StoragePushConstant16";
    case Capability::StorageInputOutput16: return "StorageInputOutput16";
    case Capability::DeviceGroup: return "DeviceGroup";
    case Capability::AtomicStorageOps: return "AtomicStorageOps";
    case Capability::SampleMaskPostDepthCoverage: return "SampleMaskPostDepthCoverage";
    case Capability::StorageBuffer8BitAccess: return "StorageBuffer8BitAccess";
    case Capability::StoragePushConstant8: return "StoragePushConstant8";
    case Capability::DenormPreserve: return "DenormPreserve";
    case Capability::DenormFlushToZero: return "DenormFlushToZero";
    case Capability::SignedZeroInfNanPreserve: return "SignedZeroInfNanPreserve";
    case Capability::RoundingModeRTE: return "RoundingModeRTE";
    case Capability::RoundingModeRTZ: return "RoundingModeRTZ";
    case Capability::ImageFootprintNV: return "ImageFootprintNV";
    case Capability::FragmentBarycentricNV: return "FragmentBarycentricNV";
    case Capability::ComputeDerivativeGroupQuadsNV: return "ComputeDerivativeGroupQuadsNV";
    case Capability::GroupNonUniformPartitionedNV: return "GroupNonUniformPartitionedNV";
    case Capability::VulkanMemoryModel: return "VulkanMemoryModel";
    case Capability::VulkanMemoryModelDeviceScope: return "VulkanMemoryModelDeviceScope";
    case Capability::ComputeDerivativeGroupLinearNV: return "ComputeDerivativeGroupLinearNV";
    case Capability::SubgroupShuffleINTEL: return "SubgroupShuffleINTEL";
    case Capability::SubgroupBufferBlockIOINTEL: return "SubgroupBufferBlockIOINTEL";
    case Capability::SubgroupImageBlockIOINTEL: return "SubgroupImageBlockIOINTEL";
    case Capability::SubgroupImageMediaBlockIOINTEL: return "SubgroupImageMediaBlockIOINTEL";
    case Capability::SubgroupAvcMotionEstimationINTEL: return "SubgroupAvcMotionEstimationINTEL";
    case Capability::SubgroupAvcMotionEstimationIntraINTEL: return "SubgroupAvcMotionEstimationIntraINTEL";
    case Capability::SubgroupAvcMotionEstimationChromaINTEL: return "SubgroupAvcMotionEstimationChromaINTEL";
    case Capability::Shader: return "Shader";
    case Capability::Vector16: return "Vector16";
    case Capability::Float16Buffer: return "Float16Buffer";
    case Capability::Int64Atomics: return "Int64Atomics";
    case Capability::ImageBasic: return "ImageBasic";
    case Capability::Pipes: return "Pipes";
    case Capability::DeviceEnqueue: return "DeviceEnqueue";
    case Capability::LiteralSampler: return "LiteralSampler";
    case Capability::GenericPointer: return "GenericPointer";
    case Capability::Image1D: return "Image1D";
    case Capability::ImageBuffer: return "ImageBuffer";
    case Capability::NamedBarrier: return "NamedBarrier";
    case Capability::GroupNonUniformVote: return "GroupNonUniformVote";
    case Capability::GroupNonUniformArithmetic: return "GroupNonUniformArithmetic";
    case Capability::GroupNonUniformBallot: return "GroupNonUniformBallot";
    case Capability::GroupNonUniformShuffle: return "GroupNonUniformShuffle";
    case Capability::GroupNonUniformShuffleRelative: return "GroupNonUniformShuffleRelative";
    case Capability::GroupNonUniformClustered: return "GroupNonUniformClustered";
    case Capability::GroupNonUniformQuad: return "GroupNonUniformQuad";
    case Capability::StorageUniform16: return "StorageUniform16";
    case Capability::UniformAndStorageBuffer8BitAccess: return "UniformAndStorageBuffer8BitAccess";
    case Capability::UniformTexelBufferArrayDynamicIndexing: return "UniformTexelBufferArrayDynamicIndexing";
    case Capability::Geometry: return "Geometry";
    case Capability::Tessellation: return "Tessellation";
    case Capability::ImageReadWrite: return "ImageReadWrite";
    case Capability::ImageMipmap: return "ImageMipmap";
    case Capability::AtomicStorage: return "AtomicStorage";
    case Capability::ImageGatherExtended: return "ImageGatherExtended";
    case Capability::StorageImageMultisample: return "StorageImageMultisample";
    case Capability::UniformBufferArrayDynamicIndexing: return "UniformBufferArrayDynamicIndexing";
    case Capability::SampledImageArrayDynamicIndexing: return "SampledImageArrayDynamicIndexing";
    case Capability::StorageBufferArrayDynamicIndexing: return "StorageBufferArrayDynamicIndexing";
    case Capability::StorageImageArrayDynamicIndexing: return "StorageImageArrayDynamicIndexing";
    case Capability::ClipDistance: return "ClipDistance";
    case Capability::CullDistance: return "CullDistance";
    case Capability::SampleRateShading: return "SampleRateShading";
    case Capability::SampledRect: return "SampledRect";
    case Capability::InputAttachment: return "InputAttachment";
    case Capability::SparseResidency: return "SparseResidency";
    case Capability::MinLod: return "MinLod";
    case Capability::SampledCubeArray: return "SampledCubeArray";
    case Capability::ImageMSArray: return "ImageMSArray";
    case Capability::StorageImageExtendedFormats: return "StorageImageExtendedFormats";
    case Capability::ImageQuery: return "ImageQuery";
    case Capability::DerivativeControl: return "DerivativeControl";
    case Capability::InterpolationFunction: return "InterpolationFunction";
    case Capability::TransformFeedback: return "TransformFeedback";
    case Capability::StorageImageReadWithoutFormat: return "StorageImageReadWithoutFormat";
    case Capability::StorageImageWriteWithoutFormat: return "StorageImageWriteWithoutFormat";
    case Capability::SubgroupDispatch: return "SubgroupDispatch";
    case Capability::PipeStorage: return "PipeStorage";
    case Capability::DrawParameters: return "DrawParameters";
    case Capability::MultiView: return "MultiView";
    case Capability::VariablePointersStorageBuffer: return "VariablePointersStorageBuffer";
    case Capability::Float16ImageAMD: return "Float16ImageAMD";
    case Capability::ImageGatherBiasLodAMD: return "ImageGatherBiasLodAMD";
    case Capability::FragmentMaskAMD: return "FragmentMaskAMD";
    case Capability::StencilExportEXT: return "StencilExportEXT";
    case Capability::ImageReadWriteLodAMD: return "ImageReadWriteLodAMD";
    case Capability::ShaderClockKHR: return "ShaderClockKHR";
    case Capability::FragmentFullyCoveredEXT: return "FragmentFullyCoveredEXT";
    case Capability::MeshShadingNV: return "MeshShadingNV";
    case Capability::FragmentDensityEXT: return "FragmentDensityEXT";
    case Capability::ShaderNonUniform: return "ShaderNonUniform";
    case Capability::RuntimeDescriptorArray: return "RuntimeDescriptorArray";
    case Capability::StorageTexelBufferArrayDynamicIndexing: return "StorageTexelBufferArrayDynamicIndexing";
    case Capability::RayTracingNV: return "RayTracingNV";
    case Capability::PhysicalStorageBufferAddresses: return "PhysicalStorageBufferAddresses";
    case Capability::CooperativeMatrixNV: return "CooperativeMatrixNV";
    case Capability::FragmentShaderSampleInterlockEXT: return "FragmentShaderSampleInterlockEXT";
    case Capability::FragmentShaderShadingRateInterlockEXT: return "FragmentShaderShadingRateInterlockEXT";
    case Capability::ShaderSMBuiltinsNV: return "ShaderSMBuiltinsNV";
    case Capability::FragmentShaderPixelInterlockEXT: return "FragmentShaderPixelInterlockEXT";
    case Capability::DemoteToHelperInvocationEXT: return "DemoteToHelperInvocationEXT";
    case Capability::IntegerFunctions2INTEL: return "IntegerFunctions2INTEL";
    case Capability::TessellationPointSize: return "TessellationPointSize";
    case Capability::GeometryPointSize: return "GeometryPointSize";
    case Capability::ImageCubeArray: return "ImageCubeArray";
    case Capability::ImageRect: return "ImageRect";
    case Capability::GeometryStreams: return "GeometryStreams";
    case Capability::MultiViewport: return "MultiViewport";
    case Capability::VariablePointers: return "VariablePointers";
    case Capability::SampleMaskOverrideCoverageNV: return "SampleMaskOverrideCoverageNV";
    case Capability::GeometryShaderPassthroughNV: return "GeometryShaderPassthroughNV";
    case Capability::PerViewAttributesNV: return "PerViewAttributesNV";
    case Capability::InputAttachmentArrayDynamicIndexing: return "InputAttachmentArrayDynamicIndexing";
    case Capability::UniformBufferArrayNonUniformIndexing: return "UniformBufferArrayNonUniformIndexing";
    case Capability::SampledImageArrayNonUniformIndexing: return "SampledImageArrayNonUniformIndexing";
    case Capability::StorageBufferArrayNonUniformIndexing: return "StorageBufferArrayNonUniformIndexing";
    case Capability::StorageImageArrayNonUniformIndexing: return "StorageImageArrayNonUniformIndexing";
    case Capability::InputAttachmentArrayNonUniformIndexing: return "InputAttachmentArrayNonUniformIndexing";
    case Capability::UniformTexelBufferArrayNonUniformIndexing: return "UniformTexelBufferArrayNonUniformIndexing";
    case Capability::StorageTexelBufferArrayNonUniformIndexing: return "StorageTexelBufferArrayNonUniformIndexing";
    case Capability::ShaderViewportIndexLayerEXT: return "ShaderViewportIndexLayerEXT";
    case Capability::ShaderViewportMaskNV: return "ShaderViewportMaskNV";
    case Capability::ShaderStereoViewNV: return "ShaderStereoViewNV";
  }
  return "";
}

::llvm::Optional<Capability> symbolizeCapability(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Capability>>(str)
      .Case("Matrix", Capability::Matrix)
      .Case("Addresses", Capability::Addresses)
      .Case("Linkage", Capability::Linkage)
      .Case("Kernel", Capability::Kernel)
      .Case("Float16", Capability::Float16)
      .Case("Float64", Capability::Float64)
      .Case("Int64", Capability::Int64)
      .Case("Groups", Capability::Groups)
      .Case("Int16", Capability::Int16)
      .Case("Int8", Capability::Int8)
      .Case("Sampled1D", Capability::Sampled1D)
      .Case("SampledBuffer", Capability::SampledBuffer)
      .Case("GroupNonUniform", Capability::GroupNonUniform)
      .Case("ShaderLayer", Capability::ShaderLayer)
      .Case("ShaderViewportIndex", Capability::ShaderViewportIndex)
      .Case("SubgroupBallotKHR", Capability::SubgroupBallotKHR)
      .Case("SubgroupVoteKHR", Capability::SubgroupVoteKHR)
      .Case("StorageBuffer16BitAccess", Capability::StorageBuffer16BitAccess)
      .Case("StoragePushConstant16", Capability::StoragePushConstant16)
      .Case("StorageInputOutput16", Capability::StorageInputOutput16)
      .Case("DeviceGroup", Capability::DeviceGroup)
      .Case("AtomicStorageOps", Capability::AtomicStorageOps)
      .Case("SampleMaskPostDepthCoverage", Capability::SampleMaskPostDepthCoverage)
      .Case("StorageBuffer8BitAccess", Capability::StorageBuffer8BitAccess)
      .Case("StoragePushConstant8", Capability::StoragePushConstant8)
      .Case("DenormPreserve", Capability::DenormPreserve)
      .Case("DenormFlushToZero", Capability::DenormFlushToZero)
      .Case("SignedZeroInfNanPreserve", Capability::SignedZeroInfNanPreserve)
      .Case("RoundingModeRTE", Capability::RoundingModeRTE)
      .Case("RoundingModeRTZ", Capability::RoundingModeRTZ)
      .Case("ImageFootprintNV", Capability::ImageFootprintNV)
      .Case("FragmentBarycentricNV", Capability::FragmentBarycentricNV)
      .Case("ComputeDerivativeGroupQuadsNV", Capability::ComputeDerivativeGroupQuadsNV)
      .Case("GroupNonUniformPartitionedNV", Capability::GroupNonUniformPartitionedNV)
      .Case("VulkanMemoryModel", Capability::VulkanMemoryModel)
      .Case("VulkanMemoryModelDeviceScope", Capability::VulkanMemoryModelDeviceScope)
      .Case("ComputeDerivativeGroupLinearNV", Capability::ComputeDerivativeGroupLinearNV)
      .Case("SubgroupShuffleINTEL", Capability::SubgroupShuffleINTEL)
      .Case("SubgroupBufferBlockIOINTEL", Capability::SubgroupBufferBlockIOINTEL)
      .Case("SubgroupImageBlockIOINTEL", Capability::SubgroupImageBlockIOINTEL)
      .Case("SubgroupImageMediaBlockIOINTEL", Capability::SubgroupImageMediaBlockIOINTEL)
      .Case("SubgroupAvcMotionEstimationINTEL", Capability::SubgroupAvcMotionEstimationINTEL)
      .Case("SubgroupAvcMotionEstimationIntraINTEL", Capability::SubgroupAvcMotionEstimationIntraINTEL)
      .Case("SubgroupAvcMotionEstimationChromaINTEL", Capability::SubgroupAvcMotionEstimationChromaINTEL)
      .Case("Shader", Capability::Shader)
      .Case("Vector16", Capability::Vector16)
      .Case("Float16Buffer", Capability::Float16Buffer)
      .Case("Int64Atomics", Capability::Int64Atomics)
      .Case("ImageBasic", Capability::ImageBasic)
      .Case("Pipes", Capability::Pipes)
      .Case("DeviceEnqueue", Capability::DeviceEnqueue)
      .Case("LiteralSampler", Capability::LiteralSampler)
      .Case("GenericPointer", Capability::GenericPointer)
      .Case("Image1D", Capability::Image1D)
      .Case("ImageBuffer", Capability::ImageBuffer)
      .Case("NamedBarrier", Capability::NamedBarrier)
      .Case("GroupNonUniformVote", Capability::GroupNonUniformVote)
      .Case("GroupNonUniformArithmetic", Capability::GroupNonUniformArithmetic)
      .Case("GroupNonUniformBallot", Capability::GroupNonUniformBallot)
      .Case("GroupNonUniformShuffle", Capability::GroupNonUniformShuffle)
      .Case("GroupNonUniformShuffleRelative", Capability::GroupNonUniformShuffleRelative)
      .Case("GroupNonUniformClustered", Capability::GroupNonUniformClustered)
      .Case("GroupNonUniformQuad", Capability::GroupNonUniformQuad)
      .Case("StorageUniform16", Capability::StorageUniform16)
      .Case("UniformAndStorageBuffer8BitAccess", Capability::UniformAndStorageBuffer8BitAccess)
      .Case("UniformTexelBufferArrayDynamicIndexing", Capability::UniformTexelBufferArrayDynamicIndexing)
      .Case("Geometry", Capability::Geometry)
      .Case("Tessellation", Capability::Tessellation)
      .Case("ImageReadWrite", Capability::ImageReadWrite)
      .Case("ImageMipmap", Capability::ImageMipmap)
      .Case("AtomicStorage", Capability::AtomicStorage)
      .Case("ImageGatherExtended", Capability::ImageGatherExtended)
      .Case("StorageImageMultisample", Capability::StorageImageMultisample)
      .Case("UniformBufferArrayDynamicIndexing", Capability::UniformBufferArrayDynamicIndexing)
      .Case("SampledImageArrayDynamicIndexing", Capability::SampledImageArrayDynamicIndexing)
      .Case("StorageBufferArrayDynamicIndexing", Capability::StorageBufferArrayDynamicIndexing)
      .Case("StorageImageArrayDynamicIndexing", Capability::StorageImageArrayDynamicIndexing)
      .Case("ClipDistance", Capability::ClipDistance)
      .Case("CullDistance", Capability::CullDistance)
      .Case("SampleRateShading", Capability::SampleRateShading)
      .Case("SampledRect", Capability::SampledRect)
      .Case("InputAttachment", Capability::InputAttachment)
      .Case("SparseResidency", Capability::SparseResidency)
      .Case("MinLod", Capability::MinLod)
      .Case("SampledCubeArray", Capability::SampledCubeArray)
      .Case("ImageMSArray", Capability::ImageMSArray)
      .Case("StorageImageExtendedFormats", Capability::StorageImageExtendedFormats)
      .Case("ImageQuery", Capability::ImageQuery)
      .Case("DerivativeControl", Capability::DerivativeControl)
      .Case("InterpolationFunction", Capability::InterpolationFunction)
      .Case("TransformFeedback", Capability::TransformFeedback)
      .Case("StorageImageReadWithoutFormat", Capability::StorageImageReadWithoutFormat)
      .Case("StorageImageWriteWithoutFormat", Capability::StorageImageWriteWithoutFormat)
      .Case("SubgroupDispatch", Capability::SubgroupDispatch)
      .Case("PipeStorage", Capability::PipeStorage)
      .Case("DrawParameters", Capability::DrawParameters)
      .Case("MultiView", Capability::MultiView)
      .Case("VariablePointersStorageBuffer", Capability::VariablePointersStorageBuffer)
      .Case("Float16ImageAMD", Capability::Float16ImageAMD)
      .Case("ImageGatherBiasLodAMD", Capability::ImageGatherBiasLodAMD)
      .Case("FragmentMaskAMD", Capability::FragmentMaskAMD)
      .Case("StencilExportEXT", Capability::StencilExportEXT)
      .Case("ImageReadWriteLodAMD", Capability::ImageReadWriteLodAMD)
      .Case("ShaderClockKHR", Capability::ShaderClockKHR)
      .Case("FragmentFullyCoveredEXT", Capability::FragmentFullyCoveredEXT)
      .Case("MeshShadingNV", Capability::MeshShadingNV)
      .Case("FragmentDensityEXT", Capability::FragmentDensityEXT)
      .Case("ShaderNonUniform", Capability::ShaderNonUniform)
      .Case("RuntimeDescriptorArray", Capability::RuntimeDescriptorArray)
      .Case("StorageTexelBufferArrayDynamicIndexing", Capability::StorageTexelBufferArrayDynamicIndexing)
      .Case("RayTracingNV", Capability::RayTracingNV)
      .Case("PhysicalStorageBufferAddresses", Capability::PhysicalStorageBufferAddresses)
      .Case("CooperativeMatrixNV", Capability::CooperativeMatrixNV)
      .Case("FragmentShaderSampleInterlockEXT", Capability::FragmentShaderSampleInterlockEXT)
      .Case("FragmentShaderShadingRateInterlockEXT", Capability::FragmentShaderShadingRateInterlockEXT)
      .Case("ShaderSMBuiltinsNV", Capability::ShaderSMBuiltinsNV)
      .Case("FragmentShaderPixelInterlockEXT", Capability::FragmentShaderPixelInterlockEXT)
      .Case("DemoteToHelperInvocationEXT", Capability::DemoteToHelperInvocationEXT)
      .Case("IntegerFunctions2INTEL", Capability::IntegerFunctions2INTEL)
      .Case("TessellationPointSize", Capability::TessellationPointSize)
      .Case("GeometryPointSize", Capability::GeometryPointSize)
      .Case("ImageCubeArray", Capability::ImageCubeArray)
      .Case("ImageRect", Capability::ImageRect)
      .Case("GeometryStreams", Capability::GeometryStreams)
      .Case("MultiViewport", Capability::MultiViewport)
      .Case("VariablePointers", Capability::VariablePointers)
      .Case("SampleMaskOverrideCoverageNV", Capability::SampleMaskOverrideCoverageNV)
      .Case("GeometryShaderPassthroughNV", Capability::GeometryShaderPassthroughNV)
      .Case("PerViewAttributesNV", Capability::PerViewAttributesNV)
      .Case("InputAttachmentArrayDynamicIndexing", Capability::InputAttachmentArrayDynamicIndexing)
      .Case("UniformBufferArrayNonUniformIndexing", Capability::UniformBufferArrayNonUniformIndexing)
      .Case("SampledImageArrayNonUniformIndexing", Capability::SampledImageArrayNonUniformIndexing)
      .Case("StorageBufferArrayNonUniformIndexing", Capability::StorageBufferArrayNonUniformIndexing)
      .Case("StorageImageArrayNonUniformIndexing", Capability::StorageImageArrayNonUniformIndexing)
      .Case("InputAttachmentArrayNonUniformIndexing", Capability::InputAttachmentArrayNonUniformIndexing)
      .Case("UniformTexelBufferArrayNonUniformIndexing", Capability::UniformTexelBufferArrayNonUniformIndexing)
      .Case("StorageTexelBufferArrayNonUniformIndexing", Capability::StorageTexelBufferArrayNonUniformIndexing)
      .Case("ShaderViewportIndexLayerEXT", Capability::ShaderViewportIndexLayerEXT)
      .Case("ShaderViewportMaskNV", Capability::ShaderViewportMaskNV)
      .Case("ShaderStereoViewNV", Capability::ShaderStereoViewNV)
      .Default(::llvm::None);
}
::llvm::Optional<Capability> symbolizeCapability(uint32_t value) {
  switch (value) {
  case 0: return Capability::Matrix;
  case 4: return Capability::Addresses;
  case 5: return Capability::Linkage;
  case 6: return Capability::Kernel;
  case 9: return Capability::Float16;
  case 10: return Capability::Float64;
  case 11: return Capability::Int64;
  case 18: return Capability::Groups;
  case 22: return Capability::Int16;
  case 39: return Capability::Int8;
  case 43: return Capability::Sampled1D;
  case 46: return Capability::SampledBuffer;
  case 61: return Capability::GroupNonUniform;
  case 69: return Capability::ShaderLayer;
  case 70: return Capability::ShaderViewportIndex;
  case 4423: return Capability::SubgroupBallotKHR;
  case 4431: return Capability::SubgroupVoteKHR;
  case 4433: return Capability::StorageBuffer16BitAccess;
  case 4435: return Capability::StoragePushConstant16;
  case 4436: return Capability::StorageInputOutput16;
  case 4437: return Capability::DeviceGroup;
  case 4445: return Capability::AtomicStorageOps;
  case 4447: return Capability::SampleMaskPostDepthCoverage;
  case 4448: return Capability::StorageBuffer8BitAccess;
  case 4450: return Capability::StoragePushConstant8;
  case 4464: return Capability::DenormPreserve;
  case 4465: return Capability::DenormFlushToZero;
  case 4466: return Capability::SignedZeroInfNanPreserve;
  case 4467: return Capability::RoundingModeRTE;
  case 4468: return Capability::RoundingModeRTZ;
  case 5282: return Capability::ImageFootprintNV;
  case 5284: return Capability::FragmentBarycentricNV;
  case 5288: return Capability::ComputeDerivativeGroupQuadsNV;
  case 5297: return Capability::GroupNonUniformPartitionedNV;
  case 5345: return Capability::VulkanMemoryModel;
  case 5346: return Capability::VulkanMemoryModelDeviceScope;
  case 5350: return Capability::ComputeDerivativeGroupLinearNV;
  case 5568: return Capability::SubgroupShuffleINTEL;
  case 5569: return Capability::SubgroupBufferBlockIOINTEL;
  case 5570: return Capability::SubgroupImageBlockIOINTEL;
  case 5579: return Capability::SubgroupImageMediaBlockIOINTEL;
  case 5696: return Capability::SubgroupAvcMotionEstimationINTEL;
  case 5697: return Capability::SubgroupAvcMotionEstimationIntraINTEL;
  case 5698: return Capability::SubgroupAvcMotionEstimationChromaINTEL;
  case 1: return Capability::Shader;
  case 7: return Capability::Vector16;
  case 8: return Capability::Float16Buffer;
  case 12: return Capability::Int64Atomics;
  case 13: return Capability::ImageBasic;
  case 17: return Capability::Pipes;
  case 19: return Capability::DeviceEnqueue;
  case 20: return Capability::LiteralSampler;
  case 38: return Capability::GenericPointer;
  case 44: return Capability::Image1D;
  case 47: return Capability::ImageBuffer;
  case 59: return Capability::NamedBarrier;
  case 62: return Capability::GroupNonUniformVote;
  case 63: return Capability::GroupNonUniformArithmetic;
  case 64: return Capability::GroupNonUniformBallot;
  case 65: return Capability::GroupNonUniformShuffle;
  case 66: return Capability::GroupNonUniformShuffleRelative;
  case 67: return Capability::GroupNonUniformClustered;
  case 68: return Capability::GroupNonUniformQuad;
  case 4434: return Capability::StorageUniform16;
  case 4449: return Capability::UniformAndStorageBuffer8BitAccess;
  case 5304: return Capability::UniformTexelBufferArrayDynamicIndexing;
  case 2: return Capability::Geometry;
  case 3: return Capability::Tessellation;
  case 14: return Capability::ImageReadWrite;
  case 15: return Capability::ImageMipmap;
  case 21: return Capability::AtomicStorage;
  case 25: return Capability::ImageGatherExtended;
  case 27: return Capability::StorageImageMultisample;
  case 28: return Capability::UniformBufferArrayDynamicIndexing;
  case 29: return Capability::SampledImageArrayDynamicIndexing;
  case 30: return Capability::StorageBufferArrayDynamicIndexing;
  case 31: return Capability::StorageImageArrayDynamicIndexing;
  case 32: return Capability::ClipDistance;
  case 33: return Capability::CullDistance;
  case 35: return Capability::SampleRateShading;
  case 37: return Capability::SampledRect;
  case 40: return Capability::InputAttachment;
  case 41: return Capability::SparseResidency;
  case 42: return Capability::MinLod;
  case 45: return Capability::SampledCubeArray;
  case 48: return Capability::ImageMSArray;
  case 49: return Capability::StorageImageExtendedFormats;
  case 50: return Capability::ImageQuery;
  case 51: return Capability::DerivativeControl;
  case 52: return Capability::InterpolationFunction;
  case 53: return Capability::TransformFeedback;
  case 55: return Capability::StorageImageReadWithoutFormat;
  case 56: return Capability::StorageImageWriteWithoutFormat;
  case 58: return Capability::SubgroupDispatch;
  case 60: return Capability::PipeStorage;
  case 4427: return Capability::DrawParameters;
  case 4439: return Capability::MultiView;
  case 4441: return Capability::VariablePointersStorageBuffer;
  case 5008: return Capability::Float16ImageAMD;
  case 5009: return Capability::ImageGatherBiasLodAMD;
  case 5010: return Capability::FragmentMaskAMD;
  case 5013: return Capability::StencilExportEXT;
  case 5015: return Capability::ImageReadWriteLodAMD;
  case 5055: return Capability::ShaderClockKHR;
  case 5265: return Capability::FragmentFullyCoveredEXT;
  case 5266: return Capability::MeshShadingNV;
  case 5291: return Capability::FragmentDensityEXT;
  case 5301: return Capability::ShaderNonUniform;
  case 5302: return Capability::RuntimeDescriptorArray;
  case 5305: return Capability::StorageTexelBufferArrayDynamicIndexing;
  case 5340: return Capability::RayTracingNV;
  case 5347: return Capability::PhysicalStorageBufferAddresses;
  case 5357: return Capability::CooperativeMatrixNV;
  case 5363: return Capability::FragmentShaderSampleInterlockEXT;
  case 5372: return Capability::FragmentShaderShadingRateInterlockEXT;
  case 5373: return Capability::ShaderSMBuiltinsNV;
  case 5378: return Capability::FragmentShaderPixelInterlockEXT;
  case 5379: return Capability::DemoteToHelperInvocationEXT;
  case 5584: return Capability::IntegerFunctions2INTEL;
  case 23: return Capability::TessellationPointSize;
  case 24: return Capability::GeometryPointSize;
  case 34: return Capability::ImageCubeArray;
  case 36: return Capability::ImageRect;
  case 54: return Capability::GeometryStreams;
  case 57: return Capability::MultiViewport;
  case 4442: return Capability::VariablePointers;
  case 5249: return Capability::SampleMaskOverrideCoverageNV;
  case 5251: return Capability::GeometryShaderPassthroughNV;
  case 5260: return Capability::PerViewAttributesNV;
  case 5303: return Capability::InputAttachmentArrayDynamicIndexing;
  case 5306: return Capability::UniformBufferArrayNonUniformIndexing;
  case 5307: return Capability::SampledImageArrayNonUniformIndexing;
  case 5308: return Capability::StorageBufferArrayNonUniformIndexing;
  case 5309: return Capability::StorageImageArrayNonUniformIndexing;
  case 5310: return Capability::InputAttachmentArrayNonUniformIndexing;
  case 5311: return Capability::UniformTexelBufferArrayNonUniformIndexing;
  case 5312: return Capability::StorageTexelBufferArrayNonUniformIndexing;
  case 5254: return Capability::ShaderViewportIndexLayerEXT;
  case 5255: return Capability::ShaderViewportMaskNV;
  case 5259: return Capability::ShaderStereoViewNV;
  default: return ::llvm::None;
  }
}

bool CapabilityAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 18)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 22)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 39)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 43)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 46)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 61)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 69)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 70)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4423)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4431)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4433)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4435)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4436)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4437)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4445)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4447)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4448)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4450)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4464)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4465)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4466)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4467)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4468)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5282)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5284)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5288)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5297)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5345)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5346)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5350)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5568)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5569)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5570)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5579)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5696)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5697)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5698)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 38)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 44)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 47)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 59)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 62)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 63)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 64)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 65)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 66)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 67)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 68)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4434)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4449)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5304)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 21)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 25)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 27)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 28)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 29)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 30)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 31)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 32)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 33)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 35)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 37)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 40)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 41)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 42)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 45)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 48)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 49)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 50)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 51)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 52)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 53)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 55)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 56)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 58)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 60)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4427)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4439)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4441)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5008)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5009)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5010)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5013)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5015)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5055)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5265)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5266)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5291)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5301)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5302)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5305)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5340)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5347)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5357)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5363)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5372)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5373)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5378)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5379)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5584)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 23)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 24)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 34)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 36)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 54)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 57)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4442)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5249)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5251)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5260)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5303)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5306)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5307)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5308)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5309)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5310)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5311)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5312)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5254)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5255)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5259)));
}
CapabilityAttr CapabilityAttr::get(::mlir::MLIRContext *context, Capability val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<CapabilityAttr>();
}
Capability CapabilityAttr::getValue() const {
  return static_cast<Capability>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyDecoration(Decoration val) {
  switch (val) {
    case Decoration::RelaxedPrecision: return "RelaxedPrecision";
    case Decoration::SpecId: return "SpecId";
    case Decoration::Block: return "Block";
    case Decoration::BufferBlock: return "BufferBlock";
    case Decoration::RowMajor: return "RowMajor";
    case Decoration::ColMajor: return "ColMajor";
    case Decoration::ArrayStride: return "ArrayStride";
    case Decoration::MatrixStride: return "MatrixStride";
    case Decoration::GLSLShared: return "GLSLShared";
    case Decoration::GLSLPacked: return "GLSLPacked";
    case Decoration::CPacked: return "CPacked";
    case Decoration::BuiltIn: return "BuiltIn";
    case Decoration::NoPerspective: return "NoPerspective";
    case Decoration::Flat: return "Flat";
    case Decoration::Patch: return "Patch";
    case Decoration::Centroid: return "Centroid";
    case Decoration::Sample: return "Sample";
    case Decoration::Invariant: return "Invariant";
    case Decoration::Restrict: return "Restrict";
    case Decoration::Aliased: return "Aliased";
    case Decoration::Volatile: return "Volatile";
    case Decoration::Constant: return "Constant";
    case Decoration::Coherent: return "Coherent";
    case Decoration::NonWritable: return "NonWritable";
    case Decoration::NonReadable: return "NonReadable";
    case Decoration::Uniform: return "Uniform";
    case Decoration::UniformId: return "UniformId";
    case Decoration::SaturatedConversion: return "SaturatedConversion";
    case Decoration::Stream: return "Stream";
    case Decoration::Location: return "Location";
    case Decoration::Component: return "Component";
    case Decoration::Index: return "Index";
    case Decoration::Binding: return "Binding";
    case Decoration::DescriptorSet: return "DescriptorSet";
    case Decoration::Offset: return "Offset";
    case Decoration::XfbBuffer: return "XfbBuffer";
    case Decoration::XfbStride: return "XfbStride";
    case Decoration::FuncParamAttr: return "FuncParamAttr";
    case Decoration::FPRoundingMode: return "FPRoundingMode";
    case Decoration::FPFastMathMode: return "FPFastMathMode";
    case Decoration::LinkageAttributes: return "LinkageAttributes";
    case Decoration::NoContraction: return "NoContraction";
    case Decoration::InputAttachmentIndex: return "InputAttachmentIndex";
    case Decoration::Alignment: return "Alignment";
    case Decoration::MaxByteOffset: return "MaxByteOffset";
    case Decoration::AlignmentId: return "AlignmentId";
    case Decoration::MaxByteOffsetId: return "MaxByteOffsetId";
    case Decoration::NoSignedWrap: return "NoSignedWrap";
    case Decoration::NoUnsignedWrap: return "NoUnsignedWrap";
    case Decoration::ExplicitInterpAMD: return "ExplicitInterpAMD";
    case Decoration::OverrideCoverageNV: return "OverrideCoverageNV";
    case Decoration::PassthroughNV: return "PassthroughNV";
    case Decoration::ViewportRelativeNV: return "ViewportRelativeNV";
    case Decoration::SecondaryViewportRelativeNV: return "SecondaryViewportRelativeNV";
    case Decoration::PerPrimitiveNV: return "PerPrimitiveNV";
    case Decoration::PerViewNV: return "PerViewNV";
    case Decoration::PerTaskNV: return "PerTaskNV";
    case Decoration::PerVertexNV: return "PerVertexNV";
    case Decoration::NonUniform: return "NonUniform";
    case Decoration::RestrictPointer: return "RestrictPointer";
    case Decoration::AliasedPointer: return "AliasedPointer";
    case Decoration::CounterBuffer: return "CounterBuffer";
    case Decoration::UserSemantic: return "UserSemantic";
    case Decoration::UserTypeGOOGLE: return "UserTypeGOOGLE";
  }
  return "";
}

::llvm::Optional<Decoration> symbolizeDecoration(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Decoration>>(str)
      .Case("RelaxedPrecision", Decoration::RelaxedPrecision)
      .Case("SpecId", Decoration::SpecId)
      .Case("Block", Decoration::Block)
      .Case("BufferBlock", Decoration::BufferBlock)
      .Case("RowMajor", Decoration::RowMajor)
      .Case("ColMajor", Decoration::ColMajor)
      .Case("ArrayStride", Decoration::ArrayStride)
      .Case("MatrixStride", Decoration::MatrixStride)
      .Case("GLSLShared", Decoration::GLSLShared)
      .Case("GLSLPacked", Decoration::GLSLPacked)
      .Case("CPacked", Decoration::CPacked)
      .Case("BuiltIn", Decoration::BuiltIn)
      .Case("NoPerspective", Decoration::NoPerspective)
      .Case("Flat", Decoration::Flat)
      .Case("Patch", Decoration::Patch)
      .Case("Centroid", Decoration::Centroid)
      .Case("Sample", Decoration::Sample)
      .Case("Invariant", Decoration::Invariant)
      .Case("Restrict", Decoration::Restrict)
      .Case("Aliased", Decoration::Aliased)
      .Case("Volatile", Decoration::Volatile)
      .Case("Constant", Decoration::Constant)
      .Case("Coherent", Decoration::Coherent)
      .Case("NonWritable", Decoration::NonWritable)
      .Case("NonReadable", Decoration::NonReadable)
      .Case("Uniform", Decoration::Uniform)
      .Case("UniformId", Decoration::UniformId)
      .Case("SaturatedConversion", Decoration::SaturatedConversion)
      .Case("Stream", Decoration::Stream)
      .Case("Location", Decoration::Location)
      .Case("Component", Decoration::Component)
      .Case("Index", Decoration::Index)
      .Case("Binding", Decoration::Binding)
      .Case("DescriptorSet", Decoration::DescriptorSet)
      .Case("Offset", Decoration::Offset)
      .Case("XfbBuffer", Decoration::XfbBuffer)
      .Case("XfbStride", Decoration::XfbStride)
      .Case("FuncParamAttr", Decoration::FuncParamAttr)
      .Case("FPRoundingMode", Decoration::FPRoundingMode)
      .Case("FPFastMathMode", Decoration::FPFastMathMode)
      .Case("LinkageAttributes", Decoration::LinkageAttributes)
      .Case("NoContraction", Decoration::NoContraction)
      .Case("InputAttachmentIndex", Decoration::InputAttachmentIndex)
      .Case("Alignment", Decoration::Alignment)
      .Case("MaxByteOffset", Decoration::MaxByteOffset)
      .Case("AlignmentId", Decoration::AlignmentId)
      .Case("MaxByteOffsetId", Decoration::MaxByteOffsetId)
      .Case("NoSignedWrap", Decoration::NoSignedWrap)
      .Case("NoUnsignedWrap", Decoration::NoUnsignedWrap)
      .Case("ExplicitInterpAMD", Decoration::ExplicitInterpAMD)
      .Case("OverrideCoverageNV", Decoration::OverrideCoverageNV)
      .Case("PassthroughNV", Decoration::PassthroughNV)
      .Case("ViewportRelativeNV", Decoration::ViewportRelativeNV)
      .Case("SecondaryViewportRelativeNV", Decoration::SecondaryViewportRelativeNV)
      .Case("PerPrimitiveNV", Decoration::PerPrimitiveNV)
      .Case("PerViewNV", Decoration::PerViewNV)
      .Case("PerTaskNV", Decoration::PerTaskNV)
      .Case("PerVertexNV", Decoration::PerVertexNV)
      .Case("NonUniform", Decoration::NonUniform)
      .Case("RestrictPointer", Decoration::RestrictPointer)
      .Case("AliasedPointer", Decoration::AliasedPointer)
      .Case("CounterBuffer", Decoration::CounterBuffer)
      .Case("UserSemantic", Decoration::UserSemantic)
      .Case("UserTypeGOOGLE", Decoration::UserTypeGOOGLE)
      .Default(::llvm::None);
}
::llvm::Optional<Decoration> symbolizeDecoration(uint32_t value) {
  switch (value) {
  case 0: return Decoration::RelaxedPrecision;
  case 1: return Decoration::SpecId;
  case 2: return Decoration::Block;
  case 3: return Decoration::BufferBlock;
  case 4: return Decoration::RowMajor;
  case 5: return Decoration::ColMajor;
  case 6: return Decoration::ArrayStride;
  case 7: return Decoration::MatrixStride;
  case 8: return Decoration::GLSLShared;
  case 9: return Decoration::GLSLPacked;
  case 10: return Decoration::CPacked;
  case 11: return Decoration::BuiltIn;
  case 13: return Decoration::NoPerspective;
  case 14: return Decoration::Flat;
  case 15: return Decoration::Patch;
  case 16: return Decoration::Centroid;
  case 17: return Decoration::Sample;
  case 18: return Decoration::Invariant;
  case 19: return Decoration::Restrict;
  case 20: return Decoration::Aliased;
  case 21: return Decoration::Volatile;
  case 22: return Decoration::Constant;
  case 23: return Decoration::Coherent;
  case 24: return Decoration::NonWritable;
  case 25: return Decoration::NonReadable;
  case 26: return Decoration::Uniform;
  case 27: return Decoration::UniformId;
  case 28: return Decoration::SaturatedConversion;
  case 29: return Decoration::Stream;
  case 30: return Decoration::Location;
  case 31: return Decoration::Component;
  case 32: return Decoration::Index;
  case 33: return Decoration::Binding;
  case 34: return Decoration::DescriptorSet;
  case 35: return Decoration::Offset;
  case 36: return Decoration::XfbBuffer;
  case 37: return Decoration::XfbStride;
  case 38: return Decoration::FuncParamAttr;
  case 39: return Decoration::FPRoundingMode;
  case 40: return Decoration::FPFastMathMode;
  case 41: return Decoration::LinkageAttributes;
  case 42: return Decoration::NoContraction;
  case 43: return Decoration::InputAttachmentIndex;
  case 44: return Decoration::Alignment;
  case 45: return Decoration::MaxByteOffset;
  case 46: return Decoration::AlignmentId;
  case 47: return Decoration::MaxByteOffsetId;
  case 4469: return Decoration::NoSignedWrap;
  case 4470: return Decoration::NoUnsignedWrap;
  case 4999: return Decoration::ExplicitInterpAMD;
  case 5248: return Decoration::OverrideCoverageNV;
  case 5250: return Decoration::PassthroughNV;
  case 5252: return Decoration::ViewportRelativeNV;
  case 5256: return Decoration::SecondaryViewportRelativeNV;
  case 5271: return Decoration::PerPrimitiveNV;
  case 5272: return Decoration::PerViewNV;
  case 5273: return Decoration::PerTaskNV;
  case 5285: return Decoration::PerVertexNV;
  case 5300: return Decoration::NonUniform;
  case 5355: return Decoration::RestrictPointer;
  case 5356: return Decoration::AliasedPointer;
  case 5634: return Decoration::CounterBuffer;
  case 5635: return Decoration::UserSemantic;
  case 5636: return Decoration::UserTypeGOOGLE;
  default: return ::llvm::None;
  }
}

bool DecorationAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 18)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 21)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 22)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 23)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 24)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 25)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 26)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 27)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 28)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 29)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 30)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 31)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 32)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 33)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 34)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 35)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 36)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 37)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 38)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 39)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 40)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 41)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 42)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 43)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 44)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 45)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 46)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 47)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4469)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4470)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4999)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5248)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5250)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5252)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5256)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5271)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5272)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5273)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5285)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5300)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5355)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5356)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5634)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5635)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5636)));
}
DecorationAttr DecorationAttr::get(::mlir::MLIRContext *context, Decoration val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<DecorationAttr>();
}
Decoration DecorationAttr::getValue() const {
  return static_cast<Decoration>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyImageDepthInfo(ImageDepthInfo val) {
  switch (val) {
    case ImageDepthInfo::NoDepth: return "NoDepth";
    case ImageDepthInfo::IsDepth: return "IsDepth";
    case ImageDepthInfo::DepthUnknown: return "DepthUnknown";
  }
  return "";
}

::llvm::Optional<ImageDepthInfo> symbolizeImageDepthInfo(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ImageDepthInfo>>(str)
      .Case("NoDepth", ImageDepthInfo::NoDepth)
      .Case("IsDepth", ImageDepthInfo::IsDepth)
      .Case("DepthUnknown", ImageDepthInfo::DepthUnknown)
      .Default(::llvm::None);
}
::llvm::Optional<ImageDepthInfo> symbolizeImageDepthInfo(uint32_t value) {
  switch (value) {
  case 0: return ImageDepthInfo::NoDepth;
  case 1: return ImageDepthInfo::IsDepth;
  case 2: return ImageDepthInfo::DepthUnknown;
  default: return ::llvm::None;
  }
}

bool ImageDepthInfoAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
ImageDepthInfoAttr ImageDepthInfoAttr::get(::mlir::MLIRContext *context, ImageDepthInfo val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ImageDepthInfoAttr>();
}
ImageDepthInfo ImageDepthInfoAttr::getValue() const {
  return static_cast<ImageDepthInfo>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyDeviceType(DeviceType val) {
  switch (val) {
    case DeviceType::Other: return "Other";
    case DeviceType::IntegratedGPU: return "IntegratedGPU";
    case DeviceType::DiscreteGPU: return "DiscreteGPU";
    case DeviceType::CPU: return "CPU";
    case DeviceType::Unknown: return "Unknown";
  }
  return "";
}

::llvm::Optional<DeviceType> symbolizeDeviceType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<DeviceType>>(str)
      .Case("Other", DeviceType::Other)
      .Case("IntegratedGPU", DeviceType::IntegratedGPU)
      .Case("DiscreteGPU", DeviceType::DiscreteGPU)
      .Case("CPU", DeviceType::CPU)
      .Case("Unknown", DeviceType::Unknown)
      .Default(::llvm::None);
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyDim(Dim val) {
  switch (val) {
    case Dim::Dim1D: return "Dim1D";
    case Dim::Dim2D: return "Dim2D";
    case Dim::Dim3D: return "Dim3D";
    case Dim::Cube: return "Cube";
    case Dim::Rect: return "Rect";
    case Dim::Buffer: return "Buffer";
    case Dim::SubpassData: return "SubpassData";
  }
  return "";
}

::llvm::Optional<Dim> symbolizeDim(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Dim>>(str)
      .Case("Dim1D", Dim::Dim1D)
      .Case("Dim2D", Dim::Dim2D)
      .Case("Dim3D", Dim::Dim3D)
      .Case("Cube", Dim::Cube)
      .Case("Rect", Dim::Rect)
      .Case("Buffer", Dim::Buffer)
      .Case("SubpassData", Dim::SubpassData)
      .Default(::llvm::None);
}
::llvm::Optional<Dim> symbolizeDim(uint32_t value) {
  switch (value) {
  case 0: return Dim::Dim1D;
  case 1: return Dim::Dim2D;
  case 2: return Dim::Dim3D;
  case 3: return Dim::Cube;
  case 4: return Dim::Rect;
  case 5: return Dim::Buffer;
  case 6: return Dim::SubpassData;
  default: return ::llvm::None;
  }
}

bool DimAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)));
}
DimAttr DimAttr::get(::mlir::MLIRContext *context, Dim val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<DimAttr>();
}
Dim DimAttr::getValue() const {
  return static_cast<Dim>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyExecutionMode(ExecutionMode val) {
  switch (val) {
    case ExecutionMode::Invocations: return "Invocations";
    case ExecutionMode::SpacingEqual: return "SpacingEqual";
    case ExecutionMode::SpacingFractionalEven: return "SpacingFractionalEven";
    case ExecutionMode::SpacingFractionalOdd: return "SpacingFractionalOdd";
    case ExecutionMode::VertexOrderCw: return "VertexOrderCw";
    case ExecutionMode::VertexOrderCcw: return "VertexOrderCcw";
    case ExecutionMode::PixelCenterInteger: return "PixelCenterInteger";
    case ExecutionMode::OriginUpperLeft: return "OriginUpperLeft";
    case ExecutionMode::OriginLowerLeft: return "OriginLowerLeft";
    case ExecutionMode::EarlyFragmentTests: return "EarlyFragmentTests";
    case ExecutionMode::PointMode: return "PointMode";
    case ExecutionMode::Xfb: return "Xfb";
    case ExecutionMode::DepthReplacing: return "DepthReplacing";
    case ExecutionMode::DepthGreater: return "DepthGreater";
    case ExecutionMode::DepthLess: return "DepthLess";
    case ExecutionMode::DepthUnchanged: return "DepthUnchanged";
    case ExecutionMode::LocalSize: return "LocalSize";
    case ExecutionMode::LocalSizeHint: return "LocalSizeHint";
    case ExecutionMode::InputPoints: return "InputPoints";
    case ExecutionMode::InputLines: return "InputLines";
    case ExecutionMode::InputLinesAdjacency: return "InputLinesAdjacency";
    case ExecutionMode::Triangles: return "Triangles";
    case ExecutionMode::InputTrianglesAdjacency: return "InputTrianglesAdjacency";
    case ExecutionMode::Quads: return "Quads";
    case ExecutionMode::Isolines: return "Isolines";
    case ExecutionMode::OutputVertices: return "OutputVertices";
    case ExecutionMode::OutputPoints: return "OutputPoints";
    case ExecutionMode::OutputLineStrip: return "OutputLineStrip";
    case ExecutionMode::OutputTriangleStrip: return "OutputTriangleStrip";
    case ExecutionMode::VecTypeHint: return "VecTypeHint";
    case ExecutionMode::ContractionOff: return "ContractionOff";
    case ExecutionMode::Initializer: return "Initializer";
    case ExecutionMode::Finalizer: return "Finalizer";
    case ExecutionMode::SubgroupSize: return "SubgroupSize";
    case ExecutionMode::SubgroupsPerWorkgroup: return "SubgroupsPerWorkgroup";
    case ExecutionMode::SubgroupsPerWorkgroupId: return "SubgroupsPerWorkgroupId";
    case ExecutionMode::LocalSizeId: return "LocalSizeId";
    case ExecutionMode::LocalSizeHintId: return "LocalSizeHintId";
    case ExecutionMode::PostDepthCoverage: return "PostDepthCoverage";
    case ExecutionMode::DenormPreserve: return "DenormPreserve";
    case ExecutionMode::DenormFlushToZero: return "DenormFlushToZero";
    case ExecutionMode::SignedZeroInfNanPreserve: return "SignedZeroInfNanPreserve";
    case ExecutionMode::RoundingModeRTE: return "RoundingModeRTE";
    case ExecutionMode::RoundingModeRTZ: return "RoundingModeRTZ";
    case ExecutionMode::StencilRefReplacingEXT: return "StencilRefReplacingEXT";
    case ExecutionMode::OutputLinesNV: return "OutputLinesNV";
    case ExecutionMode::OutputPrimitivesNV: return "OutputPrimitivesNV";
    case ExecutionMode::DerivativeGroupQuadsNV: return "DerivativeGroupQuadsNV";
    case ExecutionMode::DerivativeGroupLinearNV: return "DerivativeGroupLinearNV";
    case ExecutionMode::OutputTrianglesNV: return "OutputTrianglesNV";
    case ExecutionMode::PixelInterlockOrderedEXT: return "PixelInterlockOrderedEXT";
    case ExecutionMode::PixelInterlockUnorderedEXT: return "PixelInterlockUnorderedEXT";
    case ExecutionMode::SampleInterlockOrderedEXT: return "SampleInterlockOrderedEXT";
    case ExecutionMode::SampleInterlockUnorderedEXT: return "SampleInterlockUnorderedEXT";
    case ExecutionMode::ShadingRateInterlockOrderedEXT: return "ShadingRateInterlockOrderedEXT";
    case ExecutionMode::ShadingRateInterlockUnorderedEXT: return "ShadingRateInterlockUnorderedEXT";
  }
  return "";
}

::llvm::Optional<ExecutionMode> symbolizeExecutionMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ExecutionMode>>(str)
      .Case("Invocations", ExecutionMode::Invocations)
      .Case("SpacingEqual", ExecutionMode::SpacingEqual)
      .Case("SpacingFractionalEven", ExecutionMode::SpacingFractionalEven)
      .Case("SpacingFractionalOdd", ExecutionMode::SpacingFractionalOdd)
      .Case("VertexOrderCw", ExecutionMode::VertexOrderCw)
      .Case("VertexOrderCcw", ExecutionMode::VertexOrderCcw)
      .Case("PixelCenterInteger", ExecutionMode::PixelCenterInteger)
      .Case("OriginUpperLeft", ExecutionMode::OriginUpperLeft)
      .Case("OriginLowerLeft", ExecutionMode::OriginLowerLeft)
      .Case("EarlyFragmentTests", ExecutionMode::EarlyFragmentTests)
      .Case("PointMode", ExecutionMode::PointMode)
      .Case("Xfb", ExecutionMode::Xfb)
      .Case("DepthReplacing", ExecutionMode::DepthReplacing)
      .Case("DepthGreater", ExecutionMode::DepthGreater)
      .Case("DepthLess", ExecutionMode::DepthLess)
      .Case("DepthUnchanged", ExecutionMode::DepthUnchanged)
      .Case("LocalSize", ExecutionMode::LocalSize)
      .Case("LocalSizeHint", ExecutionMode::LocalSizeHint)
      .Case("InputPoints", ExecutionMode::InputPoints)
      .Case("InputLines", ExecutionMode::InputLines)
      .Case("InputLinesAdjacency", ExecutionMode::InputLinesAdjacency)
      .Case("Triangles", ExecutionMode::Triangles)
      .Case("InputTrianglesAdjacency", ExecutionMode::InputTrianglesAdjacency)
      .Case("Quads", ExecutionMode::Quads)
      .Case("Isolines", ExecutionMode::Isolines)
      .Case("OutputVertices", ExecutionMode::OutputVertices)
      .Case("OutputPoints", ExecutionMode::OutputPoints)
      .Case("OutputLineStrip", ExecutionMode::OutputLineStrip)
      .Case("OutputTriangleStrip", ExecutionMode::OutputTriangleStrip)
      .Case("VecTypeHint", ExecutionMode::VecTypeHint)
      .Case("ContractionOff", ExecutionMode::ContractionOff)
      .Case("Initializer", ExecutionMode::Initializer)
      .Case("Finalizer", ExecutionMode::Finalizer)
      .Case("SubgroupSize", ExecutionMode::SubgroupSize)
      .Case("SubgroupsPerWorkgroup", ExecutionMode::SubgroupsPerWorkgroup)
      .Case("SubgroupsPerWorkgroupId", ExecutionMode::SubgroupsPerWorkgroupId)
      .Case("LocalSizeId", ExecutionMode::LocalSizeId)
      .Case("LocalSizeHintId", ExecutionMode::LocalSizeHintId)
      .Case("PostDepthCoverage", ExecutionMode::PostDepthCoverage)
      .Case("DenormPreserve", ExecutionMode::DenormPreserve)
      .Case("DenormFlushToZero", ExecutionMode::DenormFlushToZero)
      .Case("SignedZeroInfNanPreserve", ExecutionMode::SignedZeroInfNanPreserve)
      .Case("RoundingModeRTE", ExecutionMode::RoundingModeRTE)
      .Case("RoundingModeRTZ", ExecutionMode::RoundingModeRTZ)
      .Case("StencilRefReplacingEXT", ExecutionMode::StencilRefReplacingEXT)
      .Case("OutputLinesNV", ExecutionMode::OutputLinesNV)
      .Case("OutputPrimitivesNV", ExecutionMode::OutputPrimitivesNV)
      .Case("DerivativeGroupQuadsNV", ExecutionMode::DerivativeGroupQuadsNV)
      .Case("DerivativeGroupLinearNV", ExecutionMode::DerivativeGroupLinearNV)
      .Case("OutputTrianglesNV", ExecutionMode::OutputTrianglesNV)
      .Case("PixelInterlockOrderedEXT", ExecutionMode::PixelInterlockOrderedEXT)
      .Case("PixelInterlockUnorderedEXT", ExecutionMode::PixelInterlockUnorderedEXT)
      .Case("SampleInterlockOrderedEXT", ExecutionMode::SampleInterlockOrderedEXT)
      .Case("SampleInterlockUnorderedEXT", ExecutionMode::SampleInterlockUnorderedEXT)
      .Case("ShadingRateInterlockOrderedEXT", ExecutionMode::ShadingRateInterlockOrderedEXT)
      .Case("ShadingRateInterlockUnorderedEXT", ExecutionMode::ShadingRateInterlockUnorderedEXT)
      .Default(::llvm::None);
}
::llvm::Optional<ExecutionMode> symbolizeExecutionMode(uint32_t value) {
  switch (value) {
  case 0: return ExecutionMode::Invocations;
  case 1: return ExecutionMode::SpacingEqual;
  case 2: return ExecutionMode::SpacingFractionalEven;
  case 3: return ExecutionMode::SpacingFractionalOdd;
  case 4: return ExecutionMode::VertexOrderCw;
  case 5: return ExecutionMode::VertexOrderCcw;
  case 6: return ExecutionMode::PixelCenterInteger;
  case 7: return ExecutionMode::OriginUpperLeft;
  case 8: return ExecutionMode::OriginLowerLeft;
  case 9: return ExecutionMode::EarlyFragmentTests;
  case 10: return ExecutionMode::PointMode;
  case 11: return ExecutionMode::Xfb;
  case 12: return ExecutionMode::DepthReplacing;
  case 14: return ExecutionMode::DepthGreater;
  case 15: return ExecutionMode::DepthLess;
  case 16: return ExecutionMode::DepthUnchanged;
  case 17: return ExecutionMode::LocalSize;
  case 18: return ExecutionMode::LocalSizeHint;
  case 19: return ExecutionMode::InputPoints;
  case 20: return ExecutionMode::InputLines;
  case 21: return ExecutionMode::InputLinesAdjacency;
  case 22: return ExecutionMode::Triangles;
  case 23: return ExecutionMode::InputTrianglesAdjacency;
  case 24: return ExecutionMode::Quads;
  case 25: return ExecutionMode::Isolines;
  case 26: return ExecutionMode::OutputVertices;
  case 27: return ExecutionMode::OutputPoints;
  case 28: return ExecutionMode::OutputLineStrip;
  case 29: return ExecutionMode::OutputTriangleStrip;
  case 30: return ExecutionMode::VecTypeHint;
  case 31: return ExecutionMode::ContractionOff;
  case 33: return ExecutionMode::Initializer;
  case 34: return ExecutionMode::Finalizer;
  case 35: return ExecutionMode::SubgroupSize;
  case 36: return ExecutionMode::SubgroupsPerWorkgroup;
  case 37: return ExecutionMode::SubgroupsPerWorkgroupId;
  case 38: return ExecutionMode::LocalSizeId;
  case 39: return ExecutionMode::LocalSizeHintId;
  case 4446: return ExecutionMode::PostDepthCoverage;
  case 4459: return ExecutionMode::DenormPreserve;
  case 4460: return ExecutionMode::DenormFlushToZero;
  case 4461: return ExecutionMode::SignedZeroInfNanPreserve;
  case 4462: return ExecutionMode::RoundingModeRTE;
  case 4463: return ExecutionMode::RoundingModeRTZ;
  case 5027: return ExecutionMode::StencilRefReplacingEXT;
  case 5269: return ExecutionMode::OutputLinesNV;
  case 5270: return ExecutionMode::OutputPrimitivesNV;
  case 5289: return ExecutionMode::DerivativeGroupQuadsNV;
  case 5290: return ExecutionMode::DerivativeGroupLinearNV;
  case 5298: return ExecutionMode::OutputTrianglesNV;
  case 5366: return ExecutionMode::PixelInterlockOrderedEXT;
  case 5367: return ExecutionMode::PixelInterlockUnorderedEXT;
  case 5368: return ExecutionMode::SampleInterlockOrderedEXT;
  case 5369: return ExecutionMode::SampleInterlockUnorderedEXT;
  case 5370: return ExecutionMode::ShadingRateInterlockOrderedEXT;
  case 5371: return ExecutionMode::ShadingRateInterlockUnorderedEXT;
  default: return ::llvm::None;
  }
}

bool ExecutionModeAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 18)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 21)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 22)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 23)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 24)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 25)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 26)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 27)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 28)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 29)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 30)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 31)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 33)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 34)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 35)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 36)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 37)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 38)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 39)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4446)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4459)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4460)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4461)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4462)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4463)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5027)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5269)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5270)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5289)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5290)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5298)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5366)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5367)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5368)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5369)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5370)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5371)));
}
ExecutionModeAttr ExecutionModeAttr::get(::mlir::MLIRContext *context, ExecutionMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ExecutionModeAttr>();
}
ExecutionMode ExecutionModeAttr::getValue() const {
  return static_cast<ExecutionMode>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyExecutionModel(ExecutionModel val) {
  switch (val) {
    case ExecutionModel::Vertex: return "Vertex";
    case ExecutionModel::TessellationControl: return "TessellationControl";
    case ExecutionModel::TessellationEvaluation: return "TessellationEvaluation";
    case ExecutionModel::Geometry: return "Geometry";
    case ExecutionModel::Fragment: return "Fragment";
    case ExecutionModel::GLCompute: return "GLCompute";
    case ExecutionModel::Kernel: return "Kernel";
    case ExecutionModel::TaskNV: return "TaskNV";
    case ExecutionModel::MeshNV: return "MeshNV";
    case ExecutionModel::RayGenerationNV: return "RayGenerationNV";
    case ExecutionModel::IntersectionNV: return "IntersectionNV";
    case ExecutionModel::AnyHitNV: return "AnyHitNV";
    case ExecutionModel::ClosestHitNV: return "ClosestHitNV";
    case ExecutionModel::MissNV: return "MissNV";
    case ExecutionModel::CallableNV: return "CallableNV";
  }
  return "";
}

::llvm::Optional<ExecutionModel> symbolizeExecutionModel(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ExecutionModel>>(str)
      .Case("Vertex", ExecutionModel::Vertex)
      .Case("TessellationControl", ExecutionModel::TessellationControl)
      .Case("TessellationEvaluation", ExecutionModel::TessellationEvaluation)
      .Case("Geometry", ExecutionModel::Geometry)
      .Case("Fragment", ExecutionModel::Fragment)
      .Case("GLCompute", ExecutionModel::GLCompute)
      .Case("Kernel", ExecutionModel::Kernel)
      .Case("TaskNV", ExecutionModel::TaskNV)
      .Case("MeshNV", ExecutionModel::MeshNV)
      .Case("RayGenerationNV", ExecutionModel::RayGenerationNV)
      .Case("IntersectionNV", ExecutionModel::IntersectionNV)
      .Case("AnyHitNV", ExecutionModel::AnyHitNV)
      .Case("ClosestHitNV", ExecutionModel::ClosestHitNV)
      .Case("MissNV", ExecutionModel::MissNV)
      .Case("CallableNV", ExecutionModel::CallableNV)
      .Default(::llvm::None);
}
::llvm::Optional<ExecutionModel> symbolizeExecutionModel(uint32_t value) {
  switch (value) {
  case 0: return ExecutionModel::Vertex;
  case 1: return ExecutionModel::TessellationControl;
  case 2: return ExecutionModel::TessellationEvaluation;
  case 3: return ExecutionModel::Geometry;
  case 4: return ExecutionModel::Fragment;
  case 5: return ExecutionModel::GLCompute;
  case 6: return ExecutionModel::Kernel;
  case 5267: return ExecutionModel::TaskNV;
  case 5268: return ExecutionModel::MeshNV;
  case 5313: return ExecutionModel::RayGenerationNV;
  case 5314: return ExecutionModel::IntersectionNV;
  case 5315: return ExecutionModel::AnyHitNV;
  case 5316: return ExecutionModel::ClosestHitNV;
  case 5317: return ExecutionModel::MissNV;
  case 5318: return ExecutionModel::CallableNV;
  default: return ::llvm::None;
  }
}

bool ExecutionModelAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5267)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5268)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5313)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5314)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5315)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5316)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5317)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5318)));
}
ExecutionModelAttr ExecutionModelAttr::get(::mlir::MLIRContext *context, ExecutionModel val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ExecutionModelAttr>();
}
ExecutionModel ExecutionModelAttr::getValue() const {
  return static_cast<ExecutionModel>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyExtension(Extension val) {
  switch (val) {
    case Extension::SPV_KHR_16bit_storage: return "SPV_KHR_16bit_storage";
    case Extension::SPV_KHR_8bit_storage: return "SPV_KHR_8bit_storage";
    case Extension::SPV_KHR_device_group: return "SPV_KHR_device_group";
    case Extension::SPV_KHR_float_controls: return "SPV_KHR_float_controls";
    case Extension::SPV_KHR_physical_storage_buffer: return "SPV_KHR_physical_storage_buffer";
    case Extension::SPV_KHR_multiview: return "SPV_KHR_multiview";
    case Extension::SPV_KHR_no_integer_wrap_decoration: return "SPV_KHR_no_integer_wrap_decoration";
    case Extension::SPV_KHR_post_depth_coverage: return "SPV_KHR_post_depth_coverage";
    case Extension::SPV_KHR_shader_atomic_counter_ops: return "SPV_KHR_shader_atomic_counter_ops";
    case Extension::SPV_KHR_shader_ballot: return "SPV_KHR_shader_ballot";
    case Extension::SPV_KHR_shader_clock: return "SPV_KHR_shader_clock";
    case Extension::SPV_KHR_shader_draw_parameters: return "SPV_KHR_shader_draw_parameters";
    case Extension::SPV_KHR_storage_buffer_storage_class: return "SPV_KHR_storage_buffer_storage_class";
    case Extension::SPV_KHR_subgroup_vote: return "SPV_KHR_subgroup_vote";
    case Extension::SPV_KHR_variable_pointers: return "SPV_KHR_variable_pointers";
    case Extension::SPV_KHR_vulkan_memory_model: return "SPV_KHR_vulkan_memory_model";
    case Extension::SPV_EXT_demote_to_helper_invocation: return "SPV_EXT_demote_to_helper_invocation";
    case Extension::SPV_EXT_descriptor_indexing: return "SPV_EXT_descriptor_indexing";
    case Extension::SPV_EXT_fragment_fully_covered: return "SPV_EXT_fragment_fully_covered";
    case Extension::SPV_EXT_fragment_invocation_density: return "SPV_EXT_fragment_invocation_density";
    case Extension::SPV_EXT_fragment_shader_interlock: return "SPV_EXT_fragment_shader_interlock";
    case Extension::SPV_EXT_physical_storage_buffer: return "SPV_EXT_physical_storage_buffer";
    case Extension::SPV_EXT_shader_stencil_export: return "SPV_EXT_shader_stencil_export";
    case Extension::SPV_EXT_shader_viewport_index_layer: return "SPV_EXT_shader_viewport_index_layer";
    case Extension::SPV_AMD_gpu_shader_half_float_fetch: return "SPV_AMD_gpu_shader_half_float_fetch";
    case Extension::SPV_AMD_shader_ballot: return "SPV_AMD_shader_ballot";
    case Extension::SPV_AMD_shader_explicit_vertex_parameter: return "SPV_AMD_shader_explicit_vertex_parameter";
    case Extension::SPV_AMD_shader_fragment_mask: return "SPV_AMD_shader_fragment_mask";
    case Extension::SPV_AMD_shader_image_load_store_lod: return "SPV_AMD_shader_image_load_store_lod";
    case Extension::SPV_AMD_texture_gather_bias_lod: return "SPV_AMD_texture_gather_bias_lod";
    case Extension::SPV_GOOGLE_decorate_string: return "SPV_GOOGLE_decorate_string";
    case Extension::SPV_GOOGLE_hlsl_functionality1: return "SPV_GOOGLE_hlsl_functionality1";
    case Extension::SPV_GOOGLE_user_type: return "SPV_GOOGLE_user_type";
    case Extension::SPV_INTEL_device_side_avc_motion_estimation: return "SPV_INTEL_device_side_avc_motion_estimation";
    case Extension::SPV_INTEL_media_block_io: return "SPV_INTEL_media_block_io";
    case Extension::SPV_INTEL_shader_integer_functions2: return "SPV_INTEL_shader_integer_functions2";
    case Extension::SPV_INTEL_subgroups: return "SPV_INTEL_subgroups";
    case Extension::SPV_NV_compute_shader_derivatives: return "SPV_NV_compute_shader_derivatives";
    case Extension::SPV_NV_cooperative_matrix: return "SPV_NV_cooperative_matrix";
    case Extension::SPV_NV_fragment_shader_barycentric: return "SPV_NV_fragment_shader_barycentric";
    case Extension::SPV_NV_geometry_shader_passthrough: return "SPV_NV_geometry_shader_passthrough";
    case Extension::SPV_NV_mesh_shader: return "SPV_NV_mesh_shader";
    case Extension::SPV_NV_ray_tracing: return "SPV_NV_ray_tracing";
    case Extension::SPV_NV_sample_mask_override_coverage: return "SPV_NV_sample_mask_override_coverage";
    case Extension::SPV_NV_shader_image_footprint: return "SPV_NV_shader_image_footprint";
    case Extension::SPV_NV_shader_sm_builtins: return "SPV_NV_shader_sm_builtins";
    case Extension::SPV_NV_shader_subgroup_partitioned: return "SPV_NV_shader_subgroup_partitioned";
    case Extension::SPV_NV_shading_rate: return "SPV_NV_shading_rate";
    case Extension::SPV_NV_stereo_view_rendering: return "SPV_NV_stereo_view_rendering";
    case Extension::SPV_NV_viewport_array2: return "SPV_NV_viewport_array2";
    case Extension::SPV_NVX_multiview_per_view_attributes: return "SPV_NVX_multiview_per_view_attributes";
  }
  return "";
}

::llvm::Optional<Extension> symbolizeExtension(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Extension>>(str)
      .Case("SPV_KHR_16bit_storage", Extension::SPV_KHR_16bit_storage)
      .Case("SPV_KHR_8bit_storage", Extension::SPV_KHR_8bit_storage)
      .Case("SPV_KHR_device_group", Extension::SPV_KHR_device_group)
      .Case("SPV_KHR_float_controls", Extension::SPV_KHR_float_controls)
      .Case("SPV_KHR_physical_storage_buffer", Extension::SPV_KHR_physical_storage_buffer)
      .Case("SPV_KHR_multiview", Extension::SPV_KHR_multiview)
      .Case("SPV_KHR_no_integer_wrap_decoration", Extension::SPV_KHR_no_integer_wrap_decoration)
      .Case("SPV_KHR_post_depth_coverage", Extension::SPV_KHR_post_depth_coverage)
      .Case("SPV_KHR_shader_atomic_counter_ops", Extension::SPV_KHR_shader_atomic_counter_ops)
      .Case("SPV_KHR_shader_ballot", Extension::SPV_KHR_shader_ballot)
      .Case("SPV_KHR_shader_clock", Extension::SPV_KHR_shader_clock)
      .Case("SPV_KHR_shader_draw_parameters", Extension::SPV_KHR_shader_draw_parameters)
      .Case("SPV_KHR_storage_buffer_storage_class", Extension::SPV_KHR_storage_buffer_storage_class)
      .Case("SPV_KHR_subgroup_vote", Extension::SPV_KHR_subgroup_vote)
      .Case("SPV_KHR_variable_pointers", Extension::SPV_KHR_variable_pointers)
      .Case("SPV_KHR_vulkan_memory_model", Extension::SPV_KHR_vulkan_memory_model)
      .Case("SPV_EXT_demote_to_helper_invocation", Extension::SPV_EXT_demote_to_helper_invocation)
      .Case("SPV_EXT_descriptor_indexing", Extension::SPV_EXT_descriptor_indexing)
      .Case("SPV_EXT_fragment_fully_covered", Extension::SPV_EXT_fragment_fully_covered)
      .Case("SPV_EXT_fragment_invocation_density", Extension::SPV_EXT_fragment_invocation_density)
      .Case("SPV_EXT_fragment_shader_interlock", Extension::SPV_EXT_fragment_shader_interlock)
      .Case("SPV_EXT_physical_storage_buffer", Extension::SPV_EXT_physical_storage_buffer)
      .Case("SPV_EXT_shader_stencil_export", Extension::SPV_EXT_shader_stencil_export)
      .Case("SPV_EXT_shader_viewport_index_layer", Extension::SPV_EXT_shader_viewport_index_layer)
      .Case("SPV_AMD_gpu_shader_half_float_fetch", Extension::SPV_AMD_gpu_shader_half_float_fetch)
      .Case("SPV_AMD_shader_ballot", Extension::SPV_AMD_shader_ballot)
      .Case("SPV_AMD_shader_explicit_vertex_parameter", Extension::SPV_AMD_shader_explicit_vertex_parameter)
      .Case("SPV_AMD_shader_fragment_mask", Extension::SPV_AMD_shader_fragment_mask)
      .Case("SPV_AMD_shader_image_load_store_lod", Extension::SPV_AMD_shader_image_load_store_lod)
      .Case("SPV_AMD_texture_gather_bias_lod", Extension::SPV_AMD_texture_gather_bias_lod)
      .Case("SPV_GOOGLE_decorate_string", Extension::SPV_GOOGLE_decorate_string)
      .Case("SPV_GOOGLE_hlsl_functionality1", Extension::SPV_GOOGLE_hlsl_functionality1)
      .Case("SPV_GOOGLE_user_type", Extension::SPV_GOOGLE_user_type)
      .Case("SPV_INTEL_device_side_avc_motion_estimation", Extension::SPV_INTEL_device_side_avc_motion_estimation)
      .Case("SPV_INTEL_media_block_io", Extension::SPV_INTEL_media_block_io)
      .Case("SPV_INTEL_shader_integer_functions2", Extension::SPV_INTEL_shader_integer_functions2)
      .Case("SPV_INTEL_subgroups", Extension::SPV_INTEL_subgroups)
      .Case("SPV_NV_compute_shader_derivatives", Extension::SPV_NV_compute_shader_derivatives)
      .Case("SPV_NV_cooperative_matrix", Extension::SPV_NV_cooperative_matrix)
      .Case("SPV_NV_fragment_shader_barycentric", Extension::SPV_NV_fragment_shader_barycentric)
      .Case("SPV_NV_geometry_shader_passthrough", Extension::SPV_NV_geometry_shader_passthrough)
      .Case("SPV_NV_mesh_shader", Extension::SPV_NV_mesh_shader)
      .Case("SPV_NV_ray_tracing", Extension::SPV_NV_ray_tracing)
      .Case("SPV_NV_sample_mask_override_coverage", Extension::SPV_NV_sample_mask_override_coverage)
      .Case("SPV_NV_shader_image_footprint", Extension::SPV_NV_shader_image_footprint)
      .Case("SPV_NV_shader_sm_builtins", Extension::SPV_NV_shader_sm_builtins)
      .Case("SPV_NV_shader_subgroup_partitioned", Extension::SPV_NV_shader_subgroup_partitioned)
      .Case("SPV_NV_shading_rate", Extension::SPV_NV_shading_rate)
      .Case("SPV_NV_stereo_view_rendering", Extension::SPV_NV_stereo_view_rendering)
      .Case("SPV_NV_viewport_array2", Extension::SPV_NV_viewport_array2)
      .Case("SPV_NVX_multiview_per_view_attributes", Extension::SPV_NVX_multiview_per_view_attributes)
      .Default(::llvm::None);
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
std::string stringifyFunctionControl(FunctionControl symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Inline"); val &= ~1u; }
  if (2u & val) { strs.push_back("DontInline"); val &= ~2u; }
  if (4u & val) { strs.push_back("Pure"); val &= ~4u; }
  if (8u & val) { strs.push_back("Const"); val &= ~8u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<FunctionControl> symbolizeFunctionControl(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return FunctionControl::None;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("Inline", 1)
      .Case("DontInline", 2)
      .Case("Pure", 4)
      .Case("Const", 8)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<FunctionControl>(val);
}

::llvm::Optional<FunctionControl> symbolizeFunctionControl(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return FunctionControl::None;

  if (value & ~(1u | 2u | 4u | 8u)) return llvm::None;
  return static_cast<FunctionControl>(value);
}
bool FunctionControlAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(0u|1u|2u|4u|8u)))));
}
FunctionControlAttr FunctionControlAttr::get(::mlir::MLIRContext *context, FunctionControl val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<FunctionControlAttr>();
}
FunctionControl FunctionControlAttr::getValue() const {
  return static_cast<FunctionControl>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyGroupOperation(GroupOperation val) {
  switch (val) {
    case GroupOperation::Reduce: return "Reduce";
    case GroupOperation::InclusiveScan: return "InclusiveScan";
    case GroupOperation::ExclusiveScan: return "ExclusiveScan";
    case GroupOperation::ClusteredReduce: return "ClusteredReduce";
    case GroupOperation::PartitionedReduceNV: return "PartitionedReduceNV";
    case GroupOperation::PartitionedInclusiveScanNV: return "PartitionedInclusiveScanNV";
    case GroupOperation::PartitionedExclusiveScanNV: return "PartitionedExclusiveScanNV";
  }
  return "";
}

::llvm::Optional<GroupOperation> symbolizeGroupOperation(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<GroupOperation>>(str)
      .Case("Reduce", GroupOperation::Reduce)
      .Case("InclusiveScan", GroupOperation::InclusiveScan)
      .Case("ExclusiveScan", GroupOperation::ExclusiveScan)
      .Case("ClusteredReduce", GroupOperation::ClusteredReduce)
      .Case("PartitionedReduceNV", GroupOperation::PartitionedReduceNV)
      .Case("PartitionedInclusiveScanNV", GroupOperation::PartitionedInclusiveScanNV)
      .Case("PartitionedExclusiveScanNV", GroupOperation::PartitionedExclusiveScanNV)
      .Default(::llvm::None);
}
::llvm::Optional<GroupOperation> symbolizeGroupOperation(uint32_t value) {
  switch (value) {
  case 0: return GroupOperation::Reduce;
  case 1: return GroupOperation::InclusiveScan;
  case 2: return GroupOperation::ExclusiveScan;
  case 3: return GroupOperation::ClusteredReduce;
  case 6: return GroupOperation::PartitionedReduceNV;
  case 7: return GroupOperation::PartitionedInclusiveScanNV;
  case 8: return GroupOperation::PartitionedExclusiveScanNV;
  default: return ::llvm::None;
  }
}

bool GroupOperationAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)));
}
GroupOperationAttr GroupOperationAttr::get(::mlir::MLIRContext *context, GroupOperation val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<GroupOperationAttr>();
}
GroupOperation GroupOperationAttr::getValue() const {
  return static_cast<GroupOperation>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyImageFormat(ImageFormat val) {
  switch (val) {
    case ImageFormat::Unknown: return "Unknown";
    case ImageFormat::Rgba32f: return "Rgba32f";
    case ImageFormat::Rgba16f: return "Rgba16f";
    case ImageFormat::R32f: return "R32f";
    case ImageFormat::Rgba8: return "Rgba8";
    case ImageFormat::Rgba8Snorm: return "Rgba8Snorm";
    case ImageFormat::Rg32f: return "Rg32f";
    case ImageFormat::Rg16f: return "Rg16f";
    case ImageFormat::R11fG11fB10f: return "R11fG11fB10f";
    case ImageFormat::R16f: return "R16f";
    case ImageFormat::Rgba16: return "Rgba16";
    case ImageFormat::Rgb10A2: return "Rgb10A2";
    case ImageFormat::Rg16: return "Rg16";
    case ImageFormat::Rg8: return "Rg8";
    case ImageFormat::R16: return "R16";
    case ImageFormat::R8: return "R8";
    case ImageFormat::Rgba16Snorm: return "Rgba16Snorm";
    case ImageFormat::Rg16Snorm: return "Rg16Snorm";
    case ImageFormat::Rg8Snorm: return "Rg8Snorm";
    case ImageFormat::R16Snorm: return "R16Snorm";
    case ImageFormat::R8Snorm: return "R8Snorm";
    case ImageFormat::Rgba32i: return "Rgba32i";
    case ImageFormat::Rgba16i: return "Rgba16i";
    case ImageFormat::Rgba8i: return "Rgba8i";
    case ImageFormat::R32i: return "R32i";
    case ImageFormat::Rg32i: return "Rg32i";
    case ImageFormat::Rg16i: return "Rg16i";
    case ImageFormat::Rg8i: return "Rg8i";
    case ImageFormat::R16i: return "R16i";
    case ImageFormat::R8i: return "R8i";
    case ImageFormat::Rgba32ui: return "Rgba32ui";
    case ImageFormat::Rgba16ui: return "Rgba16ui";
    case ImageFormat::Rgba8ui: return "Rgba8ui";
    case ImageFormat::R32ui: return "R32ui";
    case ImageFormat::Rgb10a2ui: return "Rgb10a2ui";
    case ImageFormat::Rg32ui: return "Rg32ui";
    case ImageFormat::Rg16ui: return "Rg16ui";
    case ImageFormat::Rg8ui: return "Rg8ui";
    case ImageFormat::R16ui: return "R16ui";
    case ImageFormat::R8ui: return "R8ui";
  }
  return "";
}

::llvm::Optional<ImageFormat> symbolizeImageFormat(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ImageFormat>>(str)
      .Case("Unknown", ImageFormat::Unknown)
      .Case("Rgba32f", ImageFormat::Rgba32f)
      .Case("Rgba16f", ImageFormat::Rgba16f)
      .Case("R32f", ImageFormat::R32f)
      .Case("Rgba8", ImageFormat::Rgba8)
      .Case("Rgba8Snorm", ImageFormat::Rgba8Snorm)
      .Case("Rg32f", ImageFormat::Rg32f)
      .Case("Rg16f", ImageFormat::Rg16f)
      .Case("R11fG11fB10f", ImageFormat::R11fG11fB10f)
      .Case("R16f", ImageFormat::R16f)
      .Case("Rgba16", ImageFormat::Rgba16)
      .Case("Rgb10A2", ImageFormat::Rgb10A2)
      .Case("Rg16", ImageFormat::Rg16)
      .Case("Rg8", ImageFormat::Rg8)
      .Case("R16", ImageFormat::R16)
      .Case("R8", ImageFormat::R8)
      .Case("Rgba16Snorm", ImageFormat::Rgba16Snorm)
      .Case("Rg16Snorm", ImageFormat::Rg16Snorm)
      .Case("Rg8Snorm", ImageFormat::Rg8Snorm)
      .Case("R16Snorm", ImageFormat::R16Snorm)
      .Case("R8Snorm", ImageFormat::R8Snorm)
      .Case("Rgba32i", ImageFormat::Rgba32i)
      .Case("Rgba16i", ImageFormat::Rgba16i)
      .Case("Rgba8i", ImageFormat::Rgba8i)
      .Case("R32i", ImageFormat::R32i)
      .Case("Rg32i", ImageFormat::Rg32i)
      .Case("Rg16i", ImageFormat::Rg16i)
      .Case("Rg8i", ImageFormat::Rg8i)
      .Case("R16i", ImageFormat::R16i)
      .Case("R8i", ImageFormat::R8i)
      .Case("Rgba32ui", ImageFormat::Rgba32ui)
      .Case("Rgba16ui", ImageFormat::Rgba16ui)
      .Case("Rgba8ui", ImageFormat::Rgba8ui)
      .Case("R32ui", ImageFormat::R32ui)
      .Case("Rgb10a2ui", ImageFormat::Rgb10a2ui)
      .Case("Rg32ui", ImageFormat::Rg32ui)
      .Case("Rg16ui", ImageFormat::Rg16ui)
      .Case("Rg8ui", ImageFormat::Rg8ui)
      .Case("R16ui", ImageFormat::R16ui)
      .Case("R8ui", ImageFormat::R8ui)
      .Default(::llvm::None);
}
::llvm::Optional<ImageFormat> symbolizeImageFormat(uint32_t value) {
  switch (value) {
  case 0: return ImageFormat::Unknown;
  case 1: return ImageFormat::Rgba32f;
  case 2: return ImageFormat::Rgba16f;
  case 3: return ImageFormat::R32f;
  case 4: return ImageFormat::Rgba8;
  case 5: return ImageFormat::Rgba8Snorm;
  case 6: return ImageFormat::Rg32f;
  case 7: return ImageFormat::Rg16f;
  case 8: return ImageFormat::R11fG11fB10f;
  case 9: return ImageFormat::R16f;
  case 10: return ImageFormat::Rgba16;
  case 11: return ImageFormat::Rgb10A2;
  case 12: return ImageFormat::Rg16;
  case 13: return ImageFormat::Rg8;
  case 14: return ImageFormat::R16;
  case 15: return ImageFormat::R8;
  case 16: return ImageFormat::Rgba16Snorm;
  case 17: return ImageFormat::Rg16Snorm;
  case 18: return ImageFormat::Rg8Snorm;
  case 19: return ImageFormat::R16Snorm;
  case 20: return ImageFormat::R8Snorm;
  case 21: return ImageFormat::Rgba32i;
  case 22: return ImageFormat::Rgba16i;
  case 23: return ImageFormat::Rgba8i;
  case 24: return ImageFormat::R32i;
  case 25: return ImageFormat::Rg32i;
  case 26: return ImageFormat::Rg16i;
  case 27: return ImageFormat::Rg8i;
  case 28: return ImageFormat::R16i;
  case 29: return ImageFormat::R8i;
  case 30: return ImageFormat::Rgba32ui;
  case 31: return ImageFormat::Rgba16ui;
  case 32: return ImageFormat::Rgba8ui;
  case 33: return ImageFormat::R32ui;
  case 34: return ImageFormat::Rgb10a2ui;
  case 35: return ImageFormat::Rg32ui;
  case 36: return ImageFormat::Rg16ui;
  case 37: return ImageFormat::Rg8ui;
  case 38: return ImageFormat::R16ui;
  case 39: return ImageFormat::R8ui;
  default: return ::llvm::None;
  }
}

bool ImageFormatAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 18)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 21)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 22)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 23)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 24)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 25)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 26)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 27)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 28)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 29)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 30)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 31)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 32)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 33)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 34)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 35)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 36)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 37)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 38)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 39)));
}
ImageFormatAttr ImageFormatAttr::get(::mlir::MLIRContext *context, ImageFormat val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ImageFormatAttr>();
}
ImageFormat ImageFormatAttr::getValue() const {
  return static_cast<ImageFormat>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyLinkageType(LinkageType val) {
  switch (val) {
    case LinkageType::Export: return "Export";
    case LinkageType::Import: return "Import";
  }
  return "";
}

::llvm::Optional<LinkageType> symbolizeLinkageType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<LinkageType>>(str)
      .Case("Export", LinkageType::Export)
      .Case("Import", LinkageType::Import)
      .Default(::llvm::None);
}
::llvm::Optional<LinkageType> symbolizeLinkageType(uint32_t value) {
  switch (value) {
  case 0: return LinkageType::Export;
  case 1: return LinkageType::Import;
  default: return ::llvm::None;
  }
}

bool LinkageTypeAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)));
}
LinkageTypeAttr LinkageTypeAttr::get(::mlir::MLIRContext *context, LinkageType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<LinkageTypeAttr>();
}
LinkageType LinkageTypeAttr::getValue() const {
  return static_cast<LinkageType>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
std::string stringifyLoopControl(LoopControl symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Unroll"); val &= ~1u; }
  if (2u & val) { strs.push_back("DontUnroll"); val &= ~2u; }
  if (4u & val) { strs.push_back("DependencyInfinite"); val &= ~4u; }
  if (8u & val) { strs.push_back("DependencyLength"); val &= ~8u; }
  if (16u & val) { strs.push_back("MinIterations"); val &= ~16u; }
  if (32u & val) { strs.push_back("MaxIterations"); val &= ~32u; }
  if (64u & val) { strs.push_back("IterationMultiple"); val &= ~64u; }
  if (128u & val) { strs.push_back("PeelCount"); val &= ~128u; }
  if (256u & val) { strs.push_back("PartialCount"); val &= ~256u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<LoopControl> symbolizeLoopControl(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return LoopControl::None;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("Unroll", 1)
      .Case("DontUnroll", 2)
      .Case("DependencyInfinite", 4)
      .Case("DependencyLength", 8)
      .Case("MinIterations", 16)
      .Case("MaxIterations", 32)
      .Case("IterationMultiple", 64)
      .Case("PeelCount", 128)
      .Case("PartialCount", 256)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<LoopControl>(val);
}

::llvm::Optional<LoopControl> symbolizeLoopControl(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return LoopControl::None;

  if (value & ~(1u | 2u | 4u | 8u | 16u | 32u | 64u | 128u | 256u)) return llvm::None;
  return static_cast<LoopControl>(value);
}
bool LoopControlAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(0u|1u|2u|4u|8u|16u|32u|64u|128u|256u)))));
}
LoopControlAttr LoopControlAttr::get(::mlir::MLIRContext *context, LoopControl val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<LoopControlAttr>();
}
LoopControl LoopControlAttr::getValue() const {
  return static_cast<LoopControl>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
std::string stringifyMemoryAccess(MemoryAccess symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Volatile"); val &= ~1u; }
  if (2u & val) { strs.push_back("Aligned"); val &= ~2u; }
  if (4u & val) { strs.push_back("Nontemporal"); val &= ~4u; }
  if (8u & val) { strs.push_back("MakePointerAvailable"); val &= ~8u; }
  if (16u & val) { strs.push_back("MakePointerVisible"); val &= ~16u; }
  if (32u & val) { strs.push_back("NonPrivatePointer"); val &= ~32u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<MemoryAccess> symbolizeMemoryAccess(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return MemoryAccess::None;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("Volatile", 1)
      .Case("Aligned", 2)
      .Case("Nontemporal", 4)
      .Case("MakePointerAvailable", 8)
      .Case("MakePointerVisible", 16)
      .Case("NonPrivatePointer", 32)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<MemoryAccess>(val);
}

::llvm::Optional<MemoryAccess> symbolizeMemoryAccess(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return MemoryAccess::None;

  if (value & ~(1u | 2u | 4u | 8u | 16u | 32u)) return llvm::None;
  return static_cast<MemoryAccess>(value);
}
bool MemoryAccessAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(0u|1u|2u|4u|8u|16u|32u)))));
}
MemoryAccessAttr MemoryAccessAttr::get(::mlir::MLIRContext *context, MemoryAccess val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<MemoryAccessAttr>();
}
MemoryAccess MemoryAccessAttr::getValue() const {
  return static_cast<MemoryAccess>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyMemoryModel(MemoryModel val) {
  switch (val) {
    case MemoryModel::Simple: return "Simple";
    case MemoryModel::GLSL450: return "GLSL450";
    case MemoryModel::OpenCL: return "OpenCL";
    case MemoryModel::Vulkan: return "Vulkan";
  }
  return "";
}

::llvm::Optional<MemoryModel> symbolizeMemoryModel(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MemoryModel>>(str)
      .Case("Simple", MemoryModel::Simple)
      .Case("GLSL450", MemoryModel::GLSL450)
      .Case("OpenCL", MemoryModel::OpenCL)
      .Case("Vulkan", MemoryModel::Vulkan)
      .Default(::llvm::None);
}
::llvm::Optional<MemoryModel> symbolizeMemoryModel(uint32_t value) {
  switch (value) {
  case 0: return MemoryModel::Simple;
  case 1: return MemoryModel::GLSL450;
  case 2: return MemoryModel::OpenCL;
  case 3: return MemoryModel::Vulkan;
  default: return ::llvm::None;
  }
}

bool MemoryModelAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)));
}
MemoryModelAttr MemoryModelAttr::get(::mlir::MLIRContext *context, MemoryModel val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<MemoryModelAttr>();
}
MemoryModel MemoryModelAttr::getValue() const {
  return static_cast<MemoryModel>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
std::string stringifyMemorySemantics(MemorySemantics symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (2u & val) { strs.push_back("Acquire"); val &= ~2u; }
  if (4u & val) { strs.push_back("Release"); val &= ~4u; }
  if (8u & val) { strs.push_back("AcquireRelease"); val &= ~8u; }
  if (16u & val) { strs.push_back("SequentiallyConsistent"); val &= ~16u; }
  if (64u & val) { strs.push_back("UniformMemory"); val &= ~64u; }
  if (128u & val) { strs.push_back("SubgroupMemory"); val &= ~128u; }
  if (256u & val) { strs.push_back("WorkgroupMemory"); val &= ~256u; }
  if (512u & val) { strs.push_back("CrossWorkgroupMemory"); val &= ~512u; }
  if (1024u & val) { strs.push_back("AtomicCounterMemory"); val &= ~1024u; }
  if (2048u & val) { strs.push_back("ImageMemory"); val &= ~2048u; }
  if (4096u & val) { strs.push_back("OutputMemory"); val &= ~4096u; }
  if (8192u & val) { strs.push_back("MakeAvailable"); val &= ~8192u; }
  if (16384u & val) { strs.push_back("MakeVisible"); val &= ~16384u; }
  if (32768u & val) { strs.push_back("Volatile"); val &= ~32768u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<MemorySemantics> symbolizeMemorySemantics(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return MemorySemantics::None;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("Acquire", 2)
      .Case("Release", 4)
      .Case("AcquireRelease", 8)
      .Case("SequentiallyConsistent", 16)
      .Case("UniformMemory", 64)
      .Case("SubgroupMemory", 128)
      .Case("WorkgroupMemory", 256)
      .Case("CrossWorkgroupMemory", 512)
      .Case("AtomicCounterMemory", 1024)
      .Case("ImageMemory", 2048)
      .Case("OutputMemory", 4096)
      .Case("MakeAvailable", 8192)
      .Case("MakeVisible", 16384)
      .Case("Volatile", 32768)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<MemorySemantics>(val);
}

::llvm::Optional<MemorySemantics> symbolizeMemorySemantics(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return MemorySemantics::None;

  if (value & ~(2u | 4u | 8u | 16u | 64u | 128u | 256u | 512u | 1024u | 2048u | 4096u | 8192u | 16384u | 32768u)) return llvm::None;
  return static_cast<MemorySemantics>(value);
}
bool MemorySemanticsAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(0u|2u|4u|8u|16u|64u|128u|256u|512u|1024u|2048u|4096u|8192u|16384u|32768u)))));
}
MemorySemanticsAttr MemorySemanticsAttr::get(::mlir::MLIRContext *context, MemorySemantics val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<MemorySemanticsAttr>();
}
MemorySemantics MemorySemanticsAttr::getValue() const {
  return static_cast<MemorySemantics>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyOpcode(Opcode val) {
  switch (val) {
    case Opcode::OpNop: return "OpNop";
    case Opcode::OpUndef: return "OpUndef";
    case Opcode::OpSourceContinued: return "OpSourceContinued";
    case Opcode::OpSource: return "OpSource";
    case Opcode::OpSourceExtension: return "OpSourceExtension";
    case Opcode::OpName: return "OpName";
    case Opcode::OpMemberName: return "OpMemberName";
    case Opcode::OpString: return "OpString";
    case Opcode::OpLine: return "OpLine";
    case Opcode::OpExtension: return "OpExtension";
    case Opcode::OpExtInstImport: return "OpExtInstImport";
    case Opcode::OpExtInst: return "OpExtInst";
    case Opcode::OpMemoryModel: return "OpMemoryModel";
    case Opcode::OpEntryPoint: return "OpEntryPoint";
    case Opcode::OpExecutionMode: return "OpExecutionMode";
    case Opcode::OpCapability: return "OpCapability";
    case Opcode::OpTypeVoid: return "OpTypeVoid";
    case Opcode::OpTypeBool: return "OpTypeBool";
    case Opcode::OpTypeInt: return "OpTypeInt";
    case Opcode::OpTypeFloat: return "OpTypeFloat";
    case Opcode::OpTypeVector: return "OpTypeVector";
    case Opcode::OpTypeMatrix: return "OpTypeMatrix";
    case Opcode::OpTypeImage: return "OpTypeImage";
    case Opcode::OpTypeSampledImage: return "OpTypeSampledImage";
    case Opcode::OpTypeArray: return "OpTypeArray";
    case Opcode::OpTypeRuntimeArray: return "OpTypeRuntimeArray";
    case Opcode::OpTypeStruct: return "OpTypeStruct";
    case Opcode::OpTypePointer: return "OpTypePointer";
    case Opcode::OpTypeFunction: return "OpTypeFunction";
    case Opcode::OpTypeForwardPointer: return "OpTypeForwardPointer";
    case Opcode::OpConstantTrue: return "OpConstantTrue";
    case Opcode::OpConstantFalse: return "OpConstantFalse";
    case Opcode::OpConstant: return "OpConstant";
    case Opcode::OpConstantComposite: return "OpConstantComposite";
    case Opcode::OpConstantNull: return "OpConstantNull";
    case Opcode::OpSpecConstantTrue: return "OpSpecConstantTrue";
    case Opcode::OpSpecConstantFalse: return "OpSpecConstantFalse";
    case Opcode::OpSpecConstant: return "OpSpecConstant";
    case Opcode::OpSpecConstantComposite: return "OpSpecConstantComposite";
    case Opcode::OpSpecConstantOp: return "OpSpecConstantOp";
    case Opcode::OpFunction: return "OpFunction";
    case Opcode::OpFunctionParameter: return "OpFunctionParameter";
    case Opcode::OpFunctionEnd: return "OpFunctionEnd";
    case Opcode::OpFunctionCall: return "OpFunctionCall";
    case Opcode::OpVariable: return "OpVariable";
    case Opcode::OpLoad: return "OpLoad";
    case Opcode::OpStore: return "OpStore";
    case Opcode::OpCopyMemory: return "OpCopyMemory";
    case Opcode::OpAccessChain: return "OpAccessChain";
    case Opcode::OpDecorate: return "OpDecorate";
    case Opcode::OpMemberDecorate: return "OpMemberDecorate";
    case Opcode::OpVectorExtractDynamic: return "OpVectorExtractDynamic";
    case Opcode::OpVectorInsertDynamic: return "OpVectorInsertDynamic";
    case Opcode::OpVectorShuffle: return "OpVectorShuffle";
    case Opcode::OpCompositeConstruct: return "OpCompositeConstruct";
    case Opcode::OpCompositeExtract: return "OpCompositeExtract";
    case Opcode::OpCompositeInsert: return "OpCompositeInsert";
    case Opcode::OpTranspose: return "OpTranspose";
    case Opcode::OpImage: return "OpImage";
    case Opcode::OpConvertFToU: return "OpConvertFToU";
    case Opcode::OpConvertFToS: return "OpConvertFToS";
    case Opcode::OpConvertSToF: return "OpConvertSToF";
    case Opcode::OpConvertUToF: return "OpConvertUToF";
    case Opcode::OpUConvert: return "OpUConvert";
    case Opcode::OpSConvert: return "OpSConvert";
    case Opcode::OpFConvert: return "OpFConvert";
    case Opcode::OpBitcast: return "OpBitcast";
    case Opcode::OpSNegate: return "OpSNegate";
    case Opcode::OpFNegate: return "OpFNegate";
    case Opcode::OpIAdd: return "OpIAdd";
    case Opcode::OpFAdd: return "OpFAdd";
    case Opcode::OpISub: return "OpISub";
    case Opcode::OpFSub: return "OpFSub";
    case Opcode::OpIMul: return "OpIMul";
    case Opcode::OpFMul: return "OpFMul";
    case Opcode::OpUDiv: return "OpUDiv";
    case Opcode::OpSDiv: return "OpSDiv";
    case Opcode::OpFDiv: return "OpFDiv";
    case Opcode::OpUMod: return "OpUMod";
    case Opcode::OpSRem: return "OpSRem";
    case Opcode::OpSMod: return "OpSMod";
    case Opcode::OpFRem: return "OpFRem";
    case Opcode::OpFMod: return "OpFMod";
    case Opcode::OpMatrixTimesScalar: return "OpMatrixTimesScalar";
    case Opcode::OpMatrixTimesMatrix: return "OpMatrixTimesMatrix";
    case Opcode::OpIsNan: return "OpIsNan";
    case Opcode::OpIsInf: return "OpIsInf";
    case Opcode::OpOrdered: return "OpOrdered";
    case Opcode::OpUnordered: return "OpUnordered";
    case Opcode::OpLogicalEqual: return "OpLogicalEqual";
    case Opcode::OpLogicalNotEqual: return "OpLogicalNotEqual";
    case Opcode::OpLogicalOr: return "OpLogicalOr";
    case Opcode::OpLogicalAnd: return "OpLogicalAnd";
    case Opcode::OpLogicalNot: return "OpLogicalNot";
    case Opcode::OpSelect: return "OpSelect";
    case Opcode::OpIEqual: return "OpIEqual";
    case Opcode::OpINotEqual: return "OpINotEqual";
    case Opcode::OpUGreaterThan: return "OpUGreaterThan";
    case Opcode::OpSGreaterThan: return "OpSGreaterThan";
    case Opcode::OpUGreaterThanEqual: return "OpUGreaterThanEqual";
    case Opcode::OpSGreaterThanEqual: return "OpSGreaterThanEqual";
    case Opcode::OpULessThan: return "OpULessThan";
    case Opcode::OpSLessThan: return "OpSLessThan";
    case Opcode::OpULessThanEqual: return "OpULessThanEqual";
    case Opcode::OpSLessThanEqual: return "OpSLessThanEqual";
    case Opcode::OpFOrdEqual: return "OpFOrdEqual";
    case Opcode::OpFUnordEqual: return "OpFUnordEqual";
    case Opcode::OpFOrdNotEqual: return "OpFOrdNotEqual";
    case Opcode::OpFUnordNotEqual: return "OpFUnordNotEqual";
    case Opcode::OpFOrdLessThan: return "OpFOrdLessThan";
    case Opcode::OpFUnordLessThan: return "OpFUnordLessThan";
    case Opcode::OpFOrdGreaterThan: return "OpFOrdGreaterThan";
    case Opcode::OpFUnordGreaterThan: return "OpFUnordGreaterThan";
    case Opcode::OpFOrdLessThanEqual: return "OpFOrdLessThanEqual";
    case Opcode::OpFUnordLessThanEqual: return "OpFUnordLessThanEqual";
    case Opcode::OpFOrdGreaterThanEqual: return "OpFOrdGreaterThanEqual";
    case Opcode::OpFUnordGreaterThanEqual: return "OpFUnordGreaterThanEqual";
    case Opcode::OpShiftRightLogical: return "OpShiftRightLogical";
    case Opcode::OpShiftRightArithmetic: return "OpShiftRightArithmetic";
    case Opcode::OpShiftLeftLogical: return "OpShiftLeftLogical";
    case Opcode::OpBitwiseOr: return "OpBitwiseOr";
    case Opcode::OpBitwiseXor: return "OpBitwiseXor";
    case Opcode::OpBitwiseAnd: return "OpBitwiseAnd";
    case Opcode::OpNot: return "OpNot";
    case Opcode::OpBitFieldInsert: return "OpBitFieldInsert";
    case Opcode::OpBitFieldSExtract: return "OpBitFieldSExtract";
    case Opcode::OpBitFieldUExtract: return "OpBitFieldUExtract";
    case Opcode::OpBitReverse: return "OpBitReverse";
    case Opcode::OpBitCount: return "OpBitCount";
    case Opcode::OpControlBarrier: return "OpControlBarrier";
    case Opcode::OpMemoryBarrier: return "OpMemoryBarrier";
    case Opcode::OpAtomicCompareExchangeWeak: return "OpAtomicCompareExchangeWeak";
    case Opcode::OpAtomicIIncrement: return "OpAtomicIIncrement";
    case Opcode::OpAtomicIDecrement: return "OpAtomicIDecrement";
    case Opcode::OpAtomicIAdd: return "OpAtomicIAdd";
    case Opcode::OpAtomicISub: return "OpAtomicISub";
    case Opcode::OpAtomicSMin: return "OpAtomicSMin";
    case Opcode::OpAtomicUMin: return "OpAtomicUMin";
    case Opcode::OpAtomicSMax: return "OpAtomicSMax";
    case Opcode::OpAtomicUMax: return "OpAtomicUMax";
    case Opcode::OpAtomicAnd: return "OpAtomicAnd";
    case Opcode::OpAtomicOr: return "OpAtomicOr";
    case Opcode::OpAtomicXor: return "OpAtomicXor";
    case Opcode::OpPhi: return "OpPhi";
    case Opcode::OpLoopMerge: return "OpLoopMerge";
    case Opcode::OpSelectionMerge: return "OpSelectionMerge";
    case Opcode::OpLabel: return "OpLabel";
    case Opcode::OpBranch: return "OpBranch";
    case Opcode::OpBranchConditional: return "OpBranchConditional";
    case Opcode::OpReturn: return "OpReturn";
    case Opcode::OpReturnValue: return "OpReturnValue";
    case Opcode::OpUnreachable: return "OpUnreachable";
    case Opcode::OpGroupBroadcast: return "OpGroupBroadcast";
    case Opcode::OpNoLine: return "OpNoLine";
    case Opcode::OpModuleProcessed: return "OpModuleProcessed";
    case Opcode::OpGroupNonUniformElect: return "OpGroupNonUniformElect";
    case Opcode::OpGroupNonUniformBroadcast: return "OpGroupNonUniformBroadcast";
    case Opcode::OpGroupNonUniformBallot: return "OpGroupNonUniformBallot";
    case Opcode::OpGroupNonUniformIAdd: return "OpGroupNonUniformIAdd";
    case Opcode::OpGroupNonUniformFAdd: return "OpGroupNonUniformFAdd";
    case Opcode::OpGroupNonUniformIMul: return "OpGroupNonUniformIMul";
    case Opcode::OpGroupNonUniformFMul: return "OpGroupNonUniformFMul";
    case Opcode::OpGroupNonUniformSMin: return "OpGroupNonUniformSMin";
    case Opcode::OpGroupNonUniformUMin: return "OpGroupNonUniformUMin";
    case Opcode::OpGroupNonUniformFMin: return "OpGroupNonUniformFMin";
    case Opcode::OpGroupNonUniformSMax: return "OpGroupNonUniformSMax";
    case Opcode::OpGroupNonUniformUMax: return "OpGroupNonUniformUMax";
    case Opcode::OpGroupNonUniformFMax: return "OpGroupNonUniformFMax";
    case Opcode::OpSubgroupBallotKHR: return "OpSubgroupBallotKHR";
    case Opcode::OpTypeCooperativeMatrixNV: return "OpTypeCooperativeMatrixNV";
    case Opcode::OpCooperativeMatrixLoadNV: return "OpCooperativeMatrixLoadNV";
    case Opcode::OpCooperativeMatrixStoreNV: return "OpCooperativeMatrixStoreNV";
    case Opcode::OpCooperativeMatrixMulAddNV: return "OpCooperativeMatrixMulAddNV";
    case Opcode::OpCooperativeMatrixLengthNV: return "OpCooperativeMatrixLengthNV";
    case Opcode::OpSubgroupBlockReadINTEL: return "OpSubgroupBlockReadINTEL";
    case Opcode::OpSubgroupBlockWriteINTEL: return "OpSubgroupBlockWriteINTEL";
  }
  return "";
}

::llvm::Optional<Opcode> symbolizeOpcode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Opcode>>(str)
      .Case("OpNop", Opcode::OpNop)
      .Case("OpUndef", Opcode::OpUndef)
      .Case("OpSourceContinued", Opcode::OpSourceContinued)
      .Case("OpSource", Opcode::OpSource)
      .Case("OpSourceExtension", Opcode::OpSourceExtension)
      .Case("OpName", Opcode::OpName)
      .Case("OpMemberName", Opcode::OpMemberName)
      .Case("OpString", Opcode::OpString)
      .Case("OpLine", Opcode::OpLine)
      .Case("OpExtension", Opcode::OpExtension)
      .Case("OpExtInstImport", Opcode::OpExtInstImport)
      .Case("OpExtInst", Opcode::OpExtInst)
      .Case("OpMemoryModel", Opcode::OpMemoryModel)
      .Case("OpEntryPoint", Opcode::OpEntryPoint)
      .Case("OpExecutionMode", Opcode::OpExecutionMode)
      .Case("OpCapability", Opcode::OpCapability)
      .Case("OpTypeVoid", Opcode::OpTypeVoid)
      .Case("OpTypeBool", Opcode::OpTypeBool)
      .Case("OpTypeInt", Opcode::OpTypeInt)
      .Case("OpTypeFloat", Opcode::OpTypeFloat)
      .Case("OpTypeVector", Opcode::OpTypeVector)
      .Case("OpTypeMatrix", Opcode::OpTypeMatrix)
      .Case("OpTypeImage", Opcode::OpTypeImage)
      .Case("OpTypeSampledImage", Opcode::OpTypeSampledImage)
      .Case("OpTypeArray", Opcode::OpTypeArray)
      .Case("OpTypeRuntimeArray", Opcode::OpTypeRuntimeArray)
      .Case("OpTypeStruct", Opcode::OpTypeStruct)
      .Case("OpTypePointer", Opcode::OpTypePointer)
      .Case("OpTypeFunction", Opcode::OpTypeFunction)
      .Case("OpTypeForwardPointer", Opcode::OpTypeForwardPointer)
      .Case("OpConstantTrue", Opcode::OpConstantTrue)
      .Case("OpConstantFalse", Opcode::OpConstantFalse)
      .Case("OpConstant", Opcode::OpConstant)
      .Case("OpConstantComposite", Opcode::OpConstantComposite)
      .Case("OpConstantNull", Opcode::OpConstantNull)
      .Case("OpSpecConstantTrue", Opcode::OpSpecConstantTrue)
      .Case("OpSpecConstantFalse", Opcode::OpSpecConstantFalse)
      .Case("OpSpecConstant", Opcode::OpSpecConstant)
      .Case("OpSpecConstantComposite", Opcode::OpSpecConstantComposite)
      .Case("OpSpecConstantOp", Opcode::OpSpecConstantOp)
      .Case("OpFunction", Opcode::OpFunction)
      .Case("OpFunctionParameter", Opcode::OpFunctionParameter)
      .Case("OpFunctionEnd", Opcode::OpFunctionEnd)
      .Case("OpFunctionCall", Opcode::OpFunctionCall)
      .Case("OpVariable", Opcode::OpVariable)
      .Case("OpLoad", Opcode::OpLoad)
      .Case("OpStore", Opcode::OpStore)
      .Case("OpCopyMemory", Opcode::OpCopyMemory)
      .Case("OpAccessChain", Opcode::OpAccessChain)
      .Case("OpDecorate", Opcode::OpDecorate)
      .Case("OpMemberDecorate", Opcode::OpMemberDecorate)
      .Case("OpVectorExtractDynamic", Opcode::OpVectorExtractDynamic)
      .Case("OpVectorInsertDynamic", Opcode::OpVectorInsertDynamic)
      .Case("OpVectorShuffle", Opcode::OpVectorShuffle)
      .Case("OpCompositeConstruct", Opcode::OpCompositeConstruct)
      .Case("OpCompositeExtract", Opcode::OpCompositeExtract)
      .Case("OpCompositeInsert", Opcode::OpCompositeInsert)
      .Case("OpTranspose", Opcode::OpTranspose)
      .Case("OpImage", Opcode::OpImage)
      .Case("OpConvertFToU", Opcode::OpConvertFToU)
      .Case("OpConvertFToS", Opcode::OpConvertFToS)
      .Case("OpConvertSToF", Opcode::OpConvertSToF)
      .Case("OpConvertUToF", Opcode::OpConvertUToF)
      .Case("OpUConvert", Opcode::OpUConvert)
      .Case("OpSConvert", Opcode::OpSConvert)
      .Case("OpFConvert", Opcode::OpFConvert)
      .Case("OpBitcast", Opcode::OpBitcast)
      .Case("OpSNegate", Opcode::OpSNegate)
      .Case("OpFNegate", Opcode::OpFNegate)
      .Case("OpIAdd", Opcode::OpIAdd)
      .Case("OpFAdd", Opcode::OpFAdd)
      .Case("OpISub", Opcode::OpISub)
      .Case("OpFSub", Opcode::OpFSub)
      .Case("OpIMul", Opcode::OpIMul)
      .Case("OpFMul", Opcode::OpFMul)
      .Case("OpUDiv", Opcode::OpUDiv)
      .Case("OpSDiv", Opcode::OpSDiv)
      .Case("OpFDiv", Opcode::OpFDiv)
      .Case("OpUMod", Opcode::OpUMod)
      .Case("OpSRem", Opcode::OpSRem)
      .Case("OpSMod", Opcode::OpSMod)
      .Case("OpFRem", Opcode::OpFRem)
      .Case("OpFMod", Opcode::OpFMod)
      .Case("OpMatrixTimesScalar", Opcode::OpMatrixTimesScalar)
      .Case("OpMatrixTimesMatrix", Opcode::OpMatrixTimesMatrix)
      .Case("OpIsNan", Opcode::OpIsNan)
      .Case("OpIsInf", Opcode::OpIsInf)
      .Case("OpOrdered", Opcode::OpOrdered)
      .Case("OpUnordered", Opcode::OpUnordered)
      .Case("OpLogicalEqual", Opcode::OpLogicalEqual)
      .Case("OpLogicalNotEqual", Opcode::OpLogicalNotEqual)
      .Case("OpLogicalOr", Opcode::OpLogicalOr)
      .Case("OpLogicalAnd", Opcode::OpLogicalAnd)
      .Case("OpLogicalNot", Opcode::OpLogicalNot)
      .Case("OpSelect", Opcode::OpSelect)
      .Case("OpIEqual", Opcode::OpIEqual)
      .Case("OpINotEqual", Opcode::OpINotEqual)
      .Case("OpUGreaterThan", Opcode::OpUGreaterThan)
      .Case("OpSGreaterThan", Opcode::OpSGreaterThan)
      .Case("OpUGreaterThanEqual", Opcode::OpUGreaterThanEqual)
      .Case("OpSGreaterThanEqual", Opcode::OpSGreaterThanEqual)
      .Case("OpULessThan", Opcode::OpULessThan)
      .Case("OpSLessThan", Opcode::OpSLessThan)
      .Case("OpULessThanEqual", Opcode::OpULessThanEqual)
      .Case("OpSLessThanEqual", Opcode::OpSLessThanEqual)
      .Case("OpFOrdEqual", Opcode::OpFOrdEqual)
      .Case("OpFUnordEqual", Opcode::OpFUnordEqual)
      .Case("OpFOrdNotEqual", Opcode::OpFOrdNotEqual)
      .Case("OpFUnordNotEqual", Opcode::OpFUnordNotEqual)
      .Case("OpFOrdLessThan", Opcode::OpFOrdLessThan)
      .Case("OpFUnordLessThan", Opcode::OpFUnordLessThan)
      .Case("OpFOrdGreaterThan", Opcode::OpFOrdGreaterThan)
      .Case("OpFUnordGreaterThan", Opcode::OpFUnordGreaterThan)
      .Case("OpFOrdLessThanEqual", Opcode::OpFOrdLessThanEqual)
      .Case("OpFUnordLessThanEqual", Opcode::OpFUnordLessThanEqual)
      .Case("OpFOrdGreaterThanEqual", Opcode::OpFOrdGreaterThanEqual)
      .Case("OpFUnordGreaterThanEqual", Opcode::OpFUnordGreaterThanEqual)
      .Case("OpShiftRightLogical", Opcode::OpShiftRightLogical)
      .Case("OpShiftRightArithmetic", Opcode::OpShiftRightArithmetic)
      .Case("OpShiftLeftLogical", Opcode::OpShiftLeftLogical)
      .Case("OpBitwiseOr", Opcode::OpBitwiseOr)
      .Case("OpBitwiseXor", Opcode::OpBitwiseXor)
      .Case("OpBitwiseAnd", Opcode::OpBitwiseAnd)
      .Case("OpNot", Opcode::OpNot)
      .Case("OpBitFieldInsert", Opcode::OpBitFieldInsert)
      .Case("OpBitFieldSExtract", Opcode::OpBitFieldSExtract)
      .Case("OpBitFieldUExtract", Opcode::OpBitFieldUExtract)
      .Case("OpBitReverse", Opcode::OpBitReverse)
      .Case("OpBitCount", Opcode::OpBitCount)
      .Case("OpControlBarrier", Opcode::OpControlBarrier)
      .Case("OpMemoryBarrier", Opcode::OpMemoryBarrier)
      .Case("OpAtomicCompareExchangeWeak", Opcode::OpAtomicCompareExchangeWeak)
      .Case("OpAtomicIIncrement", Opcode::OpAtomicIIncrement)
      .Case("OpAtomicIDecrement", Opcode::OpAtomicIDecrement)
      .Case("OpAtomicIAdd", Opcode::OpAtomicIAdd)
      .Case("OpAtomicISub", Opcode::OpAtomicISub)
      .Case("OpAtomicSMin", Opcode::OpAtomicSMin)
      .Case("OpAtomicUMin", Opcode::OpAtomicUMin)
      .Case("OpAtomicSMax", Opcode::OpAtomicSMax)
      .Case("OpAtomicUMax", Opcode::OpAtomicUMax)
      .Case("OpAtomicAnd", Opcode::OpAtomicAnd)
      .Case("OpAtomicOr", Opcode::OpAtomicOr)
      .Case("OpAtomicXor", Opcode::OpAtomicXor)
      .Case("OpPhi", Opcode::OpPhi)
      .Case("OpLoopMerge", Opcode::OpLoopMerge)
      .Case("OpSelectionMerge", Opcode::OpSelectionMerge)
      .Case("OpLabel", Opcode::OpLabel)
      .Case("OpBranch", Opcode::OpBranch)
      .Case("OpBranchConditional", Opcode::OpBranchConditional)
      .Case("OpReturn", Opcode::OpReturn)
      .Case("OpReturnValue", Opcode::OpReturnValue)
      .Case("OpUnreachable", Opcode::OpUnreachable)
      .Case("OpGroupBroadcast", Opcode::OpGroupBroadcast)
      .Case("OpNoLine", Opcode::OpNoLine)
      .Case("OpModuleProcessed", Opcode::OpModuleProcessed)
      .Case("OpGroupNonUniformElect", Opcode::OpGroupNonUniformElect)
      .Case("OpGroupNonUniformBroadcast", Opcode::OpGroupNonUniformBroadcast)
      .Case("OpGroupNonUniformBallot", Opcode::OpGroupNonUniformBallot)
      .Case("OpGroupNonUniformIAdd", Opcode::OpGroupNonUniformIAdd)
      .Case("OpGroupNonUniformFAdd", Opcode::OpGroupNonUniformFAdd)
      .Case("OpGroupNonUniformIMul", Opcode::OpGroupNonUniformIMul)
      .Case("OpGroupNonUniformFMul", Opcode::OpGroupNonUniformFMul)
      .Case("OpGroupNonUniformSMin", Opcode::OpGroupNonUniformSMin)
      .Case("OpGroupNonUniformUMin", Opcode::OpGroupNonUniformUMin)
      .Case("OpGroupNonUniformFMin", Opcode::OpGroupNonUniformFMin)
      .Case("OpGroupNonUniformSMax", Opcode::OpGroupNonUniformSMax)
      .Case("OpGroupNonUniformUMax", Opcode::OpGroupNonUniformUMax)
      .Case("OpGroupNonUniformFMax", Opcode::OpGroupNonUniformFMax)
      .Case("OpSubgroupBallotKHR", Opcode::OpSubgroupBallotKHR)
      .Case("OpTypeCooperativeMatrixNV", Opcode::OpTypeCooperativeMatrixNV)
      .Case("OpCooperativeMatrixLoadNV", Opcode::OpCooperativeMatrixLoadNV)
      .Case("OpCooperativeMatrixStoreNV", Opcode::OpCooperativeMatrixStoreNV)
      .Case("OpCooperativeMatrixMulAddNV", Opcode::OpCooperativeMatrixMulAddNV)
      .Case("OpCooperativeMatrixLengthNV", Opcode::OpCooperativeMatrixLengthNV)
      .Case("OpSubgroupBlockReadINTEL", Opcode::OpSubgroupBlockReadINTEL)
      .Case("OpSubgroupBlockWriteINTEL", Opcode::OpSubgroupBlockWriteINTEL)
      .Default(::llvm::None);
}
::llvm::Optional<Opcode> symbolizeOpcode(uint32_t value) {
  switch (value) {
  case 0: return Opcode::OpNop;
  case 1: return Opcode::OpUndef;
  case 2: return Opcode::OpSourceContinued;
  case 3: return Opcode::OpSource;
  case 4: return Opcode::OpSourceExtension;
  case 5: return Opcode::OpName;
  case 6: return Opcode::OpMemberName;
  case 7: return Opcode::OpString;
  case 8: return Opcode::OpLine;
  case 10: return Opcode::OpExtension;
  case 11: return Opcode::OpExtInstImport;
  case 12: return Opcode::OpExtInst;
  case 14: return Opcode::OpMemoryModel;
  case 15: return Opcode::OpEntryPoint;
  case 16: return Opcode::OpExecutionMode;
  case 17: return Opcode::OpCapability;
  case 19: return Opcode::OpTypeVoid;
  case 20: return Opcode::OpTypeBool;
  case 21: return Opcode::OpTypeInt;
  case 22: return Opcode::OpTypeFloat;
  case 23: return Opcode::OpTypeVector;
  case 24: return Opcode::OpTypeMatrix;
  case 25: return Opcode::OpTypeImage;
  case 27: return Opcode::OpTypeSampledImage;
  case 28: return Opcode::OpTypeArray;
  case 29: return Opcode::OpTypeRuntimeArray;
  case 30: return Opcode::OpTypeStruct;
  case 32: return Opcode::OpTypePointer;
  case 33: return Opcode::OpTypeFunction;
  case 39: return Opcode::OpTypeForwardPointer;
  case 41: return Opcode::OpConstantTrue;
  case 42: return Opcode::OpConstantFalse;
  case 43: return Opcode::OpConstant;
  case 44: return Opcode::OpConstantComposite;
  case 46: return Opcode::OpConstantNull;
  case 48: return Opcode::OpSpecConstantTrue;
  case 49: return Opcode::OpSpecConstantFalse;
  case 50: return Opcode::OpSpecConstant;
  case 51: return Opcode::OpSpecConstantComposite;
  case 52: return Opcode::OpSpecConstantOp;
  case 54: return Opcode::OpFunction;
  case 55: return Opcode::OpFunctionParameter;
  case 56: return Opcode::OpFunctionEnd;
  case 57: return Opcode::OpFunctionCall;
  case 59: return Opcode::OpVariable;
  case 61: return Opcode::OpLoad;
  case 62: return Opcode::OpStore;
  case 63: return Opcode::OpCopyMemory;
  case 65: return Opcode::OpAccessChain;
  case 71: return Opcode::OpDecorate;
  case 72: return Opcode::OpMemberDecorate;
  case 77: return Opcode::OpVectorExtractDynamic;
  case 78: return Opcode::OpVectorInsertDynamic;
  case 79: return Opcode::OpVectorShuffle;
  case 80: return Opcode::OpCompositeConstruct;
  case 81: return Opcode::OpCompositeExtract;
  case 82: return Opcode::OpCompositeInsert;
  case 84: return Opcode::OpTranspose;
  case 100: return Opcode::OpImage;
  case 109: return Opcode::OpConvertFToU;
  case 110: return Opcode::OpConvertFToS;
  case 111: return Opcode::OpConvertSToF;
  case 112: return Opcode::OpConvertUToF;
  case 113: return Opcode::OpUConvert;
  case 114: return Opcode::OpSConvert;
  case 115: return Opcode::OpFConvert;
  case 124: return Opcode::OpBitcast;
  case 126: return Opcode::OpSNegate;
  case 127: return Opcode::OpFNegate;
  case 128: return Opcode::OpIAdd;
  case 129: return Opcode::OpFAdd;
  case 130: return Opcode::OpISub;
  case 131: return Opcode::OpFSub;
  case 132: return Opcode::OpIMul;
  case 133: return Opcode::OpFMul;
  case 134: return Opcode::OpUDiv;
  case 135: return Opcode::OpSDiv;
  case 136: return Opcode::OpFDiv;
  case 137: return Opcode::OpUMod;
  case 138: return Opcode::OpSRem;
  case 139: return Opcode::OpSMod;
  case 140: return Opcode::OpFRem;
  case 141: return Opcode::OpFMod;
  case 143: return Opcode::OpMatrixTimesScalar;
  case 146: return Opcode::OpMatrixTimesMatrix;
  case 156: return Opcode::OpIsNan;
  case 157: return Opcode::OpIsInf;
  case 162: return Opcode::OpOrdered;
  case 163: return Opcode::OpUnordered;
  case 164: return Opcode::OpLogicalEqual;
  case 165: return Opcode::OpLogicalNotEqual;
  case 166: return Opcode::OpLogicalOr;
  case 167: return Opcode::OpLogicalAnd;
  case 168: return Opcode::OpLogicalNot;
  case 169: return Opcode::OpSelect;
  case 170: return Opcode::OpIEqual;
  case 171: return Opcode::OpINotEqual;
  case 172: return Opcode::OpUGreaterThan;
  case 173: return Opcode::OpSGreaterThan;
  case 174: return Opcode::OpUGreaterThanEqual;
  case 175: return Opcode::OpSGreaterThanEqual;
  case 176: return Opcode::OpULessThan;
  case 177: return Opcode::OpSLessThan;
  case 178: return Opcode::OpULessThanEqual;
  case 179: return Opcode::OpSLessThanEqual;
  case 180: return Opcode::OpFOrdEqual;
  case 181: return Opcode::OpFUnordEqual;
  case 182: return Opcode::OpFOrdNotEqual;
  case 183: return Opcode::OpFUnordNotEqual;
  case 184: return Opcode::OpFOrdLessThan;
  case 185: return Opcode::OpFUnordLessThan;
  case 186: return Opcode::OpFOrdGreaterThan;
  case 187: return Opcode::OpFUnordGreaterThan;
  case 188: return Opcode::OpFOrdLessThanEqual;
  case 189: return Opcode::OpFUnordLessThanEqual;
  case 190: return Opcode::OpFOrdGreaterThanEqual;
  case 191: return Opcode::OpFUnordGreaterThanEqual;
  case 194: return Opcode::OpShiftRightLogical;
  case 195: return Opcode::OpShiftRightArithmetic;
  case 196: return Opcode::OpShiftLeftLogical;
  case 197: return Opcode::OpBitwiseOr;
  case 198: return Opcode::OpBitwiseXor;
  case 199: return Opcode::OpBitwiseAnd;
  case 200: return Opcode::OpNot;
  case 201: return Opcode::OpBitFieldInsert;
  case 202: return Opcode::OpBitFieldSExtract;
  case 203: return Opcode::OpBitFieldUExtract;
  case 204: return Opcode::OpBitReverse;
  case 205: return Opcode::OpBitCount;
  case 224: return Opcode::OpControlBarrier;
  case 225: return Opcode::OpMemoryBarrier;
  case 231: return Opcode::OpAtomicCompareExchangeWeak;
  case 232: return Opcode::OpAtomicIIncrement;
  case 233: return Opcode::OpAtomicIDecrement;
  case 234: return Opcode::OpAtomicIAdd;
  case 235: return Opcode::OpAtomicISub;
  case 236: return Opcode::OpAtomicSMin;
  case 237: return Opcode::OpAtomicUMin;
  case 238: return Opcode::OpAtomicSMax;
  case 239: return Opcode::OpAtomicUMax;
  case 240: return Opcode::OpAtomicAnd;
  case 241: return Opcode::OpAtomicOr;
  case 242: return Opcode::OpAtomicXor;
  case 245: return Opcode::OpPhi;
  case 246: return Opcode::OpLoopMerge;
  case 247: return Opcode::OpSelectionMerge;
  case 248: return Opcode::OpLabel;
  case 249: return Opcode::OpBranch;
  case 250: return Opcode::OpBranchConditional;
  case 253: return Opcode::OpReturn;
  case 254: return Opcode::OpReturnValue;
  case 255: return Opcode::OpUnreachable;
  case 263: return Opcode::OpGroupBroadcast;
  case 317: return Opcode::OpNoLine;
  case 330: return Opcode::OpModuleProcessed;
  case 333: return Opcode::OpGroupNonUniformElect;
  case 337: return Opcode::OpGroupNonUniformBroadcast;
  case 339: return Opcode::OpGroupNonUniformBallot;
  case 349: return Opcode::OpGroupNonUniformIAdd;
  case 350: return Opcode::OpGroupNonUniformFAdd;
  case 351: return Opcode::OpGroupNonUniformIMul;
  case 352: return Opcode::OpGroupNonUniformFMul;
  case 353: return Opcode::OpGroupNonUniformSMin;
  case 354: return Opcode::OpGroupNonUniformUMin;
  case 355: return Opcode::OpGroupNonUniformFMin;
  case 356: return Opcode::OpGroupNonUniformSMax;
  case 357: return Opcode::OpGroupNonUniformUMax;
  case 358: return Opcode::OpGroupNonUniformFMax;
  case 4421: return Opcode::OpSubgroupBallotKHR;
  case 5358: return Opcode::OpTypeCooperativeMatrixNV;
  case 5359: return Opcode::OpCooperativeMatrixLoadNV;
  case 5360: return Opcode::OpCooperativeMatrixStoreNV;
  case 5361: return Opcode::OpCooperativeMatrixMulAddNV;
  case 5362: return Opcode::OpCooperativeMatrixLengthNV;
  case 5575: return Opcode::OpSubgroupBlockReadINTEL;
  case 5576: return Opcode::OpSubgroupBlockWriteINTEL;
  default: return ::llvm::None;
  }
}

bool OpcodeAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 21)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 22)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 23)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 24)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 25)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 27)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 28)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 29)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 30)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 32)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 33)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 39)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 41)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 42)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 43)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 44)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 46)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 48)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 49)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 50)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 51)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 52)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 54)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 55)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 56)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 57)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 59)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 61)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 62)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 63)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 65)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 71)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 72)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 77)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 78)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 79)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 80)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 81)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 82)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 84)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 100)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 109)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 110)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 111)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 112)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 113)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 114)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 115)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 124)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 126)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 127)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 128)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 129)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 130)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 131)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 132)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 133)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 134)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 135)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 136)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 137)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 138)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 139)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 140)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 141)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 143)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 146)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 156)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 157)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 162)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 163)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 164)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 165)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 166)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 167)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 168)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 169)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 170)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 171)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 172)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 173)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 174)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 175)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 176)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 177)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 178)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 179)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 180)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 181)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 182)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 183)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 184)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 185)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 186)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 187)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 188)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 189)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 190)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 191)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 194)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 195)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 196)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 197)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 198)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 199)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 200)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 201)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 202)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 203)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 204)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 205)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 224)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 225)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 231)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 232)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 233)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 234)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 235)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 236)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 237)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 238)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 239)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 240)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 241)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 242)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 245)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 246)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 247)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 248)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 249)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 250)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 253)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 254)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 255)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 263)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 317)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 330)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 333)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 337)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 339)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 349)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 350)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 351)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 352)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 353)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 354)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 355)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 356)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 357)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 358)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4421)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5358)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5359)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5360)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5361)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5362)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5575)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5576)));
}
OpcodeAttr OpcodeAttr::get(::mlir::MLIRContext *context, Opcode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<OpcodeAttr>();
}
Opcode OpcodeAttr::getValue() const {
  return static_cast<Opcode>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyImageSamplerUseInfo(ImageSamplerUseInfo val) {
  switch (val) {
    case ImageSamplerUseInfo::SamplerUnknown: return "SamplerUnknown";
    case ImageSamplerUseInfo::NeedSampler: return "NeedSampler";
    case ImageSamplerUseInfo::NoSampler: return "NoSampler";
  }
  return "";
}

::llvm::Optional<ImageSamplerUseInfo> symbolizeImageSamplerUseInfo(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ImageSamplerUseInfo>>(str)
      .Case("SamplerUnknown", ImageSamplerUseInfo::SamplerUnknown)
      .Case("NeedSampler", ImageSamplerUseInfo::NeedSampler)
      .Case("NoSampler", ImageSamplerUseInfo::NoSampler)
      .Default(::llvm::None);
}
::llvm::Optional<ImageSamplerUseInfo> symbolizeImageSamplerUseInfo(uint32_t value) {
  switch (value) {
  case 0: return ImageSamplerUseInfo::SamplerUnknown;
  case 1: return ImageSamplerUseInfo::NeedSampler;
  case 2: return ImageSamplerUseInfo::NoSampler;
  default: return ::llvm::None;
  }
}

bool ImageSamplerUseInfoAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
ImageSamplerUseInfoAttr ImageSamplerUseInfoAttr::get(::mlir::MLIRContext *context, ImageSamplerUseInfo val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ImageSamplerUseInfoAttr>();
}
ImageSamplerUseInfo ImageSamplerUseInfoAttr::getValue() const {
  return static_cast<ImageSamplerUseInfo>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyImageSamplingInfo(ImageSamplingInfo val) {
  switch (val) {
    case ImageSamplingInfo::SingleSampled: return "SingleSampled";
    case ImageSamplingInfo::MultiSampled: return "MultiSampled";
  }
  return "";
}

::llvm::Optional<ImageSamplingInfo> symbolizeImageSamplingInfo(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ImageSamplingInfo>>(str)
      .Case("SingleSampled", ImageSamplingInfo::SingleSampled)
      .Case("MultiSampled", ImageSamplingInfo::MultiSampled)
      .Default(::llvm::None);
}
::llvm::Optional<ImageSamplingInfo> symbolizeImageSamplingInfo(uint32_t value) {
  switch (value) {
  case 0: return ImageSamplingInfo::SingleSampled;
  case 1: return ImageSamplingInfo::MultiSampled;
  default: return ::llvm::None;
  }
}

bool ImageSamplingInfoAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)));
}
ImageSamplingInfoAttr ImageSamplingInfoAttr::get(::mlir::MLIRContext *context, ImageSamplingInfo val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ImageSamplingInfoAttr>();
}
ImageSamplingInfo ImageSamplingInfoAttr::getValue() const {
  return static_cast<ImageSamplingInfo>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyScope(Scope val) {
  switch (val) {
    case Scope::CrossDevice: return "CrossDevice";
    case Scope::Device: return "Device";
    case Scope::Workgroup: return "Workgroup";
    case Scope::Subgroup: return "Subgroup";
    case Scope::Invocation: return "Invocation";
    case Scope::QueueFamily: return "QueueFamily";
  }
  return "";
}

::llvm::Optional<Scope> symbolizeScope(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Scope>>(str)
      .Case("CrossDevice", Scope::CrossDevice)
      .Case("Device", Scope::Device)
      .Case("Workgroup", Scope::Workgroup)
      .Case("Subgroup", Scope::Subgroup)
      .Case("Invocation", Scope::Invocation)
      .Case("QueueFamily", Scope::QueueFamily)
      .Default(::llvm::None);
}
::llvm::Optional<Scope> symbolizeScope(uint32_t value) {
  switch (value) {
  case 0: return Scope::CrossDevice;
  case 1: return Scope::Device;
  case 2: return Scope::Workgroup;
  case 3: return Scope::Subgroup;
  case 4: return Scope::Invocation;
  case 5: return Scope::QueueFamily;
  default: return ::llvm::None;
  }
}

bool ScopeAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)));
}
ScopeAttr ScopeAttr::get(::mlir::MLIRContext *context, Scope val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<ScopeAttr>();
}
Scope ScopeAttr::getValue() const {
  return static_cast<Scope>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
std::string stringifySelectionControl(SelectionControl symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Flatten"); val &= ~1u; }
  if (2u & val) { strs.push_back("DontFlatten"); val &= ~2u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<SelectionControl> symbolizeSelectionControl(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return SelectionControl::None;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("Flatten", 1)
      .Case("DontFlatten", 2)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<SelectionControl>(val);
}

::llvm::Optional<SelectionControl> symbolizeSelectionControl(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return SelectionControl::None;

  if (value & ~(1u | 2u)) return llvm::None;
  return static_cast<SelectionControl>(value);
}
bool SelectionControlAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(0u|1u|2u)))));
}
SelectionControlAttr SelectionControlAttr::get(::mlir::MLIRContext *context, SelectionControl val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<SelectionControlAttr>();
}
SelectionControl SelectionControlAttr::getValue() const {
  return static_cast<SelectionControl>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyStorageClass(StorageClass val) {
  switch (val) {
    case StorageClass::UniformConstant: return "UniformConstant";
    case StorageClass::Input: return "Input";
    case StorageClass::Uniform: return "Uniform";
    case StorageClass::Output: return "Output";
    case StorageClass::Workgroup: return "Workgroup";
    case StorageClass::CrossWorkgroup: return "CrossWorkgroup";
    case StorageClass::Private: return "Private";
    case StorageClass::Function: return "Function";
    case StorageClass::Generic: return "Generic";
    case StorageClass::PushConstant: return "PushConstant";
    case StorageClass::AtomicCounter: return "AtomicCounter";
    case StorageClass::Image: return "Image";
    case StorageClass::StorageBuffer: return "StorageBuffer";
    case StorageClass::CallableDataNV: return "CallableDataNV";
    case StorageClass::IncomingCallableDataNV: return "IncomingCallableDataNV";
    case StorageClass::RayPayloadNV: return "RayPayloadNV";
    case StorageClass::HitAttributeNV: return "HitAttributeNV";
    case StorageClass::IncomingRayPayloadNV: return "IncomingRayPayloadNV";
    case StorageClass::ShaderRecordBufferNV: return "ShaderRecordBufferNV";
    case StorageClass::PhysicalStorageBuffer: return "PhysicalStorageBuffer";
  }
  return "";
}

::llvm::Optional<StorageClass> symbolizeStorageClass(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<StorageClass>>(str)
      .Case("UniformConstant", StorageClass::UniformConstant)
      .Case("Input", StorageClass::Input)
      .Case("Uniform", StorageClass::Uniform)
      .Case("Output", StorageClass::Output)
      .Case("Workgroup", StorageClass::Workgroup)
      .Case("CrossWorkgroup", StorageClass::CrossWorkgroup)
      .Case("Private", StorageClass::Private)
      .Case("Function", StorageClass::Function)
      .Case("Generic", StorageClass::Generic)
      .Case("PushConstant", StorageClass::PushConstant)
      .Case("AtomicCounter", StorageClass::AtomicCounter)
      .Case("Image", StorageClass::Image)
      .Case("StorageBuffer", StorageClass::StorageBuffer)
      .Case("CallableDataNV", StorageClass::CallableDataNV)
      .Case("IncomingCallableDataNV", StorageClass::IncomingCallableDataNV)
      .Case("RayPayloadNV", StorageClass::RayPayloadNV)
      .Case("HitAttributeNV", StorageClass::HitAttributeNV)
      .Case("IncomingRayPayloadNV", StorageClass::IncomingRayPayloadNV)
      .Case("ShaderRecordBufferNV", StorageClass::ShaderRecordBufferNV)
      .Case("PhysicalStorageBuffer", StorageClass::PhysicalStorageBuffer)
      .Default(::llvm::None);
}
::llvm::Optional<StorageClass> symbolizeStorageClass(uint32_t value) {
  switch (value) {
  case 0: return StorageClass::UniformConstant;
  case 1: return StorageClass::Input;
  case 2: return StorageClass::Uniform;
  case 3: return StorageClass::Output;
  case 4: return StorageClass::Workgroup;
  case 5: return StorageClass::CrossWorkgroup;
  case 6: return StorageClass::Private;
  case 7: return StorageClass::Function;
  case 8: return StorageClass::Generic;
  case 9: return StorageClass::PushConstant;
  case 10: return StorageClass::AtomicCounter;
  case 11: return StorageClass::Image;
  case 12: return StorageClass::StorageBuffer;
  case 5328: return StorageClass::CallableDataNV;
  case 5329: return StorageClass::IncomingCallableDataNV;
  case 5338: return StorageClass::RayPayloadNV;
  case 5339: return StorageClass::HitAttributeNV;
  case 5342: return StorageClass::IncomingRayPayloadNV;
  case 5343: return StorageClass::ShaderRecordBufferNV;
  case 5349: return StorageClass::PhysicalStorageBuffer;
  default: return ::llvm::None;
  }
}

bool StorageClassAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5328)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5329)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5338)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5339)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5342)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5343)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5349)));
}
StorageClassAttr StorageClassAttr::get(::mlir::MLIRContext *context, StorageClass val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<StorageClassAttr>();
}
StorageClass StorageClassAttr::getValue() const {
  return static_cast<StorageClass>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyVendor(Vendor val) {
  switch (val) {
    case Vendor::AMD: return "AMD";
    case Vendor::ARM: return "ARM";
    case Vendor::Imagination: return "Imagination";
    case Vendor::Intel: return "Intel";
    case Vendor::NVIDIA: return "NVIDIA";
    case Vendor::Qualcomm: return "Qualcomm";
    case Vendor::SwiftShader: return "SwiftShader";
    case Vendor::Unknown: return "Unknown";
  }
  return "";
}

::llvm::Optional<Vendor> symbolizeVendor(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Vendor>>(str)
      .Case("AMD", Vendor::AMD)
      .Case("ARM", Vendor::ARM)
      .Case("Imagination", Vendor::Imagination)
      .Case("Intel", Vendor::Intel)
      .Case("NVIDIA", Vendor::NVIDIA)
      .Case("Qualcomm", Vendor::Qualcomm)
      .Case("SwiftShader", Vendor::SwiftShader)
      .Case("Unknown", Vendor::Unknown)
      .Default(::llvm::None);
}
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
::llvm::StringRef stringifyVersion(Version val) {
  switch (val) {
    case Version::V_1_0: return "v1.0";
    case Version::V_1_1: return "v1.1";
    case Version::V_1_2: return "v1.2";
    case Version::V_1_3: return "v1.3";
    case Version::V_1_4: return "v1.4";
    case Version::V_1_5: return "v1.5";
  }
  return "";
}

::llvm::Optional<Version> symbolizeVersion(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Version>>(str)
      .Case("v1.0", Version::V_1_0)
      .Case("v1.1", Version::V_1_1)
      .Case("v1.2", Version::V_1_2)
      .Case("v1.3", Version::V_1_3)
      .Case("v1.4", Version::V_1_4)
      .Case("v1.5", Version::V_1_5)
      .Default(::llvm::None);
}
::llvm::Optional<Version> symbolizeVersion(uint32_t value) {
  switch (value) {
  case 0: return Version::V_1_0;
  case 1: return Version::V_1_1;
  case 2: return Version::V_1_2;
  case 3: return Version::V_1_3;
  case 4: return Version::V_1_4;
  case 5: return Version::V_1_5;
  default: return ::llvm::None;
  }
}

bool VersionAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)));
}
VersionAttr VersionAttr::get(::mlir::MLIRContext *context, Version val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<VersionAttr>();
}
Version VersionAttr::getValue() const {
  return static_cast<Version>(::mlir::IntegerAttr::getInt());
}
} // namespace spirv
} // namespace mlir

