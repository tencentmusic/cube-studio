/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace spirv {
// valid SPIR-V AddressingModel
enum class AddressingModel : uint32_t {
  Logical = 0,
  Physical32 = 1,
  Physical64 = 2,
  PhysicalStorageBuffer64 = 5348,
};

::llvm::Optional<AddressingModel> symbolizeAddressingModel(uint32_t);
::llvm::StringRef stringifyAddressingModel(AddressingModel);
::llvm::Optional<AddressingModel> symbolizeAddressingModel(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAddressingModel() {
  return 5348;
}


inline ::llvm::StringRef stringifyEnum(AddressingModel enumValue) {
  return stringifyAddressingModel(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AddressingModel> symbolizeEnum<AddressingModel>(::llvm::StringRef str) {
  return symbolizeAddressingModel(str);
}

class AddressingModelAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AddressingModel;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AddressingModelAttr get(::mlir::MLIRContext *context, AddressingModel val);
  AddressingModel getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::AddressingModel> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::AddressingModel getEmptyKey() {
    return static_cast<::mlir::spirv::AddressingModel>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::AddressingModel getTombstoneKey() {
    return static_cast<::mlir::spirv::AddressingModel>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::AddressingModel &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::AddressingModel &lhs, const ::mlir::spirv::AddressingModel &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Image Arrayed specification
enum class ImageArrayedInfo : uint32_t {
  NonArrayed = 0,
  Arrayed = 1,
};

::llvm::Optional<ImageArrayedInfo> symbolizeImageArrayedInfo(uint32_t);
::llvm::StringRef stringifyImageArrayedInfo(ImageArrayedInfo);
::llvm::Optional<ImageArrayedInfo> symbolizeImageArrayedInfo(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForImageArrayedInfo() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ImageArrayedInfo enumValue) {
  return stringifyImageArrayedInfo(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ImageArrayedInfo> symbolizeEnum<ImageArrayedInfo>(::llvm::StringRef str) {
  return symbolizeImageArrayedInfo(str);
}

class ImageArrayedInfoAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ImageArrayedInfo;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ImageArrayedInfoAttr get(::mlir::MLIRContext *context, ImageArrayedInfo val);
  ImageArrayedInfo getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ImageArrayedInfo> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ImageArrayedInfo getEmptyKey() {
    return static_cast<::mlir::spirv::ImageArrayedInfo>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ImageArrayedInfo getTombstoneKey() {
    return static_cast<::mlir::spirv::ImageArrayedInfo>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ImageArrayedInfo &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ImageArrayedInfo &lhs, const ::mlir::spirv::ImageArrayedInfo &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V BuiltIn
enum class BuiltIn : uint32_t {
  Position = 0,
  PointSize = 1,
  ClipDistance = 3,
  CullDistance = 4,
  VertexId = 5,
  InstanceId = 6,
  PrimitiveId = 7,
  InvocationId = 8,
  Layer = 9,
  ViewportIndex = 10,
  TessLevelOuter = 11,
  TessLevelInner = 12,
  TessCoord = 13,
  PatchVertices = 14,
  FragCoord = 15,
  PointCoord = 16,
  FrontFacing = 17,
  SampleId = 18,
  SamplePosition = 19,
  SampleMask = 20,
  FragDepth = 22,
  HelperInvocation = 23,
  NumWorkgroups = 24,
  WorkgroupSize = 25,
  WorkgroupId = 26,
  LocalInvocationId = 27,
  GlobalInvocationId = 28,
  LocalInvocationIndex = 29,
  WorkDim = 30,
  GlobalSize = 31,
  EnqueuedWorkgroupSize = 32,
  GlobalOffset = 33,
  GlobalLinearId = 34,
  SubgroupSize = 36,
  SubgroupMaxSize = 37,
  NumSubgroups = 38,
  NumEnqueuedSubgroups = 39,
  SubgroupId = 40,
  SubgroupLocalInvocationId = 41,
  VertexIndex = 42,
  InstanceIndex = 43,
  SubgroupEqMask = 4416,
  SubgroupGeMask = 4417,
  SubgroupGtMask = 4418,
  SubgroupLeMask = 4419,
  SubgroupLtMask = 4420,
  BaseVertex = 4424,
  BaseInstance = 4425,
  DrawIndex = 4426,
  DeviceIndex = 4438,
  ViewIndex = 4440,
  BaryCoordNoPerspAMD = 4992,
  BaryCoordNoPerspCentroidAMD = 4993,
  BaryCoordNoPerspSampleAMD = 4994,
  BaryCoordSmoothAMD = 4995,
  BaryCoordSmoothCentroidAMD = 4996,
  BaryCoordSmoothSampleAMD = 4997,
  BaryCoordPullModelAMD = 4998,
  FragStencilRefEXT = 5014,
  ViewportMaskNV = 5253,
  SecondaryPositionNV = 5257,
  SecondaryViewportMaskNV = 5258,
  PositionPerViewNV = 5261,
  ViewportMaskPerViewNV = 5262,
  FullyCoveredEXT = 5264,
  TaskCountNV = 5274,
  PrimitiveCountNV = 5275,
  PrimitiveIndicesNV = 5276,
  ClipDistancePerViewNV = 5277,
  CullDistancePerViewNV = 5278,
  LayerPerViewNV = 5279,
  MeshViewCountNV = 5280,
  MeshViewIndicesNV = 5281,
  BaryCoordNV = 5286,
  BaryCoordNoPerspNV = 5287,
  FragSizeEXT = 5292,
  FragInvocationCountEXT = 5293,
  LaunchIdNV = 5319,
  LaunchSizeNV = 5320,
  WorldRayOriginNV = 5321,
  WorldRayDirectionNV = 5322,
  ObjectRayOriginNV = 5323,
  ObjectRayDirectionNV = 5324,
  RayTminNV = 5325,
  RayTmaxNV = 5326,
  InstanceCustomIndexNV = 5327,
  ObjectToWorldNV = 5330,
  WorldToObjectNV = 5331,
  HitTNV = 5332,
  HitKindNV = 5333,
  IncomingRayFlagsNV = 5351,
  WarpsPerSMNV = 5374,
  SMCountNV = 5375,
  WarpIDNV = 5376,
  SMIDNV = 5377,
};

::llvm::Optional<BuiltIn> symbolizeBuiltIn(uint32_t);
::llvm::StringRef stringifyBuiltIn(BuiltIn);
::llvm::Optional<BuiltIn> symbolizeBuiltIn(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBuiltIn() {
  return 5377;
}


inline ::llvm::StringRef stringifyEnum(BuiltIn enumValue) {
  return stringifyBuiltIn(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<BuiltIn> symbolizeEnum<BuiltIn>(::llvm::StringRef str) {
  return symbolizeBuiltIn(str);
}

class BuiltInAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = BuiltIn;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static BuiltInAttr get(::mlir::MLIRContext *context, BuiltIn val);
  BuiltIn getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::BuiltIn> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::BuiltIn getEmptyKey() {
    return static_cast<::mlir::spirv::BuiltIn>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::BuiltIn getTombstoneKey() {
    return static_cast<::mlir::spirv::BuiltIn>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::BuiltIn &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::BuiltIn &lhs, const ::mlir::spirv::BuiltIn &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Capability
enum class Capability : uint32_t {
  Matrix = 0,
  Addresses = 4,
  Linkage = 5,
  Kernel = 6,
  Float16 = 9,
  Float64 = 10,
  Int64 = 11,
  Groups = 18,
  Int16 = 22,
  Int8 = 39,
  Sampled1D = 43,
  SampledBuffer = 46,
  GroupNonUniform = 61,
  ShaderLayer = 69,
  ShaderViewportIndex = 70,
  SubgroupBallotKHR = 4423,
  SubgroupVoteKHR = 4431,
  StorageBuffer16BitAccess = 4433,
  StoragePushConstant16 = 4435,
  StorageInputOutput16 = 4436,
  DeviceGroup = 4437,
  AtomicStorageOps = 4445,
  SampleMaskPostDepthCoverage = 4447,
  StorageBuffer8BitAccess = 4448,
  StoragePushConstant8 = 4450,
  DenormPreserve = 4464,
  DenormFlushToZero = 4465,
  SignedZeroInfNanPreserve = 4466,
  RoundingModeRTE = 4467,
  RoundingModeRTZ = 4468,
  ImageFootprintNV = 5282,
  FragmentBarycentricNV = 5284,
  ComputeDerivativeGroupQuadsNV = 5288,
  GroupNonUniformPartitionedNV = 5297,
  VulkanMemoryModel = 5345,
  VulkanMemoryModelDeviceScope = 5346,
  ComputeDerivativeGroupLinearNV = 5350,
  SubgroupShuffleINTEL = 5568,
  SubgroupBufferBlockIOINTEL = 5569,
  SubgroupImageBlockIOINTEL = 5570,
  SubgroupImageMediaBlockIOINTEL = 5579,
  SubgroupAvcMotionEstimationINTEL = 5696,
  SubgroupAvcMotionEstimationIntraINTEL = 5697,
  SubgroupAvcMotionEstimationChromaINTEL = 5698,
  Shader = 1,
  Vector16 = 7,
  Float16Buffer = 8,
  Int64Atomics = 12,
  ImageBasic = 13,
  Pipes = 17,
  DeviceEnqueue = 19,
  LiteralSampler = 20,
  GenericPointer = 38,
  Image1D = 44,
  ImageBuffer = 47,
  NamedBarrier = 59,
  GroupNonUniformVote = 62,
  GroupNonUniformArithmetic = 63,
  GroupNonUniformBallot = 64,
  GroupNonUniformShuffle = 65,
  GroupNonUniformShuffleRelative = 66,
  GroupNonUniformClustered = 67,
  GroupNonUniformQuad = 68,
  StorageUniform16 = 4434,
  UniformAndStorageBuffer8BitAccess = 4449,
  UniformTexelBufferArrayDynamicIndexing = 5304,
  Geometry = 2,
  Tessellation = 3,
  ImageReadWrite = 14,
  ImageMipmap = 15,
  AtomicStorage = 21,
  ImageGatherExtended = 25,
  StorageImageMultisample = 27,
  UniformBufferArrayDynamicIndexing = 28,
  SampledImageArrayDynamicIndexing = 29,
  StorageBufferArrayDynamicIndexing = 30,
  StorageImageArrayDynamicIndexing = 31,
  ClipDistance = 32,
  CullDistance = 33,
  SampleRateShading = 35,
  SampledRect = 37,
  InputAttachment = 40,
  SparseResidency = 41,
  MinLod = 42,
  SampledCubeArray = 45,
  ImageMSArray = 48,
  StorageImageExtendedFormats = 49,
  ImageQuery = 50,
  DerivativeControl = 51,
  InterpolationFunction = 52,
  TransformFeedback = 53,
  StorageImageReadWithoutFormat = 55,
  StorageImageWriteWithoutFormat = 56,
  SubgroupDispatch = 58,
  PipeStorage = 60,
  DrawParameters = 4427,
  MultiView = 4439,
  VariablePointersStorageBuffer = 4441,
  Float16ImageAMD = 5008,
  ImageGatherBiasLodAMD = 5009,
  FragmentMaskAMD = 5010,
  StencilExportEXT = 5013,
  ImageReadWriteLodAMD = 5015,
  ShaderClockKHR = 5055,
  FragmentFullyCoveredEXT = 5265,
  MeshShadingNV = 5266,
  FragmentDensityEXT = 5291,
  ShaderNonUniform = 5301,
  RuntimeDescriptorArray = 5302,
  StorageTexelBufferArrayDynamicIndexing = 5305,
  RayTracingNV = 5340,
  PhysicalStorageBufferAddresses = 5347,
  CooperativeMatrixNV = 5357,
  FragmentShaderSampleInterlockEXT = 5363,
  FragmentShaderShadingRateInterlockEXT = 5372,
  ShaderSMBuiltinsNV = 5373,
  FragmentShaderPixelInterlockEXT = 5378,
  DemoteToHelperInvocationEXT = 5379,
  IntegerFunctions2INTEL = 5584,
  TessellationPointSize = 23,
  GeometryPointSize = 24,
  ImageCubeArray = 34,
  ImageRect = 36,
  GeometryStreams = 54,
  MultiViewport = 57,
  VariablePointers = 4442,
  SampleMaskOverrideCoverageNV = 5249,
  GeometryShaderPassthroughNV = 5251,
  PerViewAttributesNV = 5260,
  InputAttachmentArrayDynamicIndexing = 5303,
  UniformBufferArrayNonUniformIndexing = 5306,
  SampledImageArrayNonUniformIndexing = 5307,
  StorageBufferArrayNonUniformIndexing = 5308,
  StorageImageArrayNonUniformIndexing = 5309,
  InputAttachmentArrayNonUniformIndexing = 5310,
  UniformTexelBufferArrayNonUniformIndexing = 5311,
  StorageTexelBufferArrayNonUniformIndexing = 5312,
  ShaderViewportIndexLayerEXT = 5254,
  ShaderViewportMaskNV = 5255,
  ShaderStereoViewNV = 5259,
};

::llvm::Optional<Capability> symbolizeCapability(uint32_t);
::llvm::StringRef stringifyCapability(Capability);
::llvm::Optional<Capability> symbolizeCapability(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCapability() {
  return 5698;
}


inline ::llvm::StringRef stringifyEnum(Capability enumValue) {
  return stringifyCapability(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Capability> symbolizeEnum<Capability>(::llvm::StringRef str) {
  return symbolizeCapability(str);
}

class CapabilityAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Capability;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CapabilityAttr get(::mlir::MLIRContext *context, Capability val);
  Capability getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Capability> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::Capability getEmptyKey() {
    return static_cast<::mlir::spirv::Capability>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Capability getTombstoneKey() {
    return static_cast<::mlir::spirv::Capability>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Capability &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::Capability &lhs, const ::mlir::spirv::Capability &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Decoration
enum class Decoration : uint32_t {
  RelaxedPrecision = 0,
  SpecId = 1,
  Block = 2,
  BufferBlock = 3,
  RowMajor = 4,
  ColMajor = 5,
  ArrayStride = 6,
  MatrixStride = 7,
  GLSLShared = 8,
  GLSLPacked = 9,
  CPacked = 10,
  BuiltIn = 11,
  NoPerspective = 13,
  Flat = 14,
  Patch = 15,
  Centroid = 16,
  Sample = 17,
  Invariant = 18,
  Restrict = 19,
  Aliased = 20,
  Volatile = 21,
  Constant = 22,
  Coherent = 23,
  NonWritable = 24,
  NonReadable = 25,
  Uniform = 26,
  UniformId = 27,
  SaturatedConversion = 28,
  Stream = 29,
  Location = 30,
  Component = 31,
  Index = 32,
  Binding = 33,
  DescriptorSet = 34,
  Offset = 35,
  XfbBuffer = 36,
  XfbStride = 37,
  FuncParamAttr = 38,
  FPRoundingMode = 39,
  FPFastMathMode = 40,
  LinkageAttributes = 41,
  NoContraction = 42,
  InputAttachmentIndex = 43,
  Alignment = 44,
  MaxByteOffset = 45,
  AlignmentId = 46,
  MaxByteOffsetId = 47,
  NoSignedWrap = 4469,
  NoUnsignedWrap = 4470,
  ExplicitInterpAMD = 4999,
  OverrideCoverageNV = 5248,
  PassthroughNV = 5250,
  ViewportRelativeNV = 5252,
  SecondaryViewportRelativeNV = 5256,
  PerPrimitiveNV = 5271,
  PerViewNV = 5272,
  PerTaskNV = 5273,
  PerVertexNV = 5285,
  NonUniform = 5300,
  RestrictPointer = 5355,
  AliasedPointer = 5356,
  CounterBuffer = 5634,
  UserSemantic = 5635,
  UserTypeGOOGLE = 5636,
};

::llvm::Optional<Decoration> symbolizeDecoration(uint32_t);
::llvm::StringRef stringifyDecoration(Decoration);
::llvm::Optional<Decoration> symbolizeDecoration(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDecoration() {
  return 5636;
}


inline ::llvm::StringRef stringifyEnum(Decoration enumValue) {
  return stringifyDecoration(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Decoration> symbolizeEnum<Decoration>(::llvm::StringRef str) {
  return symbolizeDecoration(str);
}

class DecorationAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Decoration;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DecorationAttr get(::mlir::MLIRContext *context, Decoration val);
  Decoration getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Decoration> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::Decoration getEmptyKey() {
    return static_cast<::mlir::spirv::Decoration>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Decoration getTombstoneKey() {
    return static_cast<::mlir::spirv::Decoration>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Decoration &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::Decoration &lhs, const ::mlir::spirv::Decoration &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Image Depth specification
enum class ImageDepthInfo : uint32_t {
  NoDepth = 0,
  IsDepth = 1,
  DepthUnknown = 2,
};

::llvm::Optional<ImageDepthInfo> symbolizeImageDepthInfo(uint32_t);
::llvm::StringRef stringifyImageDepthInfo(ImageDepthInfo);
::llvm::Optional<ImageDepthInfo> symbolizeImageDepthInfo(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForImageDepthInfo() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(ImageDepthInfo enumValue) {
  return stringifyImageDepthInfo(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ImageDepthInfo> symbolizeEnum<ImageDepthInfo>(::llvm::StringRef str) {
  return symbolizeImageDepthInfo(str);
}

class ImageDepthInfoAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ImageDepthInfo;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ImageDepthInfoAttr get(::mlir::MLIRContext *context, ImageDepthInfo val);
  ImageDepthInfo getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ImageDepthInfo> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ImageDepthInfo getEmptyKey() {
    return static_cast<::mlir::spirv::ImageDepthInfo>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ImageDepthInfo getTombstoneKey() {
    return static_cast<::mlir::spirv::ImageDepthInfo>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ImageDepthInfo &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ImageDepthInfo &lhs, const ::mlir::spirv::ImageDepthInfo &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V device types
enum class DeviceType {
  Other,
  IntegratedGPU,
  DiscreteGPU,
  CPU,
  Unknown,
};

::llvm::StringRef stringifyDeviceType(DeviceType);
::llvm::Optional<DeviceType> symbolizeDeviceType(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(DeviceType enumValue) {
  return stringifyDeviceType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DeviceType> symbolizeEnum<DeviceType>(::llvm::StringRef str) {
  return symbolizeDeviceType(str);
}
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::DeviceType> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::spirv::DeviceType>::type>;

  static inline ::mlir::spirv::DeviceType getEmptyKey() {
    return static_cast<::mlir::spirv::DeviceType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::DeviceType getTombstoneKey() {
    return static_cast<::mlir::spirv::DeviceType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::DeviceType &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::spirv::DeviceType>::type>(val));
  }

  static bool isEqual(const ::mlir::spirv::DeviceType &lhs, const ::mlir::spirv::DeviceType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Dim
enum class Dim : uint32_t {
  Dim1D = 0,
  Dim2D = 1,
  Dim3D = 2,
  Cube = 3,
  Rect = 4,
  Buffer = 5,
  SubpassData = 6,
};

::llvm::Optional<Dim> symbolizeDim(uint32_t);
::llvm::StringRef stringifyDim(Dim);
::llvm::Optional<Dim> symbolizeDim(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDim() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(Dim enumValue) {
  return stringifyDim(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Dim> symbolizeEnum<Dim>(::llvm::StringRef str) {
  return symbolizeDim(str);
}

class DimAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Dim;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DimAttr get(::mlir::MLIRContext *context, Dim val);
  Dim getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Dim> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::Dim getEmptyKey() {
    return static_cast<::mlir::spirv::Dim>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Dim getTombstoneKey() {
    return static_cast<::mlir::spirv::Dim>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Dim &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::Dim &lhs, const ::mlir::spirv::Dim &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V ExecutionMode
enum class ExecutionMode : uint32_t {
  Invocations = 0,
  SpacingEqual = 1,
  SpacingFractionalEven = 2,
  SpacingFractionalOdd = 3,
  VertexOrderCw = 4,
  VertexOrderCcw = 5,
  PixelCenterInteger = 6,
  OriginUpperLeft = 7,
  OriginLowerLeft = 8,
  EarlyFragmentTests = 9,
  PointMode = 10,
  Xfb = 11,
  DepthReplacing = 12,
  DepthGreater = 14,
  DepthLess = 15,
  DepthUnchanged = 16,
  LocalSize = 17,
  LocalSizeHint = 18,
  InputPoints = 19,
  InputLines = 20,
  InputLinesAdjacency = 21,
  Triangles = 22,
  InputTrianglesAdjacency = 23,
  Quads = 24,
  Isolines = 25,
  OutputVertices = 26,
  OutputPoints = 27,
  OutputLineStrip = 28,
  OutputTriangleStrip = 29,
  VecTypeHint = 30,
  ContractionOff = 31,
  Initializer = 33,
  Finalizer = 34,
  SubgroupSize = 35,
  SubgroupsPerWorkgroup = 36,
  SubgroupsPerWorkgroupId = 37,
  LocalSizeId = 38,
  LocalSizeHintId = 39,
  PostDepthCoverage = 4446,
  DenormPreserve = 4459,
  DenormFlushToZero = 4460,
  SignedZeroInfNanPreserve = 4461,
  RoundingModeRTE = 4462,
  RoundingModeRTZ = 4463,
  StencilRefReplacingEXT = 5027,
  OutputLinesNV = 5269,
  OutputPrimitivesNV = 5270,
  DerivativeGroupQuadsNV = 5289,
  DerivativeGroupLinearNV = 5290,
  OutputTrianglesNV = 5298,
  PixelInterlockOrderedEXT = 5366,
  PixelInterlockUnorderedEXT = 5367,
  SampleInterlockOrderedEXT = 5368,
  SampleInterlockUnorderedEXT = 5369,
  ShadingRateInterlockOrderedEXT = 5370,
  ShadingRateInterlockUnorderedEXT = 5371,
};

::llvm::Optional<ExecutionMode> symbolizeExecutionMode(uint32_t);
::llvm::StringRef stringifyExecutionMode(ExecutionMode);
::llvm::Optional<ExecutionMode> symbolizeExecutionMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForExecutionMode() {
  return 5371;
}


inline ::llvm::StringRef stringifyEnum(ExecutionMode enumValue) {
  return stringifyExecutionMode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ExecutionMode> symbolizeEnum<ExecutionMode>(::llvm::StringRef str) {
  return symbolizeExecutionMode(str);
}

class ExecutionModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ExecutionMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ExecutionModeAttr get(::mlir::MLIRContext *context, ExecutionMode val);
  ExecutionMode getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ExecutionMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ExecutionMode getEmptyKey() {
    return static_cast<::mlir::spirv::ExecutionMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ExecutionMode getTombstoneKey() {
    return static_cast<::mlir::spirv::ExecutionMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ExecutionMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ExecutionMode &lhs, const ::mlir::spirv::ExecutionMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V ExecutionModel
enum class ExecutionModel : uint32_t {
  Vertex = 0,
  TessellationControl = 1,
  TessellationEvaluation = 2,
  Geometry = 3,
  Fragment = 4,
  GLCompute = 5,
  Kernel = 6,
  TaskNV = 5267,
  MeshNV = 5268,
  RayGenerationNV = 5313,
  IntersectionNV = 5314,
  AnyHitNV = 5315,
  ClosestHitNV = 5316,
  MissNV = 5317,
  CallableNV = 5318,
};

::llvm::Optional<ExecutionModel> symbolizeExecutionModel(uint32_t);
::llvm::StringRef stringifyExecutionModel(ExecutionModel);
::llvm::Optional<ExecutionModel> symbolizeExecutionModel(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForExecutionModel() {
  return 5318;
}


inline ::llvm::StringRef stringifyEnum(ExecutionModel enumValue) {
  return stringifyExecutionModel(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ExecutionModel> symbolizeEnum<ExecutionModel>(::llvm::StringRef str) {
  return symbolizeExecutionModel(str);
}

class ExecutionModelAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ExecutionModel;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ExecutionModelAttr get(::mlir::MLIRContext *context, ExecutionModel val);
  ExecutionModel getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ExecutionModel> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ExecutionModel getEmptyKey() {
    return static_cast<::mlir::spirv::ExecutionModel>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ExecutionModel getTombstoneKey() {
    return static_cast<::mlir::spirv::ExecutionModel>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ExecutionModel &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ExecutionModel &lhs, const ::mlir::spirv::ExecutionModel &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// supported SPIR-V extensions
enum class Extension {
  SPV_KHR_16bit_storage,
  SPV_KHR_8bit_storage,
  SPV_KHR_device_group,
  SPV_KHR_float_controls,
  SPV_KHR_physical_storage_buffer,
  SPV_KHR_multiview,
  SPV_KHR_no_integer_wrap_decoration,
  SPV_KHR_post_depth_coverage,
  SPV_KHR_shader_atomic_counter_ops,
  SPV_KHR_shader_ballot,
  SPV_KHR_shader_clock,
  SPV_KHR_shader_draw_parameters,
  SPV_KHR_storage_buffer_storage_class,
  SPV_KHR_subgroup_vote,
  SPV_KHR_variable_pointers,
  SPV_KHR_vulkan_memory_model,
  SPV_EXT_demote_to_helper_invocation,
  SPV_EXT_descriptor_indexing,
  SPV_EXT_fragment_fully_covered,
  SPV_EXT_fragment_invocation_density,
  SPV_EXT_fragment_shader_interlock,
  SPV_EXT_physical_storage_buffer,
  SPV_EXT_shader_stencil_export,
  SPV_EXT_shader_viewport_index_layer,
  SPV_AMD_gpu_shader_half_float_fetch,
  SPV_AMD_shader_ballot,
  SPV_AMD_shader_explicit_vertex_parameter,
  SPV_AMD_shader_fragment_mask,
  SPV_AMD_shader_image_load_store_lod,
  SPV_AMD_texture_gather_bias_lod,
  SPV_GOOGLE_decorate_string,
  SPV_GOOGLE_hlsl_functionality1,
  SPV_GOOGLE_user_type,
  SPV_INTEL_device_side_avc_motion_estimation,
  SPV_INTEL_media_block_io,
  SPV_INTEL_shader_integer_functions2,
  SPV_INTEL_subgroups,
  SPV_NV_compute_shader_derivatives,
  SPV_NV_cooperative_matrix,
  SPV_NV_fragment_shader_barycentric,
  SPV_NV_geometry_shader_passthrough,
  SPV_NV_mesh_shader,
  SPV_NV_ray_tracing,
  SPV_NV_sample_mask_override_coverage,
  SPV_NV_shader_image_footprint,
  SPV_NV_shader_sm_builtins,
  SPV_NV_shader_subgroup_partitioned,
  SPV_NV_shading_rate,
  SPV_NV_stereo_view_rendering,
  SPV_NV_viewport_array2,
  SPV_NVX_multiview_per_view_attributes,
};

::llvm::StringRef stringifyExtension(Extension);
::llvm::Optional<Extension> symbolizeExtension(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(Extension enumValue) {
  return stringifyExtension(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Extension> symbolizeEnum<Extension>(::llvm::StringRef str) {
  return symbolizeExtension(str);
}
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Extension> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::spirv::Extension>::type>;

  static inline ::mlir::spirv::Extension getEmptyKey() {
    return static_cast<::mlir::spirv::Extension>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Extension getTombstoneKey() {
    return static_cast<::mlir::spirv::Extension>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Extension &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::spirv::Extension>::type>(val));
  }

  static bool isEqual(const ::mlir::spirv::Extension &lhs, const ::mlir::spirv::Extension &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V FunctionControl
enum class FunctionControl : uint32_t {
  None = 0,
  Inline = 1,
  DontInline = 2,
  Pure = 4,
  Const = 8,
};

::llvm::Optional<FunctionControl> symbolizeFunctionControl(uint32_t);
std::string stringifyFunctionControl(FunctionControl);
::llvm::Optional<FunctionControl> symbolizeFunctionControl(::llvm::StringRef);
inline FunctionControl operator|(FunctionControl lhs, FunctionControl rhs) {
  return static_cast<FunctionControl>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline FunctionControl operator&(FunctionControl lhs, FunctionControl rhs) {
  return static_cast<FunctionControl>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(FunctionControl bits, FunctionControl bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

inline std::string stringifyEnum(FunctionControl enumValue) {
  return stringifyFunctionControl(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FunctionControl> symbolizeEnum<FunctionControl>(::llvm::StringRef str) {
  return symbolizeFunctionControl(str);
}

class FunctionControlAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FunctionControl;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FunctionControlAttr get(::mlir::MLIRContext *context, FunctionControl val);
  FunctionControl getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::FunctionControl> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::FunctionControl getEmptyKey() {
    return static_cast<::mlir::spirv::FunctionControl>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::FunctionControl getTombstoneKey() {
    return static_cast<::mlir::spirv::FunctionControl>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::FunctionControl &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::FunctionControl &lhs, const ::mlir::spirv::FunctionControl &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V GroupOperation
enum class GroupOperation : uint32_t {
  Reduce = 0,
  InclusiveScan = 1,
  ExclusiveScan = 2,
  ClusteredReduce = 3,
  PartitionedReduceNV = 6,
  PartitionedInclusiveScanNV = 7,
  PartitionedExclusiveScanNV = 8,
};

::llvm::Optional<GroupOperation> symbolizeGroupOperation(uint32_t);
::llvm::StringRef stringifyGroupOperation(GroupOperation);
::llvm::Optional<GroupOperation> symbolizeGroupOperation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForGroupOperation() {
  return 8;
}


inline ::llvm::StringRef stringifyEnum(GroupOperation enumValue) {
  return stringifyGroupOperation(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<GroupOperation> symbolizeEnum<GroupOperation>(::llvm::StringRef str) {
  return symbolizeGroupOperation(str);
}

class GroupOperationAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = GroupOperation;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static GroupOperationAttr get(::mlir::MLIRContext *context, GroupOperation val);
  GroupOperation getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::GroupOperation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::GroupOperation getEmptyKey() {
    return static_cast<::mlir::spirv::GroupOperation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::GroupOperation getTombstoneKey() {
    return static_cast<::mlir::spirv::GroupOperation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::GroupOperation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::GroupOperation &lhs, const ::mlir::spirv::GroupOperation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V ImageFormat
enum class ImageFormat : uint32_t {
  Unknown = 0,
  Rgba32f = 1,
  Rgba16f = 2,
  R32f = 3,
  Rgba8 = 4,
  Rgba8Snorm = 5,
  Rg32f = 6,
  Rg16f = 7,
  R11fG11fB10f = 8,
  R16f = 9,
  Rgba16 = 10,
  Rgb10A2 = 11,
  Rg16 = 12,
  Rg8 = 13,
  R16 = 14,
  R8 = 15,
  Rgba16Snorm = 16,
  Rg16Snorm = 17,
  Rg8Snorm = 18,
  R16Snorm = 19,
  R8Snorm = 20,
  Rgba32i = 21,
  Rgba16i = 22,
  Rgba8i = 23,
  R32i = 24,
  Rg32i = 25,
  Rg16i = 26,
  Rg8i = 27,
  R16i = 28,
  R8i = 29,
  Rgba32ui = 30,
  Rgba16ui = 31,
  Rgba8ui = 32,
  R32ui = 33,
  Rgb10a2ui = 34,
  Rg32ui = 35,
  Rg16ui = 36,
  Rg8ui = 37,
  R16ui = 38,
  R8ui = 39,
};

::llvm::Optional<ImageFormat> symbolizeImageFormat(uint32_t);
::llvm::StringRef stringifyImageFormat(ImageFormat);
::llvm::Optional<ImageFormat> symbolizeImageFormat(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForImageFormat() {
  return 39;
}


inline ::llvm::StringRef stringifyEnum(ImageFormat enumValue) {
  return stringifyImageFormat(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ImageFormat> symbolizeEnum<ImageFormat>(::llvm::StringRef str) {
  return symbolizeImageFormat(str);
}

class ImageFormatAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ImageFormat;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ImageFormatAttr get(::mlir::MLIRContext *context, ImageFormat val);
  ImageFormat getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ImageFormat> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ImageFormat getEmptyKey() {
    return static_cast<::mlir::spirv::ImageFormat>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ImageFormat getTombstoneKey() {
    return static_cast<::mlir::spirv::ImageFormat>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ImageFormat &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ImageFormat &lhs, const ::mlir::spirv::ImageFormat &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V LinkageType
enum class LinkageType : uint32_t {
  Export = 0,
  Import = 1,
};

::llvm::Optional<LinkageType> symbolizeLinkageType(uint32_t);
::llvm::StringRef stringifyLinkageType(LinkageType);
::llvm::Optional<LinkageType> symbolizeLinkageType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLinkageType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(LinkageType enumValue) {
  return stringifyLinkageType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<LinkageType> symbolizeEnum<LinkageType>(::llvm::StringRef str) {
  return symbolizeLinkageType(str);
}

class LinkageTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = LinkageType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LinkageTypeAttr get(::mlir::MLIRContext *context, LinkageType val);
  LinkageType getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::LinkageType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::LinkageType getEmptyKey() {
    return static_cast<::mlir::spirv::LinkageType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::LinkageType getTombstoneKey() {
    return static_cast<::mlir::spirv::LinkageType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::LinkageType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::LinkageType &lhs, const ::mlir::spirv::LinkageType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V LoopControl
enum class LoopControl : uint32_t {
  None = 0,
  Unroll = 1,
  DontUnroll = 2,
  DependencyInfinite = 4,
  DependencyLength = 8,
  MinIterations = 16,
  MaxIterations = 32,
  IterationMultiple = 64,
  PeelCount = 128,
  PartialCount = 256,
};

::llvm::Optional<LoopControl> symbolizeLoopControl(uint32_t);
std::string stringifyLoopControl(LoopControl);
::llvm::Optional<LoopControl> symbolizeLoopControl(::llvm::StringRef);
inline LoopControl operator|(LoopControl lhs, LoopControl rhs) {
  return static_cast<LoopControl>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline LoopControl operator&(LoopControl lhs, LoopControl rhs) {
  return static_cast<LoopControl>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(LoopControl bits, LoopControl bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

inline std::string stringifyEnum(LoopControl enumValue) {
  return stringifyLoopControl(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<LoopControl> symbolizeEnum<LoopControl>(::llvm::StringRef str) {
  return symbolizeLoopControl(str);
}

class LoopControlAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = LoopControl;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LoopControlAttr get(::mlir::MLIRContext *context, LoopControl val);
  LoopControl getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::LoopControl> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::LoopControl getEmptyKey() {
    return static_cast<::mlir::spirv::LoopControl>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::LoopControl getTombstoneKey() {
    return static_cast<::mlir::spirv::LoopControl>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::LoopControl &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::LoopControl &lhs, const ::mlir::spirv::LoopControl &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V MemoryAccess
enum class MemoryAccess : uint32_t {
  None = 0,
  Volatile = 1,
  Aligned = 2,
  Nontemporal = 4,
  MakePointerAvailable = 8,
  MakePointerVisible = 16,
  NonPrivatePointer = 32,
};

::llvm::Optional<MemoryAccess> symbolizeMemoryAccess(uint32_t);
std::string stringifyMemoryAccess(MemoryAccess);
::llvm::Optional<MemoryAccess> symbolizeMemoryAccess(::llvm::StringRef);
inline MemoryAccess operator|(MemoryAccess lhs, MemoryAccess rhs) {
  return static_cast<MemoryAccess>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline MemoryAccess operator&(MemoryAccess lhs, MemoryAccess rhs) {
  return static_cast<MemoryAccess>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(MemoryAccess bits, MemoryAccess bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

inline std::string stringifyEnum(MemoryAccess enumValue) {
  return stringifyMemoryAccess(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MemoryAccess> symbolizeEnum<MemoryAccess>(::llvm::StringRef str) {
  return symbolizeMemoryAccess(str);
}

class MemoryAccessAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MemoryAccess;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MemoryAccessAttr get(::mlir::MLIRContext *context, MemoryAccess val);
  MemoryAccess getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::MemoryAccess> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::MemoryAccess getEmptyKey() {
    return static_cast<::mlir::spirv::MemoryAccess>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::MemoryAccess getTombstoneKey() {
    return static_cast<::mlir::spirv::MemoryAccess>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::MemoryAccess &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::MemoryAccess &lhs, const ::mlir::spirv::MemoryAccess &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V MemoryModel
enum class MemoryModel : uint32_t {
  Simple = 0,
  GLSL450 = 1,
  OpenCL = 2,
  Vulkan = 3,
};

::llvm::Optional<MemoryModel> symbolizeMemoryModel(uint32_t);
::llvm::StringRef stringifyMemoryModel(MemoryModel);
::llvm::Optional<MemoryModel> symbolizeMemoryModel(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemoryModel() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(MemoryModel enumValue) {
  return stringifyMemoryModel(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MemoryModel> symbolizeEnum<MemoryModel>(::llvm::StringRef str) {
  return symbolizeMemoryModel(str);
}

class MemoryModelAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MemoryModel;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MemoryModelAttr get(::mlir::MLIRContext *context, MemoryModel val);
  MemoryModel getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::MemoryModel> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::MemoryModel getEmptyKey() {
    return static_cast<::mlir::spirv::MemoryModel>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::MemoryModel getTombstoneKey() {
    return static_cast<::mlir::spirv::MemoryModel>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::MemoryModel &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::MemoryModel &lhs, const ::mlir::spirv::MemoryModel &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V MemorySemantics
enum class MemorySemantics : uint32_t {
  None = 0,
  Acquire = 2,
  Release = 4,
  AcquireRelease = 8,
  SequentiallyConsistent = 16,
  UniformMemory = 64,
  SubgroupMemory = 128,
  WorkgroupMemory = 256,
  CrossWorkgroupMemory = 512,
  AtomicCounterMemory = 1024,
  ImageMemory = 2048,
  OutputMemory = 4096,
  MakeAvailable = 8192,
  MakeVisible = 16384,
  Volatile = 32768,
};

::llvm::Optional<MemorySemantics> symbolizeMemorySemantics(uint32_t);
std::string stringifyMemorySemantics(MemorySemantics);
::llvm::Optional<MemorySemantics> symbolizeMemorySemantics(::llvm::StringRef);
inline MemorySemantics operator|(MemorySemantics lhs, MemorySemantics rhs) {
  return static_cast<MemorySemantics>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline MemorySemantics operator&(MemorySemantics lhs, MemorySemantics rhs) {
  return static_cast<MemorySemantics>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(MemorySemantics bits, MemorySemantics bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

inline std::string stringifyEnum(MemorySemantics enumValue) {
  return stringifyMemorySemantics(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MemorySemantics> symbolizeEnum<MemorySemantics>(::llvm::StringRef str) {
  return symbolizeMemorySemantics(str);
}

class MemorySemanticsAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MemorySemantics;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MemorySemanticsAttr get(::mlir::MLIRContext *context, MemorySemantics val);
  MemorySemantics getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::MemorySemantics> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::MemorySemantics getEmptyKey() {
    return static_cast<::mlir::spirv::MemorySemantics>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::MemorySemantics getTombstoneKey() {
    return static_cast<::mlir::spirv::MemorySemantics>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::MemorySemantics &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::MemorySemantics &lhs, const ::mlir::spirv::MemorySemantics &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V instructions
enum class Opcode : uint32_t {
  OpNop = 0,
  OpUndef = 1,
  OpSourceContinued = 2,
  OpSource = 3,
  OpSourceExtension = 4,
  OpName = 5,
  OpMemberName = 6,
  OpString = 7,
  OpLine = 8,
  OpExtension = 10,
  OpExtInstImport = 11,
  OpExtInst = 12,
  OpMemoryModel = 14,
  OpEntryPoint = 15,
  OpExecutionMode = 16,
  OpCapability = 17,
  OpTypeVoid = 19,
  OpTypeBool = 20,
  OpTypeInt = 21,
  OpTypeFloat = 22,
  OpTypeVector = 23,
  OpTypeMatrix = 24,
  OpTypeImage = 25,
  OpTypeSampledImage = 27,
  OpTypeArray = 28,
  OpTypeRuntimeArray = 29,
  OpTypeStruct = 30,
  OpTypePointer = 32,
  OpTypeFunction = 33,
  OpTypeForwardPointer = 39,
  OpConstantTrue = 41,
  OpConstantFalse = 42,
  OpConstant = 43,
  OpConstantComposite = 44,
  OpConstantNull = 46,
  OpSpecConstantTrue = 48,
  OpSpecConstantFalse = 49,
  OpSpecConstant = 50,
  OpSpecConstantComposite = 51,
  OpSpecConstantOp = 52,
  OpFunction = 54,
  OpFunctionParameter = 55,
  OpFunctionEnd = 56,
  OpFunctionCall = 57,
  OpVariable = 59,
  OpLoad = 61,
  OpStore = 62,
  OpCopyMemory = 63,
  OpAccessChain = 65,
  OpDecorate = 71,
  OpMemberDecorate = 72,
  OpVectorExtractDynamic = 77,
  OpVectorInsertDynamic = 78,
  OpVectorShuffle = 79,
  OpCompositeConstruct = 80,
  OpCompositeExtract = 81,
  OpCompositeInsert = 82,
  OpTranspose = 84,
  OpImage = 100,
  OpConvertFToU = 109,
  OpConvertFToS = 110,
  OpConvertSToF = 111,
  OpConvertUToF = 112,
  OpUConvert = 113,
  OpSConvert = 114,
  OpFConvert = 115,
  OpBitcast = 124,
  OpSNegate = 126,
  OpFNegate = 127,
  OpIAdd = 128,
  OpFAdd = 129,
  OpISub = 130,
  OpFSub = 131,
  OpIMul = 132,
  OpFMul = 133,
  OpUDiv = 134,
  OpSDiv = 135,
  OpFDiv = 136,
  OpUMod = 137,
  OpSRem = 138,
  OpSMod = 139,
  OpFRem = 140,
  OpFMod = 141,
  OpMatrixTimesScalar = 143,
  OpMatrixTimesMatrix = 146,
  OpIsNan = 156,
  OpIsInf = 157,
  OpOrdered = 162,
  OpUnordered = 163,
  OpLogicalEqual = 164,
  OpLogicalNotEqual = 165,
  OpLogicalOr = 166,
  OpLogicalAnd = 167,
  OpLogicalNot = 168,
  OpSelect = 169,
  OpIEqual = 170,
  OpINotEqual = 171,
  OpUGreaterThan = 172,
  OpSGreaterThan = 173,
  OpUGreaterThanEqual = 174,
  OpSGreaterThanEqual = 175,
  OpULessThan = 176,
  OpSLessThan = 177,
  OpULessThanEqual = 178,
  OpSLessThanEqual = 179,
  OpFOrdEqual = 180,
  OpFUnordEqual = 181,
  OpFOrdNotEqual = 182,
  OpFUnordNotEqual = 183,
  OpFOrdLessThan = 184,
  OpFUnordLessThan = 185,
  OpFOrdGreaterThan = 186,
  OpFUnordGreaterThan = 187,
  OpFOrdLessThanEqual = 188,
  OpFUnordLessThanEqual = 189,
  OpFOrdGreaterThanEqual = 190,
  OpFUnordGreaterThanEqual = 191,
  OpShiftRightLogical = 194,
  OpShiftRightArithmetic = 195,
  OpShiftLeftLogical = 196,
  OpBitwiseOr = 197,
  OpBitwiseXor = 198,
  OpBitwiseAnd = 199,
  OpNot = 200,
  OpBitFieldInsert = 201,
  OpBitFieldSExtract = 202,
  OpBitFieldUExtract = 203,
  OpBitReverse = 204,
  OpBitCount = 205,
  OpControlBarrier = 224,
  OpMemoryBarrier = 225,
  OpAtomicCompareExchangeWeak = 231,
  OpAtomicIIncrement = 232,
  OpAtomicIDecrement = 233,
  OpAtomicIAdd = 234,
  OpAtomicISub = 235,
  OpAtomicSMin = 236,
  OpAtomicUMin = 237,
  OpAtomicSMax = 238,
  OpAtomicUMax = 239,
  OpAtomicAnd = 240,
  OpAtomicOr = 241,
  OpAtomicXor = 242,
  OpPhi = 245,
  OpLoopMerge = 246,
  OpSelectionMerge = 247,
  OpLabel = 248,
  OpBranch = 249,
  OpBranchConditional = 250,
  OpReturn = 253,
  OpReturnValue = 254,
  OpUnreachable = 255,
  OpGroupBroadcast = 263,
  OpNoLine = 317,
  OpModuleProcessed = 330,
  OpGroupNonUniformElect = 333,
  OpGroupNonUniformBroadcast = 337,
  OpGroupNonUniformBallot = 339,
  OpGroupNonUniformIAdd = 349,
  OpGroupNonUniformFAdd = 350,
  OpGroupNonUniformIMul = 351,
  OpGroupNonUniformFMul = 352,
  OpGroupNonUniformSMin = 353,
  OpGroupNonUniformUMin = 354,
  OpGroupNonUniformFMin = 355,
  OpGroupNonUniformSMax = 356,
  OpGroupNonUniformUMax = 357,
  OpGroupNonUniformFMax = 358,
  OpSubgroupBallotKHR = 4421,
  OpTypeCooperativeMatrixNV = 5358,
  OpCooperativeMatrixLoadNV = 5359,
  OpCooperativeMatrixStoreNV = 5360,
  OpCooperativeMatrixMulAddNV = 5361,
  OpCooperativeMatrixLengthNV = 5362,
  OpSubgroupBlockReadINTEL = 5575,
  OpSubgroupBlockWriteINTEL = 5576,
};

::llvm::Optional<Opcode> symbolizeOpcode(uint32_t);
::llvm::StringRef stringifyOpcode(Opcode);
::llvm::Optional<Opcode> symbolizeOpcode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForOpcode() {
  return 5576;
}


inline ::llvm::StringRef stringifyEnum(Opcode enumValue) {
  return stringifyOpcode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Opcode> symbolizeEnum<Opcode>(::llvm::StringRef str) {
  return symbolizeOpcode(str);
}

class OpcodeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Opcode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static OpcodeAttr get(::mlir::MLIRContext *context, Opcode val);
  Opcode getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Opcode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::Opcode getEmptyKey() {
    return static_cast<::mlir::spirv::Opcode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Opcode getTombstoneKey() {
    return static_cast<::mlir::spirv::Opcode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Opcode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::Opcode &lhs, const ::mlir::spirv::Opcode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Sampler Use specification
enum class ImageSamplerUseInfo : uint32_t {
  SamplerUnknown = 0,
  NeedSampler = 1,
  NoSampler = 2,
};

::llvm::Optional<ImageSamplerUseInfo> symbolizeImageSamplerUseInfo(uint32_t);
::llvm::StringRef stringifyImageSamplerUseInfo(ImageSamplerUseInfo);
::llvm::Optional<ImageSamplerUseInfo> symbolizeImageSamplerUseInfo(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForImageSamplerUseInfo() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(ImageSamplerUseInfo enumValue) {
  return stringifyImageSamplerUseInfo(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ImageSamplerUseInfo> symbolizeEnum<ImageSamplerUseInfo>(::llvm::StringRef str) {
  return symbolizeImageSamplerUseInfo(str);
}

class ImageSamplerUseInfoAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ImageSamplerUseInfo;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ImageSamplerUseInfoAttr get(::mlir::MLIRContext *context, ImageSamplerUseInfo val);
  ImageSamplerUseInfo getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ImageSamplerUseInfo> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ImageSamplerUseInfo getEmptyKey() {
    return static_cast<::mlir::spirv::ImageSamplerUseInfo>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ImageSamplerUseInfo getTombstoneKey() {
    return static_cast<::mlir::spirv::ImageSamplerUseInfo>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ImageSamplerUseInfo &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ImageSamplerUseInfo &lhs, const ::mlir::spirv::ImageSamplerUseInfo &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Image Sampling specification
enum class ImageSamplingInfo : uint32_t {
  SingleSampled = 0,
  MultiSampled = 1,
};

::llvm::Optional<ImageSamplingInfo> symbolizeImageSamplingInfo(uint32_t);
::llvm::StringRef stringifyImageSamplingInfo(ImageSamplingInfo);
::llvm::Optional<ImageSamplingInfo> symbolizeImageSamplingInfo(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForImageSamplingInfo() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ImageSamplingInfo enumValue) {
  return stringifyImageSamplingInfo(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ImageSamplingInfo> symbolizeEnum<ImageSamplingInfo>(::llvm::StringRef str) {
  return symbolizeImageSamplingInfo(str);
}

class ImageSamplingInfoAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ImageSamplingInfo;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ImageSamplingInfoAttr get(::mlir::MLIRContext *context, ImageSamplingInfo val);
  ImageSamplingInfo getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::ImageSamplingInfo> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::ImageSamplingInfo getEmptyKey() {
    return static_cast<::mlir::spirv::ImageSamplingInfo>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::ImageSamplingInfo getTombstoneKey() {
    return static_cast<::mlir::spirv::ImageSamplingInfo>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::ImageSamplingInfo &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::ImageSamplingInfo &lhs, const ::mlir::spirv::ImageSamplingInfo &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V Scope
enum class Scope : uint32_t {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
  QueueFamily = 5,
};

::llvm::Optional<Scope> symbolizeScope(uint32_t);
::llvm::StringRef stringifyScope(Scope);
::llvm::Optional<Scope> symbolizeScope(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForScope() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(Scope enumValue) {
  return stringifyScope(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Scope> symbolizeEnum<Scope>(::llvm::StringRef str) {
  return symbolizeScope(str);
}

class ScopeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Scope;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ScopeAttr get(::mlir::MLIRContext *context, Scope val);
  Scope getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Scope> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::Scope getEmptyKey() {
    return static_cast<::mlir::spirv::Scope>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Scope getTombstoneKey() {
    return static_cast<::mlir::spirv::Scope>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Scope &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::Scope &lhs, const ::mlir::spirv::Scope &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V SelectionControl
enum class SelectionControl : uint32_t {
  None = 0,
  Flatten = 1,
  DontFlatten = 2,
};

::llvm::Optional<SelectionControl> symbolizeSelectionControl(uint32_t);
std::string stringifySelectionControl(SelectionControl);
::llvm::Optional<SelectionControl> symbolizeSelectionControl(::llvm::StringRef);
inline SelectionControl operator|(SelectionControl lhs, SelectionControl rhs) {
  return static_cast<SelectionControl>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline SelectionControl operator&(SelectionControl lhs, SelectionControl rhs) {
  return static_cast<SelectionControl>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(SelectionControl bits, SelectionControl bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

inline std::string stringifyEnum(SelectionControl enumValue) {
  return stringifySelectionControl(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<SelectionControl> symbolizeEnum<SelectionControl>(::llvm::StringRef str) {
  return symbolizeSelectionControl(str);
}

class SelectionControlAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = SelectionControl;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static SelectionControlAttr get(::mlir::MLIRContext *context, SelectionControl val);
  SelectionControl getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::SelectionControl> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::SelectionControl getEmptyKey() {
    return static_cast<::mlir::spirv::SelectionControl>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::SelectionControl getTombstoneKey() {
    return static_cast<::mlir::spirv::SelectionControl>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::SelectionControl &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::SelectionControl &lhs, const ::mlir::spirv::SelectionControl &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V StorageClass
enum class StorageClass : uint32_t {
  UniformConstant = 0,
  Input = 1,
  Uniform = 2,
  Output = 3,
  Workgroup = 4,
  CrossWorkgroup = 5,
  Private = 6,
  Function = 7,
  Generic = 8,
  PushConstant = 9,
  AtomicCounter = 10,
  Image = 11,
  StorageBuffer = 12,
  CallableDataNV = 5328,
  IncomingCallableDataNV = 5329,
  RayPayloadNV = 5338,
  HitAttributeNV = 5339,
  IncomingRayPayloadNV = 5342,
  ShaderRecordBufferNV = 5343,
  PhysicalStorageBuffer = 5349,
};

::llvm::Optional<StorageClass> symbolizeStorageClass(uint32_t);
::llvm::StringRef stringifyStorageClass(StorageClass);
::llvm::Optional<StorageClass> symbolizeStorageClass(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForStorageClass() {
  return 5349;
}


inline ::llvm::StringRef stringifyEnum(StorageClass enumValue) {
  return stringifyStorageClass(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<StorageClass> symbolizeEnum<StorageClass>(::llvm::StringRef str) {
  return symbolizeStorageClass(str);
}

class StorageClassAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = StorageClass;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static StorageClassAttr get(::mlir::MLIRContext *context, StorageClass val);
  StorageClass getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::StorageClass> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::StorageClass getEmptyKey() {
    return static_cast<::mlir::spirv::StorageClass>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::StorageClass getTombstoneKey() {
    return static_cast<::mlir::spirv::StorageClass>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::StorageClass &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::StorageClass &lhs, const ::mlir::spirv::StorageClass &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// recognized SPIR-V vendor strings
enum class Vendor {
  AMD,
  ARM,
  Imagination,
  Intel,
  NVIDIA,
  Qualcomm,
  SwiftShader,
  Unknown,
};

::llvm::StringRef stringifyVendor(Vendor);
::llvm::Optional<Vendor> symbolizeVendor(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(Vendor enumValue) {
  return stringifyVendor(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Vendor> symbolizeEnum<Vendor>(::llvm::StringRef str) {
  return symbolizeVendor(str);
}
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Vendor> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::spirv::Vendor>::type>;

  static inline ::mlir::spirv::Vendor getEmptyKey() {
    return static_cast<::mlir::spirv::Vendor>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Vendor getTombstoneKey() {
    return static_cast<::mlir::spirv::Vendor>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Vendor &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::spirv::Vendor>::type>(val));
  }

  static bool isEqual(const ::mlir::spirv::Vendor &lhs, const ::mlir::spirv::Vendor &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace spirv {
// valid SPIR-V version
enum class Version : uint32_t {
  V_1_0 = 0,
  V_1_1 = 1,
  V_1_2 = 2,
  V_1_3 = 3,
  V_1_4 = 4,
  V_1_5 = 5,
};

::llvm::Optional<Version> symbolizeVersion(uint32_t);
::llvm::StringRef stringifyVersion(Version);
::llvm::Optional<Version> symbolizeVersion(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVersion() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(Version enumValue) {
  return stringifyVersion(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Version> symbolizeEnum<Version>(::llvm::StringRef str) {
  return symbolizeVersion(str);
}

class VersionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Version;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static VersionAttr get(::mlir::MLIRContext *context, Version val);
  Version getValue() const;
};
} // namespace spirv
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::spirv::Version> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::spirv::Version getEmptyKey() {
    return static_cast<::mlir::spirv::Version>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::spirv::Version getTombstoneKey() {
    return static_cast<::mlir::spirv::Version>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::spirv::Version &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::spirv::Version &lhs, const ::mlir::spirv::Version &rhs) {
    return lhs == rhs;
  }
};
}

