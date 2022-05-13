/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Op Availability Implementations                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AccessChainOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version AccessChainOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AccessChainOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AccessChainOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AddressOfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version AddressOfOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AddressOfOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AddressOfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicAndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicAndOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicAndOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicCompareExchangeWeakOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicCompareExchangeWeakOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicCompareExchangeWeakOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicCompareExchangeWeakOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->equal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->unequal_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicIAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicIAddOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicIAddOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicIAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicIDecrementOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicIDecrementOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicIDecrementOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicIDecrementOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicIIncrementOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicIIncrementOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicIIncrementOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicIIncrementOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicISubOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicISubOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicISubOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicISubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicOrOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicOrOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicOrOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicSMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicSMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicSMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicSMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicSMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicSMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicSMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicSMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicUMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicUMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicUMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicUMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicUMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicUMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicUMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicUMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> AtomicXorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicXorOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version AtomicXorOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> AtomicXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitCountOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BitCountOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitCountOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitCountOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitFieldInsertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitFieldInsertOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitFieldInsertOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitFieldInsertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitFieldSExtractOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitFieldSExtractOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitFieldSExtractOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitFieldSExtractOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitFieldUExtractOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitFieldUExtractOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitFieldUExtractOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitFieldUExtractOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitReverseOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Shader}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitReverseOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitReverseOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitReverseOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitcastOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BitcastOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitcastOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitcastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitwiseAndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BitwiseAndOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitwiseAndOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitwiseAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitwiseOrOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BitwiseOrOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitwiseOrOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitwiseOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BitwiseXorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BitwiseXorOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BitwiseXorOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BitwiseXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BranchConditionalOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BranchConditionalOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BranchConditionalOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BranchConditionalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> BranchOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version BranchOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version BranchOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> BranchOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CompositeConstructOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version CompositeConstructOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CompositeConstructOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CompositeConstructOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CompositeExtractOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version CompositeExtractOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CompositeExtractOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CompositeExtractOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CompositeInsertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version CompositeInsertOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CompositeInsertOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CompositeInsertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConstantOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ConstantOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ConstantOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConstantOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ControlBarrierOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version ControlBarrierOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ControlBarrierOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ControlBarrierOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertFToSOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ConvertFToSOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ConvertFToSOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertFToSOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertFToUOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ConvertFToUOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ConvertFToUOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertFToUOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertSToFOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ConvertSToFOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ConvertSToFOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertSToFOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ConvertUToFOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ConvertUToFOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ConvertUToFOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ConvertUToFOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CooperativeMatrixLengthNVOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixLengthNVOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixLengthNVOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CooperativeMatrixLengthNVOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CooperativeMatrixLoadNVOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixLoadNVOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixLoadNVOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CooperativeMatrixLoadNVOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CooperativeMatrixMulAddNVOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixMulAddNVOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixMulAddNVOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CooperativeMatrixMulAddNVOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CooperativeMatrixStoreNVOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::CooperativeMatrixNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixStoreNVOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CooperativeMatrixStoreNVOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CooperativeMatrixStoreNVOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_NV_cooperative_matrix}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> CopyMemoryOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version CopyMemoryOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version CopyMemoryOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> CopyMemoryOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> EntryPointOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_model();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version EntryPointOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version EntryPointOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> EntryPointOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ExecutionModeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_mode();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version ExecutionModeOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->execution_mode();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ExecutionModeOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ExecutionModeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->execution_mode();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FAddOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FAddOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FConvertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FConvertOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FConvertOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FConvertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FDivOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FDivOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FDivOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FModOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FModOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FModOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FModOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FMulOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FMulOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FNegateOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FNegateOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FNegateOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FNegateOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FOrdEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FOrdEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FOrdGreaterThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FOrdGreaterThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FOrdGreaterThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FOrdGreaterThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdLessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FOrdLessThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FOrdLessThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdLessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdLessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FOrdLessThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FOrdLessThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdLessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FOrdNotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FOrdNotEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FOrdNotEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FOrdNotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FRemOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FRemOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FRemOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FRemOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FSubOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FSubOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FSubOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FSubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FUnordEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FUnordEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FUnordGreaterThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FUnordGreaterThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FUnordGreaterThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FUnordGreaterThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordLessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FUnordLessThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FUnordLessThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordLessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordLessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FUnordLessThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FUnordLessThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordLessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FUnordNotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FUnordNotEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FUnordNotEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FUnordNotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FuncOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FuncOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FuncOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FuncOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> FunctionCallOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version FunctionCallOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version FunctionCallOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> FunctionCallOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLAcosOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLAcosOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLAcosOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLAcosOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLAsinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLAsinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLAsinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLAsinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLAtanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLAtanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLAtanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLAtanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLCeilOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLCeilOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLCeilOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLCeilOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLCosOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLCosOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLCosOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLCosOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLCoshOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLCoshOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLCoshOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLCoshOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLExpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLExpOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLExpOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLExpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFAbsOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFAbsOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFClampOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFClampOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFClampOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFSignOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFSignOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFSignOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFSignOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFloorOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFloorOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFloorOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFloorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFmaOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFmaOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFmaOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFmaOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLFrexpStructOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFrexpStructOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLFrexpStructOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLFrexpStructOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLInverseSqrtOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLInverseSqrtOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLInverseSqrtOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLInverseSqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLLdexpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLLdexpOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLLdexpOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLLdexpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLLogOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLLogOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLLogOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLLogOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLPowOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLPowOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLPowOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLPowOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLRoundOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLRoundOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLRoundOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLRoundOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSAbsOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSAbsOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSClampOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSClampOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSClampOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSSignOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSSignOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSSignOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSSignOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSinhOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSinhOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSinhOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSinhOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLSqrtOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSqrtOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLSqrtOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLSqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLTanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLTanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLTanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLTanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLTanhOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLTanhOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLTanhOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLTanhOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GLSLUClampOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GLSLUClampOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GLSLUClampOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GLSLUClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GlobalVariableOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version GlobalVariableOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GlobalVariableOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GlobalVariableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupBroadcastOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Groups}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupBroadcastOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupBroadcastOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupBroadcastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformBallotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformBallotOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformBallotOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformBallotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformBroadcastOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformBallot}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformBroadcastOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformBroadcastOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformBroadcastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformElectOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniform}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformElectOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformElectOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformElectOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFAddOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFAddOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformFMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFMulOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformFMulOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformFMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformIAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformIAddOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformIAddOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformIAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformIMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformIMulOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformIMulOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformIMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformSMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformSMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformSMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformSMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformSMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformSMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformSMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformSMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformUMaxOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformUMaxOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformUMaxOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformUMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> GroupNonUniformUMinOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::GroupNonUniformArithmetic, ::mlir::spirv::Capability::GroupNonUniformClustered, ::mlir::spirv::Capability::GroupNonUniformPartitionedNV}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformUMinOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_3));
  }
  {
    auto tblgen_attrVal = this->execution_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version GroupNonUniformUMinOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> GroupNonUniformUMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->group_operation();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IAddOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version IAddOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version IAddOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IAddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version IEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version IEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IMulOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version IMulOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version IMulOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> INotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version INotEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version INotEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> INotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ISubOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ISubOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ISubOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ISubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ImageOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ImageOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ImageOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ImageOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IsInfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version IsInfOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version IsInfOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IsInfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> IsNanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version IsNanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version IsNanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> IsNanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LoadOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LoadOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LoadOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LoadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalAndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LogicalAndOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LogicalAndOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LogicalEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LogicalEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalNotEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LogicalNotEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LogicalNotEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalNotEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalNotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LogicalNotOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LogicalNotOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalNotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LogicalOrOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LogicalOrOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LogicalOrOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LogicalOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> LoopOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version LoopOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::LoopControl tblgen_attrVal = this->loop_control() & static_cast<::mlir::spirv::LoopControl>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version LoopOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> LoopOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MatrixTimesMatrixOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version MatrixTimesMatrixOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version MatrixTimesMatrixOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MatrixTimesMatrixOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MatrixTimesScalarOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version MatrixTimesScalarOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version MatrixTimesScalarOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MatrixTimesScalarOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MemoryBarrierOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version MemoryBarrierOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  {
    auto tblgen_attrVal = this->memory_scope();
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getMinVersion(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, *tblgen_instance));
  }
  return tblgen_overall;
}
::mlir::spirv::Version MemoryBarrierOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MemoryBarrierOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  for (unsigned i = 0; i < std::numeric_limits<uint32_t>::digits; ++i) {
    ::mlir::spirv::MemorySemantics tblgen_attrVal = this->memory_semantics() & static_cast<::mlir::spirv::MemorySemantics>(1 << i);
    if (static_cast<uint32_t>(tblgen_attrVal) == 0) continue;
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> MergeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version MergeOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version MergeOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> MergeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ModuleEndOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ModuleEndOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ModuleEndOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ModuleEndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ModuleOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->addressing_model();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->memory_model();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version ModuleOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ModuleOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ModuleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->addressing_model();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  {
    auto tblgen_attrVal = this->memory_model();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> NotOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version NotOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version NotOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> NotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> OCLExpOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version OCLExpOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version OCLExpOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> OCLExpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> OCLFAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version OCLFAbsOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version OCLFAbsOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> OCLFAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> OCLSAbsOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version OCLSAbsOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version OCLSAbsOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> OCLSAbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> OrderedOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version OrderedOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version OrderedOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> OrderedOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ReferenceOfOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ReferenceOfOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ReferenceOfOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ReferenceOfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ReturnOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ReturnOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ReturnOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ReturnOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ReturnValueOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ReturnValueOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ReturnValueOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ReturnValueOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SConvertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SConvertOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SConvertOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SConvertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SDivOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SDivOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SDivOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SGreaterThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SGreaterThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SGreaterThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SGreaterThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SLessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SLessThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SLessThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SLessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SLessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SLessThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SLessThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SLessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SModOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SModOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SModOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SModOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SNegateOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SNegateOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SNegateOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SNegateOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SRemOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SRemOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SRemOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SRemOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SelectOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SelectOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SelectOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SelectOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SelectionOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SelectionOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SelectionOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SelectionOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ShiftLeftLogicalOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ShiftLeftLogicalOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ShiftLeftLogicalOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ShiftLeftLogicalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ShiftRightArithmeticOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ShiftRightArithmeticOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ShiftRightArithmeticOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ShiftRightArithmeticOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ShiftRightLogicalOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ShiftRightLogicalOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ShiftRightLogicalOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ShiftRightLogicalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SpecConstantCompositeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SpecConstantCompositeOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SpecConstantCompositeOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SpecConstantCompositeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SpecConstantOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SpecConstantOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SpecConstantOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SpecConstantOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SpecConstantOperationOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version SpecConstantOperationOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SpecConstantOperationOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SpecConstantOperationOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> StoreOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version StoreOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version StoreOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> StoreOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SubgroupBallotKHROp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupBallotKHR}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version SubgroupBallotKHROp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SubgroupBallotKHROp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SubgroupBallotKHROp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_KHR_shader_ballot}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SubgroupBlockReadINTELOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupBufferBlockIOINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version SubgroupBlockReadINTELOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SubgroupBlockReadINTELOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SubgroupBlockReadINTELOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> SubgroupBlockWriteINTELOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::SubgroupBufferBlockIOINTEL}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version SubgroupBlockWriteINTELOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version SubgroupBlockWriteINTELOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> SubgroupBlockWriteINTELOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Extension exts[] = {::mlir::spirv::Extension::SPV_INTEL_subgroups}; ArrayRef<::mlir::spirv::Extension> ref(exts, ::llvm::array_lengthof(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> TransposeOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Matrix}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version TransposeOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version TransposeOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> TransposeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UConvertOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UConvertOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UConvertOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UConvertOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UDivOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UDivOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UDivOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UGreaterThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UGreaterThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UGreaterThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UGreaterThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UGreaterThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UGreaterThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UGreaterThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UGreaterThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ULessThanEqualOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ULessThanEqualOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ULessThanEqualOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ULessThanEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> ULessThanOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version ULessThanOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version ULessThanOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> ULessThanOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UModOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UModOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UModOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UModOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UndefOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UndefOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UndefOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UndefOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UnorderedOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    static const ::mlir::spirv::Capability caps[] = {::mlir::spirv::Capability::Kernel}; ArrayRef<::mlir::spirv::Capability> ref(caps, ::llvm::array_lengthof(caps));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::mlir::spirv::Version UnorderedOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UnorderedOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UnorderedOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> UnreachableOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version UnreachableOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version UnreachableOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> UnreachableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VariableOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->storage_class();
    auto tblgen_instance = ::mlir::spirv::getCapabilities(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::mlir::spirv::Version VariableOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version VariableOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VariableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  {
    auto tblgen_attrVal = this->storage_class();
    auto tblgen_instance = ::mlir::spirv::getExtensions(tblgen_attrVal);
    if (tblgen_instance) tblgen_overall.emplace_back(*tblgen_instance);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorExtractDynamicOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version VectorExtractDynamicOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version VectorExtractDynamicOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorExtractDynamicOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorInsertDynamicOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version VectorInsertDynamicOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version VectorInsertDynamicOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorInsertDynamicOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> VectorShuffleOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version VectorShuffleOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version VectorShuffleOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> VectorShuffleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> YieldOp::getCapabilities() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::mlir::spirv::Version YieldOp::getMinVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::max(tblgen_overall, ::mlir::spirv::Version::V_1_0));
  }
  return tblgen_overall;
}
::mlir::spirv::Version YieldOp::getMaxVersion() {
  ::mlir::spirv::Version tblgen_overall = static_cast<::mlir::spirv::Version>(~uint32_t(0));
  {
    
    tblgen_overall = static_cast<::mlir::spirv::Version>(std::min(tblgen_overall, ::mlir::spirv::Version::V_1_5));
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> YieldOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
