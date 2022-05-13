inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::AsmDialect convertAsmDialectFromLLVM(::llvm::InlineAsm::AsmDialect value) {
  switch (value) {
  case ::llvm::InlineAsm::AsmDialect::AD_ATT:
    return ::mlir::LLVM::AsmDialect::AD_ATT;
  case ::llvm::InlineAsm::AsmDialect::AD_Intel:
    return ::mlir::LLVM::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown ::llvm::InlineAsm::AsmDialect type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::Linkage convertLinkageFromLLVM(::llvm::GlobalValue::LinkageTypes value) {
  switch (value) {
  case ::llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    return ::mlir::LLVM::Linkage::Private;
  case ::llvm::GlobalValue::LinkageTypes::InternalLinkage:
    return ::mlir::LLVM::Linkage::Internal;
  case ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    return ::mlir::LLVM::Linkage::AvailableExternally;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    return ::mlir::LLVM::Linkage::Linkonce;
  case ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    return ::mlir::LLVM::Linkage::Weak;
  case ::llvm::GlobalValue::LinkageTypes::CommonLinkage:
    return ::mlir::LLVM::Linkage::Common;
  case ::llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    return ::mlir::LLVM::Linkage::Appending;
  case ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    return ::mlir::LLVM::Linkage::ExternWeak;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    return ::mlir::LLVM::Linkage::LinkonceODR;
  case ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    return ::mlir::LLVM::Linkage::WeakODR;
  case ::llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    return ::mlir::LLVM::Linkage::External;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::LinkageTypes type");}

