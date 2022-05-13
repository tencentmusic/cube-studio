static LLVM_ATTRIBUTE_UNUSED ::llvm::InlineAsm::AsmDialect convertAsmDialectToLLVM(::mlir::LLVM::AsmDialect value) {
  switch (value) {
  case ::mlir::LLVM::AsmDialect::AD_ATT:
    return ::llvm::InlineAsm::AsmDialect::AD_ATT;
  case ::mlir::LLVM::AsmDialect::AD_Intel:
    return ::llvm::InlineAsm::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown AsmDialect type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::LinkageTypes convertLinkageToLLVM(::mlir::LLVM::Linkage value) {
  switch (value) {
  case ::mlir::LLVM::Linkage::Private:
    return ::llvm::GlobalValue::LinkageTypes::PrivateLinkage;
  case ::mlir::LLVM::Linkage::Internal:
    return ::llvm::GlobalValue::LinkageTypes::InternalLinkage;
  case ::mlir::LLVM::Linkage::AvailableExternally:
    return ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage;
  case ::mlir::LLVM::Linkage::Linkonce:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage;
  case ::mlir::LLVM::Linkage::Weak:
    return ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage;
  case ::mlir::LLVM::Linkage::Common:
    return ::llvm::GlobalValue::LinkageTypes::CommonLinkage;
  case ::mlir::LLVM::Linkage::Appending:
    return ::llvm::GlobalValue::LinkageTypes::AppendingLinkage;
  case ::mlir::LLVM::Linkage::ExternWeak:
    return ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage;
  case ::mlir::LLVM::Linkage::LinkonceODR:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage;
  case ::mlir::LLVM::Linkage::WeakODR:
    return ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage;
  case ::mlir::LLVM::Linkage::External:
    return ::llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  }
  llvm_unreachable("unknown Linkage type");
}

