if (auto op = dyn_cast<::mlir::LLVM::AShrOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateAShr(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateAdd(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddrSpaceCastOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateAddrSpaceCast(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AllocaOp>(opInst)) {

    auto *inst = builder.CreateAlloca(
      moduleTranslation.convertType(op.getResult().getType())->getPointerElementType(), moduleTranslation.lookupValue(op.arraySize()));
    
    if (op.alignment().hasValue()) {
      auto align = op.alignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    moduleTranslation.mapValue(op.res()) = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AndOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateAnd(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AssumeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn =
        llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::assume, {});
    builder.CreateCall(fn, {moduleTranslation.lookupValue(op.cond())});
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicCmpXchgOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateAtomicCmpXchg(moduleTranslation.lookupValue(op.ptr()), moduleTranslation.lookupValue(op.cmp()), moduleTranslation.lookupValue(op.val()), llvm::MaybeAlign(),
                   getLLVMAtomicOrdering(op.success_ordering()),
                   getLLVMAtomicOrdering(op.failure_ordering()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicRMWOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateAtomicRMW(getLLVMAtomicBinOp(op.bin_op()), moduleTranslation.lookupValue(op.ptr()), moduleTranslation.lookupValue(op.val()),
                                   llvm::MaybeAlign(),
                                   getLLVMAtomicOrdering(op.ordering()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitReverseOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::bitreverse,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitcastOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateBitCast(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ConstantOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = getLLVMConstant(moduleTranslation.convertType(op.getResult().getType()), op.value(), opInst.getLoc(),
                                            moduleTranslation);
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CopySignOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::copysign,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroBeginOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_begin,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroEndOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_end,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroFreeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_free,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroIdOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_id,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroResumeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_resume,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSaveOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_save,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSizeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_size,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSuspendOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::coro_suspend,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CosOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::cos,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CtPopOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ctpop,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Exp2Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::exp2,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExpOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::exp,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractElementOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateExtractElement(moduleTranslation.lookupValue(op.vector()), moduleTranslation.lookupValue(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractValueOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateExtractValue(moduleTranslation.lookupValue(op.container()), extractPosition(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAbsOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fabs,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAddOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFAdd(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCeilOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ceil,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCmpOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateFCmp(getLLVMCmpPredicate(op.predicate()), moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FDivOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFDiv(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FFloorOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::floor,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMAOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fma,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulAddOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fmuladd,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFMul(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FNegOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFNeg(moduleTranslation.lookupValue(op.operand()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPExtOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFPExt(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToSIOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFPToSI(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToUIOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFPToUI(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPTruncOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFPTrunc(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FRemOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFRem(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FSubOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateFSub(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FenceOp>(opInst)) {

    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering(op.ordering()),
      llvmContext.getOrInsertSyncScopeID(op.syncscope()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FreezeOp>(opInst)) {
builder.CreateFreeze(moduleTranslation.lookupValue(op.val()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GEPOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateGEP( moduleTranslation.lookupValue(op.base())->getType()->getPointerElementType(), moduleTranslation.lookupValue(op.base()), moduleTranslation.lookupValues(op.indices()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GetActiveLaneMaskOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::get_active_lane_mask,
        { moduleTranslation.convertType(opInst.getResult(0).getType()), moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ICmpOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateICmp(getLLVMCmpPredicate(op.predicate()), moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertElementOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateInsertElement(moduleTranslation.lookupValue(op.vector()), moduleTranslation.lookupValue(op.value()), moduleTranslation.lookupValue(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertValueOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = builder.CreateInsertValue(moduleTranslation.lookupValue(op.container()), moduleTranslation.lookupValue(op.value()),
                                     extractPosition(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::IntToPtrOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateIntToPtr(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LShrOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateLShr(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LoadOp>(opInst)) {

    auto *inst = builder.CreateLoad(
      moduleTranslation.lookupValue(op.addr())->getType()->getPointerElementType(), moduleTranslation.lookupValue(op.addr()), op.volatile_());
  
    if (op.alignment().hasValue()) {
      auto align = op.alignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.nontemporal()) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.mapValue(op.res()) = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log10Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log10,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log2Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log2,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LogOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedLoadOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = moduleTranslation.lookupValues(op.pass_thru()).empty() ? builder.CreateMaskedLoad(
      moduleTranslation.lookupValue(op.data()), llvm::Align(op.alignment()), moduleTranslation.lookupValue(op.mask())) :
      builder.CreateMaskedLoad(
        moduleTranslation.lookupValue(op.data()), llvm::Align(op.alignment()), moduleTranslation.lookupValue(op.mask()), moduleTranslation.lookupValues(op.pass_thru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedStoreOp>(opInst)) {

    builder.CreateMaskedStore(
      moduleTranslation.lookupValue(op.value()), moduleTranslation.lookupValue(op.data()), llvm::Align(op.alignment()), moduleTranslation.lookupValue(op.mask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorLoadOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      moduleTranslation.lookupValue(op.data())->getType()->getPointerElementType());
    moduleTranslation.mapValue(op.res()) = mb.CreateColumnMajorLoad(
      moduleTranslation.lookupValue(op.data()), align, moduleTranslation.lookupValue(op.stride()), op.isVolatile(), op.rows(),
      op.columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorStoreOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      moduleTranslation.lookupValue(op.data())->getType()->getPointerElementType());
    mb.CreateColumnMajorStore(
      moduleTranslation.lookupValue(op.matrix()), moduleTranslation.lookupValue(op.data()), align, moduleTranslation.lookupValue(op.stride()), op.isVolatile(),
      op.rows(), op.columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixMultiplyOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    moduleTranslation.mapValue(op.res()) = mb.CreateMatrixMultiply(
      moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()), op.lhs_rows(), op.lhs_columns(),
      op.rhs_columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixTransposeOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    moduleTranslation.mapValue(op.res()) = mb.CreateMatrixTranspose(
      moduleTranslation.lookupValue(op.matrix()), op.rows(), op.columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaxNumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::maxnum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaximumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::maximum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyInlineOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memcpy_inline,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType()), moduleTranslation.convertType(opInst.getOperand(2).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memcpy,
        { moduleTranslation.convertType(opInst.getOperand(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType()), moduleTranslation.convertType(opInst.getOperand(2).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinNumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::minnum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinimumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::minimum,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MulOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateMul(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::NullOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = llvm::ConstantPointerNull::get(    cast<llvm::PointerType>(moduleTranslation.convertType(op.getResult().getType())));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::OrOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateOr(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::pow,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Prefetch>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::prefetch,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PtrToIntOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreatePtrToInt(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ResumeOp>(opInst)) {
 builder.CreateResume(moduleTranslation.lookupValue(op.value())); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ReturnOp>(opInst)) {

    if (opInst.getNumOperands() != 0)
      builder.CreateRet(moduleTranslation.lookupValues(op.args())[0]);
    else
      builder.CreateRetVoid();
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SAddWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sadd_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SDivOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateSDiv(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SExtOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateSExt(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SIToFPOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateSIToFP(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMaxOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMulWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smul_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SRemOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateSRem(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SSubWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ssub_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SelectOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateSelect(moduleTranslation.lookupValue(op.condition()), moduleTranslation.lookupValue(op.trueValue()), moduleTranslation.lookupValue(op.falseValue()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShlOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateShl(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShuffleVectorOp>(opInst)) {

      SmallVector<unsigned, 4> position = extractPosition(op.mask());
      SmallVector<int, 4> mask(position.begin(), position.end());
      moduleTranslation.mapValue(op.res()) = builder.CreateShuffleVector(moduleTranslation.lookupValue(op.v1()), moduleTranslation.lookupValue(op.v2()), mask);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SqrtOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sqrt,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StackRestoreOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::stackrestore,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StackSaveOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::stacksave,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StoreOp>(opInst)) {

    auto *inst = builder.CreateStore(moduleTranslation.lookupValue(op.value()), moduleTranslation.lookupValue(op.addr()), op.volatile_());
  
    if (op.alignment().hasValue()) {
      auto align = op.alignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.nontemporal()) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SubOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateSub(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::TruncOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateTrunc(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UAddWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::uadd_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UDivOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateUDiv(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UIToFPOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateUIToFP(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UMulWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::umul_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::URemOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateURem(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::USubWithOverflowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::usub_with_overflow,
        { moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[0])
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UndefOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = llvm::UndefValue::get(moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UnreachableOp>(opInst)) {
 builder.CreateUnreachable(); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::XOrOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateXor(moduleTranslation.lookupValue(op.lhs()), moduleTranslation.lookupValue(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ZExtOp>(opInst)) {
moduleTranslation.mapValue(op.res()) = builder.CreateZExt(moduleTranslation.lookupValue(op.arg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_compressstore>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::masked_compressstore,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_expandload>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::masked_expandload,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_gather>(opInst)) {

    moduleTranslation.mapValue(op.res()) = moduleTranslation.lookupValues(op.pass_thru()).empty() ? builder.CreateMaskedGather(
      moduleTranslation.lookupValue(op.ptrs()), llvm::Align(op.alignment()), moduleTranslation.lookupValue(op.mask())) :
      builder.CreateMaskedGather(
        moduleTranslation.lookupValue(op.ptrs()), llvm::Align(op.alignment()), moduleTranslation.lookupValue(op.mask()), moduleTranslation.lookupValues(op.pass_thru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_scatter>(opInst)) {

    builder.CreateMaskedScatter(
      moduleTranslation.lookupValue(op.value()), moduleTranslation.lookupValue(op.ptrs()), llvm::Align(op.alignment()), moduleTranslation.lookupValue(op.mask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_add>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_add,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_and>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_and,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fadd>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fadd,
        { moduleTranslation.convertType(opInst.getOperand(1).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc(op.reassoc());
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    moduleTranslation.mapValue(op.res()) = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmul>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmul,
        { moduleTranslation.convertType(opInst.getOperand(1).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc(op.reassoc());
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    moduleTranslation.mapValue(op.res()) = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_mul>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_mul,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_or>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_or,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_smax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_smin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_umax,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_umin,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_xor>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_xor,
        { moduleTranslation.convertType(opInst.getOperand(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
