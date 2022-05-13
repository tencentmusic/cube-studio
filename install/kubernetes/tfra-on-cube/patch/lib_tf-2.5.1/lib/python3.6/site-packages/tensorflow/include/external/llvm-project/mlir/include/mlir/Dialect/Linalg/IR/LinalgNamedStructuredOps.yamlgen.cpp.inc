
//===----------------------------------------------------------------------===//
// Implementation of MatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr MatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulOp::getNumRegionArgs() { return 3; }

std::string MatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulOp::regionBuilder(Block &block, ValueRange captures) {
  RegionBuilderHelper helper(block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

void MatmulOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context) {
  results.add<EraseDeadLinalgOp>();
  results.add<FoldTensorCastOp>();
}
LogicalResult MatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
  getGenericEffectsImpl(effects,
    getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr BatchMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr BatchMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatmulOp::regionBuilder(Block &block, ValueRange captures) {
  RegionBuilderHelper helper(block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

void BatchMatmulOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context) {
  results.add<EraseDeadLinalgOp>();
  results.add<FoldTensorCastOp>();
}
LogicalResult BatchMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void BatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
  getGenericEffectsImpl(effects,
    getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
}

//===----------------------------------------------------------------------===//
// Implementation of MatvecOp
//===----------------------------------------------------------------------===//

ArrayAttr MatvecOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr MatvecOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatvecOp::getNumRegionArgs() { return 3; }

std::string MatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatvecOp::regionBuilder(Block &block, ValueRange captures) {
  RegionBuilderHelper helper(block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

void MatvecOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context) {
  results.add<EraseDeadLinalgOp>();
  results.add<FoldTensorCastOp>();
}
LogicalResult MatvecOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
  getGenericEffectsImpl(effects,
    getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
}

//===----------------------------------------------------------------------===//
// Implementation of VecmatOp
//===----------------------------------------------------------------------===//

ArrayAttr VecmatOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(VecmatOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr VecmatOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1, d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned VecmatOp::getNumRegionArgs() { return 3; }

std::string VecmatOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void VecmatOp::regionBuilder(Block &block, ValueRange captures) {
  RegionBuilderHelper helper(block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

void VecmatOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context) {
  results.add<EraseDeadLinalgOp>();
  results.add<FoldTensorCastOp>();
}
LogicalResult VecmatOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void VecmatOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
  getGenericEffectsImpl(effects,
    getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
}

//===----------------------------------------------------------------------===//
// Implementation of DotOp
//===----------------------------------------------------------------------===//

ArrayAttr DotOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DotOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  return exprs;
}

ArrayAttr DotOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DotOp::getNumRegionArgs() { return 3; }

std::string DotOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void DotOp::regionBuilder(Block &block, ValueRange captures) {
  RegionBuilderHelper helper(block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

void DotOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context) {
  results.add<EraseDeadLinalgOp>();
  results.add<FoldTensorCastOp>();
}
LogicalResult DotOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DotOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
  getGenericEffectsImpl(effects,
    getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
}
