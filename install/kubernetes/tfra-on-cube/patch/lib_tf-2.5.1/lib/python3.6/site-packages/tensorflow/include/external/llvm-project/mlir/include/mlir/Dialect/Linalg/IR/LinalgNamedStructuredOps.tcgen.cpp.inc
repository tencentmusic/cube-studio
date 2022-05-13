
    ArrayAttr MatmulColumnMajorOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatmulColumnMajorOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2;
    bindDims(context, d0, d1, d2);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto map0 = AffineMap::get(3, 3, {d2, d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(3, 3, {d0, d2}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(3, 3, {d0, d1}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatmulColumnMajorOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void MatmulColumnMajorOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatmulColumnMajorOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatmulColumnMajorOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr MatmulI8I8I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatmulI8I8I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2;
    bindDims(context, d0, d1, d2);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto map0 = AffineMap::get(3, 3, {d0, d2}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(3, 3, {d2, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(3, 3, {d0, d1}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatmulI8I8I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void MatmulI8I8I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatmulI8I8I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatmulI8I8I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr MatmulI16I16I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatmulI16I16I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2;
    bindDims(context, d0, d1, d2);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto map0 = AffineMap::get(3, 3, {d0, d2}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(3, 3, {d2, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(3, 3, {d0, d1}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatmulI16I16I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void MatmulI16I16I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatmulI16I16I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatmulI16I16I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr MatmulI32I32I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatmulI32I32I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2;
    bindDims(context, d0, d1, d2);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto map0 = AffineMap::get(3, 3, {d0, d2}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(3, 3, {d2, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(3, 3, {d0, d1}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatmulI32I32I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_muli(_0, _1);
    Value _5 = std_addi(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void MatmulI32I32I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatmulI32I32I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatmulI32I32I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr MatvecI8I8I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatvecI8I8I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map0 = AffineMap::get(2, 2, {d0, d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
	auto map1 = AffineMap::get(2, 2, {d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatvecI8I8I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void MatvecI8I8I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatvecI8I8I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatvecI8I8I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr MatvecI16I16I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatvecI16I16I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map0 = AffineMap::get(2, 2, {d0, d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
	auto map1 = AffineMap::get(2, 2, {d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatvecI16I16I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void MatvecI16I16I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatvecI16I16I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatvecI16I16I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr MatvecI32I32I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatvecI32I32I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map0 = AffineMap::get(2, 2, {d0, d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
	auto map1 = AffineMap::get(2, 2, {d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatvecI32I32I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_muli(_0, _1);
    Value _5 = std_addi(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void MatvecI32I32I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult MatvecI32I32I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatvecI32I32I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr VecmatI8I8I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr VecmatI8I8I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map1 = AffineMap::get(2, 2, {d1, d0}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
	auto map0 = AffineMap::get(2, 2, {d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void VecmatI8I8I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void VecmatI8I8I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult VecmatI8I8I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void VecmatI8I8I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr VecmatI16I16I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr VecmatI16I16I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map1 = AffineMap::get(2, 2, {d1, d0}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
	auto map0 = AffineMap::get(2, 2, {d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void VecmatI16I16I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void VecmatI16I16I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult VecmatI16I16I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void VecmatI16I16I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr VecmatI32I32I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr VecmatI32I32I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map1 = AffineMap::get(2, 2, {d1, d0}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
	auto map0 = AffineMap::get(2, 2, {d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void VecmatI32I32I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_muli(_0, _1);
    Value _5 = std_addi(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void VecmatI32I32I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult VecmatI32I32I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void VecmatI32I32I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr DotI8I8I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr DotI8I8I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0;
    bindDims(context, d0);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto map0 = AffineMap::get(1, 1, {d0}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(1, 1, {d0}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(1, 1, {}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void DotI8I8I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void DotI8I8I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult DotI8I8I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void DotI8I8I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr DotI16I16I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr DotI16I16I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0;
    bindDims(context, d0);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto map0 = AffineMap::get(1, 1, {d0}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(1, 1, {d0}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(1, 1, {}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void DotI16I16I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void DotI16I16I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult DotI16I16I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void DotI16I16I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr DotI32I32I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr DotI32I32I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0;
    bindDims(context, d0);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto map0 = AffineMap::get(1, 1, {d0}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(1, 1, {d0}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(1, 1, {}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0 }, 1, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void DotI32I32I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_muli(_0, _1);
    Value _5 = std_addi(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void DotI32I32I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult DotI32I32I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void DotI32I32I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr BatchMatmulI8I8I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr BatchMatmulI8I8I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3;
    bindDims(context, d0, d1, d2, d3);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto map0 = AffineMap::get(4, 4, {d0, d1, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(4, 4, {d0, d3, d2}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(4, 4, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void BatchMatmulI8I8I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void BatchMatmulI8I8I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult BatchMatmulI8I8I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void BatchMatmulI8I8I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr BatchMatmulI16I16I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr BatchMatmulI16I16I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3;
    bindDims(context, d0, d1, d2, d3);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto map0 = AffineMap::get(4, 4, {d0, d1, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(4, 4, {d0, d3, d2}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(4, 4, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void BatchMatmulI16I16I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_sexti32(_0);
    Value _5 = std_sexti32(_1);
    Value _6 = std_muli(_4, _5);
    Value _7 = std_addi(_2, _6);
    (linalg_yield(ValueRange{ _7 }));
  }
    void BatchMatmulI16I16I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult BatchMatmulI16I16I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void BatchMatmulI16I16I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr BatchMatmulI32I32I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr BatchMatmulI32I32I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3;
    bindDims(context, d0, d1, d2, d3);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto map0 = AffineMap::get(4, 4, {d0, d1, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(4, 4, {d0, d3, d2}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(4, 4, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void BatchMatmulI32I32I32Op::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_muli(_0, _1);
    Value _5 = std_addi(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void BatchMatmulI32I32I32Op::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult BatchMatmulI32I32I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void BatchMatmulI32I32I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvWOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvWOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto map0 = AffineMap::get(2, 2, {d0 + d1}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(2, 2, {d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(2, 2, {d0}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1 }, 2, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvWOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvWOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvWOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvWOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvNWCOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvNWCOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4;
    bindDims(context, d0, d1, d2, d3, d4);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto map0 = AffineMap::get(5, 5, {d0, d1 + d3, d4}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4 }, 5, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(5, 5, {d2, d3, d4}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4 }, 5, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(5, 5, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4 }, 5, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvNWCOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvNWCOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvNWCOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvNWCOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvNCWOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvNCWOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4;
    bindDims(context, d0, d1, d2, d3, d4);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto map0 = AffineMap::get(5, 5, {d0, d4, d2 + d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4 }, 5, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(5, 5, {d1, d4, d3}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4 }, 5, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(5, 5, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4 }, 5, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvNCWOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvNCWOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvNCWOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvNCWOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvHWOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvHWOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3;
    bindDims(context, d0, d1, d2, d3);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto map0 = AffineMap::get(4, 4, {d0 + d2, d1 + d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(4, 4, {d2, d3}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(4, 4, {d0, d1}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3 }, 4, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvHWOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvHWOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvHWOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvHWOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvNHWCOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvNHWCOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto map0 = AffineMap::get(7, 7, {d0, d1 + d4, d2 + d5, d6}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6 }, 7, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(7, 7, {d3, d4, d5, d6}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6 }, 7, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(7, 7, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6 }, 7, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvNHWCOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvNHWCOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvNHWCOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvNHWCOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvNCHWOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvNCHWOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto map0 = AffineMap::get(7, 7, {d0, d6, d2 + d4, d3 + d5}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6 }, 7, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(7, 7, {d1, d6, d4, d5}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6 }, 7, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(7, 7, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6 }, 7, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvNCHWOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvNCHWOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvNCHWOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvNCHWOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvDHWOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvDHWOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5;
    bindDims(context, d0, d1, d2, d3, d4, d5);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto map0 = AffineMap::get(6, 6, {d0 + d3, d1 + d4, d2 + d5}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5 }, 6, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(6, 6, {d3, d4, d5}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5 }, 6, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(6, 6, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5 }, 6, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvDHWOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvDHWOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvDHWOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvDHWOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvNDHWCOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvNDHWCOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6, d7, d8;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6, d7, d8);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto map0 = AffineMap::get(9, 9, {d0, d1 + d5, d2 + d6, d3 + d7, d8}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8 }, 9, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(9, 9, {d4, d5, d6, d7, d8}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8 }, 9, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(9, 9, {d0, d1, d2, d3, d4}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8 }, 9, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvNDHWCOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvNDHWCOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvNDHWCOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvNDHWCOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvNCDHWOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvNCDHWOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6, d7, d8;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6, d7, d8);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto map0 = AffineMap::get(9, 9, {d0, d8, d2 + d5, d3 + d6, d4 + d7}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8 }, 9, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(9, 9, {d1, d8, d5, d6, d7}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8 }, 9, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(9, 9, {d0, d1, d2, d3, d4}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8 }, 9, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvNCDHWOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvNCDHWOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvNCDHWOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvNCDHWOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr DepthwiseConvInputNHWCFilterHWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  bool DepthwiseConvInputNHWCFilterHWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult DepthwiseConvInputNHWCFilterHWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr DepthwiseConvInputNHWCFilterHWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(7, 11, {d0, d1 * s9 + d5, d2 * s10 + d6, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1 }, 7, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(7, 11, {d5, d6, d3, d4}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1 }, 7, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(7, 9, {d0, d1, d2, d3, d4}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1 }, 7, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void DepthwiseConvInputNHWCFilterHWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void DepthwiseConvInputNHWCFilterHWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult DepthwiseConvInputNHWCFilterHWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void DepthwiseConvInputNHWCFilterHWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr DepthwiseConvInputNHWCFilterHWCOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  bool DepthwiseConvInputNHWCFilterHWCOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult DepthwiseConvInputNHWCFilterHWCOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr DepthwiseConvInputNHWCFilterHWCOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5;
    bindDims(context, d0, d1, d2, d3, d4, d5);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(6, 10, {d0, d1 * s8 + d4, d2 * s9 + d5, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1 }, 6, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(6, 10, {d4, d5, d3}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1 }, 6, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(6, 8, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1 }, 6, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void DepthwiseConvInputNHWCFilterHWCOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void DepthwiseConvInputNHWCFilterHWCOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult DepthwiseConvInputNHWCFilterHWCOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void DepthwiseConvInputNHWCFilterHWCOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvInputNWCFilterWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  bool ConvInputNWCFilterWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult ConvInputNWCFilterWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvInputNWCFilterWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4;
    bindDims(context, d0, d1, d2, d3, d4);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto map0 = AffineMap::get(5, 7, {d0, d1 * s5 + d3 * s6, d4}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, cst0, cst1 }, 5, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(5, 7, {d3, d4, d2}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, cst0, cst1 }, 5, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(5, 5, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, cst0, cst1 }, 5, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvInputNWCFilterWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvInputNWCFilterWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvInputNWCFilterWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvInputNWCFilterWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvInputNCWFilterWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  bool ConvInputNCWFilterWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult ConvInputNCWFilterWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvInputNCWFilterWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4;
    bindDims(context, d0, d1, d2, d3, d4);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto map0 = AffineMap::get(5, 7, {d0, d4, d2 * s5 + d3 * s6}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, cst0, cst1 }, 5, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(5, 7, {d3, d4, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, cst0, cst1 }, 5, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(5, 5, {d0, d1, d2}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, cst0, cst1 }, 5, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvInputNCWFilterWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvInputNCWFilterWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvInputNCWFilterWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvInputNCWFilterWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvInputNHWCFilterHWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  bool ConvInputNHWCFilterHWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult ConvInputNHWCFilterHWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvInputNHWCFilterHWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(7, 11, {d0, d1 * s7 + d4 * s8, d2 * s9 + d5 * s10, d6}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, cst0, cst1, cst2, cst3 }, 7, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(7, 11, {d4, d5, d6, d3}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, cst0, cst1, cst2, cst3 }, 7, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(7, 7, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, cst0, cst1, cst2, cst3 }, 7, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvInputNHWCFilterHWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvInputNHWCFilterHWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvInputNHWCFilterHWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvInputNHWCFilterHWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvInputNCHWFilterHWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  bool ConvInputNCHWFilterHWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult ConvInputNCHWFilterHWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvInputNCHWFilterHWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(7, 11, {d0, d6, d2 * s7 + d4 * s8, d3 * s9 + d5 * s10}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, cst0, cst1, cst2, cst3 }, 7, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(7, 11, {d4, d5, d6, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, cst0, cst1, cst2, cst3 }, 7, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(7, 7, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, cst0, cst1, cst2, cst3 }, 7, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvInputNCHWFilterHWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvInputNCHWFilterHWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvInputNCHWFilterHWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvInputNCHWFilterHWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvInputNDHWCFilterDHWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  bool ConvInputNDHWCFilterDHWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult ConvInputNDHWCFilterDHWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvInputNDHWCFilterDHWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6, d7, d8;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6, d7, d8);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto s11 = getAffineSymbolExpr(11, context); (void)s11;
	auto s12 = getAffineSymbolExpr(12, context); (void)s12;
	auto s13 = getAffineSymbolExpr(13, context); (void)s13;
	auto s14 = getAffineSymbolExpr(14, context); (void)s14;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto cst4 = getAffineConstantExpr(strides().getValue<int64_t>({ 2 }), context);
	auto cst5 = getAffineConstantExpr(dilations().getValue<int64_t>({ 2 }), context);
	auto map0 = AffineMap::get(9, 15, {d0, d1 * s9 + d5 * s10, d2 * s11 + d6 * s12, d3 * s13 + d7 * s14, d8}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1, cst2, cst3, cst4, cst5 }, 9, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(9, 15, {d5, d6, d7, d8, d4}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1, cst2, cst3, cst4, cst5 }, 9, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(9, 9, {d0, d1, d2, d3, d4}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1, cst2, cst3, cst4, cst5 }, 9, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvInputNDHWCFilterDHWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvInputNDHWCFilterDHWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvInputNDHWCFilterDHWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvInputNDHWCFilterDHWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr ConvInputNCDHWFilterDHWCFOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
    }
  bool ConvInputNCDHWFilterDHWCFOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult ConvInputNCDHWFilterDHWCFOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr ConvInputNCDHWFilterDHWCFOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5, d6, d7, d8;
    bindDims(context, d0, d1, d2, d3, d4, d5, d6, d7, d8);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto s11 = getAffineSymbolExpr(11, context); (void)s11;
	auto s12 = getAffineSymbolExpr(12, context); (void)s12;
	auto s13 = getAffineSymbolExpr(13, context); (void)s13;
	auto s14 = getAffineSymbolExpr(14, context); (void)s14;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto cst4 = getAffineConstantExpr(strides().getValue<int64_t>({ 2 }), context);
	auto cst5 = getAffineConstantExpr(dilations().getValue<int64_t>({ 2 }), context);
	auto map0 = AffineMap::get(9, 15, {d0, d8, d2 * s9 + d5 * s10, d3 * s11 + d6 * s12, d4 * s13 + d7 * s14}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1, cst2, cst3, cst4, cst5 }, 9, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(9, 15, {d5, d6, d7, d8, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1, cst2, cst3, cst4, cst5 }, 9, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(9, 9, {d0, d1, d2, d3, d4}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, s8, cst0, cst1, cst2, cst3, cst4, cst5 }, 9, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void ConvInputNCDHWFilterDHWCFOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = std_mulf(_0, _1);
    Value _5 = std_addf(_2, _4);
    (linalg_yield(ValueRange{ _5 }));
  }
    void ConvInputNCDHWFilterDHWCFOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult ConvInputNCDHWFilterDHWCFOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void ConvInputNCDHWFilterDHWCFOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr PoolingNHWCSumOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  bool PoolingNHWCSumOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult PoolingNHWCSumOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr PoolingNHWCSumOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5;
    bindDims(context, d0, d1, d2, d3, d4, d5);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto s11 = getAffineSymbolExpr(11, context); (void)s11;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(6, 12, {d0, d1 * s8 + d4 * s9, d2 * s10 + d5 * s11, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(6, 12, {d4, d5}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(6, 8, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void PoolingNHWCSumOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _2(args[2]);
    
    Value _4 = std_addf(_2, _0);
    (linalg_yield(ValueRange{ _4 }));
  }
    void PoolingNHWCSumOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult PoolingNHWCSumOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void PoolingNHWCSumOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr PoolingNHWCMaxOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  bool PoolingNHWCMaxOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult PoolingNHWCMaxOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr PoolingNHWCMaxOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5;
    bindDims(context, d0, d1, d2, d3, d4, d5);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto s11 = getAffineSymbolExpr(11, context); (void)s11;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(6, 12, {d0, d1 * s8 + d4 * s9, d2 * s10 + d5 * s11, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(6, 12, {d4, d5}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(6, 12, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void PoolingNHWCMaxOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _2(args[2]);
    
    Value _4 = std_cmpf_ogt(_0, _2);
    Value _5 = std_select(_4, _0, _2);
    (linalg_yield(ValueRange{ _5 }));
  }
    void PoolingNHWCMaxOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult PoolingNHWCMaxOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void PoolingNHWCMaxOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }

    ArrayAttr PoolingNHWCMinOp::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
    }
  bool PoolingNHWCMinOp::hasDynamicIndexingMaps() { return true; }

  LogicalResult PoolingNHWCMinOp::verifyIndexingMapRequiredAttributes() {
    Operation *op = getOperation();
    
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'dilations'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'dilations'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'dilations'");
        }
      

        if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
          if (!attr.getType().getElementType().isInteger(64)) return op->emitError(
            "incorrect element type for indexing map required attribute 'strides'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
            return op->emitError(
              "incorrect shape for indexing map required attribute 'strides'");
        } else {
          return op->emitError(
            "missing indexing map required attribute 'strides'");
        }
      
    return success();
  }
  
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr PoolingNHWCMinOp::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2, d3, d4, d5;
    bindDims(context, d0, d1, d2, d3, d4, d5);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto s3 = getAffineSymbolExpr(3, context); (void)s3;
	auto s4 = getAffineSymbolExpr(4, context); (void)s4;
	auto s5 = getAffineSymbolExpr(5, context); (void)s5;
	auto s6 = getAffineSymbolExpr(6, context); (void)s6;
	auto s7 = getAffineSymbolExpr(7, context); (void)s7;
	auto s8 = getAffineSymbolExpr(8, context); (void)s8;
	auto s9 = getAffineSymbolExpr(9, context); (void)s9;
	auto s10 = getAffineSymbolExpr(10, context); (void)s10;
	auto s11 = getAffineSymbolExpr(11, context); (void)s11;
	auto cst0 = getAffineConstantExpr(strides().getValue<int64_t>({ 0 }), context);
	auto cst1 = getAffineConstantExpr(dilations().getValue<int64_t>({ 0 }), context);
	auto cst2 = getAffineConstantExpr(strides().getValue<int64_t>({ 1 }), context);
	auto cst3 = getAffineConstantExpr(dilations().getValue<int64_t>({ 1 }), context);
	auto map0 = AffineMap::get(6, 12, {d0, d1 * s8 + d4 * s9, d2 * s10 + d5 * s11, d3}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(6, 12, {d4, d5}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(6, 12, {d0, d1, d2, d3}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2, s3, s4, s5, s6, s7, cst0, cst1, cst2, cst3 }, 6, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void PoolingNHWCMinOp::regionBuilder(Block &block, ValueRange captures) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value _0(args[0]), _2(args[2]);
    
    Value _4 = std_cmpf_olt(_0, _2);
    Value _5 = std_select(_4, _0, _2);
    (linalg_yield(ValueRange{ _5 }));
  }
    void PoolingNHWCMinOp::getCanonicalizationPatterns(
        RewritePatternSet &results,
        MLIRContext *context) {
      results.add<EraseDeadLinalgOp>();
      results.add<FoldTensorCastOp>();
    }
    LogicalResult PoolingNHWCMinOp::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void PoolingNHWCMinOp::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    }
