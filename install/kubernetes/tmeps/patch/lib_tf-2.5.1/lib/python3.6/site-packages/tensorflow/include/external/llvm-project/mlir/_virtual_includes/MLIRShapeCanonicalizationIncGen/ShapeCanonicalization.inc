/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td:17
*/
struct AssumingAllOneOp : public ::mlir::RewritePattern {
  AssumingAllOneOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.assuming_all", {}, 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range args(op0->getOperands());
    ::mlir::Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::AssumingAllOp>(op0); (void)castedOp0;
    args = castedOp0.getODSOperands(0);
    if (!((
      args.size() == 1
    ))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'args' failed to satisfy constraint: ";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ args }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td:21
*/
struct CstrBroadcastableEqOps : public ::mlir::RewritePattern {
  CstrBroadcastableEqOps(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.cstr_broadcastable", {"shape.const_witness"}, 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range shapes(op0->getOperands());
    ::mlir::shape::CstrBroadcastableOp op;
    ::mlir::Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::CstrBroadcastableOp>(op0); (void)castedOp0;
    op = castedOp0;
    shapes = castedOp0.getODSOperands(0);
    if (!((
      llvm::all_of(shapes, [&](mlir::Value val) {
        return shapes[0] == val;
      })
    ))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'shapes' failed to satisfy constraint: ";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::ConstWitnessOp tblgen_ConstWitnessOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = rewriter.getBoolAttr(true)) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("passing"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ConstWitnessOp_0 = rewriter.create<::mlir::shape::ConstWitnessOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ConstWitnessOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td:25
*/
struct CstrEqEqOps : public ::mlir::RewritePattern {
  CstrEqEqOps(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.cstr_eq", {"shape.const_witness"}, 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range shapes(op0->getOperands());
    ::mlir::shape::CstrEqOp op;
    ::mlir::Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::CstrEqOp>(op0); (void)castedOp0;
    op = castedOp0;
    shapes = castedOp0.getODSOperands(0);
    if (!((
      llvm::all_of(shapes, [&](mlir::Value val) {
        return shapes[0] == val;
      })
    ))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'shapes' failed to satisfy constraint: ";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::ConstWitnessOp tblgen_ConstWitnessOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = rewriter.getBoolAttr(true)) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("passing"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ConstWitnessOp_0 = rewriter.create<::mlir::shape::ConstWitnessOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ConstWitnessOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td:29
*/
struct IndexToSizeToIndexCanonicalization : public ::mlir::RewritePattern {
  IndexToSizeToIndexCanonicalization(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.size_to_index", {}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::SizeToIndexOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::shape::IndexToSizeOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      arg = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ arg }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td:33
*/
struct SizeToIndexToSizeCanonicalization : public ::mlir::RewritePattern {
  SizeToIndexToSizeCanonicalization(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.index_to_size", {}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::IndexToSizeOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::shape::SizeToIndexOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      arg = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ arg }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td:39
*/
struct TensorCastConstShape : public ::mlir::RewritePattern {
  TensorCastConstShape(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tensor.cast", {"shape.const_shape"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::DenseIntElementsAttr arg;
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::tensor::CastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::shape::ConstShapeOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::DenseIntElementsAttr>("shape");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'shape.const_shape' to have attribute 'shape' of type '::mlir::DenseIntElementsAttr'";
          });
        }
        arg = tblgen_attr;
      }
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::ConstShapeOp tblgen_ConstShapeOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = arg) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("shape"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ConstShapeOp_0 = rewriter.create<::mlir::shape::ConstShapeOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ConstShapeOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<AssumingAllOneOp>(patterns.getContext());
  patterns.add<CstrBroadcastableEqOps>(patterns.getContext());
  patterns.add<CstrEqEqOps>(patterns.getContext());
  patterns.add<IndexToSizeToIndexCanonicalization>(patterns.getContext());
  patterns.add<SizeToIndexToSizeCanonicalization>(patterns.getContext());
  patterns.add<TensorCastConstShape>(patterns.getContext());
}
