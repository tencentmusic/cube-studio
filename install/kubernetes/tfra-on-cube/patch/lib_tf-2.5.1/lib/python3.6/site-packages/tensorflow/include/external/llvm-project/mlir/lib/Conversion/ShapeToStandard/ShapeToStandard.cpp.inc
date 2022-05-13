/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/llvm-project/mlir/lib/Conversion/ShapeToStandard/ShapeToStandard.td:22
*/
struct CstrBroadcastableToRequire : public ::mlir::RewritePattern {
  CstrBroadcastableToRequire(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.cstr_broadcastable", {"shape.cstr_require", "shape.is_broadcastable"}, 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range shapes(op0->getOperands());
    ::mlir::Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::CstrBroadcastableOp>(op0); (void)castedOp0;
    shapes = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::IsBroadcastableOp tblgen_IsBroadcastableOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values_0;
      for (auto v: shapes) {
        tblgen_values_0.push_back(v);
      }
      tblgen_IsBroadcastableOp_0 = rewriter.create<::mlir::shape::IsBroadcastableOp>(odsLoc,
        /*shapes=*/tblgen_values_0
      );
    }
    ::mlir::shape::CstrRequireOp tblgen_CstrRequireOp_1;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_IsBroadcastableOp_0.getODSResults(0).begin()));
      if (auto tmpAttr = 
        rewriter.getStringAttr("required broadcastable shapes")
      ) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("msg"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CstrRequireOp_1 = rewriter.create<::mlir::shape::CstrRequireOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CstrRequireOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Conversion/ShapeToStandard/ShapeToStandard.td:31
*/
struct CstrEqToRequire : public ::mlir::RewritePattern {
  CstrEqToRequire(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.cstr_eq", {"shape.cstr_require", "shape.shape_eq"}, 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range shapes(op0->getOperands());
    ::mlir::Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::shape::CstrEqOp>(op0); (void)castedOp0;
    shapes = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::ShapeEqOp tblgen_ShapeEqOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values_0;
      for (auto v: shapes) {
        tblgen_values_0.push_back(v);
      }
      tblgen_ShapeEqOp_0 = rewriter.create<::mlir::shape::ShapeEqOp>(odsLoc,
        /*shapes=*/tblgen_values_0
      );
    }
    ::mlir::shape::CstrRequireOp tblgen_CstrRequireOp_1;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ShapeEqOp_0.getODSResults(0).begin()));
      if (auto tmpAttr = 
        rewriter.getStringAttr("required equal shapes")
      ) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("msg"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CstrRequireOp_1 = rewriter.create<::mlir::shape::CstrRequireOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CstrRequireOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<CstrBroadcastableToRequire>(patterns.getContext());
  patterns.add<CstrEqToRequire>(patterns.getContext());
}
