/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:19
*/
struct ConvertChainedBitcast : public ::mlir::RewritePattern {
  ConvertChainedBitcast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Bitcast", {"spv.Bitcast"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::BitcastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::BitcastOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      operand = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::BitcastOp tblgen_BitcastOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*operand.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BitcastOp_0 = rewriter.create<::mlir::spirv::BitcastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BitcastOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:56
*/
struct ConvertComparisonIntoClampSPV_FOrdLessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClampSPV_FOrdLessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Select", {"spv.GLSL.FClamp"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::FOrdLessThanEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!castedOp2) return ::mlir::failure();
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          auto castedOp3 = ::llvm::dyn_cast_or_null<::mlir::spirv::FOrdLessThanEqualOp>(op3); (void)castedOp3;
          if (!castedOp3) return ::mlir::failure();
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops[1] = op3;
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops[2] = op2;
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops[3] = op1;
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSLFClampOp tblgen_GLSLFClampOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSLFClampOp_0 = rewriter.create<::mlir::spirv::GLSLFClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSLFClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:56
*/
struct ConvertComparisonIntoClampSPV_FOrdLessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClampSPV_FOrdLessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Select", {"spv.GLSL.FClamp"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::FOrdLessThanOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!castedOp2) return ::mlir::failure();
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          auto castedOp3 = ::llvm::dyn_cast_or_null<::mlir::spirv::FOrdLessThanOp>(op3); (void)castedOp3;
          if (!castedOp3) return ::mlir::failure();
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops[1] = op3;
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops[2] = op2;
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops[3] = op1;
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSLFClampOp tblgen_GLSLFClampOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSLFClampOp_0 = rewriter.create<::mlir::spirv::GLSLFClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSLFClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:56
*/
struct ConvertComparisonIntoClampSPV_SLessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClampSPV_SLessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Select", {"spv.GLSL.SClamp"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::SLessThanEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!castedOp2) return ::mlir::failure();
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          auto castedOp3 = ::llvm::dyn_cast_or_null<::mlir::spirv::SLessThanEqualOp>(op3); (void)castedOp3;
          if (!castedOp3) return ::mlir::failure();
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops[1] = op3;
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops[2] = op2;
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops[3] = op1;
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSLSClampOp tblgen_GLSLSClampOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSLSClampOp_0 = rewriter.create<::mlir::spirv::GLSLSClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSLSClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:56
*/
struct ConvertComparisonIntoClampSPV_SLessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClampSPV_SLessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Select", {"spv.GLSL.SClamp"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::SLessThanOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!castedOp2) return ::mlir::failure();
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          auto castedOp3 = ::llvm::dyn_cast_or_null<::mlir::spirv::SLessThanOp>(op3); (void)castedOp3;
          if (!castedOp3) return ::mlir::failure();
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops[1] = op3;
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops[2] = op2;
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops[3] = op1;
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSLSClampOp tblgen_GLSLSClampOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSLSClampOp_0 = rewriter.create<::mlir::spirv::GLSLSClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSLSClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:56
*/
struct ConvertComparisonIntoClampSPV_ULessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClampSPV_ULessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Select", {"spv.GLSL.UClamp"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::ULessThanEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!castedOp2) return ::mlir::failure();
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          auto castedOp3 = ::llvm::dyn_cast_or_null<::mlir::spirv::ULessThanEqualOp>(op3); (void)castedOp3;
          if (!castedOp3) return ::mlir::failure();
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops[1] = op3;
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops[2] = op2;
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops[3] = op1;
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSLUClampOp tblgen_GLSLUClampOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSLUClampOp_0 = rewriter.create<::mlir::spirv::GLSLUClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSLUClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:56
*/
struct ConvertComparisonIntoClampSPV_ULessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClampSPV_ULessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.Select", {"spv.GLSL.UClamp"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::ULessThanOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!castedOp2) return ::mlir::failure();
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          auto castedOp3 = ::llvm::dyn_cast_or_null<::mlir::spirv::ULessThanOp>(op3); (void)castedOp3;
          if (!castedOp3) return ::mlir::failure();
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops[1] = op3;
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops[2] = op2;
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops[3] = op1;
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSLUClampOp tblgen_GLSLUClampOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSLUClampOp_0 = rewriter.create<::mlir::spirv::GLSLUClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSLUClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:26
*/
struct ConvertLogicalNotOfIEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfIEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.LogicalNot", {"spv.INotEqual"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::IEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::INotEqualOp tblgen_INotEqualOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_INotEqualOp_0 = rewriter.create<::mlir::spirv::INotEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_INotEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:30
*/
struct ConvertLogicalNotOfINotEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfINotEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.LogicalNot", {"spv.IEqual"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::INotEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::IEqualOp tblgen_IEqualOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IEqualOp_0 = rewriter.create<::mlir::spirv::IEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:34
*/
struct ConvertLogicalNotOfLogicalEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfLogicalEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.LogicalNot", {"spv.LogicalNotEqual"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::LogicalEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::LogicalNotEqualOp tblgen_LogicalNotEqualOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_LogicalNotEqualOp_0 = rewriter.create<::mlir::spirv::LogicalNotEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_LogicalNotEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:38
*/
struct ConvertLogicalNotOfLogicalNotEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfLogicalNotEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spv.LogicalNot", {"spv.LogicalEqual"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::spirv::LogicalNotEqualOp>(op1); (void)castedOp1;
      if (!castedOp1) return ::mlir::failure();
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::LogicalEqualOp tblgen_LogicalEqualOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_LogicalEqualOp_0 = rewriter.create<::mlir::spirv::LogicalEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_LogicalEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertChainedBitcast>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClampSPV_FOrdLessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClampSPV_FOrdLessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClampSPV_SLessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClampSPV_SLessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClampSPV_ULessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClampSPV_ULessThanOp>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfIEqual>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfINotEqual>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfLogicalEqual>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfLogicalNotEqual>(patterns.getContext());
}
