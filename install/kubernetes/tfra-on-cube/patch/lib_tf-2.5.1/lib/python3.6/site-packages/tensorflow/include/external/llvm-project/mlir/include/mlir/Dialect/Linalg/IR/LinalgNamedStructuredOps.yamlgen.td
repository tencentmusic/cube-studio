
//===----------------------------------------------------------------------===//
// Op definition for MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : LinalgStructuredBase_Op<"matmul", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
      let summary = [{ Performs a matrix multiplacation of two 2D inputs. }];
      let description = [{
        Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
      }];
    
    let arguments = (ins
      Variadic<AnyShaped>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<MatmulOp>(parser, result/*TODO:, captures*/);
    }];
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(Block &block, ValueRange captures);
      static std::function<void(Block &, ValueRange)> getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulOp
//===----------------------------------------------------------------------===//

def BatchMatmulOp : LinalgStructuredBase_Op<"batch_matmul", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
      let summary = [{ Performs a batched matrix multiplacation of two 3D inputs. }];
      let description = [{
        Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
      }];
    
    let arguments = (ins
      Variadic<AnyShaped>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result/*TODO:, captures*/);
    }];
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(Block &block, ValueRange captures);
      static std::function<void(Block &, ValueRange)> getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatvecOp
//===----------------------------------------------------------------------===//

def MatvecOp : LinalgStructuredBase_Op<"matvec", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
      let summary = [{ Performs a matrix-vector multiplication. }];
      let description = [{
        Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
      }];
    
    let arguments = (ins
      Variadic<AnyShaped>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<MatvecOp>(parser, result/*TODO:, captures*/);
    }];
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(Block &block, ValueRange captures);
      static std::function<void(Block &, ValueRange)> getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for VecmatOp
//===----------------------------------------------------------------------===//

def VecmatOp : LinalgStructuredBase_Op<"vecmat", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
      let summary = [{ Performs a vector-matrix multiplacation. }];
      let description = [{
        Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
      }];
    
    let arguments = (ins
      Variadic<AnyShaped>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<VecmatOp>(parser, result/*TODO:, captures*/);
    }];
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(Block &block, ValueRange captures);
      static std::function<void(Block &, ValueRange)> getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DotOp
//===----------------------------------------------------------------------===//

def DotOp : LinalgStructuredBase_Op<"dot", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
      let summary = [{ Performs a dot product of two vectors to a scalar result. }];
      let description = [{
        Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
      }];
    
    let arguments = (ins
      Variadic<AnyShaped>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<DotOp>(parser, result/*TODO:, captures*/);
    }];
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(Block &block, ValueRange captures);
      static std::function<void(Block &, ValueRange)> getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}
