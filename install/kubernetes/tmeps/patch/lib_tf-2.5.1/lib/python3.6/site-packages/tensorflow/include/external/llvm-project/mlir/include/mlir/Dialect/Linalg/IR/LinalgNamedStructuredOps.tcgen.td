  def MatmulColumnMajorOp : LinalgStructuredBase_Op<"matmul_column_major", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatmulColumnMajorOp>(
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
          createAndFillStructuredOpRegion<MatmulColumnMajorOp>(
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
        return ::parseNamedStructuredOp<MatmulColumnMajorOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def MatmulI8I8I32Op : LinalgStructuredBase_Op<"matmul_i8_i8_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
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
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
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
        return ::parseNamedStructuredOp<MatmulI8I8I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def MatmulI16I16I32Op : LinalgStructuredBase_Op<"matmul_i16_i16_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatmulI16I16I32Op>(
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
          createAndFillStructuredOpRegion<MatmulI16I16I32Op>(
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
        return ::parseNamedStructuredOp<MatmulI16I16I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def MatmulI32I32I32Op : LinalgStructuredBase_Op<"matmul_i32_i32_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatmulI32I32I32Op>(
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
          createAndFillStructuredOpRegion<MatmulI32I32I32Op>(
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
        return ::parseNamedStructuredOp<MatmulI32I32I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def MatvecI8I8I32Op : LinalgStructuredBase_Op<"matvec_i8_i8_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatvecI8I8I32Op>(
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
          createAndFillStructuredOpRegion<MatvecI8I8I32Op>(
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
        return ::parseNamedStructuredOp<MatvecI8I8I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def MatvecI16I16I32Op : LinalgStructuredBase_Op<"matvec_i16_i16_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatvecI16I16I32Op>(
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
          createAndFillStructuredOpRegion<MatvecI16I16I32Op>(
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
        return ::parseNamedStructuredOp<MatvecI16I16I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def MatvecI32I32I32Op : LinalgStructuredBase_Op<"matvec_i32_i32_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<MatvecI32I32I32Op>(
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
          createAndFillStructuredOpRegion<MatvecI32I32I32Op>(
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
        return ::parseNamedStructuredOp<MatvecI32I32I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def VecmatI8I8I32Op : LinalgStructuredBase_Op<"vecmat_i8_i8_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<VecmatI8I8I32Op>(
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
          createAndFillStructuredOpRegion<VecmatI8I8I32Op>(
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
        return ::parseNamedStructuredOp<VecmatI8I8I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def VecmatI16I16I32Op : LinalgStructuredBase_Op<"vecmat_i16_i16_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<VecmatI16I16I32Op>(
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
          createAndFillStructuredOpRegion<VecmatI16I16I32Op>(
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
        return ::parseNamedStructuredOp<VecmatI16I16I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def VecmatI32I32I32Op : LinalgStructuredBase_Op<"vecmat_i32_i32_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<VecmatI32I32I32Op>(
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
          createAndFillStructuredOpRegion<VecmatI32I32I32Op>(
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
        return ::parseNamedStructuredOp<VecmatI32I32I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def DotI8I8I32Op : LinalgStructuredBase_Op<"dot_i8_i8_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<DotI8I8I32Op>(
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
          createAndFillStructuredOpRegion<DotI8I8I32Op>(
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
        return ::parseNamedStructuredOp<DotI8I8I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def DotI16I16I32Op : LinalgStructuredBase_Op<"dot_i16_i16_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<DotI16I16I32Op>(
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
          createAndFillStructuredOpRegion<DotI16I16I32Op>(
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
        return ::parseNamedStructuredOp<DotI16I16I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def DotI32I32I32Op : LinalgStructuredBase_Op<"dot_i32_i32_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<DotI32I32I32Op>(
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
          createAndFillStructuredOpRegion<DotI32I32I32Op>(
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
        return ::parseNamedStructuredOp<DotI32I32I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def BatchMatmulI8I8I32Op : LinalgStructuredBase_Op<"batch_matmul_i8_i8_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<BatchMatmulI8I8I32Op>(
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
          createAndFillStructuredOpRegion<BatchMatmulI8I8I32Op>(
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
        return ::parseNamedStructuredOp<BatchMatmulI8I8I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def BatchMatmulI16I16I32Op : LinalgStructuredBase_Op<"batch_matmul_i16_i16_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<BatchMatmulI16I16I32Op>(
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
          createAndFillStructuredOpRegion<BatchMatmulI16I16I32Op>(
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
        return ::parseNamedStructuredOp<BatchMatmulI16I16I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def BatchMatmulI32I32I32Op : LinalgStructuredBase_Op<"batch_matmul_i32_i32_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
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
          createAndFillStructuredOpRegion<BatchMatmulI32I32I32Op>(
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
          createAndFillStructuredOpRegion<BatchMatmulI32I32I32Op>(
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
        return ::parseNamedStructuredOp<BatchMatmulI32I32I32Op>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvWOp : LinalgStructuredBase_Op<"conv_1d", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvWOp>(
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
          createAndFillStructuredOpRegion<ConvWOp>(
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
        return ::parseNamedStructuredOp<ConvWOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvNWCOp : LinalgStructuredBase_Op<"conv_1d_nwc", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvNWCOp>(
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
          createAndFillStructuredOpRegion<ConvNWCOp>(
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
        return ::parseNamedStructuredOp<ConvNWCOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvNCWOp : LinalgStructuredBase_Op<"conv_1d_ncw", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvNCWOp>(
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
          createAndFillStructuredOpRegion<ConvNCWOp>(
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
        return ::parseNamedStructuredOp<ConvNCWOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvHWOp : LinalgStructuredBase_Op<"conv_2d", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvHWOp>(
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
          createAndFillStructuredOpRegion<ConvHWOp>(
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
        return ::parseNamedStructuredOp<ConvHWOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvNHWCOp : LinalgStructuredBase_Op<"conv_2d_nhwc", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvNHWCOp>(
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
          createAndFillStructuredOpRegion<ConvNHWCOp>(
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
        return ::parseNamedStructuredOp<ConvNHWCOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvNCHWOp : LinalgStructuredBase_Op<"conv_2d_nchw", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvNCHWOp>(
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
          createAndFillStructuredOpRegion<ConvNCHWOp>(
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
        return ::parseNamedStructuredOp<ConvNCHWOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvDHWOp : LinalgStructuredBase_Op<"conv_3d", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvDHWOp>(
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
          createAndFillStructuredOpRegion<ConvDHWOp>(
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
        return ::parseNamedStructuredOp<ConvDHWOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvNDHWCOp : LinalgStructuredBase_Op<"conv_3d_ndhwc", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvNDHWCOp>(
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
          createAndFillStructuredOpRegion<ConvNDHWCOp>(
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
        return ::parseNamedStructuredOp<ConvNDHWCOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def ConvNCDHWOp : LinalgStructuredBase_Op<"conv_3d_ncdhw", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
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
          createAndFillStructuredOpRegion<ConvNCDHWOp>(
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
          createAndFillStructuredOpRegion<ConvNCDHWOp>(
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
        return ::parseNamedStructuredOp<ConvNCDHWOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
  def DepthwiseConvInputNHWCFilterHWCFOp : LinalgStructuredBase_Op<"depthwise_conv_2d_input_nhwc_filter_hwcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A general depth-wise 2-D convolution operation. }];
      let description = [{
        This operation performs depth-wise 2-D convolution over an input `I` and filter
`F` and generates output `O` using the following computation:

```
  O(n, oh, ow, ci, co) = std_addf<kh, kw>(
      O(n, oh, ow, ci, co),
      std_mulf(I(n, oh * strides[0] + kh, ow * strides[1] + kw, ci),
               K(kh, kw, ci, co)));
```

where

* `I` is a 4-D tensor with shape `(N, IH, IW, CI)`.
* `F` is a 4-D tensor with shape `(KH, KW, CI, CO)`.
* `O` is a 5-D tensor with shape `(N, OH, OW, CI, CO)`.
* `strides` is a 2-element vector attribute for window strides along the
  height/width dimension.

The indexing maps for these three tensors contain 7 dimensions, following the
order of (`N`, `OH`, `OW`, `CI`, `CO`, `KH`, `KW`).

Note: this op only supports any channel multiplier, which is `CO`. To map back
to 4D result as DepthwiseConvInputNHWCFilterHWCOp, you will have to create a
Linalg reshape op which collapses `CI` and `CO` into one dimension.
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCFOp>(
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
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<DepthwiseConvInputNHWCFilterHWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def DepthwiseConvInputNHWCFilterHWCOp : LinalgStructuredBase_Op<"depthwise_conv_2d_input_nhwc_filter_hwc", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A depth-wise 2-D convolution operation. }];
      let description = [{
        This operation performs depth-wise 2-D convolution over an input `I` and filter
`F` and generates output `O` using the following computation:

```
O(n, oh, ow, c) = std_addf<kh, kw>(
    O(n, oh, ow, c),
    std_mulf(I(n, oh * strides[0] + kh, ow * strides[1] + kw, c),
             K(kh, kw, c)));
```

where

* `I` is a 4-D tensor with shape `(N, IH, IW, C)`.
* `F` is a 3-D tensor with shape `(KH, KW, C)`.
* `O` is a 4-D tensor with shape `(N, OH, OW, C)`.
* `strides` is a 2-element vector attribute for window strides along the
  height/width dimension.

The indexing maps for these three tensors contain 6 dimensions, following the
order of (`N`, `OH`, `OW`, `C`, `KH`, `KW`).

Note: this op only supports channel multiplier == 1.
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCOp>(
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
          createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConvInputNHWCFilterHWCOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<DepthwiseConvInputNHWCFilterHWCOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def ConvInputNWCFilterWCFOp : LinalgStructuredBase_Op<"conv_1d_input_nwc_filter_wcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A 1-D convolution given NWC layout input and WCF layout filter. }];
      let description = [{
        Computes a 1-D convolution given 3-D input and filter. The data layout
of input is NWC and the data layout of filter is WCF.

The indexing maps for these three tensors contain 5 dimensions, following the
order of (`N`, `W`, `F`, `KW`, `C`).
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[1]>:$dilations,
RankedI64ElementsAttr<[1]>:$strides
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
          createAndFillStructuredOpRegion<ConvInputNWCFilterWCFOp>(
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
          createAndFillStructuredOpRegion<ConvInputNWCFilterWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNWCFilterWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<ConvInputNWCFilterWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def ConvInputNCWFilterWCFOp : LinalgStructuredBase_Op<"conv_1d_input_ncw_filter_wcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A 1-D convolution given NCW layout input and WCF layout filter. }];
      let description = [{
        Computes a 1-D convolution given 3-D input and filter. The data layout
of input is NCW and the data layout of filter is WCF.

The indexing maps for these three tensors contain 5 dimensions, following the
order of (`N`, `F`, `W`, `KW`, `C`).
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[1]>:$dilations,
RankedI64ElementsAttr<[1]>:$strides
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
          createAndFillStructuredOpRegion<ConvInputNCWFilterWCFOp>(
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
          createAndFillStructuredOpRegion<ConvInputNCWFilterWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNCWFilterWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<ConvInputNCWFilterWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def ConvInputNHWCFilterHWCFOp : LinalgStructuredBase_Op<"conv_2d_input_nhwc_filter_hwcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A 2-D convolution given NHWC layout input and HWCF layout filter. }];
      let description = [{
        Computes a 2-D convolution given 4-D input and filter. The data layout
of input is NHWC and the data layout of filter is HWCF.

The indexing maps for these three tensors contain 7 dimensions, following the
order of (`N`, `H`, `W`, `F`, `KH`, `KW`, `C`).
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$dilations,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<ConvInputNHWCFilterHWCFOp>(
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
          createAndFillStructuredOpRegion<ConvInputNHWCFilterHWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNHWCFilterHWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<ConvInputNHWCFilterHWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def ConvInputNCHWFilterHWCFOp : LinalgStructuredBase_Op<"conv_2d_input_nchw_filter_hwcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A 2-D convolution given NCHW layout input and HWCF layout filter. }];
      let description = [{
        Computes a 2-D convolution given 4-D input and filter. The data layout
of input is NCHW and the data layout of filter is HWCF.

The indexing maps for these three tensors contain 7 dimensions, following the
order of (`N`, `F`, `H`, `W`, `KH`, `KW`, `C`).
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$dilations,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<ConvInputNCHWFilterHWCFOp>(
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
          createAndFillStructuredOpRegion<ConvInputNCHWFilterHWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNCHWFilterHWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<ConvInputNCHWFilterHWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def ConvInputNDHWCFilterDHWCFOp : LinalgStructuredBase_Op<"conv_3d_input_ndhwc_filter_dhwcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A 3-D convolution given NDHWC layout input and DHWCF layout filter. }];
      let description = [{
        Computes a 3-D convolution given 5-D input and filter. The data layout
of input is NDHWC and the data layout of filter is DHWCF.

The indexing maps for these three tensors contain 9 dimensions, following the
order of (`N`, `D`, `H`, `W`, `F`, `KD`, `KH`, `KW`, `C`).
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[3]>:$dilations,
RankedI64ElementsAttr<[3]>:$strides
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
          createAndFillStructuredOpRegion<ConvInputNDHWCFilterDHWCFOp>(
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
          createAndFillStructuredOpRegion<ConvInputNDHWCFilterDHWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNDHWCFilterDHWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<ConvInputNDHWCFilterDHWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def ConvInputNCDHWFilterDHWCFOp : LinalgStructuredBase_Op<"conv_3d_input_ncdhw_filter_dhwcf", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let summary = [{ A 3-D convolution given NCDHW layout input and DHWCF layout filter. }];
      let description = [{
        Computes a 3-D convolution given 5-D input and filter. The data layout
of input is NCDHW and the data layout of filter is DHWCF.

The indexing maps for these three tensors contain 9 dimensions, following the
order of (`N`, `F`, `D`, `H`, `W`, `KD`, `KH`, `KW`, `C`).
      }];
    
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[3]>:$dilations,
RankedI64ElementsAttr<[3]>:$strides
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
          createAndFillStructuredOpRegion<ConvInputNCDHWFilterDHWCFOp>(
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
          createAndFillStructuredOpRegion<ConvInputNCDHWFilterDHWCFOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ConvInputNCDHWFilterDHWCFOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<ConvInputNCDHWFilterDHWCFOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def PoolingNHWCSumOp : LinalgStructuredBase_Op<"pooling_nhwc_sum", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$dilations,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<PoolingNHWCSumOp>(
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
          createAndFillStructuredOpRegion<PoolingNHWCSumOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNHWCSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<PoolingNHWCSumOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def PoolingNHWCMaxOp : LinalgStructuredBase_Op<"pooling_nhwc_max", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$dilations,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<PoolingNHWCMaxOp>(
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
          createAndFillStructuredOpRegion<PoolingNHWCMaxOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNHWCMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<PoolingNHWCMaxOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
  def PoolingNHWCMinOp : LinalgStructuredBase_Op<"pooling_nhwc_min", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/]> {
      
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$dilations,
RankedI64ElementsAttr<[2]>:$strides
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
          createAndFillStructuredOpRegion<PoolingNHWCMinOp>(
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
          createAndFillStructuredOpRegion<PoolingNHWCMinOp>(
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
        
      , OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, "Attribute":$dilations, "Attribute":$strides),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNHWCMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        $_state.addAttribute("dilations", dilations);
$_state.addAttribute("strides", strides);
      }]>
    
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{
        return ::parseNamedStructuredOp<PoolingNHWCMinOp>(parser, result/*TODO:, captures*/);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
      }];
  }
