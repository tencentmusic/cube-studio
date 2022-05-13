/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Serialization Utilities/Functions                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_SERIALIZATION_FNS

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AccessChainOp>(::mlir::spirv::AccessChainOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAccessChain, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicAndOp>(::mlir::spirv::AtomicAndOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicAnd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(::mlir::spirv::AtomicCompareExchangeWeakOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("equal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("equal_semantics");
  }
  {
    if (auto attr = op->getAttr("unequal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("unequal_semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicCompareExchangeWeak, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIAddOp>(::mlir::spirv::AtomicIAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIAdd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIDecrementOp>(::mlir::spirv::AtomicIDecrementOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op->getAttr("semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("semantics");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIDecrement, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIIncrementOp>(::mlir::spirv::AtomicIIncrementOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op->getAttr("semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("semantics");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIIncrement, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicISubOp>(::mlir::spirv::AtomicISubOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicISub, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicOrOp>(::mlir::spirv::AtomicOrOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicOr, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicSMaxOp>(::mlir::spirv::AtomicSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicSMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicSMinOp>(::mlir::spirv::AtomicSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicSMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicUMaxOp>(::mlir::spirv::AtomicUMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicUMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicUMinOp>(::mlir::spirv::AtomicUMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicUMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicXorOp>(::mlir::spirv::AtomicXorOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicXor, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitCountOp>(::mlir::spirv::BitCountOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitCount));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldInsertOp>(::mlir::spirv::BitFieldInsertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitFieldInsert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldSExtractOp>(::mlir::spirv::BitFieldSExtractOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitFieldSExtract));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldUExtractOp>(::mlir::spirv::BitFieldUExtractOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitFieldUExtract));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitReverseOp>(::mlir::spirv::BitReverseOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitReverse));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitcastOp>(::mlir::spirv::BitcastOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitcast));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseAndOp>(::mlir::spirv::BitwiseAndOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitwiseAnd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseOrOp>(::mlir::spirv::BitwiseOrOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitwiseOr));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseXorOp>(::mlir::spirv::BitwiseXorOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitwiseXor));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeConstructOp>(::mlir::spirv::CompositeConstructOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeConstruct, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeExtractOp>(::mlir::spirv::CompositeExtractOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("indices")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("indices");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeExtract, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeInsertOp>(::mlir::spirv::CompositeInsertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("indices")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("indices");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeInsert, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertFToSOp>(::mlir::spirv::ConvertFToSOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertFToS));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertFToUOp>(::mlir::spirv::ConvertFToUOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertFToU));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertSToFOp>(::mlir::spirv::ConvertSToFOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertSToF));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertUToFOp>(::mlir::spirv::ConvertUToFOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertUToF));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixLengthNVOp>(::mlir::spirv::CooperativeMatrixLengthNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op->getAttr("type")) {
    operands.push_back(static_cast<uint32_t>(getTypeID(attr.cast<TypeAttr>().getValue())));
  }
  elidedAttrs.push_back("type");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixLengthNV, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixLoadNVOp>(::mlir::spirv::CooperativeMatrixLoadNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixLoadNV, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixMulAddNVOp>(::mlir::spirv::CooperativeMatrixMulAddNVOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpCooperativeMatrixMulAddNV));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixStoreNVOp>(::mlir::spirv::CooperativeMatrixStoreNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixStoreNV, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FAddOp>(::mlir::spirv::FAddOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFAdd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FConvertOp>(::mlir::spirv::FConvertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFConvert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FDivOp>(::mlir::spirv::FDivOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFDiv));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FModOp>(::mlir::spirv::FModOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFMod));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FMulOp>(::mlir::spirv::FMulOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFMul));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FNegateOp>(::mlir::spirv::FNegateOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFNegate));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdEqualOp>(::mlir::spirv::FOrdEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(::mlir::spirv::FOrdGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdGreaterThanOp>(::mlir::spirv::FOrdGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdLessThanEqualOp>(::mlir::spirv::FOrdLessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdLessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdLessThanOp>(::mlir::spirv::FOrdLessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdLessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdNotEqualOp>(::mlir::spirv::FOrdNotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdNotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FRemOp>(::mlir::spirv::FRemOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFRem));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FSubOp>(::mlir::spirv::FSubOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFSub));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordEqualOp>(::mlir::spirv::FUnordEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(::mlir::spirv::FUnordGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordGreaterThanOp>(::mlir::spirv::FUnordGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordLessThanEqualOp>(::mlir::spirv::FUnordLessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordLessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordLessThanOp>(::mlir::spirv::FUnordLessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordLessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordNotEqualOp>(::mlir::spirv::FUnordNotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordNotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLAcosOp>(::mlir::spirv::GLSLAcosOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 17);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLAsinOp>(::mlir::spirv::GLSLAsinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 16);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLAtanOp>(::mlir::spirv::GLSLAtanOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 18);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLCeilOp>(::mlir::spirv::GLSLCeilOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 9);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLCosOp>(::mlir::spirv::GLSLCosOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 14);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLCoshOp>(::mlir::spirv::GLSLCoshOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 20);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLExpOp>(::mlir::spirv::GLSLExpOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 27);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFAbsOp>(::mlir::spirv::GLSLFAbsOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 4);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFClampOp>(::mlir::spirv::GLSLFClampOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 43);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFMaxOp>(::mlir::spirv::GLSLFMaxOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 40);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFMinOp>(::mlir::spirv::GLSLFMinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 37);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFSignOp>(::mlir::spirv::GLSLFSignOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 6);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFloorOp>(::mlir::spirv::GLSLFloorOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 8);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFmaOp>(::mlir::spirv::GLSLFmaOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 50);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFrexpStructOp>(::mlir::spirv::GLSLFrexpStructOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 52);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLInverseSqrtOp>(::mlir::spirv::GLSLInverseSqrtOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 32);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLLdexpOp>(::mlir::spirv::GLSLLdexpOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 53);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLLogOp>(::mlir::spirv::GLSLLogOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 28);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLPowOp>(::mlir::spirv::GLSLPowOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 26);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLRoundOp>(::mlir::spirv::GLSLRoundOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 1);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSAbsOp>(::mlir::spirv::GLSLSAbsOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 5);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSClampOp>(::mlir::spirv::GLSLSClampOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 45);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSMaxOp>(::mlir::spirv::GLSLSMaxOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 42);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSMinOp>(::mlir::spirv::GLSLSMinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 39);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSSignOp>(::mlir::spirv::GLSLSSignOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 7);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSinOp>(::mlir::spirv::GLSLSinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 13);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSinhOp>(::mlir::spirv::GLSLSinhOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 19);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSqrtOp>(::mlir::spirv::GLSLSqrtOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 31);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLTanOp>(::mlir::spirv::GLSLTanOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 15);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLTanhOp>(::mlir::spirv::GLSLTanhOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 21);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLUClampOp>(::mlir::spirv::GLSLUClampOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 44);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupBroadcastOp>(::mlir::spirv::GroupBroadcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupBroadcast, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformBallotOp>(::mlir::spirv::GroupNonUniformBallotOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformBallot, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(::mlir::spirv::GroupNonUniformBroadcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformBroadcast, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformElectOp>(::mlir::spirv::GroupNonUniformElectOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op->getAttr("execution_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("execution_scope");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformElect, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFAddOp>(::mlir::spirv::GroupNonUniformFAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFAdd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMaxOp>(::mlir::spirv::GroupNonUniformFMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMinOp>(::mlir::spirv::GroupNonUniformFMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMulOp>(::mlir::spirv::GroupNonUniformFMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMul, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformIAddOp>(::mlir::spirv::GroupNonUniformIAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformIAdd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformIMulOp>(::mlir::spirv::GroupNonUniformIMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformIMul, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformSMaxOp>(::mlir::spirv::GroupNonUniformSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformSMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformSMinOp>(::mlir::spirv::GroupNonUniformSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformSMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformUMaxOp>(::mlir::spirv::GroupNonUniformUMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformUMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformUMinOp>(::mlir::spirv::GroupNonUniformUMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformUMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IAddOp>(::mlir::spirv::IAddOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIAdd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IEqualOp>(::mlir::spirv::IEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IMulOp>(::mlir::spirv::IMulOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIMul));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INotEqualOp>(::mlir::spirv::INotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpINotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ISubOp>(::mlir::spirv::ISubOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpISub));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ImageOp>(::mlir::spirv::ImageOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpImage));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IsInfOp>(::mlir::spirv::IsInfOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIsInf));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IsNanOp>(::mlir::spirv::IsNanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIsNan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LoadOp>(::mlir::spirv::LoadOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  if (auto attr = op->getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("alignment");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpLoad, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalAndOp>(::mlir::spirv::LogicalAndOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalAnd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalEqualOp>(::mlir::spirv::LogicalEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalNotEqualOp>(::mlir::spirv::LogicalNotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalNotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalNotOp>(::mlir::spirv::LogicalNotOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalNot));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalOrOp>(::mlir::spirv::LogicalOrOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalOr));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MatrixTimesMatrixOp>(::mlir::spirv::MatrixTimesMatrixOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpMatrixTimesMatrix));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MatrixTimesScalarOp>(::mlir::spirv::MatrixTimesScalarOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpMatrixTimesScalar));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NotOp>(::mlir::spirv::NotOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpNot));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OCLExpOp>(::mlir::spirv::OCLExpOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 19);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OCLFAbsOp>(::mlir::spirv::OCLFAbsOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 23);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OCLSAbsOp>(::mlir::spirv::OCLSAbsOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 141);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OrderedOp>(::mlir::spirv::OrderedOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpOrdered));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ReturnOp>(::mlir::spirv::ReturnOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpReturn));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ReturnValueOp>(::mlir::spirv::ReturnValueOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpReturnValue));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SConvertOp>(::mlir::spirv::SConvertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSConvert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SDivOp>(::mlir::spirv::SDivOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSDiv));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SGreaterThanEqualOp>(::mlir::spirv::SGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SGreaterThanOp>(::mlir::spirv::SGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SLessThanEqualOp>(::mlir::spirv::SLessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSLessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SLessThanOp>(::mlir::spirv::SLessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSLessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SModOp>(::mlir::spirv::SModOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSMod));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SNegateOp>(::mlir::spirv::SNegateOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSNegate));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SRemOp>(::mlir::spirv::SRemOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSRem));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SelectOp>(::mlir::spirv::SelectOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSelect));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftLeftLogicalOp>(::mlir::spirv::ShiftLeftLogicalOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpShiftLeftLogical));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftRightArithmeticOp>(::mlir::spirv::ShiftRightArithmeticOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpShiftRightArithmetic));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftRightLogicalOp>(::mlir::spirv::ShiftRightLogicalOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpShiftRightLogical));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::StoreOp>(::mlir::spirv::StoreOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  if (auto attr = op->getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("alignment");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpStore, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SubgroupBallotKHROp>(::mlir::spirv::SubgroupBallotKHROp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSubgroupBallotKHR));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SubgroupBlockReadINTELOp>(::mlir::spirv::SubgroupBlockReadINTELOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSubgroupBlockReadINTEL));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SubgroupBlockWriteINTELOp>(::mlir::spirv::SubgroupBlockWriteINTELOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSubgroupBlockWriteINTEL));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::TransposeOp>(::mlir::spirv::TransposeOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpTranspose));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UConvertOp>(::mlir::spirv::UConvertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUConvert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UDivOp>(::mlir::spirv::UDivOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUDiv));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UGreaterThanEqualOp>(::mlir::spirv::UGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UGreaterThanOp>(::mlir::spirv::UGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ULessThanEqualOp>(::mlir::spirv::ULessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpULessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ULessThanOp>(::mlir::spirv::ULessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpULessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UModOp>(::mlir::spirv::UModOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUMod));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UnorderedOp>(::mlir::spirv::UnorderedOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUnordered));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UnreachableOp>(::mlir::spirv::UnreachableOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUnreachable));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VariableOp>(::mlir::spirv::VariableOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("storage_class")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("storage_class");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpVariable, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorExtractDynamicOp>(::mlir::spirv::VectorExtractDynamicOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpVectorExtractDynamic));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorInsertDynamicOp>(::mlir::spirv::VectorInsertDynamicOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpVectorInsertDynamic));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorShuffleOp>(::mlir::spirv::VectorShuffleOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("components")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("components");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpVectorShuffle, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.first)) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

LogicalResult Serializer::dispatchToAutogenSerialization(Operation *op) {
  if (isa<::mlir::spirv::AccessChainOp>(op)) {
    return processOp(cast<::mlir::spirv::AccessChainOp>(op));
  }
  if (isa<::mlir::spirv::AtomicAndOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicAndOp>(op));
  }
  if (isa<::mlir::spirv::AtomicCompareExchangeWeakOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicCompareExchangeWeakOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIAddOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIAddOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIDecrementOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIDecrementOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIIncrementOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIIncrementOp>(op));
  }
  if (isa<::mlir::spirv::AtomicISubOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicISubOp>(op));
  }
  if (isa<::mlir::spirv::AtomicOrOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicOrOp>(op));
  }
  if (isa<::mlir::spirv::AtomicSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicSMaxOp>(op));
  }
  if (isa<::mlir::spirv::AtomicSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicSMinOp>(op));
  }
  if (isa<::mlir::spirv::AtomicUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicUMaxOp>(op));
  }
  if (isa<::mlir::spirv::AtomicUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicUMinOp>(op));
  }
  if (isa<::mlir::spirv::AtomicXorOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicXorOp>(op));
  }
  if (isa<::mlir::spirv::BitCountOp>(op)) {
    return processOp(cast<::mlir::spirv::BitCountOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldInsertOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldInsertOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldSExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldSExtractOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldUExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldUExtractOp>(op));
  }
  if (isa<::mlir::spirv::BitReverseOp>(op)) {
    return processOp(cast<::mlir::spirv::BitReverseOp>(op));
  }
  if (isa<::mlir::spirv::BitcastOp>(op)) {
    return processOp(cast<::mlir::spirv::BitcastOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseAndOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseAndOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseOrOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseOrOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseXorOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseXorOp>(op));
  }
  if (isa<::mlir::spirv::BranchConditionalOp>(op)) {
    return processOp(cast<::mlir::spirv::BranchConditionalOp>(op));
  }
  if (isa<::mlir::spirv::BranchOp>(op)) {
    return processOp(cast<::mlir::spirv::BranchOp>(op));
  }
  if (isa<::mlir::spirv::CompositeConstructOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeConstructOp>(op));
  }
  if (isa<::mlir::spirv::CompositeExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeExtractOp>(op));
  }
  if (isa<::mlir::spirv::CompositeInsertOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeInsertOp>(op));
  }
  if (isa<::mlir::spirv::ControlBarrierOp>(op)) {
    return processOp(cast<::mlir::spirv::ControlBarrierOp>(op));
  }
  if (isa<::mlir::spirv::ConvertFToSOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertFToSOp>(op));
  }
  if (isa<::mlir::spirv::ConvertFToUOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertFToUOp>(op));
  }
  if (isa<::mlir::spirv::ConvertSToFOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertSToFOp>(op));
  }
  if (isa<::mlir::spirv::ConvertUToFOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertUToFOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixLengthNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixLengthNVOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixLoadNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixLoadNVOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixMulAddNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixMulAddNVOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixStoreNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixStoreNVOp>(op));
  }
  if (isa<::mlir::spirv::CopyMemoryOp>(op)) {
    return processOp(cast<::mlir::spirv::CopyMemoryOp>(op));
  }
  if (isa<::mlir::spirv::EntryPointOp>(op)) {
    return processOp(cast<::mlir::spirv::EntryPointOp>(op));
  }
  if (isa<::mlir::spirv::ExecutionModeOp>(op)) {
    return processOp(cast<::mlir::spirv::ExecutionModeOp>(op));
  }
  if (isa<::mlir::spirv::FAddOp>(op)) {
    return processOp(cast<::mlir::spirv::FAddOp>(op));
  }
  if (isa<::mlir::spirv::FConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::FConvertOp>(op));
  }
  if (isa<::mlir::spirv::FDivOp>(op)) {
    return processOp(cast<::mlir::spirv::FDivOp>(op));
  }
  if (isa<::mlir::spirv::FModOp>(op)) {
    return processOp(cast<::mlir::spirv::FModOp>(op));
  }
  if (isa<::mlir::spirv::FMulOp>(op)) {
    return processOp(cast<::mlir::spirv::FMulOp>(op));
  }
  if (isa<::mlir::spirv::FNegateOp>(op)) {
    return processOp(cast<::mlir::spirv::FNegateOp>(op));
  }
  if (isa<::mlir::spirv::FOrdEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::FOrdLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdLessThanOp>(op));
  }
  if (isa<::mlir::spirv::FOrdNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::FRemOp>(op)) {
    return processOp(cast<::mlir::spirv::FRemOp>(op));
  }
  if (isa<::mlir::spirv::FSubOp>(op)) {
    return processOp(cast<::mlir::spirv::FSubOp>(op));
  }
  if (isa<::mlir::spirv::FUnordEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::FUnordLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordLessThanOp>(op));
  }
  if (isa<::mlir::spirv::FUnordNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::FunctionCallOp>(op)) {
    return processOp(cast<::mlir::spirv::FunctionCallOp>(op));
  }
  if (isa<::mlir::spirv::GLSLAcosOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLAcosOp>(op));
  }
  if (isa<::mlir::spirv::GLSLAsinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLAsinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLAtanOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLAtanOp>(op));
  }
  if (isa<::mlir::spirv::GLSLCeilOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLCeilOp>(op));
  }
  if (isa<::mlir::spirv::GLSLCosOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLCosOp>(op));
  }
  if (isa<::mlir::spirv::GLSLCoshOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLCoshOp>(op));
  }
  if (isa<::mlir::spirv::GLSLExpOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLExpOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFAbsOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFClampOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFClampOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFMinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFSignOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFSignOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFloorOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFloorOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFmaOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFmaOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFrexpStructOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFrexpStructOp>(op));
  }
  if (isa<::mlir::spirv::GLSLInverseSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLInverseSqrtOp>(op));
  }
  if (isa<::mlir::spirv::GLSLLdexpOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLLdexpOp>(op));
  }
  if (isa<::mlir::spirv::GLSLLogOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLLogOp>(op));
  }
  if (isa<::mlir::spirv::GLSLPowOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLPowOp>(op));
  }
  if (isa<::mlir::spirv::GLSLRoundOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLRoundOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSAbsOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSClampOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSClampOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSMinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSSignOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSSignOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSinhOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSinhOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSqrtOp>(op));
  }
  if (isa<::mlir::spirv::GLSLTanOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLTanOp>(op));
  }
  if (isa<::mlir::spirv::GLSLTanhOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLTanhOp>(op));
  }
  if (isa<::mlir::spirv::GLSLUClampOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLUClampOp>(op));
  }
  if (isa<::mlir::spirv::GroupBroadcastOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupBroadcastOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformBallotOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformBallotOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformBroadcastOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformBroadcastOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformElectOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformElectOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFAddOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFAddOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMulOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMulOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformIAddOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformIAddOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformIMulOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformIMulOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformSMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformSMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformUMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformUMinOp>(op));
  }
  if (isa<::mlir::spirv::IAddOp>(op)) {
    return processOp(cast<::mlir::spirv::IAddOp>(op));
  }
  if (isa<::mlir::spirv::IEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::IEqualOp>(op));
  }
  if (isa<::mlir::spirv::IMulOp>(op)) {
    return processOp(cast<::mlir::spirv::IMulOp>(op));
  }
  if (isa<::mlir::spirv::INotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::INotEqualOp>(op));
  }
  if (isa<::mlir::spirv::ISubOp>(op)) {
    return processOp(cast<::mlir::spirv::ISubOp>(op));
  }
  if (isa<::mlir::spirv::ImageOp>(op)) {
    return processOp(cast<::mlir::spirv::ImageOp>(op));
  }
  if (isa<::mlir::spirv::IsInfOp>(op)) {
    return processOp(cast<::mlir::spirv::IsInfOp>(op));
  }
  if (isa<::mlir::spirv::IsNanOp>(op)) {
    return processOp(cast<::mlir::spirv::IsNanOp>(op));
  }
  if (isa<::mlir::spirv::LoadOp>(op)) {
    return processOp(cast<::mlir::spirv::LoadOp>(op));
  }
  if (isa<::mlir::spirv::LogicalAndOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalAndOp>(op));
  }
  if (isa<::mlir::spirv::LogicalEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalEqualOp>(op));
  }
  if (isa<::mlir::spirv::LogicalNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::LogicalNotOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalNotOp>(op));
  }
  if (isa<::mlir::spirv::LogicalOrOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalOrOp>(op));
  }
  if (isa<::mlir::spirv::MatrixTimesMatrixOp>(op)) {
    return processOp(cast<::mlir::spirv::MatrixTimesMatrixOp>(op));
  }
  if (isa<::mlir::spirv::MatrixTimesScalarOp>(op)) {
    return processOp(cast<::mlir::spirv::MatrixTimesScalarOp>(op));
  }
  if (isa<::mlir::spirv::MemoryBarrierOp>(op)) {
    return processOp(cast<::mlir::spirv::MemoryBarrierOp>(op));
  }
  if (isa<::mlir::spirv::NotOp>(op)) {
    return processOp(cast<::mlir::spirv::NotOp>(op));
  }
  if (isa<::mlir::spirv::OCLExpOp>(op)) {
    return processOp(cast<::mlir::spirv::OCLExpOp>(op));
  }
  if (isa<::mlir::spirv::OCLFAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::OCLFAbsOp>(op));
  }
  if (isa<::mlir::spirv::OCLSAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::OCLSAbsOp>(op));
  }
  if (isa<::mlir::spirv::OrderedOp>(op)) {
    return processOp(cast<::mlir::spirv::OrderedOp>(op));
  }
  if (isa<::mlir::spirv::ReturnOp>(op)) {
    return processOp(cast<::mlir::spirv::ReturnOp>(op));
  }
  if (isa<::mlir::spirv::ReturnValueOp>(op)) {
    return processOp(cast<::mlir::spirv::ReturnValueOp>(op));
  }
  if (isa<::mlir::spirv::SConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::SConvertOp>(op));
  }
  if (isa<::mlir::spirv::SDivOp>(op)) {
    return processOp(cast<::mlir::spirv::SDivOp>(op));
  }
  if (isa<::mlir::spirv::SGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::SGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::SGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::SGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::SLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::SLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::SLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::SLessThanOp>(op));
  }
  if (isa<::mlir::spirv::SModOp>(op)) {
    return processOp(cast<::mlir::spirv::SModOp>(op));
  }
  if (isa<::mlir::spirv::SNegateOp>(op)) {
    return processOp(cast<::mlir::spirv::SNegateOp>(op));
  }
  if (isa<::mlir::spirv::SRemOp>(op)) {
    return processOp(cast<::mlir::spirv::SRemOp>(op));
  }
  if (isa<::mlir::spirv::SelectOp>(op)) {
    return processOp(cast<::mlir::spirv::SelectOp>(op));
  }
  if (isa<::mlir::spirv::ShiftLeftLogicalOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftLeftLogicalOp>(op));
  }
  if (isa<::mlir::spirv::ShiftRightArithmeticOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftRightArithmeticOp>(op));
  }
  if (isa<::mlir::spirv::ShiftRightLogicalOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftRightLogicalOp>(op));
  }
  if (isa<::mlir::spirv::StoreOp>(op)) {
    return processOp(cast<::mlir::spirv::StoreOp>(op));
  }
  if (isa<::mlir::spirv::SubgroupBallotKHROp>(op)) {
    return processOp(cast<::mlir::spirv::SubgroupBallotKHROp>(op));
  }
  if (isa<::mlir::spirv::SubgroupBlockReadINTELOp>(op)) {
    return processOp(cast<::mlir::spirv::SubgroupBlockReadINTELOp>(op));
  }
  if (isa<::mlir::spirv::SubgroupBlockWriteINTELOp>(op)) {
    return processOp(cast<::mlir::spirv::SubgroupBlockWriteINTELOp>(op));
  }
  if (isa<::mlir::spirv::TransposeOp>(op)) {
    return processOp(cast<::mlir::spirv::TransposeOp>(op));
  }
  if (isa<::mlir::spirv::UConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::UConvertOp>(op));
  }
  if (isa<::mlir::spirv::UDivOp>(op)) {
    return processOp(cast<::mlir::spirv::UDivOp>(op));
  }
  if (isa<::mlir::spirv::UGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::UGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::UGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::UGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::ULessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::ULessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::ULessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::ULessThanOp>(op));
  }
  if (isa<::mlir::spirv::UModOp>(op)) {
    return processOp(cast<::mlir::spirv::UModOp>(op));
  }
  if (isa<::mlir::spirv::UnorderedOp>(op)) {
    return processOp(cast<::mlir::spirv::UnorderedOp>(op));
  }
  if (isa<::mlir::spirv::UnreachableOp>(op)) {
    return processOp(cast<::mlir::spirv::UnreachableOp>(op));
  }
  if (isa<::mlir::spirv::VariableOp>(op)) {
    return processOp(cast<::mlir::spirv::VariableOp>(op));
  }
  if (isa<::mlir::spirv::VectorExtractDynamicOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorExtractDynamicOp>(op));
  }
  if (isa<::mlir::spirv::VectorInsertDynamicOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorInsertDynamicOp>(op));
  }
  if (isa<::mlir::spirv::VectorShuffleOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorShuffleOp>(op));
  }
  return op->emitError("unhandled operation serialization");
}

#endif // GET_SERIALIZATION_FNS

#ifdef GET_DESERIALIZATION_FNS

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AccessChainOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AccessChainOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AccessChainOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AccessChainOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AccessChainOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicAndOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicAndOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicAndOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicAndOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicAndOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("equal_semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("unequal_semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicCompareExchangeWeakOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIDecrementOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIDecrementOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIDecrementOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIDecrementOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIDecrementOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIIncrementOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIIncrementOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIIncrementOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIIncrementOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIIncrementOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicISubOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicISubOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicISubOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicISubOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicISubOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicOrOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicOrOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicOrOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicOrOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicOrOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicUMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicUMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicUMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicUMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicUMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicUMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicUMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicUMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicUMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicUMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicXorOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicXorOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicXorOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicXorOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicXorOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitCountOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitCount", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldInsertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitFieldInsert", true, 4);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldSExtractOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitFieldSExtract", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldUExtractOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitFieldUExtract", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitReverseOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitReverse", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitcastOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Bitcast", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseAndOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitwiseAnd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseOrOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitwiseOr", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseXorOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitwiseXor", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeConstructOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeConstructOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeConstructOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeConstructOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeConstructOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeExtractOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeExtractOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeExtractOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("indices", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeExtractOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeExtractOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeInsertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeInsertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeInsertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("indices", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeInsertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeInsertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertFToSOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertFToS", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertFToUOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertFToU", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertSToFOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertSToF", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertUToFOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertUToF", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixLengthNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CooperativeMatrixLengthNVOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CooperativeMatrixLengthNVOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("type", TypeAttr::get(getType(words[wordIndex++]))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixLengthNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixLengthNVOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixLoadNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CooperativeMatrixLoadNVOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CooperativeMatrixLoadNVOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixLoadNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixLoadNVOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixMulAddNVOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CooperativeMatrixMulAddNV", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixStoreNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixStoreNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixStoreNVOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FAddOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FAdd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FConvertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FConvert", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FDivOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FDiv", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FModOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FMod", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FMulOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FMul", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FNegateOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FNegate", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdLessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdLessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdLessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdLessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdNotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdNotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FRemOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FRem", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FSubOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FSub", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordLessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordLessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordLessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordLessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordNotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordNotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLAcosOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Acos", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLAsinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Asin", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLAtanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Atan", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLCeilOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Ceil", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLCosOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Cos", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLCoshOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Cosh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLExpOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Exp", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.FAbs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFClampOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.FClamp", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFMaxOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.FMax", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFMinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.FMin", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFSignOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.FSign", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFloorOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Floor", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFmaOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Fma", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFrexpStructOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.FrexpStruct", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLInverseSqrtOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.InverseSqrt", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLLdexpOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Ldexp", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLLogOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Log", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLPowOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Pow", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLRoundOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Round", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.SAbs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSClampOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.SClamp", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSMaxOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.SMax", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSMinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.SMin", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSSignOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.SSign", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Sin", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSinhOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Sinh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSqrtOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Sqrt", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLTanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Tan", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLTanhOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.Tanh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLUClampOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GLSL.UClamp", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupBroadcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupBroadcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupBroadcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupBroadcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupBroadcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformBallotOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformBallotOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformBallotOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformBallotOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformBallotOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformBroadcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformBroadcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformBroadcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformBroadcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformElectOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformElectOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformElectOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformElectOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformElectOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformIAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformIAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformIAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformIAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformIAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformIMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformIMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformIMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformIMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformIMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformUMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformUMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformUMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformUMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformUMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformUMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformUMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformUMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformUMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformUMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IAddOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IAdd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IMulOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IMul", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.INotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ISubOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ISub", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ImageOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Image", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IsInfOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IsInf", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IsNanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IsNan", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LoadOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LoadOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LoadOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LoadOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LoadOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalAndOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalAnd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalNotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalNotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalNotOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalNot", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalOrOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalOr", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MatrixTimesMatrixOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.MatrixTimesMatrix", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MatrixTimesScalarOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.MatrixTimesScalar", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NotOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Not", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OCLExpOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.OCL.exp", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OCLFAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.OCL.fabs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OCLSAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.OCL.s_abs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OrderedOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Ordered", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ReturnOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Return", false, 0);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ReturnValueOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ReturnValue", false, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SConvertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SConvert", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SDivOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SDiv", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SLessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SLessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SLessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SLessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SModOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SMod", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SNegateOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SNegate", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SRemOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SRem", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SelectOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Select", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftLeftLogicalOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ShiftLeftLogical", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftRightArithmeticOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ShiftRightArithmetic", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftRightLogicalOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ShiftRightLogical", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::StoreOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::StoreOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::StoreOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SubgroupBallotKHROp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SubgroupBallotKHR", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SubgroupBlockReadINTELOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SubgroupBlockReadINTEL", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SubgroupBlockWriteINTELOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SubgroupBlockWriteINTEL", false, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::TransposeOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Transpose", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UConvertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UConvert", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UDivOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UDiv", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ULessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ULessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ULessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ULessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UModOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UMod", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UnorderedOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Unordered", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UnreachableOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Unreachable", false, 0);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VariableOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::VariableOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::VariableOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("storage_class", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::VariableOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::VariableOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorExtractDynamicOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.VectorExtractDynamic", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorInsertDynamicOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.VectorInsertDynamic", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorShuffleOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::VectorShuffleOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::VectorShuffleOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("components", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::VectorShuffleOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::VectorShuffleOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

LogicalResult spirv::Deserializer::dispatchToAutogenDeserialization(spirv::Opcode opcode, ArrayRef<uint32_t> words) {
  switch (opcode) {
  case spirv::Opcode::OpAccessChain:
    return processOp<::mlir::spirv::AccessChainOp>(words);
  case spirv::Opcode::OpAtomicAnd:
    return processOp<::mlir::spirv::AtomicAndOp>(words);
  case spirv::Opcode::OpAtomicCompareExchangeWeak:
    return processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(words);
  case spirv::Opcode::OpAtomicIAdd:
    return processOp<::mlir::spirv::AtomicIAddOp>(words);
  case spirv::Opcode::OpAtomicIDecrement:
    return processOp<::mlir::spirv::AtomicIDecrementOp>(words);
  case spirv::Opcode::OpAtomicIIncrement:
    return processOp<::mlir::spirv::AtomicIIncrementOp>(words);
  case spirv::Opcode::OpAtomicISub:
    return processOp<::mlir::spirv::AtomicISubOp>(words);
  case spirv::Opcode::OpAtomicOr:
    return processOp<::mlir::spirv::AtomicOrOp>(words);
  case spirv::Opcode::OpAtomicSMax:
    return processOp<::mlir::spirv::AtomicSMaxOp>(words);
  case spirv::Opcode::OpAtomicSMin:
    return processOp<::mlir::spirv::AtomicSMinOp>(words);
  case spirv::Opcode::OpAtomicUMax:
    return processOp<::mlir::spirv::AtomicUMaxOp>(words);
  case spirv::Opcode::OpAtomicUMin:
    return processOp<::mlir::spirv::AtomicUMinOp>(words);
  case spirv::Opcode::OpAtomicXor:
    return processOp<::mlir::spirv::AtomicXorOp>(words);
  case spirv::Opcode::OpBitCount:
    return processOp<::mlir::spirv::BitCountOp>(words);
  case spirv::Opcode::OpBitFieldInsert:
    return processOp<::mlir::spirv::BitFieldInsertOp>(words);
  case spirv::Opcode::OpBitFieldSExtract:
    return processOp<::mlir::spirv::BitFieldSExtractOp>(words);
  case spirv::Opcode::OpBitFieldUExtract:
    return processOp<::mlir::spirv::BitFieldUExtractOp>(words);
  case spirv::Opcode::OpBitReverse:
    return processOp<::mlir::spirv::BitReverseOp>(words);
  case spirv::Opcode::OpBitcast:
    return processOp<::mlir::spirv::BitcastOp>(words);
  case spirv::Opcode::OpBitwiseAnd:
    return processOp<::mlir::spirv::BitwiseAndOp>(words);
  case spirv::Opcode::OpBitwiseOr:
    return processOp<::mlir::spirv::BitwiseOrOp>(words);
  case spirv::Opcode::OpBitwiseXor:
    return processOp<::mlir::spirv::BitwiseXorOp>(words);
  case spirv::Opcode::OpBranchConditional:
    return processOp<::mlir::spirv::BranchConditionalOp>(words);
  case spirv::Opcode::OpBranch:
    return processOp<::mlir::spirv::BranchOp>(words);
  case spirv::Opcode::OpCompositeConstruct:
    return processOp<::mlir::spirv::CompositeConstructOp>(words);
  case spirv::Opcode::OpCompositeExtract:
    return processOp<::mlir::spirv::CompositeExtractOp>(words);
  case spirv::Opcode::OpCompositeInsert:
    return processOp<::mlir::spirv::CompositeInsertOp>(words);
  case spirv::Opcode::OpControlBarrier:
    return processOp<::mlir::spirv::ControlBarrierOp>(words);
  case spirv::Opcode::OpConvertFToS:
    return processOp<::mlir::spirv::ConvertFToSOp>(words);
  case spirv::Opcode::OpConvertFToU:
    return processOp<::mlir::spirv::ConvertFToUOp>(words);
  case spirv::Opcode::OpConvertSToF:
    return processOp<::mlir::spirv::ConvertSToFOp>(words);
  case spirv::Opcode::OpConvertUToF:
    return processOp<::mlir::spirv::ConvertUToFOp>(words);
  case spirv::Opcode::OpCooperativeMatrixLengthNV:
    return processOp<::mlir::spirv::CooperativeMatrixLengthNVOp>(words);
  case spirv::Opcode::OpCooperativeMatrixLoadNV:
    return processOp<::mlir::spirv::CooperativeMatrixLoadNVOp>(words);
  case spirv::Opcode::OpCooperativeMatrixMulAddNV:
    return processOp<::mlir::spirv::CooperativeMatrixMulAddNVOp>(words);
  case spirv::Opcode::OpCooperativeMatrixStoreNV:
    return processOp<::mlir::spirv::CooperativeMatrixStoreNVOp>(words);
  case spirv::Opcode::OpCopyMemory:
    return processOp<::mlir::spirv::CopyMemoryOp>(words);
  case spirv::Opcode::OpEntryPoint:
    return processOp<::mlir::spirv::EntryPointOp>(words);
  case spirv::Opcode::OpExecutionMode:
    return processOp<::mlir::spirv::ExecutionModeOp>(words);
  case spirv::Opcode::OpFAdd:
    return processOp<::mlir::spirv::FAddOp>(words);
  case spirv::Opcode::OpFConvert:
    return processOp<::mlir::spirv::FConvertOp>(words);
  case spirv::Opcode::OpFDiv:
    return processOp<::mlir::spirv::FDivOp>(words);
  case spirv::Opcode::OpFMod:
    return processOp<::mlir::spirv::FModOp>(words);
  case spirv::Opcode::OpFMul:
    return processOp<::mlir::spirv::FMulOp>(words);
  case spirv::Opcode::OpFNegate:
    return processOp<::mlir::spirv::FNegateOp>(words);
  case spirv::Opcode::OpFOrdEqual:
    return processOp<::mlir::spirv::FOrdEqualOp>(words);
  case spirv::Opcode::OpFOrdGreaterThanEqual:
    return processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(words);
  case spirv::Opcode::OpFOrdGreaterThan:
    return processOp<::mlir::spirv::FOrdGreaterThanOp>(words);
  case spirv::Opcode::OpFOrdLessThanEqual:
    return processOp<::mlir::spirv::FOrdLessThanEqualOp>(words);
  case spirv::Opcode::OpFOrdLessThan:
    return processOp<::mlir::spirv::FOrdLessThanOp>(words);
  case spirv::Opcode::OpFOrdNotEqual:
    return processOp<::mlir::spirv::FOrdNotEqualOp>(words);
  case spirv::Opcode::OpFRem:
    return processOp<::mlir::spirv::FRemOp>(words);
  case spirv::Opcode::OpFSub:
    return processOp<::mlir::spirv::FSubOp>(words);
  case spirv::Opcode::OpFUnordEqual:
    return processOp<::mlir::spirv::FUnordEqualOp>(words);
  case spirv::Opcode::OpFUnordGreaterThanEqual:
    return processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(words);
  case spirv::Opcode::OpFUnordGreaterThan:
    return processOp<::mlir::spirv::FUnordGreaterThanOp>(words);
  case spirv::Opcode::OpFUnordLessThanEqual:
    return processOp<::mlir::spirv::FUnordLessThanEqualOp>(words);
  case spirv::Opcode::OpFUnordLessThan:
    return processOp<::mlir::spirv::FUnordLessThanOp>(words);
  case spirv::Opcode::OpFUnordNotEqual:
    return processOp<::mlir::spirv::FUnordNotEqualOp>(words);
  case spirv::Opcode::OpFunctionCall:
    return processOp<::mlir::spirv::FunctionCallOp>(words);
  case spirv::Opcode::OpGroupBroadcast:
    return processOp<::mlir::spirv::GroupBroadcastOp>(words);
  case spirv::Opcode::OpGroupNonUniformBallot:
    return processOp<::mlir::spirv::GroupNonUniformBallotOp>(words);
  case spirv::Opcode::OpGroupNonUniformBroadcast:
    return processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(words);
  case spirv::Opcode::OpGroupNonUniformElect:
    return processOp<::mlir::spirv::GroupNonUniformElectOp>(words);
  case spirv::Opcode::OpGroupNonUniformFAdd:
    return processOp<::mlir::spirv::GroupNonUniformFAddOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMax:
    return processOp<::mlir::spirv::GroupNonUniformFMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMin:
    return processOp<::mlir::spirv::GroupNonUniformFMinOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMul:
    return processOp<::mlir::spirv::GroupNonUniformFMulOp>(words);
  case spirv::Opcode::OpGroupNonUniformIAdd:
    return processOp<::mlir::spirv::GroupNonUniformIAddOp>(words);
  case spirv::Opcode::OpGroupNonUniformIMul:
    return processOp<::mlir::spirv::GroupNonUniformIMulOp>(words);
  case spirv::Opcode::OpGroupNonUniformSMax:
    return processOp<::mlir::spirv::GroupNonUniformSMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformSMin:
    return processOp<::mlir::spirv::GroupNonUniformSMinOp>(words);
  case spirv::Opcode::OpGroupNonUniformUMax:
    return processOp<::mlir::spirv::GroupNonUniformUMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformUMin:
    return processOp<::mlir::spirv::GroupNonUniformUMinOp>(words);
  case spirv::Opcode::OpIAdd:
    return processOp<::mlir::spirv::IAddOp>(words);
  case spirv::Opcode::OpIEqual:
    return processOp<::mlir::spirv::IEqualOp>(words);
  case spirv::Opcode::OpIMul:
    return processOp<::mlir::spirv::IMulOp>(words);
  case spirv::Opcode::OpINotEqual:
    return processOp<::mlir::spirv::INotEqualOp>(words);
  case spirv::Opcode::OpISub:
    return processOp<::mlir::spirv::ISubOp>(words);
  case spirv::Opcode::OpImage:
    return processOp<::mlir::spirv::ImageOp>(words);
  case spirv::Opcode::OpIsInf:
    return processOp<::mlir::spirv::IsInfOp>(words);
  case spirv::Opcode::OpIsNan:
    return processOp<::mlir::spirv::IsNanOp>(words);
  case spirv::Opcode::OpLoad:
    return processOp<::mlir::spirv::LoadOp>(words);
  case spirv::Opcode::OpLogicalAnd:
    return processOp<::mlir::spirv::LogicalAndOp>(words);
  case spirv::Opcode::OpLogicalEqual:
    return processOp<::mlir::spirv::LogicalEqualOp>(words);
  case spirv::Opcode::OpLogicalNotEqual:
    return processOp<::mlir::spirv::LogicalNotEqualOp>(words);
  case spirv::Opcode::OpLogicalNot:
    return processOp<::mlir::spirv::LogicalNotOp>(words);
  case spirv::Opcode::OpLogicalOr:
    return processOp<::mlir::spirv::LogicalOrOp>(words);
  case spirv::Opcode::OpMatrixTimesMatrix:
    return processOp<::mlir::spirv::MatrixTimesMatrixOp>(words);
  case spirv::Opcode::OpMatrixTimesScalar:
    return processOp<::mlir::spirv::MatrixTimesScalarOp>(words);
  case spirv::Opcode::OpMemoryBarrier:
    return processOp<::mlir::spirv::MemoryBarrierOp>(words);
  case spirv::Opcode::OpNot:
    return processOp<::mlir::spirv::NotOp>(words);
  case spirv::Opcode::OpOrdered:
    return processOp<::mlir::spirv::OrderedOp>(words);
  case spirv::Opcode::OpReturn:
    return processOp<::mlir::spirv::ReturnOp>(words);
  case spirv::Opcode::OpReturnValue:
    return processOp<::mlir::spirv::ReturnValueOp>(words);
  case spirv::Opcode::OpSConvert:
    return processOp<::mlir::spirv::SConvertOp>(words);
  case spirv::Opcode::OpSDiv:
    return processOp<::mlir::spirv::SDivOp>(words);
  case spirv::Opcode::OpSGreaterThanEqual:
    return processOp<::mlir::spirv::SGreaterThanEqualOp>(words);
  case spirv::Opcode::OpSGreaterThan:
    return processOp<::mlir::spirv::SGreaterThanOp>(words);
  case spirv::Opcode::OpSLessThanEqual:
    return processOp<::mlir::spirv::SLessThanEqualOp>(words);
  case spirv::Opcode::OpSLessThan:
    return processOp<::mlir::spirv::SLessThanOp>(words);
  case spirv::Opcode::OpSMod:
    return processOp<::mlir::spirv::SModOp>(words);
  case spirv::Opcode::OpSNegate:
    return processOp<::mlir::spirv::SNegateOp>(words);
  case spirv::Opcode::OpSRem:
    return processOp<::mlir::spirv::SRemOp>(words);
  case spirv::Opcode::OpSelect:
    return processOp<::mlir::spirv::SelectOp>(words);
  case spirv::Opcode::OpShiftLeftLogical:
    return processOp<::mlir::spirv::ShiftLeftLogicalOp>(words);
  case spirv::Opcode::OpShiftRightArithmetic:
    return processOp<::mlir::spirv::ShiftRightArithmeticOp>(words);
  case spirv::Opcode::OpShiftRightLogical:
    return processOp<::mlir::spirv::ShiftRightLogicalOp>(words);
  case spirv::Opcode::OpStore:
    return processOp<::mlir::spirv::StoreOp>(words);
  case spirv::Opcode::OpSubgroupBallotKHR:
    return processOp<::mlir::spirv::SubgroupBallotKHROp>(words);
  case spirv::Opcode::OpSubgroupBlockReadINTEL:
    return processOp<::mlir::spirv::SubgroupBlockReadINTELOp>(words);
  case spirv::Opcode::OpSubgroupBlockWriteINTEL:
    return processOp<::mlir::spirv::SubgroupBlockWriteINTELOp>(words);
  case spirv::Opcode::OpTranspose:
    return processOp<::mlir::spirv::TransposeOp>(words);
  case spirv::Opcode::OpUConvert:
    return processOp<::mlir::spirv::UConvertOp>(words);
  case spirv::Opcode::OpUDiv:
    return processOp<::mlir::spirv::UDivOp>(words);
  case spirv::Opcode::OpUGreaterThanEqual:
    return processOp<::mlir::spirv::UGreaterThanEqualOp>(words);
  case spirv::Opcode::OpUGreaterThan:
    return processOp<::mlir::spirv::UGreaterThanOp>(words);
  case spirv::Opcode::OpULessThanEqual:
    return processOp<::mlir::spirv::ULessThanEqualOp>(words);
  case spirv::Opcode::OpULessThan:
    return processOp<::mlir::spirv::ULessThanOp>(words);
  case spirv::Opcode::OpUMod:
    return processOp<::mlir::spirv::UModOp>(words);
  case spirv::Opcode::OpUnordered:
    return processOp<::mlir::spirv::UnorderedOp>(words);
  case spirv::Opcode::OpUnreachable:
    return processOp<::mlir::spirv::UnreachableOp>(words);
  case spirv::Opcode::OpVariable:
    return processOp<::mlir::spirv::VariableOp>(words);
  case spirv::Opcode::OpVectorExtractDynamic:
    return processOp<::mlir::spirv::VectorExtractDynamicOp>(words);
  case spirv::Opcode::OpVectorInsertDynamic:
    return processOp<::mlir::spirv::VectorInsertDynamicOp>(words);
  case spirv::Opcode::OpVectorShuffle:
    return processOp<::mlir::spirv::VectorShuffleOp>(words);
  default:
    ;
  }
  auto opcodeString = spirv::stringifyOpcode(opcode);
  if (!opcodeString.empty()) {
    return emitError(unknownLoc, "unhandled deserialization of ") << opcodeString;
  } else {
   return emitError(unknownLoc, "unhandled opcode ") << static_cast<uint32_t>(opcode);
  }
}
LogicalResult spirv::Deserializer::dispatchToExtensionSetAutogenDeserialization(StringRef extensionSetName, uint32_t instructionID, ArrayRef<uint32_t> words) {
  if (extensionSetName == "GLSL.std.450") {
    switch (instructionID) {
    case 17:
      return processOp<::mlir::spirv::GLSLAcosOp>(words);
    case 16:
      return processOp<::mlir::spirv::GLSLAsinOp>(words);
    case 18:
      return processOp<::mlir::spirv::GLSLAtanOp>(words);
    case 9:
      return processOp<::mlir::spirv::GLSLCeilOp>(words);
    case 14:
      return processOp<::mlir::spirv::GLSLCosOp>(words);
    case 20:
      return processOp<::mlir::spirv::GLSLCoshOp>(words);
    case 27:
      return processOp<::mlir::spirv::GLSLExpOp>(words);
    case 4:
      return processOp<::mlir::spirv::GLSLFAbsOp>(words);
    case 43:
      return processOp<::mlir::spirv::GLSLFClampOp>(words);
    case 40:
      return processOp<::mlir::spirv::GLSLFMaxOp>(words);
    case 37:
      return processOp<::mlir::spirv::GLSLFMinOp>(words);
    case 6:
      return processOp<::mlir::spirv::GLSLFSignOp>(words);
    case 8:
      return processOp<::mlir::spirv::GLSLFloorOp>(words);
    case 50:
      return processOp<::mlir::spirv::GLSLFmaOp>(words);
    case 52:
      return processOp<::mlir::spirv::GLSLFrexpStructOp>(words);
    case 32:
      return processOp<::mlir::spirv::GLSLInverseSqrtOp>(words);
    case 53:
      return processOp<::mlir::spirv::GLSLLdexpOp>(words);
    case 28:
      return processOp<::mlir::spirv::GLSLLogOp>(words);
    case 26:
      return processOp<::mlir::spirv::GLSLPowOp>(words);
    case 1:
      return processOp<::mlir::spirv::GLSLRoundOp>(words);
    case 5:
      return processOp<::mlir::spirv::GLSLSAbsOp>(words);
    case 45:
      return processOp<::mlir::spirv::GLSLSClampOp>(words);
    case 42:
      return processOp<::mlir::spirv::GLSLSMaxOp>(words);
    case 39:
      return processOp<::mlir::spirv::GLSLSMinOp>(words);
    case 7:
      return processOp<::mlir::spirv::GLSLSSignOp>(words);
    case 13:
      return processOp<::mlir::spirv::GLSLSinOp>(words);
    case 19:
      return processOp<::mlir::spirv::GLSLSinhOp>(words);
    case 31:
      return processOp<::mlir::spirv::GLSLSqrtOp>(words);
    case 15:
      return processOp<::mlir::spirv::GLSLTanOp>(words);
    case 21:
      return processOp<::mlir::spirv::GLSLTanhOp>(words);
    case 44:
      return processOp<::mlir::spirv::GLSLUClampOp>(words);
    default:
      return emitError(unknownLoc, "unhandled deserializations of ") << instructionID << " from extension set " << extensionSetName;
    }
  }
  if (extensionSetName == "OpenCL.std") {
    switch (instructionID) {
    case 19:
      return processOp<::mlir::spirv::OCLExpOp>(words);
    case 23:
      return processOp<::mlir::spirv::OCLFAbsOp>(words);
    case 141:
      return processOp<::mlir::spirv::OCLSAbsOp>(words);
    default:
      return emitError(unknownLoc, "unhandled deserializations of ") << instructionID << " from extension set " << extensionSetName;
    }
  }
  return emitError(unknownLoc, "unhandled deserialization of extended instruction set extensionSetName");
}
#endif // GET_DESERIALIZATION_FNS

