/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
std::string stringifyCombiningKind(CombiningKind symbol) {
  auto val = static_cast<uint32_t>(symbol);
  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("add"); val &= ~1u; }
  if (2u & val) { strs.push_back("mul"); val &= ~2u; }
  if (4u & val) { strs.push_back("min"); val &= ~4u; }
  if (8u & val) { strs.push_back("max"); val &= ~8u; }
  if (16u & val) { strs.push_back("and"); val &= ~16u; }
  if (32u & val) { strs.push_back("or"); val &= ~32u; }
  if (64u & val) { strs.push_back("xor"); val &= ~64u; }

  if (val) return "";
  return ::llvm::join(strs, "|");
}

::llvm::Optional<CombiningKind> symbolizeCombiningKind(::llvm::StringRef str) {
  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol)
      .Case("add", 1)
      .Case("mul", 2)
      .Case("min", 4)
      .Case("max", 8)
      .Case("and", 16)
      .Case("or", 32)
      .Case("xor", 64)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<CombiningKind>(val);
}

::llvm::Optional<CombiningKind> symbolizeCombiningKind(uint32_t value) {
  if (value & ~(1u | 2u | 4u | 8u | 16u | 32u | 64u)) return llvm::None;
  return static_cast<CombiningKind>(value);
}
} // namespace vector
} // namespace mlir

