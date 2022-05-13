/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {

class BuiltinDialect : public ::mlir::Dialect {
  explicit BuiltinDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<BuiltinDialect>()) {
    
    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("");
  }

  private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Location Attributes.
    void registerLocationAttributes();
    // Register the builtin Types.
    void registerTypes();

  public:
  };
} // namespace mlir
