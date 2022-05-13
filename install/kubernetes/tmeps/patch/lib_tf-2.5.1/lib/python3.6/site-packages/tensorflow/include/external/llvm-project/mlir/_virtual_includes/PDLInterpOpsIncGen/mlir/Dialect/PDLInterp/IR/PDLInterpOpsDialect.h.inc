/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace pdl_interp {

class PDLInterpDialect : public ::mlir::Dialect {
  explicit PDLInterpDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<PDLInterpDialect>()) {
    
    getContext()->getOrLoadDialect<pdl::PDLDialect>();

    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("pdl_interp");
  }

    /// Returns the name of the function containing the matcher code. This
    /// function is called by the interpreter when matching an operation.
    static StringRef getMatcherFunctionName() { return "matcher"; }

    /// Returns the name of the module containing the rewrite functions. These
    /// functions are invoked by distinct patterns within the matcher function
    /// to rewrite the IR after a successful match.
    static StringRef getRewriterModuleName() { return "rewriters"; }
  };
} // namespace pdl_interp
} // namespace mlir
