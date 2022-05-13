/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
class FastmathFlagsInterface;
namespace detail {
struct FastmathFlagsInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LLVM::FastmathFlags (*fastmathFlags)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{fastmathFlags} {}

    static inline ::mlir::LLVM::FastmathFlags fastmathFlags(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class FastmathFlagsInterface : public ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct FastmathFlagsInterfaceTrait : public ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public FastmathFlagsInterfaceTrait<ConcreteOp> {};
  ::mlir::LLVM::FastmathFlags fastmathFlags();
};
template<typename ConcreteOp>
::mlir::LLVM::FastmathFlags detail::FastmathFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::fastmathFlags(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).fastmathFlags();
}
} // namespace LLVM
} // namespace mlir
