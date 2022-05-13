/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class TosaOp;
namespace detail {
struct TosaOpInterfaceTraits {
  struct Concept {
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{} {}

  };
};
} // end namespace detail
class TosaOp : public ::mlir::OpInterface<TosaOp, detail::TosaOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<TosaOp, detail::TosaOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct TosaOpTrait : public ::mlir::OpInterface<TosaOp, detail::TosaOpInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public TosaOpTrait<ConcreteOp> {};
};
