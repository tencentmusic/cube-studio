/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class RegionKindInterface;
namespace detail {
struct RegionKindInterfaceInterfaceTraits {
  struct Concept {
    RegionKind (*getRegionKind)(unsigned);
    bool (*hasSSADominance)(unsigned);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getRegionKind, hasSSADominance} {}

    static inline RegionKind getRegionKind(unsigned index);
    static inline bool hasSSADominance(unsigned index);
  };
};
} // end namespace detail
class RegionKindInterface : public ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct RegionKindInterfaceTrait : public ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public RegionKindInterfaceTrait<ConcreteOp> {};
  RegionKind getRegionKind(unsigned index);
  bool hasSSADominance(unsigned index);
};
template<typename ConcreteOp>
RegionKind detail::RegionKindInterfaceInterfaceTraits::Model<ConcreteOp>::getRegionKind(unsigned index) {
  return ConcreteOp::getRegionKind(index);
}
template<typename ConcreteOp>
bool detail::RegionKindInterfaceInterfaceTraits::Model<ConcreteOp>::hasSSADominance(unsigned index) {
  return getRegionKind(index) == RegionKind::SSACFG;
}
} // namespace mlir
