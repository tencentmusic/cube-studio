/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class CallOpInterface;
namespace detail {
struct CallOpInterfaceInterfaceTraits {
  struct Concept {
    CallInterfaceCallable (*getCallableForCallee)(::mlir::Operation *);
    Operation::operand_range (*getArgOperands)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getCallableForCallee, getArgOperands} {}

    static inline CallInterfaceCallable getCallableForCallee(::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getArgOperands(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class CallOpInterface : public ::mlir::OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct CallOpInterfaceTrait : public ::mlir::OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public CallOpInterfaceTrait<ConcreteOp> {};
  CallInterfaceCallable getCallableForCallee();
  Operation::operand_range getArgOperands();

    /// Resolve the callable operation for given callee to a
    /// CallableOpInterface, or nullptr if a valid callable was not resolved.
    /// `symbolTable` is an optional parameter that will allow for using a
    /// cached symbol table for symbol lookups instead of performing an O(N)
    /// scan.
    Operation *resolveCallable(SymbolTableCollection *symbolTable = nullptr);
  
};
template<typename ConcreteOp>
CallInterfaceCallable detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::getCallableForCallee(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCallableForCallee();
}
template<typename ConcreteOp>
Operation::operand_range detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgOperands(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgOperands();
}
} // namespace mlir
namespace mlir {
class CallableOpInterface;
namespace detail {
struct CallableOpInterfaceInterfaceTraits {
  struct Concept {
    Region *(*getCallableRegion)(::mlir::Operation *);
    ArrayRef<Type> (*getCallableResults)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getCallableRegion, getCallableResults} {}

    static inline Region *getCallableRegion(::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<Type> getCallableResults(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class CallableOpInterface : public ::mlir::OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct CallableOpInterfaceTrait : public ::mlir::OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public CallableOpInterfaceTrait<ConcreteOp> {};
  Region *getCallableRegion();
  ArrayRef<Type> getCallableResults();
};
template<typename ConcreteOp>
Region *detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getCallableRegion(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCallableRegion();
}
template<typename ConcreteOp>
ArrayRef<Type> detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getCallableResults(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCallableResults();
}
} // namespace mlir
