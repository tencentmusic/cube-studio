/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SymbolOpInterface;
namespace detail {
struct SymbolOpInterfaceInterfaceTraits {
  struct Concept {
    StringRef (*getName)(::mlir::Operation *);
    void (*setName)(::mlir::Operation *, StringRef);
    mlir::SymbolTable::Visibility (*getVisibility)(::mlir::Operation *);
    bool (*isNested)(::mlir::Operation *);
    bool (*isPrivate)(::mlir::Operation *);
    bool (*isPublic)(::mlir::Operation *);
    void (*setVisibility)(::mlir::Operation *, mlir::SymbolTable::Visibility);
    void (*setNested)(::mlir::Operation *);
    void (*setPrivate)(::mlir::Operation *);
    void (*setPublic)(::mlir::Operation *);
    Optional<::mlir::SymbolTable::UseRange> (*getSymbolUses)(::mlir::Operation *, Operation *);
    bool (*symbolKnownUseEmpty)(::mlir::Operation *, Operation *);
    LogicalResult (*replaceAllSymbolUses)(::mlir::Operation *, StringRef, Operation *);
    bool (*isOptionalSymbol)(::mlir::Operation *);
    bool (*canDiscardOnUseEmpty)(::mlir::Operation *);
    bool (*isDeclaration)(::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{getName, setName, getVisibility, isNested, isPrivate, isPublic, setVisibility, setNested, setPrivate, setPublic, getSymbolUses, symbolKnownUseEmpty, replaceAllSymbolUses, isOptionalSymbol, canDiscardOnUseEmpty, isDeclaration} {}

    static inline StringRef getName(::mlir::Operation *tablegen_opaque_val);
    static inline void setName(::mlir::Operation *tablegen_opaque_val, StringRef name);
    static inline mlir::SymbolTable::Visibility getVisibility(::mlir::Operation *tablegen_opaque_val);
    static inline bool isNested(::mlir::Operation *tablegen_opaque_val);
    static inline bool isPrivate(::mlir::Operation *tablegen_opaque_val);
    static inline bool isPublic(::mlir::Operation *tablegen_opaque_val);
    static inline void setVisibility(::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis);
    static inline void setNested(::mlir::Operation *tablegen_opaque_val);
    static inline void setPrivate(::mlir::Operation *tablegen_opaque_val);
    static inline void setPublic(::mlir::Operation *tablegen_opaque_val);
    static inline Optional<::mlir::SymbolTable::UseRange> getSymbolUses(::mlir::Operation *tablegen_opaque_val, Operation * from);
    static inline bool symbolKnownUseEmpty(::mlir::Operation *tablegen_opaque_val, Operation * from);
    static inline LogicalResult replaceAllSymbolUses(::mlir::Operation *tablegen_opaque_val, StringRef newSymbol, Operation * from);
    static inline bool isOptionalSymbol(::mlir::Operation *tablegen_opaque_val);
    static inline bool canDiscardOnUseEmpty(::mlir::Operation *tablegen_opaque_val);
    static inline bool isDeclaration(::mlir::Operation *tablegen_opaque_val);
  };
};
} // end namespace detail
class SymbolOpInterface : public ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct SymbolOpInterfaceTrait : public ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    StringRef getName() {
      return mlir::SymbolTable::getSymbolName(this->getOperation());
    }
    void setName(StringRef name) {
      this->getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            StringAttr::get(this->getOperation()->getContext(), name));
    }
    mlir::SymbolTable::Visibility getVisibility() {
      return mlir::SymbolTable::getSymbolVisibility(this->getOperation());
    }
    bool isNested() {
      return getVisibility() == mlir::SymbolTable::Visibility::Nested;
    }
    bool isPrivate() {
      return getVisibility() == mlir::SymbolTable::Visibility::Private;
    }
    bool isPublic() {
      return getVisibility() == mlir::SymbolTable::Visibility::Public;
    }
    void setVisibility(mlir::SymbolTable::Visibility vis) {
      mlir::SymbolTable::setSymbolVisibility(this->getOperation(), vis);
    }
    void setNested() {
      setVisibility(mlir::SymbolTable::Visibility::Nested);
    }
    void setPrivate() {
      setVisibility(mlir::SymbolTable::Visibility::Private);
    }
    void setPublic() {
      setVisibility(mlir::SymbolTable::Visibility::Public);
    }
    Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation * from) {
      return ::mlir::SymbolTable::getSymbolUses(this->getOperation(), from);
    }
    bool symbolKnownUseEmpty(Operation * from) {
      return ::mlir::SymbolTable::symbolKnownUseEmpty(this->getOperation(),
                                                        from);
    }
    LogicalResult replaceAllSymbolUses(StringRef newSymbol, Operation * from) {
      return ::mlir::SymbolTable::replaceAllSymbolUses(this->getOperation(),
                                                         newSymbol, from);
    }
    bool isOptionalSymbol() {
      return false;
    }
    bool canDiscardOnUseEmpty() {
      // By default, base this on the visibility alone. A symbol can be
        // discarded as long as it is not public. Only public symbols may be
        // visible from outside of the IR.
        return getVisibility() != ::mlir::SymbolTable::Visibility::Public;
    }
    bool isDeclaration() {
      // By default, assume that the operation defines a symbol.
        return false;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      // If this is an optional symbol, bail out early if possible.
    auto concreteOp = cast<ConcreteOp>(op);
    if (concreteOp.isOptionalSymbol()) {
      if(!concreteOp->getAttr(::mlir::SymbolTable::getSymbolAttrName()))
        return success();
    }
    if (::mlir::failed(::mlir::detail::verifySymbol(op)))
      return ::mlir::failure();
    if (concreteOp.isDeclaration() && concreteOp.isPublic())
      return concreteOp.emitOpError("symbol declaration cannot have public "
             "visibility");
    return success();
    }

    using Visibility = mlir::SymbolTable::Visibility;
  
  };
  template <typename ConcreteOp>
  struct Trait : public SymbolOpInterfaceTrait<ConcreteOp> {};
  StringRef getName();
  void setName(StringRef name);
  mlir::SymbolTable::Visibility getVisibility();
  bool isNested();
  bool isPrivate();
  bool isPublic();
  void setVisibility(mlir::SymbolTable::Visibility vis);
  void setNested();
  void setPrivate();
  void setPublic();
  Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation * from);
  bool symbolKnownUseEmpty(Operation * from);
  LogicalResult replaceAllSymbolUses(StringRef newSymbol, Operation * from);
  bool isOptionalSymbol();
  bool canDiscardOnUseEmpty();
  bool isDeclaration();

    /// Custom classof that handles the case where the symbol is optional.
    static bool classof(Operation *op) {
      auto *concept = getInterfaceFor(op);
      if (!concept)
        return false;
      return !concept->isOptionalSymbol(op) ||
             op->getAttr(::mlir::SymbolTable::getSymbolAttrName());
    }
  
};
template<typename ConcreteOp>
StringRef detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::getName(::mlir::Operation *tablegen_opaque_val) {
  // Don't rely on the trait implementation as optional symbol operations
        // may override this.
        return mlir::SymbolTable::getSymbolName((llvm::cast<ConcreteOp>(tablegen_opaque_val)));
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setName(::mlir::Operation *tablegen_opaque_val, StringRef name) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setName(name);
}
template<typename ConcreteOp>
mlir::SymbolTable::Visibility detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::getVisibility(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVisibility();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isNested(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isNested();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isPrivate(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isPrivate();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isPublic(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isPublic();
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setVisibility(::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setVisibility(vis);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setNested(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setNested();
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setPrivate(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPrivate();
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setPublic(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPublic();
}
template<typename ConcreteOp>
Optional<::mlir::SymbolTable::UseRange> detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSymbolUses(::mlir::Operation *tablegen_opaque_val, Operation * from) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSymbolUses(from);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::symbolKnownUseEmpty(::mlir::Operation *tablegen_opaque_val, Operation * from) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).symbolKnownUseEmpty(from);
}
template<typename ConcreteOp>
LogicalResult detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::replaceAllSymbolUses(::mlir::Operation *tablegen_opaque_val, StringRef newSymbol, Operation * from) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).replaceAllSymbolUses(newSymbol, from);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isOptionalSymbol(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOptionalSymbol();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::canDiscardOnUseEmpty(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canDiscardOnUseEmpty();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDeclaration(::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDeclaration();
}
} // namespace mlir
namespace mlir {
class SymbolUserOpInterface;
namespace detail {
struct SymbolUserOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*verifySymbolUses)(::mlir::Operation *, ::mlir::SymbolTableCollection &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Model() : Concept{verifySymbolUses} {}

    static inline ::mlir::LogicalResult verifySymbolUses(::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable);
  };
};
} // end namespace detail
class SymbolUserOpInterface : public ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct SymbolUserOpInterfaceTrait : public ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public SymbolUserOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection & symbolTable);
};
template<typename ConcreteOp>
::mlir::LogicalResult detail::SymbolUserOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifySymbolUses(::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifySymbolUses(symbolTable);
}
} // namespace mlir
