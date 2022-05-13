/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

RegionKind mlir::RegionKindInterface::getRegionKind(unsigned index) {
      return getImpl()->getRegionKind(index);
  }
bool mlir::RegionKindInterface::hasSSADominance(unsigned index) {
      return getImpl()->hasSSADominance(index);
  }
