package com.custom_computing_ic.maxdeep.lib;

import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;

public
abstract class LayerParameters {
 public
  String dtype;  // data type name
 public
  int BW;  // bit width
 public
  int numFracBits;  // number of fraction bits in fixed-point.

 public
  CPUTypes getCPUTypes() {
    switch (BW) {
      case 1:
        return CPUTypes.INT8;
      case 8:
        return CPUTypes.INT8;
      case 16:
        return CPUTypes.INT16;
      case 32:
        return CPUTypes.INT32;
      default:
        throw new IllegalArgumentException("BW (" + BW + ") is not supported");
    }
  }

 public
  abstract long getIfmapStreamNumElems();

 public
  abstract long getCoeffStreamNumElems();

 public
  abstract long getOfmapStreamNumElems();

 public
  abstract int getIfmapVecSize();

 public
  abstract int getCoeffVecSize();

 public
  abstract int getOfmapVecSize();
}
