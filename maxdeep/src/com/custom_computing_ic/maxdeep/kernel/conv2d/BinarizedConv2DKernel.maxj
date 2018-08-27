package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DKernel;
import com.custom_computing_ic.maxdeep.lib.BinarizedDotProduct;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;

public
class BinarizedConv2DKernel extends Conv2DKernel {
 public BinarizedConv2DKernel(KernelBase<?> owner, ConvLayerParameters cp) {
    super(owner, cp, dfeUInt(1));
  }

  @Override public DFEVar dotprod(DFEVector<DFEVar> ifmap,
                                  DFEVector<DFEVar> coeff) {
    BinarizedDotProduct bdp = new BinarizedDotProduct(getOwner(), cp.K * cp.K);
    bdp.setInputs(ifmap, coeff);
    return bdp.getOutput();
  }

  @Override public DFEType getOfmapScalarT() {
    if (cp.BW == 1) return BinarizedDotProduct.getOutT(cp.K * cp.K);

    return T;
  }
}
