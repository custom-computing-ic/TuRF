package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public abstract class ConvLayerBaseFmapBuffer extends KernelComponent {

  protected ConvLayerBaseFmapBuffer(KernelBase<?> owner) {
    super(owner);
  }

  public abstract DFEType getAddrT();

  public abstract DFEVector<DFEVar> port(DFEVector<DFEVar> data, DFEVar addr, DFEVar writeEn);

  public abstract DFEVectorType<DFEVar> getPortVecT();

  public abstract int getWidth();

  public abstract int getDepth();
}
