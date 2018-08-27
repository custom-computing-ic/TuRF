package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * Conv2D PE.
 * 
 * @author Ruizhe Zhao
 * @since 18/06/2017
 * 
 */
public class Conv2DProcessEngineKernelComponent extends KernelComponent {

  private final DFEVar            knlHeight, knlWidth;
  private final DFEVector<DFEVar> inp;
  private final DFEVector<DFEVar> wgt;
  private final DFEVector<DFEVar> out;

  public Conv2DProcessEngineKernelComponent(KernelBase<?> owner,
      int maxKnlHeight, int maxKnlWidth, DFEType type) {
    super(owner);

    DFEType indexType = dfeInt(32);
    knlHeight = indexType.newInstance(owner);
    knlWidth = indexType.newInstance(owner);

    int maxKnlSize = maxKnlHeight * maxKnlWidth;
    DFEVectorType<DFEVar> vecType = new DFEVectorType<DFEVar>(type, maxKnlSize);
    inp = vecType.newInstance(owner);
    wgt = vecType.newInstance(owner);
    out = vecType.newInstance(owner); // worst case

    DotProductKernel dotProd = new DotProductKernel(owner,
        maxKnlSize, type);
    dotProd.setInputs(inp, wgt);
    out[0] <== dotProd.getOutput();
    for (int i = 1; i < maxKnlSize; i ++)
      out[i] <== constant.var(0);
  }
  
  public void setKernelShape(DFEVar knlHeight, DFEVar knlWidth) {
    this.knlHeight <== knlHeight;
    this.knlWidth <== knlWidth;
  }
  
  public void setInput(DFEVector<DFEVar> inp) {
    this.inp <== inp;
  }
  
  public void setWeight(DFEVector<DFEVar> wgt) {
    this.wgt <== wgt;
  }
  
  public DFEVector<DFEVar> getInput() { return inp; }
  
  public DFEVector<DFEVar> getWeight() { return wgt; }
  
  public DFEVector<DFEVar> getOutput() { return out; }
  
  @Override
  protected void finalize() throws Throwable {
    super.finalize();
  }
}
