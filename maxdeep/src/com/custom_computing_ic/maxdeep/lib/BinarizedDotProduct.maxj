/**
 * 
 */
package com.custom_computing_ic.maxdeep.lib;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.utils.AdderTree;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * @author Ruizhe Zhao
 * 
 */
public class BinarizedDotProduct extends KernelComponent {

  private final DFEVector<DFEVar>     vecA;
  private final DFEVector<DFEVar>     vecB;
  private final DFEVar                out;
  private final DFEVectorType<DFEVar> vecT;
  private final int                   vecSize;

  public BinarizedDotProduct(KernelBase<?> owner, int vecSize) {
    super(owner);

    if (vecSize <= 0)
      throw new IllegalArgumentException("vecSize should be larger than 0");

    this.vecSize = vecSize;

    DFEType type = dfeUInt(1);

    vecT = new DFEVectorType<DFEVar>(type, vecSize);

    vecA = vecT.newInstance(owner);
    vecB = vecT.newInstance(owner);

    DFEVector<DFEVar> vecC = vecA ^ vecB;
    List<DFEVar> tmp = new ArrayList<DFEVar>();
    for (int i = 0; i < vecSize; i++)
      tmp.add(vecC[i].cast(getOutT()));

    // Output is the result of an adder-tree based reduction
    out = AdderTree.reduce(tmp);
  }
  
  public static DFEType getOutT(int vecSize) {
    return dfeUInt(getOutNumBits(vecSize));
  }

  public static int getOutNumBits(int vecSize) {
    return 1 + (int) Math.floor(Math.log10(vecSize) / Math.log10(2.0));
  }

  public DFEType getOutT() {
    return dfeUInt(getOutNumBits(vecSize));
  }
  
  public void setInputs(DFEVector<DFEVar> vecA, DFEVector<DFEVar> vecB) {
    this.vecA <== vecA;
    this.vecB <== vecB;
  }

  public DFEVectorType<DFEVar> getInputVecT() { return vecT; }

  public DFEVar getOutput() { return out; }

}
