package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.fuse.FusedConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public abstract class BaseConvLayerKernel extends KernelComponent {

  protected final ConvLayerParameters cp;
  protected final List<ConvLayerParameters> cps;
  protected final DFEType T;
  protected DFEVector<DFEVar> ifmap, ofmap;
  protected List<DFEVector<DFEVar>> coeffList;
  protected DFEVectorType<DFEVar> ifmapVecT, ofmapVecT;
  protected List<DFEVectorType<DFEVar>> coeffVecTList;


  public BaseConvLayerKernel(KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T,
      boolean useIfmapBuffer) {
    super(owner);

    this.cps = cps;
    this.cp = cps.get(0); // be compatible with a single layer case

    this.T = T;

    ifmapVecT = new DFEVectorType<DFEVar>(T, getIfmapVecSize());
    ifmap = ifmapVecT.newInstance(getOwner());

    ofmapVecT = new DFEVectorType<DFEVar>(T, getOfmapVecSize());
    ofmap = ofmapVecT.newInstance(getOwner());

    // create coefficents
    List<Integer> coeffVecSizeList = getCoeffVecSizeList();
    coeffVecTList = new ArrayList<DFEVectorType<DFEVar>>();
    coeffList = new ArrayList<DFEVector<DFEVar>>();
    for (int i = 0; i < coeffVecSizeList.size(); i++) {
      coeffVecTList.add(new DFEVectorType<DFEVar>(T, coeffVecSizeList[i]));
      coeffList.add(coeffVecTList[i].newInstance(owner));
    }
  }

  public BaseConvLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T) {
    this(owner, new ArrayList<ConvLayerParameters>(Arrays.asList(cp)), T);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T,
      boolean useIfmapBuffer) {
    this(owner, new ArrayList<ConvLayerParameters>(Arrays.asList(cp)), T, useIfmapBuffer);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T) {
    this(owner, cps, T, true);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, FusedConvLayerParameters fcp, DFEType T) {
    this(owner, fcp.cps, T);
  }


  public DFEVector<DFEVar> getIfmap() {
    return ifmap;
  }

  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  public abstract DFEVar getIfmapEn();

  public abstract int getIfmapVecSize();

  public List<DFEVector<DFEVar>> getCoeffList() {
    return coeffList;
  }

  public List<DFEVectorType<DFEVar>> getCoeffVecTList() {
    return coeffVecTList;
  }

  public abstract List<DFEVar> getCoeffEnList();

  public abstract List<Integer> getCoeffVecSizeList();

  public DFEVectorType<DFEVar> getOfmapVecT() {
    return ofmapVecT;
  }

  public abstract DFEVar getOfmapEn();

  public abstract int getOfmapVecSize();

  public DFEVector<DFEVar> getOfmap() {
    return ofmap;
  }

  public void setInputs(DFEVector<DFEVar> ifmap, List<DFEVector<DFEVar>> coeffList) {
    this.ifmap.connect(ifmap);

    if (this.coeffList.size() != coeffList.size())
      throw new IllegalArgumentException(String.format(
          "Coefficient lists are not matching in size: %d (kernel) != %d (parameter)",
          this.coeffList.size(), coeffList.size()));

    for (int i = 0; i < coeffList.size(); i++)
      this.coeffList[i].connect(coeffList[i]);
  }
}
