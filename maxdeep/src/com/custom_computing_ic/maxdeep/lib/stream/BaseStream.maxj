package com.custom_computing_ic.maxdeep.lib.stream;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * The base class for Stream
 * 
 * @author Ruizhe Zhao
 * 
 */
public class BaseStream {

  private final String                name;
  private final int                   vecSize;
  private final DFEVar                enable;
  private final DFEType               T;
  private final DFEVectorType<DFEVar> vecT;
  private DFEVector<DFEVar>           placeholder;
  private final boolean               isInput;

  public BaseStream(String name, int vecSize, DFEType T, DFEVar enable) {
    this(name, vecSize, T, enable, true);
  }

  public BaseStream(String name, int vecSize, DFEType T, DFEVar enable,
      boolean isInput) {
    if (vecSize <= 0)
      throw new IllegalArgumentException("vecSize should be larger than 0.");

    this.name = name;
    this.vecSize = vecSize;
    this.T = T;
    this.enable = enable;
    this.vecT = new DFEVectorType<DFEVar>(T, vecSize);
    this.isInput = isInput;
  }

  public String getName() {
    return name;
  }

  public int getVecSize() {
    return vecSize;
  }

  public DFEVar getEnable() {
    return enable;
  }

  public DFEType getT() {
    return T;
  }

  public DFEVectorType<DFEVar> getVecT() {
    return vecT;
  }

  public DFEVector<DFEVar> getPlaceholder(KernelBase<?> owner) {
    if (this.placeholder != null)
      return this.placeholder;

    this.placeholder = vecT.newInstance(owner);
    return this.placeholder;
  }

  public void setIO(Kernel owner) throws RuntimeException {
    owner.getManager().logMsg(String.format("setting IO for %s", name));

    if (placeholder == null)
      throw new RuntimeException("please initialise placeholder at first");

    if (isInput)
      placeholder.connect(owner.io.input(name, vecT, enable));
    else
      owner.io.output(name, vecT, enable).connect(placeholder);
  }

}
