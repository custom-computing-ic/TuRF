/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.Conv2DParameter.Mode;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DFactorizedModuleParameter.ShapeMode;
import com.custom_computing_ic.maxdeep.utils.AdderTree;
import com.maxeler.maxcompiler.v2.errors.MaxCompilerAPIError;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * This class implements the pointwise convolution process engine.
 *
 * @author Ruizhe Zhao
 * @since 22/06/2017
 */
public final class Conv2DPointwiseProcessEngine extends
    KernelComponent {

  private final DFEType type;
  private final DFEType indexType;
  private final int     maxHeight;
  private final int     maxWidth;
  private final int     minHeight;
  private final int     minWidth;
  private final DFEVar  height;
  private final DFEVar  width;
  private final DFEVar  numChnl;
  private final DFEVector<DFEVar>  inp;
  private final DFEVector<DFEVar>  wgt;
  private final DFEVector<DFEVar>  out;
  private final DFEVar  outVld;
  private DFEVar        cntChnl;
  @SuppressWarnings("unused")
  private DFEVar        cntIdx;
  private final Mode    mode;

  private final int numParaChnl;
  private final int numParaFltr;

  private boolean hasHeight = false;
  private boolean hasWidth = false;
  private boolean hasNumChnl = false;
  private boolean hasInput = false;
  private boolean hasWeight = false;

  /**
   * The only pointwise constructor you should use.
   * @param owner
   * @param conv2dParams
   * @param type
   */
  public Conv2DPointwiseProcessEngine(KernelBase<?> owner,
      Conv2DFactorizedModuleParameter conv2dParams, DFEType type) {
    this(owner,
        type,
        conv2dParams.getOfmapHeight(),
        conv2dParams.getOfmapWidth(),
        conv2dParams.getOfmapHeight(),
        conv2dParams.getOfmapWidth(),
        Mode.FLTR_MAJOR,
        conv2dParams.getIfmapNumParaChnl(),
        conv2dParams.getOfmapNumParaChnl());

    if (conv2dParams.getShapeMode() != ShapeMode.STATIC)
      throw new IllegalArgumentException("shape mode can only support STATIC in pointwise conv2d");

    this.setHeight(constant.var(conv2dParams.getOfmapHeight()).cast(indexType));
    this.setWidth(constant.var(conv2dParams.getOfmapWidth()).cast(indexType));
    this.setNumChnl(constant.var(conv2dParams.getIfmapNumChnl()).cast(indexType));
    // this.setNumFltr(constant.var(conv2dParams.getOfmapNumChnl()).cast(indexType));
  }

  @Deprecated
  public Conv2DPointwiseProcessEngine(KernelBase<?> owner,
      Conv2DProcessEngineParameters conv2dParams, DFEType type) {
    this(owner,
        type,
        conv2dParams.getMaxHeight(),
        conv2dParams.getMaxWidth(),
        conv2dParams.getMinHeight(),
        conv2dParams.getMinWidth(),
        conv2dParams.getMode(),
        conv2dParams.getNumParaChnl(),
        conv2dParams.getNumParaFltr());
  }

  /**
   * Constructor.
   *
   * @param owner
   */
  public Conv2DPointwiseProcessEngine(KernelBase<?> owner,
      DFEType type, int maxHeight, int maxWidth, int minHeight, int minWidth, Mode mode, int numParaChnl, int numParaFltr) {
    super(owner);

    this.maxHeight = maxHeight;
    this.maxWidth = maxWidth;
    this.minHeight = minHeight;
    this.minWidth = minWidth;
    this.type = type;
    this.indexType = dfeInt(32);
    this.height = indexType.newInstance(owner);
    this.width = indexType.newInstance(owner);
    this.numChnl = indexType.newInstance(owner);
    this.mode = mode;
    this.numParaChnl = numParaChnl;
    this.numParaFltr = numParaFltr;

    DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(type, numParaChnl);
    DFEVectorType<DFEVar> wgtT = new DFEVectorType<DFEVar>(type, numParaChnl * numParaFltr);
    DFEVectorType<DFEVar> outT = new DFEVectorType<DFEVar>(type, numParaFltr);
    inp = vecT.newInstance(owner);
    wgt = wgtT.newInstance(owner);
    out = outT.newInstance(owner);
    outVld = dfeBool().newInstance(owner);

    compute();

    debug.simPrintf("point - %KObj% %KObj% %KObj% %d\n", inp, wgt, out, outVld);
  }

  public void setInput(DFEVector<DFEVar> inp) { this.hasInput = true; this.inp <== inp; }

  public void setIfmap(DFEVector<DFEVar> ifmap) { setInput(ifmap); }

  public void setWeight(DFEVector<DFEVar> wgt) { this.hasWeight = true; this.wgt <== wgt; }

  public void setCoeff(DFEVector<DFEVar> coeff) { setWeight(coeff); }

  public void setHeight(DFEVar height) { this.hasHeight = true; this.height <== height; }

  public void setWidth(DFEVar width) { this.hasWidth = true; this.width <== width; }

  public void setNumChnl(DFEVar numChnl) { this.hasNumChnl = true; this.numChnl <== numChnl; }

  public DFEVector<DFEVar> getOutput() { return out; }

  public DFEVector<DFEVar> getOfmap() { return getOutput(); }

  public DFEVar getOutputValid() { return outVld; }

  public DFEVar getOfmapValid() { return outVld; }

  private void compute() {
    if (mode == Mode.CHNL_MAJOR)
      throw new IllegalArgumentException(
          "The channel major mode of pointwise conv2d is not implemented yet.");
    else if (mode == Mode.FLTR_MAJOR)
      computeByFltrMajor();
    else
      throw new IllegalArgumentException("Cannot recognize mode.");
  }

  private void computeByFltrMajor() {
    CounterChain chain = getOwner().control.count.makeCounterChain();
    cntChnl = chain.addCounter(numChnl.cast(dfeUInt(32)), numParaChnl).cast(indexType);
    cntIdx = chain.addCounter((height * width).cast(dfeUInt(32)), 1).cast(indexType);

    for (int f = 0; f < numParaFltr; f ++) {
      DFEVectorType<DFEVar> tmpT = new DFEVectorType<DFEVar>(type, numParaChnl);
      DFEVector<DFEVar> tmp = tmpT.newInstance(getOwner());
      for (int c = 0; c < numParaChnl; c ++) {

        DFEVar product = inp[c] * wgt[f * numParaChnl + c];
        DFEVar carriedSum = type.newInstance(getOwner());
        DFEVar sum = cntChnl === 0 ? 0 : carriedSum;
        DFEVar newSum = sum + product;

        DFEVar offset = -(height * width);
        int minOffset = -(maxHeight * maxWidth);
        int maxOffset = -(minHeight * minWidth);
        carriedSum <== stream.offset(newSum, offset, minOffset, maxOffset);

        tmp[c] <== newSum;
      }
      out[f] <== AdderTree.reduce(tmp.getElementsAsList());
    }

    outVld <== cntChnl === numChnl - numParaChnl;
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();

    if (!hasInput)
      throw new MaxCompilerAPIError("Input has not been connected.");
    if (!hasWeight)
      throw new MaxCompilerAPIError("Weight has not been connected.");
    if (!hasHeight)
      throw new MaxCompilerAPIError("Height has not been connected.");
    if (!hasWidth)
      throw new MaxCompilerAPIError("Witdh has not been connected.");
    if (!hasNumChnl)
      throw new MaxCompilerAPIError("NumChnl has not been connected.");
  }
}
