/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DFactorizedModuleParameter.ShapeMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * @author Ruizhe Zhao
 *
 */
public class Conv2DFactorizedModule extends KernelComponent {

  private final Conv2DFactorizedModuleParameter conv2dParams;
  private final DFEType                       scalarT;
  private final DFEType                       indexT;
  private final DFEVectorType<DFEVar>         ifmapVecT;
  private final DFEVectorType<DFEVar>         depthCoeffVecT;
  private final DFEVectorType<DFEVar>         pointCoeffVecT;
  private final DFEVectorType<DFEVar>         ofmapVecT;

  private final DFEVector<DFEVar>             ifmap;
  private final DFEVector<DFEVar>             depthCoeff;
  private final DFEVector<DFEVar>             pointCoeff;
  private final DFEVector<DFEVar>             ofmap;
  private final DFEVar                        ofmapValid;
  private final DFEVar                        height;
  private final DFEVar                        width;
  private final DFEVar                        numChnl;
  private final DFEVar                        numFltr;
  private final DFEVar                        f, c, i;

  /**
   * Default constructor of Conv2DFactorizedModule
   * @param owner
   * @param conv2dParams
   * @param scalarT
   */
  public Conv2DFactorizedModule(KernelBase<?> owner,
      Conv2DFactorizedModuleParameter conv2dParams, DFEType scalarT) {
    super(owner);

    if (conv2dParams.getShapeMode() != ShapeMode.STATIC)
      throw new IllegalArgumentException("Only STATIC shape mode is supported");

    this.conv2dParams = conv2dParams;
    this.scalarT = scalarT;
    this.indexT = dfeInt(32);

    this.ifmapVecT = new DFEVectorType<DFEVar>(scalarT,
        getIfmapVecSize(conv2dParams));
    this.depthCoeffVecT = new DFEVectorType<DFEVar>(scalarT,
        getDepthCoeffVecSize(conv2dParams));
    this.pointCoeffVecT = new DFEVectorType<DFEVar>(scalarT,
        getPointCoeffVecSize(conv2dParams));
    this.ofmapVecT = new DFEVectorType<DFEVar>(scalarT,
        getOfmapVecSize(conv2dParams));

    this.ifmap = ifmapVecT.newInstance(owner);
    this.depthCoeff = depthCoeffVecT.newInstance(owner);
    this.pointCoeff = pointCoeffVecT.newInstance(owner);
    this.ofmap = ofmapVecT.newInstance(owner);
    this.ofmapValid = dfeBool().newInstance(owner);

    this.height = indexT.newInstance(owner);
    this.width = indexT.newInstance(owner);
    this.numChnl = indexT.newInstance(owner);
    this.numFltr = indexT.newInstance(owner);

    CounterChain chain = getOwner().control.count.makeCounterChain();
    if (conv2dParams.getOfmapNumChnl() == 1)
      f = constant.var(0);
    else
	    f = chain.addCounter(conv2dParams.getOfmapNumChnl(),
	        conv2dParams.getOfmapNumParaChnl());

    if (conv2dParams.getIfmapNumChnl() == 1)
      c = constant.var(0);
    else
	    c = chain.addCounter(conv2dParams.getIfmapNumChnl(),
	        conv2dParams.getIfmapNumParaChnl());
    i = chain.addCounter(conv2dParams.getOfmapHeight() * conv2dParams.getOfmapWidth(), 1);

    Conv2DDepthwiseProcessEngine depthwise = new Conv2DDepthwiseProcessEngine(
        owner, conv2dParams, scalarT);
    Conv2DPointwiseProcessEngine pointwise = new Conv2DPointwiseProcessEngine(
        owner, conv2dParams, scalarT);
    Conv2DFactorizedModuleCache cache = new Conv2DFactorizedModuleCache(owner, conv2dParams, scalarT);

    depthwise.setIfmap(ifmap);
    depthwise.setCoeff(depthCoeff);
    cache.setInput(depthwise.getOfmap());
    cache.setWriteEnable(f === 0);

    pointwise.setIfmap(cache.getOutput());
    pointwise.setCoeff(pointCoeff);

    this.ofmap <== pointwise.getOfmap();
    this.ofmapValid <== pointwise.getOfmapValid();
  }

  public DFEVar getCurrOfmapChnl() {
    return f;
  }

  public DFEVar getCurrIfmapChnl() {
    return c;
  }

  public DFEVar getCurrOfmapIndex() {
    return i;
  }

  public DFEVar getIfmapEnable() {
    return f === 0;
  }

  public int getIfmapVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getKnlSize() * params.getIfmapNumParaChnl();
  }

  public int getDepthCoeffVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getKnlSize() * params.getIfmapNumParaChnl();
  }

  public int getPointCoeffVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getIfmapNumParaChnl() * params.getOfmapNumParaChnl();
  }

  public int getOfmapVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getOfmapNumParaChnl();
  }

  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  public DFEVectorType<DFEVar> getDepthCoeffVecT() {
    return depthCoeffVecT;
  }

  public DFEVectorType<DFEVar> getPointCoeffVecT() {
    return pointCoeffVecT;
  }

  public DFEVectorType<DFEVar> getOfmapVecT() {
    return ofmapVecT;
  }

  public void setIfmap(DFEVector<DFEVar> ifmap) { this.ifmap <== ifmap; }

  public void setDepthCoeff(DFEVector<DFEVar> depthCoeff) { this.depthCoeff <== depthCoeff; }

  public void setPointCoeff(DFEVector<DFEVar> pointCoeff) { this.pointCoeff <== pointCoeff; }

  public DFEVector<DFEVar> getOfmap() { return ofmap; }

  public DFEVar getOfmapValid() { return ofmapValid; }
}
