/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DFactorizedModuleParameter;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DFactorizedModuleParameter.ShapeMode;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
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
public class Conv2DDepthwiseProcessEngine extends KernelComponent {

  private final DFEType                       scalarT;
  private final DFEType                       indexT;
  private final DFEVectorType<DFEVar>         ifmapVecT;
  private final DFEVectorType<DFEVar>         coeffVecT;
  private final DFEVectorType<DFEVar>         ofmapVecT;
  private final Conv2DFactorizedModuleParameter conv2dParams;

  private final DFEVector<DFEVar>             ifmap;
  private final DFEVector<DFEVar>             coeff;
  private final DFEVector<DFEVar>             ofmap;

//  private final DFEVar                        height;
//  private final DFEVar                        width;
//  private final DFEVar                        knlHeight;
//  private final DFEVar                        knlWidth;
//  private final DFEVar                        numChnl, numFltr;

  /**
   * @param owner
   */
  public Conv2DDepthwiseProcessEngine(KernelBase<?> owner,
      Conv2DFactorizedModuleParameter conv2dParams, DFEType scalarT) {
    super(owner);

    if (conv2dParams.getShapeMode() != ShapeMode.STATIC)
      throw new IllegalArgumentException("shape mode should be STATIC");

    this.conv2dParams = conv2dParams;
    this.scalarT = scalarT;
    this.indexT = dfeInt(32);
    this.ifmapVecT = new DFEVectorType<DFEVar>(scalarT,
        getIfmapVecSize(conv2dParams));
    this.coeffVecT = new DFEVectorType<DFEVar>(scalarT,
        getCoeffVecSize(conv2dParams));
    this.ofmapVecT = new DFEVectorType<DFEVar>(scalarT,
        getOfmapVecSize(conv2dParams));

    this.ifmap = ifmapVecT.newInstance(owner);
    this.coeff = coeffVecT.newInstance(owner);
    this.ofmap = ofmapVecT.newInstance(owner);

//    this.height = indexT.newInstance(owner);
//    this.width = indexT.newInstance(owner);
//    this.knlHeight = indexT.newInstance(owner);
//    this.knlWidth = indexT.newInstance(owner);
//    this.numChnl = indexT.newInstance(owner);
//    this.numFltr = indexT.newInstance(owner);

    for (int p = 0; p < conv2dParams.getNumParaIfmapChnl(); p ++) {
      DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(scalarT, conv2dParams.getKnlSize());
      DFEVector<DFEVar> ifmapVec = vecT.newInstance(owner);
      DFEVector<DFEVar> coeffVec = vecT.newInstance(owner);
      for (int i = 0; i < conv2dParams.getKnlSize(); i ++) {
        ifmapVec[i] <== ifmap[p * conv2dParams.getKnlSize() +  i];
        coeffVec[i] <== coeff[p * conv2dParams.getKnlSize() +  i];
      }

      DotProductKernel dotProd = new DotProductKernel(owner,
          conv2dParams.getKnlSize(), scalarT);
      dotProd.setInputs(ifmapVec, coeffVec);
      this.ofmap[p] <== dotProd.getOutput();
    }

    debug.simPrintf("depth - %KObj% %KObj% %KObj%\n", ifmap, coeff, ofmap);
  }

  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  public DFEVectorType<DFEVar> getCoeffVecT() {
    return coeffVecT;
  }

  public DFEVectorType<DFEVar> getOfmapVecT() {
    return ofmapVecT;
  }

  public int getIfmapVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getKnlSize() * params.getIfmapNumParaChnl();
  }

  public int getCoeffVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getKnlSize() * params.getIfmapNumParaChnl();
  }

  public int getOfmapVecSize(Conv2DFactorizedModuleParameter params) {
    return params.getIfmapNumParaChnl();
  }

  public void setIfmap(DFEVector<DFEVar> ifmap) { this.ifmap <== ifmap; }

  public void setCoeff(DFEVector<DFEVar> coeff) { this.coeff <== coeff; }

  public DFEVector<DFEVar> getOfmap() { return ofmap; }

  public DFEVar getOfmapValid() { return constant.var(1).cast(dfeBool()); }

}
