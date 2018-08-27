package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class DepthwiseSeparableConvLayerKernel extends ConvLayerKernel {

  private ConvLayerParameters dcp, pcp;
  private DFEVector<DFEVar> depthwiseCoeff, pointwiseCoeff;

  public DepthwiseSeparableConvLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T) {
    super(owner, cp, T);
  }

  public void setInputs(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> depthwiseCoeff,
      DFEVector<DFEVar> pointwiseCoeff) {

    this.ifmap.connect(ifmap);

    this.coeffList.get(0).connect(depthwiseCoeff);
    this.coeffList.get(1).connect(pointwiseCoeff);
  }

  public void initStreams() {
    if (cp.type != Type.DEPTHWISE_SEPARABLE)
      throw new IllegalArgumentException(
          "type of the convolution layer should be DEPTHWISE_SEPARABLE");
    if (cp.seq != CompSeq.FILTER_MAJOR)
      throw new IllegalArgumentException(
          "Only FILTER_MAJOR is supported in DEPTHWISE_SEPARABLE type");

    // create convolution layer parameters for depthwise and pointwise layers
    dcp = cp.createDepthwiseParameters();
    pcp = cp.createPointwiseParameters();

    /* vector type */
    this.ifmapVecT = new DFEVectorType<DFEVar>(T, getIfmapVecSize());
    this.ofmapVecT = new DFEVectorType<DFEVar>(T, getOfmapVecSize());

    /* streams */
    this.ifmap = ifmapVecT.newInstance(getOwner());
    this.ofmap = ofmapVecT.newInstance(getOwner());

    coeffVecTList.add(getDepthwiseCoeffVecT());
    coeffVecTList.add(getPointwiseCoeffVecT());

    depthwiseCoeff = coeffVecTList[0].newInstance(getOwner());
    pointwiseCoeff = coeffVecTList[1].newInstance(getOwner());

    coeffList.add(depthwiseCoeff);
    coeffList.add(pointwiseCoeff);
  }

  @Override
  public void initConvLayer() {

    /* line buffer */
    lbuf = new ConvLayerLineBuffer(getOwner(), cp, T);
    lbuf.setInput(ifmap);
    DFEVector<DFEVar> lineBufVec = lbuf.getOutputVec();

    /* depthwise conv2d */
    DFEVector<DFEVar> depthOfmap = initDepthwiseConv2D(lineBufVec, depthwiseCoeff);

    /* ifmap buffer */
    ibuf = new ConvLayerIfmapBuffer(getOwner(), pcp, T);
    DFEVector<DFEVar> ifmapBufVec =
        ibuf.port(depthOfmap, getIfmapBufferAddr(), getIfmapBufferWriteEn());

    /* pointwise convolution */
    Conv2DKernel pwc = new Conv2DKernel(getOwner(), pcp, T);
    pwc.setInputs(ifmapBufVec, pointwiseCoeff);
    DFEVector<DFEVar> pointOfmap = pwc.getOfmap();

    /* ofmap buffer */
    obuf = new ConvLayerOfmapBuffer(getOwner(), pcp, T, prefix);
    obuf.setReset(getOfmapReset());
    ofmap.connect(obuf.port(pointOfmap, getOfmapBufferAddr(), getOfmapBufferWriteEn()));
  }

  private DFEVector<DFEVar> initDepthwiseConv2D(DFEVector<DFEVar> lineBufVec,
      DFEVector<DFEVar> coeff) {
    List<DFEVector<DFEVar>> ifmapVecList = createDepthwiseVecList(lineBufVec);
    List<DFEVector<DFEVar>> coeffVecList = createDepthwiseVecList(depthwiseCoeff);

    DFEVector<DFEVar> depthOfmap = getDepthwiseOfmapVecT().newInstance(getOwner());
    for (int i = 0; i < cp.PC; i++) {
      Conv2DKernel dwc = new Conv2DKernel(getOwner(), dcp, T);
      dwc.setInputs(ifmapVecList.get(i), coeffVecList.get(i));
      DFEVector<DFEVar> dwcOfmap = dwc.getOfmap();

      for (int j = 0; j < cp.PK; j++) {
        depthOfmap[i * cp.PK + j].connect(dwcOfmap[j]);
      }
    }

    return depthOfmap;
  }

  private int getDepthwiseOfmapVecSize() {
    return cp.PK * cp.PC;
  }

  private DFEVectorType<DFEVar> getDepthwiseOfmapVecT() {
    return new DFEVectorType<DFEVar>(T, getDepthwiseOfmapVecSize());
  }

  private List<DFEVector<DFEVar>> createDepthwiseVecList(DFEVector<DFEVar> rawVec) {
    int vecLen = rawVec.getSize() / cp.PC;
    System.out.printf("vecLen = %d rawVec.getSize() = %d cp.PC = %d\n", vecLen, rawVec.getSize(),
        cp.PC);
    List<DFEVector<DFEVar>> vecList = new ArrayList<DFEVector<DFEVar>>();

    for (int i = 0; i < cp.PC; i++) {
      DFEVector<DFEVar> vec = (new DFEVectorType<DFEVar>(T, vecLen)).newInstance(getOwner());

      for (int j = 0; j < vecLen; j++)
        vec[j].connect(rawVec[i * vecLen + j]);

      vecList.add(vec);
    }

    return vecList;
  }

  @Override
  protected DFEVar getIfmapBufferAddr() {
    return getOfmapBufferAddr(ibuf.getAddrT());
  }

  @Override
  public DFEVar getIfmapEn() {
    return f.eq(0);
  }

  @Override
  protected DFEVar getIfmapBufferWriteEn() {
    return f.eq(0) & getOfmapBufferWriteEn();
  }

  public int getDepthwiseCoeffVecSize() {
    return cp.PC * dcp.K * dcp.K;
  }

  public DFEVectorType<DFEVar> getDepthwiseCoeffVecT() {
    return new DFEVectorType<DFEVar>(T, getDepthwiseCoeffVecSize());
  }

  public DFEVar getDepthwiseCoeffEn() {
    return f.eq(0) & h.eq(0) & w.eq(0);
  }

  public int getPointwiseCoeffVecSize() {
    return pcp.PC * pcp.PF;
  }

  public DFEVectorType<DFEVar> getPointwiseCoeffVecT() {
    return new DFEVectorType<DFEVar>(T, getPointwiseCoeffVecSize());
  }

  public DFEVar getPointwiseCoeffEn() {
    return h.eq(0) & w.eq(0);
  }
}
