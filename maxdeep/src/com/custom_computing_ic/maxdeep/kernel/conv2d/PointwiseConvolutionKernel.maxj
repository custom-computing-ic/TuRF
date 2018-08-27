package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * We try to evaluate the hardware design of doing point-wise convolution on FPGA.
 * 
 * @author rz3515
 * 
 */
public class PointwiseConvolutionKernel extends BaseConvLayerKernel {

  private final DFEVar f, c, h, w;
  private final DFEVector<DFEVar> coeff;

  public PointwiseConvolutionKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T) {
    super(owner, cp, T);

    if (cp.seq != ConvLayerParameters.CompSeq.FILTER_MAJOR)
      throw new IllegalArgumentException("Only support filter major in Pointwise convolution.");

    this.coeff = coeffList.get(0);

    // initialise counters
    CounterChain chain = control.count.makeCounterChain();
    f = chain.addCounter(cp.F / cp.PF, 1);
    c = chain.addCounter(cp.C / cp.PC, 1);
    h = (cp.H == cp.PH) ? constant.var(0) : chain.addCounter(cp.H / cp.PH, 1);
    w = (cp.W == cp.PW) ? constant.var(0) : chain.addCounter(cp.W / cp.PW, 1);

    if (cp.dbg) {
      debug.simPrintf("f = %KObj% c = %KObj% h = %KObj% w = %KObj%\n", f, c, h, w);
      debug.simPrintf("ifmap[%3d, %3d, %3d] = %KObj%\n", c, h, w, ifmap);
      debug.simPrintf("ofmap[%3d, %3d, %3d] = %KObj%\n", f, h, w, ofmap);
      debug.simPrintf("coeff[%3d, %3d] = %KObj%\n", f, c, coeff);
    }

    // input feature map buffer
    ConvLayerIfmapBuffer ibuf = new ConvLayerIfmapBuffer(owner, cp, T);
    DFEVar ibufAddr = getIbufAddr(cp, ibuf.getAddrT(), c, h, w);
    DFEVar ibufWriteEn = getIbufWriteEn(f);
    DFEVector<DFEVar> ibufOutput = ibuf.port(ifmap, ibufAddr, ibufWriteEn);

    // output feature map buffer
    ConvLayerOfmapBuffer obuf = new ConvLayerOfmapBuffer(owner, cp, T);
    obuf.setReset(getObufReset(c));

    // dot-product units
    DFEVector<DFEVar> procResult =
        process(owner, ibufOutput, coeff, cp.PH, cp.PW, cp.PC, cp.PF, T, c);

    if (cp.dbg) {
      debug.simPrintf("dp_out[%3d, %3d, %3d] = %KObj%\n", f, h, w, procResult);
    }

    // output feature map buffer
    ofmap.connect(obuf.port(procResult, getObufAddr(cp, obuf.getAddrT(), h, w),
        getObufWriteEn(cp, h, w)));
  }

  @Override
  public int getIfmapVecSize() {
    return cp.PH * cp.PW * cp.PC;
  }

  @Override
  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  @Override
  public DFEVar getIfmapEn() {
    return f.eq(0);
  }

  @Override
  public List<DFEVar> getCoeffEnList() {
    List<DFEVar> coeffEnList = new ArrayList<DFEVar>();
    coeffEnList.add(h.eq(0) & w.eq(0));

    return coeffEnList;
  }

  @Override
  public List<Integer> getCoeffVecSizeList() {
    List<Integer> coeffVecSizeList = new ArrayList<Integer>();
    coeffVecSizeList.add(cp.PC * cp.PF);
    return coeffVecSizeList;
  }

  @Override
  public DFEVar getOfmapEn() {
    return c.eq(cp.C / cp.PC - 1);
  }

  @Override
  public int getOfmapVecSize() {
    return cp.PF * cp.PH * cp.PW;
  }

  public static DFEVector<DFEVar> process(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      DFEVector<DFEVar> weights, int parHeight, int parWidth, int parInDepth, int parOutDepth,
      DFEType T, DFEVar c) {
    List<DFEVector<DFEVar>> ifmapPE = getIfmapPE(owner, ifmap, parHeight, parWidth, parInDepth, T);
    List<DFEVector<DFEVar>> weightsPE = getWeightsPE(owner, weights, parInDepth, parOutDepth, T);

    DFEVectorType<DFEVar> resT = new DFEVectorType<DFEVar>(T, parHeight * parWidth * parOutDepth);
    DFEVector<DFEVar> result = resT.newInstance(owner);
    for (int ph = 0; ph < parHeight; ph++) {
      for (int pw = 0; pw < parWidth; pw++) {
        for (int pf = 0; pf < parOutDepth; pf++) {
          DFEVector<DFEVar> currIfmap = ifmapPE.get(ph * parWidth + pw);
          DFEVector<DFEVar> currWeights = weightsPE.get(pf);

          // in total we instantiate parWidth * parOutDepth * parInDepth number of multipliers
          DotProductKernel dp = new DotProductKernel(owner, parInDepth, T);
          dp.setInputs(currIfmap, currWeights);
          DFEVar out = dp.getOutput();

          result[pf * parWidth * parHeight + ph * parWidth + pw].connect(out);
        }
      }
    }

    return result;
  }

  private DFEVar getIbufAddr(ConvLayerParameters cp, DFEType addrT, DFEVar c, DFEVar h, DFEVar w) {
    DFEVar HEIGHT = constant.var(cp.H / cp.PH).cast(addrT);
    DFEVar WIDTH = constant.var(cp.W / cp.PW).cast(addrT);

    DFEVar addr = c.cast(addrT) * HEIGHT * WIDTH;
    addr += h.cast(addrT) * WIDTH;
    addr += w.cast(addrT);

    return addr;
  }

  private DFEVar getIbufWriteEn(DFEVar f) {
    return f.eq(0);
  }

  private DFEVar getObufReset(DFEVar c) {
    return c.eq(0);
  }

  private DFEVar getObufAddr(ConvLayerParameters cp, DFEType addrT, DFEVar h, DFEVar w) {
    DFEVar WIDTH = constant.var(cp.W / cp.PW).cast(addrT);

    return (h.cast(addrT) * WIDTH + w.cast(addrT));
  }

  private DFEVar getObufWriteEn(ConvLayerParameters cp, DFEVar h, DFEVar w) {
    return constant.var(1).cast(dfeBool());
  }

  public static List<DFEVector<DFEVar>> getIfmapPE(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      int parHeight, int parWidth, int parInDepth, DFEType T) {
    List<DFEVector<DFEVar>> splits = new ArrayList<DFEVector<DFEVar>>();

    DFEVectorType<DFEVar> sT = new DFEVectorType<DFEVar>(T, parInDepth);
    for (int ph = 0; ph < parHeight; ph++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DFEVector<DFEVar> split = sT.newInstance(owner);

        for (int pc = 0; pc < parInDepth; pc++) {
          split[pc].connect(ifmap[pc * parWidth * parHeight + ph * parWidth + pw]);
        }

        splits.add(split);
      }
    }

    return splits;
  }

  public static List<DFEVector<DFEVar>> getWeightsPE(KernelBase<?> owner,
      DFEVector<DFEVar> weights, int parInDepth, int parOutDepth, DFEType T) {
    List<DFEVector<DFEVar>> splits = new ArrayList<DFEVector<DFEVar>>();

    DFEVectorType<DFEVar> sT = new DFEVectorType<DFEVar>(T, parInDepth);
    for (int pf = 0; pf < parOutDepth; pf++) {
      DFEVector<DFEVar> split = sT.newInstance(owner);

      for (int pc = 0; pc < parInDepth; pc++) {
        split[pc].connect(weights[pf * parInDepth + pc]);
      }

      splits.add(split);
    }

    return splits;
  }

}
