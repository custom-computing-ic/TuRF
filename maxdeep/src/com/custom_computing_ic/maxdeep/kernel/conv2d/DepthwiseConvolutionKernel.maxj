package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradIfmapTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradInverseTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradWeightsTransform;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class DepthwiseConvolutionKernel extends Kernel {

  public static final String[] INPUTS = {"ifmap", "weights", "bias"};
  public static final String[] OUTPUTS = {"ofmap"};
  public static final boolean OPT = true;

  public DepthwiseConvolutionKernel(KernelParameters params, int tileHeight, int tileWidth,
      int tileDepth, int kernelSize, int parWidth, int parDepth, DFEType T, boolean dbg) {
    this(params, tileHeight, tileWidth, tileDepth, kernelSize, parWidth, parDepth, T, dbg, false);
  }

  public DepthwiseConvolutionKernel(KernelParameters params, int tileHeight, int tileWidth,
      int tileDepth, int kernelSize, int parWidth, int parDepth, DFEType T, boolean dbg,
      boolean useWinograd) {
    super(params);

    if (useWinograd && parWidth != 1)
      throw new IllegalArgumentException("parWidth should be 1 if useWinograd is true");
    if (useWinograd && kernelSize != 3)
      throw new IllegalArgumentException(String.format(
          "kernelSize should be 3 if you use Winograd, got: %3d", kernelSize));

    int tileInputHeight = tileHeight + kernelSize - 1;
    int tileInputWidth = tileWidth + kernelSize - 1;

    if (tileInputWidth % parWidth != 0)
      throw new IllegalArgumentException(String.format("The input width of tile (%d) be divisible "
          + "by the number of parallelised units in the width axis (%d)", tileInputWidth, parWidth));

    if (!useWinograd) {
      // counters
      DFEType cT = dfeInt(32);
      CounterChain chain = control.count.makeCounterChain();
      // DFEVar c = chain.addCounter(tileDepth / parDepth, 1).cast(cT);
      DFEVar h = chain.addCounter(tileInputHeight, 1).cast(cT);
      DFEVar w = chain.addCounter(tileInputWidth / parWidth, 1).cast(cT);

      if (dbg) {
        debug.simPrintf("h = %KObj% w = %KObj%\n", h, w);
      }

      // initialise the convolution layer parameter
      ConvLayerParameters cp =
          (new ConvLayerParameters.Builder(tileInputHeight, tileInputWidth, tileDepth, 1,
              kernelSize)).PC(parDepth).PK(parWidth).dbg(dbg).useWinograd(useWinograd).build();

      // vector size
      int ifmapVecSize = parWidth * parDepth;
      int weightsVecSize = parDepth * kernelSize * kernelSize;
      int biasVecSize = parDepth;
      int ofmapVecSize = parWidth * parDepth;

      // create streams
      DFEVar ONE = constant.var(1).cast(dfeBool());

      DFEVar ifmapEn = ONE;
      DFEVar weightsEn = h.eq(0) & w.eq(0);
      DFEVar biasEn = h.eq(0) & w.eq(0);
      DFEVar ofmapEn = getOfmapEn(kernelSize, parWidth, h, w);

      DFEVector<DFEVar> ifmap = createStream(INPUTS[0], T, ifmapVecSize, ifmapEn, true);
      DFEVector<DFEVar> weights = createStream(INPUTS[1], T, weightsVecSize, weightsEn, true);
      DFEVector<DFEVar> bias = createStream(INPUTS[2], T, biasVecSize, biasEn, true);
      DFEVector<DFEVar> ofmap = createStream(OUTPUTS[0], T, ofmapVecSize, ofmapEn, false);

      // initialise line buffer
      ConvLayerLineBuffer lineBuffer = new ConvLayerLineBuffer(getKernel(), cp, T);
      lineBuffer.setInput(ifmap);
      DFEVector<DFEVar> lbuf = lineBuffer.getOutputVec();

      // prepare input for the PE
      List<DFEVector<DFEVar>> ifmapPE = getIfmapPE(this, lbuf, parDepth, parWidth, kernelSize, T);
      List<DFEVector<DFEVar>> weightsPE =
          getWeightsPE(this, weights, parDepth, parWidth, kernelSize, T);

      // run PE
      DFEVector<DFEVar> ofmapPE =
          process(this, ifmapPE, weightsPE, parDepth, parWidth, kernelSize, T);

      // add bias
      DFEVector<DFEVar> ofmapBias = addBias(ofmapPE, bias, parDepth, parWidth);

      // connect output
      ofmap.connect(ofmapBias);
    } else {
      int M = WinogradTransform.M;
      int TILE_SIZE = WinogradTransform.TILE_SIZE;
      int maxHeightCount = (tileInputHeight - TILE_SIZE + M) / M;
      int maxWidthCount = (tileInputWidth - TILE_SIZE + M) / M;

      // counters
      DFEType cT = dfeInt(32);
      CounterChain chain = control.count.makeCounterChain();
      // DFEVar c = chain.addCounter(tileDepth / parDepth, 1).cast(cT);
      DFEVar h = chain.addCounter(tileInputHeight, 1).cast(cT);
      DFEVar w = chain.addCounter(tileInputWidth, 1).cast(cT);

      if (dbg) {
        debug.simPrintf("h = %KObj% w = %KObj%\n", h, w);
      }

      // initialise the convolution layer parameter
      ConvLayerParameters cp =
          (new ConvLayerParameters.Builder(tileInputHeight, tileInputWidth, tileDepth, 1,
              kernelSize)).PC(parDepth).PK(parWidth).dbg(dbg).useWinograd(useWinograd).build();

      // vector size
      int ifmapVecSize = parWidth * parDepth;
      int weightsVecSize = parDepth * kernelSize * kernelSize;
      int biasVecSize = parDepth;
      int ofmapVecSize = M * M * parDepth;

      // create streams
      DFEVar ONE = constant.var(1).cast(dfeBool());

      DFEVar ifmapEn = ONE;
      DFEVar weightsEn = h.eq(0) & w.eq(0);
      DFEVar biasEn = h.eq(0) & w.eq(0);
      DFEVar ofmapEn = getWinogradObufWriteEn(h, w);

      DFEVector<DFEVar> ifmap = createStream(INPUTS[0], T, ifmapVecSize, ifmapEn, true);
      DFEVector<DFEVar> weights = createStream(INPUTS[1], T, weightsVecSize, weightsEn, true);
      DFEVector<DFEVar> bias = createStream(INPUTS[2], T, biasVecSize, biasEn, true);
      DFEVector<DFEVar> ofmap = createStream(OUTPUTS[0], T, ofmapVecSize, ofmapEn, false);

      // initialise line buffer
      ConvLayerLineBuffer lineBuffer = new ConvLayerLineBuffer(getKernel(), cp, T);
      lineBuffer.setInput(ifmap);
      DFEVector<DFEVar> lbuf = lineBuffer.getOutputVec();

      // run PE
      DFEVector<DFEVar> ofmapPE = processWinograd(this, lbuf, weights, parDepth, kernelSize, T);

      // add bias
      DFEVector<DFEVar> ofmapBias = addBiasWinograd(ofmapPE, bias, parDepth);

      // connect output
      ofmap.connect(ofmapBias);
    }

  }

  public static DFEVector<DFEVar> getIfmapWinogradPE(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      int parInDepth, DFEType T) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    DFEVectorType<DFEVar> RT = new DFEVectorType<DFEVar>(T, parInDepth * TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> R = RT.newInstance(owner);

    for (int i = 0; i < parInDepth; i++) {
      WinogradIfmapTransform winogradTransform = new WinogradIfmapTransform(owner, T, OPT);

      DFEVector<DFEVar> input = winogradTransform.getInputT().newInstance(owner);
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        input[j].connect(ifmap[i * TILE_SIZE * TILE_SIZE + j]);
      winogradTransform.setInput(input);
      DFEVector<DFEVar> trans = winogradTransform.getOutput();
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        R[i * TILE_SIZE * TILE_SIZE + j].connect(trans[j]);
    }

    return R;
  }

  public static DFEVector<DFEVar> getWeightsWinogradPE(KernelBase<?> owner,
      DFEVector<DFEVar> weights, int parInDepth, int kernelSize, DFEType T) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;

    DFEVectorType<DFEVar> RT = new DFEVectorType<DFEVar>(T, parInDepth * TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> R = RT.newInstance(owner);

    for (int i = 0; i < parInDepth; i++) {
      WinogradWeightsTransform winogradTransform = new WinogradWeightsTransform(owner, T, OPT);

      DFEVector<DFEVar> input = winogradTransform.getInputT().newInstance(owner);
      for (int j = 0; j < kernelSize * kernelSize; j++)
        input[j].connect(weights[i * kernelSize * kernelSize + j]);

      winogradTransform.setInput(input);
      DFEVector<DFEVar> trans = winogradTransform.getOutput();
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        R[i * TILE_SIZE * TILE_SIZE + j].connect(trans[j]);
    }

    return R;
  }

  public static DFEVector<DFEVar> process(KernelBase<?> owner, List<DFEVector<DFEVar>> ifmapPE,
      List<DFEVector<DFEVar>> weightsPE, int parDepth, int parWidth, int kernelSize, DFEType T) {
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, parDepth * parWidth);
    DFEVector<DFEVar> v = vT.newInstance(owner);

    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        owner.optimization.pushDSPFactor(1.0);
        DotProductKernel dp = new DotProductKernel(owner, kernelSize * kernelSize, T);
        owner.optimization.popDSPFactor();

        dp.setInputs(ifmapPE.get(pd * parWidth + pw), weightsPE.get(pd * parWidth + pw));
        v[pd * parWidth + pw].connect(dp.getOutput());
      }
    }

    return v;
  }

  public static DFEVector<DFEVar> processWinograd(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      DFEVector<DFEVar> weights, int parInDepth, int kernelSize, DFEType T) {
    // prepare input for the PE
    DFEVector<DFEVar> ifmapPE = getIfmapWinogradPE(owner, ifmap, parInDepth, T);
    DFEVector<DFEVar> weightsPE = getWeightsWinogradPE(owner, weights, parInDepth, kernelSize, T);

    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    int M = WinogradTransform.M;

    DFEVectorType<DFEVar> TOT = new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    DFEVectorType<DFEVar> OT = new DFEVectorType<DFEVar>(T, parInDepth * M * M);
    DFEVector<DFEVar> O = OT.newInstance(owner);

    for (int i = 0; i < parInDepth; i++) {
      DFEVector<DFEVar> TO = TOT.newInstance(owner);

      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++) {
        int idx = i * TILE_SIZE * TILE_SIZE + j;

        owner.optimization.pushDSPFactor(1.0);
        DFEVar r = ifmapPE[idx] * weightsPE[idx];
        owner.optimization.popDSPFactor();

        TO[j].connect(r);
      }

      WinogradInverseTransform transform = new WinogradInverseTransform(owner, T, OPT);
      transform.setInputMatrix(TO);
      DFEVector<DFEVar> trans = transform.getOutput();

      for (int j = 0; j < M * M; j++)
        O[i * M * M + j].connect(trans[j]);
    }

    return O;
  }

  private DFEVector<DFEVar> addBias(DFEVector<DFEVar> ofmapPE, DFEVector<DFEVar> bias,
      int parDepth, int parWidth) {
    DFEVector<DFEVar> ofmapPEBias = ofmapPE.getType().newInstance(getKernel());
    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        ofmapPEBias[pd * parWidth + pw].connect(ofmapPE[pd * parWidth + pw] + bias[pd]);
      }
    }
    return ofmapPEBias;
  }

  private DFEVector<DFEVar> addBiasWinograd(DFEVector<DFEVar> ofmapPE, DFEVector<DFEVar> bias,
      int parDepth) {

    int M = WinogradTransform.M;
    DFEVector<DFEVar> ofmapPEBias = ofmapPE.getType().newInstance(getKernel());

    for (int i = 0; i < parDepth; i++) {
      for (int j = 0; j < M * M; j++) {
        ofmapPEBias[i * M * M + j].connect(ofmapPE[i * M * M + j] + bias[i]);
      }
    }

    return ofmapPEBias;
  }

  private DFEVar getOfmapEn(int kernelSize, int parWidth, DFEVar h, DFEVar w) {
    return getObufWriteEn(kernelSize, parWidth, h, w);
  }

  private DFEVar getObufWriteEn(int kernelSize, int parWidth, DFEVar h, DFEVar w) {
    DFEVar rowAbove = h >= (kernelSize - 1);
    DFEVar colRight = (w * parWidth) >= (kernelSize - 1);

    return rowAbove & colRight;
  }

  private DFEVar getWinogradObufWriteEn(DFEVar h, DFEVar w) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    int M = WinogradTransform.M;

    DFEVar hi = h - TILE_SIZE + M + 1;
    DFEVar wi = w - TILE_SIZE + M + 1;

    DFEVar rowAbove = h >= (TILE_SIZE - 1);
    DFEVar colRight = w >= (TILE_SIZE - 1);
    DFEVar posHeight = (hi.get(0) | hi.get(1)).eq(0);
    DFEVar posWidth = (wi.get(0) | wi.get(1)).eq(0);

    // debug.simPrintf(
    // "h = %KObj% w = %KObj% hi = %KObj% wi = %KObj% posHeight = %KObj% posWidth = %KObj%\n", h,
    // w, hi, wi, posHeight, posWidth);

    return rowAbove & colRight & posHeight & posWidth;
  }

  /**
   * Create the ifmap for PE array from the output of the line buffer.
   * 
   * @param lbuf
   * @return
   */
  public static List<DFEVector<DFEVar>> getIfmapPE(KernelBase<?> owner, DFEVector<DFEVar> lbuf,
      int parDepth, int parWidth, int kernelSize, DFEType T) {
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, kernelSize * kernelSize);
    List<DFEVector<DFEVar>> ifmapPE = new ArrayList<DFEVector<DFEVar>>();

    int lbufWidth = kernelSize + parWidth - 1;
    int lbufHeight = kernelSize;

    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DFEVector<DFEVar> v = vT.newInstance(owner);

        for (int kh = 0; kh < kernelSize; kh++) {
          for (int kw = 0; kw < kernelSize; kw++) {
            int idx = pd * lbufHeight * lbufWidth + kh * lbufWidth + (kw + pw);
            v[kh * kernelSize + kw].connect(lbuf[idx]);
          }
        }

        ifmapPE.add(v);
      }
    }

    return ifmapPE;
  }

  public static List<DFEVector<DFEVar>> getWeightsPE(KernelBase<?> owner,
      DFEVector<DFEVar> weights, int parDepth, int parWidth, int kernelSize, DFEType T) {
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, kernelSize * kernelSize);
    List<DFEVector<DFEVar>> weightsPE = new ArrayList<DFEVector<DFEVar>>();

    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DFEVector<DFEVar> v = vT.newInstance(owner);
        for (int k = 0; k < kernelSize * kernelSize; k++) {
          int idx = pd * kernelSize * kernelSize + k;
          v[k].connect(weights[idx]);
        }
        weightsPE.add(v);
      }
    }

    return weightsPE;
  }

  private DFEVector<DFEVar> createStream(String name, DFEType T, int vecSize, DFEVar enable,
      boolean isInput) {
    DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(T, vecSize);
    DFEVector<DFEVar> vec = vecT.newInstance(getKernel());

    if (isInput)
      vec.connect(io.input(name, vecT, enable));
    else
      io.output(name, vecT, enable).connect(vec);

    return vec;
  }
}
