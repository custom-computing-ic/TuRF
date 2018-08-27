package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * Kernel for generating Conv2DKernel different parameters
 * 
 * @author rz3515
 * 
 */
public class ConvLayerWrapKernel extends Kernel {
  public static final String IFMAP_NAME = "ifmap";
  public static final String COEFF_NAME = "coeff";
  public static final String DEPTHWISE_COEFF_NAME = "depthwise_coeff";
  public static final String POINTWISE_COEFF_NAME = "pointwise_coeff";
  public static final String OFMAP_NAME = "ofmap";

  private final ConvLayerParameters cp;
  private final DFEType T;

  public ConvLayerWrapKernel(KernelParameters params, ConvLayerParameters cp) {
    this(params, cp, 1);
  }

  public ConvLayerWrapKernel(KernelParameters params, ConvLayerParameters cp, int numCoeffFifoSplits) {
    super(params);

    // KernelConfiguration config = getKernelConfig();
    // config.optimization.setFIFOImplementationBRAMThreshold(2048);
    // config.optimization
    // .setDSPMulAddChainBehavior(DSPMulAddChainBehaviour.IGNORE);

    this.cp = cp;

    // TODO: This type will return a signed integer when BW > 1
    T = (cp.BW == 1) ? dfeUInt(1) : cp.getDFEType();

    /**
     * TODO: what is the proper way to initialise conv and cast the type?
     */
    if (cp.type == Type.STANDARD || cp.type == Type.POINTWISE) {
      // debug.pushEnableNumericExceptions(true);
      // optimization.pushRoundingMode(RoundingMode.TRUNCATE);
      BaseConvLayerKernel conv = ConvLayerKernelFactory.create(getKernel(), cp, T);
      // optimization.popRoundingMode();
      // debug.popEnableNumericExceptions();

      DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, conv.getIfmapVecT(), conv.getIfmapEn());
      List<DFEVector<DFEVar>> coeffList = createCoeffList(conv, numCoeffFifoSplits);

      conv.setInputs(ifmap, coeffList);
      io.output(OFMAP_NAME, conv.getOfmapVecT(), conv.getOfmapEn()).connect(conv.getOfmap());

    } else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      DepthwiseSeparableConvLayerKernel conv =
          new DepthwiseSeparableConvLayerKernel(getKernel(), cp, T);

      DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, conv.getIfmapVecT(), conv.getIfmapEn());
      DFEVector<DFEVar> depthwiseCoeff =
          io.input(DEPTHWISE_COEFF_NAME, conv.getDepthwiseCoeffVecT(), conv.getDepthwiseCoeffEn());
      DFEVector<DFEVar> pointwiseCoeff =
          io.input(POINTWISE_COEFF_NAME, conv.getPointwiseCoeffVecT(), conv.getPointwiseCoeffEn());

      conv.setInputs(ifmap, depthwiseCoeff, pointwiseCoeff);
      io.output(OFMAP_NAME, conv.getOfmapVecT(), conv.getOfmapEn()).connect(conv.getOfmap());

      if (cp.dbg) {
        debug.simPrintf("[ConvLayerWrapKernel] pointwise en = %d\n", conv.getPointwiseCoeffEn());
      }

    } else if (cp.type == Type.DEPTHWISE_SEPARABLE_V2) {
      // create the kernel for depthwise separable convolution V2
      getManager().logMsg("Initializing kernel for Depthwise Separable Convolution V2");

      DepthwiseSeparableConvLayerKernelV2 conv =
          new DepthwiseSeparableConvLayerKernelV2(getKernel(), cp, T);
      conv.setIO(getKernel());
    } else {
      throw new IllegalArgumentException("type has no been supported");
    }
  }

  public List<DFEVector<DFEVar>> createCoeffList(BaseConvLayerKernel conv, int numCoeffFifoSplits) {
    List<DFEVector<DFEVar>> coeffList = new ArrayList<DFEVector<DFEVar>>();
    List<DFEVectorType<DFEVar>> coeffVecTList = conv.getCoeffVecTList();
    List<Integer> coeffVecSizeList = conv.getCoeffVecSizeList();
    List<DFEVar> coeffEnList = conv.getCoeffEnList();

    int numCoeff = coeffVecSizeList.size();

    for (int c = 0; c < numCoeff; c++) {
      int coeffVecSize = coeffVecSizeList.get(c);
      DFEVectorType<DFEVar> coeffVecT = coeffVecTList.get(c);
      DFEVar coeffEn = coeffEnList.get(c);

      // We try to reduce the size of each FIFO
      if (numCoeffFifoSplits > 1) {
        if (coeffVecSize % numCoeffFifoSplits != 0)
          throw new IllegalArgumentException(String.format(
              "Coefficient vector size %d should be divisible by FIFO split size %d.",
              coeffVecSize, numCoeffFifoSplits));

        int splitVecSize = coeffVecSize / numCoeffFifoSplits;
        getManager().logMsg(
            String.format("Split coefficient stream: vec size %d "
                + "FIFO splits %d split vec size %d.", coeffVecSize, numCoeffFifoSplits,
                splitVecSize));

        DFEVectorType<DFEVar> CT = new DFEVectorType<DFEVar>(T, splitVecSize);

        DFEVector<DFEVar> coeff = coeffVecT.newInstance(this);
        for (int i = 0; i < numCoeffFifoSplits; i++) {
          DFEVector<DFEVar> tmp = io.input(COEFF_NAME + "_" + i, CT, coeffEn);
          for (int j = 0; j < splitVecSize; j++) {
            coeff[i * splitVecSize + j].connect(tmp[j]);
          }
        }
        coeffList.add(coeff);
      } else {
        // Just use the normal way
        coeffList.add(io.input(COEFF_NAME, coeffVecT, coeffEn));
      }
    }

    return coeffList;
  }
}
