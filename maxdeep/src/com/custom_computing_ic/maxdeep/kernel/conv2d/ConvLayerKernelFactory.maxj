package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;

/**
 * Static factory to build convolutional layer kernel
 * 
 * @author rz3515
 * 
 */
public class ConvLayerKernelFactory {
  public static BaseConvLayerKernel create(KernelBase<?> owner, ConvLayerParameters cp, DFEType T) {
    if (cp.type == Type.STANDARD)
      return new ConvLayerKernel(owner, cp, T);
    if (cp.type == Type.POINTWISE)
      return new PointwiseConvolutionKernel(owner, cp, T);
    if (cp.type == Type.DEPTHWISE_SEPARABLE)
      return new DepthwiseSeparableConvLayerKernel(owner, cp, T);
    if (cp.type == Type.DEPTHWISE_SEPARABLE_V2)
      return new DepthwiseSeparableConvLayerKernelV2(owner, cp, T);

    throw new IllegalArgumentException(String.format(
        "ConvLayerKernel type %s cannot be recognised by the factory.", cp.type.name()));
  }
}
