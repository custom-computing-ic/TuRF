package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class BinaryPackKernel extends Kernel {

  public static final String INP_NAME  = "PACK_INP";
  public static final String OUT_NAME  = "PACK_OUT";

  public static final int    BIT_WIDTH = 8;

  public BinaryPackKernel(KernelParameters p) {
    super(p);

    DFEVectorType<DFEVar> inpT =
        new DFEVectorType<DFEVar>(dfeUInt(1), BIT_WIDTH);
    DFEType outT = dfeUInt(BIT_WIDTH);

    DFEVector<DFEVar> inp = io.input(INP_NAME, inpT);
    DFEVar out = inp.pack().cast(outT);

    io.output(OUT_NAME, outT).connect(out);
  }
}
