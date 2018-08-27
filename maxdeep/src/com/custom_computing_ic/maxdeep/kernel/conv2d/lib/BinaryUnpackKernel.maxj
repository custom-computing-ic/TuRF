package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class BinaryUnpackKernel extends Kernel {

  public static final String INP_NAME  = "UNPACK_INP";
  public static final String OUT_NAME  = "UNPACK_OUT";

  public static final int    BIT_WIDTH = 8;

  public BinaryUnpackKernel(KernelParameters p) {
    super(p);

    DFEVectorType<DFEVar> outT =
        new DFEVectorType<DFEVar>(dfeUInt(1), BIT_WIDTH);
    DFEType inpT = dfeUInt(BIT_WIDTH);

    DFEVar inp = io.input(INP_NAME, inpT);
    DFEVector<DFEVar> out = outT.newInstance(getKernel());

    for (int i = 0; i < BIT_WIDTH; i++)
      out[i].connect(inp.slice(i).cast(dfeUInt(1)));

    io.output(OUT_NAME, outT).connect(out);
  }
}
