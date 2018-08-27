package com.custom_computing_ic.maxdeep.utils;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class StreamUtils {

  public static DFEVector<DFEVar> createStream(Kernel owner, String name, DFEType T, int vecSize,
      DFEVar enable, boolean isInput) {
    DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(T, vecSize);
    DFEVector<DFEVar> vec = vecT.newInstance(owner);

    if (isInput)
      vec.connect(owner.io.input(name, vecT, enable));
    else
      owner.io.output(name, vecT, enable).connect(vec);

    return vec;
  }
}
