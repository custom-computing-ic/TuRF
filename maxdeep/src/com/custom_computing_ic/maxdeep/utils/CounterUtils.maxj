package com.custom_computing_ic.maxdeep.utils;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFETypeFactory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;

public final class CounterUtils {

  public static DFEVar createCounter(
      CounterChain chain,
      int maxStep,
      KernelBase<?> owner) {

    DFEType countT = DFETypeFactory.dfeInt(32);
    DFEVar counter;
    if (maxStep == 1)
      counter = owner.constant.var(0);
    else
      counter = chain.addCounter(maxStep, 1);

    return counter.cast(countT);
  }
}
