/**
 *
 */
package com.custom_computing_ic.maxdeep.lib;

import com.custom_computing_ic.maxdeep.utils.AdderTree;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * A kernel component responsible for creating dot-product logic.
 *
 * The dot-product is implemented by an array of multipliers followed by an
 * adder tree with the same width.
 *
 * @author Ruizhe Zhao
 * @since 18/06/2017
 */
public
class DotProductKernel extends KernelComponent {
 private
  final DFEVector<DFEVar> vecA;
 private
  final DFEVector<DFEVar> vecB;
 private
  final DFEVar out;
 private
  final DFEVectorType<DFEVar> vecT;

 public DotProductKernel(KernelBase<?> owner, int vecSize, DFEType type) {
    this(owner, vecSize, type, false);
  }

  /**
   * Constructor for the DotProductKernelComponent.
   *
   * @param owner owner design of this component
   * @param vecSize size of the vectors to be computed
   * @param type data type.s
   */
 public DotProductKernel(KernelBase<?> owner, int vecSize, DFEType type, boolean dbg) {
    super(owner);

    if (vecSize <= 0)
      throw new IllegalArgumentException("vecSize should be larger than 0");

    vecT = new DFEVectorType<DFEVar>(type, vecSize);

    vecA = vecT.newInstance(owner);
    vecB = vecT.newInstance(owner);

    // An array of multipliers will be initialized here.
    DFEVector<DFEVar> vecC = vecA * vecB;
    if (dbg) {
      debug.simPrintf("[DotProd] vecA = %KObj%\n", vecA);
      debug.simPrintf("[DotProd] vecB = %KObj%\n", vecB);
      debug.simPrintf("[DotProd] vecC = %KObj%\n", vecC);
    }

    // Output is the result of an adder-tree based reduction
    out = AdderTree.reduce(vecC.getElementsAsList());
  }

 public
  void setInputs(DFEVector<DFEVar> vecA, DFEVector<DFEVar> vecB) {
    this.vecA.connect(vecA);
    this.vecB.connect(vecB);
  }

 public
  DFEVectorType<DFEVar> getInputVecT() { return vecT; }

 public
  DFEVar getOutput() { return out; }

 public
  DFEType getOutputT() { return out.getType(); }
}
