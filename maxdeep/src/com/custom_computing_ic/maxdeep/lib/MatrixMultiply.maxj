package com.custom_computing_ic.maxdeep.lib;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class MatrixMultiply extends KernelComponent {

  private final DFEVector<DFEVar> mA;
  private final DFEVector<DFEVar> mB;
  private final DFEVector<DFEVar> mC;

  public MatrixMultiply(KernelBase<?> owner, int M, int N, int K, DFEType T) {
    super(owner);

    DFEVectorType<DFEVar> mAT = new DFEVectorType<DFEVar>(T, M * K);
    DFEVectorType<DFEVar> mBT = new DFEVectorType<DFEVar>(T, K * N);
    DFEVectorType<DFEVar> mCT = new DFEVectorType<DFEVar>(T, M * N);
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, K);

    mA = mAT.newInstance(owner);
    mB = mBT.newInstance(owner);
    mC = mCT.newInstance(owner);

    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        DFEVector<DFEVar> vA = vT.newInstance(owner);
        DFEVector<DFEVar> vB = vT.newInstance(owner);

        for (int k = 0; k < K; k++) {
          vA[k].connect(mA[m * K + k]);
          vB[k].connect(mB[k * N + n]);
        }

        DotProductKernel dp = new DotProductKernel(owner, K, T);
        dp.setInputs(vA, vB);
        mC[m * N + n].connect(dp.getOutput());
      }
    }
  }

  public void setInputMatrices(DFEVector<DFEVar> matA, DFEVector<DFEVar> matB) {
    mA.connect(matA);
    mB.connect(matB);
  }

  public DFEVector<DFEVar> getOutput() {
    return mC;
  }

}
