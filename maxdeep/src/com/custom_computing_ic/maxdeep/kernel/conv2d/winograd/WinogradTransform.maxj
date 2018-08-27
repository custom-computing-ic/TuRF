package com.custom_computing_ic.maxdeep.kernel.conv2d.winograd;

import com.custom_computing_ic.maxdeep.lib.MatrixMultiply;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class WinogradTransform {


  public static final int M = 4, R = 3;
  public static final int TILE_SIZE = M + R - 1;

  public static DFEVector<DFEVar> convertToMatrix(KernelBase<?> owner, float[][] A, int M, int N,
      DFEType T, boolean transpose) {
    DFEVectorType<DFEVar> matT = new DFEVectorType<DFEVar>(T, M * N);
    DFEVector<DFEVar> mat = matT.newInstance(owner);

    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        DFEVar val = owner.constant.var(A[m][n]).cast(T);
        if (transpose)
          mat[n * M + m].connect(val);
        else
          mat[m * N + n].connect(val);
      }
    }

    return mat;
  }

  public static DFEVector<DFEVar> transpose(KernelBase<?> owner, DFEVector<DFEVar> A, int M, int N,
      DFEType T) {
    DFEVectorType<DFEVar> matT = new DFEVectorType<DFEVar>(T, M * N);
    DFEVector<DFEVar> B = matT.newInstance(owner);

    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
        B[n * M + m].connect(A[m * N + n]);

    return B;
  }

  public static DFEVector<DFEVar> transform(KernelBase<?> owner, DFEVector<DFEVar> X,
      DFEVector<DFEVar> A, int M, int N, DFEType T) {
    if (X.getSize() != N * N)
      throw new IllegalArgumentException(String.format(
          "The DFEVector to be transformed should have size %d but got %d", N * N, X.getSize()));

    owner.optimization.pushDSPFactor(0.0);
    MatrixMultiply mma = new MatrixMultiply(owner, M, N, N, T);
    MatrixMultiply mmb = new MatrixMultiply(owner, M, M, N, T);
    owner.optimization.popDSPFactor();

    mma.setInputMatrices(A, X);
    mmb.setInputMatrices(mma.getOutput(), transpose(owner, A, M, N, T));

    return mmb.getOutput();
  }
}
