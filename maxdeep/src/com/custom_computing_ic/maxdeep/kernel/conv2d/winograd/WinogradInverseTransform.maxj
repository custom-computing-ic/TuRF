package com.custom_computing_ic.maxdeep.kernel.conv2d.winograd;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public
class WinogradInverseTransform extends KernelComponent {
 private
  final float[][] A = {{1, 0, 0, 0}, {1, 1, 1, 1},   {1, -1, 1, -1},
                       {1, 2, 4, 8}, {1, -2, 4, -8}, {0, 0, 0, 1}};

 private
  final DFEVector<DFEVar> X, Y;
 private
  boolean dbg;

 public WinogradInverseTransform(KernelBase<?> owner, DFEType T, boolean opt) {
    this(owner, T, opt, false);
  }
 public WinogradInverseTransform(KernelBase<?> owner, DFEType T, boolean opt, boolean dbg) {
    super(owner);

    this.dbg = dbg;

    int M = WinogradTransform.M;
    int R = WinogradTransform.R;
    int TILE_SIZE = WinogradTransform.TILE_SIZE;

    DFEVectorType<DFEVar> XT =
        new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    X = XT.newInstance(owner);

    if (!opt) {
      DFEVector<DFEVar> AA =
          WinogradTransform.convertToMatrix(owner, A, TILE_SIZE, M, T, true);
      Y = WinogradTransform.transform(owner, X, AA, M, TILE_SIZE, T);
    } else {
      Y = getAXA(owner, T, getAX(owner, T, TILE_SIZE, M), TILE_SIZE, M);
    }
  }

 public DFEVector<DFEVar> getAX(KernelBase<?> owner, DFEType T, int TILE_SIZE, int M) {
    DFEVectorType<DFEVar> AXT = new DFEVectorType<DFEVar>(T, M * TILE_SIZE);
    DFEVector<DFEVar> AX = AXT.newInstance(owner);

    for (int j = 0; j < TILE_SIZE; j++) {
      DFEVar[] x = new DFEVar[TILE_SIZE];
      for (int k = 0; k < TILE_SIZE; k++) x[k] = X[k * TILE_SIZE + j];

      AX[j].connect(x[0] + x[1] + x[2] + x[3] + x[4]);
      AX[j + TILE_SIZE].connect(x[1] - x[2] + (x[3] << 1) - (x[4] << 1));
      AX[j + 2 * TILE_SIZE].connect(x[1] + x[2] + (x[3] << 2) + (x[4] << 2));
      AX[j + 3 * TILE_SIZE].connect(x[1] - x[2] + (x[3] << 3) - (x[4] << 3) +
                                    x[5]);
    }

    return AX;
  }

 public DFEVector<DFEVar> getAXA(KernelBase<?> owner, DFEType T, DFEVector<DFEVar> AX,
      int TILE_SIZE, int M) {
    DFEVectorType<DFEVar> AXAT = new DFEVectorType<DFEVar>(T, M * M);
    DFEVector<DFEVar> AXA = AXAT.newInstance(owner);

    if (dbg) {
      debug.simPrintf("[Winograd] ATo = %KObj%\n", AX);
    }

    for (int j = 0; j < M; j++) {
      DFEVar[] x = new DFEVar[TILE_SIZE];
      for (int k = 0; k < TILE_SIZE; k++) x[k] = AX[j * TILE_SIZE + k];

      AXA[j * M].connect(x[0] + x[1] + x[2] + x[3] + x[4]);
      AXA[j * M + 1].connect(x[1] - x[2] + (x[3] << 1) - (x[4] << 1));
      AXA[j * M + 2].connect(x[1] + x[2] + (x[3] << 2) + (x[4] << 2));
      AXA[j * M + 3].connect(x[1] - x[2] + (x[3] << 3) - (x[4] << 3) + x[5]);
    }

    return AXA;
  }

 public
  void setInputMatrix(DFEVector<DFEVar> m) { X.connect(m); }

 public
  DFEVector<DFEVar> getOutput() { return Y; }
}
