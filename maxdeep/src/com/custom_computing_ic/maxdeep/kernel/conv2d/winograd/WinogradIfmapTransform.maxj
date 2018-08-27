package com.custom_computing_ic.maxdeep.kernel.conv2d.winograd;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * Perform Winograd transformation before the dot-product
 *
 * @author rz3515
 *
 */
public
class WinogradIfmapTransform extends KernelComponent {
 private
  final float[][] B = {{4, 0, 0, 0, 0, 0},      {0, -4, 4, -2, 2, 4},
                       {-5, -4, -4, -1, -1, 0}, {0, 1, -1, 2, -2, -5},
                       {1, 1, 1, 1, 1, 0},      {0, 0, 0, 0, 0, 1}};
 private
  final DFEVectorType<DFEVar> dT;
 private
  final DFEVector<DFEVar> d, BdB;
 private
  boolean dbg;

 public WinogradIfmapTransform(KernelBase<?> owner, DFEType T, boolean opt) {
    this(owner, T, opt, false);
  }
 public WinogradIfmapTransform(KernelBase<?> owner, DFEType T, boolean opt, boolean dbg) {
    super(owner);

    this.dbg = dbg;
    int TILE_SIZE = WinogradTransform.TILE_SIZE;

    dT = new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    d = dT.newInstance(owner);

    if (!opt) {
      DFEVector<DFEVar> BB = WinogradTransform.convertToMatrix(
          owner, B, TILE_SIZE, TILE_SIZE, T, true);
      BdB = WinogradTransform.transform(owner, d, BB, TILE_SIZE, TILE_SIZE, T);
    } else {
      BdB = getBdB(owner, T, getBd(owner, T, TILE_SIZE), TILE_SIZE);
    }
  }

 public DFEVector<DFEVar> getBd(KernelBase<?> owner, DFEType T, int TILE_SIZE) {
    DFEVectorType<DFEVar> BdT =
        new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> Bd = BdT.newInstance(owner);

    for (int i = 0; i < TILE_SIZE; i++) {
      DFEVar[] dd = new DFEVar[TILE_SIZE];
      for (int j = 0; j < TILE_SIZE; j++) dd[j] = d[j * TILE_SIZE + i];

      Bd[i].connect((dd[0] << 2) - ((dd[2] << 2) + dd[2]) + dd[4]);
      Bd[i + TILE_SIZE].connect(-((dd[1] + dd[2]) << 2) + dd[3] + dd[4]);
      Bd[i + 2 * TILE_SIZE].connect(((dd[1] - dd[2]) << 2) - dd[3] + dd[4]);
      Bd[i + 3 * TILE_SIZE].connect(-(dd[1] << 1) - dd[2] + (dd[3] << 1) +
                                    dd[4]);
      Bd[i + 4 * TILE_SIZE].connect((dd[1] << 1) - dd[2] - (dd[3] << 1) +
                                    dd[4]);
      Bd[i + 5 * TILE_SIZE].connect((dd[1] << 2) - ((dd[3] << 2) + dd[3]) +
                                    dd[5]);
    }

    return Bd;
  }

 public DFEVector<DFEVar> getBdB(KernelBase<?> owner, DFEType T, DFEVector<DFEVar> Bd,
      int TILE_SIZE) {
    DFEVectorType<DFEVar> BdBT =
        new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> BdB = BdBT.newInstance(owner);

    if (dbg) {
      debug.simPrintf("[Winograd] Bd = %KObj%\n", Bd);
    }

    for (int i = 0; i < TILE_SIZE; i++) {
      DFEVar[] dd = new DFEVar[TILE_SIZE];
      for (int j = 0; j < TILE_SIZE; j++) dd[j] = Bd[i * TILE_SIZE + j];

      BdB[i * TILE_SIZE].connect((dd[0] << 2) - ((dd[2] << 2) + dd[2]) + dd[4]);
      BdB[i * TILE_SIZE + 1].connect(-((dd[1] + dd[2]) << 2) + dd[3] + dd[4]);
      BdB[i * TILE_SIZE + 2].connect(((dd[1] - dd[2]) << 2) - dd[3] + dd[4]);
      BdB[i * TILE_SIZE + 3].connect(-(dd[1] << 1) - dd[2] + (dd[3] << 1) +
                                     dd[4]);
      BdB[i * TILE_SIZE + 4].connect((dd[1] << 1) - dd[2] - (dd[3] << 1) +
                                     dd[4]);
      BdB[i * TILE_SIZE + 5].connect((dd[1] << 2) - ((dd[3] << 2) + dd[3]) +
                                     dd[5]);
    }

    return BdB;
  }

 public
  DFEVectorType<DFEVar> getInputT() { return dT; }

 public
  void setInput(DFEVector<DFEVar> mat) { d.connect(mat); }

 public
  DFEVector<DFEVar> getOutput() { return BdB; }
}
