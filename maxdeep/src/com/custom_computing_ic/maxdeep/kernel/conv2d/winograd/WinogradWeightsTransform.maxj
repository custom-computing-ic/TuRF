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
class WinogradWeightsTransform extends KernelComponent {
 private
  final float[][] G = {{0.25f, 0.0f, 0.0f},
                       {-1.f / 6, -1.f / 6, -1.f / 6},
                       {-1.f / 6, 1.f / 6, -1.f / 6},
                       {1.f / 24, 1.f / 12, 1.f / 6},
                       {1.f / 24, -1.f / 12, 1.f / 6},
                       {0.0f, 0.0f, 1.0f}};

 private
  final DFEType T;
 private
  final DFEVector<DFEVar> g, GgG;
 private
  final DFEVectorType<DFEVar> gT;

 public WinogradWeightsTransform(KernelBase<?> owner, DFEType T, boolean optimise) {
    super(owner);

    this.T = T;

    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    int R = WinogradTransform.R;
    gT = new DFEVectorType<DFEVar>(T, R * R);
    g = gT.newInstance(owner);

    if (!optimise) {
      DFEVector<DFEVar> GG =
          WinogradTransform.convertToMatrix(owner, G, TILE_SIZE, R, T, false);
      GgG = WinogradTransform.transform(owner, g, GG, TILE_SIZE, R, T);
    } else {
      owner.optimization.pushDSPFactor(0.0);
      DFEVector<DFEVar> Gg = getGg(owner, g, T, TILE_SIZE, R);
      GgG = getGgG(owner, Gg, T, TILE_SIZE, R);
      owner.optimization.popDSPFactor();
    }
  }

 public
  DFEVar shiftRightSix(DFEVar x) {
    // DFEVar oneSix = constant.var(1. / 6).cast(T);
    // DFEVar ans = x * oneSix;

    DFEVar ans = (x >> 3) + (x >> 5) + (x >> 7) + (x >> 9);

    return ans;
  }

 public DFEVector<DFEVar> getGg(KernelBase<?> owner, DFEVector<DFEVar> g, DFEType T,
      int TILE_SIZE, int R) {
    DFEVectorType<DFEVar> GgT = new DFEVectorType<DFEVar>(T, TILE_SIZE * R);
    DFEVector<DFEVar> Gg = GgT.newInstance(owner);

    Gg[0].connect(g[0] >> 2);
    Gg[1].connect(g[1] >> 2);
    Gg[2].connect(g[2] >> 2);

    Gg[3].connect(-shiftRightSix(g[0] + g[3] + g[6]));
    Gg[4].connect(-shiftRightSix(g[1] + g[4] + g[7]));
    Gg[5].connect(-shiftRightSix(g[2] + g[5] + g[8]));

    Gg[6].connect(shiftRightSix(-g[0] + g[3] - g[6]));
    Gg[7].connect(shiftRightSix(-g[1] + g[4] - g[7]));
    Gg[8].connect(shiftRightSix(-g[2] + g[5] - g[8]));

    Gg[9].connect(shiftRightSix((g[0] >> 2) + (g[3] >> 1) + g[6]));
    Gg[10].connect(shiftRightSix((g[1] >> 2) + (g[4] >> 1) + g[7]));
    Gg[11].connect(shiftRightSix((g[2] >> 2) + (g[5] >> 1) + g[8]));

    Gg[12].connect(shiftRightSix((g[0] >> 2) - (g[3] >> 1) + g[6]));
    Gg[13].connect(shiftRightSix((g[1] >> 2) - (g[4] >> 1) + g[7]));
    Gg[14].connect(shiftRightSix((g[2] >> 2) - (g[5] >> 1) + g[8]));

    Gg[15].connect(g[6]);
    Gg[16].connect(g[7]);
    Gg[17].connect(g[8]);

    return Gg;
  }

 public DFEVector<DFEVar> getGgG(KernelBase<?> owner, DFEVector<DFEVar> Gg, DFEType T,
      int TILE_SIZE, int R) {
    DFEVectorType<DFEVar> GgGT =
        new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> GgG = GgGT.newInstance(owner);

    for (int i = 0; i < TILE_SIZE; i++) {
      int j = i * TILE_SIZE;
      int k = i * R;

      GgG[j].connect(Gg[k] >> 2);
      GgG[j + 1].connect(-shiftRightSix(Gg[k] + Gg[k + 1] + Gg[k + 2]));
      GgG[j + 2].connect(shiftRightSix(-Gg[k] + Gg[k + 1] - Gg[k + 2]));
      GgG[j + 3].connect(
          shiftRightSix((Gg[k] >> 2) + (Gg[k + 1] >> 1) + Gg[k + 2]));
      GgG[j + 4].connect(
          shiftRightSix((Gg[k] >> 2) - (Gg[k + 1] >> 1) + Gg[k + 2]));
      GgG[j + 5].connect(Gg[k + 2]);
    }

    return GgG;
  }

 public
  DFEVectorType<DFEVar> getInputT() { return gT; }

 public
  void setInput(DFEVector<DFEVar> mat) { g.connect(mat); }

 public
  DFEVector<DFEVar> getOutput() { return GgG; }
}
