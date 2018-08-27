/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradIfmapTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradInverseTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradWeightsTransform;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.custom_computing_ic.maxdeep.utils.AdderTree;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * This kernel implements the conv2d block.
 *
 * INTERFACE: *described in ruby* < <<X(1 ,1), X(1 ,2), ..., X(1 ,K*(K+PK))>,
 * <X(2 ,1), X(2 ,2), ..., X(2 ,K*(K+PK))>, ..., <X(PC,1), X(PC,2), ...,
 * X(PC,K*(K+PK))>>_PF, <<W(1 ,1), W(1 ,2), ..., W(1 ,K*K)>, <W(2 ,1), W(2 ,2),
 * ..., W(2 ,K*K)>, ..., <W(PC,1), W(PC,2), ..., W(PC,K*K)>>_PF > ~ <<Y(1 ,1),
 * Y(1 ,2), ..., Y(1 ,PK)>, <Y(2 ,1), Y(2 ,2), ..., Y(2 ,PK)>, ..., <Y(PF,1),
 * Y(PF,2), ..., Y(PF,PK)>>
 *
 * @author Ruizhe Zhao
 *
 */
public
class Conv2DKernel extends KernelComponent {
 protected
  final ConvLayerParameters cp;

 protected
  final DFEVectorType<DFEVar> ifmapVecT, coeffVecT, ofmapVecT;
 protected
  final DFEVector<DFEVar> ifmap, coeff, ofmap;
 protected
  final DFEType T;

 private
  static final boolean OPTIMISE_WINOGRAD_TRANSFORM = true;

  /**
   * constructor
   *
   * @param owner
   * @param cp
   * @param T
   */
 public Conv2DKernel(KernelBase<?> owner, ConvLayerParameters cp,
      DFEType T) {
    super(owner);

    this.cp = cp;

    int K = cp.K;
    int PC = cp.PC;
    int PF = cp.PF;
    int PK = cp.PK;

    owner.getManager().logMsg(String.format(
        "Building the CORE arithmetic unit for \"%s\" ...", cp.name));
    if (cp.useWinograd) owner.getManager().logMsg("WINOGRAD is applied");

    /**
     * Verify parameters.
     */
    if (cp.useWinograd && cp.PK != 1)
      throw new IllegalArgumentException(
          "P_K should be 1 if WINOGRAD is applied, got: " + cp.PK);
    if (cp.useWinograd && cp.K != WinogradTransform.R)
      throw new IllegalArgumentException(
          String.format("Kernel size (%3d) should equal to R (%3d) in the " +
                            "WINOGRAD transform.",
                        cp.K, WinogradTransform.R));

    /**
     * Setup size of vectors in interface.
     */
    int coeffVecSize = cp.getCoeffVecSize();
    int ifmapVecSize = (cp.useWinograd) ? (WinogradTransform.TILE_SIZE *
                                           WinogradTransform.TILE_SIZE * cp.PC)
                                        : (K * (K + PK - 1) * PC);
    int ofmapVecSize = (cp.useWinograd)
                           ? WinogradTransform.M * WinogradTransform.M * cp.PF
                           : PK * PF;
    owner.getManager().logMsg(
        String.format("CORE ifmap vector size: %d", ifmapVecSize));
    owner.getManager().logMsg(
        String.format("CORE coefficient vector size: %d", coeffVecSize));
    owner.getManager().logMsg(
        String.format("CORE ofmap vector size: %d", ofmapVecSize));

    this.T = T;

    this.ifmapVecT = new DFEVectorType<DFEVar>(T, ifmapVecSize);
    this.coeffVecT = new DFEVectorType<DFEVar>(T, coeffVecSize);
    this.ofmapVecT = new DFEVectorType<DFEVar>(getOfmapScalarT(), ofmapVecSize);

    this.ifmap = ifmapVecT.newInstance(owner);
    this.coeff = coeffVecT.newInstance(owner);
    this.ofmap = ofmapVecT.newInstance(owner);

    /**
     * computation
     */
    if (cp.useWinograd)
      computeWinograd(ifmap, coeff, ofmap);
    else
      compute(ifmap, coeff, ofmap);

    if (cp.dbg) {
      debug.simPrintf("[Conv2DKernel] ifmap = %KObj%\n", ifmap);
      debug.simPrintf("[Conv2DKernel] coeff = %KObj%\n", coeff);
      debug.simPrintf("[Conv2DKernel] ofmap = %KObj%\n", ofmap);
    }
  }

 public
  DFEType getOfmapScalarT() { return T; }

  /**
   * core computation, based on dot-product.
   *
   * @param ifmap
   * @param coeff
   * @param ofmap
   */
 public
  void compute(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff,
               DFEVector<DFEVar> ofmap) {
    int K = cp.K;
    int PC = cp.PC;
    int PF = cp.PF;
    int PK = cp.PK;

    for (int pf = 0; pf < PF; pf++) {
      List<DFEVar> tmpResults = new ArrayList<DFEVar>(PK * PC);
      // TODO: initialise an ArrayList in this way might be silly
      for (int i = 0; i < PC * PK; i++) tmpResults.add(null);

      for (int pc = 0; pc < PC; pc++) {
        // create a new vector instance
        List<DFEVector<DFEVar>> ifmapChunks = getIfmapChunksAt(pc);

        // coefficient chunk
        DFEVector<DFEVar> coeffChunk = getCoeffChunkAt(pf, pc);

        for (int pk = 0; pk < PK; pk++) {
          DFEVector<DFEVar> ifmapChunk = ifmapChunks.get(pk);

          tmpResults.set(pk * PC + pc, dotprod(ifmapChunk, coeffChunk));
        }
      }

      for (int pk = 0; pk < PK; pk++)
        ofmap[pf * PK + pk].connect(
            AdderTree.reduce(tmpResults.subList(pk * PC, (pk + 1) * PC)));
    }
  }

 public
  void computeWinograd(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff,
                       DFEVector<DFEVar> ofmap) {
    getOwner().getManager().logMsg("Using WINOGRAD compute function ...");
    if (cp.winogradWeightsOffline)
      getOwner().getManager().logMsg("Winograd weights are computed offline");

    // transform ifmap and coefficient
    DFEVector<DFEVar> ifmapWino = createWinogradTransformedIfmap(ifmap);
    DFEVector<DFEVar> coeffWino = (cp.winogradWeightsOffline)
                                      ? coeff
                                      : createWinogradTransformedCoeff(coeff);

    if (cp.dbg) {
      debug.simPrintf("[Conv2DKernel] ifmap WINO = %KObj%\n", ifmapWino);
      debug.simPrintf("[Conv2DKernel] coeff WINO = %KObj%\n", coeffWino);
    }

    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    int M = WinogradTransform.M;

    // type of temporary output
    DFEVectorType<DFEVar> TOT =
        new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    // type of output
    DFEVectorType<DFEVar> OT = new DFEVectorType<DFEVar>(T, cp.PF * M * M);

    getOwner().getManager().logMsg(
        String.format("CORE consume %d number of multipliers ...",
                      cp.PC * cp.PF * TILE_SIZE * TILE_SIZE));

    for (int f = 0; f < cp.PF; f++) {
      DFEVector<DFEVar> TO = TOT.newInstance(getOwner());

      for (int c = 0; c < cp.PC; c++) {
        getOwner().getManager().logMsg(
            String.format("CORE setting up element-wise multiply (#%03d) ...",
                          f * cp.PC + c));

        DFEVector<DFEVar> tmp = TOT.newInstance(getOwner());

        if (cp.dbg) {
          debug.simPrintf("[Conv2DKernel] ifmap x coeff = %KObj%\n", tmp);
        }

        for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++) {
          int ifmapIdx = c * TILE_SIZE * TILE_SIZE + j;
          int coeffIdx = (f * cp.PC + c) * TILE_SIZE * TILE_SIZE + j;

          // getOwner().optimization.pushDSPFactor(1.0);
          DFEVar r = ifmapWino[ifmapIdx] * coeffWino[coeffIdx];
          // getOwner().optimization.popDSPFactor();

          tmp[j].connect(r);
        }

        if (c == 0)
          TO.connect(tmp);
        else
          TO += tmp;
      }

      if (cp.dbg) {
        debug.simPrintf("[Conv2DKernel] o = %KObj%\n", TO);
      }

      WinogradInverseTransform transform = new WinogradInverseTransform(
          getOwner(), T, OPTIMISE_WINOGRAD_TRANSFORM, cp.dbg);
      transform.setInputMatrix(TO);
      DFEVector<DFEVar> trans = transform.getOutput();

      for (int j = 0; j < M * M; j++) ofmap[f * M * M + j].connect(trans[j]);
    }
  }

 public
  DFEVector<DFEVar> createWinogradTransformedIfmap(DFEVector<DFEVar> ifmap) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    DFEVectorType<DFEVar> RT =
        new DFEVectorType<DFEVar>(T, cp.PC * TILE_SIZE * TILE_SIZE);

    getOwner().getManager().logMsg(String.format(
        "CORE initialised WINOGRAD transformed ifmap %d x %d x %d", cp.PC,
        TILE_SIZE, TILE_SIZE));
    DFEVector<DFEVar> R = RT.newInstance(getOwner());

    for (int i = 0; i < cp.PC; i++) {
      WinogradIfmapTransform winogradTransform = new WinogradIfmapTransform(
          getOwner(), T, OPTIMISE_WINOGRAD_TRANSFORM, cp.dbg);

      // the original input, split into TILE_SIZE^2 tiles
      DFEVector<DFEVar> input =
          winogradTransform.getInputT().newInstance(getOwner());
      getOwner().getManager().logMsg(
          String.format("CORE initialised WINOGRAD transform input matrix " +
                            "(#%04d) of length %d",
                        i, input.getSize()));

      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        input[j].connect(ifmap[i * TILE_SIZE * TILE_SIZE + j]);
      winogradTransform.setInput(input);

      // transformed results
      DFEVector<DFEVar> trans = winogradTransform.getOutput();
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        R[i * TILE_SIZE * TILE_SIZE + j].connect(trans[j]);
    }

    return R;
  }

 public
  DFEVector<DFEVar> createWinogradTransformedCoeff(DFEVector<DFEVar> coeff) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;

    DFEVectorType<DFEVar> RT =
        new DFEVectorType<DFEVar>(T, cp.PF * cp.PC * TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> R = RT.newInstance(getOwner());
    getOwner().getManager().logMsg(String.format(
        "CORE initialised WINOGRAD transformed coeff %d x %d x %d x %d", cp.PF,
        cp.PC, TILE_SIZE, TILE_SIZE));

    for (int i = 0; i < cp.PF * cp.PC; i++) {
      WinogradWeightsTransform winogradTransform = new WinogradWeightsTransform(
          getOwner(), T, OPTIMISE_WINOGRAD_TRANSFORM);

      DFEVector<DFEVar> input =
          winogradTransform.getInputT().newInstance(getOwner());
      getOwner().getManager().logMsg(
          String.format("CORE initialised WINOGRAD transform input matrix " +
                            "(#%04d) of length %d",
                        i, input.getSize()));
      for (int j = 0; j < cp.K * cp.K; j++)
        input[j].connect(coeff[i * cp.K * cp.K + j]);

      winogradTransform.setInput(input);
      DFEVector<DFEVar> trans = winogradTransform.getOutput();
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        R[i * TILE_SIZE * TILE_SIZE + j].connect(trans[j]);
    }

    return R;
  }

 public
  DFEVar dotprod(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff) {
    DotProductKernel dp =
        new DotProductKernel(this.getOwner(), cp.K * cp.K, T, cp.dbg);
    dp.setInputs(ifmap, coeff);

    return dp.getOutput();
  }

 public
  DFEVectorType<DFEVar> getIfmapT() { return ifmapVecT; }

 public
  DFEVectorType<DFEVar> getCoeffT() { return coeffVecT; }

 public
  DFEVectorType<DFEVar> getOfmapT() { return ofmapVecT; }

 public
  void setInputs(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff) {
    this.ifmap.connect(ifmap);
    this.coeff.connect(coeff);
  }

 public
  DFEVector<DFEVar> getOfmap() { return ofmap; }

  /**
   * Get ifmap data chunk at (pc)
   *
   * TODO: Each time you call this function, a new vector will be created,
   * might need caching in the future.
   *
   * @param pc
   * @return
   */
 private
  List<DFEVector<DFEVar>> getIfmapChunksAt(int pc) {
    int K = cp.K;
    int PC = cp.PC;
    int PK = cp.PK;

    if (pc >= PC)
      throw new IllegalArgumentException("pc should be smaller than PC");

    int ifmapPackedChunkSize = K * (K + PK - 1);

    DFEVectorType<DFEVar> ifmapPackedChunkT =
        new DFEVectorType<DFEVar>(T, ifmapPackedChunkSize);

    DFEVector<DFEVar> ifmapPackedChunk =
        ifmapPackedChunkT.newInstance(getOwner());

    // create packed chunk
    for (int i = 0; i < ifmapPackedChunkSize; i++)
      ifmapPackedChunk[i].connect(ifmap[pc * ifmapPackedChunkSize + i]);

    // unpack
    return unpackIfmapChunk(ifmapPackedChunk, cp);
  }

 private
  DFEVector<DFEVar> getCoeffChunkAt(int pf, int pc) {
    int K = cp.K;
    int PC = cp.PC;

    int coeffChunkSize = K * K;

    DFEVectorType<DFEVar> coeffChunkT =
        new DFEVectorType<DFEVar>(T, coeffChunkSize);
    DFEVector<DFEVar> coeffChunk = coeffChunkT.newInstance(getOwner());

    for (int i = 0; i < coeffChunkSize; i++)
      coeffChunk[i].connect(
          coeff[pf * PC * coeffChunkSize + pc * coeffChunkSize + i]);

    return coeffChunk;
  }

  /**
   * Convert vector from (K, K + PK -1) to (PK, K, K)
   *
   * @param src
   *          source vector
   * @param cp
   *
   * @author Ruizhe Zhao
   */
 private
  List<DFEVector<DFEVar>> unpackIfmapChunk(DFEVector<DFEVar> src,
                                           ConvLayerParameters cp) {
    int K = cp.K;
    int PK = cp.PK;

    int ifmapChunkSize = K * K;
    DFEVectorType<DFEVar> ifmapChunkT =
        new DFEVectorType<DFEVar>(T, ifmapChunkSize);
    List<DFEVector<DFEVar>> ifmapChunks = new ArrayList<DFEVector<DFEVar>>(PK);

    for (int p = 0; p < PK; p++) {
      ifmapChunks.add(p, ifmapChunkT.newInstance(getOwner()));

      for (int kx = 0; kx < K; kx++) {
        for (int ky = 0; ky < K; ky++) {
          int srcIdx = kx * (K + PK - 1) + (ky + p);
          int dstIdx = kx * K + ky;
          ifmapChunks.get(p)[dstIdx].connect(src[srcIdx]);
        }
      }
    }

    return ifmapChunks;
  }
}
