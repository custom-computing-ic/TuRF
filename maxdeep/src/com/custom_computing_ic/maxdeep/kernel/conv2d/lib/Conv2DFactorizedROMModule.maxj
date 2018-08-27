package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.utils.ArbitraryLengthPortROM;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;

/**
 * This component uses ROM to provide coefficients for the factorized module.
 * Each coefficient input stream of the core module will be connected to a ROM.
 *
 * @author Ruizhe Zhao
 *
 */
public class Conv2DFactorizedROMModule extends KernelComponent {

  private final DFEType                         scalarT;
  private final Conv2DFactorizedModule          core;
  private final Conv2DFactorizedModuleParameter conv2dParams;
  private final DFEVector<DFEVar>               ifmap;
  private final DFEVectorType<DFEVar>           depthVecT, pointVecT;
  private final ArbitraryLengthPortROM          depthROM, pointROM;

  public Conv2DFactorizedROMModule(KernelBase<?> owner,
      Conv2DFactorizedModuleParameter conv2dParams, DFEType scalarT) {
    super(owner);

    if (conv2dParams.getIfmapNumChnl()/conv2dParams.getIfmapNumParaChnl() < 2)
      throw new IllegalArgumentException(
          "depth coeff ROM should have depth larger than 1");

    this.conv2dParams = conv2dParams;
    this.scalarT = scalarT;

    this.core = new Conv2DFactorizedModule(owner, conv2dParams, scalarT);
    this.ifmap = core.getIfmapVecT().newInstance(owner);
    core.setIfmap(ifmap);

    this.depthVecT = core.getDepthCoeffVecT();
    this.pointVecT = core.getPointCoeffVecT();

    // initialize ROM
    this.depthROM = new ArbitraryLengthPortROM(getOwner(),
        this.depthVecT.getSize(), getDepthROMNumElems(), scalarT);
    this.pointROM = new ArbitraryLengthPortROM(getOwner(),
        this.pointVecT.getSize(), getPointROMNumElems(), scalarT);
    // setup ROM read for the core module
    core.setDepthCoeff(depthROM.read(getDepthROMAddr()));
    core.setPointCoeff(pointROM.read(getPointROMAddr()));
  }

  public DFEVectorType<DFEVar> getIfmapVecT() { return core.getIfmapVecT(); }

  public DFEVectorType<DFEVar> getOfmapVecT() { return core.getOfmapVecT(); }

  public void setDepthROMMapped(String name) {
    this.depthROM.mapToCPU(name);
  }

  public void setPointROMMapped(String name) {
    this.pointROM.mapToCPU(name);
  }

  public int getDepthROMPortWidth(int index) {
    return this.depthROM.getROMPortWitdh(index);
  }

  public int getNumDepthROMs() {
    return this.depthROM.getNumROMs();
  }

  public int getNumPointROMs() {
    return this.pointROM.getNumROMs();
  }

  public int getPointROMPortWidth(int index) {
    return this.pointROM.getROMPortWitdh(index);
  }


  public void setIfmap(DFEVector<DFEVar> ifmap) { this.ifmap <== ifmap; }

  public DFEVector<DFEVar> getOfmap() {
    return core.getOfmap();
  }

  public DFEVar getOfmapEnable() {
    return core.getOfmapValid();
  }

  public DFEVar getIfmapEnable() {
    return core.getIfmapEnable();
  }

  private int getDepthROMNumElems() {
    return conv2dParams.getIfmapNumChnl() / conv2dParams.getIfmapNumParaChnl();
  }

  private int getPointROMNumElems() {
    return (conv2dParams.getIfmapNumChnl() * conv2dParams.getOfmapNumChnl())
        / (conv2dParams.getIfmapNumParaChnl() * conv2dParams
            .getOfmapNumParaChnl());
  }

  private DFEVar getDepthROMAddr() {
    DFEType addrT = dfeUInt(MathUtils.bitsToAddress(getDepthROMNumElems()));
    DFEVar addr = core.getCurrIfmapChnl() / conv2dParams.getIfmapNumParaChnl();
    return addr.cast(addrT);
  }

  private DFEVar getPointROMAddr() {
    DFEType addrT = dfeUInt(MathUtils.bitsToAddress(getPointROMNumElems()));
    DFEVar f = core.getCurrOfmapChnl() / conv2dParams.getOfmapNumParaChnl();
    DFEVar c = core.getCurrIfmapChnl() / conv2dParams.getIfmapNumParaChnl();
    f = f.cast(dfeInt(32));
    c = c.cast(dfeInt(32));
    DFEVar C = constant
      .var(conv2dParams.getIfmapNumChnl() / conv2dParams.getIfmapNumParaChnl())
      .cast(dfeInt(32));
    DFEVar addr = (f * C) + c;
    // debug.simPrintf("f = %d c = %d addr = %d %d\n", f, c, addr, C);
    return addr.cast(addrT);
  }
}
