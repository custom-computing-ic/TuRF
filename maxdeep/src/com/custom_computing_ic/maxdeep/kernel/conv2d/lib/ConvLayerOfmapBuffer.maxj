package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Stream.OffsetExpr;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;

/**
 * Buffer for output feature map stream.
 * 
 * HOW TO USE:
 * 
 * <pre>
 * {@link ConvLayerOfmapBuffer} obuf =
 *   new {@link ConvLayerOfmapBuffer}(getKernel(), cp, T);
 * data <== obuf.port(addr, dataIn, writeEn);
 * </pre>
 * 
 * @author Ruizhe Zhao
 * @since 03/08/2017
 * 
 */
public class ConvLayerOfmapBuffer extends ConvLayerBaseFmapBuffer {
  private final ConvLayerParameters cp;

  private final DFEType T;
  private final DFEVectorType<DFEVar> portVecT;
  private final DFEType addrT;

  private final Memory<DFEVector<DFEVar>> mem;
  private final DFEVector<DFEVar> port;
  private final DFEVar addr;
  private final DFEVector<DFEVar> data;
  private final DFEVar writeEn;
  private final DFEVector<DFEVar> result;
  private final DFEVar reset;

  private final int MAX_WRITE_LATENCY = 10;

  public ConvLayerOfmapBuffer(KernelBase<?> owner, ConvLayerParameters params, DFEType T) {
    this(owner, params, T, "");
  }

  /**
   * 
   * @param owner
   * @param params
   * @param T
   */
  public ConvLayerOfmapBuffer(KernelBase<?> owner, ConvLayerParameters params, DFEType T,
      String prefix) {
    super(owner);

    this.cp = params;
    this.T = T;

    int width = getWidth();
    int depth = getDepth();

    owner.getManager().logMsg("[ConvLayerOfmapBuffer] depth = %5d addr_bits = %5d\n", depth,
        MathUtils.bitsToAddress(depth));

    this.addrT = dfeUInt(MathUtils.bitsToAddress(depth));
    this.portVecT = new DFEVectorType<DFEVar>(T, width);

    this.mem = owner.mem.alloc(portVecT, depth);
    this.addr = addrT.newInstance(owner);
    this.data = portVecT.newInstance(owner);
    this.writeEn = dfeBool().newInstance(owner);
    this.reset = dfeBool().newInstance(owner);
    this.port = mem.read(addr);

    // mem.port(addr, data, writeEn, RamWriteMode.WRITE_FIRST);

    this.result = portVecT.newInstance(owner);
    for (int i = 0; i < portVecT.getSize(); i++) {
      DFEVar prevData = (reset.eq(true)) ? constant.var(0).cast(T) : port[i];
      this.result[i].connect(data[i] + prevData);
    }

    /* TODO: we assume the read result of an unwritten address is 0 */
    OffsetExpr writeLatency = stream.makeOffsetAutoLoop(prefix + "_WRITE_LATENCY");
    DFEVar memWriteAddr = stream.offset(addr, -writeLatency);
    DFEVector<DFEVar> memWriteValue = stream.offset(result, -writeLatency);
    DFEVar memWriteEn = stream.offset(writeEn, -writeLatency);

    mem.write(memWriteAddr, memWriteValue, memWriteEn);

    if (cp.dbg) {
      debug.simPrintf("[ConvLayerOfmapBuffer] input          = %KObj%\n", data);
      debug.simPrintf("[ConvLayerOfmapBuffer] read           = %KObj%\n", port);
      debug.simPrintf("[ConvLayerOfmapBuffer] output         = %KObj%\n", result);
      debug.simPrintf("[ConvLayerOfmapBuffer] writeEn        = %KObj%\n", writeEn);
      debug.simPrintf("[ConvLayerOfmapBuffer] addr           = %KObj%\n", addr);
      debug.simPrintf("[ConvLayerOfmapBuffer] actual output  = %KObj%\n", memWriteValue);
      debug.simPrintf("[ConvLayerOfmapBuffer] actual writeEn = %KObj%\n", memWriteEn);
      debug.simPrintf("[ConvLayerOfmapBuffer] actual addr    = %KObj%\n", memWriteAddr);
      debug.simPrintf("[ConvLayerOfmapBuffer] latency        = %KObj%\n",
          writeLatency.getDFEVar(getOwner(), dfeUInt(8)));
    }
  }

  public DFEType getAddrT() {
    return addrT;
  }

  public DFEVector<DFEVar> port(DFEVector<DFEVar> data, DFEVar addr, DFEVar writeEn) {
    this.data.connect(data);
    this.addr.connect(addr);
    this.writeEn.connect(writeEn);
    return this.result;
  }

  public void setReset(DFEVar reset) {
    this.reset.connect(reset);
  }

  public DFEVectorType<DFEVar> getPortVecT() {
    return portVecT;
  }

  public int getWidth() {
    if (cp.type == Type.POINTWISE) {
      return cp.PF * cp.PH * cp.PW;
    } else {
      if (cp.useWinograd) {
        return cp.PF * WinogradTransform.M * WinogradTransform.M;
      } else if (cp.seq == CompSeq.CHANNEL_MAJOR || cp.seq == CompSeq.FILTER_MAJOR) {
        return cp.PK * cp.PF;
      } else {
        throw new IllegalArgumentException(String.format(
            "Computation sequence %s has not been supported yet", cp.seq));
      }
    }
  }

  public int getDepth() {
    if (cp.type == Type.POINTWISE) {
      return cp.PF * cp.OH * cp.OW / getWidth();
    } else {
      if (cp.seq == CompSeq.CHANNEL_MAJOR) {
        return cp.F * cp.OH * cp.OW / getWidth();
      } else if (cp.seq == CompSeq.FILTER_MAJOR) {
        return cp.PF * cp.OH * cp.OW / getWidth();
      } else {
        throw new IllegalArgumentException(String.format(
            "Computation sequence %s has not been supported yet", cp.seq));
      }
    }
  }
}
