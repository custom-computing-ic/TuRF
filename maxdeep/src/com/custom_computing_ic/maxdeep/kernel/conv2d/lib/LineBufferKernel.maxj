package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.maxeler.maxcompiler.v2.errors.MaxCompilerAPIError;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * This is a kernel for integrating line buffer with your design.
 * 
 * It allows you to write in parallel.
 * 
 * +--------- maxWidth (width) -------------+ |<-- numPipes --> | | | maxHeight | | | | |
 * +----------------------------------------+
 * 
 * @author Ruizhe Zhao
 * @since 24/04/2017
 */
public class LineBufferKernel extends KernelComponent {

  private final int maxHeight;
  private final int maxWidth;
  private final int maxWidthFolded;
  private final int numPipes;

  private final DFEType scalarType;
  private final DFEVectorType<DFEVar> valueType;
  private final DFEType indexType;

  private DFEVar height;
  private final DFEVar width;
  private final DFEVar widthFolded;
  private final DFEVar capacity;
  private DFEVar capacityFolded;

  private final DFEVector<DFEVar> buf;
  private DFEVar vld;
  private DFEVar idx;

  private static final int DEFAULT_INDEX_COUNTER_BITS = 64;

  /**
   * Constructor.
   * 
   * Folded line buffer: enable multiple elements input and output in one cycle.
   * 
   * Usage:
   * 
   * <pre>
   * LineBufferKernelLib lbuf = new LineBufferKernelLib(getKernel(), K, W, P, T);
   * lbuf.setHeight(K);
   * lbuf.setWidth(W);
   * lbuf.setCapacity(H * W);
   * lbuf.setInput(dataIn);
   * dataOut <== lbuf.getOutput();
   * </pre>
   * 
   * @param owner The parent kernel of this kernel.
   * @param maxHeight maximum height of the line buffer.
   * @param maxWidth maximum width of the line buffer.
   * @param numPipes number of parallel write pipes.
   * @param type data type
   * 
   * @author Ruizhe Zhao
   * @since 24/04/2017
   */
  public LineBufferKernel(KernelBase<?> owner, int maxHeight, int maxWidth, int numPipes,
      DFEType type, boolean hasValid) {
    super(owner);

    if (maxHeight <= 0)
      throw new MaxCompilerAPIError("maxHeight (%d) should be larger than 0", maxHeight);
    if (maxWidth <= 0)
      throw new MaxCompilerAPIError("maxWidth (%d) should be larger than 0", maxWidth);
    if (numPipes <= 0)
      throw new MaxCompilerAPIError("numPipes (%d) should be larger than 0", numPipes);
    if (maxWidth % numPipes != 0)
      throw new MaxCompilerAPIError("maxWidth (%d) should be a integer multiple of numPipes (%d)",
          maxWidth, numPipes);

    this.numPipes = numPipes;
    this.maxHeight = maxHeight;
    this.maxWidth = maxWidth;
    this.maxWidthFolded = maxWidth / numPipes;

    scalarType = type;
    valueType = new DFEVectorType<DFEVar>(type, numPipes);
    indexType = dfeInt(32);

    width = indexType.newInstance(getOwner());
    capacity = indexType.newInstance(getOwner());
    widthFolded = width / numPipes;

    buf = valueType.newInstance(getOwner());

    if (hasValid) {
      height = indexType.newInstance(getOwner());
      capacityFolded = capacity / numPipes;
      DFEVar capFolded = capacityFolded.cast(dfeUInt(DEFAULT_INDEX_COUNTER_BITS));
      idx = control.count.simpleCounter(DEFAULT_INDEX_COUNTER_BITS, capFolded).cast(indexType);
      vld = idx >= (widthFolded * (height - 1));
    }
  }

  public LineBufferKernel(KernelBase<?> owner, int maxHeight, int maxWidth, int numPipes,
      DFEType type) {
    this(owner, maxHeight, maxWidth, numPipes, type, true);
  }

  public DFEVectorType<DFEVar> getInputVecT() {
    return valueType;
  }

  public void setInput(DFEVector<DFEVar> inp) {
    if (inp.getSize() != this.buf.getSize())
      throw new MaxCompilerAPIError("inp should have length [%d] equals to buf's length [%d]",
          inp.getSize(), this.buf.getSize());

    buf.connect(inp);
  }

  public void setHeight(DFEVar height) {
    this.height.connect(height);
  }

  public void setWidth(DFEVar width) {
    this.width.connect(width);
  }

  /**
   * Capacity should not be smaller than height * width
   */
  public void setCapacity(DFEVar capacity) {
    this.capacity.connect(capacity);
  }

  public DFEVar getValid() {
    return vld;
  }

  public DFEType getIndexT() {
    return indexType;
  }

  public DFEVector<DFEVar> getOutput() {
    DFEVectorType<DFEVar> outVecType = new DFEVectorType<DFEVar>(scalarType, maxHeight * numPipes);
    DFEVector<DFEVar> outVec = outVecType.newInstance(getOwner());

    for (int j = 0; j < maxHeight; j++) {
      DFEVector<DFEVar> outVecRead = getOutputVectorAt(maxHeight - j - 1);
      for (int i = 0; i < numPipes; i++) {
        outVec[i * maxHeight + j].connect(outVecRead[i]);
      }
    }

    return outVec;
  }

  private DFEVector<DFEVar> getOutputVectorAt(int i) {
    if (i == 0)
      return buf;

    DFEVar index = constant.var(i).cast(indexType);
    DFEVar offset = index * (-widthFolded);
    int maxOffset = -maxWidthFolded * maxHeight;
    DFEVector<DFEVar> readBufByOffset = stream.offset(buf, offset, maxOffset, 0);
    return readBufByOffset;
  }
}
