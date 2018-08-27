/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * @author Ruizhe Zhao
 *
 */
public
class ConvLayerLineBuffer extends KernelComponent {
 private
  final ConvLayerParameters cp;

  @SuppressWarnings("unused") private final DFEType T;
 private
  final DFEVectorType<DFEVar> inputVecT;
 private
  final DFEVectorType<DFEVar> outputVecT;

 private
  final DFEVector<DFEVar> inputVec;
 private
  final DFEVector<DFEVar> outputVec;

 private
  final int lineBufferHeight;
 private
  final int lineBufferWidth;
 private
  final int lineBufferNumPipes;

 public
  static final int WINO_LBUF_HEIGHT = 2;
 public
  static final int WINO_LBUF_TILE_SIZE = 4;
 public
  static final int WINO_LBUF_NUM_PIPES =
      WINO_LBUF_TILE_SIZE * WINO_LBUF_TILE_SIZE;
 public
  static final int WINO_LBUF_PADDING_WIDTH = 2;

  /**
   * constructor
   *
   * @param owner
   * @param params
   * @param T
   */
 public ConvLayerLineBuffer(KernelBase<?> owner, ConvLayerParameters params, DFEType T) {
    super(owner);

    this.cp = params;
    this.T = T;
    // initialise the height of line buffer
    this.lineBufferHeight = getLineBufferHeight(cp);
    this.lineBufferWidth = getLineBufferWidth(cp);
    this.lineBufferNumPipes = getLineBufferNumPipes(cp);

    owner.getManager().logMsg(
        String.format("Building line buffer for \"%s\" ...", params.name));
    owner.getManager().logMsg(String.format(
        "Line buffer shape %d x %d, produces %d number of %d x %d tiles " +
            "per cycle",
        lineBufferHeight, cp.W, cp.PK, lineBufferHeight, lineBufferHeight));
    owner.getManager().logMsg(String.format(
        "Line buffer input vector size: %d, output vector size: %d.",
        getInputVecSize(), getOutputVecSize()));

    /* input stream */
    this.inputVecT = new DFEVectorType<DFEVar>(T, getInputVecSize());
    this.inputVec = inputVecT.newInstance(getOwner());

    /* output stream */
    this.outputVecT = new DFEVectorType<DFEVar>(T, getOutputVecSize());
    this.outputVec = outputVecT.newInstance(getOwner());

    /* initialise line buffers */
    int numLineBuffers = getNumLineBuffers();
    owner.getManager().logMsg(
        String.format("Number of separated line buffers: %d", numLineBuffers));

    for (int i = 0; i < numLineBuffers; i++) {
      /* initialise one line buffer */
      owner.getManager().logMsg(
          String.format("Initialising line buffer kernel with %d x %d x %d",
                        lineBufferHeight, lineBufferWidth, lineBufferNumPipes));

      LineBufferKernel lbuf =
          new LineBufferKernel(getOwner(), lineBufferHeight, lineBufferWidth,
                               lineBufferNumPipes, T, false);

      lbuf.setWidth(constant.var(lineBufferWidth).cast(lbuf.getIndexT()));
      lbuf.setInput(createLineBufferInputVector(i));
      // get results from line buffers, lbufOut has shape PK * K */
      DFEVector<DFEVar> lbufOutput = lbuf.getOutput();

      // the output tile size of the line buffer kernel;
      int tileSize = getLineBufferTileSize();

      // here chunk means the number of output vectors per cycle from the line
      // buffer kernel.
      int lbufOutputChunkSize = lbufOutput.getSize();
      int numLbufOutputChunks =
          cp.useWinograd ? WINO_LBUF_HEIGHT : (tileSize + cp.PK - 1) / cp.PK;

      owner.getManager().logMsg(String.format("Size of line buffer output: %d",
                                              lbufOutput.getSize()));
      owner.getManager().logMsg(String.format(
          "Number of line buffer output chunks: %d", numLbufOutputChunks));

      // organize output
      for (int j = 0; j < numLbufOutputChunks; j++) {
        // reverse chunk, the less j is the older the chunk is in the stream.
        owner.getManager().logMsg(
            String.format("Connecting outputs from chunk (#%03d) ...", j));
        DFEVector<DFEVar> lbufOutputChunk =
            stream.offset(lbufOutput, -(numLbufOutputChunks - j - 1));

        if (cp.useWinograd) {
          // outputVec: (PC, 6, 6)
          int numTilesPerChunk =
              lbufOutputChunkSize / (WINO_LBUF_TILE_SIZE * WINO_LBUF_TILE_SIZE);

          // iterate every element in the output chunk and assign them to the
          // final output
          for (int t = 0; t < numTilesPerChunk; t++) {
            for (int x = 0; x < WINO_LBUF_TILE_SIZE; x++) {
              for (int y = 0; y < WINO_LBUF_TILE_SIZE; y++) {
                int xi = t * WINO_LBUF_TILE_SIZE + x;
                int yi = j * WINO_LBUF_TILE_SIZE + y;

                if (xi < WINO_LBUF_PADDING_WIDTH ||
                    yi < WINO_LBUF_PADDING_WIDTH)
                  continue;

                int xj = xi - WINO_LBUF_PADDING_WIDTH;
                int yj = yi - WINO_LBUF_PADDING_WIDTH;
                int srcIdx =
                    (x * WINO_LBUF_TILE_SIZE + y) * numTilesPerChunk + t;
                int dstIdx = i * WinogradTransform.TILE_SIZE *
                                 WinogradTransform.TILE_SIZE +
                             xj * WinogradTransform.TILE_SIZE + yj;

                owner.getManager().logMsg(String.format(
                    "Connect FROM line buffer output (%d, %d, %d) TO " +
                        "final output (%d, %d, %d) dst %d src %d",
                    i, xi, yi, i, xj, yj, dstIdx, srcIdx));
                outputVec[dstIdx].connect(lbufOutputChunk[srcIdx]);
              }
            }
          }
        } else {
          /* outputVec: (PC, K, K + PK - 1) */
          for (int p = 0; p < cp.PK; p++) {
            for (int k = 0; k < lineBufferHeight; k++) {
              int idx =
                  i * (lineBufferHeight * (lineBufferHeight + cp.PK - 1)) +
                  k * (lineBufferHeight + cp.PK - 1) + j * cp.PK + p;

              outputVec[idx].connect(lbufOutputChunk[p * lineBufferHeight + k]);
            }
          }
        }
      }
    }

    if (cp.dbg) {
      debug.simPrintf("Line buffer inputVec = %KObj%\n", inputVec);
      debug.simPrintf("Line buffer outputVec = %KObj%\n", outputVec);
    }
  }

 public
  int getOutputVecSize() {
    if (cp.useWinograd) {
      // Return a 6 x 6 tile for each parallel channel
      return cp.PC * WinogradTransform.TILE_SIZE * WinogradTransform.TILE_SIZE;
    } else {
      return cp.PC * lineBufferHeight * (lineBufferHeight + cp.PK - 1);
    }
  }

 public
  DFEVectorType<DFEVar> getOutputVecT() { return outputVecT; }

 public
  DFEVector<DFEVar> getOutputVec() { return outputVec; }

 public
  int getInputVecSize() {
    if (cp.useWinograd) {
      // 4 x 4 x PC
      return WINO_LBUF_NUM_PIPES * cp.PC;
    } else {
      return cp.PC * cp.PK;
    }
  }

 public
  DFEVectorType<DFEVar> getInputVecT() { return inputVecT; }

 public
  void setInput(DFEVector<DFEVar> inputVec) { this.inputVec.connect(inputVec); }

 public
  int getNumLineBuffers() { return cp.PC; }

  /**
   * Decide the height of line buffer.
   *
   * @return Line buffer height
   */
 public
  static int getLineBufferHeight(ConvLayerParameters cp) {
    if (cp.useWinograd) {
      // a customised constant
      return WINO_LBUF_HEIGHT;
    } else {
      // decided by the kernel size;
      return cp.K;
    }
  }

 public
  static int getLineBufferWidth(ConvLayerParameters cp) {
    return cp.useWinograd
               ? (cp.W + WINO_LBUF_PADDING_WIDTH) * WINO_LBUF_TILE_SIZE
               : cp.W;
  }

 public
  static int getLineBufferNumPipes(ConvLayerParameters cp) {
    return cp.useWinograd ? WINO_LBUF_NUM_PIPES : cp.PK;
  }

 public
  int getLineBufferTileSize() { return lineBufferHeight; }

  /**
   * Build the input vector to the line buffer.
   *
   * @return An input vector.
   */
 private
  DFEVector<DFEVar> createLineBufferInputVector(int ci) {
    // vector per line buffer
    int vecSize = getInputVecSize() / cp.PC;
    DFEVectorType<DFEVar> VT = new DFEVectorType<DFEVar>(T, vecSize);
    DFEVector<DFEVar> vec = VT.newInstance(getOwner());

    // inputVec is the original vector
    // This function works for both Winograd and standard.
    for (int i = 0; i < vecSize; i++)
      vec[i].connect(inputVec[ci * vecSize + i]);

    return vec;
  }
}
