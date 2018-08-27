/**
 * 
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.Conv2DParameter.Mode;

/**
 * @author Ruizhe Zhao
 * 
 */
public final class Conv2DProcessEngineParameters {

  private final int  maxHeight;
  private final int  maxWidth;
  private final int  minHeight;
  private final int  minWidth;
  private final int  maxNumChnl;
  private final int  maxNumFltr;
  private final int  maxKnlHeight;
  private final int  maxKnlWidth;
  private final int  minKnlHeight;
  private final int  minKnlWidth;
  private final int  numParaChnl;
  private final int  numParaFltr;
  private final Mode mode;

  public Conv2DProcessEngineParameters(int maxHeight, int maxWidth,
      int minHeight, int minWidth, int maxNumChnl, int maxNumFltr,
      int maxKnlHeight, int maxKnlWidth, int numParaChnl, int numParaFltr) {

    if (maxHeight <= 0)
      throw new IllegalArgumentException("maxHeight should be larger than 0");
    if (maxWidth <= 0)
      throw new IllegalArgumentException("maxWidth should be larger than 0");
    if (minHeight <= 0)
      throw new IllegalArgumentException("minHeight should be larger than 0");
    if (minWidth <= 0)
      throw new IllegalArgumentException("minWidth should be larger than 0");
    if (minHeight > maxHeight)
      throw new IllegalArgumentException(
          "minHeight should not be larger than maxHeight");
    if (minWidth > minHeight)
      throw new IllegalArgumentException(
          "minWidth should not be larger than maxWidth");
    if (maxKnlHeight <= 0)
      throw new IllegalArgumentException("maxKnlHeight should be larger than 0");
    if (maxKnlWidth <= 0)
      throw new IllegalArgumentException("maxKnlHeight should be larger than 0");
    if (numParaChnl <= 0)
      throw new IllegalArgumentException("numParaChnl should be larger than 0");
    if (numParaFltr <= 0)
      throw new IllegalArgumentException("numParaFltr should be larger than 0");

    this.maxHeight = maxHeight;
    this.maxWidth = maxWidth;
    this.minHeight = minHeight;
    this.minWidth = minWidth;
    this.maxNumChnl = maxNumChnl;
    this.maxNumFltr = maxNumFltr;
    this.maxKnlHeight = maxKnlHeight;
    this.maxKnlWidth = maxKnlWidth;
    this.minKnlHeight = maxKnlHeight;
    this.minKnlWidth = maxKnlWidth;
    this.numParaChnl = numParaChnl;
    this.numParaFltr = numParaFltr;
    this.mode = Mode.FLTR_MAJOR;
  }

  public Mode getMode() {
    return mode;
  }

  public int getMaxKnlSize() {
    return maxKnlHeight * maxKnlWidth;
  }

  public int getNumParaChnl() {
    return numParaChnl;
  }

  public int getNumParaFltr() {
    return numParaFltr;
  }

  public int getMaxHeight() {
    return maxHeight;
  }

  public int getMaxWidth() {
    return maxWidth;
  }

  public int getMinHeight() {
    return minHeight;
  }

  public int getMinWidth() {
    return minWidth;
  }

  public int getMaxNumChnl() {
    return maxNumChnl;
  }

  public int getMaxNumFltr() {
    return maxNumFltr;
  }

  public int getMaxKnlHeight() {
    return maxKnlHeight;
  }

  public int getMaxKnlWidth() {
    return maxKnlWidth;
  }

  public int getMaxOutHeight() {
    return maxHeight - minKnlHeight + 1;
  }

  public int getMaxOutWidth() {
    return maxWidth - minKnlWidth + 1;
  }

  public int getMinOutHeight() {
    return minHeight - maxKnlHeight + 1;
  }

  public int getMinOutWidth() {
    return minWidth - maxKnlWidth + 1;
  }
}
