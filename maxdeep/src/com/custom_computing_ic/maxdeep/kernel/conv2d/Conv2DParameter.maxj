package com.custom_computing_ic.maxdeep.kernel.conv2d;

@Deprecated
public class Conv2DParameter {

  public enum Mode {
    CHNL_MAJOR, FLTR_MAJOR
  };

  private final int maxInpHeight;
  private final int maxInpWidth;
  private final int maxNumChnl;
  private final int maxNumFltr;
  private int       numChnlPipes;
  private int       numFltrPipes;
  private final int knlHeight;
  private final int knlWidth;
  private final int knlSize;
  private int       maxPadHeight;
  private int       maxPadWidth;
  private Mode      mode;

  public Conv2DParameter(int maxInpHeight, int maxInpWidth, int maxNumChnl,
      int maxNumFltr, int knlHeight, int knlWidth) {
    this.maxInpHeight = maxInpHeight;
    this.maxInpWidth = maxInpWidth;
    this.maxNumChnl = maxNumChnl;
    this.maxNumFltr = maxNumFltr;
    this.maxPadHeight = 0;
    this.maxPadWidth = 0;
    this.numChnlPipes = 1;
    this.numFltrPipes = 1;
    this.knlHeight = knlHeight;
    this.knlWidth = knlWidth;
    this.knlSize = this.knlHeight * this.knlWidth;

    if (this.maxInpHeight <= 0)
      throw new IllegalArgumentException("maxInpHeight should be larger than 0");
    if (this.maxInpWidth <= 0)
      throw new IllegalArgumentException("maxInpWidth should be larger than 0");
    if (this.maxNumChnl <= 0)
      throw new IllegalArgumentException("maxNumChnl should be larger than 0");
    if (this.maxNumFltr <= 0)
      throw new IllegalArgumentException("maxNumFltr should be larger than 0");
    if (this.knlHeight <= 0)
      throw new IllegalArgumentException("knlHeight should be larger than 0");
    if (this.knlWidth <= 0)
      throw new IllegalArgumentException("knlWidth should be larger than 0");
  }

  public int getMaxPadHeight() {
    return maxPadHeight;
  }

  public void setMaxPadHeight(int maxPadHeight) {
    if (maxPadHeight <= 0)
      throw new IllegalArgumentException("maxPadHeight should be larger than 0");
    this.maxPadHeight = maxPadHeight;
  }

  public int getMaxPadWidth() {
    return maxPadWidth;
  }

  public void setMaxPadWidth(int maxPadWidth) {
    if (maxPadWidth <= 0)
      throw new IllegalArgumentException("maxPadWidth should be larger than 0");
    this.maxPadWidth = maxPadWidth;
  }

  public int getMaxInpHeight() {
    return maxInpHeight;
  }

  public int getMaxInpWidth() {
    return maxInpWidth;
  }

  public int getMaxNumChnl() {
    return maxNumChnl;
  }

  public int getMaxNumFltr() {
    return maxNumFltr;
  }

  public int getNumChnlPipes() {
    return numChnlPipes;
  }

  public int getNumFltrPipes() {
    return numFltrPipes;
  }

  public int getKnlHeight() {
    return knlHeight;
  }

  public int getKnlWidth() {
    return knlWidth;
  }

  public int getKnlSize() {
    return knlSize;
  }

  public Mode getMode() {
    return mode;
  }

  public int getMaxOutHeight() {
    return this.maxInpHeight + this.maxPadHeight - this.knlHeight + 1;
  }

  public int getMaxOutWidth() {
    return this.maxInpWidth + this.maxPadWidth - this.knlWidth + 1;
  }

  public void setNumChnlPipes(int numChnlPipes) {
    if (numChnlPipes <= 0)
      throw new IllegalArgumentException("numChnlPipes should be larger than 0");
    if (numChnlPipes > maxNumChnl)
      throw new IllegalArgumentException(
          "numChnlPipes should be not be larger than maxNumChnl");
    this.numChnlPipes = numChnlPipes;
  }

  public void setNumFltrPipes(int numFltrPipes) {
    if (numFltrPipes <= 0)
      throw new IllegalArgumentException("numFltrPipes should be larger than 0");
    if (numFltrPipes > maxNumFltr)
      throw new IllegalArgumentException(
          "numFltrPipes should be not be larger than maxNumFltr");
    this.numFltrPipes = numFltrPipes;
  }

  public void setMode(Mode mode) {
    this.mode = mode;
  }
}
