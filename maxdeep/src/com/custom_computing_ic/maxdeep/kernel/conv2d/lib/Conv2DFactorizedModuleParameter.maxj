package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

public class Conv2DFactorizedModuleParameter {

  public enum ShapeMode {
    STATIC, DYNAMIC;
  }

  public static class StaticBuilder {
    private final int ifmapHeight;
    private final int ifmapWidth;
    private final int ifmapNumChnl;
    private final int ofmapNumChnl;
    private int       ifmapNumParaChnl;
    private int       ofmapNumParaChnl;
    private int       knlHeight, knlWidth;

    public StaticBuilder(int ifmapHeight, int ifmapWidth, int ifmapNumChnl,
        int ofmapNumChnl) {
      this.ifmapHeight = ifmapHeight;
      this.ifmapWidth = ifmapWidth;
      this.ifmapNumChnl = ifmapNumChnl;
      this.ofmapNumChnl = ofmapNumChnl;
      this.knlHeight = 1;
      this.knlWidth = 1;
      this.ifmapNumParaChnl = 1;
      this.ofmapNumParaChnl = 1;
    }

    public StaticBuilder knlShape(int length) {
      this.knlHeight = length;
      this.knlWidth = length;
      return this;
    }

    public StaticBuilder knlHeight(int knlHeight) {
      this.knlHeight = knlHeight;
      return this;
    }

    public StaticBuilder knlWidth(int knlWidth) {
      this.knlWidth = knlWidth;
      return this;
    }

    public StaticBuilder ifmapNumParaChnl(int ifmapNumParaChnl) {
      this.ifmapNumParaChnl = ifmapNumParaChnl;
      return this;
    }

    @Deprecated
    public StaticBuilder numParaIfmapChnl(int numParaIfmapChnl) {
      this.ifmapNumParaChnl = numParaIfmapChnl;
      return this;
    }

    public StaticBuilder ofmapNumParaChnl(int ofmapNumParaChnl) {
      this.ofmapNumParaChnl = ofmapNumParaChnl;
      return this;
    }

    @Deprecated
    public StaticBuilder numParaOfmapChnl(int numParaOfmapChnl) {
      this.ofmapNumParaChnl = numParaOfmapChnl;
      return this;
    }

    public Conv2DFactorizedModuleParameter build() {
      return new Conv2DFactorizedModuleParameter(this);
    }
  }

  private final int       ifmapHeight;
  private final int       ifmapWidth;
  private final int       ifmapNumChnl;
  private final int       ofmapNumChnl;
  private final int       knlHeight;
  private final int       knlWidth;
  private final int       ifmapNumParaChnl;
  private final int       ofmapNumParaChnl;
  private final ShapeMode shapeMode;

  private Conv2DFactorizedModuleParameter(StaticBuilder builder) {
    this.shapeMode = ShapeMode.STATIC;
    this.ifmapHeight = builder.ifmapHeight;
    this.ifmapWidth = builder.ifmapWidth;
    this.ifmapNumChnl = builder.ifmapNumChnl;
    this.ofmapNumChnl = builder.ofmapNumChnl;
    this.knlHeight = builder.knlHeight;
    this.knlWidth = builder.knlWidth;
    this.ifmapNumParaChnl = builder.ifmapNumParaChnl;
    this.ofmapNumParaChnl = builder.ofmapNumParaChnl;
  }

  public int getKnlHeight() {
    return knlHeight;
  }

  public int getKnlWidth() {
    return knlWidth;
  }

  public int getKnlSize() {
    return knlHeight * knlWidth;
  }

  public ShapeMode getShapeMode() {
    return shapeMode;
  }

  public int getOfmapHeight() {
    return ifmapHeight - knlHeight + 1;
  }

  public int getOfmapWidth() {
    return ifmapWidth - knlWidth + 1;
  }

  public int getOfmapSize() {
    return getOfmapHeight() * getOfmapWidth();
  }

  public int getOfmapNumChnl() {
    return ofmapNumChnl;
  }

  public int getIfmapHeight() {
    return ifmapHeight;
  }

  public int getIfmapWidth() {
    return ifmapWidth;
  }

  public int getIfmapNumChnl() {
    return ifmapNumChnl;
  }

  public int getOfmapNumParaChnl() {
    return ofmapNumParaChnl;
  }

  @Deprecated
  public int getNumParaOfmapChnl() {
    return ofmapNumParaChnl;
  }

  public int getIfmapTotalSize() {
    return ifmapHeight * ifmapWidth * ifmapNumChnl;
  }

  public int getCacheTotalSize() {
    return getOfmapSize() * ifmapNumChnl;
  }

  public int getIfmapNumParaChnl() {
    return ifmapNumParaChnl;
  }

  public int getNumCycles() {
    return getOfmapHeight() * getOfmapWidth() * ifmapNumChnl / getIfmapNumParaChnl() * ofmapNumChnl / getOfmapNumParaChnl();
  }

  @Deprecated
  public int getNumParaIfmapChnl() {
    return ifmapNumParaChnl;
  }

}
