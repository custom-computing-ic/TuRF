package com.custom_computing_ic.maxdeep.kernel.fuse;

import java.util.ArrayList;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;

public class FusedConvLayerParameters {
  public List<ConvLayerParameters> cps;

  public FusedConvLayerParameters(Builder builder) {
    cps = builder.cps;
  }

  public int getNumLayers() {
    return cps.size();
  }

  public class Builder {
    public List<ConvLayerParameters> cps;

    public Builder(int numLayers) {
      cps = new ArrayList<ConvLayerParameters>();
    }

    public Builder addConvLayer(ConvLayerParameters cp) {
      cps.add(cp);
      return this;
    }
  }
}
