package com.custom_computing_ic.maxdeep.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;

/**
 * Create an adder tree to reduce the input list of values.
 * 
 * @author Ruizhe Zhao
 * @since 18/06/2017
 */
public class AdderTree {

  public static DFEVar reduce(DFEVar... values) {
    return reduce(Arrays.asList(values));
  }

  public static DFEVar reduce(List<DFEVar> values) {
    if (values.size() == 1)
      return values[0];
    if (values.size() == 2)
      return values[0] + values[1];
    List<DFEVar> results = new ArrayList<DFEVar>();
    for (int i = 0; i < values.size(); i += 2) {
      int fromIndex = i;
      int toIndex = (i == values.size() - 1) ? i + 1 : i + 2;
      results.add(reduce(values.subList(fromIndex, toIndex)));
    }
    return reduce(results);
  }
}
