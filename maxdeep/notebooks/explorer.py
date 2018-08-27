""" Design space exploration.
"""
import math
from net import ConvNet
from design import DesignParams, BasicDesign
from fpga_platform import Platform
from perf_predict import BasicPredictor

import numpy as np
import itertools
from collections import OrderedDict


def factor(n):
  result = []
  for i in range(1, n + 1):
    if n % i == 0:
      result.append(i)
  return reversed(result)


class Explorer(object):
  def explore(self, net: ConvNet, platform: Platform):
    pass


class BasicExplorer(Explorer):

  def candidates(self, use_winograd=False):
    tile_shapes = OrderedDict({
        "T_H": [7, 14, 28, 56, 112, 224],
        "T_W": [7, 14, 28, 56, 112, 224],
        "T_C": [64, 128, 256, 512, 1024, 2048],
        "T_F": [64, 128, 256, 512, 1024, 2048],
    })

    for T_H,  T_W, T_C, T_F in itertools.product(*list(tile_shapes.values())):
      T_ROW = T_C
      T_COL = T_H * T_W

      params = OrderedDict({
          'P_H': [1],
          'P_W': [1],
          'P_C': factor(T_C),
          'P_F': factor(T_F)
      })

      for P_H, P_W, P_C, P_F in itertools.product(*list(params.values())):
        if use_winograd:
          P_COL = P_C
          P_ROW = P_F * 36
        else:
          P_COL = P_H * P_W * P_C
          P_ROW = P_F * 9

        yield [T_H, T_W, T_C, T_F, P_H, P_W, P_C, P_F, T_ROW, T_COL, P_ROW, P_COL]

  def explore(self, net: ConvNet, platform: Platform, use_winograd=False):
    predictor = BasicPredictor()

    results = []
    for cfg in self.candidates(use_winograd):
      params = DesignParams(T_H=cfg[0], T_W=cfg[1], T_C=cfg[2], T_F=cfg[3], P_H=cfg[4], P_W=cfg[5],
                            P_C=cfg[6], P_F=cfg[7], T_ROW=cfg[8], T_COL=cfg[9], P_ROW=cfg[10], P_COL=cfg[11],
                            use_winograd=use_winograd)

      design = BasicDesign(params)
      res = design.resource
      BRAM, DSP, ALM = res.BRAM, res.DSP, res.ALM

      if BRAM > platform.max_BRAM or DSP > platform.max_DSP:  # or ALM > platform.max_ALM:
        continue

      num_cycle, elapsed, GFLOPS, _, CTC, _ = predictor.predict(net, design)

      results.append([BRAM, DSP, ALM, num_cycle, elapsed, GFLOPS, CTC, params])

    return np.array(results)


class FusedBlockExplorer(Explorer):

  def bottleneck_candidates(self, use_winograd=False):
    tile_shapes = OrderedDict({
        "T_H": [7, 14, 28],  # , 112, 224],
        "T_W": [7, 14, 28],  # , 112, 224],
        'T_C': [1024, 2048],
        'T_F': [1024, 2048],
        'T_C2': [1024, 2048],
        'T_C3': [1024, 2048],
    })

    for T_H, T_W, T_C, T_F, T_C2, T_C3 in itertools.product(*list(tile_shapes.values())):
      T_ROW = T_F
      T_COL = T_H * T_W * T_C

      params = OrderedDict({
          'P_H': [1],
          'P_W': [1],
          'P_C1': factor(T_C),
          'P_C2': factor(T_C2),
          'P_C3': factor(T_C3),
          'P_FF': factor(T_F)
      })

      for P_H, P_W, P_C1, P_C2, P_C3, P_FF in itertools.product(*list(params.values())):
        # P_C = P_C2 * P_C3 + int(math.ceil((P_C1 * P_C2 + P_C3 * P_FF) / 36))
        # P_F = 1
        P_C = P_C2
        P_F = P_C3

        if use_winograd:
          P_COL = P_C * 4
          P_ROW = P_F * 9
        else:
          P_COL = P_H * P_W * P_C
          P_ROW = P_F * 9

        T_C1 = T_C
        T_FF = T_F

        yield [T_H, T_W, T_C, T_F, P_H, P_W, P_C, P_F, T_ROW, T_COL, P_ROW, P_COL,
               P_C1, P_C2, P_C3, P_FF, T_C1, T_C2, T_C3, T_FF]

  def depthwise_separable_candidates(self, use_winograd=False):
    tile_shapes = OrderedDict({
        "T_H": [7, 14, 28, 56, 112, 224],
        "T_W": [7, 14, 28, 56, 112, 224],
        "T_C": [64, 128, 256, 512],
        "T_F": [64, 128, 256, 512],
    })

    for T_H,  T_W, T_C, T_F in itertools.product(*list(tile_shapes.values())):
      T_ROW = T_C
      T_COL = T_H * T_W

      params = OrderedDict({
          'P_H': [1],
          'P_W': [1],
          'P_C': factor(T_C),
          'P_F': factor(T_F)
      })

      for P_H, P_W, P_C1, P_FF in itertools.product(*list(params.values())):
        P_C = P_C1
        P_F = 1
        if use_winograd:
          P_COL = P_C
          P_ROW = P_F * 36
        else:
          P_COL = P_H * P_W * P_C
          P_ROW = P_F * 9

        yield [T_H, T_W, T_C, T_F, P_H, P_W, P_C, P_F, T_ROW, T_COL, P_ROW, P_COL,
               P_C1, None, None, P_FF, None, None, None, None]

  def separable_bottleneck_candidates(self, use_winograd=False):
    tile_shapes = OrderedDict({
        "T_H": [7, 14],  # , 112, 224],
        "T_W": [7, 14],  # , 112, 224],
        'T_C1': [512, 1024, 2048],
        'T_C2': [512, 1024, 2048],
        'T_FF': [512, 1024, 2048],
    })

    for T_H, T_W, T_C1, T_C2, T_FF in itertools.product(*list(tile_shapes.values())):
      T_C = T_C1
      T_F = T_FF
      T_ROW = T_F
      T_COL = T_H * T_W * T_C

      params = OrderedDict({
          'P_H': [1],
          'P_W': [1],
          'P_C1': factor(T_C1),
          'P_C2': factor(T_C2),
          'P_FF': factor(T_FF)
      })

      for P_H, P_W, P_C1, P_C2, P_FF in itertools.product(*list(params.values())):
        # P_C = P_C2 * P_C3 + int(math.ceil((P_C1 * P_C2 + P_C3 * P_FF) / 36))
        # P_F = 1
        P_C = P_C2
        P_F = 2

        if use_winograd:
          P_COL = P_C * 4
          P_ROW = P_F * 9
        else:
          P_COL = P_H * P_W * P_C
          P_ROW = P_F * 9

        T_C1 = T_C
        T_FF = T_F

        yield [T_H, T_W, T_C, T_F, P_H, P_W, P_C, P_F, T_ROW, T_COL, P_ROW, P_COL,
               P_C1, P_C2, None, P_FF, T_C1, T_C2, None, T_FF]

  def candidates(self, block_type, use_winograd=False):
    if block_type == 'Bottleneck':
      return self.bottleneck_candidates(use_winograd=use_winograd)
    if block_type == 'DepthwiseSeparable':
      return self.depthwise_separable_candidates(use_winograd=use_winograd)
    if block_type == 'SeparableBottleneck':
      return self.separable_bottleneck_candidates(use_winograd=use_winograd)

    raise TypeError('Unrecognised block type: %s', block_type)

  def explore(self, net: ConvNet, platform: Platform, block_type, use_winograd=False, max_number_of_steps=300000):
    predictor = BasicPredictor()

    RES = []
    best_GFLOPS = 0.0
    best_CTC = 0.0
    best_DSP = 0
    best_BRAM = 0
    best_params = None
    best_results = None

    for idx, cfg in enumerate(self.candidates(block_type, use_winograd=use_winograd)):
      if idx == max_number_of_steps:
        break

      if idx % 10000 == 0 and best_results:
        print('%12d: Best GFLOPS = %10.6f CTC = %10.6f DSP = %d BRAM = %d' %
              (idx, best_GFLOPS, best_CTC, best_DSP, best_BRAM))
        for i, l in enumerate(net.layers):
          print('%30s: # cycles = %10d # data = %10d' %
                (l.name, best_results[i][0], best_results[i][1]))
        print(best_params.__dict__)

      params = DesignParams(T_H=cfg[0], T_W=cfg[1], T_C=cfg[2], T_F=cfg[3], P_H=cfg[4], P_W=cfg[5],
                            P_C=cfg[6], P_F=cfg[7], T_ROW=cfg[8], T_COL=cfg[9], P_ROW=cfg[10], P_COL=cfg[11],
                            P_C1=cfg[12], P_C2=cfg[13], P_C3=cfg[14], P_FF=cfg[15],
                            T_C1=cfg[16], T_C2=cfg[17], T_C3=cfg[18], T_FF=cfg[19],
                            use_winograd=use_winograd, block_type=block_type)

      design = BasicDesign(params)
      res = design.resource
      BRAM, DSP, ALM = res.BRAM, res.DSP, res.ALM

      if BRAM > platform.max_BRAM or DSP > platform.max_DSP:  # or ALM > platform.max_ALM:
        continue

      num_cycle, elapsed, GFLOPS, _, CTC, results = predictor.predict(
          net, design)

      if CTC * 38 < GFLOPS:
        continue

      if GFLOPS > best_GFLOPS:
        best_GFLOPS = max(GFLOPS, best_GFLOPS)
        best_CTC = CTC
        best_DSP = DSP
        best_BRAM = BRAM
        best_params = params
        best_results = results

      RES.append([BRAM, DSP, ALM, num_cycle, elapsed, GFLOPS, CTC, params])

    return np.array(RES)
