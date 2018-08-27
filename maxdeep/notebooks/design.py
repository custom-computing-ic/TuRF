""" Design
"""

import math
from collections import OrderedDict

import numpy as np
from net import ConvNet
from fpga_platform import StratixVPlatform
from utils import int_ceil
from layer import BaseLayer, ConvLayer, MatmulLayer,\
    ConvBlock, BottleneckBlock, DepthwiseSeparableBlock, SeparableBottleneckBlock,\
    STD, PNT, DPT

BIT_WIDTH = 16


class DesignResourceUsage(object):
  def __init__(self, LUT=0, FF=0, BRAM=0, DSP=0, ALM=0):
    self.LUT = int(LUT)
    self.FF = int(FF)
    self.BRAM = int(BRAM)
    self.DSP = int(DSP)
    self.ALM = int(ALM)

  def __str__(self):
    return '<%6d, %6d, %6d, %6d, %6d>' % (self.LUT, self.FF, self.BRAM, self.DSP, self.ALM)

  def __repr__(self):
    return self.__str__()


class DesignParams(object):
  def __init__(self, P_H=1, P_W=1, P_C=1, P_F=1, T_H=1, T_W=1, T_C=1, T_F=1, T_C1=1, T_C2=1, T_C3=1, T_FF=1,
               P_ROW=1, P_COL=1, T_ROW=1, T_COL=1, P_C1=1, P_C2=1, P_C3=1, P_FF=1, K=3, use_winograd=False,
               block_type=None):
    if use_winograd:
      self.P_H = 4
      self.P_W = 4
    else:
      self.P_H = P_H
      self.P_W = P_W
    self.P_C = P_C
    self.P_F = P_F
    self.P_ROW = P_ROW
    self.P_COL = P_COL

    self.T_H = T_H
    self.T_W = T_W
    self.T_C = T_C
    self.T_F = T_F
    self.T_ROW = T_ROW
    self.T_COL = T_COL

    self.K = K
    self.use_winograd = use_winograd
    self.block_type = block_type

    self.P_C1 = P_C1
    self.P_C2 = P_C2
    self.P_C3 = P_C3
    self.P_FF = P_FF
    self.T_C1 = T_C1
    self.T_C2 = T_C2
    self.T_C3 = T_C3
    self.T_FF = T_FF


class Design(object):

  def __init__(self, params: DesignParams):
    self.params = params

  def run(self, net: ConvNet):
    pass


class BasicDesign(Design):

  def run_std_conv_tile(self, layer):
    T_C = min(self.params.T_C, layer.C)
    T_F = min(self.params.T_F, layer.F)

    total_num_cycles = self.num_cycle_std_conv(self.params.T_H, self.params.T_W, T_C, T_F,
                                               self.params.P_H, self.params.P_W,
                                               self.params.P_C, self.params.P_F)
    total_num_cycles = int(math.ceil(total_num_cycles / (layer.stride ** 2)))

    total_data = ((T_C * T_F * (self.params.K**2)) +
                  (self.params.T_H * self.params.T_W * T_C) +
                  (self.params.T_H * self.params.T_W * T_F))

    return np.array([total_num_cycles, total_data])

  def run_pnt_conv_tile(self, layer):
    T_C = min(self.params.T_C, layer.C)
    T_F = min(self.params.T_F, layer.F)

    P_C = self.params.P_C
    if self.params.use_winograd:
      P_F = self.params.P_F * 36
    else:
      P_F = self.params.P_F

    total_num_cycles = self.num_cycle_std_conv(self.params.T_H, self.params.T_W, T_C, T_F,
                                               1, 1, P_C, P_F)
    total_num_cycles = int(math.ceil(total_num_cycles / (layer.stride ** 2)))

    total_data = ((T_C * T_F) +
                  (self.params.T_H * self.params.T_W * T_C) +
                  (self.params.T_H * self.params.T_W * T_F))

    return np.array([total_num_cycles, total_data])

  def run_dpt_conv_tile(self, layer):
    T_C = min(self.params.T_C, layer.C)

    num_cycle = [int_ceil(self.params.T_H, self.params.P_H),
                 int_ceil(self.params.T_W, self.params.P_W),
                 int_ceil(T_C, self.params.P_C * self.params.P_F)]

    total_num_cycles = np.prod(num_cycle)
    total_num_cycles = int(math.ceil(total_num_cycles / (layer.stride ** 2)))

    total_data = ((T_C * (self.params.K ** 2)) +
                  (self.params.T_H * self.params.T_W * T_C) * 2)

    return np.array([total_num_cycles, total_data])

  def run_std_conv(self, layer: ConvLayer):
    num_tile = [int_ceil(layer.H, self.params.T_H), int_ceil(layer.W, self.params.T_W),
                int_ceil(layer.C, self.params.T_C), int_ceil(layer.F, self.params.T_F)]
    total_num_tiles = np.prod(num_tile)

    return self.run_std_conv_tile(layer) * total_num_tiles

  def run_pnt_conv(self, layer: ConvLayer):
    num_tile = [int_ceil(layer.H, self.params.T_H), int_ceil(layer.W, self.params.T_W),
                int_ceil(layer.C, self.params.T_C), int_ceil(layer.F, self.params.T_F)]
    total_num_tiles = np.prod(num_tile)

    return self.run_pnt_conv_tile(layer) * total_num_tiles

  def run_dpt_conv(self, layer: ConvLayer):
    num_tile = [int_ceil(layer.H, self.params.T_H), int_ceil(layer.W, self.params.T_W),
                int_ceil(layer.C, self.params.T_C)]
    total_num_tiles = np.prod(num_tile)

    return self.run_dpt_conv_tile(layer) * total_num_tiles

  def run_conv(self, layer: ConvLayer):
    if layer.T == STD:
      return self.run_std_conv(layer)
    if layer.T == PNT:
      return self.run_pnt_conv(layer)
    if layer.T == DPT:
      return self.run_dpt_conv(layer)

  def run_matmul_tile(self, layer: MatmulLayer):
    T_ROW = min(self.params.T_ROW, layer.num_rows)
    T_COL = min(self.params.T_COL, layer.num_cols)
    num_cycle = [int_ceil(T_ROW, self.params.P_ROW),
                 int_ceil(T_COL, self.params.P_COL)]

    total_num_cycles = np.prod(num_cycle)
    total_data = (T_ROW * T_COL + T_COL + T_ROW)

    return np.array([total_num_cycles, total_data])

  def run_matmul(self, layer: MatmulLayer):
    num_tile = [int_ceil(layer.num_cols, self.params.T_COL),
                int_ceil(layer.num_rows, self.params.T_ROW)]
    total_num_tiles = np.prod(num_tile)

    return total_num_tiles * self.run_matmul_tile(layer)

  def num_cycle_std_conv(self, T_H, T_W, T_C, T_F, P_H, P_W, P_C, P_F):
    num_cycle = [int_ceil(T_H, P_H), int_ceil(T_W, P_W),
                 int_ceil(T_C, P_C), int_ceil(T_F, P_F)]

    return np.prod(num_cycle)

  def num_cycle_dws_conv(self, T_H, T_W, T_C, P_H, P_W, P_C):
    num_cycle = [int_ceil(T_H, P_H), int_ceil(T_W, P_W), int_ceil(T_C, P_C), ]

    return np.prod(num_cycle)

  def run_bottleneck_block_tile(self, block):
    if not self.params.use_winograd:
      raise RuntimeError('We only support Winograd mode for now.')

    T_C1 = min(self.params.T_C1, block.layers[0].C)
    T_C2 = min(self.params.T_C2, block.layers[1].C)
    T_C3 = min(self.params.T_C3, block.layers[2].C)
    T_FF = min(self.params.T_FF, block.layers[2].F)

    num_cycles = [
        self.num_cycle_std_conv(self.params.T_H, self.params.T_W,
                                T_C1, T_C2,
                                1, 1,
                                self.params.P_C1, self.params.P_C2),
        self.num_cycle_std_conv(self.params.T_H, self.params.T_W,
                                T_C2, T_C3,
                                self.params.P_H, self.params.P_W,
                                self.params.P_C2, self.params.P_C3),
        self.num_cycle_std_conv(self.params.T_H, self.params.T_W,
                                T_C3, T_FF,
                                self.params.P_H, self.params.P_W,
                                self.params.P_C3, self.params.P_FF),
    ]
    num_cycles[0] = int(math.ceil(num_cycles[0] / (block.S ** 2)))

    total_num_cycle = np.max(num_cycles)
    # print(num_cycles, total_num_cycle)
    total_data = (self.params.T_H * self.params.T_W * (T_C1 + T_FF) +
                  T_C1 * T_C2 + T_C2 * T_C3 * (self.params.K ** 2) + T_C3 * T_FF)

    return np.array([total_num_cycle, total_data])

  def run_bottleneck_block(self, block: BottleneckBlock):
    num_tile = [int_ceil(block.layers[0].H, self.params.T_H),
                int_ceil(block.layers[0].W, self.params.T_W),
                int_ceil(block.layers[0].C, self.params.T_C1),
                int_ceil(block.layers[1].C, self.params.T_C2),
                int_ceil(block.layers[2].C, self.params.T_C3),
                int_ceil(block.layers[2].F, self.params.T_FF)]
    total_num_tiles = np.prod(num_tile)

    result = self.run_bottleneck_block_tile(block) * total_num_tiles
    if len(block.layers) == 4:
      result += self.run_layer(block.layers[3])
    return result

  def run_depthwise_separable_block_tile(self, block):
    if not self.params.use_winograd:
      raise RuntimeError('We only support Winograd mode for now.')

    T_C = min(self.params.T_C, block.layers[0].C)
    T_F = min(self.params.T_F, block.layers[1].F)

    num_cycle = self.num_cycle_std_conv(self.params.T_H, self.params.T_W, T_C, T_F,
                                        self.params.P_H, self.params.P_W, self.params.P_C1, self.params.P_FF)
    total_num_cycle = int(math.ceil(num_cycle / (block.S ** 2)))

    # print(num_cycles, total_num_cycle)
    total_data = (self.params.T_H * self.params.T_W * (T_C + T_F) +
                  T_C * T_F + T_C * (self.params.K ** 2))

    return np.array([total_num_cycle, total_data])

  def run_separable_bottleneck_block_tile(self, block):
    if not self.params.use_winograd:
      raise RuntimeError('We only support Winograd mode for now.')

    T_C1 = min(self.params.T_C1, block.layers[0].C)
    T_C2 = min(self.params.T_C2, block.layers[1].C)
    T_FF = min(self.params.T_FF, block.layers[2].F)

    num_cycles = [
        self.num_cycle_std_conv(self.params.T_H, self.params.T_W, T_C1, T_C2,
                                1, 1, self.params.P_C1, self.params.P_C2),
        self.num_cycle_dws_conv(self.params.T_H, self.params.T_W, T_C2,
                                self.params.P_H, self.params.P_W, self.params.P_C2),
        self.num_cycle_std_conv(self.params.T_H, self.params.T_W, T_C2, T_FF,
                                self.params.P_H, self.params.P_W,
                                self.params.P_C2, self.params.P_FF),
    ]
    num_cycles[0] = int(math.ceil(num_cycles[0] / (block.S ** 2)))

    total_num_cycle = np.max(num_cycles)
    # print(num_cycles, total_num_cycle)
    total_data = (self.params.T_H * self.params.T_W * (T_C1 + T_FF) +
                  T_C1 * T_C2 + T_C2 * (self.params.K ** 2) + T_C2 * T_FF)

    return np.array([total_num_cycle, total_data])

  def run_depthwise_separable_block(self, block: DepthwiseSeparableBlock):
    if not isinstance(block, DepthwiseSeparableBlock):
      raise TypeError('block should be a DepthwiseSeparableBlock instance, got %s' %
                      type(block))

    num_tile = [int_ceil(block.layers[0].H, self.params.T_H),
                int_ceil(block.layers[0].W, self.params.T_W),
                int_ceil(block.layers[0].C, self.params.T_C),
                int_ceil(block.layers[1].F, self.params.T_F)]
    total_num_tiles = np.prod(num_tile)

    result = self.run_depthwise_separable_block_tile(block) * total_num_tiles
    return result

  def run_separable_bottleneck_block(self, block: SeparableBottleneckBlock):
    if not isinstance(block, SeparableBottleneckBlock):
      raise TypeError('block should be a SeparableBottleneckBlock instance, got %s' %
                      type(block))

    num_tile = [int_ceil(block.layers[0].H, self.params.T_H),
                int_ceil(block.layers[0].W, self.params.T_W),
                int_ceil(block.layers[0].C, self.params.T_C1),
                int_ceil(block.layers[1].C, self.params.T_C2),
                int_ceil(block.layers[2].F, self.params.T_FF)]
    total_num_tiles = np.prod(num_tile)

    result = self.run_separable_bottleneck_block_tile(block) * total_num_tiles
    return result

  def run_conv_block(self, block: ConvBlock):
    if isinstance(block, BottleneckBlock):
      return self.run_bottleneck_block(block)
    if isinstance(block, DepthwiseSeparableBlock):
      return self.run_depthwise_separable_block(block)
    if isinstance(block, SeparableBottleneckBlock):
      return self.run_separable_bottleneck_block(block)

    raise RuntimeError("Type of block %s is not supported: %s" %
                       (block.name, type(block)))

  def run_layer(self, layer: BaseLayer):
    if isinstance(layer, ConvLayer):
      return self.run_conv(layer)
    if isinstance(layer, MatmulLayer):
      return self.run_matmul(layer)
    if isinstance(layer, ConvBlock):
      return self.run_conv_block(layer)

    raise RuntimeError("Type of layer %s is not supported: %s" %
                       (layer.name, type(layer)))

  def run(self, net: ConvNet):
    # print(self.params.__dict__)
    results = []
    for layer in net.layers:
      result = self.run_layer(layer)

      results.append(result)

    total_results = np.sum(results, axis=0)
    # print('GFLOPS = %10.6f CTC = %10.6f' %
    #       (total_results[0] * 1e-9, total_results[1]))

    return total_results, results

  @property
  def resource(self):
    num_ALM = 2500 * self.params.P_C + 1000 * self.params.P_F

    if self.params.block_type == 'Bottleneck':
      mem_elems = np.array([(self.params.T_C1 * self.params.T_H * self.params.T_W),
                            (self.params.T_W * self.params.K * self.params.P_C2),
                            (self.params.T_C2 * self.params.T_H * self.params.T_W),
                            (self.params.T_C3 * self.params.T_H * self.params.T_W),
                            (self.params.P_FF * self.params.T_H * self.params.T_W)])
    elif self.params.block_type == 'DepthwiseSeparable':
      mem_elems = np.array([(self.params.T_C * self.params.T_H * self.params.T_W),
                            (self.params.T_W * self.params.K * self.params.P_C1),
                            (self.params.P_FF * self.params.T_H * self.params.T_W)])
    elif self.params.block_type == 'SeparableBottleneck':
      mem_elems = np.array([(self.params.T_C1 * self.params.T_H * self.params.T_W),
                            (self.params.T_W * self.params.K * self.params.P_C2),
                            (self.params.T_C2 * self.params.T_H * self.params.T_W),
                            (self.params.T_C2 * self.params.T_H * self.params.T_W),
                            (self.params.P_FF * self.params.T_H * self.params.T_W)])
    else:
      mem_elems = np.array([(self.params.T_C * self.params.T_H * self.params.T_W),
                            (self.params.T_W * self.params.K * self.params.P_C),
                            (self.params.P_F * self.params.T_H * self.params.T_W)])
    mem_size = mem_elems * BIT_WIDTH

    if self.params.use_winograd:
      if self.params.block_type == 'Bottleneck':
        num_mult = (self.params.P_C1 * self.params.P_C2 +
                    self.params.P_C2 * self.params.P_C3 * 36 +
                    self.params.P_C3 * self.params.P_FF)
      elif self.params.block_type == 'DepthwiseSeparable':
        num_mult = (self.params.P_C1 * self.params.P_FF +
                    self.params.P_C1 * 36)
      elif self.params.block_type == 'SeparableBottleneck':
        num_mult = (self.params.P_C1 * self.params.P_C2 +
                    self.params.P_C2 * 36 +
                    self.params.P_C2 * self.params.P_FF)
      else:
        num_mult = (self.params.P_C * self.params.P_F * 36)
    else:
      num_mult = (self.params.P_C * self.params.P_W *
                  self.params.P_F * (self.params.K ** 2))

    return DesignResourceUsage(ALM=num_ALM,
                               BRAM=np.sum([StratixVPlatform.get_BRAM(s)
                                            for s in mem_size]),
                               DSP=StratixVPlatform.get_DSP(num_mult, BIT_WIDTH))
