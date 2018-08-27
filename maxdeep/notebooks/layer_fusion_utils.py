import math
import numpy as np
from layer import ConvLayer

FILTER = "filter"
CHANNEL = "channel"


class BaseConvSimDesign(object):
  def __init__(self, layer, P_W, P_C, P_F, seq, is_buf_input=True):
    self.layer = layer
    self.p_f = P_F
    self.p_c = P_C
    self.p_w = P_W
    self.prev = None
    self.next = None
    self.seq = seq
    self.tick = 0
    self.is_buf_input = is_buf_input

    self.H = int(self.layer.H)
    self.W = int(self.layer.W / P_W)
    self.C = int(self.layer.C / P_C)
    if self.layer.F:
      self.F = int(self.layer.F / P_F)
    else:
      self.F = None

  @property
  def state(self):
    pass

  @property
  def input_addr(self):
    pass

  @property
  def output_addr(self):
    pass


class DepthwiseConvSimDesign(BaseConvSimDesign):
  @property
  def state(self):
    w = self.tick % self.W
    h = int(math.floor(self.tick / self.W)) % self.H
    c = int(math.floor(self.tick / self.W / self.H)) % self.C

    return c, h, w


class PointwiseConvSimDesign(BaseConvSimDesign):
  @property
  def state(self):
    w = self.tick % self.W
    h = int(math.floor(self.tick / self.W)) % self.H
    if self.seq == FILTER:
      c = int(math.floor(self.tick / self.W / self.H)) % self.C
      f = int(math.floor(self.tick / self.W / self.H / self.C)) % self.F
    elif self.seq == CHANNEL:
      f = int(math.floor(self.tick / self.W / self.H)) % self.F
      c = int(math.floor(self.tick / self.W / self.H / self.F)) % self.C

    return f, c, h, w

  @property
  def input_addr(self):
    f, c, h, w = self.state
    if self.is_buf_input:
      if self.seq == CHANNEL:
        return h * self.W + w
      elif self.seq == FILTER:
        return c * self.H * self.W + h * self.W + w
    else:
      if self.prev:
        if self.prev.seq == FILTER:
          return h * self.W + w
        elif self.prev.seq == CHANNEL:
          return c * self.H * self.W + h * self.W + w
      else:
        if self.seq == CHANNEL:
          return h * self.W + w
        elif self.seq == FILTER:
          return c * self.H * self.W + h * self.W + w

  @property
  def output_addr(self):
    f, c, h, w = self.state
    if self.is_buf_input:
      if self.next:
        if self.next.seq == CHANNEL:
          return h * self.W + w
        elif self.next.seq == FILTER:
          return f * self.H * self.W + h * self.W + w
      else:
        if self.seq == FILTER:
          return h * self.W + w
        elif self.seq == CHANNEL:
          return f * self.H * self.W + h * self.W + w
    else:
      if self.seq == FILTER:
        return h * self.W + w
      elif self.seq == CHANNEL:
        return f * self.H * self.W + h * self.W + w

  @property
  def OH(self):
    return int(self.H)

  @property
  def OW(self):
    return int(self.layer.W / self.p_w)

  @property
  def done(self):
    f, c, h, w = self.state
    return f == self.F - 1 and c == self.C - 1 and h == self.H - 1 and w == self.W - 1

  @property
  def output_partial_done(self):
    _, c, _, _ = self.state
    return c == self.C - 1

  @property
  def input_partial_done(self):
    f, _, _, _ = self.state
    return f == self.F - 1


class StandardConvSimDesign(BaseConvSimDesign):
  @property
  def OH(self):
    return int(self.H - self.layer.K + 1)

  @property
  def OW(self):
    return int((self.layer.W - self.layer.K + 1) / self.p_w)

  def out_h(self, h):
    return int(0 if h <= self.layer.K - 1 else h - self.layer.K + 1)

  def out_w(self, w):
    return int(0 if w * self.p_w < self.layer.K - 1
               else (w * self.p_w - self.layer.K + 1) / self.p_w)

  @property
  def state(self):
    w = self.tick % self.W
    h = int(math.floor(self.tick / self.W)) % self.H
    if self.seq == FILTER:
      c = int(math.floor(self.tick / self.W / self.H)) % self.C
      f = int(math.floor(self.tick / self.W / self.H / self.C)) % self.F
    elif self.seq == CHANNEL:
      f = int(math.floor(self.tick / self.W / self.H)) % self.F
      c = int(math.floor(self.tick / self.W / self.H / self.F)) % self.C

    return f, c, h, w

  @property
  def input_addr(self):
    f, c, h, w = self.state
    if self.is_buf_input:
      if self.seq == CHANNEL:
        return h * self.W + w
      elif self.seq == FILTER:
        return c * self.H * self.W + h * self.W + w
    else:
      if self.prev:
        if self.prev.seq == FILTER:
          return h * self.W + w
        elif self.prev.seq == CHANNEL:
          return c * self.H * self.W + h * self.W + w
      else:
        if self.seq == CHANNEL:
          return h * self.W + w
        elif self.seq == FILTER:
          return c * self.H * self.W + h * self.W + w

  @property
  def output_addr(self):
    f, c, h, w = self.state
    out_h = self.out_h(h)
    out_w = self.out_w(w)
    if self.is_buf_input:  # output depends on the next
      if self.next:
        if self.next.seq == CHANNEL:
          return out_h * self.OW + out_w
        elif self.next.seq == FILTER:
          return f * self.OH * self.OW + out_h * self.OW + out_w
      else:
        if self.seq == FILTER:
          return out_h * self.OW + out_w
        elif self.seq == CHANNEL:
          return f * self.OH * self.OW + out_h * self.OW + out_w
    else:
      if self.seq == FILTER:
        return out_h * self.OW + out_w
      elif self.seq == CHANNEL:
        return f * self.OH * self.OW + out_h * self.OW + out_w

  @property
  def done(self):
    f, c, h, w = self.state
    return f == self.F - 1 and c == self.C - 1 and h == self.H - 1 and w == self.W - 1

  @property
  def output_partial_done(self):
    _, c, h, w = self.state
    return c == self.C - 1 and h >= self.layer.K - 1 and w * self.p_w >= (self.layer.K + self.p_w - 2)

  @property
  def input_partial_done(self):
    f, _, _, _ = self.state
    return f == self.F - 1


def make_conv_sim_design(layer, param, seq, is_buf_input=True):
  if layer.T == 'depthwise':
    return DepthwiseConvSimDesign(layer, param['P_W'], param['P_C'], param['P_C'],
                                  seq, is_buf_input=is_buf_input)
  if layer.T == 'pointwise':
    return PointwiseConvSimDesign(layer, param['P_W'], param['P_C'], param['P_F'],
                                  seq, is_buf_input=is_buf_input)
  if layer.T == 'standard':
    return StandardConvSimDesign(layer, param['P_W'], param['P_C'], param['P_F'],
                                 seq, is_buf_input=is_buf_input)


class Buffer(object):
  def __init__(self, depth, width):
    self.state = np.vstack([np.zeros(depth, dtype=np.bool),
                            np.ones(depth, dtype=np.bool)]).T
    self.size = width * depth

  def readable(self, addr):
    return self.state[addr][0]

  def writable(self, addr):
    return self.state[addr][1]

  def set_readable(self, addr):
    self.state[addr][0] = True

  def set_writable(self, addr):
    self.state[addr][1] = True

  def set_not_readable(self, addr):
    self.state[addr][0] = False

  def set_not_writable(self, addr):
    self.state[addr][1] = False


def get_output_buffer_depth(design, is_buf_input=True):
  if is_buf_input:
    if design.next.seq == CHANNEL:
      return int(design.OH * design.OW)
    if design.next.seq == FILTER:
      return int(design.OH * design.OW * design.F)
  else:
    if design.seq == FILTER:
      return int(design.OH * design.OW)
    if design.seq == CHANNEL:
      return int(design.OH * design.OW * design.F)


def get_output_buffer_width(design):
  return design.p_f * design.p_w


def run_dataflow_sim(N, block, params, seqs, is_buf_input=True):
  """
  :param N: batch size
  :param block: list of layers in a block
  :param params: hardware configure parameters
  :param seqs: computation sequence
  :return: ticks that finish computation of batches
  """

  # set up block designs
  block_design = [make_conv_sim_design(layer, params[layer.name], seqs[idx], is_buf_input=is_buf_input)
                  for idx, layer in enumerate(block)]
  for i in range(len(block)):
    if i < len(block) - 1:
      block_design[i].next = block_design[i+1]
    if i > 0:
      block_design[i].prev = block_design[i-1]

  # initialise buffers
  buffers = [Buffer(get_output_buffer_depth(design, is_buf_input=is_buf_input),
                    get_output_buffer_width(design))
             for design in block_design[:-1]]
  buffer_size = np.sum([b.size for b in buffers])

  batch = 0
  global_tick = 0
  output_ticks = []

  while True:
    states = [d.state for d in block_design]
    input_addrs = [d.input_addr for d in block_design]
    input_partial_dones = [d.input_partial_done for d in block_design]
    output_addrs = [d.output_addr for d in block_design]
    output_partial_dones = [d.output_partial_done for d in block_design]

    for idx, design in enumerate(block_design):
      state = states[idx]
      input_addr = input_addrs[idx]
      output_addr = output_addrs[idx]
      input_partial_done = input_partial_dones[idx]
      output_partial_done = output_partial_dones[idx]

      # print("DESIGN #%3d: <%20s, %5d, %5d>" %
      #       (idx, state, input_addr, output_addr))

      runnable = True
      if design.prev:
        runnable = runnable and buffers[idx - 1].readable(input_addr)
      if design.next:
        runnable = runnable and buffers[idx].writable(output_addr)

      if runnable:
        # print("RUNNABLE")
        if design.next and output_partial_done:
          buffers[idx].set_readable(output_addr)
          buffers[idx].set_not_writable(output_addr)
        if design.prev and input_partial_done:
          buffers[idx-1].set_not_readable(input_addr)
          buffers[idx-1].set_writable(input_addr)

        design.tick += 1

    if block_design[-1].done:
      output_ticks.append(global_tick)
      if batch == N - 1:
        break
      batch += 1

    global_tick += 1

  return output_ticks, buffer_size
