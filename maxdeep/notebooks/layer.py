""" Layer definition """

import numpy as np

STD = "standard"
DPT = "depthwise"
PNT = "pointwise"


class BaseLayer:
  def __init__(self, name):
    self.name = name

  @property
  def num_params(self):
    pass

  @property
  def num_ops(self):
    pass


class ConvLayer(BaseLayer):
  def __init__(self, name, H, W, C, F, K, T, S=1, P=0):
    super().__init__(name)
    self.H = H
    self.W = W
    self.C = C
    self.F = F
    self.K = K
    self.T = T  # type
    self.stride = S
    self.pad = P

  def __str__(self):
    return "%20s: <%3d, %3d, %4d, %4d, %d, %10s>" \
        % (self.name, self.H, self.W, self.C, self.F, self.K, self.T)

  def __repr__(self):
    return self.__str__()

  @property
  def OH(self):
    return (self.H - self.K + 2 * self.pad) / self.stride + 1

  @property
  def OW(self):
    return (self.W - self.K + 2 * self.pad) / self.stride + 1

  @property
  def num_params(self):
    if self.T == STD or self.T == PNT:
      return self.C * self.F * self.K * self.K
    else:
      return self.C * self.K * self.K

  @property
  def num_ops(self):
    if self.T == STD:
      return 2 * self.C * self.F * self.OH * self.OW * self.K * self.K
    if self.T == PNT:
      return 2 * self.C * self.F * self.H * self.W / (self.stride * self.stride)
    if self.T == DPT:
      return 2 * self.C * self.OH * self.OW * self.K * self.K


class MatmulLayer(BaseLayer):
  def __init__(self, name, H, W):
    super().__init__(name)
    self.name = name
    self.num_rows = H
    self.num_cols = W

  def __str__(self):
    return "%20s: <%4d, %4d>" % (self.name, self.num_rows, self.num_cols)

  def __repr__(self):
    return self.__str__()

  @property
  def num_params(self):
    return self.num_cols * self.num_rows

  @property
  def num_ops(self):
    return 2 * self.num_cols * self.num_rows


class ConvBlock(BaseLayer):
  """ Convolution block """

  def __init__(self, name):
    super().__init__(name)

    self.layers = []

  def __str__(self):
    return '\n'.join(['%s' % l for l in self.layers])

  def __repr__(self):
    return self.__str__()

  @property
  def block_type(self):
    pass

  @property
  def num_ops(self):
    return np.sum([l.num_ops for l in self.layers])

  @property
  def num_params(self):
    return np.sum([l.num_params for l in self.layers])


class BottleneckBlock(ConvBlock):

  def __init__(self, name, H, W, C1, C2, F, K, S):
    super().__init__(name)

    self.S = S
    HH = int(H / 2) if S == 2 else H
    WW = int(W / 2) if S == 2 else W

    self.layers = [
        ConvLayer('%s/conv1' % name, H, W, C1, C2, 1, PNT, S=S, P=0),
        ConvLayer('%s/conv2' % name, HH, WW, C2, C2, K, STD, S=1, P=1),
        ConvLayer('%s/conv3' % name, HH, WW, C2, F, 1, PNT, S=1, P=0)
    ]

    if C1 != F:
      self.layers.append(ConvLayer('%s/shortcut' %
                                   self.name, H, W, C1, F, 1, PNT, S=S, P=0))


class DepthwiseSeparableBlock(ConvBlock):

  def __init__(self, name, H, W, C, F, K, S):
    super().__init__(name)

    self.S = S
    HH = int(H / 2) if S == 2 else H
    WW = int(W / 2) if S == 2 else W

    self.layers = [
        ConvLayer('%s/depthwise' % name, H, W, C, C, 3, DPT, S=S, P=1),
        ConvLayer('%s/pointwise' % name, HH, WW, C, F, 1, PNT, S=1, P=0),
    ]


class SeparableBottleneckBlock(ConvBlock):

  def __init__(self, name, H, W, C1, C2, F, K, S):
    super().__init__(name)

    self.S = S
    HH = int(H / 2) if S == 2 else H
    WW = int(W / 2) if S == 2 else W

    self.layers = [
        ConvLayer('%s/conv1' % name, H, W, C1, C2, 1, PNT, S=1, P=0),
        ConvLayer('%s/conv2' % name, H, W, C2, C2, 3, DPT, S=S, P=1),
        ConvLayer('%s/conv3' % name, HH, WW, C2, F, 1, PNT, S=1, P=0)
    ]
