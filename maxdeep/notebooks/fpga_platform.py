from utils import int_ceil


class Platform(object):
  @property
  def max_ALM(self):
    pass

  @property
  def max_BRAM(self):
    pass

  @property
  def max_DSP(self):
    pass


class StratixVPlatform(Platform):
  DSP_BIT_WIDTH = 27
  BRAM_BLOCK_SIZE = 20 * 1024

  @property
  def max_ALM(self):
    return 262400

  @property
  def max_BRAM(self):
    return 2567

  @property
  def max_DSP(self):
    return 1963

  @staticmethod
  def get_DSP(num_mult, bit_width):
    return num_mult / int_ceil(StratixVPlatform.DSP_BIT_WIDTH, bit_width)

  @staticmethod
  def get_BRAM(size):
    return int_ceil(size, StratixVPlatform.BRAM_BLOCK_SIZE)
