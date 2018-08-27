import numpy as np

from layer import ConvLayer, STD, DPT, PNT, \
    MatmulLayer, BottleneckBlock, DepthwiseSeparableBlock, SeparableBottleneckBlock


class ConvNet:

  def __init__(self, layers):
    self.layers = layers

  @property
  def num_params(self):
    return np.sum([layer.num_params for layer in self.layers])

  @property
  def num_ops(self):
    return np.sum([layer.num_ops for layer in self.layers])

  def __str__(self):
    return '\n'.join(['%s %10.6f %10.6f' % (layer, layer.num_ops * 1e-9, layer.num_params * 1e-6)
                      for layer in self.layers])

  def __repr__(self):
    return self.__str__()


VGG16 = ConvNet([
    ConvLayer("conv1_1", 224, 224, 3, 64, 3, STD),
    ConvLayer("conv1_2", 224, 224, 64, 64, 3, STD),
    ConvLayer("conv2_1", 112, 112, 64, 128, 3, STD),
    ConvLayer("conv2_2", 112, 112, 128, 128, 3, STD),
    ConvLayer("conv3_1", 56, 56, 128, 256, 3, STD),
    ConvLayer("conv3_2", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv3_3", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv4_1", 28, 28, 256, 512, 3, STD),
    ConvLayer("conv4_2", 28, 28, 512, 512, 3, STD),
    ConvLayer("conv4_3", 28, 28, 512, 512, 3, STD),
    ConvLayer("conv5_1", 14, 14, 512, 512, 3, STD),
    ConvLayer("conv5_2", 14, 14, 512, 512, 3, STD),
    ConvLayer("conv5_3", 14, 14, 512, 512, 3, STD),
    MatmulLayer("fc6", 7 * 7 * 512, 4096),
    MatmulLayer("fc7", 4096, 4096),
    MatmulLayer("fc8", 4096, 1000),
])

VGG16_OPT = ConvNet([
    ConvLayer("conv1_1", 224, 224, 3, 64, 3, STD),
    ConvLayer("conv1_2", 224, 224, 64, 64, 3, STD),
    ConvLayer("conv2_1", 112, 112, 64, 128, 3, STD),
    ConvLayer("conv2_2", 112, 112, 128, 128, 3, STD),
    ConvLayer("conv3_1", 56, 56, 128, 256, 3, STD),
    ConvLayer("conv3_2", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv3_3", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv4_1", 28, 28, 256, 512, 3, STD),
    ConvLayer("conv4_2", 28, 28, 512, 512, 3, STD),
    ConvLayer("conv4_3", 28, 28, 512, 512, 3, STD),
    DepthwiseSeparableBlock("conv5_1", 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_2", 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_3", 14, 14, 512, 512, 3, 1),
    MatmulLayer("fc6", 7 * 7 * 512, 4096),
    MatmulLayer("fc7", 4096, 4096),
    MatmulLayer("fc8", 4096, 1000),
])

VGG16_OPT_V2 = ConvNet([
    ConvLayer("conv1_1", 224, 224, 3, 64, 3, STD),
    ConvLayer("conv1_2", 224, 224, 64, 64, 3, STD),
    ConvLayer("conv2_1", 112, 112, 64, 128, 3, STD),
    ConvLayer("conv2_2", 112, 112, 128, 128, 3, STD),
    ConvLayer("conv3_1", 56, 56, 128, 256, 3, STD),
    ConvLayer("conv3_2", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv3_3", 56, 56, 256, 256, 3, STD),
    DepthwiseSeparableBlock("conv4_1", 28, 28, 256, 512, 3, 1),
    DepthwiseSeparableBlock("conv4_2", 28, 28, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv4_3", 28, 28, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_1", 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_2", 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_3", 14, 14, 512, 512, 3, 1),
    MatmulLayer("fc6", 7 * 7 * 512, 4096),
    MatmulLayer("fc7", 4096, 4096),
    MatmulLayer("fc8", 4096, 1000),
])

VGG16_OPT_V5 = ConvNet([
    DepthwiseSeparableBlock("conv1_1", 224, 224, 3, 64, 3, 1),
    DepthwiseSeparableBlock("conv1_2", 224, 224, 64, 64, 3, 1),
    DepthwiseSeparableBlock("conv2_1", 112, 112, 64, 128, 3, 1),
    DepthwiseSeparableBlock("conv2_2", 112, 112, 128, 128, 3, 1),
    DepthwiseSeparableBlock("conv3_1", 56, 56, 128, 256, 3, 1),
    DepthwiseSeparableBlock("conv3_2", 56, 56, 256, 256, 3, 1),
    DepthwiseSeparableBlock("conv3_3", 56, 56, 256, 256, 3, 1),
    DepthwiseSeparableBlock("conv4_1", 28, 28, 256, 512, 3, 1),
    DepthwiseSeparableBlock("conv4_2", 28, 28, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv4_3", 28, 28, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_1", 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_2", 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock("conv5_3", 14, 14, 512, 512, 3, 1),
    MatmulLayer("fc6", 7 * 7 * 512, 4096),
    MatmulLayer("fc7", 4096, 4096),
    MatmulLayer("fc8", 4096, 1000),
])
RESNET50 = ConvNet([
    ConvLayer('conv2_1/conv1', 56, 56, 64, 64, 1, PNT),
    ConvLayer('conv2_1/conv2', 56, 56, 64, 64, 3, STD, P=1),
    ConvLayer('conv2_1/conv3', 56, 56, 64, 256, 1, PNT),
    ConvLayer('conv2_2/conv1', 56, 56, 256, 64, 1, PNT),
    ConvLayer('conv2_2/conv2', 56, 56, 64, 64, 3, STD, P=1),
    ConvLayer('conv2_2/conv3', 56, 56, 64, 256, 1, PNT),
    ConvLayer('conv2_3/conv1', 56, 56, 256, 64, 1, PNT),
    ConvLayer('conv2_3/conv2', 56, 56, 64, 64, 3, STD, P=1),
    ConvLayer('conv2_3/conv3', 56, 56, 64, 256, 1, PNT),

    ConvLayer('conv3_1/conv1', 56, 56, 256, 128, 1, PNT, S=2),
    ConvLayer('conv3_1/conv2', 28, 28, 128, 128, 3, STD, P=1),
    ConvLayer('conv3_1/conv3', 28, 28, 128, 512, 1, PNT),
    ConvLayer('conv3_2/conv1', 28, 28, 512, 128, 1, PNT),
    ConvLayer('conv3_2/conv2', 28, 28, 128, 128, 3, STD, P=1),
    ConvLayer('conv3_2/conv3', 28, 28, 128, 512, 1, PNT),
    ConvLayer('conv3_3/conv1', 28, 28, 512, 128, 1, PNT),
    ConvLayer('conv3_3/conv2', 28, 28, 128, 128, 3, STD, P=1),
    ConvLayer('conv3_3/conv3', 28, 28, 128, 512, 1, PNT),
    ConvLayer('conv3_4/conv1', 28, 28, 512, 128, 1, PNT),
    ConvLayer('conv3_4/conv2', 28, 28, 128, 128, 3, STD, P=1),
    ConvLayer('conv3_4/conv3', 28, 28, 128, 512, 1, PNT),

    ConvLayer('conv4_1/conv1', 28, 28, 512, 256, 1, PNT, S=2),
    ConvLayer('conv4_1/conv2', 14, 14, 256, 256, 3, STD, P=1),
    ConvLayer('conv4_1/conv3', 14, 14, 256, 1024, 1, PNT),
    ConvLayer('conv4_2/conv1', 14, 14, 1024, 256, 1, PNT),
    ConvLayer('conv4_2/conv2', 14, 14, 256, 256, 3, STD, P=1),
    ConvLayer('conv4_2/conv3', 14, 14, 256, 1024, 1, PNT),
    ConvLayer('conv4_3/conv1', 14, 14, 1024, 256, 1, PNT),
    ConvLayer('conv4_3/conv2', 14, 14, 256, 256, 3, STD, P=1),
    ConvLayer('conv4_3/conv3', 14, 14, 256, 1024, 1, PNT),
    ConvLayer('conv4_4/conv1', 14, 14, 1024, 256, 1, PNT),
    ConvLayer('conv4_4/conv2', 14, 14, 256, 256, 3, STD, P=1),
    ConvLayer('conv4_4/conv3', 14, 14, 256, 1024, 1, PNT),
    ConvLayer('conv4_5/conv1', 14, 14, 1024, 256, 1, PNT),
    ConvLayer('conv4_5/conv2', 14, 14, 256, 256, 3, STD, P=1),
    ConvLayer('conv4_5/conv3', 14, 14, 256, 1024, 1, PNT),
    ConvLayer('conv4_6/conv1', 14, 14, 1024, 256, 1, PNT),
    ConvLayer('conv4_6/conv2', 14, 14, 256, 256, 3, STD, P=1),
    ConvLayer('conv4_6/conv3', 14, 14, 256, 1024, 1, PNT),

    ConvLayer('conv5_1/conv1', 14, 14, 1024, 512, 1, PNT, S=2),
    ConvLayer('conv5_1/conv2', 7, 7, 512, 512, 3, STD, P=1),
    ConvLayer('conv5_1/conv3', 7, 7, 512, 2048, 1, PNT),
    ConvLayer('conv5_2/conv1', 7, 7, 2048, 512, 1, PNT),
    ConvLayer('conv5_2/conv2', 7, 7, 512, 512, 3, STD, P=1),
    ConvLayer('conv5_2/conv3', 7, 7, 512, 2048, 1, PNT),
    ConvLayer('conv5_3/conv1', 7, 7, 2048, 512, 1, PNT),
    ConvLayer('conv5_3/conv2', 7, 7, 512, 512, 3, STD, P=1),
    ConvLayer('conv5_3/conv3', 7, 7, 512, 2048, 1, PNT),

    MatmulLayer('fc', 2048, 1000)
])

MOBILENET_V1 = ConvNet([
    ConvLayer('conv0', 224, 224, 3, 32, 3, STD, S=2, P=1),


    ConvLayer('conv1/depthwise', 112, 112, 32, 32, 3, DPT, S=1, P=1),
    ConvLayer('conv1/pointwise', 112, 112, 32, 64, 1, PNT, S=1, P=0),
    ConvLayer('conv2/depthwise', 112, 112, 64, 64, 3, DPT, S=2, P=1),
    ConvLayer('conv2/pointwise', 56, 56, 64, 128, 1, PNT, S=1, P=0),

    ConvLayer('conv3/depthwise', 56, 56, 128, 128, 3, DPT, S=1, P=1),
    ConvLayer('conv3/pointwise', 56, 56, 128, 128, 1, PNT, S=1, P=0),
    ConvLayer('conv4/depthwise', 56, 56, 128, 128, 3, DPT, S=2, P=1),
    ConvLayer('conv4/pointwise', 28, 28, 128, 256, 1, PNT, S=1, P=0),

    ConvLayer('conv5/depthwise', 28, 28, 256, 256, 3, DPT, S=1, P=1),
    ConvLayer('conv5/pointwise', 28, 28, 256, 256, 1, PNT, S=1, P=0),
    ConvLayer('conv6/depthwise', 28, 28, 256, 256, 3, DPT, S=2, P=1),
    ConvLayer('conv6/pointwise', 14, 14, 256, 512, 1, PNT, S=1, P=0),

    ConvLayer('conv7/depthwise', 14, 14, 512, 512, 3, DPT, S=1, P=1),
    ConvLayer('conv7/pointwise', 14, 14, 512, 512, 1, PNT, S=1, P=0),
    ConvLayer('conv8/depthwise', 14, 14, 512, 512, 3, DPT, S=1, P=1),
    ConvLayer('conv8/pointwise', 14, 14, 512, 512, 1, PNT, S=1, P=0),
    ConvLayer('conv9/depthwise', 14, 14, 512, 512, 3, DPT, S=1, P=1),
    ConvLayer('conv9/pointwise', 14, 14, 512, 512, 1, PNT, S=1, P=0),
    ConvLayer('conv10/depthwise', 14, 14, 512, 512, 3, DPT, S=1, P=1),
    ConvLayer('conv10/pointwise', 14, 14, 512, 512, 1, PNT, S=1, P=0),
    ConvLayer('conv11/depthwise', 14, 14, 512, 512, 3, DPT, S=1, P=1),
    ConvLayer('conv11/pointwise', 14, 14, 512, 512, 1, PNT, S=1, P=0),

    ConvLayer('conv12/depthwise', 14, 14, 512, 512, 3, DPT, S=2, P=1),
    ConvLayer('conv12/pointwise', 7, 7, 512, 1024, 1, PNT, S=1, P=0),
    ConvLayer('conv1k/depthwise', 7, 7, 1024, 1024, 3, DPT, S=1, P=1),
    ConvLayer('conv13/pointwise', 7, 7, 1024, 1024, 1, PNT, S=1, P=0),
    MatmulLayer('fc', 1024, 1000),
])

MOBILENET_V2 = ConvNet([
    ConvLayer('conv0', 224, 224, 3, 32, 3, STD, S=2, P=1),

    ConvLayer('expand_conv/conv1', 112, 112, 32, 32, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv/conv2', 112, 112, 32, 32, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv/conv3', 112, 112, 32, 16, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_1/conv1', 112, 112, 16, 96, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_1/conv2', 112, 112, 96, 96, 3, DPT, S=2, P=1),
    ConvLayer('expand_conv_1/conv3', 56, 56, 96, 24, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_2/conv1', 56, 56, 24, 96, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_2/conv2', 56, 56, 96, 96, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_2/conv3', 56, 56, 96, 24, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_3/conv1', 56, 56, 24, 144, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_3/conv2', 56, 56, 144, 144, 3, DPT, S=2, P=1),
    ConvLayer('expand_conv_3/conv3', 28, 28, 144, 32, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_4/conv1', 28, 28, 32, 192, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_4/conv2', 28, 28, 192, 192, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_4/conv3', 28, 28, 192, 32, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_5/conv1', 28, 28, 32, 192, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_5/conv2', 28, 28, 192, 192, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_5/conv3', 28, 28, 192, 32, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_6/conv1', 28, 28, 32, 192, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_6/conv2', 28, 28, 192, 192, 3, DPT, S=2, P=1),
    ConvLayer('expand_conv_6/conv3', 14, 14, 192, 64, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_7/conv1', 14, 14, 64, 384, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_7/conv2', 14, 14, 384, 384, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_7/conv3', 14, 14, 384, 64, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_8/conv1', 14, 14, 64, 384, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_8/conv2', 14, 14, 384, 384, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_8/conv3', 14, 14, 384, 64, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_9/conv1', 14, 14, 64, 384, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_9/conv2', 14, 14, 384, 384, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_9/conv3', 14, 14, 384, 64, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_10/conv1', 14, 14, 64, 384, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_10/conv2', 14, 14, 384, 384, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_10/conv3', 14, 14, 384, 96, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_11/conv1', 14, 14, 96, 576, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_11/conv2', 14, 14, 576, 576, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_11/conv3', 14, 14, 576, 96, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_12/conv1', 14, 14, 96, 576, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_12/conv2', 14, 14, 576, 576, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_12/conv3', 14, 14, 576, 96, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_13/conv1', 14, 14, 96, 576, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_13/conv2', 14, 14, 576, 576, 3, DPT, S=2, P=1),
    ConvLayer('expand_conv_13/conv3', 7, 7, 576, 160, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_14/conv1', 7, 7, 160, 960, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_14/conv2', 7, 7, 960, 960, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_14/conv3', 7, 7, 960, 160, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_15/conv1', 7, 7, 160, 960, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_15/conv2', 7, 7, 960, 960, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_15/conv3', 7, 7, 960, 160, 1, PNT, S=1, P=0),

    ConvLayer('expand_conv_16/conv1', 7, 7, 160, 960, 1, PNT, S=1, P=0),
    ConvLayer('expand_conv_16/conv2', 7, 7, 960, 960, 3, DPT, S=1, P=1),
    ConvLayer('expand_conv_16/conv3', 7, 7, 960, 320, 1, PNT, S=1, P=0),

    ConvLayer('conv1', 7, 7, 320, 1280, 1, PNT, S=1, P=0),

    MatmulLayer('fc', 1280, 1000),
])

RESNET50_BLOCK = ConvNet([
    ConvLayer('conv1', 224, 224, 3, 64, 7, STD, S=2, P=3),

    BottleneckBlock('res2a', 56, 56, 64, 64, 256, 3, 1),
    BottleneckBlock('res2b', 56, 56, 256, 64, 256, 3, 1),
    BottleneckBlock('res2c', 56, 56, 256, 64, 256, 3, 1),

    BottleneckBlock('res3a', 56, 56, 256, 128, 512, 3, 2),
    BottleneckBlock('res3b', 28, 28, 512, 128, 512, 3, 1),
    BottleneckBlock('res3c', 28, 28, 512, 128, 512, 3, 1),
    BottleneckBlock('res3d', 28, 28, 512, 128, 512, 3, 1),

    BottleneckBlock('res4a', 28, 28, 512, 256, 1024, 3, 2),
    BottleneckBlock('res4b', 14, 14, 1024, 256, 1024, 3, 1),
    BottleneckBlock('res4c', 14, 14, 1024, 256, 1024, 3, 1),
    BottleneckBlock('res4d', 14, 14, 1024, 256, 1024, 3, 1),
    BottleneckBlock('res4e', 14, 14, 1024, 256, 1024, 3, 1),
    BottleneckBlock('res4f', 14, 14, 1024, 256, 1024, 3, 1),

    BottleneckBlock('res5a', 14, 14, 1024, 512, 2048, 3, 2),
    BottleneckBlock('res5b', 7, 7, 2048, 512, 2048, 3, 1),
    BottleneckBlock('res5c', 7, 7, 2048, 512, 2048, 3, 1),

    MatmulLayer('fc', 2048, 1000)
])

MOBILENET_V1_BLOCK = ConvNet([
    ConvLayer('conv0', 224, 224, 3, 32, 3, STD, S=2, P=1),

    DepthwiseSeparableBlock('conv1', 112, 112, 32, 64, 3, 1),
    DepthwiseSeparableBlock('conv2', 112, 112, 64, 128, 3, 2),
    DepthwiseSeparableBlock('conv3', 56, 56, 128, 128, 3, 1),
    DepthwiseSeparableBlock('conv4', 56, 56, 128, 256, 3, 2),
    DepthwiseSeparableBlock('conv5', 28, 28, 256, 256, 3, 1),
    DepthwiseSeparableBlock('conv6', 28, 28, 256, 512, 3, 2),
    DepthwiseSeparableBlock('conv7', 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock('conv8', 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock('conv9', 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock('conv10', 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock('conv11', 14, 14, 512, 512, 3, 1),
    DepthwiseSeparableBlock('conv12', 14, 14, 512, 1024, 3, 2),
    DepthwiseSeparableBlock('conv13', 7, 7, 1024, 1024, 3, 1),

    MatmulLayer('fc', 1024, 1000),
])

MOBILENET_V2_BLOCK = ConvNet([
    ConvLayer('conv0', 224, 224, 3, 32, 3, STD, S=2, P=1),

    SeparableBottleneckBlock('expand_conv', 112, 112, 32, 32, 16, 3, 1),
    SeparableBottleneckBlock('expand_conv_1', 112, 112, 16, 96, 24, 3, 2),
    SeparableBottleneckBlock('expand_conv_2', 56, 56, 24, 96, 24, 3, 1),
    SeparableBottleneckBlock('expand_conv_3', 56, 56, 24, 144, 32, 3, 2),
    SeparableBottleneckBlock('expand_conv_4', 28, 28, 32, 192, 32, 3, 1),
    SeparableBottleneckBlock('expand_conv_5', 28, 28, 32, 192, 32, 3, 1),
    SeparableBottleneckBlock('expand_conv_6', 28, 28, 32, 192, 64, 3, 2),
    SeparableBottleneckBlock('expand_conv_7', 14, 14, 64, 384, 64, 3, 1),
    SeparableBottleneckBlock('expand_conv_8', 14, 14, 64, 384, 64, 3, 1),
    SeparableBottleneckBlock('expand_conv_9', 14, 14, 64, 384, 64, 3, 1),
    SeparableBottleneckBlock('expand_conv_10', 14, 14, 64, 384, 96, 3, 1),
    SeparableBottleneckBlock('expand_conv_11', 14, 14, 96, 576, 96, 3, 1),
    SeparableBottleneckBlock('expand_conv_12', 14, 14, 96, 576, 96, 3, 1),
    SeparableBottleneckBlock('expand_conv_13', 14, 14, 96, 576, 160, 3, 2),
    SeparableBottleneckBlock('expand_conv_14', 7, 7, 160, 960, 160, 3, 1),
    SeparableBottleneckBlock('expand_conv_15', 7, 7, 160, 960, 160, 3, 1),
    SeparableBottleneckBlock('expand_conv_16', 7, 7, 160, 960, 320, 3, 1),

    ConvLayer('conv1', 7, 7, 320, 1280, 1, PNT, S=1, P=0),

    MatmulLayer('fc', 1280, 1000),
])
