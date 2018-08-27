r"""
We implement a revised convolution layer implementation in this
file.
"""

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn


@add_arg_scope
def conv2d(inputs,
           num_outputs,
           kernel_size,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           activation_fn=None,
           normalizer_fn=None,
           normalizer_params=None,
           weights_initializer=initializers.xavier_initializer(),
           weights_regularizer=None,
           biases_initializer=init_ops.zeros_initializer(),
           biases_regularizer=None,
           reuse=None,
           use_mask=False,
           variables_collections=None,
           outputs_collections=None,
           trainable=True,
           scope=None):
  """ This conv2d supports weights masking """

  with variable_scope.variable_scope(
          scope, 'Conv2D', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)

    df = ('channels_first'
          if data_format and data_format.startswith('NC') else 'channels_last')
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    num_filters_in = utils.channel_dimension(
        inputs.get_shape(), df, min_rank=4)
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')

    weights_shape = [kernel_h, kernel_w, num_filters_in, num_outputs]
    weights = variables.model_variable(
        'weights',
        shape=weights_shape,
        dtype=dtype,
        initializer=weights_initializer,
        regularizer=weights_regularizer,
        trainable=trainable,
        collections=weights_collections)
    strides = [1, stride_h, stride_w, 1]

    if use_mask:
      mask_collections = utils.get_variable_collections(
          variables_collections, 'mask')
      mask = variables.model_variable(
          'mask',
          shape=weights.shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          trainable=False,
          collections=mask_collections)

      masked_weights = math_ops.multiply(mask, weights,
                                         name='weights/masked_weights')
      weights = masked_weights

    outputs = nn.conv2d(
        inputs, weights, strides, padding, data_format=data_format)

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable(
            'biases',
            shape=[num_outputs],
            dtype=dtype,
            initializer=biases_initializer,
            regularizer=biases_regularizer,
            trainable=trainable,
            collections=biases_collections)
        outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def grouped_conv2d(inputs,
                   num_outputs,
                   kernel_size,
                   num_groups,
                   append_pointwise_conv=False,
                   stride=1,
                   padding='SAME',
                   data_format='NHWC',
                   activation_fn=None,
                   normalizer_fn=None,
                   normalizer_params=None,
                   weights_initializer=initializers.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=init_ops.zeros_initializer(),
                   biases_regularizer=None,
                   reuse=None,
                   variables_collections=None,
                   outputs_collections=None,
                   trainable=True,
                   scope=None):

  with variable_scope.variable_scope(
          scope, 'GroupedConv2D', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)

    assert data_format == 'NHWC'
    df = ('channels_first'
          if data_format and data_format.startswith('NC') else 'channels_last')
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    num_filters_in = utils.channel_dimension(
        inputs.get_shape(), df, min_rank=4)
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')
    strides = [1, stride_h, stride_w, 1]

    assert num_filters_in % num_groups == 0
    assert num_outputs % num_groups == 0

    weights_shape = [kernel_h, kernel_w,
                     int(num_filters_in / num_groups),
                     int(num_outputs / num_groups)]
    grouped_inputs = array_ops.split(inputs, num_or_size_splits=num_groups, axis=3)
    grouped_outputs = []

    for group_id in range(num_groups):
      weights = variables.model_variable(
          'grouped_weights_%d' % group_id,
          shape=weights_shape,
          dtype=dtype,
          initializer=weights_initializer,
          regularizer=weights_regularizer,
          trainable=trainable,
          collections=weights_collections)

      grouped_outputs.append(
          nn.conv2d(
            grouped_inputs[group_id],
            weights, strides, padding, data_format=data_format))

    outputs = array_ops.concat(grouped_outputs, axis=3)

    if append_pointwise_conv:
      pointwise_weights = variables.model_variable(
          'pointwise_weights',
          shape=[1, 1, outputs.shape[3], outputs.shape[3]],
          dtype=dtype,
          initializer=weights_initializer,
          regularizer=weights_regularizer,
          trainable=trainable,
          collections=weights_collections)
      outputs = nn.conv2d(outputs, pointwise_weights, [1, 1, 1, 1], padding='VALID')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable(
            'biases',
            shape=[num_outputs],
            dtype=dtype,
            initializer=biases_initializer,
            regularizer=biases_regularizer,
            trainable=trainable,
            collections=biases_collections)
        outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
