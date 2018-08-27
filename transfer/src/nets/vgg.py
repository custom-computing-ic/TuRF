# A VGG-16 implementation that can support more extensions
# than the original implementation

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import conv2d


def _create_bipartite_weights_mask(bipartite_connections, weights_shape):
  """ Create a mask for weights based on the bipartite connection
  """

  K, NC, NF = weights_shape[1:]
  GC, GF = bipartite_connections.shape
  NC_, NF_ = int(NC / GC), int(NF / GF)

  mask = np.ones((K, K, NC, NF), dtype=np.float32)

  for gc in range(GC):
    for gf in range(GF):
      if bipartite_connections[gc, gf]:
        mask[:, :, gc*NC_:(gc+1)*NC_, gf*NF_:(gf+1)*NF_] = 0

  return tf.constant(mask, name='bipartite_weights_mask')


def _replacible_conv2d(inputs, num_outputs, kernels,
                       scope=None,
                       replace=None,
                       ignore_replace=False,
                       append_pointwise_conv=False,
                       bipartite_connections_map=None,
                       weight_decay=5e-4,
                       weight_punishment_factor=1e-3):

  if not replace or scope not in replace or ignore_replace:
    # creating a standard conv2d
    if not replace or scope not in replace:
      return conv2d.conv2d(inputs, num_outputs, kernels, scope=scope)

    def weights_regularizer(weights):
      map_key = '/'.join(['vgg_16', scope, 'weights'])
      mask = _create_bipartite_weights_mask(
          bipartite_connections_map[map_key],
          weights.shape.as_list())
      masked_weights = tf.multiply(mask, weights)

      return (slim.l2_regularizer(weight_decay)(weights) +
              slim.l2_regularizer(weight_punishment_factor)(masked_weights))

    # overwrite the regularizer
    return conv2d.conv2d(inputs, num_outputs, kernels, scope=scope,
                         weights_regularizer=weights_regularizer)

  if replace[scope] == 'separable':
    return slim.separable_conv2d(inputs, num_outputs, kernels,
                                 depth_multiplier=1,
                                 scope=scope)
  if 'grouped' in replace[scope]:
    num_groups = int(replace[scope].split('_')[1])

    return conv2d.grouped_conv2d(
        inputs, num_outputs, kernels,
        num_groups=num_groups,
        append_pointwise_conv=append_pointwise_conv,
        scope=scope)

  raise ValueError('replace %s is unknown' % replace[scope])


def get_replacible_conv2d_fn(replace,
                             append_pointwise_conv=False,
                             bipartite_connections_map=None,
                             ignore_replace=False,
                             weight_decay=5e-4,
                             weight_punishment_factor=5e-4):

  def replacible_conv2d(inputs, num_outputs, kernels, scope):
    return _replacible_conv2d(inputs, num_outputs, kernels,
                              scope=scope,
                              replace=replace,
                              append_pointwise_conv=append_pointwise_conv,
                              ignore_replace=ignore_replace,
                              bipartite_connections_map=bipartite_connections_map,
                              weight_decay=weight_decay,
                              weight_punishment_factor=weight_punishment_factor)

  return replacible_conv2d


def vgg_arg_scope(use_mask=False, weight_decay=5e-4):

  with slim.arg_scope([conv2d.conv2d, conv2d.grouped_conv2d,
                       slim.conv2d, slim.separable_conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([conv2d.conv2d], use_mask=use_mask):
      with slim.arg_scope([conv2d.conv2d, conv2d.grouped_conv2d,
                           slim.conv2d, slim.separable_conv2d],
                          padding='SAME') as arg_sc:
        return arg_sc


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',  # use the same scope
           fc_conv_padding='VALID',
           global_pool=False,
           append_pointwise_conv=False,
           layer_replacement=None,
           ignore_replace=False,
           bipartite_connections_map=None,
           weight_decay=5e-4,
           weight_punishment_factor=5e-4):
  # initialize with an empty dict if not provided
  if not layer_replacement:
    layer_replacement = {}

  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    conv2d_fn = get_replacible_conv2d_fn(
        layer_replacement,
        append_pointwise_conv=append_pointwise_conv,
        ignore_replace=ignore_replace,
        bipartite_connections_map=bipartite_connections_map,
        weight_decay=weight_decay,
        weight_punishment_factor=weight_punishment_factor)

    with slim.arg_scope([slim.conv2d, conv2d.conv2d,
                         slim.fully_connected, slim.max_pool2d,
                         slim.separable_conv2d],
                        outputs_collections=end_points_collection):
      net = conv2d_fn(inputs, 64, [3, 3], scope='conv1/conv1_1')
      net = conv2d_fn(net, 64, [3, 3], scope='conv1/conv1_2')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')

      net = conv2d_fn(net, 128, [3, 3], scope='conv2/conv2_1')
      net = conv2d_fn(net, 128, [3, 3], scope='conv2/conv2_2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')

      net = conv2d_fn(net, 256, [3, 3], scope='conv3/conv3_1')
      net = conv2d_fn(net, 256, [3, 3], scope='conv3/conv3_2')
      net = conv2d_fn(net, 256, [3, 3], scope='conv3/conv3_3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')

      net = conv2d_fn(net, 512, [3, 3], scope='conv4/conv4_1')
      net = conv2d_fn(net, 512, [3, 3], scope='conv4/conv4_2')
      net = conv2d_fn(net, 512, [3, 3], scope='conv4/conv4_3')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')

      net = conv2d_fn(net, 512, [3, 3], scope='conv5/conv5_1')
      net = conv2d_fn(net, 512, [3, 3], scope='conv5/conv5_2')
      net = conv2d_fn(net, 512, [3, 3], scope='conv5/conv5_3')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      net = slim.conv2d(net, 4096, [7, 7],
                        padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None, normalizer_fn=None, scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net

      return net, end_points


vgg_16.default_image_size = 224
