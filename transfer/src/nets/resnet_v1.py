r"""
Residual Network (v1)

https://github.com/tensorflow/models/blob/master/research/slim/nets/
"""

import collections

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  pass


def resnet_v1_block(scope, base_depth, num_units, stride):
  return Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
      }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
      }])


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    with slim.arg_scope(
        [slim.conv2d, bottleneck, stack_blocks_dense],
        outputs_collections=end_points_collection):
      pass 


def resnet_v1_50(inputs,
    num_classes=None,
    is_training=True,
    global_pool=True,
    output_stride=None,
    spatial_squeeze=True,
    store_non_strided_activations=False,
    reuse=None,
    scope='resnet_v1_50'):

  blocks = [
    resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
    resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
    resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
    resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(
      inputs, blocks, num_classes, is_training,
      global_pool=global_pool, output_stride=output_stride,
      include_root_block=True, spatial_squeeze=spatial_squeeze,
      store_non_strided_activations=store_non_strided_activations,
      reuse=reuse, scope=scope)


def resnet_arg_scope(weight_decay=1e-4,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'fused': None
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_params=batch_norm_params,
      normalizer_fn=(slim.batch_norm if use_batch_norm else None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc
