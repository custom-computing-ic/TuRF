r"""
Train a layer-replaced model.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

import train_test_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The path to the dataset directory')
tf.app.flags.DEFINE_string('dataset_name', None,
                           'Name of the dataset to be evaluated')
tf.app.flags.DEFINE_string('model_name', None,
                           'Name of the model to be trained')
tf.app.flags.DEFINE_string('train_dir', None,
                           'The path to the training directory')
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'The path to the original checkpoint')
tf.app.flags.DEFINE_string('trainable_scopes', None,
                           'Scopes that can be trained')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                           'Scopes that should not be assigned from the checkpoint')
tf.app.flags.DEFINE_integer('max_number_of_epochs', 10,
                            'Maximum number of epochs for training')
tf.app.flags.DEFINE_string('separable_conv_scopes', None,
                           'Convolution layers should be implemented in separable_conv2d')
tf.app.flags.DEFINE_string('grouped_conv_scopes', None,
                           'Convolution layers should be replaced with conv2d')
tf.app.flags.DEFINE_integer('num_groups', 2,
                            'Number of groups in grouped convolution')
tf.app.flags.DEFINE_boolean('append_pointwise_conv', False,
                            'Append pointwise convolution after grouped convolution')
tf.app.flags.DEFINE_boolean('init_from_pre_trained', False,
                            'Whether to initialize coefficients from pre-trained models')
tf.app.flags.DEFINE_boolean('init_pointwise_from_pre_trained', False,
                            'Whether to initialize pointwise convolution from pre-trained values')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
tf.app.flags.DEFINE_string('weights_file', None,
                           'Name of the weights file')
tf.app.flags.DEFINE_string('bipartite_connections_file', None,
                           'Name of the bipartite connections file')
tf.app.flags.DEFINE_boolean('ignore_replace', False,
                            'Whether to ignore replacement during creating the network')
tf.app.flags.DEFINE_float('weight_punishment_factor', 1e-3,
                          'Punishment factor')


def split_scopes(scopes=None):
  return [x.strip() for x in scopes.split(',')] if scopes else []


def get_layer_replacement(separable_conv_scopes=None,
                          grouped_conv_scopes=None,
                          num_groups=2):
  layer_replacement = {}

  separable_conv_scopes = split_scopes(separable_conv_scopes)
  grouped_conv_scopes = split_scopes(grouped_conv_scopes)

  for scope in separable_conv_scopes:
    layer_replacement[scope] = 'separable'
  for scope in grouped_conv_scopes:
    layer_replacement[scope] = 'grouped_%d' % num_groups

  return layer_replacement


def main(_):
  """
  Train a given model with some of its layers replaced.
  """

  if not FLAGS.model_name:
    raise ValueError('You should provide --model_name')
  if not FLAGS.dataset_name:
    raise ValueError('You should provide --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('--dataset_dir is missing')
  if not FLAGS.train_dir:
    raise ValueError('--train_dir should be specified')

  layer_replacement = get_layer_replacement(
      FLAGS.separable_conv_scopes, FLAGS.grouped_conv_scopes, FLAGS.num_groups)

  # get weights file name
  if FLAGS.weights_file:
    weights_map_file = FLAGS.weights_file
  else:
    weights_map_file = FLAGS.model_name + '.npy'

  # get bipartite file name
  if FLAGS.bipartite_connections_file:
    bipartite_connections_file = FLAGS.bipartite_connections_file
  else:
    bipartite_connections_file = FLAGS.model_name + '_bipartite.npy'

  if os.path.isfile(weights_map_file):
    weights_map = np.load(weights_map_file).item()
  else:
    weights_map = None

  if os.path.isfile(bipartite_connections_file):
    bipartite_connections_map = np.load(bipartite_connections_file).item()
  else:
    bipartite_connections_map = None

  train_test_utils.train_eval(
      FLAGS.model_name,
      FLAGS.dataset_name,
      FLAGS.dataset_dir,
      FLAGS.train_dir,
      learning_rate=FLAGS.learning_rate,
      max_number_of_epochs=FLAGS.max_number_of_epochs,
      checkpoint_path=FLAGS.checkpoint_path,
      trainable_scopes=FLAGS.trainable_scopes,
      checkpoint_exclude_scopes=FLAGS.checkpoint_exclude_scopes,
      layer_replacement=layer_replacement,
      num_groups=FLAGS.num_groups,
      append_pointwise_conv=FLAGS.append_pointwise_conv,
      ignore_replace=FLAGS.ignore_replace,
      weight_punishment_factor=FLAGS.weight_punishment_factor,
      weights_map=weights_map,
      bipartite_connections_map=bipartite_connections_map,
      init_from_pre_trained=FLAGS.init_from_pre_trained,
      init_pointwise_from_pre_trained=FLAGS.init_pointwise_from_pre_trained)


if __name__ == '__main__':
  tf.app.run(main)
