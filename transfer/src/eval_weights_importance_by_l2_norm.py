r"""
This script tries to evaluate the importance of
different connections of a single convolution layer
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import tensorflow.contrib.slim as slim

import dataset_factory
import preprocessing_factory
import train_test_utils
from nets import network_factory

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_name', None,
                           'Name of the model to be trained')
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The path to the dataset directory')
tf.app.flags.DEFINE_string('dataset_name', None,
                           'Name of the dataset to be evaluated')
tf.app.flags.DEFINE_string('dataset_split_name', None,
                           'Name of the dataset split to be evaluated')
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'The path to the original checkpoint')
tf.app.flags.DEFINE_string('conv_scopes', None,
                           'The scope of the convolution layer to be evaluated')
tf.app.flags.DEFINE_integer('num_classes', None,
                            'Number of total classes in the dataset')
tf.app.flags.DEFINE_string('weights_file', None,
                           'Name of the weights file')
tf.app.flags.DEFINE_string('bipartite_connections_file', None,
                           'Name of the bipartite connections file')
tf.app.flags.DEFINE_integer('num_groups', None,
                            'Number of groups in grouped convolution')


def get_weights(model_name,
                dataset_name,
                dataset_dir,
                dataset_split_name,
                checkpoint_path,
                conv_scope=None,
                batch_size=1,
                is_training=False):

  with tf.Graph().as_default():
    dataset = dataset_factory.get_dataset(
        dataset_name, dataset_split_name, dataset_dir)
    preprocessing_fn = preprocessing_factory.get_preprocessing(
        model_name, is_training=is_training)
    network_fn = network_factory.get_network_fn(
        model_name, FLAGS.num_classes)
    images, _, labels = train_test_utils.load_batch(
        dataset, preprocessing_fn, is_training=is_training, batch_size=batch_size,
        shuffle=False)
    logits, end_points = network_fn(images)

    model_vars = slim.get_model_variables()

    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)

    if os.path.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    with tf.Session() as sess:
      restorer.restore(sess, checkpoint_path)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      vals = sess.run(model_vars)
      val_map = {}
      for i, var in enumerate(model_vars):
        if 'conv' in var.op.name:
          val_map[var.op.name] = vals[i]

      coord.request_stop()
      coord.join()

  return val_map


def l2_norm(kernel):
  return np.sqrt(np.sum(np.power(kernel.flatten(), 2)))


def get_bipartite_connections(weights_map,
                              num_groups=None,
                              conv_scopes=None):
  bipartite_connections = {}

  for name, weights in weights_map.items():
    if conv_scopes and name not in conv_scopes:
      continue
    if 'weights' not in name:
      continue
    print('Exporting bipartite matching of %s ...' % name)

    NC_, NF_ = weights.shape[2:]
    if num_groups is not None:
      NC = num_groups
      NF = num_groups
    else:
      NC = NC_
      NF = NF_

    importance_matrix = np.zeros((NC, NF))
    for c in range(NC):
      for f in range(NF):
        if NC == NC_ and NF == NF_:
          importance_matrix[c, f] = l2_norm(weights[:, :, c, f])
        else:
          for ci in range(int(NC_ / NC)):
            for fi in range(int(NF_ / NF)):
              importance_matrix[c, f] += l2_norm(
                  weights[:, :,
                          int(c * NC_ / NC + ci),
                          int(f * NF_ / NF + fi)])

    row_ind, col_ind = linear_sum_assignment(- importance_matrix)
    bipartite_connections[name] = np.zeros((NC, NF), dtype=np.bool)
    bipartite_connections[name][row_ind, col_ind] = True
    bipartite_connections[name + '/importance_matrix'] = importance_matrix

  return bipartite_connections


def eval_perf(bipartite_connections, conv_scopes=None):
  for name in bipartite_connections:
    scope = '/'.join(name.split('/')[1:3])
    if conv_scopes and scope not in conv_scopes:
      continue
    if 'importance_matrix' in name:
      continue

    BC = bipartite_connections[name]
    IM = bipartite_connections[name + '/importance_matrix']

    max_imp = IM[BC].sum()
    total_imp = IM.sum()
    print('%10s %10.5f %10.5f %10.5f' %
          (name, max_imp, total_imp, (total_imp - max_imp) / total_imp))


def main(_):
  tf.set_random_seed(42)
  tf.logging.set_verbosity(tf.logging.INFO)

  # get weights file name
  if FLAGS.weights_file:
    weights_file_name = FLAGS.weights_file
  else:
    weights_file_name = FLAGS.model_name + '.npy'

  # get bipartite file name
  if FLAGS.bipartite_connections_file:
    bipartite_file_name = FLAGS.bipartite_connections_file
  else:
    bipartite_file_name = FLAGS.model_name + '_bipartite.npy'

  # get conv scopes
  if FLAGS.conv_scopes:
    conv_scopes = [s.strip() for s in FLAGS.conv_scopes.split(',')]
  else:
    conv_scopes = None

  # get weights
  if os.path.isfile(weights_file_name):
    weights_map = np.load(weights_file_name).item()
    print('Loaded weights_map from %s' % weights_file_name)
  else:
    weights_map = get_weights(FLAGS.model_name,
                              FLAGS.dataset_name,
                              FLAGS.dataset_dir,
                              FLAGS.dataset_split_name,
                              FLAGS.checkpoint_path,
                              conv_scopes)
    np.save(weights_file_name, weights_map)

  # get bipartite connections file
  if os.path.isfile(bipartite_file_name):
    bipartite_connections = np.load(bipartite_file_name).item()
    print('Loaded bipartite connections from %s' % bipartite_file_name)
  else:
    bipartite_connections = get_bipartite_connections(
        weights_map, num_groups=FLAGS.num_groups)
    np.save(bipartite_file_name, bipartite_connections)

  # get conv scopes
  if FLAGS.conv_scopes:
    conv_scopes = [s.strip() for s in FLAGS.conv_scopes.split(',')]
  else:
    conv_scopes = None

  eval_perf(bipartite_connections, conv_scopes=conv_scopes)


if __name__ == '__main__':
  tf.app.run(main)
