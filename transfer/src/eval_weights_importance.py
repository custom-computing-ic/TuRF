r"""
This script tries to evaluate the importance of
different connections of a single convolution layer
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import dataset_factory
import preprocessing_factory
import train_test_utils
from nets import network_factory

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The path to the dataset directory')
tf.app.flags.DEFINE_string('dataset_name', None,
                           'Name of the dataset to be evaluated')
tf.app.flags.DEFINE_string('dataset_split_name', None,
                           'Name of the dataset split to be evaluated')
tf.app.flags.DEFINE_string('model_name', None,
                           'Name of the model to be trained')
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'The path to the original checkpoint')
tf.app.flags.DEFINE_string('eval_dir', None,
                           'The path to the directory that logs evaluation')
tf.app.flags.DEFINE_string('conv_scope', None,
                           'The scope of the convolution layer to be evaluated')
tf.app.flags.DEFINE_string('mask_channels', None,
                           'A in_channel,out_channel tuple')


def get_mask_tensor(conv_scope, graph):
  return graph.get_tensor_by_name('/'.join([conv_scope, 'mask']) + ':0')


def get_mask_assign_op(conv_scope, in_channel, out_channel, graph):
  mask_tensor = get_mask_tensor(conv_scope, graph)
  shape = mask_tensor.shape
  mask_to_assign = np.ones(shape, dtype=np.float32)
  mask_to_assign[:, :, in_channel, out_channel] = np.zeros(shape[:2])

  return tf.assign(mask_tensor, tf.constant(mask_to_assign))


def evaluate(model_name,
             dataset_name,
             dataset_dir,
             dataset_split_name,
             eval_dir,
             checkpoint_path,
             conv_scope,
             batch_size=32,
             in_channel=0,
             out_channel=0,
             is_training=False):
  """
  Evaluate a single conv_scope
  """

  with tf.Graph().as_default():
    g = tf.get_default_graph()

    dataset = dataset_factory.get_dataset(
        dataset_name, dataset_split_name, dataset_dir)
    num_batches = math.ceil(dataset.num_samples / batch_size)

    preprocessing_fn = preprocessing_factory.get_preprocessing(
        model_name, is_training=is_training)

    network_fn = network_factory.get_network_fn(
        model_name, dataset.num_classes, use_mask=True,
        is_training=is_training)

    images, _, labels = train_test_utils.load_batch(
        dataset, preprocessing_fn, is_training=is_training, batch_size=batch_size,
        shuffle=False)

    logits, end_points = network_fn(images)
    predictions = tf.argmax(logits, 1)

    mask_assign_op = get_mask_assign_op(conv_scope, in_channel, out_channel, g)

    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=one_hot_labels)

    # Evaluation
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/accuracy': tf.metrics.accuracy(predictions, labels),
        'eval/recall_5': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 5),
        'eval/mean_loss': tf.metrics.mean(loss),
    })

    mask_variables = [var for var in slim.get_model_variables()
                      if 'mask' in var.op.name]

    variables_to_restore = slim.get_variables_to_restore()
    variables_to_restore = [x for x in variables_to_restore
                            if 'mask' not in x.op.name]
    restorer = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      tf.summary.FileWriter(eval_dir, sess.graph)

      restorer.restore(
          sess, tf.train.latest_checkpoint(checkpoint_path))

      sess.run(tf.local_variables_initializer())
      sess.run(tf.variables_initializer(mask_variables))
      sess.run(mask_assign_op)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for batch_id in range(num_batches):
        if batch_id != 0 and batch_id % 10 == 0:
          tf.logging.info('Evaluated [%5d/%5d]' % (batch_id, num_batches))

        # run accuracy evaluation
        sess.run(list(names_to_updates.values()))

      metric_values = sess.run(list(names_to_values.values()))
      for metric, value in zip(names_to_values.keys(), metric_values):
        print('Metric %s has value: %f' % (metric, value))

      coord.request_stop()
      coord.join()


def main(_):
  tf.set_random_seed(42)
  tf.logging.set_verbosity(tf.logging.INFO)

  in_channel, out_channel = [int(x) for x in FLAGS.mask_channels.split(',')]

  evaluate(FLAGS.model_name,
           FLAGS.dataset_name,
           FLAGS.dataset_dir,
           FLAGS.dataset_split_name,
           FLAGS.eval_dir,
           FLAGS.checkpoint_path,
           FLAGS.conv_scope,
           in_channel=in_channel,
           out_channel=out_channel,
           is_training=False)


if __name__ == '__main__':
  tf.app.run(main)
