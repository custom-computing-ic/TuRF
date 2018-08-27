r"""
This is the script that can launch fine-tuned training for different models
and different configurations.
"""

import os
# add visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
# add slim to PYTHONPATH
sys.path.append(os.path.join(
    os.path.dirname(__file__), '../../models/research/slim'))

import numpy as np
import tensorflow as tf
import dataset_birds
import train_test_utils
import dataset_factory
from nets import network_factory

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('trainable_scopes', None,
                           'Variable scopes that are trainable')
tf.app.flags.DEFINE_integer('num_epochs', 100,
                            'Number of total training epochs')
tf.app.flags.DEFINE_string('fine_tuning_mode', 'single',
                           'Mode for fine-tuning: "iterative", "single"')
tf.app.flags.DEFINE_integer('start_index', 1,
                            'The layer index of the first layer to be fine-tuned')
tf.app.flags.DEFINE_string('model_name', 'vgg_16',
                           'Name of the model to be trained and evaluated')
tf.app.flags.DEFINE_string('dataset_name', 'birds',
                           'Name of the dataset to be evaluated')

# some constants
DATASET_DIR = '/mnt/data2/rz3515/datasets/caltech_ucsd_birds_200_2011/CUB_200_2011/'
TRAIN_DIR = '/mnt/data2/rz3515/train/vgg_16_birds/vgg_16_birds%s'
CHECKPOINT_DIR = '/mnt/data2/rz3515/checkpoints'


def run(trainable_scopes):
  # create a suffix
  encoded_suffix = '_fine_tune_%d' % len(trainable_scopes)

  # run the training function
  train_test_utils.train_eval(dataset_birds,
                              DATASET_DIR,
                              TRAIN_DIR % encoded_suffix,
                              trainable_scopes=trainable_scopes,
                              checkpoint_exclude_scopes=['vgg_16/fc8'],
                              max_number_of_epochs=FLAGS.num_epochs,
                              checkpoint_path=CHECKPOINT_DIR + '/vgg_16.ckpt')


def main(_):
  """ Iterate through every layers in the VGG-16 model and try to fine-tune until converge.
  """

  # parse the trainable scopes
  trainable_scopes = []
  if FLAGS.trainable_scopes:
    trainable_scopes = [scope.strip()
                        for scope in FLAGS.trainable_scopes.split(',')]

  # parse fine-tuning mode
  if FLAGS.fine_tuning_mode == 'single':
    run(trainable_scopes)
  elif FLAGS.fine_tuning_mode == 'iterative':
    # we will iterate through all trainable scopes
    for i in range(FLAGS.start_index - 1, len(trainable_scopes)):
      curr_trainable_scopes = trainable_scopes[-(i+1):]

      run(curr_trainable_scopes)
  else:
    raise ValueError('Unrecognised fine_tuning_mode: %s'
                     % FLAGS.fine_tuning_mode)


if __name__ == '__main__':
  tf.app.run(main)
