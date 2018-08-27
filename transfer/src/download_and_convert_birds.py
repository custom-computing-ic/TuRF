""" Download and convert Caltech-UCSD Birds-200-2011 dataset
to TensorFlow TFRecords file.
"""


import os
import sys
import math
import numpy as np
import tensorflow as tf

import dataset_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory of the dataset')

_NUM_SHARDS = 5


class ImageReader(object):
  """ A helper class to read image through TensorFlow utilities """

  def __init__(self):
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(
        self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)

    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """ Get the paths for train and test images, and the corresponding class ids
  of each file.

  Args:
    dataset_dir: The path to the dataset directory.
  Returns:
    Two lists of train and test filenames, and a dictionary to map filename to
    class id
  """

  images_file = os.path.join(dataset_dir, "images.txt")
  labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
  train_test_split_file = os.path.join(dataset_dir, "train_test_split.txt")
  # classes = os.path.join(dataset_dir, "classes.txt")

  image_id_to_filenames = dict(np.genfromtxt(images_file, dtype=None))
  image_id_to_class_ids = dict(np.genfromtxt(labels_file, dtype=None))

  train_filenames = []
  test_filenames = []
  filename_to_class_ids = {}

  for img_id, is_train in np.genfromtxt(train_test_split_file, dtype=None):
    rel_filename = image_id_to_filenames[img_id].decode()
    filename = os.path.join(dataset_dir, 'images', rel_filename)
    if is_train == 1:
      train_filenames.append(filename)
    else:
      test_filenames.append(filename)

    basename = os.path.basename(rel_filename)
    filename_to_class_ids[basename] = image_id_to_class_ids[img_id]

  return train_filenames, test_filenames, filename_to_class_ids


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'birds_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)

  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, filename_to_class_ids, dataset_dir):
  """ Convert the given list of files to a TFRecord dataset.

  Args:
    split_name: The name of the dataset split, either 'train' or 'test'.
    filenames: A list of paths to image files
    filename_to_class_ids: A dictionary from file name to its class id. 
    dataset_dir: The path to the dataset directory.
  """

  assert split_name in ['train', 'test']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
  # shuffle file names
  np.random.shuffle(filenames)

  with tf.Graph().as_default():
    image_reader = ImageReader()

    # just run on CPU
    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_idx = shard_id * num_per_shard
          end_idx = min((shard_id+1) * num_per_shard, len(filenames))

          for i in range(start_idx, end_idx):
            # write in a progress style
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # read file
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            basename = os.path.basename(filenames[i])
            class_id = filename_to_class_ids[basename] - 1

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def run(dataset_dir):
  """ Runs the download and convert process.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """

  if not tf.gfile.Exists(dataset_dir):
    raise ValueError('You must specify an existing directory since'
                     'we do not support download at the moment.')


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError(
        'You must supply the dataset directory with --dataset_dir')

  train_filenames, test_filenames, filename_to_class_ids = \
      _get_filenames_and_classes(FLAGS.dataset_dir)

  print('Train files: %d' % len(train_filenames))
  print('Test files: %d' % len(test_filenames))

  # convert dataset
  _convert_dataset('train', train_filenames, filename_to_class_ids,
                   FLAGS.dataset_dir)
  _convert_dataset('test', test_filenames, filename_to_class_ids,
                   FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run(main)
