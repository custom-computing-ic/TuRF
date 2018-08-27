r"""
This file collects utility functions for training and testing in the current
code-base.
"""

import os
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# utilities from research/slim
from nets import network_factory
import dataset_factory
import preprocessing_factory


def load_batch(dataset, preprocessing_fn, batch_size=32, height=224, width=224,
               shuffle=True, is_training=False):
  """ Load batches from the given dataset and pre-process.

  Args:
      dataset: A slim Dataset object
      batch_size: batch size
      height: train image height
      width: train image width
      preprocessing_fn: image pre-processing function
      shuffle: whether to shuffle the slim data provider
      is_training: a boolean indicates whether we are loading a batch for training or not
  Returns:
      A batch contains images, image_raw (for visualisation) and labels.
  """

  # build the data provider
  data_provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=shuffle,
      num_readers=4,
      common_queue_capacity=20 * batch_size,
      common_queue_min=10 * batch_size)

  # load single image and label
  image_raw, label = data_provider.get(['image', 'label'])

  # pre-process
  image = preprocessing_fn(image_raw, height, width)

  # pre-process for display
  image_raw = tf.expand_dims(image_raw, 0)
  image_raw = tf.image.resize_images(image_raw, [height, width])
  image_raw = tf.squeeze(image_raw)

  # build batches
  if is_training:
    images, images_raw, labels = tf.train.shuffle_batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size,
        min_after_dequeue=batch_size)
  else:
    images, images_raw, labels = tf.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)

  return images, images_raw, labels


def configure_learning_rate(learning_rate,
                            num_samples_per_epoch,
                            global_step,
                            learning_rate_decay_factor=0.94,
                            num_epochs_per_decay=10,
                            batch_size=32):
  """ Configure exponential decay for the given learning rate
  """
  decay_steps = int(math.ceil(num_samples_per_epoch / batch_size) *
                    num_epochs_per_decay)

  return tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True,
                                    name='exponential_decay_learning_rate')


def get_latest_step(train_dir):
  latest_checkpoint = tf.train.latest_checkpoint(train_dir)
  if latest_checkpoint is None:
    return 0
  else:
    return int(latest_checkpoint.split('-')[-1])


def get_train_dataset_num_samples(dataset_name, dataset_dir):
  with tf.Graph().as_default():
    dataset = dataset_factory.get_dataset(
        dataset_name, 'train', dataset_dir)

    return dataset.num_samples


def init_from_pre_trained_assign_ops(model_name,
                                     layer_replacement,
                                     variables_to_restore,
                                     init_pointwise_from_pre_trained=False,
                                     weights_map=None,
                                     bipartite_connections_map=None,
                                     num_groups=2):
  assign_ops = []

  if layer_replacement is None:
    return assign_ops

  # create a new variable scope for loading original operators
  if not weights_map:
    with tf.variable_scope('replace'):
      layer_replace_vars = {}

      for scope_name in layer_replacement:
        if ('separable' not in layer_replacement[scope_name] and
                'grouped' not in layer_replacement[scope_name]):
          continue

        var_name = scope_name + '/weights'
        # create a new variable to host the original variable from the model
        var = tf.get_variable(var_name, shape=(3, 3, 512, 512),
                              trainable=False)
        layer_replace_vars[var_name] = var

        # assign the content of the original variable from the checkpoint
        # to layer_replace_vars
        variables_to_restore[model_name + '/' + var_name] = var

  for scope_name in layer_replacement:

    if 'grouped' in layer_replacement[scope_name]:
      if not weights_map:
        src_var_name = scope_name + '/weights'
        src_var = layer_replace_vars[src_var_name]

        for group_id in range(num_groups):
          dst_var_name = 'grouped_weights_%d' % group_id
          dst_var = slim.get_unique_variable(
              '/'.join([model_name, scope_name, dst_var_name]))

          group_in_channel = dst_var.shape[2]
          group_out_channel = dst_var.shape[3]
          assign_op = tf.assign(
              dst_var,
              src_var[
                  :, :,
                  (group_in_channel * group_id):(group_in_channel * (group_id + 1)),
                  (group_out_channel * group_id):(group_out_channel * (group_id + 1))])
          assign_ops.append(assign_op)

      else:
        map_key = '/'.join([model_name, scope_name, 'weights'])
        bipartite_connections = bipartite_connections_map[map_key]
        print(bipartite_connections)

        weights = weights_map[map_key]
        assert num_groups == bipartite_connections.shape[0]

        GC, GF = bipartite_connections.shape
        NC, NF = weights.shape[2:]

        for gc in range(GC):
          dst_var_name = 'grouped_weights_%d' % gc
          dst_var = slim.get_unique_variable(
              '/'.join([model_name, scope_name, dst_var_name]))

          for gf in range(GF):
            if bipartite_connections[gc, gf]:
              # selected bipartite connection
              dst_val = weights[:, :,
                                (gc*int(NC/GC)):((gc+1)*int(NC/GC)),
                                (gf*int(NF/GF)):((gf+1)*int(NF/GF))]
              break

          assign_ops.append(tf.assign(dst_var, dst_val))

        # biases
        assign_ops.append(tf.assign(slim.get_unique_variable(
            '/'.join([model_name, scope_name, 'biases'])),
            weights_map['/'.join([model_name, scope_name, 'biases'])]))

        # assign the pointwise
        if init_pointwise_from_pre_trained:
          tf.logging.info('Init pointwise from pre-trained model')
          pointwise_var = slim.get_unique_variable(
              '/'.join([model_name, scope_name, 'pointwise_weights']))
          epsilon = 1e-2
          pointwise_weights = np.random.random(
              pointwise_var.shape.as_list()) * epsilon

          for gc in range(GC):
            for gf in range(GF):
              if bipartite_connections[gc, gf]:
                for i in range(int(NC / GC)):
                  pointwise_weights[0, 0,
                                    gc * int(NC/GC) + i,
                                    gf * int(NF/GF) + i] = (
                      1. - pointwise_weights[0, 0,
                                             gc * int(NC/GC) + i,
                                             gf * int(NF/GF) + i])

          assign_ops.append(tf.assign(pointwise_var, pointwise_weights))

    elif 'separable' in layer_replacement[scope_name]:
      dst_var_name = 'depthwise_weights'
      dst_var = slim.get_unique_variable(
          '/'.join([model_name, scope_name, dst_var_name]))

      if not weights_map:
        src_var_name = scope_name + '/weights'
        src_var = layer_replace_vars[src_var_name]

        # This part is quite slow
        for i in range(dst_var.shape[2]):
          assign_op = tf.assign(dst_var[:, :, i, 0],
                                src_var[:, :, i, i])
          assign_ops.append(assign_op)
      else:
        dst_val = np.zeros(dst_var.shape, dtype=np.float32)
        map_key = '/'.join([model_name, scope_name, 'weights'])
        for c in range(dst_var.shape[2]):
          for f in range(dst_var.shape[2]):
            if bipartite_connections_map[map_key][c, f]:
              dst_val[:, :, c, 0] = weights_map[map_key][:, :, c, f]

        assign_ops.append(tf.assign(dst_var, dst_val))
        assign_ops.append(tf.assign(slim.get_unique_variable(
            '/'.join([model_name, scope_name, 'biases'])),
            weights_map['/'.join([model_name, scope_name, 'biases'])]))

      # assign the pointwise
      if init_pointwise_from_pre_trained:
        tf.logging.info('Init pointwise from pre-trained model')
        pointwise_var = slim.get_unique_variable(
            '/'.join([model_name, scope_name, 'pointwise_weights']))
        epsilon = 1e-2
        pointwise_weights = np.random.random(
            pointwise_var.shape.as_list()) * epsilon

        if bipartite_connections_map:
          map_key = '/'.join([model_name, scope_name, 'weights'])
          for c in range(pointwise_var.shape[2]):
            for f in range(pointwise_var.shape[3]):
              if bipartite_connections_map[map_key][c, f]:
                pointwise_weights[0, 0, c, f] = 1. - \
                    pointwise_weights[0, 0, c, f]
        else:
          for i in range(pointwise_var.shape[2]):
            pointwise_weights[0, 0, i, i] = 1. - pointwise_weights[0, 0, i, i]

        assign_ops.append(tf.assign(pointwise_var, pointwise_weights))

  return assign_ops


def get_init_fn(checkpoint_path, scopes, layer_replacement, model_name,
                init_from_pre_trained=False,
                init_pointwise_from_pre_trained=False,
                weights_map=None,
                bipartite_connections_map=None,
                num_groups=2):
  """ Get model initialisation function before training.
  :param checkpoint_path: A checkpoint to be restored from.
  :param scopes: scopes to exclude.
  :param layer_replacement: how layers are replaced
  :return: a init function.
  """

  if not checkpoint_path:
    return None
  if not scopes:
    checkpoint_exclude_scopes = []
  elif isinstance(scopes, str):
    checkpoint_exclude_scopes = scopes.split(',')
  else:
    checkpoint_exclude_scopes = scopes

  exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

  variables_to_restore = {}
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore[var.op.name] = var

  # create variable assignment operations
  assign_ops = []

  tf.logging.info('Initialize from pre-trained model: %s' %
                  init_from_pre_trained)

  if init_from_pre_trained:
    assign_ops = init_from_pre_trained_assign_ops(
        model_name,
        layer_replacement,
        variables_to_restore,
        init_pointwise_from_pre_trained=init_pointwise_from_pre_trained,
        weights_map=weights_map,
        bipartite_connections_map=bipartite_connections_map,
        num_groups=num_groups)

  checkpoint_assign_op, feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  def init_fn(sess):
    tf.logging.info('Restoring from checkpoint file: %s' % checkpoint_path)
    sess.run(checkpoint_assign_op, feed_dict)
    tf.logging.info('Running variable assignments')
    for op in assign_ops:
      sess.run(op)

  return init_fn


def get_variables_to_train(scopes=None):
  """ Get trainable variables """

  if scopes is None:
    return tf.trainable_variables()
  if isinstance(scopes, str):
    scopes = scopes.split(',')

  scopes = [scope.strip() for scope in scopes]
  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def create_conv_activation_summary(name, end_points):
  activation = end_points[name]
  shape = activation.shape
  num_channels = shape[-1]

  if num_channels == 1:
    return tf.summary.image('activations/' + name, activation)
  else:
    return tf.summary.image('activations/' + name, activation[:, :, :, 0:1])


def create_train_summary_op(end_points, learning_rate, total_loss, images_raw, images, model_name=None):
  """ Create an summary op from the built model.
  """
  # add summaries
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

  for end_point in end_points:
    x = end_points[end_point]
    summaries.add(tf.summary.histogram('activations/' + end_point, x))
  for variable in slim.get_model_variables():
    summaries.add(tf.summary.histogram(variable.op.name, variable))

  summaries.add(tf.summary.scalar('learning_rate', learning_rate))
  # Add summaries for losses.
  for loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
    summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
  summaries.add(tf.summary.scalar('losses/total_loss', total_loss))
  summaries.add(tf.summary.image('images_raw', images_raw))
  summaries.add(tf.summary.image('images', images))

  if model_name == 'vgg_16':
    for name in end_points:
      if 'conv' in name:
        summaries.add(create_conv_activation_summary(name, end_points))

  summary_op = tf.summary.merge(list(summaries), name='train_summary_op')

  return summary_op


def create_eval_summary_op(names_to_values):
  """ Create evaluation summary operator.

  Don't know why but it seems that these summary operators cannot work.
  """
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

  for name in names_to_values:
    summaries.add(tf.summary.scalar(name, names_to_values[name]))
  summary_op = tf.summary.merge(list(summaries), name='eval_summary_op')

  return summary_op


def train_model(model_name,
                dataset_name,
                dataset_dir,
                train_dir,
                batch_size=32,
                checkpoint_path=None,
                max_number_of_epochs=10,
                trainable_scopes=None,
                checkpoint_exclude_scopes=None,
                use_mask=False,
                weight_decay=4e-5,
                learning_rate=1e-3,
                num_groups=2,
                init_from_pre_trained=False,
                init_pointwise_from_pre_trained=False,
                weights_map=None,
                bipartite_connections_map=None,
                **kwargs):
  """ Train the given model.

  Returns:
      A final loss of the training process.
  """

  tf.logging.set_verbosity(tf.logging.INFO)

  layer_replacement = kwargs.get('layer_replacement')

  with tf.Graph().as_default():
    dataset = dataset_factory.get_dataset(dataset_name, 'train', dataset_dir)

    preprocessing_fn = preprocessing_factory.get_preprocessing(
        model_name, is_training=True)

    network_fn = network_factory.get_network_fn(
        model_name, dataset.num_classes,
        use_mask=use_mask, weight_decay=weight_decay, is_training=True)

    max_number_of_steps = int(math.ceil(max_number_of_epochs * dataset.num_samples
                                        / batch_size))

    tf.logging.info('Training on %s' % train_dir)
    tf.logging.info('Number of samples: %d' % dataset.num_samples)
    tf.logging.info('Max number of steps: %d' % max_number_of_steps)

    """
    Load data from the dataset and pre-process.
    """
    images, images_raw, labels = load_batch(
        dataset, preprocessing_fn, is_training=True)

    # create arg_scope
    logits, end_points = network_fn(images,
        bipartite_connections_map=bipartite_connections_map,
        **kwargs)

    # compute losses
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    # configure learning rate
    global_step = slim.create_global_step()
    learning_rate = configure_learning_rate(learning_rate,
                                            dataset.num_samples,
                                            global_step)

    # create summary op
    summary_op = create_train_summary_op(end_points, learning_rate,
                                         total_loss, images_raw, images,
                                         model_name=model_name)

    """
    Configure optimizer and training.

    if we do fine-tuning, just set trainable_scopes.
    """
    variables_to_train = get_variables_to_train(trainable_scopes)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=0.9,
                                          momentum=0.9,
                                          epsilon=1.0)

    # training operator
    train_op = slim.learning.create_train_op(total_loss, optimizer,
                                             variables_to_train=variables_to_train)

    init_fn = get_init_fn(checkpoint_path,
                          checkpoint_exclude_scopes,
                          layer_replacement,
                          model_name,
                          weights_map=weights_map,
                          bipartite_connections_map=bipartite_connections_map,
                          init_from_pre_trained=init_from_pre_trained,
                          init_pointwise_from_pre_trained=init_pointwise_from_pre_trained,
                          num_groups=num_groups)
    # final loss value
    final_loss = slim.learning.train(train_op,
                                     logdir=train_dir,
                                     init_fn=init_fn,
                                     number_of_steps=max_number_of_steps,
                                     log_every_n_steps=10,
                                     save_summaries_secs=60)

    print('Finished training. Final batch loss %f' % final_loss)

    return final_loss


def eval_model(model_name,
               dataset_name,
               dataset_dir,
               train_dir,
               batch_size=32,
               use_mask=False,
               num_groups=2,
               init_from_pre_trained=False,
               init_pointwise_from_pre_trained=False,
               weights_map=None,
               **kwargs):
  """ Evaluate the performance of a model. """

  with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    tf_global_step = tf.train.get_or_create_global_step()

    dataset = dataset_factory.get_dataset(
        dataset_name, 'test', dataset_dir)

    preprocessing_fn = preprocessing_factory.get_preprocessing(
        model_name, is_training=False)

    network_fn = network_factory.get_network_fn(
        model_name, dataset.num_classes, use_mask=use_mask)

    images, images_raw, labels = load_batch(dataset,
                                            preprocessing_fn,
                                            shuffle=False,
                                            batch_size=batch_size,
                                            is_training=False)

    logits, _ = network_fn(images, **kwargs)

    predictions = tf.argmax(logits, 1)

    # Evaluation
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'eval/recall_5': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 5),
    })

    summary_op = create_eval_summary_op(names_to_values)

    num_evals = int(math.ceil(dataset.num_samples / float(batch_size)))
    variables_to_restore = slim.get_variables_to_restore()

    # We don't want to mess up with the train dir
    eval_dir = train_dir + '_eval'

    metric_values = slim.evaluation.evaluate_once(
        '',
        tf.train.latest_checkpoint(train_dir),
        eval_dir,
        num_evals=num_evals,
        eval_op=list(names_to_updates.values()),
        final_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)

    names_to_values = dict(zip(names_to_values.keys(), metric_values))
    for name in names_to_values:
      print('%s: %f' % (name, names_to_values[name]))

  return names_to_values['eval/accuracy'], names_to_values['eval/recall_5']


def train_eval_once(model_name,
                    dataset_name,
                    dataset_dir,
                    train_dir,
                    checkpoint_path=None,
                    batch_size=32,
                    max_number_of_epochs=10,
                    trainable_scopes=None,
                    checkpoint_exclude_scopes=None,
                    learning_rate=1e-3,
                    use_mask=False,
                    **kwargs):
  """ Train and evaluate the given model once.

  :return: Final loss, accuracy, recall_5
  """

  # first run training
  final_loss = train_model(model_name,
                           dataset_name,
                           dataset_dir,
                           train_dir,
                           checkpoint_path=checkpoint_path,
                           max_number_of_epochs=max_number_of_epochs,
                           trainable_scopes=trainable_scopes,
                           checkpoint_exclude_scopes=checkpoint_exclude_scopes,
                           learning_rate=learning_rate,
                           use_mask=use_mask,
                           **kwargs)

  # then run evaluation
  accuracy, recall_5 = eval_model(model_name,
                                  dataset_name,
                                  dataset_dir,
                                  train_dir,
                                  batch_size=batch_size,
                                  use_mask=use_mask,
                                  **kwargs)

  return final_loss, accuracy, recall_5


def train_eval(model_name,
               dataset_name,
               dataset_dir,
               train_dir,
               checkpoint_path=None,
               batch_size=32,
               max_number_of_epochs=10,
               num_epochs_per_eval=5,
               trainable_scopes=None,
               checkpoint_exclude_scopes=None,
               learning_rate=1e-3,
               use_mask=False,
               **kwargs):
  """ Train and evaluate a model.

  The model will be trained until we reaches the maximum number of epochs
  or the model is converged.
  """

  tf.logging.set_verbosity(tf.logging.INFO)

  latest_step = get_latest_step(train_dir)
  train_num_samples = get_train_dataset_num_samples(
      dataset_name, dataset_dir)
  num_evals = int(math.ceil(max_number_of_epochs / num_epochs_per_eval))
  results = []

  for i in range(num_evals):
    curr_max_number_of_epochs = (i + 1) * num_epochs_per_eval
    curr_max_number_of_steps = int(math.ceil(train_num_samples * curr_max_number_of_epochs
                                             / batch_size))

    tf.logging.info('Running evaluation %03d ...' % i)
    tf.logging.info('Current max number of epochs: %d' %
                    curr_max_number_of_epochs)
    tf.logging.info('Current max number of steps: %d' %
                    curr_max_number_of_steps)

    if latest_step >= curr_max_number_of_steps:
      tf.logging.info('Skipped because the latest step (%d) is larger than '
                      'the current maximal steps (%d)'
                      % (latest_step, curr_max_number_of_steps))
      continue

    result = train_eval_once(model_name,
                             dataset_name,
                             dataset_dir,
                             train_dir,
                             checkpoint_path=checkpoint_path,
                             batch_size=batch_size,
                             max_number_of_epochs=curr_max_number_of_epochs,
                             trainable_scopes=trainable_scopes,
                             checkpoint_exclude_scopes=checkpoint_exclude_scopes,
                             learning_rate=learning_rate,
                             use_mask=use_mask,
                             **kwargs)
    results.append(result)

  return results
