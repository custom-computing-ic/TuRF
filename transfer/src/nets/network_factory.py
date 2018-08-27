r"""
Return a network_fn and network_arg_scope_fn.
"""
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import vgg


networks_map = {
    'vgg_16': vgg.vgg_16,
}

arg_scopes_map = {
    'vgg_16': vgg.vgg_arg_scope,
}


def get_network_fn(name, num_classes, use_mask=False, weight_decay=0.0, is_training=False):
  """
  Return a network_fn wrapped in arg_scope

  When using the returned network_fn, there is no need to wrap it 
  with arg_scope.
  """

  if name not in networks_map:
    raise ValueError('Name of the network is unknown: %s' % name)
  func = networks_map[name]

  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_map[name](use_mask=use_mask,
                                     weight_decay=weight_decay)

    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training, **kwargs)

  # network_fn is a newly created function and doesn't have default_image_size
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
