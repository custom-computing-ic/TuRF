import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
if 'TF_SLIM_PATH' in os.environ:
  sys.path.append(os.environ['TF_SLIM_PATH'])
else:
  sys.path.append(os.path.join(
      os.path.dirname(__file__), '../../models/research/slim'))

from preprocessing import vgg_preprocessing


preprocessing_fn_map = {
    'vgg_16': vgg_preprocessing,
}


def get_preprocessing(name, is_training=False):
  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name %s not found' % name)

  def preprocesing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, output_height, output_width, is_training=is_training, **kwargs)

  return preprocesing_fn
