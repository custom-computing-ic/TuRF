r"""
Create a dataset by a given name.
"""


import dataset_birds


datasets_map = {
    'birds': dataset_birds,
}


def get_dataset(name, split_name, dataset_dir):
  """ Given a dataset name and a split_name returns a Dataset.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset is unknown: %s' % name)

  return datasets_map[name].get_split(split_name, dataset_dir)
