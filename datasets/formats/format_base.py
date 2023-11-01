""" Base dataset format interface for loading data files. """

class DatasetFormatBase():
  
  @staticmethod
  def load_data(root, split, *args, **kwargs):
    """Will return the dictionary of tensor data modes,
    (e.g.: rgbs, depths, rays).

    Args:
        root (str): root path of the dataset
        split (str): The split to use from train, val, test

    Returns:
        (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
    """
    raise NotImplementedError('DatasetFormat: **load_data** must be overriden in child dataset format classes...')