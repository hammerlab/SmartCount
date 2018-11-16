import os
import os.path as osp
import pandas as pd


def _initialize_data_dir(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


class Datastore(object):
    """Abstract class representing model for manipulating cytometry results"""
    pass

    def close(self):
        raise NotImplementedError()

    
class HDF5Datastore(Datastore):

    def __init__(self, data_dir, mode, data_file='data.h5', **kwargs):
        """Datastore implementation based on local HDF5 files"""
        self.data_dir = _initialize_data_dir(data_dir)
        self.data_file = osp.join(self.data_dir, data_file)
        self.store = pd.HDFStore(self.data_file, mode=mode, **kwargs)

    def close(self):
        self.store.close()

    def save_image(self, key, image, grayscale=True):

        # If storing grayscale images, make sure not to store as redundant RGB representation
        if grayscale and image.ndim == 3:
            image = image[..., 0]

        self.store.put(key + '/shape', pd.Series(image.shape), format='fixed')
        self.store.put(key + '/data', pd.Series(image.ravel()), format='fixed')

    def load_image(self, key):
        """Load image array for the given key, or return nothing if key is not present"""
        try:
            shape = tuple(self.store.get(key + '/shape'))
            image = self.store.get(key + '/data')
        except KeyError:
            return None
        return image.values.reshape(shape)

    def save_df(self, key, df, **kwargs):
        if len(df) > 0:
            self.store.append(key, df, **kwargs)
