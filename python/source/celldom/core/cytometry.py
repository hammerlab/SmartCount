import os
import os.path as osp
import celldom
import logging
import tempfile
import pandas as pd
from celldom import io as celldom_io
from celldom.config import cell_config, marker_config

logger = logging.getLogger(__name__)


def get_cytometer_config_from_name(name):
    """Get a cytometer config by name (resolves to file in repo)

    Args:
        name: Name of cytometer (assumed to be a filename minus extension present in CELLDOM_REPO_DIR/config/cytometer);
            e.g. 'cytometer_01'
    Returns:
        CytometerConfig
    """
    return _deserialize(celldom.get_cytometer_config(name))


def get_cytometer_config_from_path(path):
    """Get a cytometer config from local file

    Args:
        path: Path to cytometer configuration file
    Returns:
        CytometerConfig
    """
    return _deserialize(celldom.read_config(path))


def _deserialize(config):
    model_config = config['resources']['models']

    model_paths = {}
    for model_name, location in model_config.items():
        location_type = location['type']

        # If the location is a file path, use it as-is
        if location_type == 'file':
            model_paths[model_name] = location['location']

        # If it is a url, download it if not already cached
        elif location_type == 'url':
            url = location['location']
            cache_path = osp.join('models', celldom.get_version(), osp.basename(url))
            model_paths[model_name] = celldom_io.cache(url, cache_path)
        else:
            raise ValueError('Model location type "{}" not valid (should be "file" or "url")'.format(location_type))

    return CytometerConfig(model_paths)


class Datastore(object):
    """Abstract class representing model for manipulating cytometry results"""
    pass

    def close(self):
        raise NotImplementedError()


def _initialize_data_dir(data_dir):
    os.makedirs(self.data_dir, exist_ok=True)
    return data_dir


class HDF5Datastore(Datastore):

    def __init__(self, data_dir):
        """Datastore implementation based on local HDF5 files"""
        self.data_dir = _initialize_data_dir(data_dir)
        self.store = pd.HDFStore(self.data_dir)

    def close(self):
        self.store.close()


class ChipImage(object):

    def __init__(self, path, properties):
        self.path = path
        self.properties = properties


class CytometerConfig(object):

    def __init__(self, model_paths):
        self.model_paths = model_paths


class Cytometer(object):

    def __init__(self, config, data_dir, properties=None):
        self.config = config
        self.data_dir = data_dir
        self.properties = properties
        self.datastore = None
        self.initialized = False

    def initialize(self):
        self.__enter__()

    def shutdown(self):
        self.__exit__()

    def __enter__(self):
        import keras
        from cvutils.mrcnn import model as mrcnn_model
        from celldom.miq import miq

        # Initialize keras session
        tf_conf = celldom.initialize_keras_session()

        # Initialize datastore to maintain results
        self.datastore = HDF5Datastore(data_dir)

        # Initialize predictive models
        self.digit_model = keras.models.load_model(self.config.model_paths['digit'])
        self.marker_model = mrcnn_model.get_model(
            'inference', marker_config.MarkerInferenceConfig(), tempfile.mkdtemp(),
            init_with='file', file=self.config.model_paths['marker']
        )
        self.cell_model = mrcnn_model.get_model(
            'inference', cell_config.CellInferenceConfig(), tempfile.mkdtemp(),
            init_with='file', file=self.config.model_paths['cell']
        )
        self.focus_model = miq.get_classifier(tf_conf)
        self.initialized = True

    def __exit__(self, type, value, traceback):
        from keras import backend

        # Close or clear sessions as well as possible (maybe there are more comprehensive ways?)
        backend.clear_session()
        self.focus_model._sess.close()

        # Close underlying dataset and assume it will never be used again
        self.datastore.close()

    def _check_initialized(self):
        if not self.initialized:
            raise ValueError('Cytometer cannot be used until initialzed (call `cytometer.initialize` first)')

    def analyze(self, chip_image_path, properties=None):
        _check_initialized()

        pass
        # merge chip results metadata to cytometer metadata
        # Append to HDF5 store
