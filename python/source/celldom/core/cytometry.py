import os
import os.path as osp
import celldom
import logging
import tempfile
import pandas as pd
from collections import OrderedDict
from celldom import io as celldom_io
from celldom.config import cell_config, marker_config
from celldom.extract import DPF_NONE

logger = logging.getLogger(__name__)

DEFAULT_CELL_STAT_ATTRS = ['area', 'solidity', 'eccentricity']
DEFAULT_CELL_STAT_NAMES = ['mean', 'std', 'min', 'max']


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


class Acquisition(object):

    def __init__(self, path, properties):
        self.path = path
        self.properties = properties

    def load_image(self, dataset_class):
        dataset = dataset_class()
        dataset.initialize([self.path])
        dataset.prepare()
        return dataset.load_image(0)


class CytometerConfig(object):

    def __init__(self, model_paths):
        self.model_paths = model_paths


class Cytometer(object):

    def __init__(self, config, data_dir, properties=None, chip_config=None):
        self.config = config
        self.data_dir = data_dir
        self.properties = properties
        self.chip_config = chip_config or celldom.get_default_chip_config()
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

    def process(self, acquisition, dpf=DPF_NONE):
        acq_data, apt_data = self.analyze(acquisition, dfp=dpf)
        # TODO: Need to integrate experiment_config and use that to write out data


    def analyze(self, acquisition, dpf=DPF_NONE):
        _check_initialized()

        # Prepare single image dataset
        # Note that while this may seem unnecessary for loading images, it used here because
        # the "*Dataset" implementations often include image specific pre-processing that should
        # be applied -- and in this case that means reflection and uint8 conversion)
        image = acquisition.load_image(marker_dataset.MarkerDataset)

        # Extract all relevant information
        partitions, norm_image, norm_centers, neighbors, rotation, scale = apartment_extraction.extract(
            image, self.marker_model, self.chip_config,
            digit_model=self.digit_model, cell_model=self.cell_model, focus_model=self.focus_model,
            chip_scaling=False, dpf=dpf
        )

        acq_data = pd.Series({
            'acq_image_shape': image.shape,
            'acq_norm_image': norm_image if dpf.acq_norm_image else None,
            'apt_count': len(partitions),
            'rotation': rotation,
            'scale': scale,
        })
        apt_data = pd.DataFrame([_prepare(partition, dpf) for partition in partitions])

        return acq_data, apt_data

    def _prepare(self, partition, dpf):
        r = pd.Series(partition)

        # Summarize cells
        cells = r['cells']
        r['cell_count'] = len(cells)
        r = r.append(_get_cell_stats(cells))

        # Remove images in result if indicated that they should not be saved in isf (ImageSaveFlags)
        dpf_dict = dpf._asdict()
        r = r.drop([
            p for p in r.filter(regex='image$|images$').index.values
            if p in dpf_dict and not dpf_dict[p]
        ])

        return r


def _get_cell_stats(
        cells,
        attrs=DEFAULT_CELL_STAT_ATTRS,
        stats=DEFAULT_CELL_STAT_NAMES,
        percentiles=None):
    """Fetch cell statistic summaries

    Returns:
        A series with keys like "cell_area_min", "cell_area_max", "cell_area_std", "cell_solidity_mean", etc.
    """

    def get_stats(x, name):
        return (
            pd.Series(x)
            .describe(percentiles=percentiles)
            .loc[stats]
            # Replace percent signs in percentile values
            .rename(lambda v: 'p' + v.replace('%', '') if '%' in v else v)
            .add_prefix('cell_' + name + '_')
        )

    # Aggregate series of statistics for all given attributes
    return pd.concat([
        get_stats([getattr(c, attr, None) for c in cells], attr)
        for attr in attrs
    ])
