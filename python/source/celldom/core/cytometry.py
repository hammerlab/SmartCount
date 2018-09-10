import os
import os.path as osp
import celldom
import logging
import tempfile
import pandas as pd
import numpy as np
import hashlib
from skimage import io as sk_io
from collections import OrderedDict
from celldom import io as celldom_io
from celldom.config import cell_config, marker_config
from celldom.dataset import marker_dataset
from celldom.extract import NO_IMAGES, ALL_IMAGES, APT_IMAGES, apartment_extraction
from celldom.utils import assert_rgb
from celldom.config.utils import get_apartment_image_shape
from celldom.warnings import disable_skimage_warnings
from cvutils.encoding import base64_encode_image

# Disable known skimage warnings relating to this module and its usage
disable_skimage_warnings()

logger = logging.getLogger(__name__)

DEFAULT_CELL_STAT_ATTRS = ['area', 'solidity', 'eccentricity']
DEFAULT_CELL_STAT_PERCENTILES = [.1, .5, .9]
DEFAULT_CELL_STAT_NAMES = ['mean', 'std', 'p10', 'p50', 'p90']
DEFAULT_STRING_SIZE = 64
PATH_STRING_SIZE = 256
ACQ_PROP_PREFIX = 'acq_'


def _resolve_paths(config):
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
            cache_path = osp.join('models', osp.basename(url))
            model_paths[model_name] = celldom_io.cache(url, cache_path)
        else:
            raise ValueError('Model location type "{}" not valid (should be "file" or "url")'.format(location_type))

    return model_paths


class Datastore(object):
    """Abstract class representing model for manipulating cytometry results"""
    pass

    def close(self):
        raise NotImplementedError()


def _initialize_data_dir(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def image_compression_codec():
    # See: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_hdf.html
    return os.getenv(celldom.ENV_CELLDOM_IMAGE_COMPRESSION_CODEC, 'blosc:lz4')


def image_compression_level():
    # See: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_hdf.html
    return int(os.getenv(celldom.ENV_CELLDOM_IMAGE_COMPRESSION_LEVEL, 9))


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


def get_readonly_datastore(data_dir, data_file='data.h5'):
    return HDF5Datastore(data_dir, 'r', data_file).store


def get_readonly_images(data_dir, data_file='images.h5'):
    return HDF5Datastore(data_dir, 'r', data_file)


class Acquisition(object):

    def __init__(self, config, path, properties=None):
        self.config = config
        self.path = path
        # As an instance attribute, this may be useful for bypassing property inference
        # in the future (but for now it does nothing)
        self._properties = properties

    def load_image(self, dataset_class=marker_dataset.MarkerDataset):
        """Load the image associated with this acquisition while accounting for preprocessing"""
        dataset = dataset_class(
            reflect_images=self.config.acquisition_reflection,
            scale_factor=self.config.acquisition_scale_factor
        )
        dataset.initialize([self.path])
        dataset.prepare()
        return dataset.load_image(0)

    def infer_properties(self, exp_config):
        if self._properties is not None:
            raise ValueError('Cannot infer properties for acquisition if already explicitly defined')
        props = exp_config.parse_path(self.path)
        if 'id' in props:
            raise ValueError(
                'Properties inferred from paths cannot include an "id" attribute (properties = {})'
                .format(props)
            )
        props['id'] = self._generate_id(props)
        return props

    def _generate_id(self, props):
        keys = sorted(list(props.keys()))
        key = ':'.join([str(props[k]) for k in keys])
        return hashlib.md5(key.encode('utf-8')).hexdigest()


SUPPORTED_IMAGE_FIELDS = {
    'apt_image': {'key_fields': ['acq_id', 'apt_id']},
    # 'cell_image': {'key_fields': ['acq_id', 'apt_id', 'cell_id']}
}


def get_image_key(image_type, image_field, properties):
    """Generate key (as string path) used to save images in HDF5 stores

    Args:
        image_type: Object type (apartment, cell, acqusition, etc.)
        image_field: Name of image field associated with object (apt_image, raw_norm_image, cell_image, etc.)
        properties: Any dictionary containing keys used to generate store key (acq_id, apt_id, etc.)
    Returns:
        String key path for HDF5 storage
    """
    field_meta = SUPPORTED_IMAGE_FIELDS[image_field]
    keys = [image_type]

    # Add to full key a select set of identifying fields for the image
    for c in field_meta['key_fields']:
        keys.append(c + '_' + str(properties[c]))
    return '/'.join(keys)


class Cytometer(object):

    def __init__(self, config, data_dir, output_mode='w', enable_focus_scores=True):
        self.config = config
        self.model_paths = _resolve_paths(config.get_cytometer_config())
        self.chip_config = config.get_chip_config()
        self.data_dir = data_dir
        self.output_mode = output_mode
        self.enable_focus_scores = enable_focus_scores

        self.datastore = None
        self.images = None
        self.initialized = False

    def initialize(self):
        self.__enter__()

    def shutdown(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        import tensorflow as tf

        # This probably belongs elsewhere, but try to eliminate TF warnings here anyways
        tf.logging.set_verbosity(logging.ERROR)

        import keras
        from cvutils.mrcnn import model as mrcnn_model
        from celldom.miq import miq

        # Initialize keras session
        tf_conf = celldom.initialize_keras_session()

        # Initialize datastore to maintain results
        self.datastore = HDF5Datastore(self.data_dir, self.output_mode, data_file='data.h5')

        comp_codec, comp_level = image_compression_codec(), image_compression_level()
        logger.debug('Initializing image storage with codec %s (level = %s)', comp_codec, comp_level)
        self.images = HDF5Datastore(self.data_dir, self.output_mode,
                                    data_file='images.h5', complib=comp_codec, complevel=comp_level)

        # Initialize predictive models
        self.digit_model = keras.models.load_model(self.model_paths['digit'])
        self.marker_model = mrcnn_model.get_model(
            'inference', marker_config.MarkerInferenceConfig(), tempfile.mkdtemp(),
            init_with='file', file=self.model_paths['marker']
        )
        self.cell_model = mrcnn_model.get_model(
            'inference', cell_config.CellInferenceConfig(), tempfile.mkdtemp(),
            init_with='file', file=self.model_paths['cell']
        )
        if self.enable_focus_scores:
            self.focus_model = miq.get_classifier(tf_conf)
        else:
            self.focus_model = None
        self.initialized = True
        return self

    def __exit__(self, type, value, traceback):
        if type:
            raise value

        from keras import backend

        # Close or clear sessions as well as possible (maybe there are more comprehensive ways?)
        backend.clear_session()
        if self.focus_model is not None:
            self.focus_model._sess.close()

        # Close underlying datastores and assume they will never be used again
        if self.datastore is not None:
            self.datastore.close()
        if self.images is not None:
            self.images.close()

        return True

    def flush(self):
        if self.datastore is not None:
            self.datastore.store.flush()
        if self.images is not None:
            self.images.store.flush()

    def _check_initialized(self):
        if not self.initialized:
            raise ValueError('Cytometer cannot be used until initialized (call `cytometer.initialize` first)')

    def run(self, path, dpf=APT_IMAGES):
        """Run cytometry analysis on a single image

        Args:
            path: Path of image on filesystem
            dpf: "Data Persistence Flags" indicating how more expensive objects like images are handled; see
                celldom.extract for preset groupings of flags that can be used here -- for example:
                ```
                cytometer = cytometry.Cytometer(experiment_config, output_dir)
                cytometry.analyze(image_path, dpf=extract.NO_IMAGES)
                ```
                Default is `APT_IMAGES` which will only save apartment images extracted from raw images
        Returns:
            (acquisition_data, apartment_data, cell_data) where each is a dataframe containing:
                - acquisition_data: Information about the original acquisition image (like rotation, marker locations)
                - apartment_data: Apartment data like cell count, the extracted apartment image, etc.
                - cell_data: Sizes, locations, and images of individual cells with fields linking cells back to
                    apartments
        """
        return self.analyze(Acquisition(self.config, path), dpf=dpf)

    def analyze(self, acquisition, dpf=APT_IMAGES):
        """Run cytometry analysis on a single acquisition

        Args:
            acquisition: Acquisition object
            dpf: "Data Persistence Flags" indicating how more expensive objects like images are handled; see
                celldom.extract for preset groupings of flags that can be used here -- for example:
                ```
                cytometer = cytometry.Cytometer(experiment_config, output_dir)
                acquisition = cytometry.Acquisition(experiment_config, image_path)
                cytometry.analyze(acquisition, dpf=extract.NO_IMAGES)
                ```
                Default is `APT_IMAGES` which will only save apartment images extracted from raw images
        Returns:
            (acquisition_data, apartment_data, cell_data) where each is a dataframe containing:
                - acquisition_data: Information about the original acquisition image (like rotation, marker locations)
                - apartment_data: Apartment data like cell count, the extracted apartment image, etc.
                - cell_data: Sizes, locations, and images of individual cells with fields linking cells back to
                    apartments
        """
        self._check_initialized()

        # Prepare single image dataset
        # Note that while this may seem unnecessary for loading images, it used here because
        # the "*Dataset" implementations often include image specific pre-processing that should
        # be applied -- and in this case that potentially means reflection, uint8 conversion, and resizing)
        image = acquisition.load_image()

        # At this point, the image should always be 8-bit RGB
        assert_rgb(image)

        # Infer properties associated with the given acquisition based on file path
        properties = acquisition.infer_properties(self.config)
        properties['processed_at'] = pd.to_datetime('now')

        # Extract all relevant information
        partitions, norm_image, norm_centers, neighbors, rotation = apartment_extraction.extract(
            image, self.marker_model, self.chip_config,
            digit_model=self.digit_model, cell_model=self.cell_model, focus_model=self.focus_model, dpf=dpf
        )

        acq_data = pd.DataFrame([dict(
            raw_image_path=acquisition.path,
            raw_image_shape_height=image.shape[0],
            raw_image_shape_width=image.shape[1],
            raw_norm_image=norm_image if dpf.raw_norm_image else None,
            apt_count=len(partitions),
            rotation=rotation
        )])

        # Extract a data frame with rows representing each cell
        # Note: If you don't keep empty data frames out of the list to concatenate,
        # it messes up the data types of results
        cell_data = [self._prepare_cells(r) for r in partitions if len(r['cells']) > 0]
        cell_data = pd.concat(cell_data) if cell_data else pd.DataFrame()

        # Compute cell stats at apartment level and drop cells field
        apt_data = pd.DataFrame([self._prepare_apt(r, dpf) for r in partitions])

        # Assign extra global, inferred information to all datasets
        acq_data, apt_data, cell_data = [
            _assign_properties(d, ACQ_PROP_PREFIX, properties)
            for d in [acq_data, apt_data, cell_data]
        ]

        # Return column-sorted results
        return acq_data.sort_index(axis=1), apt_data.sort_index(axis=1), cell_data.sort_index(axis=1)

    def _prepare_cells(self, r):
        """Convert list of cell dicts for one apartment to data frame"""
        return pd.DataFrame(r['cells'])\
            .assign(apt_id=r['apt_id'], apt_num=r['apt_num'], st_num=r['st_num'])

    def _prepare_apt(self, r, dpf):
        r = pd.Series(r)

        # Summarize cells
        cells = r['cells']
        r['cell_count'] = len(cells)
        r = r.append(_get_cell_stats(cells))

        # Drop the value of any image fields not flagged for persistence
        for image_field in _get_image_fields(r.index.values, dpf):
            r[image_field] = None

        return r.drop('cells')

    def save(self, acq_data, apt_data, cell_data):
        for table, df in [
            ('acquisition', acq_data),
            ('apartment', apt_data),
            ('cell', cell_data)
        ]:
            if df is not None and len(df) > 0:
                self._save(table, df)
        return self

    def _save_images(self, key, df, image_field):
        field_meta = SUPPORTED_IMAGE_FIELDS[image_field]
        cols = field_meta['key_fields'] + [image_field]
        for i, r in df[cols].iterrows():
            image = r[image_field]
            if image is None:
                continue
            self.images.save_image(get_image_key(key, image_field, r), image)

    def _save(self, key, df):
        d = df.copy()

        # Convert non-numeric fields to truncated strings (and remove any image fields)
        col_sizes = {}
        object_cols = d.select_dtypes('object').columns.values
        for c in object_cols:

            # Save supported image fields in separate datastore
            if _is_image_field(c):
                if c in SUPPORTED_IMAGE_FIELDS:
                    self._save_images(key, d, c)
                d = d.drop(c, axis=1)
                continue

            # Make a string size exception for file paths (which are much longer)
            # and ids (which are fixed size)
            if _is_path_field(c):
                col_sizes[c] = PATH_STRING_SIZE
            elif c == ACQ_PROP_PREFIX + 'id':
                col_sizes[c] = 32  # These are md5s
            else:
                col_sizes[c] = DEFAULT_STRING_SIZE
            d[c] = d[c].apply(lambda v: None if v is None else str(v)[:col_sizes[c]])

        # Create list of indexed fields
        index_columns = d.filter(regex='^' + ACQ_PROP_PREFIX).columns.tolist()
        for c in ['apt_num', 'st_num']:
            if c in d:
                index_columns.append(c)
        self.datastore.save_df(key, d, min_itemsize=col_sizes, data_columns=index_columns)


def _assign_properties(d, prefix, properties):
    p = {(prefix + k): v for k, v in properties.items()}
    if isinstance(d, pd.DataFrame):
        return d.assign(**p)
    elif isinstance(d, pd.Series):
        return d.append(pd.Series(p))
    else:
        raise ValueError('Data must be frame or series not {}'.format(type(d)))


def _is_image_field(field):
    return field.endswith('_image') or field.endswith('_images')


def _is_path_field(field):
    return field.endswith('_path')


def _get_image_fields(fields, dpf):
    dpf_dict = dpf._asdict()
    return [
       f for f in fields
       if _is_image_field(f) and f in dpf_dict and not dpf_dict[f]
    ]


def _get_cell_stats(
        cells,
        attrs=DEFAULT_CELL_STAT_ATTRS,
        stats=DEFAULT_CELL_STAT_NAMES,
        percentiles=DEFAULT_CELL_STAT_PERCENTILES):
    """Fetch cell statistic summaries

    Returns:
        A series with keys like "cell_area_min", "cell_area_max", "cell_area_std", "cell_solidity_mean", etc.
    """
    def get_stats(x, name):
        return (
            pd.Series(x)
            .describe(percentiles=percentiles)
            # Replace percent signs in percentile values
            .rename(lambda v: 'p' + v.replace('%', '') if '%' in v else v)
            .filter(stats)
            .add_prefix('cell_' + name + '_')
        )

    # Aggregate series of statistics for all given attributes
    return pd.concat([
        get_stats([c.get(attr, np.nan) for c in cells], attr)
        for attr in attrs
    ])
