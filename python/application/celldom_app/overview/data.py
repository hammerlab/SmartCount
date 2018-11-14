import os
import glob
import json
import os.path as osp
import numpy as np
import pandas as pd
import time
from celldom.execute import analysis
from celldom.execute import visualization
from celldom.core import cytometry
from celldom.core.experiment import Experiment
from celldom.execute import calculation
from celldom_app.overview import config
from cvutils.encoding import base64_encode_image
from skimage import color
from tables.exceptions import HDF5ExtError
import logging

logger = logging.getLogger(__name__)

cfg = None
image_store = None
output_dir = None
cache = {}

KEY_GROWTH_DATA = 'growth_data'
KEY_APT_DATA = 'apartment_data'
KEY_ACQ_DATA = 'acquisition_data'
KEY_CELL_DATA = 'cell_data'
KEY_ARRAY_DATA = 'array_data'
# Currently, nothing is cached by application (all such functionality should be pushed to celldom.execute.calculation)
SAVE_KEYS = []
STORE_READ_MAX_ATTEMPTS = 10
STORE_READ_PAUSE_SECS = 5
KEY_SEP = ':'


def save(overwrite=False):
    for k in SAVE_KEYS:
        if k not in cache:
            continue
        f = osp.join(output_dir, k + '.h5')
        if not osp.exists(f) or overwrite:
            logger.info('Saving cached object "%s" to path "%s"', k, f)
            cache[k].to_hdf(f, key='data')


def restore():
    for f in glob.glob(osp.join(output_dir, '*.h5')):
        k = osp.basename(f).split('.')[0]
        if k in SAVE_KEYS:
            logger.info('Loading cached object "%s" from path "%s"', k, f)
            cache[k] = pd.read_hdf(f)


def get_growth_data():
    return cache[KEY_GROWTH_DATA]


def get_acquisition_data():
    return cache[KEY_ACQ_DATA]


def get_cell_data():
    return cache[KEY_CELL_DATA]


def get_apartment_data():
    return cache[KEY_APT_DATA]


def get_array_data():
    return cache[KEY_ARRAY_DATA]


def get_array_key_fields():
    return cfg.experimental_condition_fields


def get_apartment_key_fields():
    return cfg.exp_config.apartment_address_fields


def make_key_string(values):
    return KEY_SEP.join(values)


def append_key(key, values):
    return key + KEY_SEP + make_key_string(values)


def _get_key(row, fields):
    vals = []
    for field in fields:
        if field not in row:
            raise ValueError('Row "{}" does not contain required key field "{}"'.format(row, field))
        vals.append(row[field])
    return make_key_string(vals)


def get_apartment_key(row):
    return _get_key(row, get_apartment_key_fields())


def get_array_key(row):
    return _get_key(row, get_array_key_fields())


def _prep(experiment, df):
    """Add or transform any globally available fields"""
    return calculation.add_measurement_times(experiment, df)


def get_output_path(path):
    """Get relative path from app output path (i.e. $EXP_DATA/app/overview)"""
    return osp.join(output_dir, path)


def initialize(data_dir):
    global cfg
    global image_store
    cfg = config.get()

    global output_dir
    logger.info('Initializing app data for data directory "%s"', data_dir)

    output_dir = osp.join(data_dir, 'app', 'overview')
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    restore()

    experiment = Experiment(cfg.exp_config, data_dir=data_dir)

    # Initialize image datastore
    try:
        image_store = experiment.get_image_store()
    except Exception:
        logger.warning('Image data not available (processor is likely still running)')

    # Extract necessary data from experiment output stores
    store = experiment.get_data_store()
    cache[KEY_CELL_DATA] = _prep(experiment, store.get('cell'))
    cache[KEY_APT_DATA] = _prep(experiment, store.get('apartment'))
    cache[KEY_ACQ_DATA] = _prep(experiment, store.get('acquisition'))

    # Load/compute growth rate data
    if KEY_GROWTH_DATA not in cache:
        df = calculation.calculate_apartment_growth_rates(experiment, overwrite=cfg.force_result_calculation)
        cache[KEY_GROWTH_DATA] = df

    if KEY_ARRAY_DATA not in cache:
        df = cache[KEY_GROWTH_DATA]
        df = df.groupby(cfg.experimental_condition_fields).agg({
            'growth_rate': ['count', 'median', 'mean'],
            'first_date': 'min',
            'last_date': 'max'
        })
        df.columns = [':'.join(c) for c in df]
        df = df.rename(columns={
            'growth_rate:count': 'num_apartments',
            'growth_rate:median': 'median_growth_rate',
            'growth_rate:mean': 'mean_growth_rate',
            'first_date:min': 'first_date',
            'last_date:max': 'last_date'
        })
        df = df.reset_index()
        cache[KEY_ARRAY_DATA] = df

    # Save any cached objects that haven't already been saved
    save(overwrite=False)

    # Remove OOB apartments if configured to do so
    if cfg.remove_oob_address:
        logger.info('Experiment data loaded with OOB apartment addresses removed')
        for k in [KEY_CELL_DATA, KEY_APT_DATA, KEY_GROWTH_DATA]:
            cache[k] = calculation.remove_oob_apartments(experiment, cache[k])
    else:
        logger.info('Experiment data loaded without removing OOB apartment addresses')


def get_apartment_image_data(df, marker_color=visualization.COLOR_RED):
    key_fields = get_apartment_key_fields()

    # Index growth data and use unique index values to subset other datasets
    idx = df.set_index(key_fields).index.unique()

    # Select apartment/cell data based on the given growth data
    apt_data = get_apartment_data().set_index(key_fields).loc[idx].copy()
    cell_data = get_cell_data().set_index(key_fields).loc[idx].copy()

    # Add externally saved apartment images to apartment data
    images = []
    for i, r in apt_data.iterrows():
        key = cytometry.get_image_key('apartment', 'apt_image', r)
        image = image_store.load_image(key)
        if image is None:
            images.append(None)
        else:
            # Convert grayscale images to rgb (this function preserves uint8 types)
            if image.ndim == 2:
                image = color.gray2rgb(image)
            images.append(image)
    apt_data['apt_image'] = images
    apt_data = apt_data[apt_data['apt_image'].notnull()]

    image_data = visualization.process_results(apt_data, cell_data, cfg.experimental_condition_fields, marker_color)

    res = []
    for k, g in image_data.groupby(key_fields):
        g = g.sort_values('acq_datetime')
        row = {key_fields[i]: k[i] for i in range(len(k))}
        row['n'] = len(g)
        row['encoded_images'] = [base64_encode_image(img) for img in g['image']]
        row['images'] = g['image'].values
        row['dates'] = g['acq_datetime'].tolist()
        row['cell_counts'] = g['cell_count'].tolist()
        res.append(row)
    res = pd.DataFrame(res)

    res['key'] = res.apply(get_apartment_key, axis=1)
    res = res.set_index('key')
    return res
