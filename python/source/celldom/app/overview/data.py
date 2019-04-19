import os
import glob
import os.path as osp
import numpy as np
import pandas as pd
import time
from celldom.execute import analysis
from celldom.execute import visualization
from celldom.core import cytometry
from celldom.execute import view
from cvutils.encoding import base64_encode_image
from celldom.app.overview import config
from skimage import color
from tables.exceptions import HDF5ExtError
import logging

logger = logging.getLogger(__name__)

cfg = None
image_store = None
cache = {}

KEY_APT = 'apartment'
KEY_APT_SUMMARY = 'apartment_summary'
KEY_ACQ = 'acquisition'
KEY_CELL = 'cell'
KEY_ARRAY = 'array'
KEY_ARRAY_SUMMARY = 'array_summary'

# Currently, nothing is cached by application (all such functionality should be pushed to celldom.execute.calculation)
SAVE_KEYS = []
STORE_READ_MAX_ATTEMPTS = 10
STORE_READ_PAUSE_SECS = 5
KEY_SEP = ':'


def save(overwrite=False):
    for k in SAVE_KEYS:
        if k not in cache:
            continue
        f = osp.join(cfg.app_output_dir, k + '.h5')
        if not osp.exists(f) or overwrite:
            logger.info('Saving cached object "%s" to path "%s"', k, f)
            cache[k].to_hdf(f, key='data')


def restore():
    for f in glob.glob(osp.join(cfg.app_output_dir, '*.h5')):
        k = osp.basename(f).split('.')[0]
        if k in SAVE_KEYS:
            logger.info('Loading cached object "%s" from path "%s"', k, f)
            cache[k] = pd.read_hdf(f)


def get_dataset(dataset):
    return cache[dataset]


def get_dataset_names():
    return cache.keys()


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


def get_output_path(path):
    """Get relative path from app output path (i.e. $EXP_DATA/app/overview)"""
    return osp.join(cfg.app_output_dir, path)


def _get_array_data(df):
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
    return df.reset_index()


def _get_array_summary_data(df_acq, df_arr):
    # Count raw files processed for each experimental condition
    df_acq = df_acq.groupby(cfg.experimental_condition_fields).size().rename('num_raw_images')

    # Select fields from array data
    df_arr = df_arr.set_index(cfg.experimental_condition_fields)[[
        'mean_growth_rate', 'median_growth_rate', 'num_apartments']]

    # Merge acquisition summary and array data
    return pd.concat([df_acq, df_arr], axis=1).reset_index()


def initialize():
    global cfg
    global image_store
    cfg = config.get()
    logger.info('Initializing app data for data directory "%s"', cfg.exp_data_dir)

    if not osp.exists(cfg.app_output_dir):
        os.makedirs(cfg.app_output_dir, exist_ok=True)
    restore()

    # Initialize image datastore
    try:
        image_store = cfg.exp.get_image_store()
    except Exception:
        logger.warning('Image data not available (processor is likely still running)')

    # Extract necessary data from experiment view stores
    view.build_all(cfg.exp, force=cfg.force_view_calculation)
    store = cfg.exp.get_view_store()
    for k in [KEY_APT, KEY_APT_SUMMARY, KEY_ACQ, KEY_CELL]:
        cache[k] = store.get(k)

    if KEY_ARRAY not in cache:
        cache[KEY_ARRAY] = _get_array_data(cache[KEY_APT_SUMMARY])

    if KEY_ARRAY_SUMMARY not in cache:
        cache[KEY_ARRAY_SUMMARY] = _get_array_summary_data(cache[KEY_ACQ], cache[KEY_ARRAY])

    # Save any cached objects that haven't already been saved
    save(overwrite=False)

    # Remove OOB apartments if configured to do so
    if cfg.remove_oob_address:
        logger.info('Experiment data loaded with OOB apartment addresses removed')
        for k in [KEY_CELL, KEY_APT, KEY_APT_SUMMARY]:
            cache[k] = view.remove_oob_apartments(cfg.exp, cache[k])
    else:
        logger.info('Experiment data loaded without removing OOB apartment addresses')


def get_apartment_image_data(df, marker_color=visualization.COLOR_RED):
    key_fields = get_apartment_key_fields()

    # Index growth data and use unique index values to subset other datasets
    idx = df.set_index(key_fields).index.drop_duplicates()

    # Select apartment/cell data based on the given growth data
    apt_data = cache[KEY_APT].set_index(key_fields).loc[idx].reset_index().copy()
    cell_data = cache[KEY_CELL].set_index(key_fields)
    if len(idx.intersection(cell_data.index)) > 0:
        cell_data = cell_data.loc[idx].reset_index().copy()
    else:
        cell_data = pd.DataFrame()

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
        row['hours'] = g['elapsed_hours_group'].tolist()
        row['cell_counts'] = g['cell_count'].tolist()
        res.append(row)
    res = pd.DataFrame(res)

    res['key'] = res.apply(get_apartment_key, axis=1)
    res = res.set_index('key')
    return res
