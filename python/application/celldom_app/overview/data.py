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
from celldom_app.overview import config
from cvutils.encoding import base64_encode_image
from skimage import color
from tables.exceptions import HDF5ExtError
import logging

logger = logging.getLogger(__name__)

cfg = None
image_store = None
cache_dir = None
cache = {}

KEY_GROWTH_DATA = 'growth_data'
KEY_APT_DATA = 'apartment_data'
KEY_ACQ_DATA = 'acquisition_data'
KEY_CELL_DATA = 'cell_data'
KEY_ARRAY_DATA = 'array_data'
KEY_APT_IMG_DATA = 'apartment_image_data'
SAVE_KEYS = [KEY_GROWTH_DATA, KEY_APT_IMG_DATA]
STORE_READ_MAX_ATTEMPTS = 10
STORE_READ_PAUSE_SECS = 5
KEY_SEP = ':'


def save(overwrite=False, keys=SAVE_KEYS):
    for k in keys:
        if k not in cache:
            continue
        f = osp.join(cache_dir, k + '.h5')
        if not osp.exists(f) or overwrite:
            logger.info('Saving cached object "%s" to path "%s"', k, f)
            cache[k].to_hdf(f, key='data')


def restore():
    for f in glob.glob(osp.join(cache_dir, '*.h5')):
        k = osp.basename(f).split('.')[0]
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
    return cfg.experimental_condition_fields + ['apt_num', 'st_num']


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


def _clean(df):
    """Apply any global cleaning/filtering"""
    if 'st_num' in df and 'apt_num' in df:
        chip_config = cfg.exp_config.get_chip_config()
        mask = df['st_num'].astype(float).between(*chip_config['st_num_range']) & \
               df['apt_num'].astype(float).between(*chip_config['apt_num_range'])
        mask = mask.values
        n_remove = np.sum(~mask)
        if n_remove > 0:
            n = len(df)
            logger.info(
                'Removing %s rows of %s from dataframe with apartment/street numbers outside of '
                'expected range (apartment range = %s, street range = %s)',
                n_remove, n, chip_config['apt_num_range'], chip_config['st_num_range']
            )
            df = df.loc[mask]
            assert len(df) == n - n_remove
    return df


def initialize(data_dir):
    global cfg
    global image_store
    cfg = config.get()

    global cache_dir
    logger.info('Initializing app data for data directory "%s"', data_dir)

    cache_dir = osp.join(data_dir, 'app', 'overview')
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    restore()

    # Extract necessary data from experiment output store
    ct = 0
    while ct < STORE_READ_MAX_ATTEMPTS:
        try:
            store = cytometry.get_readonly_datastore(data_dir)
            if cfg.remove_oob_address:
                logger.info('Loading experiment data with OOB apartment addresses removed')
                cache[KEY_CELL_DATA] = _clean(store.get('cell'))
                cache[KEY_APT_DATA] = _clean(store.get('apartment'))
            else:
                logger.info('Loading experiment data without removing OOB apartment addresses')
                cache[KEY_CELL_DATA] = store.get('cell')
                cache[KEY_APT_DATA] = store.get('apartment')
            cache[KEY_ACQ_DATA] = store.get('acquisition')
            break
        except (AttributeError, HDF5ExtError):
            ct += 1
            logger.warning(
                'Failed to open cytometry datastore (processor may still be running); '
                'will pause for %s second(s) and try again (attempt %s of %s)',
                STORE_READ_PAUSE_SECS, ct, STORE_READ_MAX_ATTEMPTS
            )
            time.sleep(STORE_READ_PAUSE_SECS)
    if ct >= STORE_READ_MAX_ATTEMPTS:
        raise ValueError(
            'Failed to open experiment datastore after {} attempts (processor may still be running)'
            .format(STORE_READ_MAX_ATTEMPTS)
        )

    # Initialize image datastore
    try:
        image_store = cytometry.get_readonly_images(data_dir)
    except Exception:
        logger.warning('Image data not available (processor is likely still running)')

    if KEY_GROWTH_DATA not in cache:
        logger.info('Computing growth rate statistics (this may take 60 seconds or so the first time)')
        df = analysis.get_growth_rate_data(cache[KEY_APT_DATA], cfg.experimental_condition_fields)

        # Convert any non-string objects to json strings to avoid plotly serialization errors
        df['cell_counts'] = df['cell_counts'].apply(lambda m: json.dumps({str(k): v for k, v in m.items()}))
        df['acq_ids'] = df['acq_ids'].apply(lambda v: json.dumps(list(v)))
        cache[KEY_GROWTH_DATA] = df

    if KEY_ARRAY_DATA not in cache:
        logger.info('Computing array summary data')
        df = cache[KEY_GROWTH_DATA]
        df = df.groupby(cfg.experimental_condition_fields).size().rename('num_apartments').reset_index()
        cache[KEY_ARRAY_DATA] = df

    # Save any cached objects that haven't already been saved
    save(overwrite=False)


def get_apartment_image_data(df):
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

    image_data = visualization.process_results(
        apt_data, cell_data, cfg.experimental_condition_fields, visualization.COLOR_RED
    )

    res = []
    for k, g in image_data.groupby(key_fields):
        g = g.sort_values('acq_datetime')
        row = {key_fields[i]: k[i] for i in range(len(k))}
        row['n'] = len(g)
        row['encoded_images'] = [base64_encode_image(img) for img in g['image']]
        row['dates'] = g['acq_datetime'].tolist()
        row['cell_counts'] = g['cell_count'].tolist()
        res.append(row)
    res = pd.DataFrame(res)

    res['key'] = res.apply(get_apartment_key, axis=1)
    res = res.set_index('key')
    return res
