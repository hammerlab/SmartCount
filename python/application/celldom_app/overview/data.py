import os
import glob
import json
import os.path as osp
import numpy as np
import pandas as pd
from celldom.execute import analysis
from celldom.execute import visualization
from celldom.core import cytometry
from celldom_app.overview import config
from cvutils.encoding import base64_encode_image
from skimage import color
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
KEY_APT_IMG_DATA = 'apartment_image_data'
SAVE_KEYS = [KEY_GROWTH_DATA, KEY_APT_IMG_DATA]


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


def get_apartment_key(row):
    return ':'.join(row[get_apartment_key_fields()])


def get_apartment_key_fields():
    return cfg.experimental_condition_fields + ['apt_num', 'st_num']


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
    store = cytometry.get_readonly_datastore(data_dir)
    cache[KEY_CELL_DATA] = store.get('cell')
    cache[KEY_APT_DATA] = store.get('apartment')
    cache[KEY_ACQ_DATA] = store.get('acquisition')

    # Initialize image datastore
    image_store = cytometry.get_readonly_images(data_dir)

    if KEY_GROWTH_DATA not in cache:
        logger.info('Computing growth rate statistics (this may take 60 seconds or so the first time)')
        df = analysis.get_growth_rate_data(cache[KEY_APT_DATA], cfg.experimental_condition_fields)

        # Convert any non-string objects to json strings to avoid plotly serialization errors
        df['cell_counts'] = df['cell_counts'].apply(lambda m: json.dumps({str(k): v for k, v in m.items()}))
        df['acq_ids'] = df['acq_ids'].apply(lambda v: json.dumps(list(v)))
        cache[KEY_GROWTH_DATA] = df

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

# def get_apartment_image_data(df):
#     if KEY_APT_IMG_DATA not in cache:
#         cache[KEY_APT_IMG_DATA] = pd.DataFrame()
#     res = cache[KEY_APT_IMG_DATA]
#
#     keys = df.apply(get_apartment_key, axis=1)
#     in_cache = np.array([k in res.index for k in keys])
#
#     if np.all(in_cache):
#         return res
#
#     df_new = _get_apartment_image_data(df[~in_cache])
#     df_new['key'] = df_new.apply(get_apartment_key, axis=1)
#     df_new = df_new.set_index('key')
#
#     cache[KEY_APT_IMG_DATA] = df_new if len(res) == 0 else res.append(df_new)
#
#     save(overwrite=True, keys=[KEY_APT_IMG_DATA])
#     return cache[KEY_APT_IMG_DATA]

# def _get_apartment_image_data(df):
#
#     files = get_acquisition_data().set_index('acq_id')['raw_image_path'].to_dict()
#     files = [files[acq_id] for acq_ids in df['acq_ids'] for acq_id in acq_ids]
#     files = list(set(files))
#
#     logger.info('Generating apartment images for %s raw image files ...', len(files))
#     output_dir = cfg.apt_img_tmpdir
#     image_data = visualization.get_apartment_image_data(cfg.exp_config, files, output_dir)
#     logger.info('Apartment image generation complete')
#
#     fields = get_apartment_key_fields()
#
#     idx = df.set_index(fields).index
#
#     res = []
#     for k, g in image_data.groupby(fields):
#         # Ignore images related to exp conditions + st/apt numbers not explicitly provided
#         if k not in idx:
#             continue
#         g = g.sort_values('acq_datetime')
#         row = {fields[i]: k[i] for i in range(len(k))}
#         row['n'] = len(g)
#         row['encoded_images'] = [base64_encode_image(img) for img in g['image']]
#         row['dates'] = g['acq_datetime'].tolist()
#         row['cell_counts'] = g['cell_count'].tolist()
#         res.append(row)
#     return pd.DataFrame(res)

