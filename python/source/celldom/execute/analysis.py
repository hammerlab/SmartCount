
import os
import os.path as osp
import numpy as np
import pandas as pd
from celldom.core import cytometry
from celldom.core import modeling
import logging

logger = logging.getLogger(__name__)


def get_growth_rate_data(cell_data, strata_cols, start_of_experiment=None, min_cell_size=None, max_cell_size=None):
    df = cell_data
    if start_of_experiment is not None:
        start_of_experiment = pd.to_datetime(start_of_experiment)
    else:
        start_of_experiment = df['acq_datetime'].min()
        logger.info(
            'Experiment start time inferred as "{}" (make sure this is close to the actual starting datetime)'
            .format(start_of_experiment)
        )

    # Remove cells outside of size filter
    if min_cell_size is not None or max_cell_size is not None:
        df = df[df['area'].between(min_cell_size, max_cell_size)]

    # Group by what is effectively the original image id (acq_id) and location of apartment within image (apt_id) --
    # everything else included is less unique than these things but is necessary for downstream processing
    df = df.groupby(strata_cols + ['acq_id', 'apt_id', 'st_num', 'apt_num', 'acq_datetime'])

    # Count number of cells
    df = df.size().rename('cell_count').reset_index()

    # Compute time elapsed since start of experiment
    df['elapsed_hours'] = (df['acq_datetime'] - start_of_experiment) / np.timedelta64(1, 'h')

    # Regroup by apartment + time
    df = df.groupby(strata_cols + ['st_num', 'apt_num', 'elapsed_hours'])
    # Compute median cell count across all measurements
    df = df.agg({'cell_count': 'median'}).reset_index()

    # Compute growth rates (in days)
    df = df.groupby(strata_cols + ['st_num', 'apt_num']) \
        .apply(lambda g: modeling.get_growth_rate(g['elapsed_hours'] / 24, g['cell_count'])) \
        .rename('growth_rate').reset_index()

    return df


