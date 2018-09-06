
import os
import os.path as osp
import numpy as np
import pandas as pd
from celldom.core import cytometry
from celldom.core import modeling
import logging

logger = logging.getLogger(__name__)


def get_growth_rate_data(apt_data, exp_cond_fields, cell_data=None, start_of_experiment=None):

    df = apt_data

    # If cell data was provided, replace apartment counts with the data provided
    if cell_data is not None:

        # Group by original image id, id of apartment within that image, and time to compute new cell count
        dfct = cell_data.groupby(['acq_id', 'apt_id', 'acq_datetime']).size().rename('cell_count').reset_index()

        # Merge apt counts onto dynamic cell counts and then keep only rows that have either zero counts
        # or some value in the dynamic counts
        assert len(dfct.columns.difference(df.columns)) == 0, \
            'Apartment data does not have required fields "{}"'.format(dfct.columns)
        df = df.rename(columns={'cell_count': 'original_cell_count'})
        df = pd.merge(df, dfct, how='left', on=dfct.columns.values)
        df['cell_count'] = np.where(np.isclose(df['original_cell_count'], 0), 0, df['cell_count'])
        df = df[df['cell_count'].notnull()].drop('original_cell_count', axis=1)

    # Infer experiment start dates if not given, based on earliest timestamps in data
    if start_of_experiment is None:
        dfdt = df.groupby(exp_cond_fields)['acq_datetime'].min().rename('min_date')
        exp_start_dates = df.set_index(exp_cond_fields).index.to_series().map(dfdt)
    # Otherwise, assume the given date is specific to all experimental conditions
    else:
        exp_start_dates = pd.Series([pd.to_datetime(start_of_experiment)])

    # Compute elapsed time since beginning of experiment
    df['elapsed_hours'] = (df['acq_datetime'].values - exp_start_dates.values) / np.timedelta64(1, 'h')

    # Regroup by apartment + time and compute median cell count across all measurements in time
    # (and retain sets of unique acquisition ids as this is very helpful for tracing data back to images)
    df = df.groupby(exp_cond_fields + ['st_num', 'apt_num', 'elapsed_hours', 'acq_datetime'])\
        .agg({'cell_count': 'median', 'acq_id': (lambda x: set(x))}).reset_index()

    def grm(g):
        tsct = g.set_index('acq_datetime')['cell_count'].sort_index()
        # Compute growth rate estimation and other useful statistics (including count timeseries)
        return pd.Series({
            'growth_rate': modeling.get_growth_rate(g['elapsed_hours'] / 24, g['cell_count']),
            'max_cell_count': tsct.max(),
            'min_cell_count': tsct.min(),
            'first_date': tsct.index[0],
            'last_date': tsct.index[-1],
            'first_count': tsct.iloc[0],
            'last_count': tsct.iloc[-1],
            'n': len(tsct),
            'elapsed_hours_min': g['elapsed_hours'].min(),
            'cell_counts': tsct.to_dict(),
            # Flatten the array of acquisition id sets back into a single set
            'acq_ids': set([acq_id for acq_ids in g['acq_id'].values for acq_id in acq_ids])
        })

    # Regroup by st/apt alone (with experimental conditions) and compute growth rates
    grmcols = ['elapsed_hours', 'acq_datetime', 'cell_count', 'acq_id']
    df = df.groupby(exp_cond_fields + ['st_num', 'apt_num'])[grmcols].apply(grm).reset_index()

    return df


