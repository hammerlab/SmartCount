
import os
import os.path as osp
import numpy as np
import pandas as pd
from celldom.core import cytometry
from celldom.core import modeling
import logging

logger = logging.getLogger(__name__)


def get_experiment_start_dates(apt_data, exp_cond_fields):
    return apt_data.groupby(exp_cond_fields)['acq_datetime'].min().rename('experiment_start_date')


def get_experiment_elapsed_hours(apt_data, exp_cond_fields):
    # Get group -> min date index
    exp_start_dates = get_experiment_start_dates(apt_data, exp_cond_fields)

    # Map min dates to given data frame, giving a N [= len(apt_data)] vector of start dates
    exp_start_dates = apt_data.set_index(exp_cond_fields).index.to_series().map(exp_start_dates)

    # Return computed time since beginning of experiment
    return (apt_data['acq_datetime'].values - exp_start_dates.values) / np.timedelta64(1, 'h')


def get_growth_rate_data(apt_data, exp_cond_fields, cell_data=None, occupancy_threshold=.5):

    df = apt_data.copy()

    # If cell data was provided, replace apartment counts with the data provided (the cell data may have been
    # filtered in a way that should be reflected in apartment cell counts)
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
    df['elapsed_hours'] = get_experiment_elapsed_hours(df, exp_cond_fields)

    # Regroup by apartment + time and compute median cell count across all measurements in time
    # (and retain sets of unique acquisition ids as this is very helpful for tracing data back to images)
    df = df.groupby(exp_cond_fields + ['st_num', 'apt_num', 'elapsed_hours', 'acq_datetime'])\
        .agg({
            'cell_count': 'median',
            'occupancy_chamber': 'median',
            'acq_id': (lambda x: set(x))
        }).reset_index()

    def grm(g):
        gts = g.set_index('acq_datetime')[['cell_count', 'occupancy_chamber']].sort_index()
        tsct, tso = gts['cell_count'], gts['occupancy_chamber']

        # Determine "confluence" marker as when previous measurement occupancy is beyond a threshold
        # and the current count is less than the previous (which when true, is always true forward in time)
        # Note: any inequality operators evaluated against NaN result in False, as is desired here
        tsconf = ((tsct.diff() <= 0) & (tso.shift() >= occupancy_threshold)).cummax()

        # Set mask used to select time points for growth modeling
        vm = ~tsconf.values

        # Compute growth rate estimation and other useful statistics (including timeseries)
        return pd.Series({
            'growth_rate': modeling.get_growth_rate(g[vm]['elapsed_hours'] / 24, g[vm]['cell_count']),
            'max_cell_count': tsct.max(),
            'min_cell_count': tsct.min(),
            'first_date': tsct.index[0],
            'last_date': tsct.index[-1],
            'first_count': tsct.iloc[0],
            'last_count': tsct.iloc[-1],
            'n': len(tsct),
            'elapsed_hours_min': g['elapsed_hours'].min(),
            'cell_counts': tsct.to_dict(),
            'occupancies': tso.to_dict(),
            'confluence': tsconf.to_dict(),
            # Flatten the array of acquisition id sets back into a single set
            'acq_ids': set([acq_id for acq_ids in g['acq_id'].values for acq_id in acq_ids])
        })

    # Regroup by st/apt alone (with experimental conditions) and compute growth rates
    grmcols = ['elapsed_hours', 'acq_datetime', 'cell_count', 'occupancy_chamber', 'acq_id']
    df = df.groupby(exp_cond_fields + ['st_num', 'apt_num'])[grmcols].apply(grm).reset_index()

    return df


