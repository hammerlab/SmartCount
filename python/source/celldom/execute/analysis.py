
import os
import os.path as osp
import numpy as np
import pandas as pd
from celldom.core import cytometry
from celldom.core import modeling
import logging

logger = logging.getLogger(__name__)

DEFAULT_DATE_GROUP_GAP_SECONDS = 3600
DATE_GROUP_FIELDS = ['elapsed_hours', 'elapsed_hours_group', 'acq_datetime_group']
GROWTH_RATE_DICT_FIELDS = ['cell_counts', 'hours', 'dates', 'occupancies', 'confluence']
GROWTH_RATE_LIST_FIELDS = ['acq_ids']
GROWTH_RATE_OBJ_FIELDS = GROWTH_RATE_DICT_FIELDS + GROWTH_RATE_LIST_FIELDS


def get_experiment_date_groups(dates, min_gap_seconds=DEFAULT_DATE_GROUP_GAP_SECONDS):
    # Make sure dates are sorted
    if not dates.is_monotonic_increasing:
        dates = dates.sort_values()

    # Make sure dates are also unique
    if not dates.is_unique:
        dates = dates.drop_duplicates()

    # Create a new group index each time the difference between steps exceeds the given threshold (in seconds)
    groups = (dates.diff().dt.total_seconds() >= min_gap_seconds).cumsum()

    # Get the minimum date for each group and then get a vector of len(dates) containaing the group date
    # for each original date
    groups = groups.map(dates.groupby(groups).min())

    # Return a series mapping the original dates to the grouped date
    return pd.Series(groups.values, index=dates.values)


def add_experiment_date_groups(df, exp_cond_fields, min_gap_seconds=DEFAULT_DATE_GROUP_GAP_SECONDS):
    """Group dates and elapsed time fields by inferred date groups

    This often produces time period groupings that are much more meaningful than raw dates rounded down to a less
    granular frequency.
    """
    # Add elapsed hours if not present
    if 'elapsed_hours' not in df:
        df['elapsed_hours'] = get_experiment_elapsed_hours(df, exp_cond_fields)

    # Compute date groupings for each experimental condition group and concatenate the results
    res = []
    for _, g in df.groupby(exp_cond_fields):
        dg = g.copy()
        date_map = get_experiment_date_groups(dg['acq_datetime'], min_gap_seconds)
        dg['acq_datetime_group'] = dg['acq_datetime'].map(date_map)
        dg['elapsed_hours_group'] = dg['acq_datetime_group'].map(
            dg.groupby('acq_datetime_group')['elapsed_hours'].min()).astype(int)
        res.append(dg)
    return pd.concat(res).reset_index(drop=True)


def get_experiment_start_dates(df, exp_cond_fields):
    return df.groupby(exp_cond_fields)['acq_datetime'].min().rename('experiment_start_date')


def get_experiment_elapsed_hours(df, exp_cond_fields):
    # Get group -> min date index
    exp_start_dates = get_experiment_start_dates(df, exp_cond_fields)

    # Map min dates to given data frame, giving a N [= len(apt_data)] vector of start dates
    exp_start_dates = df.set_index(exp_cond_fields).index.to_series().map(exp_start_dates)

    # Return computed time since beginning of experiment
    # pylint: disable=too-many-function-args
    return (df['acq_datetime'].values - exp_start_dates.values) / np.timedelta64(1, 'h')


def get_growth_rate_data(apt_data, exp_cond_fields,
                         cell_count_field='cell_count', occupancy_field='occupancy', occupancy_threshold=.5,
                         fit_intercept=True):
    df = apt_data.copy()

    def grm(g):
        gts = g.set_index('acq_datetime_group').sort_index()
        assert gts.index.is_unique, \
            'Apartment "{}" has cell count timeseries with duplicated dates: Timeseries = {}'\
            .format(g[['acq_id', 'apt_id', 'st_num', 'apt_num']].iloc[0].to_dict(), gts.to_dict())
        tsct, tso = gts[cell_count_field], gts[occupancy_field]
        tsh, tsd = gts['elapsed_hours_group'], gts['acq_datetime']

        # Get time zero counts for all cell types
        ctz = gts.filter(regex='^cell_count_.*').rename(columns=lambda c: c.replace('cell_', 'tz_')).loc[tsh == 0]
        ctz = ctz.iloc[0].to_dict() if len(ctz) > 0 else {c: np.nan for c in ctz.columns}

        # Determine "confluence" marker as when previous measurement occupancy is beyond a threshold
        # and the current count is less than the previous (which when true, is always true forward in time)
        # Note: any inequality operators evaluated against NaN result in False, as is desired here
        tsconf = ((tsct.diff() <= 0) & (tso.shift() >= occupancy_threshold)).cummax()

        # Set mask used to select time points for growth modeling
        vm = ~tsconf.values

        # Compute growth rate estimation and other useful statistics (including timeseries)
        growth_rate_0, growth_rate_1 = modeling.get_growth_rate(
            g[vm]['elapsed_hours'] / 24, g[vm][cell_count_field], fit_intercept=fit_intercept)
        res = {
            'growth_rate': growth_rate_1,
            'growth_rate_intercept': growth_rate_0,
            'max_count': tsct.max(),
            'min_count': tsct.min(),
            'first_hour': tsh.iloc[0],
            'last_hour': tsh.iloc[-1],
            'first_date': tsd.iloc[0],
            'last_date': tsd.iloc[-1],
            'n': len(tsct),
            'initial_condition': gts['initial_condition'].iloc[0],
            'hours': tsh.to_json(),
            'dates': tsd.to_json(),
            'cell_counts': tsct.to_json(),
            'occupancies': tso.to_json(),
            'confluence': tsconf.to_json(),
            # Flatten the array of acquisition id sets back into a single set
            'acq_ids': pd.Series([
                acq_id for acq_ids in g['acq_id'].values for acq_id in acq_ids
            ]).drop_duplicates().to_json()
        }
        res.update(ctz)
        return pd.Series(res)

    # Regroup by st/apt alone (with experimental conditions) and compute growth rates
    df = df.groupby(exp_cond_fields + ['st_num', 'apt_num']).apply(grm).reset_index()

    return df


