# Utility methods for building result databases based on processing data
#
# Note that in contrast to the `analysis` module, every method provided here should accept nothing more
# than an `experiment` instance as a means of specifying all necessary parameters.  Any configurable behavior
# should be controlled by external files (available via the experiment instance) rather than runtime arguments
# so that these functions may be referenced from any type of application without concern for consistency between calls
from celldom.execute import analysis
import os.path as osp
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

VIEW_RESULT_FILE = 'views.h5'
VIEW_ACQUISITION = 'acquisition'
VIEW_CELL = 'cell'
VIEW_APARTMENT = 'apartment'
VIEW_APARTMENT_SUMMARY = 'apartment_summary'


def get_store_path(data_dir, data_file=VIEW_RESULT_FILE):
    return osp.join(data_dir, data_file)


def get_readonly_views(data_dir, data_file=VIEW_RESULT_FILE):
    return pd.HDFStore(get_store_path(data_dir, data_file=data_file), 'r')


def add_measurement_times(experiment, df):
    """Infer time periods associated with imaging intervals"""
    exp_cond_fields = experiment.config.experimental_condition_fields
    analysis_config = experiment.config.get_analysis_config()
    min_measurement_gap_seconds = analysis_config['measurement_inference']['min_measurement_gap_seconds']
    df = analysis.add_experiment_date_groups(df, exp_cond_fields, min_measurement_gap_seconds)
    return df


def remove_oob_apartments(experiment, df):
    """Apply any apt/st number range filtering to ignore any misclassified apartments"""
    if 'st_num' in df and 'apt_num' in df:
        chip_config = experiment.config.get_chip_config()
        mask = df['st_num'].astype(float).between(*chip_config['st_num_range']) & \
            df['apt_num'].astype(float).between(*chip_config['apt_num_range'])
        mask = mask.values
        n_remove = np.sum(~mask)
        if n_remove > 0:
            n = len(df)
            logger.info(
                'Removing %s rows of %s from data frame with apartment/street numbers outside of '
                'expected range (apartment range = %s, street range = %s)',
                n_remove, n, chip_config['apt_num_range'], chip_config['st_num_range']
            )
            df = df.loc[mask]
            assert len(df) == n - n_remove
    return df


def get_acquisition_view(experiment):
    logger.info('Building acquisition view')
    df = experiment.get_data_store().get('acquisition')
    return add_measurement_times(experiment, df)


def merge_measurement_times(df, df_acq):
    fields = analysis.DATE_GROUP_FIELDS
    assert df_acq['acq_id'].is_unique
    for f in fields:
        assert f not in df
    df = pd.merge(df, df_acq[['acq_id'] + fields], on='acq_id', how='left')
    assert df[fields].notnull().all().all()
    return df


def get_cell_count_field(type, component):
    return 'cell_count_{}_{}'.format(type, component)


def get_occupancy_field(component):
    return 'occupancy_{}'.format(component)


def get_cell_view(experiment, acquisition_data):
    logger.info('Building cell view')
    df_acq = acquisition_data
    df = experiment.get_data_store().get('cell')
    df = merge_measurement_times(df, df_acq)

    ################
    # Cell Selection
    ################

    # TODO: Add configuration property to potentially allow for filtering of bad
    # cells (probably based on size) here as it is early enough to affect everything
    # else downstream

    # Collapse multiple measurements for the same apartment + date grouping, using
    # the one with the most cells in the event of multiple captures
    apt_addr_fields = experiment.config.apartment_address_fields
    apt_meas_fields = apt_addr_fields + ['acq_datetime_group']
    df_ct = df.groupby(apt_meas_fields + ['acq_id', 'apt_id']).size().rename('count').reset_index()
    df_keep = pd.merge(
        df_ct.groupby(apt_meas_fields).size().rename('num_measurements').reset_index(),
        df_ct.sort_values(apt_meas_fields + ['count']).drop('count', axis=1)
        .drop_duplicates(subset=apt_meas_fields, keep='last')
    )
    df = pd.merge(df, df_keep)

    #####################
    # Cell Classification
    #####################
    analysis_config = experiment.config.get_analysis_config()
    mode = analysis_config.cell_classification_mode
    if mode != 'default':
        raise ValueError('Cell classification mode "{}" not valid'.format(mode))
    df['is_type_any'] = True

    return df


def get_apartment_view(experiment, acquisition_data, cell_data):
    logger.info('Building apartment view (this may take a few minutes)')
    df_acq, df_cell = acquisition_data, cell_data

    df = experiment.get_data_store().get('apartment')
    df = merge_measurement_times(df, df_acq)

    apt_addr_fields = experiment.config.apartment_address_fields
    analysis_config = experiment.config.get_analysis_config()

    # Add the number of measurements taken for each apartment + date combination
    df_ct = df.groupby(apt_addr_fields + ['acq_datetime_group']).size().rename('num_measurements').reset_index()
    n = len(df)
    df = pd.merge(df, df_ct)
    assert n == len(df), 'Measurement count addition resulted in a loss of records'

    # Sum cell counts by apartment for combinations of component and cell type
    def summarize(g):
        gs = g.groupby(['type', 'component']).size()
        gs.index = [
            get_cell_count_field(i[0].replace('is_type_', ''), i[1].replace('in_', ''))
            for i in gs.index
        ]
        return gs
    df_ct = df_cell.assign(
        type=df_cell.filter(regex='^is_type_.*').idxmax(axis=1),
        component=df_cell.filter(regex='^in_.*').idxmax(axis=1)
    ).groupby(['acq_id', 'apt_id']).apply(summarize).reset_index()

    # Merge counts back into original apartment data
    df = pd.merge(df, df_ct, on=['acq_id', 'apt_id'], how='left')

    # Replace null counts with zeros as it is possible for apartments to have data
    # but no associated cell records
    ct_cols = df.filter(regex='^cell_count_.*').columns
    df[ct_cols] = df[ct_cols].fillna(0)

    # Replace raw cell count with sum across classified cell count fields
    df['cell_count'] = df.filter(regex='^cell_count_.*').sum(axis=1)

    # Collapse multiple measurements for the same datetime group, preferring captures
    # with the highest raw cell count in the event of multiple
    # NOTE: There should not be many groups with multiple records since the same
    # collapsing is applied to cell data, though it will be more common with empty apartments
    df = df.sort_values(apt_addr_fields + ['acq_datetime_group', 'cell_count'])\
        .drop_duplicates(subset=apt_addr_fields + ['acq_datetime_group'], keep='last')

    # Add apartment classifications
    init_cond_field = get_cell_count_field(**analysis_config.apartment_classification_cell_class)

    def classify(g):
        cts = g[g['elapsed_hours_group'] == 0][init_cond_field]
        assert len(cts) <= 1, 'Found multiple measurements for time zero (data = {})'.format(cts)
        if len(cts) == 0:
            initial_condition = 'no_time_zero_data'
        else:
            initial_condition = {
                0: 'no_cell',
                1: 'single_cell',
                2: 'double_cell',
                3: 'triple_cell'
            }.get(cts.iloc[0], 'many_cells')
        g['initial_condition'] = initial_condition
        return g
    df = df.groupby(apt_addr_fields, group_keys=False).apply(classify)
    return df


def get_apartment_summary_view(experiment, acquisition_data, apartment_data):
    logger.info('Building apartment summary view (this may take a few minutes)')
    df_acq, df_apt = acquisition_data, apartment_data.copy()

    # Collect relevant calculation settings/parameters
    analysis_config = experiment.config.get_analysis_config()
    exp_cond_fields = experiment.config.experimental_condition_fields
    apt_addr_fields = experiment.config.apartment_address_fields

    df = analysis.get_growth_rate_data(
        df_apt, exp_cond_fields,
        cell_count_field=get_cell_count_field(**analysis_config.apartment_summary_cell_class),
        occupancy_field=get_occupancy_field(analysis_config.confluence_detection_component),
        occupancy_threshold=analysis_config.confluence_detection_threshold
    )

    cond_filter = analysis_config.apartment_summary_initial_condition
    if cond_filter:
        apt_idx = df_apt[df_apt['initial_condition'].isin(cond_filter)]\
            .set_index(apt_addr_fields).index.unique()
        if len(apt_idx) == 0:
            raise ValueError(
                'Apartment initial conditions filter "{}" did not match any apartment records -- '
                'either none of these initial conditions occurred or the condition names are invalid'
                .format(cond_filter)
            )

        # Compute boolean mask with length == len(df) where True indicates that the associated apartment matches
        # the specified initial condition filters (note that the result here is a numpy array)
        mask_match = df.set_index(apt_addr_fields).index.isin(apt_idx)

        # Set all growth rates for apartments NOT matching initial conditions to nan
        df['growth_rate'] = np.where(mask_match, df['growth_rate'], np.nan)
        logger.info(
            'Invalidating growth rates for %s apartments (of %s) due to lack of match to initial condition filter "%s"',
            len(df) - mask_match.sum(), len(df), cond_filter
        )

    return df


def build(experiment, fn, key, save=True, force=False, **kwargs):
    """Run a calculation cached as experimental results

    The behavior of this function is to either fetch previously computed results if they
    already exist under the "key" for each calculation in a single hdf5 database, or to compute
    those results and then save them under that same key.

    Args:
        experiment: Experiment instance
        fn: Function associated with calculation; must produce a single data frame
        key: String key associated with calculation; should match [_a-zA-Z0-9]+ for hdf5 keys
        save: True if any persistence/caching is desired (otherwise, nothing will ever be saved)
        force: Forces recalculation, regardless of any cached results present, if set to true
        kwargs: Keyword arguments passed to `fn` (note that `experiment` will be passed as the sole positional argument)
    Returns:
        Data frame resulting from calculation
    """
    path = get_store_path(experiment.data_dir)
    if save and not force:
        # If not explicitly told to overwrite results, try to fetch them for the "key" associated
        # with this calculation, if possible (to avoid recalculation)
        if osp.exists(path) and not force:
            try:
                df = pd.read_hdf(path, key=key)
                logger.debug('Retrieved results for calculation "%s" from path "%s"', key, path)
                return df
            except KeyError:
                pass
    df = fn(experiment, **kwargs)
    if save:
        logger.info('Saving results for calculation "%s" to path "%s"', key, path)
        # Arguments here will append to an existing hdf5 file if present, overwriting
        # whatever data frame is currently associated with the key (if any)
        df.to_hdf(path, key=key, mode='a', append=False)

    return df


def build_all(experiment, force=False):
    save = True
    df_acq = build(
        experiment, get_acquisition_view, VIEW_ACQUISITION,
        save=save, force=force
    )
    df_cell = build(
        experiment, get_cell_view, VIEW_CELL,
        save=save, force=force, acquisition_data=df_acq
    )
    df_apt = build(
        experiment, get_apartment_view, VIEW_APARTMENT,
        save=save, force=force, acquisition_data=df_acq, cell_data=df_cell
    )
    build(
        experiment, get_apartment_summary_view, VIEW_APARTMENT_SUMMARY,
        save=save, force=force, acquisition_data=df_acq, apartment_data=df_apt
    )