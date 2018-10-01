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

CALC_RESULT_FILE = 'results.h5'
CALC_APT_GROWTH_RATE = 'apartment_growth_rate'
CALC_APT_CLASSIFICATION = 'apartment_classification'


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


def add_measurement_times(experiment, df):
    """Infer time periods associated with imaging intervals"""
    exp_cond_fields = experiment.config.experimental_condition_fields
    analysis_config = experiment.config.get_analysis_config()
    min_measurement_gap_seconds = analysis_config['measurement_inference']['min_measurement_gap_seconds']
    df = analysis.add_experiment_date_groups(df, exp_cond_fields, min_measurement_gap_seconds)
    return df


def run_calculation(experiment, fn, key, save=True, overwrite=False, **kwargs):
    """Run a calculation cached as experimental results

    The behavior of this function is to either fetch previously computed results if they
    already exist under the "key" for each calculation in a single hdf5 database, or to compute
    those results and then save them under that same key.

    Args:
        experiment: Experiment instance
        fn: Function associated with calculation; must produce a single data frame
        key: String key associated with calculation; should match [_a-zA-Z0-9]+ for hdf5 keys
        save: True if any persistence/caching is desired (otherwise, nothing will ever be saved)
        overwrite: Forces recalculation and caching if set to true
        kwargs: Keyword arguments passed to `fn` (note that `experiment` will be passed as the sole positional argument)
    Returns:
        Data frame resulting from calculation
    """
    path = osp.join(experiment.data_dir, CALC_RESULT_FILE)
    if save and not overwrite:
        # If not explicitly told to overwrite results, try to fetch them for the "key" associated
        # with this calculation, if possible (to avoid recalculation)
        if osp.exists(path) and not overwrite:
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


def _calculate_apartment_classifications(experiment):
    logger.info('Computing apartment classifications (this may take a couple minutes the first time)')
    store = experiment.get_data_store()
    df = store.get('apartment')
    return __calculate_apartment_classifications(df, experiment.config.apartment_address_fields)


def __calculate_apartment_classifications(df, apt_addr_fields):
    def classify(g):
        gs = g.sort_values('acq_datetime')

        ct0 = gs.iloc[0]['cell_count']
        initial_condition = 'other'
        if ct0 == 0:
            initial_condition = 'no_cell'
        if ct0 == 1:
            initial_condition = 'single_cell'
        if ct0 == 2:
            initial_condition = 'double_cell'
        if ct0 == 3:
            initial_condition = 'triple_cell'

        return pd.Series({
            'initial_condition': initial_condition
        })
    return df.groupby(apt_addr_fields).apply(classify).reset_index()


def calculate_apartment_classifications(experiment, save=True, overwrite=False):
    """Compute apartment "classifications" based on a variety of possible conditions

    Apartment "classifications" might include things like:
    - Whether or not only a single cell was present at time zero
    - Whether or not there was EVER a cell present in an apartment
    - Whether or not there was ever a decrease in cell count across the measurment period

    At TOW, only the single cell indicator is available but the intention of this function is to tag
    apartments with individual categorizations useful in other scenarios where repeating the same
    classification logic is burdensome.

    Note that the resulting data frame contains all apartment address fields but no date/time fields as
    classifications are specific to apartments, but not to individual measurement intervals.
    """
    return run_calculation(
        experiment, _calculate_apartment_classifications, CALC_APT_CLASSIFICATION,
        save=save, overwrite=overwrite
    )


def _calculate_apartment_growth_rates(experiment):
    logger.info('Computing apartment growth rate statistics (this may take a couple minutes the first time)')

    # Collect relevant calculation settings/parameters
    analysis_config = experiment.config.get_analysis_config()
    exp_cond_fields = experiment.config.experimental_condition_fields
    apt_addr_fields = experiment.config.apartment_address_fields
    occupancy_threshold = analysis_config['confluence_detection']['component_occupancy_threshold']['chamber']

    # Load necessary data for calculation
    store = experiment.get_data_store()

    df_apt = store.get('apartment')
    df_apt = add_measurement_times(experiment, df_apt)

    # TODO: Load cell data here and add arguments for redefining cell counts through filters on cell properties

    df = analysis.get_growth_rate_data(df_apt, exp_cond_fields, occupancy_threshold=occupancy_threshold)

    # JSON encode any fields stored as data structures as these will not serialize to HDF5 well otherwise
    for c in analysis.GROWTH_RATE_DICT_FIELDS:
        df[c] = df[c].apply(lambda m: json.dumps({str(k): v for k, v in m.items()}))
    for c in analysis.GROWTH_RATE_LIST_FIELDS:
        df[c] = df[c].apply(lambda v: json.dumps(v))

    initial_conditions_filter = analysis_config['apartment_growth_rate']['initial_conditions_filter']
    if initial_conditions_filter:
        df_apt_class = __calculate_apartment_classifications(df_apt, apt_addr_fields)
        df_apt_class = df_apt_class[df_apt_class['initial_condition'].isin(initial_conditions_filter)]
        if len(df_apt_class) == 0:
            raise ValueError(
                'Apartment initial conditions filter "{}" did not match any apartment records -- '
                'either none of these initial conditions occurred or the condition names are invalid'
                .format(initial_conditions_filter)
            )

        # Compute boolean mask with length == len(df) where True indicates that the associated apartment matches
        # the specified initial condition filters (note that the result here is a numpy array)
        mask_match = df.set_index(apt_addr_fields).index.isin(df_apt_class.set_index(apt_addr_fields).index)

        # Set all growth rates for apartments NOT matching initial conditions to nan
        df['growth_rate'] = np.where(mask_match, df['growth_rate'], np.nan)
        logger.info(
            'Invalidating growth rates for %s apartments (of %s) due to lack of match to initial condition filter "%s"',
            len(df) - mask_match.sum(), len(df), initial_conditions_filter
        )

    return df


def calculate_apartment_growth_rates(experiment, save=True, overwrite=False):
    """Compute growth rates for all apartments in an experiment

    Growth rates are calculated as a linear regression of log(cell_count + 1) ~ time where
    time is expressed in days.  Note that the following conditions result in NaN growth rates:
    - Only one cell count
    - Apartments not matching initial conditions specified in analysis configuration (typically, this might include
        filters ignore growth rates outside of apartments with 1 cell at time 0)

    Note that the resulting data frame contains all apartment address fields but no date/time fields as
    growth rates are an aggregation over time.
    """
    return run_calculation(
        experiment, _calculate_apartment_growth_rates, CALC_APT_GROWTH_RATE,
        save=save, overwrite=overwrite
    )

