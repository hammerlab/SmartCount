#!/usr/bin/env python3
import fire
import os
import os.path as osp
import numpy as np
import pandas as pd
import celldom
import glob
import signal
import copy
import faulthandler
from celldom.execute import processing, query
from celldom.config import experiment_config
from celldom.extract import NO_IMAGES, DataPersistenceFlags
import logging

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger('cli')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

IMAGE_NAMES = ['apartment', 'cell', 'acquisition']
DEFAULT_IMAGES_TO_SAVE = ['apartment']


def _nb_path(nb_name):
    return osp.join(celldom.get_repo_dir(), 'python', 'notebook', 'templates', nb_name)


def _exec_nb(nb_name, data_dir, params, output_path=None, output_filename=None):
    import papermill as pm
    input_path = _nb_path(nb_name)

    # Remove any arguments with null values since papermill passes these to notebooks
    # as string "None" (defaults to None should be handled in first cell)
    params = {k: v for k, v in params.items() if v is not None}

    # Create path to output notebook by assuming that it should be placed under
    # the output directory for the experiment
    if output_path is None:
        output_path = osp.join(data_dir, 'notebook')
        os.makedirs(output_path, exist_ok=True)
        output_path = osp.join(output_path, output_filename or osp.basename(input_path))

    logger.info('Executing notebook "%s"', input_path)
    pm.execute_notebook(input_path, output_path, parameters=params)
    return output_path


def _persistence_flags_from_names(names):
    flags = dict(NO_IMAGES._asdict())
    if names is None:
        return flags
    if 'apartment' in names:
        flags['apt_image'] = True
    if 'cell' in names:
        flags['cell_image'] = True
    if 'acquisition' in names:
        flags['raw_norm_image'] = True
    return DataPersistenceFlags(**flags)


class Celldom(object):
    """Celldom CLI Application"""

    def run_processor(
            self, experiment_config_path, data_file_patterns, output_dir,
            sample_rate=None, sample_count=None, max_failures=10, images_to_save=DEFAULT_IMAGES_TO_SAVE,
            output_mode='w', enable_focus_scores=False, cell_detection_threshold=None):
        """Run cell counting/cytometry for a given experiment configuration and set of raw data files

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            data_file_patterns: Input image path glob patterns as either a single string or list of strings; Examples:
                - "/lab/data/dataset/dataset03/*/*.tif"
                - ["/lab/data/dataset/dataset03/*Chip1/*.tif","/lab/data/dataset/dataset03/*Chip3/*.tif"]
            output_dir: Directory in which results will be stored
            sample_rate: Float in (0, 1] indicating a fractional sampling rate of raw files to use
            sample_count: Fixed number of raw files to limit processing to
            max_failures: Maximum number of allowable image processing failures before entire command fails (default
                is 10)
            images_to_save: List of image types to store with experiment results; Must be any or all of ['apartment',
                'cell', 'acquisition']; Default is 'apartment' only
            output_mode: {'a', 'w', 'r+'}, default 'w'
                - ``'w'``: Write; new output files are created and old ones are overwritten if they exist
                - ``'a'``: Append; existing output files are opened for reading and writing, and if the files do
                    not exist they are created.
                ``'r+'``: Similar to ``'a'``, but the output files must already exist.
            enable_focus_scores: Whether or not focus scores should be computed for each image (which is a relatively
                expensive operation); default is True
            cell_detection_threshold: Confidence threshold for cell detections; this should be a number between
                0 and 1 and if not set, a default in celldom.config.cell_config.CellInferenceConfig will be used instead
        """
        # Get all matching files, deduplicate and sort
        files = []
        if isinstance(data_file_patterns, str):
            data_file_patterns = [data_file_patterns]
        for pattern in data_file_patterns:
            files.extend(glob.glob(pattern))
        files = sorted(list(set(files)))

        if len(files) == 0:
            raise ValueError('No data files found to process for patterns "{}"'.format(data_file_patterns))

        # Resolve flags for images to save
        if images_to_save is not None:
            for name in images_to_save:
                if name not in IMAGE_NAMES:
                    raise ValueError('Image type "{}" is not valid (must be one of {})'.format(name, IMAGE_NAMES))
        dpf = _persistence_flags_from_names(images_to_save)

        logger.info('Found %s raw data files to process', len(files))
        if sample_count is not None:
            if sample_count < 1:
                raise ValueError('Sample count must be >= 1 (not {})'.format(sample_count))
            logger.info('Randomly selecting (at most) %s files to process', sample_count)
            n = min(len(files), sample_count)
            files = pd.Series(files).sample(n=n, random_state=celldom.seed)
        elif sample_rate is not None:
            if sample_rate <= 0 or sample_rate > 1:
                raise ValueError('Sample rate must in (0, 1] (not {})'.format(sample_rate))
            logger.info('Sampling raw files using given rate %s', sample_rate)
            files = pd.Series(files).sample(frac=sample_rate, random_state=celldom.seed)
        logger.info('Number of data files chosen by sampling: %s', len(files))

        logger.info('Loading experiment configuration from path: %s', experiment_config_path)
        exp_config = experiment_config.ExperimentConfig(celldom.read_config(experiment_config_path))

        if not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('Running data processor (output dir = %s) ...', output_dir)
        processing.run_cytometer(
            exp_config, output_dir, files,
            max_failures=max_failures, dpf=dpf,
            # Cytometer arguments
            output_mode=output_mode,
            enable_focus_scores=enable_focus_scores,
            cell_detection_threshold=cell_detection_threshold
        )
        logger.info('Processing complete')

    def run_overview_app(self, experiment_config_path, output_dir, debug=False):
        """Run the experiment output overview application

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
                (e.g. /lab/data/celldom/output/20180820-G3-full)
            debug: Flag indicating that Dash server should run in debug mode, which makes it easier
                to test source code changes without restarting the app
        """
        from celldom_app.overview import config as app_config
        app_config.initialize(experiment_config_path, output_dir)

        from celldom_app.overview import data as app_data
        app_data.initialize()

        from celldom_app.overview import app
        app.run_server(debug=debug)

    def run_array_analysis(self, experiment_config_path, output_dir,
                           na_growth_rate_fill_value=None,
                           nb_filename=None, **kwargs):
        """Run the apartment array analysis template notebook

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
                (e.g. /lab/data/celldom/output/20180820-G3-full)
            na_growth_rate_fill_value: Placeholder value for growth rates in apartments ineligible for growth rate
                calculations (due to initial conditions or lack of data); This only applies to growth rate heatmap
                visualizations and can be used to fill in missing values in grids for better continuity in the display
            nb_filename: Name of notebook file to create (in `output_dir`); defaults to name of template notebook
        """
        params = dict(
            experiment_config_path=experiment_config_path,
            experiment_output_dir=output_dir,
            na_growth_rate_fill_value=na_growth_rate_fill_value
        )
        params.update(kwargs)
        path = _exec_nb('array_analysis.ipynb', output_dir, params, output_filename=nb_filename)
        logger.info('Analysis complete; see results at "{}"'.format(path))

    def run_resistant_cell_analysis(self, experiment_config_path, output_dir, concentration_fields,
                                    concentration_values_order=None, concentration_values_map=None,
                                    drug_fields=None, array_fields=None,
                                    lambda_grid='0,1,21', nb_filename=None, **kwargs):
        """Run the resistant cell analysis template notebook

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
                (e.g. /lab/data/celldom/output/20180820-G3-full)
            concentration_fields: Comma separated string with experiment field names corresponding to concentration
                (e.g. 'acq_concentration')
            concentration_values_order: Comma separated string with ordering of all possible concentrations
                (e.g. if concentrations are "Control", ".1uM" and "1.0uM" the order would be given as
                'Control,.1uM,1.0uM'); default is None meaning that names of concentrations are assumed to
                sort lexically
            concentration_values_map: Comma separated string with numeric (float) values corresponding to each
                concentration name/label (e.g. if concentrations are "Control", ".1uM" and "1.0uM" the map
                would be given as '0,.1,1.0')
            drug_fields: Comma separated string with experiment field names corresponding to concentration
                (e.g. 'acq_concentration'); default is None which implies that the entire experiment
                involved only one drug
            array_fields: Comma separated string with experiment field names corresponding to arrays; default is
                None implying that only one array is present.  Note that unlike the other field groups, this could
                also correspond to combinations of metadata fields that don't necessarily identify arrays.  For example,
                fields identifying a chip could be used instead and cell counts in the analysis will be aggregated
                across the chip instead of an array
            lambda_grid: 3 item comma separated string with values `start`, `stop`, and `n` used to indicate the
                growth rate thresholds used in the analysis (e.g. '0,1,21' means use lambda grid from 0 to 1 with
                21 items, which implies increments of .05)
            nb_filename: Name of notebook file to create (in `output_dir`); defaults to name of template notebook
        """
        params = dict(
            experiment_config_path=experiment_config_path,
            experiment_output_dir=output_dir,
            concentration_fields=concentration_fields,
            concentration_values_order=concentration_values_order,
            concentration_values_map=concentration_values_map,
            drug_fields=drug_fields,
            array_fields=array_fields,
            lambda_grid=lambda_grid
        )
        params.update(kwargs)
        path = _exec_nb('resistant_cell_analysis.ipynb', output_dir, params, output_filename=nb_filename)
        logger.info('Analysis complete; see results at "{}"'.format(path))

    def get_apartment_info(self, experiment_config_path, output_dir, keys):
        """Get apartment data for a specific set of "keys"

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
                (e.g. /lab/data/celldom/output/20180820-G3-full)
            keys: One or more key string(s) representing apartment address in the form experimental condition fields +
                apartment number + street number (':' delimited); Examples:
                - gravity:White:3:Control:01:70
                - gravity:Pink:3:0.5uM:27:02
                - gravity:Blue:3:Control:04:04
        """
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)
        return query.get_apartment_info(experiment_config_path, output_dir, keys)

    def create_chip_configuration(self, annotation_csv, chip_name, marker_spacing, apt_num_range, st_num_range,
                                  apt_num_rotation=0, st_num_rotation=0, single_digit_pad=3):
        """Create a chip configuration from a VIA annotations export file (as csv)

        Note the the configuration will be returned as a string (containing yaml document)

        Args:
            annotation_csv: Path to exported annotations; must contain bounding box and point annotations necessary
                to create a chip configuration (see utils/config/chip/README.md for more details)
            chip_name: Name of chip to be used (should be informative but does not need to comply with naming patterns)
            marker_spacing: 2-tuple as (horizontal, vertical) distances in pixels between marker centers
                (at 10x magnification)
            apt_num_range: 2-tuple as (min, max) specifying the minimum and maximum apartment numbers present on a chip
            st_num_range: 2-tuple as (min, max) specifying the minimum and maximum street numbers present on a chip
            apt_num_rotation: Rotation in degrees of apartment numbers (default is 0)
            st_num_rotation: Rotation in degrees of street numbers (default is 0)
            single_digit_pad: Number of pixels to pad around individual digit images
        Returns:
            String representation of yaml configuration
        """
        from celldom.config import generator
        import yaml
        import io
        config = generator.create_chip_configuration(
            annotation_csv, chip_name, marker_spacing, apt_num_range, st_num_range,
            apt_num_rotation=apt_num_rotation, st_num_rotation=st_num_rotation, single_digit_pad=single_digit_pad
        )
        sio = io.StringIO()
        yaml.dump(config, sio)
        return sio.getvalue()

    def build_views(self, experiment_config_path, output_dir):
        """Build (or rebuild) views based on raw experiment data

        This is necessary because apartment growth rate calculations and aggregations by inferred date groupings
        are parameterized by analysis configurations and take a while to run (so creating materialized "views"
        only needs to happen once, and all analysis afterwards is faster)

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
                (e.g. /lab/data/celldom/output/20180820-G3-full)
        """
        from celldom.execute import view
        from celldom.core import experiment
        exp_config = experiment_config.ExperimentConfig(experiment_config_path)
        view.build_all(experiment.Experiment(exp_config, output_dir), force=True)
        logger.info('View construction complete')


if __name__ == '__main__':
    # Register debugging handler
    faulthandler.register(signal.SIGUSR1)
    fire.Fire(Celldom)
