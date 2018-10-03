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
            output_mode='w', enable_focus_scores=True):
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
            enable_focus_scores=enable_focus_scores
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
        app_config.initialize(experiment_config_path)

        from celldom_app.overview import data as app_data
        app_data.initialize(output_dir)

        from celldom_app.overview import app
        app.run_server(debug=debug)

    def run_array_analysis(self, experiment_config_path, output_dir,
                           apartment_initial_conditions=None,
                           na_growth_rate_fill_value=None,
                           nb_filename=None):
        """Run the apartment array analysis template notebook

        Args:
            experiment_config_path: Path to experiment configuration
                (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
            output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
                (e.g. /lab/data/celldom/output/20180820-G3-full)
            apartment_initial_conditions: Initial conditions (at time zero) of apartments as a comma separated
                list of any number of the following: ['other', 'no_cell', 'single_cell', 'double_cell', 'triple_cell'];
                Example: --apartment-initial-conditions=['no_cell','single_cell']
                Default value is nothing, which means that no filters of any kind are applied
            na_growth_rate_fill_value: Placeholder value for apartments with cell count data but either too few
                data points for growth rate calculations or off-target initial conditions; default is nothing, which
                means that NA growth rates are excluded from visualizations and analysis (another common choice is 0)
            nb_filename: Name of notebook file to create (in `output_dir`); defaults to name of template notebook
        """
        params = dict(
            experiment_config_path=experiment_config_path,
            experiment_output_dir=output_dir,
            apartment_initial_conditions=apartment_initial_conditions,
            na_growth_rate_fill_value=na_growth_rate_fill_value
        )
        path = _exec_nb('array_analysis.ipynb', output_dir, params, output_filename=nb_filename)
        print('Analysis complete; see results at "{}"'.format(path))

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


if __name__ == '__main__':
    # Register debugging handler
    faulthandler.register(signal.SIGUSR1)
    fire.Fire(Celldom)
