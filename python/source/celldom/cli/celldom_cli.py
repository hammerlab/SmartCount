#!/usr/bin/env python
import fire
import os
import os.path as osp
import numpy as np
import pandas as pd
import celldom
import glob
from celldom.execute import processing
from celldom.config import experiment_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('cli')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def _nb_path(nb_name):
    return osp.join(celldom.get_repo_dir(), 'python', 'notebook', 'templates', nb_name)


def _exec_nb(nb_name, data_dir, params, output_path=None):
    import papermill as pm
    input_path = _nb_path(nb_name)
    if output_path is None:
        output_path = osp.join(data_dir, 'notebook')
        os.makedirs(output_path, exist_ok=True)
        output_path = osp.join(output_path, osp.basename(input_path))

    logger.info('Executing notebook "%s"', input_path)
    pm.execute_notebook(input_path, output_path, parameters=params)
    return output_path


class Celldom(object):
    """Celldom CLI Application"""

    def run_processor(
            self, experiment_config_path, data_file_patterns, output_dir,
            sample_rate=None, sample_count=None):
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

        logger.info('Found %s raw data files to process', len(files))
        if sample_count is not None:
            if sample_count < 1:
                raise ValueError('Sample count must be >= 1 (not {})'.format(sample_count))
            logger.info('Randomly selecting %s files to process', sample_count)
            files = pd.Series(files).sample(n=sample_count, random_state=celldom.seed)
        elif sample_rate  is not None:
            if sample_rate <= 0 or sample_rate > 1:
                raise ValueError('Sample rate must in (0, 1] (not {})'.format(sample_rate))
            logger.info('Sampling raw files using given rate %s', sample_rate)
            files = pd.Series(files).sample(frac=sample_rate, random_state=celldom.seed)
        logger.info('Number of raw data files after sample: %s', len(files))

        logger.info('Loading experiment configuration from path: %s', experiment_config_path)
        exp_config = experiment_config.ExperimentConfig(celldom.read_config(experiment_config_path))

        if not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('Running data processor (output dir = %s) ...', output_dir)
        processing.run_cytometer(exp_config, output_dir, files)
        logger.info('Processing complete')

    def analyze_growth_rates(
            self, data_dir, min_cell_size=25, max_cell_size=400,
            start_of_experiment=None, output_path=None):
        """Compute and export growth rates for a processed sample

        Args:
            data_dir: Output directory for processor
            min_cell_size: Minimum area threshold for cells (any smaller cells are ignored);
                default is 25 which corresponds to a radius of about 3 pixels.  For reference,
                at 10x the mode of cell radii is about 7 pixels
            max_cell_size: Maximum area threshold for cells (any bigger cells are ignored);
                default is 400 which corresponds to a radius of about 11 pixels.  Again, at
                10x the mode is around 7 pixels for radii so together the min/max cell size
                defaults assume a radius of 7 +/- 4 pixels as the allowable range
            start_of_experiment: Date in YYYY-MM-dd HH:MM:SS format indicating when the experiment
                started, which is necessary to determine how much time has elapsed for growth rate
                measurments (e.g. '2018-05-31 16:10:00')
            output_path: Path in which to save resulting notebook; defaults to
                $data_dir/analysis/growth_rate_analysis.ipynb
        Example:
            celldom analyze_growth_rates --data-dir=/lab/data/celldom/output/experiment_00
        """
        params = dict(
            data_dir=data_dir,
            min_cell_size=min_cell_size,
            max_cell_size=max_cell_size,
            start_of_experiment=start_of_experiment or ''
        )
        output_path = _exec_nb('growth_rate_analysis_01.ipynb', data_dir, params, output_path=output_path)
        logger.info('Analysis complete; results saved to %s`', output_path)


if __name__ == '__main__':
    fire.Fire(Celldom)