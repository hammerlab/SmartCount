import unittest
import os.path as osp
import celldom
from celldom.core import acquisition
from celldom.config import experiment_config


class TestAcquisition(unittest.TestCase):

    def test_collapse_acquisitions(self):
        paths = [
            # APT 000
            # Group 1
            './chipML-day0-cohort40mbar/BF_ST_000_APT_000_20180101000000.tif',
            './chipML-day0-cohort40mbar/GFP_ST_000_APT_000_20180101002900.tif',
            './chipML-day0-cohort40mbar/CY5_ST_000_APT_000_20180101003000.tif',

            # Group 2
            './chipML-day0-cohort40mbar/BF_ST_000_APT_000_20180101013000.tif',
            './chipML-day0-cohort40mbar/GFP_ST_000_APT_000_20180101013000.tif',
            './chipML-day0-cohort40mbar/CY5_ST_000_APT_000_20180101013000.tif',

            # APT 001
            # Group 3
            './chipML-day0-cohort40mbar/BF_ST_000_APT_001_20180102000000.tif',
            './chipML-day0-cohort40mbar/GFP_ST_000_APT_001_20180102000001.tif',
            './chipML-day0-cohort40mbar/CY5_ST_000_APT_001_20180102000000.tif',

            # Group 4 (invalid -- missing channel in hour 1 group)
            './chipML-day0-cohort40mbar/BF_ST_000_APT_001_20180102010000.tif',
            './chipML-day0-cohort40mbar/GFP_ST_000_APT_001_20180102010001.tif',

            # Group 4 (invalid -- duplicate channel in hour 2 group)
            './chipML-day0-cohort40mbar/BF_ST_000_APT_001_20180102020000.tif',
            './chipML-day0-cohort40mbar/GFP_ST_000_APT_001_20180102020001.tif',
            './chipML-day0-cohort40mbar/CY5_ST_000_APT_001_20180102020100.tif',
            './chipML-day0-cohort40mbar/CY5_ST_000_APT_001_20180102020200.tif',

            # APT 002
            # Group 3 (invalid - missing channels)
            './chipML-day0-cohort40mbar/BF_ST_000_APT_002_20180102000000.tif'
        ]
        config = experiment_config.ExperimentConfig(
            osp.join(celldom.test_data_dir, 'config', 'experiment', 'fluoro-expression.yaml'))

        # Test correct parameterization
        df = acquisition.collapse_acquisitions(
            paths, config, min_gap_seconds=30 * 60,
            max_invalid_acquisitions=3, return_df=True
        )
        self.assertEqual(len(df), 6)
        self.assertEqual(df['valid'].sum(), 3)
        self.assertEqual((~df['valid']).sum(), 3)

        # Test invalid threshold too low
        with self.assertRaises(ValueError):
            acquisition.collapse_acquisitions(paths, config, max_invalid_acquisitions=0)

        # Test time gap too large (should then be one row per apartment)
        df = acquisition.collapse_acquisitions(
            paths, config, min_gap_seconds=86400,
            max_invalid_acquisitions=3,
            return_df=True
        )
        self.assertEqual(len(df), 3)
        self.assertEqual(df['valid'].sum(), 0)



