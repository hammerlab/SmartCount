import os
import os.path as osp
import tqdm
import numpy as np
import pandas as pd
import celldom
import glob
import matplotlib.pyplot as plt
from cvutils.visualize import display_images
from skimage import io
from skimage import draw
from celldom.core import cytometry
from celldom.execute import processing
from celldom.config import experiment_config
from celldom.nb import avutils


def extract_video_data(exp_config, data_files, output_dir):
    """Generate annotated videos of individual apartments at best focus level

    Args:
        exp_config: Experiment configuration
        data_files: Paths to raw data files (i.e. multi-apartment images)
        output_dir: Directory in which to save results
    """
    # WIP
    pass
