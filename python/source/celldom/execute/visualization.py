import os
import os.path as osp
import numpy as np
import pandas as pd
from skimage import draw
from celldom.execute import processing
from celldom.extract import ALL_IMAGES
from celldom.nb import avutils
import logging

logger = logging.getLogger(__name__)

COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]


def generate_apartment_videos(
        exp_config, files, output_dir,
        cell_marker_color=COLOR_RED, video_type='gif', fps=1):
    """Generate annotated videos of individual apartments at best focus level

    Args:
        exp_config: Experiment configuration
        files: Paths to raw data files (i.e. multi-apartment images)
        output_dir: Directory in which to save results
        cell_marker_color: Color of cell centroids in images; set to none to disable markers
        video_type: One of 'gif' or 'mp4'; default is 'gif'
        fps: Frames per second; default is 1
    """
    if video_type not in ['gif', 'mp4']:
        raise ValueError('Video type to generate must be either "gif" or "mp4" (not "{}")'.format(video_type))
    df = get_apartment_image_data(exp_config, files, output_dir, cell_marker_color)

    video_dir = osp.join(output_dir, 'videos')
    if not osp.exists(video_dir):
        os.makedirs(video_dir)

    logger.info('Begin video generation within directory "{}"'.format(video_dir))
    fields = exp_config.experimental_condition_fields + ['apt_num', 'st_num']
    for k, g in df.groupby(fields):
        images = g.sort_values('acq_datetime')['image'].tolist()
        video = avutils.make_video(images)

        filename = ['{}={}'.format(fields[i], k[i]) for i in range(len(k))] + ['nframes={}'.format(len(images))]
        filename = '-'.join(filename)
        if video_type == 'gif':
            video.write_gif(osp.join(video_dir, filename + '.gif'), fps=fps, verbose=False, progress_bar=False)
        else:
            video.write_videofile(osp.join(video_dir, filename + '.mp4'), fps=fps, verbose=False, progress_bar=False)
    logger.info('Video generation complete (see results at "{}")'.format(video_dir))


def get_apartment_image_data(exp_config, files, output_dir, cell_marker_color=COLOR_RED):
    logger.info('Collecting data necessary to generate apartment images ...')
    exp_cond = exp_config.experimental_condition_fields
    apt_data, cell_data = run_processing(exp_config, output_dir, files)
    return process_results(apt_data, cell_data, exp_cond, cell_marker_color)


def run_processing(exp_config, output_dir, files):
    results = processing.run_cytometer(
        exp_config, output_dir, files, return_results=True, dpf=ALL_IMAGES)
    apt_data = pd.concat([r[1] for r in results], ignore_index=True)
    cell_data = pd.concat([r[2] for r in results], ignore_index=True)
    return apt_data, cell_data


def process_results(apt_data, cell_data, exp_cond_fields, cell_marker_color):
    keys = ['acq_id', 'apt_id']

    # Of all the apt/st images, choose the one with best focus (regardless of raw file it came from)
    apt_data = apt_data.groupby(exp_cond_fields + ['st_num', 'apt_num', 'acq_datetime', 'elapsed_hours_group']) \
        .apply(lambda g: g[['apt_image', 'focus_score', 'acq_id', 'apt_id']].sort_values('focus_score').iloc[0]) \
        .reset_index().set_index(keys)

    if not apt_data.index.is_unique:
        apt_data_dupe = apt_data[apt_data.index.duplicated()]
        raise AssertionError('Apartment data index should be unique; Duplicate rows:\n{}'.format(apt_data_dupe))

    # Set index on cell data to make it searchable
    cell_data = cell_data.set_index(keys).sort_index()

    df = []
    for i, r in apt_data.iterrows():
        img = r['apt_image'].copy()

        # Get cell data for this apartment (if it exists, which it may not if there
        # really are 0 cells present)
        cdf = cell_data.loc[[i]] if i in cell_data.index else pd.DataFrame()

        if len(cdf) > 0 and cdf[['centroid_x', 'centroid_y']].isnull().any().any():
            raise AssertionError(
                'Apartment with index (acq_id, apt_id) = "{}" has cell data w/ null coordinates'
                .format(i)
            )

        # Draw centroids of cells on image
        if cell_marker_color is not None:
            for _, cr in cdf.iterrows():
                cx, cy = cr['centroid_x'], cr['centroid_y']
                rr, cc = draw.circle(cy, cx, 2, shape=img.shape)
                img[rr, cc] = np.array(cell_marker_color, dtype=np.uint8)

        row = r.rename({'apt_image': 'original_image'}).to_dict()
        row['image'] = img
        row['cell_count'] = len(cdf)
        df.append(row)

    return pd.DataFrame(df)

