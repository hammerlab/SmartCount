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


def generate_apartment_videos(exp_config, files, output_dir, cell_marker_color=COLOR_RED, video_type='gif', fps=1):
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

    logger.info('Collecting data necessary to generate videos ...')
    apt_data, cell_data = run_processing(exp_config, output_dir, files)

    df = process_results(apt_data, cell_data, cell_marker_color)

    video_dir = osp.join(output_dir, 'videos')
    if not osp.exists(video_dir):
        os.makedirs(video_dir)

    logger.info('Begin video generation within directory "{}"'.format(video_dir))
    for k, g in df.groupby(['apt_num', 'st_num']):
        images = g.sort_values('acq_datetime')['image'].tolist()
        video = avutils.make_video(images)

        filename = 'apt{}_st{}_nfrm{}'.format(k[0], k[1], len(images))
        if video_type == 'gif':
            video.write_gif(osp.join(video_dir, filename + '.gif'), fps=fps, verbose=False, progress_bar=False)
        else:
            video.write_videofile(osp.join(video_dir, filename + '.mp4'), fps=fps, verbose=False, progress_bar=False)
    logger.info('Video generation complete (see results at "{}")'.format(video_dir))


def run_processing(exp_config, output_dir, files):
    results = processing.run_cytometer(
        exp_config, output_dir, files, return_results=True, dpf=ALL_IMAGES)
    apt_data = pd.concat([r[1] for r in results], ignore_index=True)
    cell_data = pd.concat([r[2] for r in results], ignore_index=True)
    return apt_data, cell_data


def process_results(apt_data, cell_data, cell_marker_color):
    keys = ['acq_id', 'apt_id']

    # Of all the apt/st images, choose the one with best focus (regardless of raw file it came from)
    apt_data = apt_data.groupby(['st_num', 'apt_num', 'acq_datetime']) \
        .apply(lambda g: g[['apt_image', 'focus_score', 'acq_id', 'apt_id']].sort_values('focus_score').iloc[0]) \
        .reset_index().set_index(keys)

    # Limit cell data to only that which pertains to the above apartments
    cell_data = cell_data.set_index(keys).loc[apt_data.index]

    df = []
    for i, r in apt_data.iterrows():
        img = r['apt_image'].copy()

        # Get cell data for this apartment
        cdf = cell_data.loc[i]

        # Draw centroids of cells on image
        if cell_marker_color is not None:
            for _, cr in cdf.iterrows():
                cx, cy = cr['centroid_x'], cr['centroid_y']
                rr, cc = draw.circle(cy, cx, 2, shape=img.shape)
                img[rr, cc] = np.array(cell_marker_color, dtype=np.uint8)

        df.append(dict(
            st_num=r['st_num'],
            apt_num=r['apt_num'],
            image=img,
            focus_score=r['focus_score'],
            acq_datetime=r['acq_datetime'],
            cell_count=len(cdf)
        ))

    return pd.DataFrame(df)

