import os.path as osp
import os
import re
import pandas as pd
import numpy as np
from celldom.app.overview import data
from celldom import io as celldom_io
from celldom import annotation
from skimage import io as sk_io
import logging
logger = logging.getLogger(__name__)


class Dataset(object):

    def __init__(self, df, name):
        self.df = df
        self.name = name

    def select(self, rows):
        """Plotly-specific row selection"""
        return self.df if not rows else self.df.iloc[rows]


class Datasets(object):

    def __init__(self, loaders):
        self.loaders = loaders
        self.data = {}

    def __getattr__(self, dataset):
        if dataset not in self.data:
            self.data[dataset] = self.load(dataset)
        return self.data[dataset]

    def __getitem__(self, dataset):
        return self.__getattr__(dataset)

    def load(self, dataset):
        return Dataset(self.loaders[dataset](), dataset)

    def update(self, dataset, code):
        if code is None or not code.strip():
            return self[dataset]
        df = self.load(dataset).df.copy()
        logger.debug('Applying custom code to dataset "%s":\n%s', dataset, code)
        local_vars = {'df': df}
        exec(code, globals(), local_vars)
        df = local_vars['df']
        self.data[dataset] = Dataset(df, dataset)
        return self.data[dataset]


def get_array_graph_figure(df, metric, enable_normalize, value_range=None, agg_func=np.median, fill_value=None,
                           date_field=None, elapsed_field=None):
    fig_data = []

    if value_range is None:
        value_range = df[metric].min(), df[metric].max()

    # Group data based on date field, if possible
    if date_field is None:
        groups = [('', df)]
    else:
        if elapsed_field:
            groups = df.groupby([date_field, elapsed_field])
        else:
            groups = df.groupby(date_field)

    for i, (k, g) in enumerate(groups):

        gp = g.pivot_table(
            index=['st_num'], columns=['apt_num'], values=metric,
            aggfunc=agg_func, fill_value=fill_value
        )
        name = str(k)
        if len(k) == 2:
            name = '{} (hr {:.0f})'.format(*k)
        trace = dict(
            visible=False,
            y=['st {}'.format(v) for v in gp.index],
            x=['apt {}'.format(v) for v in gp.columns],
            z=gp.values,
            type='heatmap',
            colorscale='Portland',
            name=name
        )
        if enable_normalize:
            trace['zmin'] = value_range[0]
            trace['zmax'] = value_range[1]
        fig_data.append(trace)

    if len(fig_data) > 0:
        fig_data[0]['visible'] = True

    fig_layout = dict(margin=dict(b=60, t=53))

    # Add sliders for heatmaps available over time, if possible
    if date_field is not None and len(fig_data) > 0:
        # See: https://plot.ly/python/reference/#layout-updatemenus
        steps = []
        for i in range(len(fig_data)):
            step = dict(
                method='restyle',
                args=['visible', [False] * len(fig_data)],
                label=fig_data[i]['name']
            )
            step['args'][1][i] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Date: "},
            pad={"t": 75},
            steps=steps
        )]

        fig_layout['sliders'] = sliders

    return dict(data=fig_data, layout=fig_layout)


def _normalize_path(p):
    p = re.sub(r'[^ ;_.:\-a-zA-Z0-9]', '', p)
    return re.sub(r'[:; ]', '-', p)


def export_apartment_annots(exp_config, identifier, images, titles, relpath=osp.join('export', 'apartment')):
    # First export all the images as png
    export_dir, paths = export_apartment_images(identifier, images, titles, type='png', relpath=relpath)

    # Run cell detection on the exported images
    logger.info('Beginning cell detection and XML doc generation for %s files', len(paths))
    docs = annotation.generate(paths, exp_config)

    # Export XML alongside exported pngs
    logger.info('Writing XML annotation files to path "%s"', export_dir)
    annotation.export(docs, export_dir, copy=False)
    return export_dir, docs


def export_apartment_images(identifier, images, titles, type='tif', relpath=osp.join('export', 'apartment')):
    identifier = _normalize_path(identifier)
    if len(images) != len(titles):
        raise ValueError(
            'Titles list length ({}) does not match image list length ({})'
            .format(len(titles), len(images)))

    # Create path to export files (just tif for now)
    exp_dir = data.get_output_path(relpath)
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # If exporting a tif, save a single image file as a stack of images
    if type == 'tif':
        path = osp.join(exp_dir, identifier + '.tif')
        celldom_io.save_tiff(path, images, titles)
        paths = [path]
    # If exporting pngs, save each image separately
    elif type == 'png':
        paths = []
        for image, title in zip(images, titles):
            path = osp.join(exp_dir, identifier + '-' + _normalize_path(title) + '.png')
            sk_io.imsave(path, image)
            paths.append(path)
    else:
        raise ValueError('Export type "{}" not supported'.format(type))
    return exp_dir, paths
