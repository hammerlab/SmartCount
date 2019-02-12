import os.path as osp
import os
import re
import pandas as pd
import numpy as np
from celldom_app.overview import data
from celldom import io as celldom_io
from skimage import io as sk_io


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

#def _clean_

def export_apartment_images(identifier, images, titles, type='tif', relpath=osp.join('export', 'apartment')):
    # Replace non-alnum with hyphen in identifier
    identifier = re.sub(r'\W+', '-', identifier)

    # Create path to export files (just tif for now)
    exp_dir = data.get_output_path(relpath)
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    if type == 'tif':
        path = osp.join(exp_dir, identifier + '.tif')
        celldom_io.save_tiff(path, images, titles)
    elif type == 'png':
        for image in images:
            path = osp.join(exp_dir, identifier + '.tif')
            sk_io.imsave()
    return exp_dir
