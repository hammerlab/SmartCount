import pandas as pd
import numpy as np


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
