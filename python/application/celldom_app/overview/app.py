import fire
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import numpy as np
import os
import json
import plotly
import logging
import glob
from celldom_app.overview import data
from celldom_app.overview import config
from celldom_app.overview import lib
from celldom_app.overview import utils as app_utils

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

app = dash.Dash()
# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
cfg = config.get()

PAGE_NAMES = ['summary', 'apartments', 'arrays']
GRAPH_GROWTH_DATA_MARGINS = {'l': 50, 'r': 20, 't': 40, 'b': 80}

# app.scripts.config.serve_locally = True
# app.css.config.serve_locally = True


def get_header_layout():
    return html.Div(
        className='row',
        children=html.Div([
            html.Div(
                'Celldom Experiment Overview',
                className='four columns',
                style={
                    'color': 'white', 'padding-left': '15px',
                    'padding-top': '10px', 'font': '400 16px system-ui'
                }
            ),
            html.Div(
                'Experiment: {}'.format(cfg.exp_config.name),
                id='experiment-name',
                className='four columns',
                style={
                    'color': 'white', 'text-align': 'center',
                    'padding-top': '10px', 'font': '400 16px system-ui'
                }
            ),
            html.Div([
                    html.Div(
                        html.Button(
                            'Arrays', id='link-arrays',
                            style={'color': 'white', 'border-width': '0px'}
                        ),
                        style={'display': 'inline', 'float': 'right'}
                    ),
                    html.Div(
                        html.Button(
                            'Apartments', id='link-apartments',
                            style={'color': 'white', 'border-width': '0px'}
                        ),
                        style={'display': 'inline', 'float': 'right'}
                    ),
                    html.Div(
                        html.Button(
                            'Summary', id='link-summary',
                            style={'color': 'white', 'border-width': '0px'}
                        ),
                        style={'display': 'inline', 'float': 'right'}
                    )
                ],
                className='four columns'
            )
        ]),
        style={'backgroundColor': 'rgb(31, 119, 180)'}
    )


def get_array_data():
    return data.get_array_data()


def get_acquisition_data():
    return data.get_acquisition_data()


def get_apartment_data():
    return data.get_growth_data()


def get_page_apartments():
    df = get_apartment_data()

    fields = data.get_apartment_key_fields() + [
        'growth_rate', 'min_cell_count', 'max_cell_count', 'first_count', 'first_date', 'elapsed_hours_min', 'n'
    ]

    # TODO: Apply filters on first_count and elapsed_hours_min
    return [
        html.Div(id='table-info-apartments', style={'float': 'right'}),
        html.Details([
                html.Summary('Apartment Data', style={'font-size': '18'}),
                html.Div([
                    dcc.Markdown(
                        'Each row of this table corresponds to an individual apartment (for {} total apartments '
                        'across all experimental conditions) and includes the estimated growth rate of cells within'
                        'that apartment as well as a few other statistics useful for sorting/filtering.'
                        '\n\n**Field Definitions**\n'
                        '- {}: Experimental conditions\n'
                        '- **growth_rate**: 24hr growth rate estimate (log 2)\n'
                        '- **min_cell_count**: Minimum cell count across all images taken\n'
                        '- **max_cell_count**: Maximum cell count across all images taken\n'
                        '- **first_date**: Date associated with first image found for an apartment\n'
                        '- **n**: Number of patches from raw images extracted and associated with an apartment '
                        '(this is typically the number of times a single apartment was imaged but there are exceptions)'
                        .format(
                            len(df),
                            ', '.join(['**{}**'.format(f) for f in cfg.experimental_condition_fields])
                        )
                    ),
                    html.Div([
                        dcc.Markdown(
                            '-----\n'
                            '**Custom data operations**\n\n'
                            'Enter any custom apartment data processing code in the field below, and it will be\n'
                            'applied to the data shown in the following table (after pressing the "Apply" button).\n'
                            'Examples:\n'
                            '```\n'
                            'df = df[df[\'apt_num\'] == \'01\']\n'
                            'df.to_csv(\'/tmp/apartment_data.csv\')\n'
                            'print(\'Data filtered and saved to csv at "/tmp/apartment_data.csv"\')\n'
                            '```\n\n'
                        ),
                        dcc.Textarea(
                            placeholder='',
                            value='',
                            style={'width': '100%', 'height': '125px', 'margin-top': '10px'},
                            id='code-apartments'
                        ),
                        html.Button(
                            'Apply',
                            id='code-apartments-apply',
                        ),

                    ])
                ])
            ]
        ),
        html.Div([
            dt.DataTable(
                rows=df.to_dict('records'),
                columns=fields,
                editable=False,
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                max_rows_in_viewport=cfg.max_table_rows,
                id='table-apartments-data'
            )
        ]),
        html.Div([
                html.Div(
                    dcc.Graph(id='graph-growth-data'),
                    className='seven columns',
                    style={'margin-top': '10px'}
                ),
                html.Div([
                        dcc.Dropdown(
                            id='apartment-dropdown',
                            placeholder='Choose apartment to view images for (must be selected in table first)'
                        ),
                        html.Div(
                            id='apartment-animation',
                            style={'overflow-x': 'scroll', 'white-space': 'nowrap'}
                        )
                    ],
                    className='five columns',
                    style={'margin-top': '10px'}
                )
            ],
            className='row'
        )
    ]


ARRAY_METRICS = [
    {'label': 'Number of Measurements', 'value': 'measurement_count'},
    {'label': 'Cell Count', 'value': 'cell_count'},
    {'label': 'Growth Rate (24hr log2)', 'value': 'growth_rate'}
]


def get_page_arrays():
    df = get_array_data()
    return [
        html.Div(id='table-info-arrays', style={'float': 'right'}),
        html.Details([
            html.Summary('Array Data', style={'font-size': '18'}),
            html.Div([
                dcc.Markdown('TBD')
            ])
        ]
        ),
        html.Div([
            dt.DataTable(
                rows=df.to_dict('records'),
                columns=df.columns.tolist(),
                editable=False,
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                max_rows_in_viewport=cfg.max_table_rows,
                id='table-arrays-data'
            )
        ]),
        html.Div([
                html.Div(
                    dcc.Dropdown(
                        id='array-dropdown',
                        placeholder='Choose array to view data for (must be selected in table first)'
                    ),
                    className='three columns'
                ),
                html.Div(
                    dcc.Dropdown(
                        id='array-metric-dropdown',
                        placeholder='Choose metric to view data for',
                        options=ARRAY_METRICS,
                        value='cell_count'
                    ),
                    className='three columns',
                    style={'margin-left': '0'}
                ),
                html.Div(
                    dcc.Checklist(
                        options=[
                            {'label': 'Normalize Across Time Steps', 'value': 'normalize'}
                        ],
                        values=['normalize'],
                        id='enable-array-normalize',
                        style={'float': 'right'}
                    ),
                    className='six columns'
                )
            ],
            className='row'
        ),
        html.Div([
                dcc.Graph(id='graph-array-data')
            ],
            className='row'
        ),
    ]


def get_summary_acquisition_layout():
    df_acq = get_acquisition_data()
    df_grd = get_apartment_data()
    n_raw_files = len(df_acq)

    # Count raw files processed for each experimental condition
    df_acq = df_acq.groupby(cfg.experimental_condition_fields).size().rename('num_raw_images')

    # Compute mean growth rate and apartment count by experimental condition
    df_grd = (
        df_grd.assign(address=df_grd['apt_num'].str.cat(df_grd['st_num'], sep=':'))
        .groupby(cfg.experimental_condition_fields)
        .agg({'address': 'nunique', 'growth_rate': 'median'})
        .rename(columns={'address': 'num_apartments', 'growth_rate': 'median_growth_rate'})
    )

    df = pd.concat([df_acq, df_grd], axis=1).reset_index()

    return [
        html.Div(id='table-info-summary', style={'float': 'right'}),
        html.Details([
            html.Summary('Summary Data', style={'font-size': '18'}),
            html.Div([
                dcc.Markdown(
                    'Each row of the following table corresponds to a specific set of experimental conditions and '
                    'summarizes the data collected for each one.'
                    '\n\n**Field Definitions**\n\n'
                    '- {}: Experimental conditions\n'
                    '- **median_growth_rate**: Median 24hr growth rate estimate (log 2) across all apartments\n'
                    '- **num_raw_images**: Number of raw, multi-apartment image files processed '
                    '(excluding any that had errors); **NOTE** There were {} raw images processed across all '
                    'experimental conditions\n'
                    '- **num_apartments**: Number of individual apartments (i.e. number of unique apartment + '
                    'street number combinations)\n'
                    .format(
                        ', '.join(['**{}**'.format(f) for f in cfg.experimental_condition_fields]),
                        n_raw_files
                    )
                )
            ])
        ]),
        html.Div([
            dt.DataTable(
                rows=df.to_dict('records'),
                columns=df.columns.tolist(),
                editable=False,
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                max_rows_in_viewport=cfg.max_table_rows,
                id='table-summary-data'
            )
        ])
    ]


def get_page_summary():
    return [
        html.Div(get_summary_acquisition_layout()),
        dcc.Graph(id='graph-summary-distributions')
    ]


def get_layout():
    return html.Div([
            # dcc.Location(id='url', refresh=False),
            get_header_layout(),
            # Add empty table to avoid: https://community.plot.ly/t/unable-to-load-table-on-multipage-dash/6347
            html.Div([
                html.Div(get_page_summary(), id='page-summary'),
                html.Div(get_page_apartments(), id='page-apartments'),
                html.Div(get_page_arrays(), id='page-arrays')
            ])

        ]
    )


app.layout = get_layout


def get_selected_button_index(click_timestamps):
    return np.argmax([(ts or 0) for ts in click_timestamps])


def add_page_callback(page_name, page_index):
    @app.callback(
        Output('page-' + page_name, 'style'),
        [Input('link-' + pn, 'n_clicks_timestamp') for pn in PAGE_NAMES]
    )
    def select_page(*timestamps):
        if get_selected_button_index(timestamps) == page_index:
            return {'display': 'block'}
        else:
            return {'display': 'none'}


for page_index, page_name in enumerate(PAGE_NAMES):
    add_page_callback(page_name, page_index)


for page_name in PAGE_NAMES:
    @app.callback(
        Output('link-' + page_name, 'style'),
        [Input('page-' + page_name, 'style')],
        [State('link-' + page_name, 'style')]
    )
    def update_link_style(page_style, style):
        if page_style and page_style['display'] == 'block':
            style['color'] = '#17becf'
        else:
            style['color'] = 'white'
        return style


def add_table_info_callback(page_name):
    @app.callback(
        Output('table-info-' + page_name, 'children'),
        [Input('table-' + page_name + '-data', 'rows')]
    )
    def update_table_info(rows):
        if rows is None:
            return None
        return 'Number of rows: {}'.format(len(rows))


for page_name in PAGE_NAMES:
    add_table_info_callback(page_name)


def get_selected_growth_data(rows, selected_row_indices):
    df = pd.DataFrame([rows[i] for i in selected_row_indices])
    for c in ['cell_counts', 'acq_ids']:
        df[c] = df[c].apply(json.loads)
    return df


@app.callback(
    Output('table-apartments-data', 'rows'),
    [Input('code-apartments-apply', 'n_clicks')],
    [State('code-apartments', 'value')]
)
def update_apartment_table(_, code):
    df = get_apartment_data()
    if code:
        logger.info('Applying custom code to apartment data:\n%s', code)
        local_vars = {'df': df}
        exec(code, globals(), local_vars)
        df = local_vars['df']
    return df.to_dict(orient='records')


@app.callback(
    Output('apartment-dropdown', 'options'),
    [Input('table-apartments-data', 'selected_row_indices'), Input('table-apartments-data', 'rows')]
)
def update_apartment_dropdown_options(selected_row_indices, rows):
    if not selected_row_indices:
        return []
    df = get_selected_growth_data(rows, selected_row_indices)
    options = []
    for i, r in df.iterrows():
        key = data.get_apartment_key(r)
        options.append({'label': key, 'value': key})
    return options


@app.callback(
    Output('apartment-animation', 'children'),
    [Input('apartment-dropdown', 'value')],
    [State('table-apartments-data', 'selected_row_indices'), State('table-apartments-data', 'rows')]
)
def update_apartment_animations(selected_apartment, selected_row_indices, rows):
    if not selected_row_indices or not rows or not selected_apartment:
        return None
    # Get growth data for all selected apartments (in table)
    df = get_selected_growth_data(rows, selected_row_indices)

    # Get growth data for the selected (single) apartment
    mask = df.apply(data.get_apartment_key, axis=1) == selected_apartment
    if not np.any(mask):
        logger.error('Failed to find growth data for apartment %s (this should not be possible)', selected_apartment)
        return []
    if np.sum(mask) > 1:
        logger.error('Apartment key %s matches multiple table rows (this should not be possible)', selected_apartment)
        return []

    # Pass one-row data frame to image data processor
    df = data.get_apartment_image_data(df.loc[list(mask.values)])
    if selected_apartment not in df.index:
        logger.error('Apartment image data does not contain apartment %s', selected_apartment)
    r = df.loc[selected_apartment]

    children = []
    for i in range(r['n']):
        title = '{} - {}'.format(r['dates'][i], r['cell_counts'][i])
        children.append(html.Div([
                html.Div(title, style={'text-align': 'center'}),
                html.Img(
                    src='data:image/png;base64,{}'.format(r['encoded_images'][i]),
                    style={'height': cfg.apartment_image_height}
                )
            ],
            style={'display': 'inline-block', 'margin': '10px'}
        ))
    return children


@app.callback(
    Output('graph-growth-data', 'figure'),
    [Input('table-apartments-data', 'selected_row_indices'), Input('table-apartments-data', 'rows')]
)
def update_apartment_growth_graph(selected_row_indices, rows):
    if not selected_row_indices or not rows:
        return {
            'data': [],
            'layout': {'title': 'Apartment Cell Counts', 'margin': GRAPH_GROWTH_DATA_MARGINS}
        }

    df = get_selected_growth_data(rows, selected_row_indices)
    fig_data = []

    for i, r in df.iterrows():
        ts = pd.Series({pd.to_datetime(k): v for k, v in r['cell_counts'].items()}).sort_index()
        fig_data.append({
            'x': ts.index,
            'y': ts.values,
            'name': data.get_apartment_key(r),
            'type': 'line'
        })
    fig_layout = {
        'title': 'Apartment Cell Counts',
        'margin': GRAPH_GROWTH_DATA_MARGINS,
        'xaxis': {'title': 'Acquisition Date'},
        'yaxis': {'title': 'Number of Cells'},
        'showlegend': True
    }
    return {'data': fig_data, 'layout': fig_layout}


@app.callback(
    Output('table-apartments-data', 'selected_row_indices'),
    [Input('graph-array-data', 'clickData')],
    [
        State('array-dropdown', 'value'),
        State('table-apartments-data', 'selected_row_indices'),
        State('table-apartments-data', 'rows')
    ]
)
def update_growth_table_selected_rows(click_data, array, selected_row_indices, rows):
    # {'points': [{'z': 2, 'curveNumber': 0, 'y': 'st 08', 'x': 'apt 11'}]}
    if not array or not rows or not click_data or 'points' not in click_data or not click_data['points']:
        return selected_row_indices

    selected_row_indices = selected_row_indices or []
    apt_num, st_num = click_data['points'][0]['x'], click_data['points'][0]['y']
    apt_num, st_num = apt_num.split(' ')[1], st_num.split(' ')[1]
    key = data.append_key(array, [apt_num, st_num])

    keys = np.array([data.get_apartment_key(r) for r in rows])

    # Get indices where keys are equal, avoiding argwhere since results are grouped by element
    indices = list(np.flatnonzero(keys == key))
    selected_row_indices.extend(indices)
    return selected_row_indices


@app.callback(
    Output('graph-summary-distributions', 'figure'),
    [Input('table-summary-data', 'selected_row_indices'), Input('table-summary-data', 'rows')]
)
def update_summary_distribution_graph(selected_row_indices, rows):
    if not selected_row_indices or not rows:
        return {
            'data': [],
            'layout': {'title': 'Growth Rate Distributions'}
        }

    # Determine keys corresponding to selected experimental conditions
    keys = pd.DataFrame(rows).iloc[selected_row_indices].set_index(cfg.experimental_condition_fields).index

    # Subset growth data to only experimental conditions selected
    df = get_apartment_data()
    df = df.set_index(cfg.experimental_condition_fields).loc[keys]

    # Determine whether or not enough groups of experimental conditions were selected
    # such that a boxplot is more useful than a large number of histograms
    enable_boxplot = len(keys) > cfg.summary_n_group_treshold

    fig_data = []
    fig_layout = {'title': 'Growth Rate Distributions'}

    # Iterate through each experimental condition based on median growth rate and add distribution figure
    groups = df.groupby(df.index)
    keys = groups['growth_rate'].median().sort_values().index
    for k in keys:
        g = groups.get_group(k)
        if enable_boxplot:
            fig_data.append({
                'x': g['growth_rate'].clip(*cfg.growth_rate_range),
                'name': ':'.join(k),
                'type': 'box'
            })
            fig_layout['xaxis'] = {'title': '24hr Growth Rate (log2)'}
            fig_layout['margin'] = {'l': 250}
        else:
            fig_data.append({
                'x': g['growth_rate'].clip(*cfg.growth_rate_range),
                'name': ':'.join(k),
                'type': 'histogram',
                'xbins': {'start': cfg.growth_rate_range[0], 'end': cfg.growth_rate_range[1], 'size': .05},
                'opacity': .3
            })
            fig_layout['barmode'] = 'overlay'
            fig_layout['xaxis'] = {'title': '24hr Growth Rate (log2)'}
            fig_layout['yaxis'] = {'title': 'Number of Apartments'}
    return {'data': fig_data, 'layout': fig_layout}


@app.callback(
    Output('array-dropdown', 'options'),
    [Input('table-arrays-data', 'selected_row_indices'), Input('table-arrays-data', 'rows')]
)
def update_array_dropdown_options(selected_row_indices, rows):
    if not selected_row_indices or not rows:
        return []

    df = pd.DataFrame([rows[i] for i in selected_row_indices])
    options = []
    for _, r in df.iterrows():
        key = data.get_array_key(r)
        options.append({'label': key, 'value': key})
    return options


def _get_metric_label(metric_value):
    for m in ARRAY_METRICS:
        if m['value'] == metric_value:
            return m['label']
    return metric_value


@app.callback(
    Output('graph-array-data', 'figure'),
    [
        Input('array-dropdown', 'value'),
        Input('array-metric-dropdown', 'value'),
        Input('enable-array-normalize', 'values')
    ]
)
def update_array_graph(array, metric, enable_normalize):
    if not array or not metric:
        return dict(data=[], layout={})

    metric_label = _get_metric_label(metric)
    title = '{}<br><i>{}</i>'.format(metric_label, array)

    def prep(d):
        array_key = tuple(array.split(':'))
        return d.set_index(data.get_array_key_fields()).loc[array_key].copy()

    if metric in ['cell_count', 'measurement_count']:
        # Subset data to selected array TODO: choose data based on metric
        df = data.get_apartment_data()
        df = prep(df)
        if len(df) == 0:
            return dict(data=[], layout={})
        date_map = app_utils.group_dates(df['acq_datetime'], min_gap_seconds=cfg.min_measurement_gap_seconds)
        df['acq_datetime_group'] = df['acq_datetime'].map(date_map)
        df['elapsed_hours_group'] = df['acq_datetime_group'].map(
            df.groupby('acq_datetime_group')['elapsed_hours'].min())
        df['measurement_count'] = 1

        if metric == 'cell_count':
            agg_func, fill_value, value_range = np.median, -1, None
        else:
            agg_func, fill_value, value_range = np.sum, 0, (0, 5)

        fig = lib.get_array_graph_figure(
            df, metric, enable_normalize,
            date_field='acq_datetime_group', elapsed_field='elapsed_hours_group',
            fill_value=fill_value, agg_func=agg_func, value_range=value_range
        )
        fig['layout']['title'] = title
        return fig
    elif metric == 'growth_rate':
        df = data.get_growth_data()
        df = prep(df)
        if len(df) == 0:
            return dict(data=[], layout={})

        def agg_func(v):
            if len(v) > 1:
                return np.nan
            return v[0]

        fig = lib.get_array_graph_figure(
            df, 'growth_rate', enable_normalize,
            fill_value=None, agg_func=agg_func
        )
        fig['layout']['title'] = title
        return fig
    else:
        raise NotImplementedError('Metric "{}" not yet supported'.format(metric))


def run_server():
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)
