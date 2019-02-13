import json
import os
import plotly
import logging
import glob
import fire
import dash
import pandas as pd
import numpy as np
import os.path as osp
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from celldom.execute.analysis import add_experiment_date_groups, GROWTH_RATE_OBJ_FIELDS
from celldom_app.overview import data
from celldom_app.overview import config
from celldom_app.overview import lib

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

app = dash.Dash()
# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
cfg = config.get()
cache = {}

PAGE_NAMES = ['summary', 'apartments', 'arrays']
GRAPH_GROWTH_DATA_MARGINS = {'l': 50, 'r': 20, 't': 40, 'b': 80}
MIN_ROWS_VIRTUALIZATION = 500

#app.scripts.config.serve_locally = True
#app.css.config.serve_locally = True


def get_datatable(id, df, cols=None, **kwargs):
    cols = df.columns.tolist() if cols is None else cols
    args = dict(
        id=id,
        columns=[{'id': c, 'name': c} for c in cols],
        data=df.to_dict(orient='records'),
        editable=False,
        filtering=True,
        sorting=True,
        n_fixed_rows=2,
        sorting_type="multi",
        row_selectable="multi",
        content_style="grow",
        row_deletable=False,
        selected_rows=[],
        style_table={'maxHeight': '500px', 'overflowX': 'scroll', 'overflowY': 'scroll'},
        style_cell={'minWidth': '100px'}
    )

    # Add virtualization for larger tables
    if len(df) >= MIN_ROWS_VIRTUALIZATION:
        args.update(dict(virtualization=True, pagination_mode=False))

    # Add kwargs last to override defaults
    args.update(kwargs)

    return dash_table.DataTable(**args)


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


def get_page_apartments():
    df = data.get_apartment_summary_data()

    fields = data.get_apartment_key_fields()
    fields += ['growth_rate', 'min_count', 'max_count', 'initial_condition', 'n']
    fields += df.filter(regex='^tz_count_.*').columns.tolist()

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
                        '- **min_count**: Minimum cell count across all images taken'
                        ' (cell type/component set in analysis configuration)\n'
                        '- **max_count**: Maximum cell count across all images taken'
                        ' (cell type/component set in analysis configuration)\n'
                        '- **initial_condition**: String name associated with classification of time zero'
                        ' conditions within apartment (e.g. single_cell, no_cell, double_cell, many_cells)\n'
                        '- **tz_count_\*_\***: Time zero counts for each cell type and component\n'
                        '- **n**: Number of time points (i.e. hours) for which measurements exist for apartment'
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
            get_datatable('table-apartments-data', df, fields)
        ]),
        html.Div([
                html.Div(
                    dcc.Graph(id='graph-growth-data'),
                    className='seven columns',
                    style={'margin-top': '10px'}
                ),
                html.Div([
                        html.Div(
                            dcc.Dropdown(
                                id='apartment-dropdown',
                                placeholder='Apartment (must be selected in table first)'
                            ),
                            style={'display': 'inline-block', 'width': '50%'}
                        ),
                        html.Div(
                            dcc.Textarea(
                                placeholder='Filter ("hr in [0,24] and hr != 72")',
                                style={'width': '100%', 'margin': '0px', 'min-height': '35px', 'height': '35px'},
                                id='apartment-time-filter'
                            ),
                            style={'display': 'inline-block', 'width': '38%'}
                        ),
                        html.Div(
                            dcc.Checklist(
                                options=[{'label': 'Centroids', 'value': 'enabled'}],
                                values=['enabled'],
                                id='enable-cell-marker'
                            ),
                            style={
                                'display': 'inline-block', 'width': '12%',
                                'vertical-align': 'top', 'padding-top': '4px'
                            }
                        ),
                        html.Div(
                            id='apartment-animation',
                            style={'overflow-x': 'scroll', 'white-space': 'nowrap'}
                        ),
                        html.Div([
                            html.Div(
                                dcc.Dropdown(
                                    id='export-apartment-type',
                                    options=[
                                        {'label': 'TIF', 'value': 'tif'},
                                        {'label': 'TIF (+centroids)', 'value': 'tif_markers'},
                                        {'label': 'PNG', 'value': 'png'},
                                        {'label': 'PNG (+centroids)', 'value': 'png_markers'},
                                        {'label': 'RectLabel Annotations', 'value': 'rectlabel_annotations'}
                                    ],
                                    placeholder='Export Type',
                                    clearable=False
                                ),
                                style={'display': 'inline-block', 'width': '50%', 'margin-top': '1px'}
                            ),
                            html.Div(
                                html.Button(
                                    'Export',
                                    id='export-apartment-images',
                                    style={'width': '100%', 'height': '35px', 'line-height': '18px'}
                                ),
                                style={
                                    'display': 'inline-block', 'width': '50%',
                                    'vertical-align': 'top', 'padding-top': '1px'
                                }
                            )
                        ])
                    ],
                    className='five columns',
                    style={'margin-top': '0'}
                )
            ],
            className='row'
        ),
        html.Div(id='null1', style={'display': 'none'})
    ]


def _get_array_metrics():
    df = data.get_apartment_data()
    metrics = [
        {'label': 'Growth Rate (24hr log2)', 'value': 'growth_rate'},
        {'label': 'Number of Measurements', 'value': 'num_measurements'},
        {'label': 'Cell Count (Chamber)', 'value': 'cell_count_any_chamber'},
        {'label': 'Cell Count (Trap)', 'value': 'cell_count_any_trap'},
        {'label': 'Occupancy Percentage (Chamber)', 'value': 'occupancy_chamber'}
    ]
    return [m for m in metrics if m['value'] in df or m['value'] in ['growth_rate']]


ARRAY_METRICS = _get_array_metrics()


def get_page_arrays():
    df = data.get_array_data()
    return [
        html.Div(id='table-info-arrays', style={'float': 'right'}),
        html.Details([
            html.Summary('Array Data', style={'font-size': '18'}),
            html.Div([
                dcc.Markdown(
                    'Each row of this table corresponds to an individual array and can be selected to produce '
                    'heatmaps of various metrics over the array\'s individual apartments'
                    '\n\n**Field Definitions**\n'
                    '- {}: Experimental conditions\n'
                    '- **num_apartments**: Number of distinct apartments for which data was collected '
                    '(should be close to number of streets times number of apartment addresses)\n'
                    '- **median_growth_rate**: Median growth rate across all apartments in the array\n'
                    '- **first_date**: Earliest date associated with a measurement of any apartment in the array\n'
                    '- **last_date**: Latest date associated with a measurement of any apartment in the array\n'
                    .format(
                        ', '.join(['**{}**'.format(f) for f in cfg.experimental_condition_fields])
                    )
                )
            ])
        ]
        ),
        html.Div([
            get_datatable('table-arrays-data', df)
        ]),
        html.Div([
                html.Div(
                    dcc.Dropdown(
                        id='array-dropdown',
                        placeholder='Choose array (must be selected in table first)'
                    ),
                    className='three columns'
                ),
                html.Div(
                    dcc.Dropdown(
                        id='array-metric-dropdown',
                        placeholder='Choose metric to view data for',
                        options=ARRAY_METRICS,
                        value=ARRAY_METRICS[0]['value'],
                        clearable=False,
                        searchable=False
                    ),
                    className='three columns',
                    style={'margin-left': '0'}
                ),
                html.Div(
                    dcc.Checklist(
                        options=[
                            {'label': 'Normalize Across Time Steps', 'value': 'enabled'}
                        ],
                        values=['enabled'],
                        id='enable-array-normalize'
                    ),
                    className='six columns',
                    style={'margin-left': '0', 'padding-top': '4px'}
                )
            ],
            className='row',
            style={'margin-top': '0'}
        ),
        html.Div(
            dcc.Graph(id='graph-array-data'),
            className='row'
        ),
    ]


def get_page_summary():
    df_acq = data.get_acquisition_data()
    df_arr = data.get_array_data()
    n_raw_files = len(df_acq)

    # Count raw files processed for each experimental condition
    df_acq = df_acq.groupby(cfg.experimental_condition_fields).size().rename('num_raw_images')

    # Select fields from array data
    df_arr = df_arr.set_index(cfg.experimental_condition_fields)[['mean_growth_rate', 'median_growth_rate', 'num_apartments']]

    # Merge acquisition summary and array data
    df = pd.concat([df_acq, df_arr], axis=1).reset_index()

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
                    '- **mean_growth_rate**: Mean 24hr growth rate estimate (log 2) across all apartments\n'
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
            get_datatable('table-summary-data', df)
        ]),
        html.Div([
                html.Div(
                    'Distribution Grouping Fields:',
                    className='two columns',
                    style={'text-align': 'right', 'margin-top': '6px'}
                ),
                html.Div(
                    dcc.Dropdown(
                        options=[
                            {'label': f, 'value': f}
                            for f in cfg.experimental_condition_fields
                        ],
                        multi=True,
                        value=cfg.experimental_condition_fields,
                        clearable=False,
                        searchable=False,
                        id='summary-distribution-grouping'
                    ),
                    className='six columns',
                    style={'margin-left': '10px'}
                ),
                html.Div(
                    dcc.Dropdown(
                        options=[
                            {'label': 'Box', 'value': 'box'},
                            {'label': 'Histogram (Count)', 'value': 'histogram'},
                            {'label': 'Histogram (Percent)', 'value': 'histogram_percent'},
                            {'label': 'Violin', 'value': 'violin'}
                        ],
                        placeholder='Plot Type',
                        multi=False,
                        clearable=True,
                        searchable=False,
                        id='summary-distribution-plot-type'
                    ),
                    className='two columns',
                    style={'margin-left': '0'}
                )
            ],
            className='row',
            style={'margin-top': '0'}
        ),
        html.Div(
            dcc.Graph(id='graph-summary-distributions'),
            className='row'
        )
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


# TODO: figure out how to do this w/o full dataset transfer
# def add_table_info_callback(page_name):
#     @app.callback(
#         Output('table-info-' + page_name, 'children'),
#         [Input('table-' + page_name + '-data', 'rows')]
#     )
#     def update_table_info(rows):
#         if rows is None:
#             return None
#         return 'Number of rows: {}'.format(len(rows))
#
#
# for page_name in PAGE_NAMES:
#     add_table_info_callback(page_name)


def update_apartment_summary_data(code=None):
    df = data.get_apartment_summary_data()
    if code is not None and code.strip():
        logger.info('Applying custom code to apartment summary data:\n%s', code)
        local_vars = {'df': df}
        exec(code, globals(), local_vars)
        df = local_vars['df']
    cache['apartment_summary_data'] = df
    return df


def get_apartment_summary_data(selected_rows=None):
    if 'apartment_summary_data' in cache:
        df = cache['apartment_summary_data']
    else:
        df = update_apartment_summary_data()
    return df.iloc[selected_rows] if selected_rows else df


@app.callback(
    Output('table-apartments-data', 'data'),
    [Input('code-apartments-apply', 'n_clicks')],
    [State('code-apartments', 'value')]
)
def update_apartment_table(_, code):
    return update_apartment_summary_data(code).to_dict(orient='records')


@app.callback(
    Output('apartment-dropdown', 'options'),
    [Input('table-apartments-data', 'selected_rows')]
)
def update_apartment_dropdown_options(selected_rows):
    if not selected_rows:
        return []
    df = get_apartment_summary_data(selected_rows)
    options = []
    for i, r in df.iterrows():
        key = data.get_apartment_key(r)
        options.append({'label': key, 'value': key})
    return options


def get_apartment_image_data(selected_apartment, show_cell_marker, selected_rows):
    if not selected_rows or not selected_apartment:
        return None
    # Get growth data for all selected apartments (in table)
    df = get_apartment_summary_data(selected_rows)

    # Get growth data for the selected (single) apartment
    mask = df.apply(data.get_apartment_key, axis=1) == selected_apartment
    if not np.any(mask):
        logger.error('Failed to find growth data for apartment %s (this should not be possible)', selected_apartment)
        return None
    if np.sum(mask) > 1:
        logger.error('Apartment key %s matches multiple table rows (this should not be possible)', selected_apartment)
        return None

    # Pass one-row data frame to image data processor
    df = data.get_apartment_image_data(
        df.loc[list(mask.values)],
        marker_color=data.visualization.COLOR_RED if show_cell_marker else None
    )
    if selected_apartment not in df.index:
        logger.error('Apartment image data does not contain apartment %s', selected_apartment)
        return None
    return df.loc[selected_apartment]


def get_apartment_time_filter_predicate(filter):
    if filter is None or not filter.strip():
        return lambda *args: True

    def predicate(hr):
        return eval(filter, {'hr': hr})
    return predicate


@app.callback(
    Output('apartment-animation', 'children'),
    [Input('apartment-dropdown', 'value'), Input('enable-cell-marker', 'values')],
    [
        State('table-apartments-data', 'selected_rows'),
        State('apartment-time-filter', 'value')
    ]
)
def update_apartment_animations(selected_apartment, show_cell_marker, selected_rows, time_filter):
    r = get_apartment_image_data(selected_apartment, show_cell_marker, selected_rows)
    if r is None:
        return None

    p = get_apartment_time_filter_predicate(time_filter)
    children = []
    for i in range(r['n']):
        if not p(r['hours'][i]):
            continue
        title = 'H{} D{} C{}'.format(r['hours'][i], r['dates'][i], r['cell_counts'][i])
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
    Output('null1', 'children'),
    [Input('export-apartment-images', 'n_clicks')],
    [
        State('apartment-dropdown', 'value'),
        State('table-apartments-data', 'selected_rows'),
        State('apartment-time-filter', 'value'),
        State('export-apartment-type', 'value'),
    ]
)
def export_apartment_animations(_, selected_apartment, selected_rows, time_filter, export_type):
    if export_type is None or not export_type.strip():
        return None
    show_cell_marker = export_type.split('_')[-1] == 'markers'
    r = get_apartment_image_data(selected_apartment, show_cell_marker, selected_rows)
    if r is None:
        return None

    p = get_apartment_time_filter_predicate(time_filter)
    # Create more informative, easy-to-read titles if they are to be used in stacks
    if export_type in ['tif', 'tif_markers']:
        titles = [
            'H{} D{} C{}'.format(r['hours'][i], r['dates'][i], r['cell_counts'][i])
            for i in range(r['n']) if p(r['hours'][i])
        ]
    # Otherwise, create more sortable path names as titles
    else:
        titles = [
            'H{:03d}:{}'.format(r['hours'][i], r['dates'][i].strftime('%Y%m%d:%H%M%S'))
            for i in range(r['n']) if p(r['hours'][i])
        ]

    # Export images directly if not also adding annotations
    if export_type in ['tif', 'tif_markers', 'png', 'png_markers']:
        typ = export_type.split('_')[0]
        export_dir, _ = lib.export_apartment_images(
            selected_apartment, r['images'].tolist(), titles,
            type=typ, relpath=osp.join('export', 'apartment', typ))
    # Otherwise, export as png first before adding annotation files
    elif export_type in ['rectlabel_annotations']:
        export_dir, _ = lib.export_apartment_annots(
            cfg.exp_config, selected_apartment, r['images'].tolist(), titles,
            relpath=osp.join('export', 'apartment', 'annotated'))
    else:
        raise ValueError('Export type "{}" not yet supported'.format(export_type))
    logger.info('Saved exported apartment images for apartment "%s" to path %s', selected_apartment, export_dir)
    

@app.callback(
    Output('graph-growth-data', 'figure'),
    [Input('table-apartments-data', 'selected_rows')]
)
def update_apartment_growth_graph(selected_rows):
    if not selected_rows:
        return {
            'data': [],
            'layout': {'title': 'Apartment Cell Counts', 'margin': GRAPH_GROWTH_DATA_MARGINS}
        }
    df = get_apartment_summary_data(selected_rows)

    # Deserialize any json objects encoded in execute.view.get_apartment_summary_view
    # Note that this is VERY slow so make sure to only use it on subsets
    # for c in GROWTH_RATE_OBJ_FIELDS:
    #     df[c] = df[c].apply(lambda x: pd.read_json(x, typ='series').to_dict())

    fig_data = []

    def get_ts(json_string):
        return pd.read_json(json_string, typ='series').sort_index()

    for i, r in df.iterrows():
        tsct = get_ts(r['cell_counts'])
        tshr = get_ts(r['hours'])
        tsdt = get_ts(r['dates'])
        tso = get_ts(r['occupancies'])
        tsconf = get_ts(r['confluence'])
        fig_data.append({
            'x': tshr.values,
            'y': tsct.values,
            'name': data.get_apartment_key(r),
            'type': 'line',
            'marker': {'symbol': ['circle-open' if v else 'circle' for v in tsconf.values]},
            'text': [
                'Occupancy: {:.0f}%<br>Date: {}'.format(100*tso.loc[dt], tsdt.loc[dt])
                for dt in tsct.index
            ]
        })
    title = 'Apartment Cell Counts<br><i>Type: {type} Component: {component}</i>'\
        .format(**cfg.analysis_config.apartment_summary_cell_class)
    fig_layout = {
        'title': title,
        'margin': GRAPH_GROWTH_DATA_MARGINS,
        'xaxis': {'title': 'Acquisition Hour'},
        'yaxis': {'title': 'Number of Cells'},
        'showlegend': True,
        'hovermode': 'closest'
    }
    return {'data': fig_data, 'layout': fig_layout}


@app.callback(
    Output('table-apartments-data', 'selected_rows'),
    [Input('graph-array-data', 'clickData')],
    [
        State('array-dropdown', 'value'),
        State('table-apartments-data', 'selected_rows')
    ]
)
def update_apartments_table_selected_rows(click_data, array, selected_rows):
    # {'points': [{'z': 2, 'curveNumber': 0, 'y': 'st 08', 'x': 'apt 11'}]}
    if not array or not click_data or 'points' not in click_data or not click_data['points']:
        return selected_rows

    # Fetch the full apartment summary data frame (not currently selected frame)
    df = get_apartment_summary_data()
    selected_row_indices = selected_rows or []
    apt_num, st_num = click_data['points'][0]['x'], click_data['points'][0]['y']
    apt_num, st_num = apt_num.split(' ')[1], st_num.split(' ')[1]
    key = data.append_key(array, [apt_num, st_num])

    keys = np.array([data.get_apartment_key(r) for _, r in df.iterrows()])

    # Get indices where keys are equal, avoiding argwhere since results are grouped by element
    indices = list(np.flatnonzero(keys == key))
    if len(indices) == 0:
        logger.info('No apartment/growth rate data found for key "%s"', key)
    selected_row_indices.extend(indices)

    # De-duplicated selected indexes as it is possible to trigger this method multiple times with the same
    # target array
    return list(np.unique(selected_row_indices))


@app.callback(
    Output('graph-summary-distributions', 'figure'),
    [
        Input('table-summary-data', 'selected_rows'),
        Input('table-summary-data', 'data'),
        Input('summary-distribution-grouping', 'value'),
        Input('summary-distribution-plot-type', 'value')
    ]
)
def update_summary_distribution_graph(selected_row_indices, rows, fields, plot_type):
    fig_layout = {
        'title': 'Growth Rate Distributions',
        'margin': {'l': 250, 'r': 100, 't': 40, 'b': 40, 'pad': 0}
    }
    if not selected_row_indices or not rows or not fields:
        return {
            'data': [],
            'layout': fig_layout
        }

    # Determine keys corresponding to selected grouping fields (`fields` was initially
    # cfg.experimental_condition_fields but can be dynamic in this context)
    keys = pd.DataFrame(rows).iloc[selected_row_indices].set_index(fields).index.unique()

    # Subset growth data to only experimental conditions selected
    df = data.get_apartment_summary_data()
    df = df.set_index(fields).loc[keys]

    # If plot type not explicitly set, use default based on number of distributions in graph
    if not plot_type:
        if len(keys) <= 2:
            plot_type = 'histogram'
        elif len(keys) <= 10:
            plot_type = 'violin'
        else:
            plot_type = 'box'

    fig_data = []

    # Iterate through each experimental condition based on mean growth rate and add distribution figure
    groups = df.groupby(df.index)
    keys = groups['growth_rate'].median().sort_values().index
    for k in keys:
        name = k if isinstance(k, str) else ':'.join(k)
        g = groups.get_group(k)
        if plot_type in ['box', 'violin']:
            fig_data.append({
                'x': g['growth_rate'].clip(*cfg.growth_rate_range),
                'name': name,
                'type': plot_type,
                'box': {
                    'visible': True
                },
                'meanline': {
                    'visible': True
                },
                'boxmean': True
            })
            fig_layout['xaxis'] = {'title': '24hr Growth Rate (log2)'}
        elif 'histogram' in plot_type:
            histnorm = plot_type.split('_')
            histnorm = histnorm[1] if len(histnorm) > 1 else ''
            fig_data.append({
                'x': g['growth_rate'].clip(*cfg.growth_rate_range),
                'name': name,
                'type': 'histogram',
                'histnorm': histnorm,
                'xbins': {'start': cfg.growth_rate_range[0], 'end': cfg.growth_rate_range[1], 'size': .05},
                'opacity': .3
            })
            fig_layout['barmode'] = 'overlay'
            fig_layout['xaxis'] = {'title': '24hr Growth Rate (log2)'}
            if histnorm == '':
                fig_layout['yaxis'] = {'title': 'Number of Apartments'}
            elif histnorm == 'percent':
                fig_layout['yaxis'] = {'title': 'Percentage of Apartments'}
            else:
                fig_layout['yaxis'] = {'title': ''}
        else:
            raise ValueError('Plot type "{}" not yet supported'.format(plot_type))
    return {'data': fig_data, 'layout': fig_layout}


@app.callback(
    Output('array-dropdown', 'options'),
    [Input('table-arrays-data', 'selected_rows'), Input('table-arrays-data', 'data')]
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
        return d.set_index(data.get_array_key_fields()).sort_index().loc[array_key].copy()

    def one_or_error(v):
        if len(v) > 1:
            raise ValueError('Found multiple values in pivot: {}'.format(v))
        return v[0]

    if metric == 'growth_rate':
        df = data.get_apartment_summary_data()
        df = prep(df)
        if len(df) == 0:
            return dict(data=[], layout={})
        fig = lib.get_array_graph_figure(
            df, 'growth_rate', enable_normalize,
            fill_value=None, agg_func=one_or_error
        )
        fig['layout']['title'] = title
        return fig
    elif 'cell_count' in metric or 'occupancy' in metric or metric == 'num_measurements':
        # Subset data to selected array TODO: choose data based on metric
        df = data.get_apartment_data()
        df = prep(df)
        if len(df) == 0:
            return dict(data=[], layout={})

        # Metric-specific transformations
        if 'occupancy' in metric:
            df[metric] = df[metric] * 100

        # Pivot configuration
        if 'cell_count' in metric:
            agg_func, fill_value, value_range = one_or_error, cfg.array_cell_count_fill, None
        elif metric == 'num_measurements':
            agg_func, fill_value, value_range = one_or_error, 0, (0, 5)
        else:
            agg_func, fill_value, value_range = one_or_error, None, None

        fig = lib.get_array_graph_figure(
            df, metric, enable_normalize,
            date_field='acq_datetime_group', elapsed_field='elapsed_hours_group',
            fill_value=fill_value, agg_func=agg_func, value_range=value_range
        )
        fig['layout']['title'] = title
        return fig
    else:
        raise NotImplementedError('Metric "{}" not yet supported'.format(metric))


def run_server(debug=False):
    app.run_server(debug=debug, port=cfg.app_port, host=cfg.app_host_ip)
