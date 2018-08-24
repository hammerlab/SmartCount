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

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

app = dash.Dash()
# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
cfg = config.get()

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
    df = data.get_growth_data()

    fields = data.get_apartment_key_fields() + [
        'growth_rate', 'min_cell_count', 'max_cell_count', 'first_date', 'n'
    ]
    # TODO: Apply filters on first_count and elapsed_hours_min
    return [
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
                    )
                ])
            ]
        ),
        html.Div([
            html.Button(
                'Unselect All',
                id='reset-growth-table',
                style={'position': 'relative', 'top': '40px', 'border-width': '0px', 'padding': '10px'}
            ),
            dt.DataTable(
                rows=df.to_dict('records'),
                columns=fields,
                editable=False,
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                max_rows_in_viewport=cfg.max_table_rows,
                id='table-growth-data'
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


def get_summary_acquisition_layout():
    df_acq = data.get_acquisition_data()
    df_grd = data.get_growth_data()
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
        html.Div(
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
        )
    ]


def get_page_summary():
    return [
        html.Div(get_summary_acquisition_layout()),
        #dcc.Graph(figure=get_summary_distribution_figure(), id='graph-distributions')
        dcc.Graph(id='graph-summary-distributions')
    ]


def get_layout():
    return html.Div([
            # dcc.Location(id='url', refresh=False),
            get_header_layout(),
            # Add empty table to avoid: https://community.plot.ly/t/unable-to-load-table-on-multipage-dash/6347
            html.Div([
                html.Div(get_page_summary(), id='page-summary'),
                html.Div(get_page_apartments(), id='page-apartments')
            ])

        ]
    )


app.layout = get_layout


@app.callback(
    Output('page-summary', 'style'),
    [Input('link-summary', 'n_clicks_timestamp'), Input('link-apartments', 'n_clicks_timestamp')]
)
def select_page_summary(click1, click2):
    if click1 is None and click2 is None:
        return {'display': 'block'}
    if (click1 or 0) > (click2 or 0):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('page-apartments', 'style'),
    [Input('link-summary', 'n_clicks_timestamp'), Input('link-apartments', 'n_clicks_timestamp')]
)
def select_page_apartments(click1, click2):
    # if click1 is None and click2 is None:
    #     return {'display': 'block'}
    if (click2 or 0) > (click1 or 0):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


for link_type in ['summary', 'apartments']:
    @app.callback(
        Output('link-' + link_type, 'style'),
        [Input('page-' + link_type, 'style')],
        [State('link-' + link_type, 'style')]
    )
    def update_link_style(page_style, style):
        if page_style and page_style['display'] == 'block':
            style['color'] = '#17becf'
        else:
            style['color'] = 'white'
        return style


def get_selected_growth_data(rows, selected_row_indices):
    df = pd.DataFrame([rows[i] for i in selected_row_indices])
    df['cell_counts'] = df['cell_counts'].apply(json.loads)
    df['acq_ids'] = df['acq_ids'].apply(json.loads)
    return df


@app.callback(
    Output('apartment-dropdown', 'options'),
    [Input('table-growth-data', 'selected_row_indices'), Input('table-growth-data', 'rows')]
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
    [State('table-growth-data', 'selected_row_indices'), State('table-growth-data', 'rows')]
)
def update_apartment_animations(selected_apartment, selected_row_indices, rows):
    if not selected_row_indices or not rows or not selected_apartment:
        return None
    df = get_selected_growth_data(rows, selected_row_indices)
    df = data.get_apartment_image_data(df)
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
    [Input('table-growth-data', 'selected_row_indices'), Input('table-growth-data', 'rows')]
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
    Output('table-growth-data', 'selected_row_indices'),
    [Input('reset-growth-table', 'n_clicks')]
)
def reset_growth_table_selected_rows(_):
    return []


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
    df = data.get_growth_data()
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


def run_server():
    app.run_server(debug=True, port=cfg.app_port, host=cfg.app_host_ip)

# celldom run_overview_app /lab/repos/celldom/config/experiment/experiment_example_G3.yaml /lab/data/celldom/output/20180820-G3-full