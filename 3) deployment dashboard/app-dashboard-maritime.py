# -*- coding: utf-8 -*-
"""
TOTAL WAVE HEIGHT PREDICTION
DASHBOARD APP
@author: Mingyu, Agung, Gian

"""
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash import no_update
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash_daq as daq
import sqlalchemy
import copy

# ========================================================================================= #
#                           SET DATABASE CONNECTION - LOAD DATA                             #
# ========================================================================================= #

database_username = 'root'
database_password = 'balab'
database_ip       = '172.17.0.5'
database_name     = 'data_hackathon'
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                               format(database_username, database_password,
                                                      database_ip, database_name))

# ========================================================================================= #
#                                    CREATING THE APPS                                      #
# ========================================================================================= #

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets = [dbc.themes.SUPERHERO,'/assets/styles.css']
) 

server=app.server


# ========================================================================================= #
#                                       NAVIGATION BAR                                      #
# ========================================================================================= #


def get_navbar():
    PLOTLY_LOGO = app.get_asset_url("balab_grey_rev.png")

    logo = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=PLOTLY_LOGO, height="60px",style={'margin': '8px 0px'})),
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="https://plot.ly",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
            ]
        ),
        color="primary",
        dark=True,
        className="mb-5",
    )

    return logo


# DATA ANALYSIS LAYOUT

layout = dict(
    autosize=True,
    #automargin=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hovermode="closest",
    plot_bgcolor="#16103a",
    paper_bgcolor="#16103a",
    legend=dict(font=dict(size=10), orientation="h"),
    font_color ="#e0e1e6",
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_visible=False,
)

# TAB OF PREDICTION DASHBOARD


tab_prediction_features = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            dbc.InputGroup(
                                    [
                                        dbc.InputGroupAddon("Sensor Inputs", addon_type="prepend"),
                                        dbc.Select(
                                            options=[
                                                {"label": "Total Wave Height", "value": "TOTAL_WAVE_HEIGHT"},
                                                {"label": "Rotation per Minute", "value": "ME1_RPM"},
                                                {"label": "Fuel Consumption", "value": "ME1_FOC"},
                                            ], id = "columns_1", value="TOTAL_WAVE_HEIGHT"
                                        )
                                    ]
                                ),
                        ])
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Spinner(
                                    size="md",
                                    color="light",
                                    children=[
                                        dcc.Graph(id="line-1", config = {"displayModeBar": False}, style = {"height": "32vh"})
                                    ]
                                ),
                            ),style = {"background-color": "#16103a"}
                        ),

                    dbc.Card(
                        dbc.CardBody([
                            dbc.InputGroup(
                                    [
                                        dbc.InputGroupAddon("Sensor Inputs", addon_type="prepend"),
                                        dbc.Select(
                                            options=[
                                                {"label": "Total Wave Height", "value": "TOTAL_WAVE_HEIGHT"},
                                                {"label": "Rotation per Minute", "value": "ME1_RPM"},
                                                {"label": "Fuel Consumption", "value": "ME1_FOC"},
                                            ], id = "columns_2", value="ME1_RPM"
                                        )
                                    ]
                                ),
                        ])
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Spinner(
                                    size="md",
                                    color="light",
                                    children=[
                                        dcc.Graph(id="line-2", config = {"displayModeBar": False}, style = {"height": "32vh"})
                                    ]
                                ),
                            ),style = {"background-color": "#16103a"}
                        ),

                    dbc.Card(
                        dbc.CardBody([
                            dbc.InputGroup(
                                    [
                                        dbc.InputGroupAddon("Sensor Inputs", addon_type="prepend"),
                                        dbc.Select(
                                            options=[
                                                {"label": "Total Wave Height", "value": "TOTAL_WAVE_HEIGHT"},
                                                {"label": "Rotation per Minute", "value": "ME1_RPM"},
                                                {"label": "Fuel Consumption", "value": "ME1_FOC"},
                                            ], id = "columns_3", value="ME1_FOC"
                                        )
                                    ]
                                ),
                        ])
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Spinner(
                                    size="md",
                                    color="light",
                                    children=[
                                        dcc.Graph(id="line-3", config = {"displayModeBar": False}, style = {"height": "32vh"})
                                    ]
                                ),
                            ),style = {"background-color": "#16103a"}
                        ),
                ],lg="7", sm=12), 

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                            dbc.CardBody([
                                html.P("Total Wave Height Trend Prediction",style={'textAlign': 'center'}),
                                                            dbc.Spinner(
                                    size="md",
                                    color="light",
                                    children=[
                                        dcc.Graph(id="trend_prophet", config = {"displayModeBar": False}, style = {"height": "35vh"})
                                    ]
                                ),
                            ]),style = {"background-color": "#16103a"}
                        )
                        ],lg="12", sm=12),
                        ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Total Wave Height next 10 minutes",style={'textAlign': 'center'}),
                                    html.Div(children=[
                                        daq.LEDDisplay(
                                            id='LED-display-WAVEHEIGHT',
                                            value=00,
                                            size=48,
                                            color='#10A5F5',
                                            backgroundColor="black"
                                        ),
                                    ],style={'textAlign': 'right','verticalAlign': 'middle',"height": "12vh",})
                                ]),style = {"background-color": "#16103a"}
                            )
                        ], lg="6", sm=12),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Wave Height Gauge",style={'textAlign': 'center'}),
                                    html.Div(children=[
                                        daq.Gauge(
                                            id='level-gauge',
                                            color={
                                                "gradient": True,
                                                "ranges": {"#E0FFFF": [0., 3],"#3CB371": [3, 6] ,"#4B0082": [6, 10]},
                                            },
                                            style={'stroke-width': '42px',},
                                            size=225,
                                            max=10,
                                            min=0,
                                            value=2
                                        )
                                    ],  style={
                                                "widh": "1%",
                                                "height": "1%",
                                                "border": "1px solid rgb(190, 190, 190)",
                                                "margin": "1px 5px 0px 5px",
                                                "padding": "0px 0px 0px 0px",
                                            },)
                                ]),style = {"background-color": "#16103a"}
                            )
                        ],lg="6", sm=12)
                    ]),
                ],lg="5", sm=12),
            ]),
        ]
    ),
    className="mt-3", style = {"background-color": "#272953"}
)

# Tab Prediction Content

tab_prediction_content = [
    tab_prediction_features,
]

# Tabs Content

tabs = dbc.Tabs(
    [
        dbc.Tab(tab_prediction_content, label="Wave Height Prediction"),
    ]
)

# Jumbotron

jumbotron = dbc.Jumbotron(
    html.H4("Early Warning Service based on Sea Level Condition"),
    className="cover"
)

# ========================================================================================= #
#                                       APP CALLBACKS                                       #
# ========================================================================================= #

@app.callback(
    Output("line-1", "figure"),
    [
        Input("columns_1", "value"),
    ],
)

def line_chart(plot_columns):
    df = pd.read_sql(""" SELECT * FROM data_hackathon.data_source ORDER BY TIME_STAMP DESC LIMIT 1008""", 
                con=database_connection)# get data last 1008 rows

    dff_temp = df.sort_values('TIME_STAMP').reset_index(drop=True)
    fig = px.line(df, x="TIME_STAMP", y=plot_columns,)

    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    return fig

@app.callback(
    Output("line-2", "figure"),
    [
        Input("columns_2", "value"),
    ],
)

def line_chart(plot_columns):
    df = pd.read_sql(""" SELECT * FROM data_hackathon.data_source ORDER BY TIME_STAMP DESC LIMIT 1008""", 
                con=database_connection)# get data last 1008 rows

    dff_temp = df.sort_values('TIME_STAMP').reset_index(drop=True)
    fig = px.line(df, x="TIME_STAMP", y=plot_columns,)
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)

    return fig


@app.callback(
    Output("line-3", "figure"),
    [
        Input("columns_3", "value"),
    ],
)

def line_chart(plot_columns):
    df = pd.read_sql(""" SELECT * FROM data_hackathon.data_source ORDER BY TIME_STAMP DESC LIMIT 1008""", 
                con=database_connection)# get data last 1008 rows

    dff_temp = df.sort_values('TIME_STAMP').reset_index(drop=True)
    fig = px.line(df, x="TIME_STAMP", y=plot_columns,)
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)

    return fig


@app.callback(
    Output("trend_prophet", "figure"),
    [
        Input("columns_1", "value"),
    ],
)

def trend_pred(plot_columns):
    df_trend = pd.read_sql('select * from dm_prophet_prediction',con=database_connection)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend['ds'], y=df_trend['yhat_upper'],
        fill=None,
        mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=df_trend['ds'], y=df_trend['yhat_lower'],    
        fill='tonexty',
        mode='lines',
    ))

    fig.add_trace(go.Scatter(x=df_trend['ds'], y=df_trend['yhat'],
                            line = dict(color='royalblue', width=5, dash='dash')))

    fig.update_layout(showlegend=False)
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)

    return fig


@app.callback(
    Output("LED-display-WAVEHEIGHT", "value"),
    [
        Input("columns_1", "value"),
    ],
)
def one_step(plot_columns):
    data_lstm = pd.read_sql('select * from dm_lstm_prediction',con=database_connection)
    value = data_lstm['TOTAL_WAVE_HEIGHT'].values
    return value


@app.callback(
    Output("level-gauge", "value"),
    [
        Input("columns_1", "value"),
    ],
)
def gauge_plot(plot_columns):
    data_lstm = pd.read_sql('select * from dm_lstm_prediction',con=database_connection)
    value = data_lstm['TOTAL_WAVE_HEIGHT'].values
    return value



# ========================================================================================= #
#                                   APPLICATION LAYOUT                                      #
# ========================================================================================= #


app.layout = html.Div(
    [
        get_navbar(),
        jumbotron,
        html.Div(
            dbc.Row(dbc.Col(tabs, width=12)),
            id="mainContainer",
            style={"display": "flex", "flex-direction": "column"}
        ),
        html.P("Copyright © Business Analytics Lab MGA Team", className="footer")
    ],
)

# Navbar Toggle Callback for Mobile Devices

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(host = "0.0.0.0", port=1223)
