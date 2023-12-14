import json
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash, html, dcc, callback, Output, Input

app = Dash(__name__)

df = pd.read_csv('datasets/names_by_state_cleaned.csv')
f = open("us-states/us-states.json")
us_states = json.load(f)

year_values = df["year"].unique()
rank_values = df["rank"].unique()

initial_rank = 5
initial_sex = "Male"
initial_year = 1910
initial_state = "AK"

ranks = []
for i in range(0, initial_rank):
    ranks.append(i+1)

names_in_initial_rank = df[(df["rank"].isin(ranks)) & (df["sex"] == initial_sex[0]) & (df["year"] == initial_year) & (df["state"] == initial_state)]["name"].unique()


app.layout = html.Div(className = 'dashboard', children = [
    html.H1(children='Popular Names Per US State', style={'textAlign':'center'}),
    html.Div(className='filters', children=[
        html.Div(children=[
            html.H2(['Gender'], style={'font-weight': 'bold', "text-align": "center","offset":1}),
            dcc.RadioItems(
                id='sex-radio-selection', 
                options=[{'label': k, 'value': k} for k in ["Male", "Female"]],
                value=initial_sex,
                inline=True,
                style={"text-align": "center"}
            )], style=dict(width='33.33%')),
        html.Div(children=[
            html.H2(['Top N Names'], style={'font-weight': 'bold', "text-align": "center","offset":1}),
            dcc.RadioItems(
                id='rank-dropdown-selection', 
                options=[5, 10, 15],
                value=initial_rank,
                inline=True,
                style={"text-align": "center"}
            )], style=dict(width='33.33%')),
        html.Div(children=[
            html.H3(['Choose Name To See Popularity Per State On Map'], style={'font-weight': 'bold', "text-align": "center","offset":6}),
            dcc.Dropdown(
                options=[{'label': k, 'value': k} for k in names_in_initial_rank],
                id='top-rank-name-dropdown-selection',
                value=names_in_initial_rank[0],
                style={"text-align": "center"}
            ),
            html.Label(['**if map is fully gray, make sure a name is selected in dropdown above'], style={"text-align": "center","offset":6}),
            ], style=dict(width='15%')),
            html.Div(children=[
                html.Label(['Click on state in map below to view Top <N> Popular <Gender> Names in bar graph below the map.'], style={"text-align": "right"}),
            ],  style=dict(width='33.33%')),
        ], style=dict(display='flex')),
     dcc.Loading(
        id="loading-2",
        children=[
            dcc.Graph(
                id='map'
            ),
            dcc.Graph(id='bar_top_n')
        ],
        type="circle"
     )
])

@callback(
    [
        Output('map', 'figure'),
        Output('bar_top_n', 'figure'),
        Output("top-rank-name-dropdown-selection", "options")
    ],
    [
        Input('sex-radio-selection', 'value'),
        Input('rank-dropdown-selection', 'value'),
        Input('top-rank-name-dropdown-selection', 'value'),
        Input('map', 'clickData')
    ]
)
def update_graphs(sex_value, rank_value, name_for_map_value, click_data):

    df_filter = df[df["sex"] == sex_value[0]]
    ranks = []
    for i in range(0, rank_value):
        ranks.append(i+1)

    state = initial_state
    year = initial_year
    print(state)
    if click_data:
        print(click_data)
        state = click_data['points'][0]['location']
        year = click_data['points'][0]['customdata'][0]

    top_rank_name_dropdown_options = df[(df["rank"].isin(ranks)) & (df["sex"] == sex_value[0]) & (df["state"] == state)]["name"].unique()

    if not name_for_map_value:
        name_for_map_value = top_rank_name_dropdown_options[0]

    RENAMED_COUNT_COLUMN = "Population with Name"
    df_filter = df_filter.rename(columns={"count": RENAMED_COUNT_COLUMN})

    df_filter = df_filter.sort_values(by=["year"])

    df_map_filter = df_filter[(df_filter["rank"].isin(ranks)) & (df_filter["name"] == name_for_map_value)]

    df_bar_filter = df_filter[(df_filter["state"] == state) & (df_filter["rank"].isin(ranks)) & (df_filter["year"] == year)]
    df_bar_filter = df_bar_filter.sort_values(by=["rank"])

    fig_map = px.choropleth(df_map_filter, geojson=us_states, 
                            locations='state', 
                            color=RENAMED_COUNT_COLUMN,
                            animation_frame='year',
                            animation_group='state',
                            hover_name="state",
                            custom_data='year',
                            color_continuous_scale="Reds",
                            range_color=(0, 1100),
                            scope="usa"
                          )


    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


    bar_chart_title = "Top %i Popular %s Names In State %s Of %i"%(rank_value, sex_value, state, year)
    
    fig_bar = px.bar(df_bar_filter, 
                     x="name", 
                     y=RENAMED_COUNT_COLUMN, 
                     title=bar_chart_title)
    fig_bar.update_traces(marker_color='#D62728')
    fig_bar.update_layout(margin={"r":50,"t":50,"l":0,"b":0})

    return [fig_map, fig_bar, top_rank_name_dropdown_options]

if __name__ == '__main__':
    app.run(debug=True)
