import dash
import plotly.graph_objects as go
import plotly.express as px
import pickle
import tropycal.tracks as tracks
import pandas as pd
import numpy as np
import cachetools
import functools
import hashlib
import os
import argparse
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import statsmodels.api as sm
import schedule
import time
import threading
import requests
from io import StringIO
import tempfile
import csv
from collections import defaultdict
import shutil
import filecmp

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Typhoon Analysis Dashboard')
parser.add_argument('--data_path', type=str, default=os.getcwd(), help='Path to the data directory')
args = parser.parse_args()

# Set data paths
DATA_PATH = args.data_path

ONI_DATA_PATH = os.path.join(DATA_PATH, 'oni_data.csv')
TYPHOON_DATA_PATH = os.path.join(DATA_PATH, 'processed_typhoon_data.csv')
LOCAL_IBTRACS_PATH = os.path.join(DATA_PATH, 'ibtracs.WP.list.v04r01.csv')
IBTRACS_URI = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv'

CACHE_FILE = 'ibtracs_cache.pkl'
CACHE_EXPIRY_DAYS = 1
last_oni_update = None

color_map = {
    'C5 Super Typhoon': 'rgb(255, 0, 0)',      # Red
    'C4 Very Strong Typhoon': 'rgb(255, 63, 0)',  # Red-Orange
    'C3 Strong Typhoon': 'rgb(255, 127, 0)',    # Orange
    'C2 Typhoon': 'rgb(255, 191, 0)',          # Orange-Yellow
    'C1 Typhoon': 'rgb(255, 255, 0)',          # Yellow
    'Tropical Storm': 'rgb(0, 255, 255)',       # Cyan
    'Tropical Depression': 'rgb(173, 216, 230)'  # Light Blue
}

def convert_typhoon_data(input_file, output_file):
    with open(input_file, 'r') as infile:
        # Skip header and units lines
        next(infile)
        next(infile)
        
        reader = csv.reader(infile)
        
        # Store data for each SID
        sid_data = defaultdict(list)
        
        for row in reader:
            if not row:
                continue
            
            sid = row[0]
            iso_time = row[6]
            sid_data[sid].append((row, iso_time))

    with open(output_file, 'w', newline='') as outfile:
        fieldnames = ['SID', 'ISO_TIME', 'LAT', 'LON', 'SEASON', 'NAME', 'WMO_WIND', 'WMO_PRES', 'USA_WIND', 'USA_PRES', 'START_DATE', 'END_DATE']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for sid, data in sid_data.items():
            start_date = min(data, key=lambda x: x[1])[1]
            end_date = max(data, key=lambda x: x[1])[1]
            
            for row, iso_time in data:
                writer.writerow({
                    'SID': row[0],
                    'ISO_TIME': iso_time,
                    'LAT': row[8],
                    'LON': row[9],
                    'SEASON': row[1],
                    'NAME': row[5],
                    'WMO_WIND': row[10].strip() or ' ',
                    'WMO_PRES': row[11].strip() or ' ',
                    'USA_WIND': row[23].strip() or ' ',
                    'USA_PRES': row[24].strip() or ' ',
                    'START_DATE': start_date,
                    'END_DATE': end_date
                })

def load_ibtracs_data():
    if os.path.exists(CACHE_FILE):
        cache_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
            print("Loading data from cache...")
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    
    if os.path.exists(LOCAL_IBTRACS_PATH):
        print("Using local IBTrACS file...")
        ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_IBTRACS_PATH)
    else:
        print("Local IBTrACS file not found. Fetching data from remote server...")
        try:
            response = requests.get(IBTRACS_URI)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name
            
            shutil.move(temp_file_path, LOCAL_IBTRACS_PATH)
            print(f"Downloaded data saved to {LOCAL_IBTRACS_PATH}")
            
            ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_IBTRACS_PATH)
        except requests.RequestException as e:
            print(f"Error downloading data: {e}")
            print("No local file available and download failed. Unable to load IBTrACS data.")
            return None
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(ibtracs, f)
    
    return ibtracs

def update_ibtracs_data():
    global ibtracs
    print("Checking for IBTrACS data updates...")

    try:
        # Get last modified time of remote file
        response = requests.head(IBTRACS_URI)
        remote_last_modified = datetime.strptime(response.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S GMT')

        # Get last modified time of local file
        if os.path.exists(LOCAL_IBTRACS_PATH):
            local_last_modified = datetime.fromtimestamp(os.path.getmtime(LOCAL_IBTRACS_PATH))
        else:
            local_last_modified = datetime.min

        # Compare modification times
        if remote_last_modified <= local_last_modified:
            print("Local IBTrACS data is up to date. No update needed.")
            if os.path.exists(CACHE_FILE):
                os.utime(CACHE_FILE, None)
                print("Cache file timestamp updated.")
            return

        print("Remote data is newer. Updating IBTrACS data...")
        
        # Download new data
        response = requests.get(IBTRACS_URI)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_file.write(response.text)
            temp_file_path = temp_file.name
        
        shutil.move(temp_file_path, LOCAL_IBTRACS_PATH)
        print(f"Downloaded data saved to {LOCAL_IBTRACS_PATH}")
        
        os.utime(LOCAL_IBTRACS_PATH, (remote_last_modified.timestamp(), remote_last_modified.timestamp()))
        
        ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_IBTRACS_PATH)
        
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(ibtracs, f)
        print("IBTrACS data updated and cache refreshed.")

    except requests.RequestException as e:
        print(f"Error checking or downloading data: {e}")
        if os.path.exists(LOCAL_IBTRACS_PATH):
            print("Using existing local file.")
            ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_IBTRACS_PATH)
            if os.path.exists(CACHE_FILE):
                os.utime(CACHE_FILE, None)
                print("Cache file timestamp updated.")
        else:
            print("No local file available. Update failed.")

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

def cache_key_generator(*args, **kwargs):
    key = hashlib.md5()
    for arg in args:
        key.update(str(arg).encode())
    for k, v in sorted(kwargs.items()):
        key.update(str(k).encode())
        key.update(str(v).encode())
    return key.hexdigest()

def categorize_typhoon(wind_speed):
    wind_speed_kt = wind_speed / 2  # Convert m/s to kt
    
    if wind_speed_kt >= 137 / 2.35:
        return 'C5 Super Typhoon'
    elif wind_speed_kt >= 113 / 2.35:
        return 'C4 Very Strong Typhoon'
    elif wind_speed_kt >= 96 / 2.35:
        return 'C3 Strong Typhoon'
    elif wind_speed_kt >= 83 / 2.35:
        return 'C2 Typhoon'
    elif wind_speed_kt >= 64 / 2.35:
        return 'C1 Typhoon'
    elif wind_speed_kt >= 34 / 2.35:
        return 'Tropical Storm'
    else:
        return 'Tropical Depression'

@functools.lru_cache(maxsize=None)
def process_typhoon_data_cached(typhoon_data_hash):
    return process_typhoon_data(typhoon_data)

def process_typhoon_data(typhoon_data):
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    typhoon_data['USA_WIND'] = pd.to_numeric(typhoon_data['USA_WIND'], errors='coerce')
    typhoon_data['USA_PRES'] = pd.to_numeric(typhoon_data['USA_PRES'], errors='coerce')
    typhoon_data['LON'] = pd.to_numeric(typhoon_data['LON'], errors='coerce')
    
    typhoon_max = typhoon_data.groupby('SID').agg({
        'USA_WIND': 'max',
        'USA_PRES': 'min',
        'ISO_TIME': 'first',
        'SEASON': 'first',
        'NAME': 'first',
        'LAT': 'first',
        'LON': 'first'
    }).reset_index()
    
    typhoon_max['Month'] = typhoon_max['ISO_TIME'].dt.strftime('%m')
    typhoon_max['Year'] = typhoon_max['ISO_TIME'].dt.year
    typhoon_max['Category'] = typhoon_max['USA_WIND'].apply(categorize_typhoon)
    return typhoon_max

def process_typhoon_data_with_cache(typhoon_data):
    typhoon_data_hash = cache_key_generator(typhoon_data.to_json())
    return process_typhoon_data_cached(typhoon_data_hash)

def load_data(typhoon_data_path):
    typhoon_data = pd.read_csv(typhoon_data_path, low_memory=False)
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    typhoon_data = typhoon_data.dropna(subset=['ISO_TIME'])
    
    print(f"Typhoon data shape after cleaning: {typhoon_data.shape}")
    print(f"Year range: {int(typhoon_data['ISO_TIME'].dt.year.min())} - {int(typhoon_data['ISO_TIME'].dt.year.max())}")
    
    return typhoon_data

@functools.lru_cache(maxsize=None)
def get_storm_data(storm_id):
    return ibtracs.get_storm(storm_id)

def filter_west_pacific_coordinates(lons, lats):
    mask = (100 <= lons) & (lons <= 180) & (0 <= lats) & (lats <= 40)
    return lons[mask], lats[mask]

def generate_cluster_equations(cluster_indices, standardized_routes):

    # Collect all routes for the current cluster
    cluster_route_vectors = standardized_routes[cluster_indices]
    
    # Reshape and combine all the routes
    all_lons = []
    all_lats = []
    for route_vector in cluster_route_vectors:
        route_points = route_vector.reshape(-1, 2)
        lons = route_points[:, 0]
        lats = route_points[:, 1]
        all_lons.extend(lons)
        all_lats.extend(lats)
    
    X = np.array(all_lons)
    y = np.array(all_lats)
    
    # Ensure there are enough data points
    if len(X) < 9:
        print(f"Not enough data points to fit Fourier series for cluster (need at least 9, have {len(X)})")
        equations = []
        x_min, x_max = X.min(), X.max()
        return equations, (x_min, x_max)
    
    x_min = X.min()
    x_max = X.max()
    
    # Normalize X to [0, 2π]
    if x_max == x_min:
        X_normalized = np.zeros_like(X)
    else:
        X_normalized = 2 * np.pi * (X - x_min) / (x_max - x_min)
    
    # Define Fourier series up to the 4th order
    def fourier_series(x, a0, a1, b1, a2, b2, a3, b3, a4, b4):
        return (a0 +
                a1 * np.cos(x) + b1 * np.sin(x) +
                a2 * np.cos(2 * x) + b2 * np.sin(2 * x) +
                a3 * np.cos(3 * x) + b3 * np.sin(3 * x) +
                a4 * np.cos(4 * x) + b4 * np.sin(4 * x))
    
    try:
        params, _ = curve_fit(fourier_series, X_normalized, y)
        a0, a1, b1, a2, b2, a3, b3, a4, b4 = params
        # Construct the equation as a string for display
        equation_str = f"y = {a0:.3f} + {a1:.3f}*cos(x) + {b1:.3f}*sin(x) + {a2:.3f}*cos(2x) + {b2:.3f}*sin(2x) + {a3:.3f}*cos(3x) + {b3:.3f}*sin(3x) + {a4:.3f}*cos(4x) + {b4:.3f}*sin(4x)"
        equations = [("Fourier Series Fit", equation_str)]
    except Exception as e:
        print(f"Error fitting Fourier series for cluster: {e}")
        equations = []
        
    return equations, (x_min, x_max)

app = dash.Dash(__name__)

# Classification standards
atlantic_standard = {
    'C5 Super Typhoon': {'wind_speed': 137, 'color': 'rgb(255, 0, 0)'},
    'C4 Very Strong Typhoon': {'wind_speed': 113, 'color': 'rgb(255, 63, 0)'},
    'C3 Strong Typhoon': {'wind_speed': 96, 'color': 'rgb(255, 127, 0)'},
    'C2 Typhoon': {'wind_speed': 83, 'color': 'rgb(255, 191, 0)'},
    'C1 Typhoon': {'wind_speed': 64, 'color': 'rgb(255, 255, 0)'},
    'Tropical Storm': {'wind_speed': 34, 'color': 'rgb(0, 255, 255)'},
    'Tropical Depression': {'wind_speed': 0, 'color': 'rgb(173, 216, 230)'}
}

taiwan_standard = {
    'Strong Typhoon': {'wind_speed': 51.0, 'color': 'rgb(255, 0, 0)'},       # >= 51.0 m/s
    'Medium Typhoon': {'wind_speed': 33.7, 'color': 'rgb(255, 127, 0)'},     # 33.7-50.9 m/s
    'Mild Typhoon': {'wind_speed': 17.2, 'color': 'rgb(255, 255, 0)'},       # 17.2-33.6 m/s
    'Tropical Depression': {'wind_speed': 0, 'color': 'rgb(173, 216, 230)'}  # < 17.2 m/s
}

app.layout = html.Div([
    html.H1("Typhoon Analysis Dashboard"),
    
    html.Div([
        dcc.Input(id='start-year', type='number', placeholder='Start Year', value=2000, min=1900, max=2024, step=1),
        dcc.Input(id='start-month', type='number', placeholder='Start Month', value=1, min=1, max=12, step=1),
        dcc.Input(id='end-year', type='number', placeholder='End Year', value=2024, min=1900, max=2024, step=1),
        dcc.Input(id='end-month', type='number', placeholder='End Month', value=6, min=1, max=12, step=1),
        dcc.Dropdown(
            id='enso-dropdown',
            options=[
                {'label': 'All Years', 'value': 'all'},
                {'label': 'El Niño', 'value': 'el_nino'},
                {'label': 'La Niña', 'value': 'la_nina'},
                {'label': 'Neutral Years', 'value': 'neutral'}
            ],
            value='all'
        ),
        html.Button('Analyze', id='analyze-button', n_clicks=0),
    ]),
    
    html.Div([
        html.P("Number of Clusters"),
        dcc.Input(id='n-clusters', type='number', placeholder='Number of Clusters', value=5, min=1, max=20, step=1),
        html.Button('Show Clusters', id='show-clusters-button', n_clicks=0),
        html.Button('Show Typhoon Routes', id='show-routes-button', n_clicks=0),
    ]),

    dcc.Graph(id='typhoon-routes-graph'),
    
    html.Div(id='cluster-equation-results'),

    html.H2("Typhoon Path Analysis"),
    html.Div([
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in range(1950, 2025)],
            value=2024,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='typhoon-dropdown',
            style={'width': '300px'}
        ),
        dcc.Dropdown(
            id='classification-standard',
            options=[
                {'label': 'Atlantic Standard', 'value': 'atlantic'},
                {'label': 'Taiwan Standard', 'value': 'taiwan'}
            ],
            value='atlantic',
            style={'width': '200px'}
        )
    ], style={'display': 'flex', 'gap': '10px'}),
    
    dcc.Graph(id='typhoon-path-animation'),
    html.Div([
        html.H3("Typhoon Generation Analysis"),
        html.Div(id='typhoon-count-analysis'),
        html.Div(id='concentrated-months-analysis'),
    ]),
    html.Div(id='cluster-info'),
    
    html.Div([
        dcc.Dropdown(
            id='classification-standard',
            options=[
                {'label': 'Atlantic Standard', 'value': 'atlantic'},
                {'label': 'Taiwan Standard', 'value': 'taiwan'}
            ],
            value='atlantic',
            style={'width': '200px'}
        )
    ], style={'margin': '10px'}),
    
], style={'font-family': 'Arial, sans-serif'})

@app.callback(
    Output('year-dropdown', 'options'),
    Input('typhoon-path-animation', 'figure')
)
def initialize_year_dropdown(_):
    try:
        years = typhoon_data['ISO_TIME'].dt.year.unique()
        years = years[~np.isnan(years)]
        years = sorted(years)
        
        options = [{'label': str(int(year)), 'value': int(year)} for year in years]
        print(f"Generated options: {options[:5]}...")
        return options
    except Exception as e:
        print(f"Error in initialize_year_dropdown: {str(e)}")
        return [{'label': 'Error', 'value': 'error'}]

@app.callback(
    [Output('typhoon-dropdown', 'options'),
     Output('typhoon-dropdown', 'value')],
    [Input('year-dropdown', 'value')]
)
def update_typhoon_dropdown(selected_year):
    if not selected_year:
        raise PreventUpdate
    
    selected_year = int(selected_year)
    
    season = ibtracs.get_season(selected_year)
    storm_summary = season.summary()
    
    typhoon_options = []
    for i in range(storm_summary['season_storms']):
        storm_id = storm_summary['id'][i]
        storm_name = storm_summary['name'][i]
        typhoon_options.append({'label': f"{storm_name} ({storm_id})", 'value': storm_id})
    
    selected_typhoon = typhoon_options[0]['value'] if typhoon_options else None
    return typhoon_options, selected_typhoon

@app.callback(
    Output('typhoon-path-animation', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('typhoon-dropdown', 'value'),
     Input('classification-standard', 'value')]
)
def update_typhoon_path(selected_year, selected_sid, standard):
    if not selected_year or not selected_sid:
        raise PreventUpdate

    storm = ibtracs.get_storm(selected_sid)
    return create_typhoon_path_figure(storm, selected_year, standard)

def create_typhoon_path_figure(storm, selected_year, standard='atlantic'):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=storm.lon,
            lat=storm.lat,
            mode='lines',
            line=dict(width=2, color='gray'),
            name='Path',
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=[storm.lon[0]],
            lat=[storm.lat[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='star'),
            name='Starting Point',
            text=storm.time[0].strftime('%Y-%m-%d %H:%M'),
            hoverinfo='text+name',
        )
    )

    frames = []
    for i in range(len(storm.time)):
        category, color = categorize_typhoon_by_standard(storm.vmax[i], standard)

        frame_data = [
            go.Scattergeo(
                lon=storm.lon[:i+1],
                lat=storm.lat[:i+1],
                mode='lines',
                line=dict(width=2, color='blue'),
                name='Path Traveled',
                showlegend=False,
            ),
            go.Scattergeo(
                lon=[storm.lon[i]],
                lat=[storm.lat[i]],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='star'),
                text=category,
                textposition="top center",
                textfont=dict(size=12, color=color),
                name='Current Location',
                hovertext=f"{storm.time[i].strftime('%Y-%m-%d %H:%M')}<br>"
                          f"Category: {category}<br>"
                          f"Wind Speed: {storm.vmax[i]:.1f} m/s",
                hoverinfo='text',
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))

    fig.frames = frames

    fig.update_layout(
        title=f"{selected_year} {storm.name} Typhoon Path",
        showlegend=False,
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(100, 100, 100)',
            showocean=True,
            oceancolor='rgb(230, 250, 255)',
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 100, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f"frame{k}"],
                             {"frame": {"duration": 100, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 0}}
                            ],
                    "label": storm.time[k].strftime('%Y-%m-%d %H:%M'),
                    "method": "animate"
                }
                for k in range(len(storm.time))
            ]
        }]
    )

    return fig

@app.callback(
    [Output('typhoon-routes-graph', 'figure'),
     Output('cluster-equation-results', 'children')],
    [Input('analyze-button', 'n_clicks'),
     Input('show-clusters-button', 'n_clicks'),
     Input('show-routes-button', 'n_clicks')],
    [State('start-year', 'value'),
     State('start-month', 'value'),
     State('end-year', 'value'),
     State('end-month', 'value'),
     State('n-clusters', 'value'),
     State('enso-dropdown', 'value')]
)

def update_route_clusters(analyze_clicks, show_clusters_clicks, show_routes_clicks,
                          start_year, start_month, end_year, end_month,
                          n_clusters, enso_value):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    
    fig_routes = go.Figure()

    clusters = np.array([])
    cluster_equations = []
    
    # Clustering analysis
    west_pacific_storms = []
    for year in range(start_year, end_year + 1):
        season = ibtracs.get_season(year)
        for storm_id in season.summary()['id']:
            storm = get_storm_data(storm_id)
            storm_date = storm.time[0]
            
            lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
            if len(lons) > 1:
                west_pacific_storms.append((lons, lats))

    max_length = max(len(storm[0]) for storm in west_pacific_storms)
    standardized_routes = []
    
    for lons, lats in west_pacific_storms:
        if len(lons) < 2:
            continue
        t = np.linspace(0, 1, len(lons))
        t_new = np.linspace(0, 1, max_length)
        lon_interp = interp1d(t, lons, kind='linear')(t_new)
        lat_interp = interp1d(t, lats, kind='linear')(t_new)
        route_vector = np.column_stack((lon_interp, lat_interp)).flatten()
        standardized_routes.append(route_vector)

    # Convert the list to a NumPy array
    standardized_routes = np.array(standardized_routes)

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(standardized_routes)

    # Use KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tsne_results)

    # Count the number of typhoons in each cluster
    cluster_counts = np.bincount(clusters)

    # Plot the t-SNE reduced data points
    for i in range(n_clusters):
        cluster_points = tsne_results[clusters == i]
        fig_routes.add_trace(go.Scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            mode='markers',
            name=f'Cluster {i+1} (n={cluster_counts[i]})',
            marker=dict(size=5),
            showlegend=True,
            visible=(button_id == 'show-clusters-button')
        ))

    equations_output = []
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        equations, (lon_min, lon_max) = generate_cluster_equations(cluster_indices, standardized_routes)
        
        equations_output.append(html.H4([
            f"Cluster {i+1} (Typhoon Count: ",
                html.Span(f"{cluster_counts[i]}", style={'color': 'blue'}),
                    ")"
        ]))
        for name, eq in equations:
            equations_output.append(html.P(f"{name}: {eq}"))
        
        # Optionally, plot the cluster center or mean route

    enso_phase_text = {
        'all': 'All Years',
        'el_nino': 'El Niño',
        'la_nina': 'La Niña',
        'neutral': 'Neutral Years'
    }
    fig_routes.update_layout(
        title=f'West Pacific Typhoon Route Clustering ({start_year}-{end_year}) - {enso_phase_text[enso_value]}',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        legend_title='Clusters'
    )
    
    return fig_routes, html.Div(equations_output)

def categorize_typhoon_by_standard(wind_speed, standard='atlantic'):
    """
    Classify typhoon based on wind speed and selected standard.
    Wind speed is in knots.
    """
    if standard == 'taiwan':
        # Convert knots to m/s (Taiwan standard)
        wind_speed_ms = wind_speed * 0.514444
        
        if wind_speed_ms >= 51.0:
            return 'Strong Typhoon', taiwan_standard['Strong Typhoon']['color']
        elif wind_speed_ms >= 33.7:
            return 'Medium Typhoon', taiwan_standard['Medium Typhoon']['color']
        elif wind_speed_ms >= 17.2:
            return 'Mild Typhoon', taiwan_standard['Mild Typhoon']['color']
        else:
            return 'Tropical Depression', taiwan_standard['Tropical Depression']['color']
    else:
        # Atlantic standard uses knots
        if wind_speed >= 137:
            return 'C5 Super Typhoon', atlantic_standard['C5 Super Typhoon']['color']
        elif wind_speed >= 113:
            return 'C4 Very Strong Typhoon', atlantic_standard['C4 Very Strong Typhoon']['color']
        elif wind_speed >= 96:
            return 'C3 Strong Typhoon', atlantic_standard['C3 Strong Typhoon']['color']
        elif wind_speed >= 83:
            return 'C2 Typhoon', atlantic_standard['C2 Typhoon']['color']
        elif wind_speed >= 64:
            return 'C1 Typhoon', atlantic_standard['C1 Typhoon']['color']
        elif wind_speed >= 34:
            return 'Tropical Storm', atlantic_standard['Tropical Storm']['color']
        else:
            return 'Tropical Depression', atlantic_standard['Tropical Depression']['color']

if __name__ == "__main__":
    print(f"Using data path: {DATA_PATH}")
    ibtracs = load_ibtracs_data()
    convert_typhoon_data(LOCAL_IBTRACS_PATH, TYPHOON_DATA_PATH)
    typhoon_data = load_data(TYPHOON_DATA_PATH)

    # Daily update of IBTrACS data
    schedule.every().day.at("01:00").do(update_ibtracs_data)
    
    # Run the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_schedule)
    scheduler_thread.start()
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)
