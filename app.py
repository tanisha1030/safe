import os
import pandas as pd
import folium
import requests
from shapely.geometry import shape
from rapidfuzz import process, fuzz
from streamlit_folium import st_folium
import streamlit as st

# Constants
DATA_DIR = "data"
GEOJSON_URL = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/districts/INDIA_DISTRICTS.geojson"

# Load datasets
@st.cache_data
def load_data():
    # Load all CSVs in the data folder
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    dfs = [pd.read_csv(f) for f in all_files]
    crime_data = pd.concat(dfs, ignore_index=True)
    crime_data['District'] = crime_data['District'].str.strip()
    
    # Load GeoJSON dynamically from GitHub
    r = requests.get(GEOJSON_URL)
    r.raise_for_status()
    geojson = r.json()
    
    return crime_data, geojson

crime_data, geojson = load_data()

# Extract district names from GeoJSON
district_names = [feature['properties']['DISTRICT'] for feature in geojson['features']]

# Fuzzy matching
matched = {}
unmatched = []

for district in crime_data['District'].unique():
    if district in district_names:
        matched[district] = district
    else:
        match, score, _ = process.extractOne(district, district_names, scorer=fuzz.token_sort_ratio)
        if score >= 80:
            matched[district] = match
        else:
            unmatched.append(district)

crime_data['Geo_District'] = crime_data['District'].map(matched)

# Streamlit UI
st.title("India District-Level Crime Heatmap")
st.markdown("### Select Crime Type(s) to Visualize")

# Get unique crime types
crime_types = crime_data['Crime_Type'].unique().tolist()
selected_crimes = st.multiselect("Crime Types", crime_types, default=crime_types[:3])

# Filter based on selection
filtered_data = crime_data[crime_data['Crime_Type'].isin(selected_crimes)]

# Aggregate by district
district_crime = filtered_data.groupby('Geo_District')['Crime_Count'].sum().reset_index()

# Compute quantiles for coloring
quantiles = district_crime['Crime_Count'].quantile([0.33, 0.66]).tolist()

def get_color(score):
    if score <= quantiles[0]:
        return 'green'
    elif score <= quantiles[1]:
        return 'yellow'
    else:
        return 'red'

# Map style selection
map_style = st.selectbox("Select Map Style", ["Stamen Terrain", "Stamen Toner", "Stamen Watercolor", "CartoDB positron"])

# Initialize map
m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles=map_style)

# Add choropleth
for feature in geojson['features']:
    district_name = feature['properties']['DISTRICT']
    crime_row = district_crime[district_crime['Geo_District'] == district_name]
    if not crime_row.empty:
        score = int(crime_row['Crime_Count'])
        folium.GeoJson(
            feature,
            style_function=lambda x, s=score: {
                'fillColor': get_color(s),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.6
            },
            tooltip=folium.Tooltip(f"{district_name}<br>Crime Score: {score}")
        ).add_to(m)

# Add markers with all selected crime types counts
from folium.plugins import MarkerCluster
marker_cluster = MarkerCluster().add_to(m)

for feature in geojson['features']:
    district_name = feature['properties']['DISTRICT']
    crime_row = filtered_data[filtered_data['Geo_District'] == district_name]
    if not crime_row.empty:
        # Create popup text for all selected crimes
        popup_text = f"<b>{district_name}</b><br>"
        for ctype in selected_crimes:
            count = crime_row[crime_row['Crime_Type'] == ctype]['Crime_Count'].sum()
            popup_text += f"{ctype}: {count}<br>"
        geom = shape(feature['geometry'])
        if geom.is_valid:
            lon, lat = geom.centroid.x, geom.centroid.y
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=get_color(crime_row['Crime_Count'].sum()),
                fill=True,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(marker_cluster)

# Display map
st_data = st_folium(m, width=900, height=600)

# Show unmatched districts
if unmatched:
    st.warning(f"The following districts could not be matched: {unmatched}")
