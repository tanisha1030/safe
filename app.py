import os
import pandas as pd
import folium
import requests
from folium.plugins import MarkerCluster
from rapidfuzz import process, fuzz
from shapely.geometry import shape
import streamlit as st
from streamlit_folium import st_folium

# Constants
DATA_DIR = "data"  # Folder containing all 57 CSV files
GEOJSON_URL = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/districts/INDIA_DISTRICTS.geojson"
OUTPUT_JSON = "district_crime_scores.json"

# Load and aggregate all CSV files
@st.cache_data
def load_crime_data():
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    crime_data = pd.concat(df_list, ignore_index=True)
    crime_data['District'] = crime_data['District'].str.strip()
    district_crime = crime_data.groupby('District')['Crime_Count'].sum().reset_index()
    return district_crime

# Load GeoJSON
@st.cache_data
def load_geojson():
    r = requests.get(GEOJSON_URL)
    r.raise_for_status()
    return r.json()

district_crime = load_crime_data()
geojson = load_geojson()

# Extract district names from GeoJSON
district_names = [feature['properties']['DISTRICT'] for feature in geojson['features']]

# Fuzzy matching of district names
matched = {}
unmatched = []

for district in district_crime['District']:
    if district in district_names:
        matched[district] = district
    else:
        match, score, _ = process.extractOne(district, district_names, scorer=fuzz.token_sort_ratio)
        if score >= 80:
            matched[district] = match
        else:
            unmatched.append(district)

district_crime['Geo_District'] = district_crime['District'].map(matched)

# Compute crime scores and quantiles for coloring
district_crime['Crime_Score'] = district_crime['Crime_Count']
quantiles = district_crime['Crime_Score'].quantile([0.33, 0.66]).tolist()

def get_color(score):
    if score <= quantiles[0]:
        return 'green'
    elif score <= quantiles[1]:
        return 'yellow'
    else:
        return 'red'

# Save precomputed scores
district_crime.to_json(OUTPUT_JSON, orient='records', indent=2)

# Streamlit UI
st.title("India District-Level Crime Heatmap")
st.markdown("### Interactive Map of Crime Scores by District")

map_style = st.selectbox("Select Map Style", ["Stamen Terrain", "Stamen Toner", "Stamen Watercolor", "CartoDB positron"])

# Initialize map
m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles=map_style)

# Add choropleth
for feature in geojson['features']:
    district_name = feature['properties']['DISTRICT']
    crime_row = district_crime[district_crime['Geo_District'] == district_name]
    if not crime_row.empty:
        score = int(crime_row['Crime_Score'])
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

# Add marker clusters
marker_cluster = MarkerCluster().add_to(m)
for feature in geojson['features']:
    district_name = feature['properties']['DISTRICT']
    crime_row = district_crime[district_crime['Geo_District'] == district_name]
    if not crime_row.empty:
        score = int(crime_row['Crime_Score'])
        geom = shape(feature['geometry'])
        if geom.is_valid:
            lon, lat = geom.centroid.x, geom.centroid.y
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=get_color(score),
                fill=True,
                fill_opacity=0.7,
                popup=f"{district_name}: {score}"
            ).add_to(marker_cluster)

# Display map in Streamlit
st_data = st_folium(m, width=800, height=600)

# Show unmatched districts
if unmatched:
    st.warning(f"The following districts could not be matched: {unmatched}")
