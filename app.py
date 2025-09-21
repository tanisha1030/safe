# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob
import json
from shapely.geometry import Point
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
import requests

st.set_page_config(page_title="India Crime Heatmap", layout="wide")

st.title("India Crime Heatmap & Women's Safety Analysis")

# ==============================
# Load and aggregate CSV data
# ==============================
data_files = glob.glob("data/*.csv")
all_data = []

for file in data_files:
    df = pd.read_csv(file)
    all_data.append(df)

if all_data:
    crime_df = pd.concat(all_data, ignore_index=True)
    st.success(f"Loaded and aggregated {len(data_files)} CSV(s); found {crime_df['District'].nunique()} distinct normalized districts.")
else:
    st.warning("No CSV files found in 'data/' folder. Please upload your crime CSV files.")

# ==============================
# Load India districts GeoJSON
# ==============================
geojson_url = "https://raw.githubusercontent.com/datameet/maps/master/Districts/India_districts.geojson"
try:
    geojson_data = requests.get(geojson_url).json()
    st.success("Loaded India districts GeoJSON from GitHub.")
except Exception as e:
    st.warning(f"Could not load remote GeoJSON. Please upload a district-level GeoJSON file. Error: {e}")
    geojson_file = st.file_uploader("Upload District GeoJSON", type="geojson")
    if geojson_file:
        geojson_data = json.load(geojson_file)

# ==============================
# Merge crime data with GeoJSON
# ==============================
if 'crime_df' in locals() and 'geojson_data' in locals():
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    merged = gdf.merge(crime_df, left_on="NAME_2", right_on="District", how="left")
else:
    st.stop()

# ==============================
# Define color scale for crimes
# ==============================
crime_values = merged['Total_Crimes'].fillna(0)
bins = [0, 50, 200, 500, np.inf]
colors = ["green", "yellow", "orange", "red"]
colormap = StepColormap(colors=colors, index=bins, vmin=0, vmax=crime_values.max())

# ==============================
# Folium map
# ==============================
m = folium.Map(location=[23.5, 82.5], zoom_start=5)
folium.GeoJson(
    merged,
    style_function=lambda feature: {
        "fillColor": colormap(feature['properties']['Total_Crimes'] or 0),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=["NAME_2", "Total_Crimes"],
                                  aliases=["District", "Total Crimes"])
).add_to(m)

colormap.caption = "Crime Level"
colormap.add_to(m)

st.subheader("India Crime Heatmap")
st_data = st_folium(m, width=1000, height=600)

# ==============================
# Local safety analysis
# ==============================
st.subheader("Local Safety & Nearby POIs")
address = st.text_input("Enter your address or location:")
if address:
    geolocator = Nominatim(user_agent="crime_app")
    location = geolocator.geocode(address)
    if location:
        st.write(f"Coordinates: {location.latitude}, {location.longitude}")
        point = Point(location.longitude, location.latitude)
        # Find district
        district_name = None
        for _, row in merged.iterrows():
            if row['geometry'].contains(point):
                district_name = row['NAME_2']
                st.write(f"This location is in **{district_name}** district.")
                st.write(f"Total Crimes in district: {row['Total_Crimes']}")
                break
        if not district_name:
            st.warning("Location is not inside any district polygon.")
    else:
        st.warning("Could not geocode the address. Try a more specific location.")

# ==============================
# Nearby POIs
# ==============================
st.subheader("Nearby POIs (Police, Hospitals, Shelters)")
poi_df = pd.DataFrame({
    'Name': ["Police Station A", "Hospital B", "Shelter C", "Police Station D"],
    'Type': ["Police", "Hospital", "Shelter", "Police"],
    'Latitude': [28.6139, 28.615, 28.617, 28.619],
    'Longitude': [77.209, 77.21, 77.215, 77.22]
})
st.map(poi_df)
