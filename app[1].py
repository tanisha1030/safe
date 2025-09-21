
import streamlit as st
import pandas as pd
import geopandas as gpd
import zipfile, io, os, json, requests
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from branca.colormap import linear
from geopy.geocoders import Nominatim
import math
import tempfile

st.set_page_config(layout="wide", page_title="India Crime Heatmap")

st.title("India District-level Crime Heatmap")
st.markdown("""
Upload the ZIP file containing district-wise crime CSVs (or place your processed CSV as `data/crime_district.csv`).
The app will generate a choropleth showing low (green), medium (yellow) and high (red) crime zones by district.
""")

# Sidebar - upload
uploaded = st.sidebar.file_uploader("Upload archive.zip with crime CSVs (optional)", type=["zip"])
geojson_choice = st.sidebar.selectbox("GeoJSON source", ("Download from recommended GitHub (default)", "Upload my geojson"))

geojson_file = None
if geojson_choice == "Upload my geojson":
    geojson_file = st.sidebar.file_uploader("Upload India districts GeoJSON", type=["geojson","json"])
else:
    st.sidebar.write("Will try to download district geojson from a community repo if not uploaded.")
    # default raw URL (community-maintained)
    geojson_url = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"

# Load crime CSV from uploaded ZIP or local
@st.cache_data
def load_crime_from_zip(zfile_bytes):
    import zipfile, pandas as pd, io
    z = zipfile.ZipFile(io.BytesIO(zfile_bytes))
    # heuristics: find district-wise IPC or crimes against women files
    csvs = [n for n in z.namelist() if n.lower().endswith('.csv')]
    # prefer files with "District" and "IPC" or "crimes_committed"
    candidates = [n for n in csvs if "district" in n.lower() or "district_wise" in n.lower()]
    if not candidates:
        candidates = csvs
    # try to read the largest candidate
    best = max(candidates, key=lambda x: z.getinfo(x).file_size)
    df = pd.read_csv(z.open(best), low_memory=False)
    return df, best

crime_df = None
crime_source_name = None
if uploaded is not None:
    bytes_data = uploaded.read()
    try:
        df, name = load_crime_from_zip(bytes_data)
        crime_df = df
        crime_source_name = name
        st.success(f"Loaded crime CSV from ZIP: {name}")
    except Exception as e:
        st.error("Failed to read uploaded ZIP: " + str(e))

# fallback: look for a preloaded CSV in data/
if crime_df is None:
    local_path = "data/crime_district.csv"
    if os.path.exists(local_path):
        crime_df = pd.read_csv(local_path, low_memory=False)
        crime_source_name = local_path
        st.info(f"Loaded local crime CSV: {local_path}")
    else:
        st.info("No crime CSV found. You can upload a zip or place a processed CSV at data/crime_district.csv")
        st.stop()

st.write("Preview of crime data (first rows):")
st.dataframe(crime_df.head())

# Basic preprocessing: try to find district name and a crime count column
def preprocess_crime_df(df):
    # Look for columns that look like district and total crimes or year columns
    col_candidates = {c:c.lower() for c in df.columns}
    # find district-like column
    district_col = None
    for c,l in col_candidates.items():
        if "district" in l or "district_name" in l or "name of district" in l or "name"==l:
            district_col = c
            break
    if district_col is None:
        # fallback to first column
        district_col = df.columns[0]
    # find a total crime column - prefer 'total' or year-like
    total_col = None
    for c,l in col_candidates.items():
        if "total" in l or "cases" in l or "crime" in l:
            total_col = c
            break
    # if multiple year columns, pick the most recent numeric column
    if total_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            total_col = numeric_cols[-1]
    # create cleaned df
    out = df[[district_col]].copy()
    out.columns = ["district"]
    if total_col is not None:
        out["crime_count"] = pd.to_numeric(df[total_col], errors='coerce').fillna(0)
    else:
        out["crime_count"] = 0
    # normalize district strings
    out['district_norm'] = out['district'].astype(str).str.lower().str.replace(r'[^a-z0-9 ]','', regex=True).str.strip()
    return out

crime_clean = preprocess_crime_df(crime_df)
st.write("Processed district -> crime_count preview:")
st.dataframe(crime_clean.head())

# Load GeoJSON
@st.cache_data
def load_geojson(geojson_file, geojson_url=None):
    import json, requests
    if geojson_file is not None:
        gj = json.load(geojson_file)
    else:
        r = requests.get(geojson_url, timeout=30)
        r.raise_for_status()
        gj = r.json()
    # convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(gj["features"])
    # ensure district name column exists - try 'DISTRICT' 'district' 'NAME'
    possible = [c for c in gdf.columns if 'dist' in c.lower() or 'name' in c.lower()]
    return gdf, possible

gdf = None
if geojson_file is not None:
    try:
        gdf, possible_cols = load_geojson(geojson_file)
    except Exception as e:
        st.error("Failed to load uploaded geojson: "+str(e))
        st.stop()
else:
    try:
        gdf, possible_cols = load_geojson(None, geojson_url)
    except Exception as e:
        st.error("Failed to download geojson from remote: "+str(e))
        st.stop()

st.write("GeoJSON properties columns detected:", possible_cols)
st.write(gdf.head())

# match district names
def merge_crime_geo(gdf, crime_clean):
    gdf2 = gdf.copy()
    # try to find district name column
    name_col = None
    for c in gdf2.columns:
        if 'dist' in c.lower() or 'name' in c.lower():
            name_col = c
            break
    if name_col is None:
        st.error("Could not find district name column in geojson.")
        st.stop()
    gdf2['district_norm'] = gdf2[name_col].astype(str).str.lower().str.replace(r'[^a-z0-9 ]','', regex=True).str.strip()
    merged = gdf2.merge(crime_clean[['district_norm','crime_count']], on='district_norm', how='left')
    merged['crime_count'] = merged['crime_count'].fillna(0)
    return merged, name_col

merged, name_col = merge_crime_geo(gdf, crime_clean)
st.write("Merged preview:")
st.dataframe(merged[[name_col,'crime_count']].head())

# Classify into low/medium/high using quantiles
q1 = merged['crime_count'].quantile(0.33)
q2 = merged['crime_count'].quantile(0.66)
def classify(n):
    if n<=q1:
        return "Low"
    elif n<=q2:
        return "Medium"
    else:
        return "High"
merged['safety_level'] = merged['crime_count'].apply(classify)

# Build a folium map
m = folium.Map(location=[22.0,80.0], zoom_start=5)
# color map
colormap = linear.StepColormap(['green','yellow','red'], vmin=merged['crime_count'].min(), vmax=merged['crime_count'].max(), index=[merged['crime_count'].min(), q1, q2, merged['crime_count'].max()])
colormap.caption = "Crime count (low → high)"
colormap.add_to(m)

folium.GeoJson(
    merged.to_json(),
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['crime_count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=[name_col,'crime_count','safety_level'], aliases=["District","Crime Count","Safety Level"])
).add_to(m)

st.write("Map (choropleth):")
st_data = st_folium(m, width=900, height=600)

# Location input box
st.sidebar.header("Find safety around your location")
loc_input = st.sidebar.text_input("Provide me info about the place you are in exactly", placeholder="Type address, city or 'lat,lon'")

def geocode_text(s):
    geolocator = Nominatim(user_agent="crime-heatmap-app")
    # if it's lat,lon
    try:
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            if len(parts)==2:
                lat = float(parts[0]); lon=float(parts[1])
                return (lat, lon, None)
    except:
        pass
    try:
        loc = geolocator.geocode(s, timeout=10)
        if loc:
            return (loc.latitude, loc.longitude, loc.address)
    except Exception as e:
        return None

if loc_input:
    with st.spinner("Geocoding and analyzing..."):
        geo = geocode_text(loc_input)
        if geo is None:
            st.error("Could not geocode the input. Try 'city, state' or 'lat,lon'.")
        else:
            lat, lon, address = geo
            st.sidebar.markdown(f"**Resolved:** {address} ({lat:.5f}, {lon:.5f})")
            # find which district polygon contains the point
            pt = Point(lon, lat)
            # ensure same CRS
            point_gdf = gpd.GeoDataFrame([[pt]], columns=['geometry'], crs=merged.crs)
            # if crs mismatch, try to set both to EPSG:4326
            try:
                merged = merged.to_crs(epsg=4326)
            except:
                pass
            found = merged[merged.contains(pt)]
            if len(found)==0:
                # try within buffer distance (0.1 deg)
                found = merged[merged.distance(pt) < 0.3]
            if len(found)>0:
                row = found.iloc[0]
                st.success(f"You are in district: **{row[name_col]}** — Safety Level: **{row['safety_level']}** (Crime count: {int(row['crime_count'])})")
            else:
                st.info("Could not confidently map your point to a district in the GeoJSON.")
            # Query Overpass API for nearest police stations and residential amenities
            overpass_url = "https://overpass-api.de/api/interpreter"
            # radius in meters
            radius = 5000
            query = f\"\"\"
            [out:json][timeout:25];
            (
              node["amenity"="police"](around:{radius},{lat},{lon});
              node["amenity"="police_station"](around:{radius},{lat},{lon});
              node["amenity"="police"](around:{radius},{lat},{lon});
              way["amenity"="police"](around:{radius},{lat},{lon});
            );
            out center 20;
            \"\"\"
            try:
                res = requests.post(overpass_url, data={'data': query}, timeout=60)
                data = res.json()
                elems = data.get('elements', [])
            except Exception as e:
                elems = []
            markers = []
            # add user marker
            folium.Marker([lat, lon], tooltip="You are here", icon=folium.Icon(color='blue',icon='user')).add_to(m)
            for el in elems:
                if 'lat' in el:
                    el_lat = el['lat']; el_lon = el['lon']
                elif 'center' in el:
                    el_lat = el['center']['lat']; el_lon = el['center']['lon']
                else:
                    continue
                name = el.get('tags', {}).get('name', 'Police Station')
                popup = f\"{name}\"
                folium.Marker([el_lat, el_lon], tooltip=name, popup=popup, icon=folium.Icon(color='red',icon='info-sign')).add_to(m)
                markers.append((name, el_lat, el_lon))
            st.write("Nearest police stations (within 5 km):")
            if markers:
                for nm, alat, alon in markers:
                    st.write(f"- {nm} — ({alat:.5f}, {alon:.5f})")
            else:
                st.write("No police stations found within 5 km via Overpass API.")
            # show updated map
            st_folium(m, width=900, height=600)
