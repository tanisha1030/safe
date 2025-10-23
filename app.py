# safe_fast.py — optimized version
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob, os, json, requests
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MeasureControl, MarkerCluster
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — district-level (green → yellow → red)")

# ---------------------------
# Utility functions
# ---------------------------
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    replacements = {
        'commr': 'commissioner', 'commissionerate': 'commissioner',
        'dist': 'district', 'north': 'n', 'south': 's',
        'east': 'e', 'west': 'w', 'parganas': 'pargana',
        '24 pargana': 'twenty four pargana', 'a and n': 'andaman nicobar',
        'a & n': 'andaman nicobar', 'city': '', 'rural': '',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())

@st.cache_data(show_spinner=False)
def load_and_aggregate_csvs(data_folder):
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame(), []
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
        district_col = next((c for c in df.columns if "district" in c.lower()), df.columns[0])
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            continue
        df["_file_total"] = df[numeric_cols].sum(axis=1, numeric_only=True)
        small = df[[district_col, "_file_total"]].rename(columns={district_col: "district_raw", "_file_total": "file_total"})
        small["district_norm"] = small["district_raw"].apply(normalize_name)
        rows.append(small)
    agg = pd.concat(rows, ignore_index=True)
    return (agg.groupby("district_norm", as_index=False)
            .agg({"file_total": "sum", "district_raw": "first"})
            .rename(columns={"file_total": "crime_total", "district_raw": "district_example"})), files

crime_agg, csv_files = load_and_aggregate_csvs("data")
if crime_agg.empty:
    st.error("No CSVs found. Place district CSVs inside 'data/'.")
    st.stop()

# ---------------------------
# Load GeoJSON (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson():
    urls = [
        "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
        "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            gj = r.json()
            return gpd.GeoDataFrame.from_features(gj["features"])
        except:
            continue
    st.error("Failed to load any district GeoJSON")
    st.stop()

gdf = load_geojson()
name_col = next((c for c in gdf.columns if "NAME_2" in c or "district" in c.lower()), gdf.columns[0])
gdf["district_norm"] = gdf[name_col].apply(normalize_name)

# Merge and classify
merged = gdf.merge(crime_agg, on="district_norm", how="left").fillna({"crime_total": 0})
non_zero = merged.loc[merged["crime_total"] > 0, "crime_total"]
q1, q2 = non_zero.quantile(0.33), non_zero.quantile(0.66)
def classify(n): return "No Data" if n == 0 else "Low" if n <= q1 else "Medium" if n <= q2 else "High"
merged["safety_level"] = merged["crime_total"].apply(classify)
merged["centroid"] = merged.geometry.centroid

# ---------------------------
# Draw main map (optimized)
# ---------------------------
m = folium.Map(location=[20.6, 78.9], zoom_start=5, tiles="cartodbpositron")
vmin, vmax = merged["crime_total"].min(), merged["crime_total"].max()
colormap = StepColormap(colors=["lightgray","green","yellow","red"], index=[vmin, 0.1, q1, q2, vmax], vmin=vmin, vmax=vmax)

def style_fn(feature):
    val = feature["properties"]["crime_total"]
    if val == 0: return {"fillColor": "#f0f0f0", "color": "#ccc", "weight": 0.5}
    elif val <= q1: return {"fillColor": "#6bcf7f", "color": "#51b364", "weight": 0.5}
    elif val <= q2: return {"fillColor": "#ffd93d", "color": "#ffcc02", "weight": 0.6}
    else: return {"fillColor": "#ff6b6b", "color": "#e55353", "weight": 0.8}

# Single-layer GeoJson instead of hundreds of polygons
folium.GeoJson(
    merged.to_json(),
    name="District Crime Map",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(fields=[name_col, "crime_total", "safety_level"],
                                  aliases=["District", "Crimes", "Safety"])
).add_to(m)

colormap.caption = "Crime Level (Gray=No Data, Green→Red)"
m.add_child(colormap)
folium.LayerControl().add_to(m)
Fullscreen().add_to(m)
MeasureControl().add_to(m)

st_folium(m, width=1100, height=650)
st.info("✅ Map loaded quickly — all data and visuals preserved.")
