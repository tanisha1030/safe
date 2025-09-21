# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob, os, json, requests
from shapely.geometry import Point
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

# Folium plugins
try:
    from folium.plugins import Fullscreen, MeasureControl, MarkerCluster
except ImportError:
    pass

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — district-level (green → yellow → red)")
st.markdown(
    "This app aggregates all CSVs in `data/` (district-wise crime tables), "
    "builds a district-level crime score, plots a choropleth for India, and "
    "lets you check safety around a supplied place (address or `lat,lon`)."
)

DATA_FOLDER = "data"
NEAREST_RADIUS_KM = 5
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
]

# ---------------------------
# UTILS
# ---------------------------
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    replacements = {
        'commr': 'commissioner',
        'commissionerate': 'commissioner', 
        'dist': 'district',
        'north': 'n',
        'south': 's', 
        'east': 'e',
        'west': 'w',
        'parganas': 'pargana',
        '24 pargana': 'twenty four pargana',
        'a and n': 'andaman nicobar',
        'a & n': 'andaman nicobar',
        'city': '',
        'rural': '',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s

# ---------------------------
# LOAD AND AGGREGATE CSVs
# ---------------------------
@st.cache_data(show_spinner=False)
def load_and_aggregate_csvs(data_folder):
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame(), []
    aggregated_rows = []
    failed = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except:
            try:
                df = pd.read_csv(f, encoding="latin1", low_memory=False)
            except Exception as e:
                failed.append((f, str(e)))
                continue
        # Find district column
        cols = df.columns
        district_col = None
        for c in cols:
            lc = c.lower()
            if "district" in lc or "district name" in lc or ("name" == lc and len(cols)>1):
                district_col = c
                break
        if district_col is None:
            district_col = cols[0]
        # sum numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df['_file_total'] = df[numeric_cols].apply(lambda row: pd.to_numeric(row, errors='coerce').fillna(0).sum(), axis=1)
        else:
            df['_file_total'] = 1
        small = df[[district_col, '_file_total']].copy()
        small.columns = ['district_raw', 'file_total']
        small['district_norm'] = small['district_raw'].apply(normalize_name)
        aggregated_rows.append(small)
    if not aggregated_rows:
        return pd.DataFrame(), files, failed
    all_small = pd.concat(aggregated_rows, ignore_index=True)
    agg = all_small.groupby('district_norm', as_index=False).agg({
        'file_total': 'sum',
        'district_raw': lambda x: x.iloc[0]
    }).rename(columns={'file_total':'crime_total','district_raw':'district_example'})
    return agg, files, failed

crime_agg, csv_files, failed_reads = load_and_aggregate_csvs(DATA_FOLDER)
if crime_agg.empty:
    st.error(f"No CSVs found in `{DATA_FOLDER}` or failed to parse files.")
    if failed_reads:
        st.write("Failed reads (examples):", failed_reads[:5])
    st.stop()
st.success(f"Loaded and aggregated {len(csv_files)} CSV(s); found {len(crime_agg)} normalized districts.")

# ---------------------------
# GEOJSON LOADING
# ---------------------------
st.sidebar.header("Data & GeoJSON options")
geo_choice = st.sidebar.radio("GeoJSON source:", ("Download default district GeoJSON", "Upload my geojson"))
uploaded_geo = None
if geo_choice == "Upload my geojson":
    uploaded_geo = st.sidebar.file_uploader("Upload India districts GeoJSON", type=["json","geojson"])

@st.cache_data(show_spinner=False)
def load_geojson_from_url_or_upload(uploaded_file, urls):
    if uploaded_file is not None:
        gj = json.load(uploaded_file)
        gdf = gpd.GeoDataFrame.from_features(gj["features"])
        return gdf
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            gj = r.json()
            gdf = gpd.GeoDataFrame.from_features(gj["features"])
            return gdf
        except:
            continue
    raise Exception("All GeoJSON sources failed.")

try:
    gdf_districts = load_geojson_from_url_or_upload(uploaded_geo, DEFAULT_GEOJSON_URLS)
except Exception as e:
    st.error("Could not load GeoJSON. Upload a district-level geojson.")
    st.stop()

# Identify district name column
name_col = None
for c in ['NAME_2','NAME_3','DTNAME','name_2','district_name','NAME','name','district','DISTRICT']:
    if c in gdf_districts.columns and gdf_districts[c].nunique() > 10:
        name_col = c
        break
if name_col is None:
    st.error("Could not identify a district-name column.")
    st.stop()

gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# Merge crime data
merged = gdf_districts.merge(crime_agg[['district_norm','crime_total']], on='district_norm', how='left')
merged['crime_total'] = merged['crime_total'].fillna(0)

# classify Low/Medium/High
non_zero = merged[merged['crime_total']>0]['crime_total']
q1,q2 = non_zero.quantile(0.33), non_zero.quantile(0.66) if len(non_zero)>0 else (0,0)
def classify_val(n):
    if n==0: return "No Data"
    elif n<=q1: return "Low"
    elif n<=q2: return "Medium"
    else: return "High"
merged['safety_level'] = merged['crime_total'].apply(classify_val)

# ---------------------------
# NATIONAL CHOROPLETH
# ---------------------------
st.subheader("🗺️ India — Interactive Crime Safety Map")
map_style = st.selectbox("Map Style:", ["Street Map","Satellite","Terrain","Dark Mode"], index=0)
show_pois = st.checkbox("Show Police Stations", value=True)
danger_threshold = st.slider("Danger Zone Threshold",0.1,1.0,0.7,0.1)

tile_configs = {
    "Street Map":"OpenStreetMap",
    "Satellite":"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain":"https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
    "Dark Mode":"CartoDB dark_matter"
}
tile_attr = {"Street Map":"OpenStreetMap","Satellite":"Esri","Terrain":"Esri","Dark Mode":"CartoDB"}

m = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles=tile_configs[map_style], attr=tile_attr[map_style])

def style_function(feature):
    val = feature['properties'].get('crime_total',0)
    safety_level = feature['properties'].get('safety_level','No Data')
    max_crime = merged['crime_total'].max()
    is_danger_zone = val > (max_crime*danger_threshold) if max_crime>0 else False
    if safety_level=="No Data":
        color_fill='#f0f0f0'; border_color='#cccccc'; border_weight=0.5
    elif is_danger_zone:
        color_fill='#ff0000'; border_color='#990000'; border_weight=1.5
    elif safety_level=="High":
        color_fill='#ff8000'; border_color='#cc6600'; border_weight=1
    elif safety_level=="Medium":
        color_fill='#ffff00'; border_color='#cccc00'; border_weight=1
    else: # Low
        color_fill='#00ff00'; border_color='#009900'; border_weight=1
    return {
        'fillColor': color_fill,
        'color': border_color,
        'weight': border_weight,
        'fillOpacity':0.5
    }

folium.GeoJson(
    merged,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=[name_col,'crime_total','safety_level'],
        aliases=["District","Crime Score","Level"],
        localize=True
    )
).add_to(m)

# Folium plugins
Fullscreen().add_to(m)
MeasureControl(primary_length_unit='kilometers').add_to(m)

# ---------------------------
# SHOW MAP
# ---------------------------
st_data = st_folium(m, width=1000, height=600)
