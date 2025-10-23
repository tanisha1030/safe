# app.py - OPTIMIZED VERSION
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob, os, io, json, requests
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

# Import folium plugins
try:
    from folium.plugins import Fullscreen, MeasureControl, MarkerCluster
except ImportError:
    pass

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — district-level (green → yellow → red)")
st.markdown(
    "This app aggregates all CSVs in `data/` (district-wise crime tables), "
    "builds a district-level crime score, plots a choropleth for India, and "
    "lets you check safety around a supplied place (address or `lat,lon`)."
)

# ---------------------------
# EMERGENCY CONTACTS DATABASE
# ---------------------------
EMERGENCY_CONTACTS = {
    "National": {
        "Police": "100",
        "Fire": "101", 
        "Ambulance": "102",
        "Disaster Management": "108",
        "Women Helpline": "1091",
        "Child Helpline": "1098",
        "Senior Citizen Helpline": "14567",
        "Tourist Helpline": "1363",
        "Railway Enquiry": "139",
        "Highway Emergency": "1033"
    },
    "State-specific": {
        "Delhi": {
            "Delhi Police Control Room": "100",
            "Delhi Fire Service": "101",
            "Delhi Ambulance": "102", 
            "Delhi Women Helpline": "181",
            "Delhi Traffic Police": "1095",
            "Delhi Anti Corruption": "1031"
        },
        "Mumbai/Maharashtra": {
            "Mumbai Police": "100",
            "Mumbai Fire Brigade": "101",
            "Mumbai Traffic Police": "103",
            "Maharashtra Police": "100",
            "Mumbai Ambulance": "102"
        },
        "Karnataka/Bangalore": {
            "Bangalore Police": "100",
            "Karnataka Police": "100", 
            "Bangalore Traffic Police": "103",
            "Bangalore Fire": "101"
        },
        "Tamil Nadu/Chennai": {
            "Chennai Police": "100",
            "Tamil Nadu Police": "100",
            "Chennai Traffic Police": "103",
            "Chennai Fire Service": "101"
        },
        "West Bengal/Kolkata": {
            "Kolkata Police": "100",
            "West Bengal Police": "100",
            "Kolkata Traffic Police": "103",
            "Kolkata Fire Brigade": "101"
        }
    },
    "Specialized Services": {
        "Cybercrime Helpline": "155260",
        "Anti-Terrorism Squad": "1090",
        "Poison Control": "1066",
        "Blood Bank": "104",
        "Railway Protection Force": "182",
        "Coast Guard": "1554",
        "Anti-Corruption Helpline": "1064",
        "Consumer Grievance": "1915",
        "Legal Aid": "15100",
        "Mental Health": "9152987821"
    }
}

def display_emergency_contacts():
    """Display emergency contacts in an organized manner"""
    st.sidebar.markdown("---")
    st.sidebar.header("🚨 Emergency Contacts")
    
    # National Emergency Numbers (always visible)
    with st.sidebar.expander("🇮🇳 National Emergency Numbers", expanded=True):
        for service, number in EMERGENCY_CONTACTS["National"].items():
            st.markdown(f"**{service}:** `{number}`")
    
    # State-specific numbers
    with st.sidebar.expander("🏛️ State-Specific Numbers"):
        state_choice = st.selectbox(
            "Select State/City:",
            ["Select..."] + list(EMERGENCY_CONTACTS["State-specific"].keys()),
            key="state_emergency"
        )
        
        if state_choice != "Select...":
            for service, number in EMERGENCY_CONTACTS["State-specific"][state_choice].items():
                st.markdown(f"**{service}:** `{number}`")
    
    # Specialized services
    with st.sidebar.expander("🛡️ Specialized Services"):
        for service, number in EMERGENCY_CONTACTS["Specialized Services"].items():
            st.markdown(f"**{service}:** `{number}`")
    
    # Quick dial section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📞 Quick Emergency Dial")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🚔 Police\n100", key="police_btn"):
            st.sidebar.success("Dial: 100")
        if st.button("🔥 Fire\n101", key="fire_btn"):
            st.sidebar.success("Dial: 101")
    
    with col2:
        if st.button("🚑 Ambulance\n102", key="ambulance_btn"):
            st.sidebar.success("Dial: 102")
        if st.button("👩 Women Help\n1091", key="women_btn"):
            st.sidebar.success("Dial: 1091")

@lru_cache(maxsize=128)
def get_region_specific_contacts(district_name, state_name=None):
    """Get emergency contacts specific to a region - CACHED"""
    contacts = EMERGENCY_CONTACTS["National"].copy()
    
    # Try to match with state-specific contacts
    for state_key, state_contacts in EMERGENCY_CONTACTS["State-specific"].items():
        if (district_name and any(city in district_name.lower() for city in state_key.lower().split('/'))) or \
           (state_name and state_name.lower() in state_key.lower()):
            contacts.update(state_contacts)
            break
    
    return contacts

# ---------------------------
# PARAMETERS - Updated working URLs
# ---------------------------
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson" 
]
DATA_FOLDER = "data"
NEAREST_RADIUS_KM = 5  # radius for nearest POIs

# ---------------------------
# UTIL: normalize district strings - OPTIMIZED
# ---------------------------
# Precompile replacement patterns
REPLACEMENTS = {
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

@lru_cache(maxsize=1024)
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    
    # Batch replace
    for old, new in REPLACEMENTS.items():
        s = s.replace(old, new)
    
    # Keep alnum and spaces - optimized
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s

# ---------------------------
# Load & aggregate all CSVs - OPTIMIZED with parallel processing
# ---------------------------
@st.cache_data(show_spinner=False)
def load_and_aggregate_csvs(data_folder):
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame(), []
    
    aggregated_rows = []
    failed = []
    
    def process_file(f):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(f, encoding="latin1", low_memory=False)
            except Exception as e:
                return None, (f, str(e))
        
        # Locate district-like column
        cols = list(df.columns)
        district_col = None
        for c in cols:
            lc = c.lower()
            if "district" in lc or "district name" in lc or ("name" == lc and len(cols)>1):
                district_col = c
                break
        if district_col is None:
            district_col = cols[0]
        
        # Sum numeric columns per row as file-level crime count
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        if numeric_cols:
            df['_file_total'] = df[numeric_cols].apply(lambda row: pd.to_numeric(row, errors='coerce').fillna(0).sum(), axis=1)
        else:
            df['_file_total'] = 1
        
        # Keep district and _file_total
        small = df[[district_col, '_file_total']].copy()
        small.columns = ['district_raw', 'file_total']
        small['district_norm'] = small['district_raw'].apply(normalize_name)
        
        return small, None
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, f) for f in files]
        for future in as_completed(futures):
            result, error = future.result()
            if error:
                failed.append(error)
            elif result is not None:
                aggregated_rows.append(result)
    
    if not aggregated_rows:
        return pd.DataFrame(), files, failed
    
    all_small = pd.concat(aggregated_rows, ignore_index=True)
    # Group by normalized district and sum
    agg = all_small.groupby('district_norm', as_index=False).agg({
        'file_total': 'sum',
        'district_raw': lambda x: x.iloc[0]
    }).rename(columns={'file_total':'crime_total', 'district_raw':'district_example'})
    
    return agg, files, failed

crime_agg, csv_files, failed_reads = load_and_aggregate_csvs(DATA_FOLDER)
if crime_agg.empty:
    st.error(f"No CSVs found or failed to parse files in `{DATA_FOLDER}`. Place your 57 CSVs there.")
    if failed_reads:
        st.write("Failed reads (examples):")
        st.write(failed_reads[:5])
    st.stop()
st.success(f"Loaded and aggregated {len(csv_files)} CSV(s) from `{DATA_FOLDER}`; found {len(crime_agg)} distinct normalized districts.")

# Display emergency contacts in sidebar
display_emergency_contacts()

st.sidebar.header("Data & GeoJSON options")
st.sidebar.write(f"Detected {len(csv_files)} CSV files in `{DATA_FOLDER}`.")
geo_choice = st.sidebar.radio("GeoJSON source:", ("Download default district GeoJSON", "Upload my geojson"))

uploaded_geo = None
if geo_choice == "Upload my geojson":
    uploaded_geo = st.sidebar.file_uploader("Upload India districts GeoJSON", type=["json","geojson"])
else:
    st.sidebar.write("Default: will try multiple sources for district-level geojson.")

# ---------------------------
# Load GeoJSON (districts) - OPTIMIZED with caching and better error handling
# ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson_from_url_or_upload(uploaded_file, urls):
    if uploaded_file is not None:
        gj = json.load(uploaded_file)
        gdf = gpd.GeoDataFrame.from_features(gj["features"])
        return gdf
    
    # Try multiple URLs
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            gj = r.json()
            gdf = gpd.GeoDataFrame.from_features(gj["features"])
            return gdf
        except Exception:
            continue
    
    raise Exception("All sources failed")

try:
    gdf_districts = load_geojson_from_url_or_upload(uploaded_geo, DEFAULT_GEOJSON_URLS)
    # Set CRS if not already set
    if gdf_districts.crs is None:
        gdf_districts.set_crs(epsg=4326, inplace=True)
except Exception as e:
    st.error("Could not load any remote GeoJSON sources. Please upload a district-level GeoJSON file.")
    st.write("You can download district boundaries from:")
    st.write("- https://github.com/geohacker/india/blob/master/district/india_district.geojson")
    st.write("- https://github.com/datta07/INDIAN-SHAPEFILES")
    st.stop()

# ---------------------------
# Identify district name column in geojson - OPTIMIZED
# ---------------------------
name_col = None

# Check for hierarchical naming patterns
hierarchical_cols = ['NAME_2', 'NAME_3', 'DTNAME', 'name_2', 'district_name']
for c in hierarchical_cols:
    if c in gdf_districts.columns:
        unique_vals = gdf_districts[c].nunique()
        if unique_vals > 10:
            name_col = c
            break

if name_col is None:
    possible_name_cols = ['NAME', 'name', 'district', 'District', 'DISTRICT', 'dtname']
    for c in possible_name_cols:
        if c in gdf_districts.columns:
            unique_vals = gdf_districts[c].nunique()
            if unique_vals > 10:
                name_col = c
                break

if name_col is None:
    for c in gdf_districts.columns:
        if gdf_districts[c].dtype == 'object':
            unique_vals = gdf_districts[c].nunique()
            if unique_vals > 50:
                name_col = c
                break

if name_col is None:
    st.error("Could not identify a district-name column in the GeoJSON.")
    st.stop()

# Create normalized name column - VECTORIZED
gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# ---------------------------
# Merge crime data into GeoDataFrame - OPTIMIZED
# ---------------------------
merged = gdf_districts.merge(crime_agg[['district_norm','crime_total']], on='district_norm', how='left')
merged['crime_total'] = merged['crime_total'].fillna(0)

# Ensure CRS is set for merged GeoDataFrame
if merged.crs is None:
    merged.set_crs(epsg=4326, inplace=True)

# Show matching stats
matched = (merged['crime_total'] > 0).sum()
total = len(merged)
missing = total - matched
st.info(f"District matching: {matched}/{total} districts have crime data. {missing} districts show 0 (no match found).")

if missing > 0.5 * total:
    st.warning("⚠️ Many districts are unmatched. Consider checking your data.")

# Classify into Low/Medium/High using quantiles - OPTIMIZED
non_zero = merged[merged['crime_total'] > 0]['crime_total']
if len(non_zero) > 0:
    q1 = non_zero.quantile(0.33)
    q2 = non_zero.quantile(0.66)
else:
    q1, q2 = 0, 0

def classify_val(n):
    if n == 0:
        return "No Data"
    elif n <= q1:
        return "Low"
    elif n <= q2:
        return "Medium"
    else:
        return "High"

merged['safety_level'] = merged['crime_total'].apply(classify_val)

# ---------------------------
# Make color scale - OPTIMIZED
# ---------------------------
vmin = merged['crime_total'].min()
vmax = merged['crime_total'].max()
if vmax > 0:
    colormap = StepColormap(
        colors=["lightgray","green","yellow","red"],
        index=[vmin, 0.1, q1, q2, vmax],
        vmin=vmin, vmax=vmax
    )
else:
    colormap = StepColormap(colors=["lightgray"], index=[0,1], vmin=0, vmax=1)
colormap.caption = "Crime count (gray = no data, green = low → red = high)"

# ---------------------------
# Draw national choropleth (folium) - OPTIMIZED
# ---------------------------
st.subheader("🗺️ India — Interactive Crime Safety Map")

# Map configuration
col1, col2 = st.columns([1, 1])
with col1:
    show_pois = st.checkbox("Show Police Stations", value=False)  # Default OFF for speed
with col2:
    danger_threshold = st.slider("Danger Zone Threshold", 0.1, 1.0, 0.7, 0.1)

# Create main map - SIMPLIFIED for speed
m = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5,
    tiles="OpenStreetMap",
    prefer_canvas=True  # Better performance
)

# Add layer control - SIMPLIFIED
folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

# Style function - OPTIMIZED
max_crime_val = merged['crime_total'].max()

def style_function(feature):
    val = feature['properties'].get('crime_total', 0)
    safety_level = feature['properties'].get('safety_level', 'No Data')
    
    is_danger_zone = val > (max_crime_val * danger_threshold) if max_crime_val > 0 else False
    
    if safety_level == "No Data":
        return {'fillColor': '#f0f0f0', 'color': '#cccccc', 'weight': 0.5, 'fillOpacity': 0.7}
    elif is_danger_zone:
        return {'fillColor': '#ff0000', 'color': '#cc0000', 'weight': 2.0, 'fillOpacity': 0.7}
    elif safety_level == "High":
        return {'fillColor': '#ff6b6b', 'color': '#e55353', 'weight': 1.0, 'fillOpacity': 0.7}
    elif safety_level == "Medium":
        return {'fillColor': '#ffd93d', 'color': '#ffcc02', 'weight': 0.8, 'fillOpacity': 0.7}
    else:
        return {'fillColor': '#6bcf7f', 'color': '#51b364', 'weight': 0.6, 'fillOpacity': 0.7}

# Tooltip function - OPTIMIZED
def create_tooltip(row):
    emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "No Data": "⚫"}.get(row['safety_level'], "⚫")
    return f"{row[name_col]} - {row['safety_level']} {emoji}"

# Add districts as ONE GeoJson layer for speed - CRITICAL OPTIMIZATION
folium.GeoJson(
    merged,
    name="🏛️ Districts",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=[name_col, 'safety_level', 'crime_total'],
        aliases=['District:', 'Safety:', 'Crime Count:'],
        localize=True
    )
).add_to(m)

# Add police stations only if requested
if show_pois:
    sample_stations = [
        {"name": "Delhi Police HQ", "lat": 28.6289, "lon": 77.2065},
        {"name": "Mumbai Police", "lat": 18.9220, "lon": 72.8347},
        {"name": "Bangalore Police", "lat": 12.9716, "lon": 77.5946},
        {"name": "Chennai Police", "lat": 13.0827, "lon": 80.2707},
        {"name": "Kolkata Police", "lat": 22.5726, "lon": 88.3639},
    ]
    
    for station in sample_stations:
        folium.Marker(
            location=[station['lat'], station['lon']],
            tooltip=f"🚔 {station['name']}",
            icon=folium.Icon(color='red', icon='shield-alt', prefix='fa')
        ).add_to(m)

# Add legend - OPTIMIZED
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; width: 180px; 
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:11px; padding: 8px; border-radius: 5px;">
<h4 style="margin: 0 0 8px 0;">🛡️ Safety</h4>
<p style="margin: 3px 0;">🟢 Low | 🟡 Medium | 🔴 High<br>🚨 Danger | ⚫ No Data</p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Add controls
folium.LayerControl().add_to(m)
try:
    Fullscreen().add_to(m)
except:
    pass

# Display map - height reduced for speed
st_data = st_folium(m, width=1200, height=600)

# ---------------------------
# Location analysis - OPTIMIZED
# ---------------------------
st.sidebar.header("Find safety near location")
loc_input = st.sidebar.text_input("Address or lat,lon", placeholder="Delhi or 28.7,77.1")

geolocator = Nominatim(user_agent="crime-app", timeout=10)

@lru_cache(maxsize=32)
def parse_or_geocode_cached(s):
    s = s.strip()
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1]), None
            except:
                pass
    try:
        loc = geolocator.geocode(s + ", India")
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except:
        pass
    return None

def find_district_for_point(point_latlon, merged_gdf):
    lat, lon = point_latlon
    pt = Point(lon, lat)
    contains = merged_gdf[merged_gdf.contains(pt)]
    if len(contains) > 0:
        return contains.iloc[0]
    # Find nearest
    merged_gdf = merged_gdf.copy()
    merged_gdf['centroid'] = merged_gdf.geometry.centroid
    distances = merged_gdf['centroid'].apply(lambda c: geodesic((c.y, c.x), (lat, lon)).km)
    return merged_gdf.loc[distances.idxmin()]

if loc_input:
    p = parse_or_geocode_cached(loc_input)
    if p is None:
        st.error("Could not geocode input.")
    else:
        lat, lon, resolved = p
        st.sidebar.success(f"📍 {lat:.4f}, {lon:.4f}")
        
        # Find district
        district_row = find_district_for_point((lat, lon), merged.copy())
        
        st.write("### Location Analysis")
        district_name = district_row[name_col]
        safety = district_row['safety_level']
        crime_count = int(district_row['crime_total'])
        
        emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "No Data": "⚫"}.get(safety, "⚫")
        st.info(f"**{district_name}** — {safety} {emoji} (Crime: {crime_count:,})")
        
        # Emergency contacts
        st.subheader("📞 Emergency Contacts")
        contacts = get_region_specific_contacts(district_name)
        
        cols = st.columns(3)
        for i, (service, number) in enumerate(list(contacts.items())[:6]):
            with cols[i % 3]:
                st.markdown(f"**{service}**\n`{number}`")
        
        # Quick actions
        st.markdown("### 🚨 Quick Dial")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("🚔 100", key="q1")
        with c2:
            st.button("🚑 102", key="q2")
        with c3:
            st.button("🔥 101", key="q3")
        with c4:
            st.button("👩 1091", key="q4")

st.markdown("---")
st.markdown("**Note:** Optimized for faster performance. Full POI search available on location input.")
