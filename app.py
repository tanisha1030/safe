# app.py
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
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
# EMERGENCY CONTACTS DATABASE (Optimized)
# ---------------------------
@st.cache_data
def get_emergency_contacts():
    return {
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

@lru_cache(maxsize=100)
def get_region_specific_contacts(district_name, state_name=None):
    """Cached function to get emergency contacts specific to a region"""
    EMERGENCY_CONTACTS = get_emergency_contacts()
    contacts = EMERGENCY_CONTACTS["National"].copy()
    
    if district_name:
        district_lower = district_name.lower()
        for state_key, state_contacts in EMERGENCY_CONTACTS["State-specific"].items():
            if any(city.lower() in district_lower for city in state_key.split('/')):
                contacts.update(state_contacts)
                break
    
    return contacts

def display_emergency_contacts():
    """Optimized emergency contacts display"""
    st.sidebar.markdown("---")
    st.sidebar.header("🚨 Emergency Contacts")
    
    EMERGENCY_CONTACTS = get_emergency_contacts()
    
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

# --------------------------- 
# OPTIMIZED PARAMETERS
# ---------------------------
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson" 
]
DATA_FOLDER = "data"
NEAREST_RADIUS_KM = 5

# ---------------------------
# OPTIMIZED UTIL FUNCTIONS
# ---------------------------
@lru_cache(maxsize=1000)
def normalize_name(s):
    """Cached normalization function"""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    
    # Optimized replacements
    replacements = {
        'commr': 'commissioner', 'commissionerate': 'commissioner', 
        'dist': 'district', 'north': 'n', 'south': 's', 
        'east': 'e', 'west': 'w', 'parganas': 'pargana',
        '24 pargana': 'twenty four pargana',
        'a and n': 'andaman nicobar', 'a & n': 'andaman nicobar',
        'city': '', 'rural': '',
    }
    
    for old, new in replacements.items():
        s = s.replace(old, new)
    
    # Vectorized cleaning
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s

# ---------------------------
# OPTIMIZED DATA LOADING
# ---------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_aggregate_csvs(data_folder):
    """Optimized CSV loading with parallel processing"""
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame(), [], []

    def process_file(f):
        try:
            # Try different encodings efficiently
            for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(f, low_memory=False, encoding=encoding, nrows=10000)  # Limit rows for speed
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, f, "Encoding error"
            
            # Quick column detection
            cols = df.columns.tolist()
            district_col = next((c for c in cols if 'district' in c.lower() or ('name' in c.lower() and len(cols) > 1)), cols[0])
            
            # Optimized numeric column detection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                # Quick coercion test on sample
                sample_size = min(100, len(df))
                for c in cols:
                    if c != district_col:
                        try:
                            pd.to_numeric(df[c].iloc[:sample_size].astype(str).str.replace(',',''), errors='raise')
                            numeric_cols.append(c)
                        except:
                            continue
            
            if numeric_cols:
                # Vectorized sum
                df['_file_total'] = df[numeric_cols].sum(axis=1, numeric_only=True)
            else:
                df['_file_total'] = 1
            
            # Return minimal data
            result = df[[district_col, '_file_total']].copy()
            result.columns = ['district_raw', 'file_total']
            result['district_norm'] = result['district_raw'].apply(normalize_name)
            return result, None, None
            
        except Exception as e:
            return None, f, str(e)
    
    # Process files in parallel (limit threads to avoid overwhelming system)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, files))
    
    aggregated_rows = []
    failed = []
    
    for result, failed_file, error in results:
        if result is not None:
            aggregated_rows.append(result)
        elif failed_file:
            failed.append((failed_file, error))
    
    if not aggregated_rows:
        return pd.DataFrame(), files, failed

    # Efficient concatenation and grouping
    all_small = pd.concat(aggregated_rows, ignore_index=True)
    agg = all_small.groupby('district_norm', as_index=False).agg({
        'file_total': 'sum',
        'district_raw': 'first'  # Changed from lambda for speed
    }).rename(columns={'file_total':'crime_total', 'district_raw':'district_example'})

    return agg, files, failed

# Load data with caching
crime_agg, csv_files, failed_reads = load_and_aggregate_csvs(DATA_FOLDER)
if crime_agg.empty:
    st.error(f"No CSVs found or failed to parse files in `{DATA_FOLDER}`. Place your 57 CSVs there.")
    if failed_reads:
        st.write("Failed reads (sample):")
        st.write(failed_reads[:3])  # Show fewer for speed
    st.stop()

st.success(f"Loaded {len(csv_files)} CSV(s); found {len(crime_agg)} districts.")

# Display emergency contacts
display_emergency_contacts()

st.sidebar.header("Data & GeoJSON options")
st.sidebar.write(f"Detected {len(csv_files)} CSV files.")
geo_choice = st.sidebar.radio("GeoJSON source:", ("Download default", "Upload"))

uploaded_geo = None
if geo_choice == "Upload":
    uploaded_geo = st.sidebar.file_uploader("Upload GeoJSON", type=["json","geojson"])

# ---------------------------
# OPTIMIZED GEOJSON LOADING
# ---------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_geojson_from_url_or_upload(uploaded_file, urls):
    """Optimized GeoJSON loading"""
    if uploaded_file is not None:
        gj = json.load(uploaded_file)
        return gpd.GeoDataFrame.from_features(gj["features"])
    
    # Try URLs with timeout and efficient parsing
    for url in urls:
        try:
            r = requests.get(url, timeout=20, stream=True)
            r.raise_for_status()
            gj = r.json()
            return gpd.GeoDataFrame.from_features(gj["features"])
        except:
            continue
    
    raise Exception("All sources failed")

try:
    gdf_districts = load_geojson_from_url_or_upload(uploaded_geo, DEFAULT_GEOJSON_URLS)
except Exception:
    st.error("Could not load GeoJSON. Please upload a district-level GeoJSON file.")
    st.stop()

# ---------------------------
# OPTIMIZED DISTRICT NAME DETECTION
# ---------------------------
# Quick column detection
hierarchical_cols = ['NAME_2', 'NAME_3', 'DTNAME', 'name_2', 'district_name']
name_col = None

for c in hierarchical_cols:
    if c in gdf_districts.columns and gdf_districts[c].nunique() > 10:
        name_col = c
        break

if name_col is None:
    possible_cols = ['NAME', 'name', 'district', 'District', 'DISTRICT']
    for c in possible_cols:
        if c in gdf_districts.columns and gdf_districts[c].nunique() > 10:
            name_col = c
            break

if name_col is None:
    # Find best column efficiently
    for c in gdf_districts.columns:
        if gdf_districts[c].dtype == 'object' and gdf_districts[c].nunique() > 50:
            name_col = c
            break

if name_col is None:
    st.error("Could not identify district column.")
    st.stop()

# Vectorized normalization
gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# ---------------------------
# OPTIMIZED DATA MERGING
# ---------------------------
merged = gdf_districts.merge(
    crime_agg[['district_norm','crime_total']], 
    on='district_norm', 
    how='left'
).fillna({'crime_total': 0})

# Quick stats
matched = (merged['crime_total'] > 0).sum()
total = len(merged)
st.info(f"Matched: {matched}/{total} districts")

# Efficient quantile calculation
non_zero_crimes = merged.loc[merged['crime_total'] > 0, 'crime_total']
if len(non_zero_crimes) > 0:
    q1, q2 = non_zero_crimes.quantile([0.33, 0.66])
else:
    q1, q2 = 0, 0

# Vectorized classification
def classify_crime_level(crime_counts):
    """Vectorized classification function"""
    conditions = [
        crime_counts == 0,
        crime_counts <= q1,
        crime_counts <= q2
    ]
    choices = ["No Data", "Low", "Medium"]
    return np.select(conditions, choices, default="High")

merged['safety_level'] = classify_crime_level(merged['crime_total'])

# ---------------------------
# OPTIMIZED MAP CREATION
# ---------------------------
vmin, vmax = merged['crime_total'].min(), merged['crime_total'].max()
if vmax > 0:
    colormap = StepColormap(
        colors=["lightgray","green","yellow","red"],
        index=[vmin, 0.1, q1, q2, vmax],
        vmin=vmin, vmax=vmax
    )
else:
    colormap = StepColormap(colors=["lightgray"], index=[0,1], vmin=0, vmax=1)

st.subheader("🗺️ India — Interactive Crime Safety Map")

# Simplified controls
col1, col2 = st.columns(2)
with col1:
    show_pois = st.checkbox("Show Police Stations", value=False)  # Default false for speed
with col2:
    danger_threshold = st.slider("Danger Threshold", 0.1, 1.0, 0.7, 0.1)

# Create optimized map
m = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5,
    tiles="OpenStreetMap",
    prefer_canvas=True  # Better performance
)

# Add minimal tile layers
folium.TileLayer('OpenStreetMap', name='Street').add_to(m)
folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)

# Optimized style function
max_crime = merged['crime_total'].max()
danger_cutoff = max_crime * danger_threshold if max_crime > 0 else 0

def get_style(crime_total, safety_level):
    """Pre-computed styling for better performance"""
    if safety_level == "No Data":
        return {'fillColor': '#f0f0f0', 'color': '#ccc', 'weight': 0.5, 'fillOpacity': 0.6}
    elif crime_total > danger_cutoff:
        return {'fillColor': '#ff0000', 'color': '#cc0000', 'weight': 2, 'fillOpacity': 0.7}
    elif safety_level == "High":
        return {'fillColor': '#ff6b6b', 'color': '#e55353', 'weight': 1, 'fillOpacity': 0.7}
    elif safety_level == "Medium":
        return {'fillColor': '#ffd93d', 'color': '#ffcc02', 'weight': 0.8, 'fillOpacity': 0.7}
    else:
        return {'fillColor': '#6bcf7f', 'color': '#51b364', 'weight': 0.6, 'fillOpacity': 0.7}

# Add districts with optimized rendering
district_layer = folium.FeatureGroup(name="Districts")

# Batch process districts for better performance
for _, row in merged.iterrows():
    district_name = row[name_col]
    crime_count = int(row['crime_total'])
    safety = row['safety_level']
    
    # Simplified popup
    popup_html = f"""
    <div style="font-family: Arial; max-width: 250px;">
        <h4>{district_name}</h4>
        <p><b>Safety:</b> {safety}<br>
        <b>Crime Count:</b> {crime_count:,}</p>
        <p><b>Emergency:</b> Police 100 | Fire 101 | Medical 102</p>
    </div>
    """
    
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, style=get_style(row['crime_total'], row['safety_level']): style,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{district_name} - {safety}"
    ).add_to(district_layer)

district_layer.add_to(m)

# Enhanced police stations with district crime data
if show_pois:
    police_layer = folium.FeatureGroup(name="Police")
    key_stations = [
        {"name": "Delhi Police HQ", "lat": 28.6289, "lon": 77.2065, "city": "Delhi"},
        {"name": "Mumbai Police Commissioner", "lat": 18.9220, "lon": 72.8347, "city": "Mumbai"},
        {"name": "Bangalore City Police", "lat": 12.9716, "lon": 77.5946, "city": "Bangalore"},
        {"name": "Chennai Police Station", "lat": 13.0827, "lon": 80.2707, "city": "Chennai"},
        {"name": "Kolkata Police HQ", "lat": 22.5726, "lon": 88.3639, "city": "Kolkata"},
    ]
    
    for station in key_stations:
        # Find the district this police station is located in
        station_point = Point(station['lon'], station['lat'])
        containing_district = merged[merged.geometry.contains(station_point)]
        
        if len(containing_district) > 0:
            district_row = containing_district.iloc[0]
            district_name = district_row[name_col]
            safety_level = district_row['safety_level']
            crime_count = int(district_row['crime_total'])
            
            # Get emergency contacts for this district
            region_contacts = get_region_specific_contacts(district_name)
            emergency_info = ""
            for service, number in list(region_contacts.items())[:4]:
                emergency_info += f"<tr><td>{service}:</td><td><b>{number}</b></td></tr>"
            
            # Determine danger status
            is_danger = crime_count > danger_cutoff if danger_cutoff > 0 else False
            danger_text = "🚨 HIGH ALERT AREA" if is_danger else ""
            
            # Safety color coding
            safety_colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545", "No Data": "#6c757d"}
            safety_color = safety_colors.get(safety_level, "#6c757d")
            
            popup_html = f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4 style="margin: 0 0 8px 0; color: #2c3e50;">🚔 {station['name']}</h4>
                <hr style="margin: 5px 0;">
                
                <h5 style="margin: 5px 0; color: {safety_color};">District: {district_name}</h5>
                <table style="width: 100%; font-size: 12px; margin-bottom: 10px;">
                    <tr><td><b>Safety Level:</b></td><td style="color: {safety_color};"><b>{safety_level}</b></td></tr>
                    <tr><td><b>Crime Count:</b></td><td><b>{crime_count:,}</b></td></tr>
                    <tr><td><b>Risk Status:</b></td><td>{'High Risk Zone' if is_danger else 'Moderate/Low Risk'}</td></tr>
                </table>
                
                {f'<div style="background: #ffebee; border: 1px solid #f44336; padding: 5px; border-radius: 3px; margin: 5px 0;"><b style="color: #d32f2f;">{danger_text}</b></div>' if is_danger else ''}
                
                <hr style="margin: 8px 0;">
                <h6 style="margin: 5px 0; color: #e74c3c;">📞 Emergency Contacts:</h6>
                <table style="width: 100%; font-size: 11px;">
                    {emergency_info}
                </table>
                
                <hr style="margin: 8px 0;">
                <p style="margin: 5px 0; font-size: 10px; color: #666;">
                    <b>Location:</b> {station['lat']:.4f}, {station['lon']:.4f}<br>
                    <b>Coverage Area:</b> {station['city']} and surrounding areas
                </p>
            </div>
            """
            
            # Color code the police station marker based on district safety
            marker_color = 'red' if safety_level == 'High' or is_danger else ('orange' if safety_level == 'Medium' else 'green')
            
            tooltip_text = f"🚔 {station['name']} - {district_name} ({safety_level})"
            if is_danger:
                tooltip_text += " ⚠️ HIGH RISK"
                
        else:
            # Fallback for stations not in any district
            popup_html = f"""
            <div style="font-family: Arial; max-width: 250px;">
                <h4 style="margin: 0; color: #2c3e50;">🚔 {station['name']}</h4>
                <hr style="margin: 5px 0;">
                <p><b>Location:</b> {station['city']}<br>
                <b>Coordinates:</b> {station['lat']:.4f}, {station['lon']:.4f}</p>
                
                <hr style="margin: 8px 0;">
                <h6 style="color: #e74c3c;">📞 Emergency Numbers:</h6>
                <p style="font-size: 12px;">
                Police: <b>100</b><br>
                Fire: <b>101</b><br>
                Medical: <b>102</b><br>
                Women Help: <b>1091</b>
                </p>
                
                <p style="font-size: 10px; color: #666;">
                District crime data not available for this location
                </p>
            </div>
            """
            marker_color = 'blue'  # Default color
            tooltip_text = f"🚔 {station['name']} - {station['city']}"
        
        folium.Marker(
            [station['lat'], station['lon']],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip_text,
            icon=folium.Icon(
                color=marker_color, 
                icon='shield-alt' if marker_color == 'red' else 'info-sign',
                prefix='fa'
            )
        ).add_to(police_layer)
    
    police_layer.add_to(m)

# Simplified legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 120px; 
     background: white; border: 2px solid grey; z-index: 9999; padding: 10px;">
<h4>Safety Legend</h4>
<p>🟢 Low Risk</p>
<p>🟡 Medium Risk</p>
<p>🔴 High Risk</p>
<p>🚨 Danger Zone</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)

# Display map
st_data = st_folium(m, width=1200, height=600, returned_objects=["last_clicked"])

# ---------------------------
# OPTIMIZED LOCATION ANALYSIS
# ---------------------------
st.sidebar.header("Location Analysis")
loc_input = st.sidebar.text_input("Enter location", placeholder="Address or lat,lon")

# Cached geocoder
@st.cache_data(ttl=3600)
def cached_geocode(location_string):
    """Cached geocoding function"""
    geolocator = Nominatim(user_agent="crime-app", timeout=5)
    try:
        if ',' in location_string and len(location_string.split(',')) == 2:
            parts = location_string.split(',')
            try:
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
                return lat, lon, f"{lat:.4f}, {lon:.4f}"
            except ValueError:
                pass
        
        location = geolocator.geocode(location_string + ", India")
        if location:
            return location.latitude, location.longitude, location.address
    except Exception:
        pass
    return None

if loc_input:
    result = cached_geocode(loc_input)
    if result:
        lat, lon, address = result
        st.sidebar.success(f"Found: {address}")
        
        # Quick district lookup
        point = Point(lon, lat)
        containing = merged[merged.geometry.contains(point)]
        
        if len(containing) > 0:
            district_row = containing.iloc[0]
            district_name = district_row[name_col]
            safety = district_row['safety_level']
            crime_count = int(district_row['crime_total'])
            
            st.write("### Location Analysis")
            
            color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "No Data": "⚫"}
            st.success(f"**{district_name}** - {safety} {color_map.get(safety, '⚫')} (Crimes: {crime_count:,})")
            
            # Emergency contacts for region
            contacts = get_region_specific_contacts(district_name)
            
            st.subheader("Emergency Contacts")
            contact_cols = st.columns(3)
            
            essential_contacts = [
                ("Police", contacts.get("Police", "100")),
                ("Fire", contacts.get("Fire", "101")),
                ("Ambulance", contacts.get("Ambulance", "102")),
                ("Women Helpline", contacts.get("Women Helpline", "1091")),
                ("Tourist Helpline", contacts.get("Tourist Helpline", "1363")),
                ("Disaster Management", contacts.get("Disaster Management", "108"))
            ]
            
            for i, (service, number) in enumerate(essential_contacts):
                with contact_cols[i % 3]:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 2px;">
                        <strong>{service}</strong><br>
                        <h4 style="margin: 0; color: #dc3545;">{number}</h4>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Simplified local map
            st.subheader("Local Area Map")
            
            local_map = folium.Map(
                location=[lat, lon],
                zoom_start=12,
                tiles="OpenStreetMap",
                prefer_canvas=True
            )
            
            # Add user location
            folium.CircleMarker(
                [lat, lon],
                radius=10,
                color="blue",
                fill=True,
                popup=f"Your Location<br>Emergency: 100",
                tooltip="You are here"
            ).add_to(local_map)
            
            # Add nearby districts (simplified)
            merged_copy = merged.copy()
            merged_copy['centroid'] = merged_copy.geometry.centroid
            merged_copy['distance'] = merged_copy['centroid'].apply(
                lambda c: ((c.y - lat)**2 + (c.x - lon)**2)**0.5  # Simplified distance
            )
            
            nearby = merged_copy.nsmallest(5, 'distance')  # Top 5 nearest
            
            for _, row in nearby.iterrows():
                style = get_style(row['crime_total'], row['safety_level'])
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, s=style: s,
                    tooltip=f"{row[name_col]} - {row['safety_level']}"
                ).add_to(local_map)
            
            st_folium(local_map, width=800, height=400)
            
        else:
            st.warning("Location not found in district boundaries")
    else:
        st.sidebar.error("Could not geocode location")

# Simplified safety tips
st.markdown("---")
st.subheader("Safety Tips")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Emergency Numbers:**
    - Police: **100**
    - Fire: **101**
    - Medical: **102**
    - Women Helpline: **1091**
    """)

with col2:
    st.markdown("""
    **Safety Tips:**
    - Stay alert in unfamiliar areas
    - Keep emergency contacts handy
    - Share location with trusted contacts
    - Trust your instincts
    """)

st.markdown("---")
st.markdown("""
**Performance Notes:** 
Data processing optimized for speed. Emergency contacts cached for quick access.
For immediate danger, call emergency services directly.
""")
