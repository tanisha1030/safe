# app.py - Optimized for Speed
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob
import os
import json
import requests
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

# Import folium plugins with error handling
try:
    from folium.plugins import Fullscreen, MarkerCluster
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

st.set_page_config(page_title="India Crime Heatmap", layout="wide")

# Cache static content
@st.cache_data
def get_app_header():
    return {
        'title': "India Crime Heatmap — District-level Safety Analysis",
        'description': "This app aggregates CSV files containing district-wise crime data, creates a safety score visualization, and provides location-based safety analysis."
    }

header = get_app_header()
st.title(header['title'])
st.markdown(header['description'])

# Configuration - moved to top for early initialization
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
]
DATA_FOLDER = "data"

# Optimized normalize function with precompiled replacements
@st.cache_data
def get_normalization_rules():
    return {
        'commr': 'commissioner', 'commissionerate': 'commissioner',
        'dist': 'district', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
        'parganas': 'pargana', '24 pargana': 'twenty four pargana',
        'a and n': 'andaman nicobar', 'a & n': 'andaman nicobar',
        'city': '', 'rural': '',
    }

REPLACEMENTS = get_normalization_rules()

def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    for old, new in REPLACEMENTS.items():
        s = s.replace(old, new)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())

# Optimized CSV loading with better caching
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_aggregate_csvs(data_folder):
    if not os.path.exists(data_folder):
        return pd.DataFrame(), [], [("Folder not found", f"Directory '{data_folder}' does not exist")]
    
    csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not csv_files:
        return pd.DataFrame(), [], [("No files", "No CSV files found in data folder")]
    
    all_data = []
    failed = []
    
    for file_path in csv_files:
        try:
            # Try reading with different encodings efficiently
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                failed.append((os.path.basename(file_path), "Encoding error"))
                continue
            
            if df.empty:
                continue
            
            # Fast column detection
            district_col = next((col for col in df.columns 
                               if any(keyword in col.lower() for keyword in ['district', 'area', 'region'])), 
                              df.columns[0])
            
            # Vectorized numeric conversion
            numeric_cols = []
            for col in df.columns:
                if col != district_col:
                    try:
                        temp_series = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                        if not temp_series.isna().all():
                            numeric_cols.append(col)
                    except:
                        continue
            
            if numeric_cols:
                # Vectorized operations
                df_numeric = df[numeric_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce')).fillna(0)
                df['crime_total'] = df_numeric.sum(axis=1)
            else:
                df['crime_total'] = 1
            
            # Keep only necessary columns
            summary_df = df[[district_col, 'crime_total']].copy()
            summary_df.columns = ['district_raw', 'crime_total']
            summary_df = summary_df[summary_df['district_raw'].notna()]
            
            if not summary_df.empty:
                all_data.append(summary_df)
                
        except Exception as e:
            failed.append((os.path.basename(file_path), str(e)))
    
    if not all_data:
        return pd.DataFrame(), csv_files, failed
    
    # Efficient concatenation and grouping
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['district_norm'] = combined_df['district_raw'].apply(normalize_name)
    
    final_agg = combined_df.groupby('district_norm', as_index=False).agg({
        'crime_total': 'sum',
        'district_raw': 'first'
    })
    
    return final_agg, csv_files, failed

# Load data with progress indicator
with st.spinner("Loading crime data..."):
    crime_data, csv_files, failed_files = load_and_aggregate_csvs(DATA_FOLDER)

if crime_data.empty:
    st.error(f"No valid crime data found in '{DATA_FOLDER}' folder.")
    if failed_files:
        with st.expander("Failed files"):
            for filename, error in failed_files[:5]:
                st.write(f"- {filename}: {error}")
    st.stop()

st.success(f"Loaded {len(csv_files)} CSV files with data for {len(crime_data)} districts.")

# Sidebar - optimized with session state
if 'geo_source' not in st.session_state:
    st.session_state.geo_source = "Use default India districts"

st.sidebar.header("Emergency Numbers")
st.sidebar.markdown("""
**Quick Access:**
- **Police**: 100 | **Fire**: 101 | **Ambulance**: 102
- **Women Safety**: 1091 | **Child Helpline**: 1098
""")
st.sidebar.error("In emergency, call 100!")
st.sidebar.markdown("---")

st.sidebar.header("Configuration")
geo_source = st.sidebar.radio("GeoJSON Source:", 
    ("Use default India districts", "Upload custom GeoJSON"),
    key='geo_source')

uploaded_geojson = None
if geo_source == "Upload custom GeoJSON":
    uploaded_geojson = st.sidebar.file_uploader("Upload GeoJSON", type=["json", "geojson"])

# Optimized GeoJSON loading with better caching
@st.cache_data(show_spinner=False, ttl=3600)
def load_geojson(uploaded_data, default_urls):
    if uploaded_data is not None:
        geojson_data = json.load(uploaded_data)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
        return gdf, "uploaded file"
    
    for i, url in enumerate(default_urls):
        try:
            response = requests.get(url, timeout=15)  # Reduced timeout
            response.raise_for_status()
            geojson_data = response.json()
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            return gdf, f"URL {i+1}"
        except:
            continue
    
    raise Exception("All GeoJSON sources failed")

# Load districts
try:
    with st.spinner("Loading district boundaries..."):
        districts_gdf, source_used = load_geojson(uploaded_geojson, DEFAULT_GEOJSON_URLS)
except Exception:
    st.error("Failed to load district boundaries. Please upload a GeoJSON file.")
    st.stop()

# Fast district column detection
@st.cache_data
def find_district_column(columns):
    priority_cols = ['NAME_2', 'NAME_3', 'DTNAME', 'name_2', 'district_name', 'NAME', 'name', 'district']
    for col in priority_cols:
        if col in columns:
            return col
    return next((col for col in columns if col.lower() in ['name', 'district']), columns[0])

district_name_col = find_district_column(list(districts_gdf.columns))
districts_gdf['district_norm'] = districts_gdf[district_name_col].apply(normalize_name)

# Efficient merge and classification
merged_data = districts_gdf.merge(crime_data[['district_norm', 'crime_total']], 
                                 on='district_norm', how='left')
merged_data['crime_total'] = merged_data['crime_total'].fillna(0)

# Fast quantile calculation
non_zero_crimes = merged_data[merged_data['crime_total'] > 0]['crime_total']
if len(non_zero_crimes) > 0:
    q33, q66 = non_zero_crimes.quantile([0.33, 0.66])
else:
    q33, q66 = 0, 0

# Vectorized safety level assignment
def get_safety_level_vectorized(crime_counts):
    return pd.cut(crime_counts, 
                 bins=[-0.1, 0, q33, q66, float('inf')], 
                 labels=['No Data', 'Low', 'Medium', 'High'])

merged_data['safety_level'] = get_safety_level_vectorized(merged_data['crime_total'])

# Color mapping
COLOR_MAP = {"No Data": "#f0f0f0", "Low": "#6bcf7f", "Medium": "#ffd93d", "High": "#ff6b6b"}

# Police Station Data - Add major cities and their police stations
@st.cache_data
def get_police_stations_data():
    """Get police station data for major cities in India"""
    police_stations = [
        # Delhi
        {"name": "Delhi Police HQ", "lat": 28.6289, "lon": 77.2065, "city": "Delhi", "district": "New Delhi", "phone": "011-23490085"},
        {"name": "Connaught Place PS", "lat": 28.6315, "lon": 77.2167, "city": "Delhi", "district": "New Delhi", "phone": "011-23417434"},
        {"name": "Karol Bagh PS", "lat": 28.6519, "lon": 77.1909, "city": "Delhi", "district": "New Delhi", "phone": "011-25752346"},
        
        # Mumbai  
        {"name": "Mumbai Police HQ", "lat": 18.9220, "lon": 72.8347, "city": "Mumbai", "district": "Mumbai", "phone": "022-22633333"},
        {"name": "Colaba PS", "lat": 18.9067, "lon": 72.8147, "city": "Mumbai", "district": "Mumbai", "phone": "022-22672444"},
        {"name": "Bandra PS", "lat": 19.0596, "lon": 72.8295, "city": "Mumbai", "district": "Mumbai", "phone": "022-26420440"},
        
        # Bangalore
        {"name": "Bangalore City Police", "lat": 12.9716, "lon": 77.5946, "city": "Bangalore", "district": "Bangalore Urban", "phone": "080-22208446"},
        {"name": "Cubbon Park PS", "lat": 12.9698, "lon": 77.5802, "city": "Bangalore", "district": "Bangalore Urban", "phone": "080-22867332"},
        {"name": "Koramangala PS", "lat": 12.9279, "lon": 77.6271, "city": "Bangalore", "district": "Bangalore Urban", "phone": "080-25537601"},
        
        # Chennai
        {"name": "Chennai Police HQ", "lat": 13.0827, "lon": 80.2707, "city": "Chennai", "district": "Chennai", "phone": "044-23452348"},
        {"name": "T Nagar PS", "lat": 13.0418, "lon": 80.2341, "city": "Chennai", "district": "Chennai", "phone": "044-24330440"},
        {"name": "Marina PS", "lat": 13.0475, "lon": 80.2843, "city": "Chennai", "district": "Chennai", "phone": "044-25361425"},
        
        # Kolkata
        {"name": "Kolkata Police HQ", "lat": 22.5726, "lon": 88.3639, "city": "Kolkata", "district": "Kolkata", "phone": "033-22143526"},
        {"name": "Park Street PS", "lat": 22.5535, "lon": 88.3507, "city": "Kolkata", "district": "Kolkata", "phone": "033-22299514"},
        {"name": "New Market PS", "lat": 22.5564, "lon": 88.3501, "city": "Kolkata", "district": "Kolkata", "phone": "033-22127506"},
        
        # Hyderabad
        {"name": "Hyderabad Police HQ", "lat": 17.3850, "lon": 78.4867, "city": "Hyderabad", "district": "Hyderabad", "phone": "040-27853508"},
        {"name": "Abids PS", "lat": 17.3847, "lon": 78.4735, "city": "Hyderabad", "district": "Hyderabad", "phone": "040-24608341"},
        {"name": "Banjara Hills PS", "lat": 17.4239, "lon": 78.4738, "city": "Hyderabad", "district": "Hyderabad", "phone": "040-23354891"},
        
        # Pune
        {"name": "Pune Police HQ", "lat": 18.5204, "lon": 73.8567, "city": "Pune", "district": "Pune", "phone": "020-26128570"},
        {"name": "Koregaon Park PS", "lat": 18.5362, "lon": 73.8977, "city": "Pune", "district": "Pune", "phone": "020-26139503"},
        {"name": "Shivaji Nagar PS", "lat": 18.5304, "lon": 73.8424, "city": "Pune", "district": "Pune", "phone": "020-25534982"},
        
        # Ahmedabad
        {"name": "Ahmedabad Police HQ", "lat": 23.0225, "lon": 72.5714, "city": "Ahmedabad", "district": "Ahmedabad", "phone": "079-25506444"},
        {"name": "Ellis Bridge PS", "lat": 23.0395, "lon": 72.5610, "city": "Ahmedabad", "district": "Ahmedabad", "phone": "079-26579424"},
        {"name": "Navrangpura PS", "lat": 23.0395, "lon": 72.5439, "city": "Ahmedabad", "district": "Ahmedabad", "phone": "079-26301318"},
        
        # Jaipur
        {"name": "Jaipur Police HQ", "lat": 26.9124, "lon": 75.7873, "city": "Jaipur", "district": "Jaipur", "phone": "0141-2743000"},
        {"name": "MI Road PS", "lat": 26.9157, "lon": 75.8103, "city": "Jaipur", "district": "Jaipur", "phone": "0141-2374832"},
        
        # Lucknow
        {"name": "Lucknow Police HQ", "lat": 26.8467, "lon": 80.9462, "city": "Lucknow", "district": "Lucknow", "phone": "0522-2624015"},
        {"name": "Hazratganj PS", "lat": 26.8489, "lon": 80.9319, "city": "Lucknow", "district": "Lucknow", "phone": "0522-2623456"},
        
        # Additional state capitals
        {"name": "Gandhinagar Police HQ", "lat": 23.2156, "lon": 72.6369, "city": "Gandhinagar", "district": "Gandhinagar", "phone": "079-23977007"},
        {"name": "Bhopal Police HQ", "lat": 23.2599, "lon": 77.4126, "city": "Bhopal", "district": "Bhopal", "phone": "0755-2778100"},
        {"name": "Chandigarh Police HQ", "lat": 30.7333, "lon": 76.7794, "city": "Chandigarh", "district": "Chandigarh", "phone": "0172-2740100"},
        {"name": "Thiruvananthapuram PS", "lat": 8.5241, "lon": 76.9366, "city": "Thiruvananthapuram", "district": "Thiruvananthapuram", "phone": "0471-2721547"},
        {"name": "Panaji Police HQ", "lat": 15.4989, "lon": 73.8278, "city": "Panaji", "district": "North Goa", "phone": "0832-2420016"},
    ]
    return pd.DataFrame(police_stations)

police_data = get_police_stations_data()

# Function to find nearest police station for a district
@st.cache_data
def find_nearest_police_stations(district_centroid, police_df, top_n=3):
    """Find nearest police stations to a district centroid"""
    distances = []
    for _, station in police_df.iterrows():
        dist = geodesic((district_centroid.y, district_centroid.x), 
                       (station['lat'], station['lon'])).kilometers
        distances.append(dist)
    
    police_df_copy = police_df.copy()
    police_df_copy['distance'] = distances
    return police_df_copy.nsmallest(top_n, 'distance')

# Create optimized map
st.subheader("India Crime Safety Map")

col1, col2 = st.columns(2)
with col1:
    map_height = st.slider("Map height", 400, 800, 600)
    show_police_stations = st.checkbox("Show Police Stations", value=True)
with col2:
    tile_style = st.selectbox("Map style", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])
    police_radius = st.slider("Police Station Search Radius (km)", 10, 100, 50)

# Optimized map creation - use choropleth instead of individual GeoJson objects
@st.cache_data
def create_base_map():
    return folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")

main_map = create_base_map()

# Add tile layers
folium.TileLayer(tile_style, name=tile_style.replace('_', ' ').title()).add_to(main_map)

# Use choropleth for much faster rendering
folium.Choropleth(
    geo_data=districts_gdf.to_json(),
    name='Districts',
    data=merged_data,
    columns=['district_norm', 'crime_total'],
    key_on='properties.district_norm',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Crime Count'
).add_to(main_map)

# Add minimal interactive features with police stations
sample_districts = merged_data.sample(min(100, len(merged_data)))
for _, row in sample_districts.iterrows():
    # Find nearest police stations for this district
    nearest_stations = find_nearest_police_stations(row.geometry.centroid, police_data, top_n=2)
    
    station_info = ""
    if not nearest_stations.empty:
        station_list = []
        for _, station in nearest_stations.iterrows():
            station_list.append(f"• {station['name']} ({station['distance']:.1f}km)")
        station_info = f"<br><b>Nearest Police:</b><br>{'<br>'.join(station_list)}"
    
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=f"<div style='max-width:200px'><b>{row[district_name_col]}</b><br>Safety: {row['safety_level']}<br>Crimes: {int(row['crime_total']):,}{station_info}</div>",
        icon=folium.Icon(color='red' if row['safety_level'] == 'High' else 
                        'orange' if row['safety_level'] == 'Medium' else 'green',
                        icon='info-sign', prefix='glyphicon')
    ).add_to(main_map)

# Add police station markers if enabled
if show_police_stations:
    police_cluster = MarkerCluster(name="Police Stations") if PLUGINS_AVAILABLE else main_map
    
    for _, station in police_data.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; max-width: 220px;">
            <h4 style="margin: 0 0 8px 0; color: #c0392b;">🚔 {station['name']}</h4>
            <hr style="margin: 5px 0;">
            <b>City:</b> {station['city']}<br>
            <b>District:</b> {station['district']}<br>
            <b>Phone:</b> {station['phone']}<br>
            <b>Location:</b> {station['lat']:.4f}, {station['lon']:.4f}<br>
            <hr style="margin: 5px 0;">
            <small style="color: #7f8c8d;">Emergency: 100</small>
        </div>
        """
        
        folium.Marker(
            location=[station['lat'], station['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"🚔 {station['name']} - {station['city']}",
            icon=folium.Icon(color='blue', icon='shield', prefix='fa')
        ).add_to(police_cluster)
    
    if PLUGINS_AVAILABLE and hasattr(police_cluster, 'add_to'):
        police_cluster.add_to(main_map)

# Simplified legend
legend_html = f'''
<div style="position: fixed; top: 10px; right: 10px; width: 140px; height: 110px; 
     background-color: white; border: 2px solid grey; z-index: 9999; 
     font-size: 11px; padding: 8px;">
<b>Safety Levels</b><br>
<span style="color: {COLOR_MAP["Low"]}">■</span> Low Risk<br>
<span style="color: {COLOR_MAP["Medium"]}">■</span> Medium Risk<br>
<span style="color: {COLOR_MAP["High"]}">■</span> High Risk<br>
<span style="color: {COLOR_MAP["No Data"]}">■</span> No Data
</div>
'''

emergency_btn = '''
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999;">
<button style="background: #ff4444; color: white; border: none; border-radius: 25px; 
               padding: 12px 16px; font-weight: bold; cursor: pointer; 
               box-shadow: 0 2px 4px rgba(0,0,0,0.3);"
        onclick="alert('EMERGENCY:\\nPolice: 100 | Fire: 101 | Ambulance: 102')">
🚨 EMERGENCY
</button>
</div>
'''

main_map.get_root().html.add_child(folium.Element(legend_html))
main_map.get_root().html.add_child(folium.Element(emergency_btn))

if PLUGINS_AVAILABLE:
    Fullscreen().add_to(main_map)

folium.LayerControl().add_to(main_map)

# Display map
map_data = st_folium(main_map, width=1200, height=map_height)

# Simplified emergency section
st.subheader("Emergency Helplines")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Primary Emergency**\n- Police: 100\n- Fire: 101\n- Ambulance: 102")
with col2:
    st.markdown("**Safety Support**\n- Women: 1091\n- Child: 1098\n- Seniors: 14567")
with col3:
    st.markdown("**Health Crisis**\n- Mental Health: 9152987821\n- Poison: 1066\n- Traffic: 1073")

# Optimized location analysis
st.sidebar.header("Location Analysis")
location_input = st.sidebar.text_input("Enter location:", placeholder="City, State or lat,lon")

if location_input:
    @st.cache_data
    def geocode_location(location_str):
        geolocator = Nominatim(user_agent="crime-app", timeout=5)
        if ',' in location_str and len(location_str.split(',')) == 2:
            try:
                lat, lon = map(float, location_str.split(','))
                return lat, lon, f"Coordinates: {lat}, {lon}"
            except:
                pass
        try:
            location = geolocator.geocode(location_str + ", India")
            if location:
                return location.latitude, location.longitude, location.address
        except:
            pass
        return None, None, None
    
    lat, lon, address = geocode_location(location_input)
    
    if lat and lon:
        st.sidebar.success(f"Found: {address}")
        
        # Fast nearest district calculation
        point = Point(lon, lat)
        distances = merged_data.geometry.centroid.apply(
            lambda x: geodesic((lat, lon), (x.y, x.x)).kilometers
        )
        nearest_idx = distances.idxmin()
        nearest_district = merged_data.iloc[nearest_idx]
        
        st.sidebar.write(f"**Nearest District:** {nearest_district[district_name_col]}")
        st.sidebar.write(f"**Safety Level:** {nearest_district['safety_level']}")
        st.sidebar.write(f"**Distance:** {distances.iloc[nearest_idx]:.1f} km")
        
        # Find nearest police stations to the location
        police_distances = []
        for _, station in police_data.iterrows():
            dist = geodesic((lat, lon), (station['lat'], station['lon'])).kilometers
            police_distances.append(dist)
        
        police_with_dist = police_data.copy()
        police_with_dist['distance'] = police_distances
        nearest_police = police_with_dist.nsmallest(3, 'distance')
        
        if not nearest_police.empty:
            st.sidebar.write("**Nearest Police Stations:**")
            for _, station in nearest_police.iterrows():
                st.sidebar.write(f"• {station['name']} ({station['distance']:.1f}km)")
                st.sidebar.write(f"  📞 {station['phone']}")
    else:
        st.sidebar.error("Location not found")

# Police Station Directory
st.subheader("🚔 Police Station Directory")

# Filter options
col1, col2 = st.columns(2)
with col1:
    selected_city = st.selectbox("Filter by City:", 
                                ["All"] + sorted(police_data['city'].unique().tolist()))
with col2:
    show_contact_details = st.checkbox("Show Contact Details", value=False)

# Filter police data
if selected_city != "All":
    filtered_police = police_data[police_data['city'] == selected_city]
else:
    filtered_police = police_data

# Display police stations
if show_contact_details:
    display_cols = ['name', 'city', 'district', 'phone']
    st.dataframe(filtered_police[display_cols].rename(columns={
        'name': 'Police Station',
        'city': 'City', 
        'district': 'District',
        'phone': 'Phone'
    }), use_container_width=True)
else:
    display_cols = ['name', 'city', 'district']
    st.dataframe(filtered_police[display_cols].rename(columns={
        'name': 'Police Station',
        'city': 'City',
        'district': 'District'
    }), use_container_width=True)

st.info(f"Showing {len(filtered_police)} police stations. Emergency number: **100**")

# Quick stats
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Districts", len(merged_data))
with col2:
    st.metric("With Data", (merged_data['crime_total'] > 0).sum())
with col3:
    st.metric("High Risk", (merged_data['safety_level'] == 'High').sum())
with col4:
    st.metric("Total Crimes", f"{int(merged_data['crime_total'].sum()):,}")

st.markdown("---")
st.markdown("**Note:** Optimized for speed. Some features simplified for faster loading.")
