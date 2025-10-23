# app.py - OPTIMIZED VERSION
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob, os, json, requests
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("🗺️ India Crime Heatmap - District Level Analysis")

# --------------------------- 
# CONFIGURATION
# ---------------------------
DATA_FOLDER = "data"
# Multiple GeoJSON sources as fallbacks
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson",
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/India_Districts.geojson"
]

# --------------------------- 
# IMPROVED DISTRICT NAME NORMALIZATION
# ---------------------------
def normalize_name(s):
    """Enhanced normalization for better district matching"""
    if pd.isna(s):
        return ""
    
    s = str(s).lower().strip()
    
    # Common replacements for Indian districts
    replacements = {
        'bengaluru': 'bangalore',
        'bangalore urban': 'bangalore',
        'bengaluru urban': 'bangalore',
        'mumbai city': 'mumbai',
        'mumbai suburban': 'mumbai',
        'delhi': 'new delhi',
        'new delhi district': 'new delhi',
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
        'y s r': 'ysr',
        'sri potti sriramulu nellore': 'nellore',
    }
    
    for old, new in replacements.items():
        s = s.replace(old, new)
    
    # Remove common suffixes
    remove_words = ['city', 'rural', 'district', 'commissionerate', 'urban', 'metropolitan']
    for word in remove_words:
        s = s.replace(f' {word}', '')
    
    # Clean special characters but keep spaces
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())  # Remove extra spaces
    
    return s

# --------------------------- 
# LOAD CSV DATA - OPTIMIZED
# ---------------------------
@st.cache_data(show_spinner=False)
def load_and_aggregate_csvs(data_folder):
    """Fast CSV loading with better error handling"""
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame(), []
    
    all_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, f in enumerate(files):
        status_text.text(f"Loading {os.path.basename(f)}...")
        progress_bar.progress((idx + 1) / len(files))
        
        try:
            df = pd.read_csv(f, low_memory=False, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(f, low_memory=False, encoding='latin1')
            except Exception as e:
                continue
        
        # Find district column
        district_col = None
        for col in df.columns:
            if 'district' in col.lower() or col.lower() == 'name':
                district_col = col
                break
        
        if district_col is None:
            district_col = df.columns[0]
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            df['crime_count'] = df[numeric_cols].sum(axis=1)
        else:
            df['crime_count'] = 1
        
        df['district_norm'] = df[district_col].apply(normalize_name)
        all_data.append(df[['district_norm', 'crime_count']])
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        return pd.DataFrame(), files
    
    combined = pd.concat(all_data, ignore_index=True)
    aggregated = combined.groupby('district_norm', as_index=False)['crime_count'].sum()
    
    return aggregated, files

# Load data
with st.spinner("Loading crime data..."):
    crime_agg, csv_files = load_and_aggregate_csvs(DATA_FOLDER)

if crime_agg.empty:
    st.error(f"No CSV files found in `{DATA_FOLDER}`. Please add your crime data files.")
    st.stop()

st.success(f"✅ Loaded {len(csv_files)} CSV files, found {len(crime_agg)} districts")

# --------------------------- 
# LOAD GEOJSON - OPTIMIZED WITH FALLBACKS
# ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson():
    """Load GeoJSON with multiple fallback sources"""
    errors = []
    
    for idx, url in enumerate(GEOJSON_URLS):
        try:
            st.info(f"Trying GeoJSON source {idx + 1}/{len(GEOJSON_URLS)}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            gj = response.json()
            gdf = gpd.GeoDataFrame.from_features(gj['features'])
            st.success(f"✅ Loaded GeoJSON from source {idx + 1}")
            return gdf
        except Exception as e:
            errors.append(f"Source {idx + 1}: {str(e)}")
            continue
    
    # If all sources fail, show error with upload option
    st.error("❌ All GeoJSON sources failed. Please upload a file manually.")
    st.write("**Attempted sources and errors:**")
    for error in errors:
        st.write(f"- {error}")
    
    st.write("\n**Upload your own GeoJSON:**")
    uploaded_file = st.file_uploader(
        "Upload India Districts GeoJSON",
        type=["json", "geojson"],
        help="Download from: https://github.com/geohacker/india/blob/master/district/india_district.geojson"
    )
    
    if uploaded_file:
        try:
            gj = json.load(uploaded_file)
            gdf = gpd.GeoDataFrame.from_features(gj['features'])
            st.success("✅ Loaded uploaded GeoJSON successfully!")
            return gdf
        except Exception as e:
            st.error(f"Failed to load uploaded file: {str(e)}")
    
    st.stop()

with st.spinner("Loading map boundaries..."):
    gdf_districts = load_geojson()

# Find district name column
name_col = None
for col in ['NAME_2', 'district', 'DISTRICT', 'NAME', 'name']:
    if col in gdf_districts.columns:
        if gdf_districts[col].nunique() > 50:
            name_col = col
            break

if name_col is None:
    st.error("Could not identify district column in GeoJSON")
    st.stop()

gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# --------------------------- 
# MERGE DATA
# ---------------------------
merged = gdf_districts.merge(
    crime_agg[['district_norm', 'crime_count']], 
    on='district_norm', 
    how='left'
)
merged['crime_count'] = merged['crime_count'].fillna(0)

# Calculate safety levels
non_zero = merged[merged['crime_count'] > 0]['crime_count']
if len(non_zero) > 0:
    q1 = non_zero.quantile(0.33)
    q2 = non_zero.quantile(0.66)
else:
    q1, q2 = 0, 0

def classify_safety(val):
    if val == 0:
        return "No Data"
    elif val <= q1:
        return "Low"
    elif val <= q2:
        return "Medium"
    else:
        return "High"

merged['safety_level'] = merged['crime_count'].apply(classify_safety)

# Show matching stats
matched = (merged['crime_count'] > 0).sum()
st.info(f"📊 Matched {matched}/{len(merged)} districts with crime data")

# --------------------------- 
# MAIN MAP - SIMPLIFIED AND FAST
# ---------------------------
st.subheader("🗺️ National Crime Heatmap")

def create_main_map(merged_json, name_col):
    """Create optimized choropleth map"""
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="OpenStreetMap",
        prefer_canvas=True  # Performance improvement
    )
    
    # Simplified styling
    def style_function(feature):
        safety = feature['properties'].get('safety_level', 'No Data')
        colors = {
            'No Data': '#e0e0e0',
            'Low': '#4caf50',
            'Medium': '#ffc107',
            'High': '#f44336'
        }
        return {
            'fillColor': colors.get(safety, '#e0e0e0'),
            'color': '#ffffff',
            'weight': 0.5,
            'fillOpacity': 0.7
        }
    
    # Add choropleth layer
    folium.GeoJson(
        merged_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[name_col, 'crime_count', 'safety_level'],
            aliases=['District:', 'Crime Count:', 'Safety:'],
            localize=True,
            sticky=False
        ),
        popup=folium.GeoJsonPopup(
            fields=[name_col, 'crime_count', 'safety_level', 'district_norm'],
            aliases=['District:', 'Crime Count:', 'Safety Level:', 'Normalized Name:'],
            localize=True,
            max_width=300
        )
    ).add_to(m)
    
    # Simple legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; padding:10px; border:2px solid grey; border-radius:5px;">
    <p><strong>Safety Level</strong></p>
    <p><span style="color:#4caf50;">⬤</span> Low Risk</p>
    <p><span style="color:#ffc107;">⬤</span> Medium Risk</p>
    <p><span style="color:#f44336;">⬤</span> High Risk</p>
    <p><span style="color:#e0e0e0;">⬤</span> No Data (0 crimes)</p>
    <p style="margin-top:10px; font-size:11px; color:#666;">
    💡 Click districts to see details<br/>
    Hover for quick info
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Create and display main map
with st.spinner("Rendering map..."):
    # Convert to GeoJSON format for faster rendering
    merged_json = json.loads(merged.to_json())
    main_map = create_main_map(merged_json, name_col)
    st_folium(main_map, width=1200, height=600, returned_objects=[])

# Show district data summary
st.markdown("---")
st.subheader("📊 District Crime Data Summary")

col1, col2 = st.columns(2)

with col1:
    # Statistics
    st.metric("Total Districts", len(merged))
    st.metric("Districts with Crime Data", (merged['crime_count'] > 0).sum())
    st.metric("Districts with No Data", (merged['crime_count'] == 0).sum())

with col2:
    # Top 5 highest crime districts
    st.write("**Top 5 Highest Crime Districts:**")
    top_districts = merged.nlargest(5, 'crime_count')[[name_col, 'crime_count', 'safety_level']]
    for idx, row in top_districts.iterrows():
        st.write(f"• {row[name_col]}: {int(row['crime_count']):,} crimes ({row['safety_level']})")

# Searchable data table
st.subheader("🔍 Search All Districts")
search_term = st.text_input("Search district name:", placeholder="Type to filter...")

display_df = merged[[name_col, 'crime_count', 'safety_level', 'district_norm']].copy()
display_df.columns = ['District Name', 'Crime Count', 'Safety Level', 'Normalized Name']
display_df = display_df.sort_values('Crime Count', ascending=False)

if search_term:
    display_df = display_df[display_df['District Name'].str.contains(search_term, case=False, na=False) | 
                            display_df['Normalized Name'].str.contains(search_term, case=False, na=False)]

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    column_config={
        "Crime Count": st.column_config.NumberColumn(format="%d"),
        "Safety Level": st.column_config.TextColumn()
    }
)

# --------------------------- 
# LOCATION SEARCH - OPTIMIZED
# ---------------------------
st.markdown("---")
st.subheader("🔍 Search Location Safety")

col1, col2 = st.columns([3, 1])
with col1:
    location_input = st.text_input(
        "Enter location (address or lat,lon):",
        placeholder="e.g., Bangalore or 12.9716,77.5946"
    )
with col2:
    search_button = st.button("🔍 Search", type="primary")

if search_button and location_input:
    geolocator = Nominatim(user_agent="crime-map", timeout=10)
    
    # Parse input
    lat, lon = None, None
    if ',' in location_input:
        try:
            parts = location_input.split(',')
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
        except:
            pass
    
    if lat is None:
        with st.spinner("Geocoding location..."):
            try:
                location = geolocator.geocode(location_input + ", India")
                if location:
                    lat, lon = location.latitude, location.longitude
                else:
                    st.error("Could not find location. Try a different query.")
                    st.stop()
            except Exception as e:
                st.error(f"Geocoding failed: {str(e)}")
                st.stop()
    
    st.success(f"📍 Found location: {lat:.4f}, {lon:.4f}")
    
    # Find containing district
    point = Point(lon, lat)
    containing = merged[merged.contains(point)]
    
    if len(containing) > 0:
        district_row = containing.iloc[0]
    else:
        # Find nearest district
        merged_copy = merged.copy()
        merged_copy['centroid'] = merged_copy.geometry.centroid
        merged_copy['dist'] = merged_copy['centroid'].apply(
            lambda c: geodesic((c.y, c.x), (lat, lon)).km
        )
        district_row = merged_copy.nsmallest(1, 'dist').iloc[0]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("District", district_row[name_col])
    with col2:
        safety = district_row['safety_level']
        color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴', 'No Data': '⚫'}
        st.metric("Safety Level", f"{safety} {color.get(safety, '⚫')}")
    with col3:
        st.metric("Crime Count", int(district_row['crime_count']))
    
    # Create local map
    st.subheader("📍 Local Area Map")
    
    local_map = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    
    # Add marker
    folium.Marker(
        location=[lat, lon],
        popup=f"📍 Your Location<br>{lat:.4f}, {lon:.4f}",
        tooltip="You are here",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(local_map)
    
    # Add nearby districts
    merged_nearby = merged.copy()
    merged_nearby['centroid'] = merged_nearby.geometry.centroid
    merged_nearby['dist_km'] = merged_nearby['centroid'].apply(
        lambda c: geodesic((c.y, c.x), (lat, lon)).km
    )
    nearby = merged_nearby[merged_nearby['dist_km'] <= 20].copy()
    
    def style_nearby(feature):
        safety = feature['properties'].get('safety_level', 'No Data')
        colors = {'No Data': '#e0e0e0', 'Low': '#4caf50', 'Medium': '#ffc107', 'High': '#f44336'}
        return {
            'fillColor': colors.get(safety, '#e0e0e0'),
            'color': '#ffffff',
            'weight': 1,
            'fillOpacity': 0.6
        }
    
    folium.GeoJson(
        nearby,
        style_function=style_nearby,
        tooltip=folium.GeoJsonTooltip(
            fields=[name_col, 'safety_level', 'dist_km'],
            aliases=['District:', 'Safety:', 'Distance (km):'],
            localize=True
        )
    ).add_to(local_map)
    
    st_folium(local_map, width=1000, height=500, returned_objects=[])
    
    # Show nearby districts table
    st.subheader("📋 Nearby Districts")
    nearby_display = nearby[[name_col, 'crime_count', 'safety_level', 'dist_km']].copy()
    nearby_display.columns = ['District', 'Crime Count', 'Safety Level', 'Distance (km)']
    nearby_display = nearby_display.sort_values('Distance (km)').head(10)
    st.dataframe(nearby_display, use_container_width=True)
    
    # Emergency contact
    st.info("🚨 **Emergency Services**: Police: 100 | Ambulance: 102 | Fire: 101")

# Footer
st.markdown("---")
st.markdown("""
**Notes:**
- Crime scores are relative rankings based on your dataset
- Safety levels: Low/Medium/High are determined by quantiles
- Maps may take a few seconds to render
- For emergencies, always call local emergency services
""")
