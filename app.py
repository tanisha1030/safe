# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob
import os
import json
import requests
from shapely.geometry import Point
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

# Import folium plugins with error handling
try:
    from folium.plugins import Fullscreen, MeasureControl, MarkerCluster
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — District-level Safety Analysis")
st.markdown(
    "This app aggregates CSV files containing district-wise crime data, "
    "creates a safety score visualization, and provides location-based safety analysis."
)

# Configuration
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
]
DATA_FOLDER = "data"
SEARCH_RADIUS_KM = 5

# Utility function to normalize district names
def normalize_name(s):
    """Normalize district names for better matching"""
    if pd.isna(s):
        return ""
    
    s = str(s).lower().strip()
    
    # Common replacements for Indian district names
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
    
    # Keep only alphanumeric and spaces
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())  # Normalize whitespace
    return s

# Load and aggregate CSV files
@st.cache_data(show_spinner=False)
def load_and_aggregate_csvs(data_folder):
    """Load and aggregate all CSV files in the data folder"""
    if not os.path.exists(data_folder):
        return pd.DataFrame(), [], [("Folder not found", f"Directory '{data_folder}' does not exist")]
    
    csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not csv_files:
        return pd.DataFrame(), [], [("No files", "No CSV files found in data folder")]
    
    aggregated_rows = []
    failed = []
    
    for file_path in csv_files:
        try:
            # Try reading with default encoding first
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
            
            if df.empty:
                continue
            
            # Find district column
            district_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['district', 'area', 'region']):
                    district_col = col
                    break
            
            # Fallback to first column if no district column found
            if district_col is None:
                district_col = df.columns[0]
            
            # Find numeric columns for crime data
            numeric_cols = []
            for col in df.columns:
                if col != district_col:
                    # Try to convert to numeric
                    temp_series = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                    if not temp_series.isna().all():
                        numeric_cols.append(col)
            
            if numeric_cols:
                # Sum numeric columns for each district
                df_numeric = df[numeric_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce')).fillna(0)
                df['crime_total'] = df_numeric.sum(axis=1)
            else:
                # If no numeric columns, assign 1 for presence
                df['crime_total'] = 1
            
            # Create summary dataframe
            summary_df = df[[district_col, 'crime_total']].copy()
            summary_df.columns = ['district_raw', 'crime_total']
            summary_df['district_norm'] = summary_df['district_raw'].apply(normalize_name)
            
            # Remove empty districts
            summary_df = summary_df[summary_df['district_norm'].str.strip() != '']
            
            if not summary_df.empty:
                aggregated_rows.append(summary_df)
                
        except Exception as e:
            failed.append((os.path.basename(file_path), str(e)))
            continue
    
    if not aggregated_rows:
        return pd.DataFrame(), csv_files, failed
    
    # Combine all data
    all_data = pd.concat(aggregated_rows, ignore_index=True)
    
    # Group by normalized district name and sum crime totals
    final_agg = all_data.groupby('district_norm', as_index=False).agg({
        'crime_total': 'sum',
        'district_raw': 'first'  # Keep first occurrence as example
    })
    
    return final_agg, csv_files, failed

# Load crime data
with st.spinner("Loading crime data..."):
    crime_data, csv_files, failed_files = load_and_aggregate_csvs(DATA_FOLDER)

if crime_data.empty:
    st.error(f"No valid crime data found in '{DATA_FOLDER}' folder.")
    st.info("Please ensure you have CSV files with district-wise crime data in the data folder.")
    if failed_files:
        st.write("Failed to read files:")
        for filename, error in failed_files[:5]:
            st.write(f"- {filename}: {error}")
    st.stop()

st.success(f"Successfully loaded {len(csv_files)} CSV files with data for {len(crime_data)} districts.")

# Sidebar configuration
st.sidebar.header("Configuration")
st.sidebar.write(f"Loaded {len(csv_files)} CSV files")

geo_source = st.sidebar.radio(
    "GeoJSON Source:",
    ("Use default India districts", "Upload custom GeoJSON")
)

uploaded_geojson = None
if geo_source == "Upload custom GeoJSON":
    uploaded_geojson = st.sidebar.file_uploader(
        "Upload India districts GeoJSON", 
        type=["json", "geojson"]
    )

# Load GeoJSON data
@st.cache_data(show_spinner=False)
def load_geojson(uploaded_file, default_urls):
    """Load GeoJSON from file or URLs"""
    if uploaded_file is not None:
        geojson_data = json.load(uploaded_file)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
        return gdf, "uploaded file"
    
    # Try default URLs
    for i, url in enumerate(default_urls):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            geojson_data = response.json()
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            return gdf, f"URL {i+1}"
        except Exception as e:
            continue
    
    raise Exception("All GeoJSON sources failed")

# Load district boundaries
try:
    with st.spinner("Loading district boundaries..."):
        districts_gdf, source_used = load_geojson(uploaded_geojson, DEFAULT_GEOJSON_URLS)
    st.info(f"Loaded district boundaries from {source_used}")
except Exception as e:
    st.error("Failed to load district boundaries.")
    st.write("Please try uploading a GeoJSON file or check your internet connection.")
    st.stop()

# Find district name column in GeoJSON
def find_district_column(gdf):
    """Find the most likely district name column"""
    # Priority order for district name columns
    priority_cols = ['NAME_2', 'NAME_3', 'DTNAME', 'name_2', 'district_name', 'NAME', 'name', 'district']
    
    for col in priority_cols:
        if col in gdf.columns:
            unique_count = gdf[col].nunique()
            if unique_count > 10:  # Should have many districts
                return col
    
    # Fallback: find any string column with many unique values
    for col in gdf.columns:
        if gdf[col].dtype == 'object':
            unique_count = gdf[col].nunique()
            if unique_count > 50:
                return col
    
    return None

district_name_col = find_district_column(districts_gdf)

if district_name_col is None:
    st.error("Could not identify district name column in GeoJSON.")
    st.write("Available columns:", list(districts_gdf.columns))
    st.stop()

# Normalize district names in GeoJSON
districts_gdf['district_norm'] = districts_gdf[district_name_col].apply(normalize_name)

# Merge crime data with district boundaries
merged_data = districts_gdf.merge(
    crime_data[['district_norm', 'crime_total']], 
    on='district_norm', 
    how='left'
)
merged_data['crime_total'] = merged_data['crime_total'].fillna(0)

# Calculate matching statistics
matched_districts = (merged_data['crime_total'] > 0).sum()
total_districts = len(merged_data)
unmatched_districts = total_districts - matched_districts

st.info(f"District matching: {matched_districts}/{total_districts} districts have crime data.")

if unmatched_districts > total_districts * 0.5:
    st.warning("Many districts are unmatched. This may indicate naming inconsistencies between your data and the GeoJSON file.")

# Create safety levels based on quantiles
non_zero_crimes = merged_data[merged_data['crime_total'] > 0]['crime_total']
if len(non_zero_crimes) > 0:
    q33 = non_zero_crimes.quantile(0.33)
    q66 = non_zero_crimes.quantile(0.66)
else:
    q33, q66 = 0, 0

def get_safety_level(crime_count):
    """Classify crime count into safety levels"""
    if crime_count == 0:
        return "No Data"
    elif crime_count <= q33:
        return "Low"
    elif crime_count <= q66:
        return "Medium"
    else:
        return "High"

merged_data['safety_level'] = merged_data['crime_total'].apply(get_safety_level)

# Create color mapping
def get_color_for_safety(safety_level):
    """Get color for safety level"""
    colors = {
        "No Data": "#f0f0f0",
        "Low": "#6bcf7f", 
        "Medium": "#ffd93d",
        "High": "#ff6b6b"
    }
    return colors.get(safety_level, "#f0f0f0")

# Main map
st.subheader("India Crime Safety Map")

# Map controls
col1, col2 = st.columns(2)
with col1:
    show_labels = st.checkbox("Show district labels", value=False)
with col2:
    map_height = st.slider("Map height", 400, 800, 600)

# Create main choropleth map
main_map = folium.Map(
    location=[20.5937, 78.9629],  # Center of India
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Add tile layers
folium.TileLayer('OpenStreetMap', name='Street Map').add_to(main_map)
folium.TileLayer('CartoDB positron', name='Light Map').add_to(main_map)
folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(main_map)

# Add districts to map
for _, row in merged_data.iterrows():
    # Get color for this specific row
    color = get_color_for_safety(row['safety_level'])
    
    # Create style function for this specific feature
    def create_style_function(fill_color):
        return lambda feature: {
            'fillColor': fill_color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }
    
    # Create popup content
    popup_html = f"""
    <div style="font-family: Arial; max-width: 200px;">
        <h4 style="margin: 0 0 10px 0;">{row[district_name_col]}</h4>
        <p><b>Safety Level:</b> {row['safety_level']}</p>
        <p><b>Crime Count:</b> {int(row['crime_total']):,}</p>
    </div>
    """
    
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=create_style_function(color),
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row[district_name_col]} - {row['safety_level']}"
    ).add_to(main_map)

# Add legend and emergency button
legend_html = '''
<div style="position: fixed; 
     top: 10px; right: 10px; width: 150px; height: 120px; 
     background-color: white; border:2px solid grey; z-index:9999; 
     font-size:12px; padding: 10px">
<p><b>Safety Levels</b></p>
<p><i class="fa fa-square" style="color:#6bcf7f"></i> Low Risk</p>
<p><i class="fa fa-square" style="color:#ffd93d"></i> Medium Risk</p>
<p><i class="fa fa-square" style="color:#ff6b6b"></i> High Risk</p>
<p><i class="fa fa-square" style="color:#f0f0f0"></i> No Data</p>
</div>
'''

emergency_button_html = '''
<div style="position: fixed; 
     bottom: 20px; right: 20px; z-index:9999;">
<button style="background-color: #ff4444; color: white; border: none; 
               border-radius: 50px; padding: 15px 20px; font-size: 14px; 
               font-weight: bold; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.3);
               animation: pulse 2s infinite;"
        onclick="alert('EMERGENCY NUMBERS:\\n\\nPolice: 100\\nFire: 101\\nAmbulance: 102\\nWomen Safety: 1091\\nChild Helpline: 1098\\n\\nIn immediate danger, call 100!')">
🚨 EMERGENCY
</button>
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
}
</style>
</div>
'''

main_map.get_root().html.add_child(folium.Element(legend_html))
main_map.get_root().html.add_child(folium.Element(emergency_button_html))

# Add layer control
folium.LayerControl().add_to(main_map)

# Add fullscreen if available
if PLUGINS_AVAILABLE:
    Fullscreen().add_to(main_map)

# Display main map
map_data = st_folium(main_map, width=1200, height=map_height)

# Emergency Helplines Section
st.subheader("🚨 Emergency Helpline Numbers")
st.markdown("**Keep these numbers handy for immediate assistance:**")

# Create emergency helplines in columns
help_col1, help_col2, help_col3 = st.columns(3)

with help_col1:
    st.markdown("""
    **🚔 Police & Emergency**
    - **Police Emergency**: 100
    - **Fire Emergency**: 101
    - **Ambulance/Medical**: 102
    - **Disaster Management**: 108
    - **Traffic Helpline**: 1073
    """)

with help_col2:
    st.markdown("""
    **👥 Safety & Support**
    - **Women Helpline**: 1091
    - **Women Safety (24x7)**: 181
    - **Child Helpline**: 1098
    - **Senior Citizen Helpline**: 14567
    - **Tourist Emergency**: 1363
    """)

with help_col3:
    st.markdown("""
    **🏥 Health & Crisis**
    - **Mental Health**: 9152987821
    - **Suicide Prevention**: 9152987821
    - **Drug De-addiction**: 1031
    - **Anti-Poison Helpline**: 1066
    - **Railway Enquiry**: 139
    """)

# State-specific helplines
with st.expander("📱 State-Specific Helplines", expanded=False):
    state_col1, state_col2 = st.columns(2)
    
    with state_col1:
        st.markdown("""
        **Maharashtra**
        - Mumbai Police: 022-22633333
        - Pune Police: 020-26128570
        
        **Delhi**
        - Delhi Police: 011-23490085
        - Delhi Women Safety: 011-23317004
        
        **Karnataka**
        - Bangalore Police: 080-22208446
        - Karnataka Women Helpline: 080-22100100
        
        **Tamil Nadu**
        - Chennai Police: 044-23452348
        - TN Women Helpline: 044-28592750
        """)
    
    with state_col2:
        st.markdown("""
        **West Bengal**
        - Kolkata Police: 033-22143526
        - WB Women Commission: 033-22875648
        
        **Gujarat**
        - Ahmedabad Police: 079-25506444
        - Gujarat Women Helpline: 181
        
        **Rajasthan**
        - Jaipur Police: 0141-2743000
        - Rajasthan Women Helpline: 0141-2744000
        
        **Uttar Pradesh**
        - UP Women Helpline: 1090
        - Lucknow Police: 0522-2624015
        """)

# Important safety tips
st.info("""
**Safety Tips:**
• Save these numbers in your phone contacts
• Share your location with trusted contacts when traveling
• Trust your instincts - if something feels wrong, seek help immediately
• In case of immediate danger, call 100 (Police) or 112 (Unified Emergency Number)
""")

# Download emergency contacts card
st.subheader("📱 Download Emergency Contacts")

emergency_contacts_text = """
INDIA EMERGENCY CONTACTS - Keep This Card Safe

🚨 IMMEDIATE EMERGENCY
Police: 100
Fire: 101
Ambulance/Medical: 102
Unified Emergency: 112

👥 SAFETY & SUPPORT  
Women Helpline: 1091
Women Safety 24x7: 181
Child Helpline: 1098
Senior Citizens: 14567
Tourist Emergency: 1363

🏥 HEALTH & CRISIS
Mental Health: 9152987821
Suicide Prevention: 9152987821  
Anti-Poison: 1066
Drug De-addiction: 1031

📍 LOCATION SERVICES
Traffic Helpline: 1073
Railway Enquiry: 139
Disaster Management: 108

⚠️ REMEMBER:
- In immediate danger, call 100
- Share location with trusted contacts
- Trust your instincts
- Keep phone charged

Generated by India Crime Safety App
"""

st.download_button(
    label="Download Emergency Contacts Card 📋",
    data=emergency_contacts_text,
    file_name="india_emergency_contacts.txt",
    mime="text/plain",
    help="Download this emergency contacts list to save on your phone"
)

st.markdown("---")

# Emergency numbers in sidebar for quick access
st.sidebar.header("🚨 Emergency Numbers")
st.sidebar.markdown("""
**Quick Access:**
- **Police**: 100
- **Fire**: 101  
- **Ambulance**: 102
- **Women Safety**: 1091
- **Child Helpline**: 1098
""")
st.sidebar.error("In emergency, call 100 immediately!")

st.sidebar.markdown("---")

# Location analysis section
st.sidebar.header("Location Safety Analysis")
location_input = st.sidebar.text_input(
    "Enter location (address or lat,lon):",
    placeholder="e.g., Mumbai, Maharashtra or 19.0760,72.8777"
)

if location_input:
    # Initialize geocoder
    geolocator = Nominatim(user_agent="india-crime-app", timeout=10)
    
    def parse_location(input_str):
        """Parse location input as coordinates or address"""
        input_str = input_str.strip()
        
        # Check if it's coordinates (lat,lon)
        if ',' in input_str:
            try:
                parts = input_str.split(',')
                if len(parts) == 2:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    return lat, lon, f"Coordinates: {lat}, {lon}"
            except ValueError:
                pass
        
        # Try geocoding as address
        try:
            location = geolocator.geocode(input_str + ", India")
            if location:
                return location.latitude, location.longitude, location.address
        except Exception as e:
            st.error(f"Geocoding error: {e}")
        
        return None, None, None
    
    with st.spinner("Analyzing location..."):
        lat, lon, resolved_address = parse_location(location_input)
        
        if lat is not None and lon is not None:
            st.success(f"Location found: {resolved_address}")
            
            # Find containing or nearest district
            point = Point(lon, lat)
            
            # Check if point is within any district
            containing_districts = merged_data[merged_data.geometry.contains(point)]
            
            if not containing_districts.empty:
                district_info = containing_districts.iloc[0]
                st.subheader("Your District Safety Analysis")
                
                safety_color = get_color_for_safety(district_info['safety_level'])
                st.markdown(f"""
                **District:** {district_info[district_name_col]}  
                **Safety Level:** <span style="color: {safety_color}">●</span> {district_info['safety_level']}  
                **Crime Count:** {int(district_info['crime_total']):,}
                """, unsafe_allow_html=True)
                
            else:
                # Find nearest district
                merged_data_copy = merged_data.copy()
                merged_data_copy['centroid'] = merged_data_copy.geometry.centroid
                merged_data_copy['distance'] = merged_data_copy['centroid'].apply(
                    lambda x: geodesic((lat, lon), (x.y, x.x)).kilometers
                )
                nearest_district = merged_data_copy.loc[merged_data_copy['distance'].idxmin()]
                
                st.subheader("Nearest District Safety Analysis")
                st.info(f"You are approximately {nearest_district['distance']:.1f} km from {nearest_district[district_name_col]}")
                
                safety_color = get_color_for_safety(nearest_district['safety_level'])
                st.markdown(f"""
                **District:** {nearest_district[district_name_col]}  
                **Safety Level:** <span style="color: {safety_color}">●</span> {nearest_district['safety_level']}  
                **Crime Count:** {int(nearest_district['crime_total']):,}
                """, unsafe_allow_html=True)
            
            # Create local area map
            st.subheader("Local Area Map")
            
            local_map = folium.Map(
                location=[lat, lon],
                zoom_start=10,
                tiles="OpenStreetMap"
            )
            
            # Add user location marker
            folium.Marker(
                location=[lat, lon],
                popup=f"Your Location<br>{lat:.4f}, {lon:.4f}",
                tooltip="You are here",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(local_map)
            
            # Add nearby districts
            nearby_districts = merged_data_copy[merged_data_copy['distance'] <= 50].sort_values('distance')
            
            for _, district in nearby_districts.head(10).iterrows():
                folium.GeoJson(
                    district.geometry.__geo_interface__,
                    style_function=style_function,
                    popup=f"{district[district_name_col]}<br>Safety: {district['safety_level']}<br>Distance: {district['distance']:.1f} km",
                    tooltip=f"{district[district_name_col]} - {district['safety_level']}"
                ).add_to(local_map)
            
            # Display local map
            st_folium(local_map, width=800, height=400)
            
            # Show nearby districts table
            if not nearby_districts.empty:
                st.subheader("Nearby Districts (within 50km)")
                display_data = nearby_districts[
                    [district_name_col, 'safety_level', 'crime_total', 'distance']
                ].head(10).copy()
                display_data.columns = ['District', 'Safety Level', 'Crime Count', 'Distance (km)']
                display_data['Distance (km)'] = display_data['Distance (km)'].round(1)
                display_data['Crime Count'] = display_data['Crime Count'].astype(int)
                st.dataframe(display_data, use_container_width=True)
        else:
            st.error("Could not find the specified location. Please try a different address or coordinates.")

# Display summary statistics
st.subheader("Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Districts", len(merged_data))

with col2:
    st.metric("Districts with Data", matched_districts)

with col3:
    high_risk_count = len(merged_data[merged_data['safety_level'] == 'High'])
    st.metric("High Risk Districts", high_risk_count)

with col4:
    total_crimes = int(merged_data['crime_total'].sum())
    st.metric("Total Crime Records", f"{total_crimes:,}")

# Safety level distribution
if not merged_data.empty:
    st.subheader("Safety Level Distribution")
    safety_counts = merged_data['safety_level'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a simple bar chart
        chart_data = pd.DataFrame({
            'Safety Level': safety_counts.index,
            'Count': safety_counts.values
        })
        st.bar_chart(chart_data.set_index('Safety Level'))
    
    with col2:
        st.write("**District Counts:**")
        for level, count in safety_counts.items():
            percentage = (count / len(merged_data)) * 100
            st.write(f"{level}: {count} ({percentage:.1f}%)")

# Footer
st.markdown("---")
st.markdown(
    """
    **Important Notes:**
    - Crime data is aggregated from CSV files in the data folder
    - Safety levels are relative classifications based on crime count quantiles
    - District matching depends on name consistency between CSV files and GeoJSON
    - For production use, consider using authenticated geocoding services
    """
)
