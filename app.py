# app.py - ALL DISTRICTS VERSION WITH SYNTHETIC DATA
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob, os, json, requests
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
import numpy as np
warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("🗺️ India Crime Heatmap - District Level Analysis")

# ---------------------------
# CONFIGURATION
# ---------------------------
DATA_FOLDER = "data"
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
    
    remove_words = ['city', 'rural', 'district', 'commissionerate', 'urban', 'metropolitan']
    for word in remove_words:
        s = s.replace(f' {word}', '')
    
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    
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
        
        district_col = None
        for col in df.columns:
            if 'district' in col.lower() or col.lower() == 'name':
                district_col = col
                break
        
        if district_col is None:
            district_col = df.columns[0]
        
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
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            gj = response.json()
            gdf = gpd.GeoDataFrame.from_features(gj['features'])
            return gdf
        except Exception as e:
            errors.append(f"Source {idx + 1}: {str(e)}")
            continue
    
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
            st.success("✅ Loaded map boundaries successfully!")
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
# MERGE DATA WITH SYNTHETIC DATA GENERATION
# ---------------------------
merged = gdf_districts.merge(
    crime_agg[['district_norm', 'crime_count']], 
    on='district_norm', 
    how='left'
)

# Generate synthetic low crime data for missing districts
districts_without_data = merged['crime_count'].isna()
num_missing = districts_without_data.sum()

if num_missing > 0:
    # Get the lowest 10% of actual crime counts to use as reference
    actual_crimes = crime_agg['crime_count'].values
    low_crime_threshold = np.percentile(actual_crimes, 10)
    
    # Generate unique synthetic data: random WHOLE numbers between 1 and low_crime_threshold
    min_val = 50
    max_val = int(max(500, low_crime_threshold * 0.5))
    
    # Generate unique random integers
    synthetic_values = np.random.choice(
        range(min_val, max_val + 1), 
        size=num_missing, 
        replace=False if num_missing <= (max_val - min_val + 1) else True
    ).astype(int)
    
    # Apply synthetic data as whole numbers
    merged.loc[districts_without_data, 'crime_count'] = synthetic_values
    merged.loc[districts_without_data, 'synthetic'] = True
    merged['synthetic'] = merged['synthetic'].fillna(False)

# Ensure no zero or NaN values and convert all to integers
merged['crime_count'] = merged['crime_count'].fillna(1)
merged['crime_count'] = merged['crime_count'].replace(0, 1)
merged['crime_count'] = merged['crime_count'].astype(int)

# Calculate safety levels (all districts will have a level now)
q1 = merged['crime_count'].quantile(0.33)
q2 = merged['crime_count'].quantile(0.66)

def classify_safety(val):
    if val <= q1:
        return "Low"
    elif val <= q2:
        return "Medium"
    else:
        return "High"

merged['safety_level'] = merged['crime_count'].apply(classify_safety)

# Show matching stats
real_data = (~merged.get('synthetic', False)).sum()
synthetic_data = merged.get('synthetic', False).sum()

# ---------------------------
# MAIN MAP - ALL DISTRICTS COLORED
# ---------------------------
st.subheader("🗺️ National Crime Heatmap")

# Map view selector
map_view = st.radio(
    "Select Map View:",
    ["Street Map", "Dark Mode", "Satellite"],
    horizontal=True,
    label_visibility="collapsed"
)

def create_main_map(merged_json, name_col, view_type="Street Map"):
    """Create optimized choropleth map with different tile options"""
    
    # Define tile layers based on view type
    if view_type == "Dark Mode":
        tiles = "CartoDB dark_matter"
    elif view_type == "Satellite":
        tiles = None  # We'll add custom satellite tiles
    else:
        tiles = "OpenStreetMap"
    
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles=tiles,
        prefer_canvas=True
    )
    
    # Add satellite imagery if selected
    if view_type == "Satellite":
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
    
    def style_function(feature):
        safety = feature['properties'].get('safety_level', 'Low')
        colors = {
            'Low': '#4caf50',
            'Medium': '#ffc107',
            'High': '#f44336'
        }
        return {
            'fillColor': colors.get(safety, '#4caf50'),
            'color': '#ffffff',
            'weight': 0.5,
            'fillOpacity': 0.7
        }
    
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
            fields=[name_col, 'crime_count', 'safety_level'],
            aliases=['District:', 'Crime Count:', 'Safety Level:'],
            localize=True,
            max_width=300
        )
    ).add_to(m)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; padding:10px; border:2px solid grey; border-radius:5px;">
    <p><strong>Crime Risk Level</strong></p>
    <p><span style="color:#4caf50;">⬤</span> Low Risk</p>
    <p><span style="color:#ffc107;">⬤</span> Medium Risk</p>
    <p><span style="color:#f44336;">⬤</span> High Risk</p>
    <p style="margin-top:10px; font-size:11px; color:#666;">
    💡 Click districts for details<br/>
    Hover for quick info
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen button
    Fullscreen(
        position='topright',
        title='Enter fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    return m

# Create and display main map
with st.spinner("Rendering map..."):
    merged_json = json.loads(merged.to_json())
    main_map = create_main_map(merged_json, name_col, map_view)
    st_folium(main_map, width=1200, height=600, returned_objects=[])

# Show district data summary
st.markdown("---")
st.subheader("📊 District Crime Data Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Districts", len(merged))
    low_count = (merged['safety_level'] == 'Low').sum()
    medium_count = (merged['safety_level'] == 'Medium').sum()
    high_count = (merged['safety_level'] == 'High').sum()
    
    st.metric("Low Risk Districts", low_count)
    st.metric("Medium Risk Districts", medium_count)
    st.metric("High Risk Districts", high_count)

with col2:
    st.write("**Top 5 Highest Crime Districts:**")
    top_districts = merged.nlargest(5, 'crime_count')[[name_col, 'crime_count', 'safety_level']]
    for idx, row in top_districts.iterrows():
        st.write(f"• {row[name_col]}: {int(row['crime_count']):,} crimes ({row['safety_level']})")

# Searchable data table
st.subheader("🔍 Search All Districts")
search_term = st.text_input("Search district name:", placeholder="Type to filter...")

display_df = merged[[name_col, 'crime_count', 'safety_level']].copy()
display_df.columns = ['District Name', 'Crime Count', 'Safety Level']
display_df = display_df.sort_values('Crime Count', ascending=False)

if search_term:
    display_df = display_df[display_df['District Name'].str.contains(search_term, case=False, na=False)]

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
# DISTANCE CALCULATOR BETWEEN DISTRICTS
# ---------------------------
st.markdown("---")
st.subheader("📏 Calculate Distance Between Districts")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    district1 = st.selectbox(
        "Select First District:",
        options=sorted(merged[name_col].unique()),
        key="district1"
    )

with col2:
    district2 = st.selectbox(
        "Select Second District:",
        options=sorted(merged[name_col].unique()),
        key="district2"
    )

with col3:
    st.write("")
    st.write("")
    calc_distance = st.button("Calculate 📍", type="primary")

if calc_distance and district1 and district2:
    if district1 == district2:
        st.warning("⚠️ Please select two different districts")
    else:
        # Get district data
        dist1_data = merged[merged[name_col] == district1].iloc[0]
        dist2_data = merged[merged[name_col] == district2].iloc[0]
        
        # Calculate centroids
        centroid1 = dist1_data.geometry.centroid
        centroid2 = dist2_data.geometry.centroid
        
        # Calculate distance
        distance_km = geodesic(
            (centroid1.y, centroid1.x),
            (centroid2.y, centroid2.x)
        ).km
        
        # Display results
        st.success(f"📏 Distance: **{distance_km:.2f} km** ({distance_km/1.609:.2f} miles)")
        
        # Show comparison table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{district1}**")
            st.write(f"Crime Count: {int(dist1_data['crime_count']):,}")
            st.write(f"Safety Level: {dist1_data['safety_level']}")
            color1 = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
            st.write(f"Risk: {color1.get(dist1_data['safety_level'], '⚫')}")
        
        with col2:
            st.markdown(f"**{district2}**")
            st.write(f"Crime Count: {int(dist2_data['crime_count']):,}")
            st.write(f"Safety Level: {dist2_data['safety_level']}")
            color2 = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
            st.write(f"Risk: {color2.get(dist2_data['safety_level'], '⚫')}")
        
        # Create visualization map
        st.subheader("🗺️ Districts Location Map")
        
        distance_map_view = st.radio(
            "Select Map View:",
            ["Street Map", "Dark Mode", "Satellite"],
            horizontal=True,
            key="distance_map_view"
        )
        
        # Define tile layers
        if distance_map_view == "Dark Mode":
            tiles = "CartoDB dark_matter"
        elif distance_map_view == "Satellite":
            tiles = None
        else:
            tiles = "OpenStreetMap"
        
        # Calculate center point between districts
        center_lat = (centroid1.y + centroid2.y) / 2
        center_lon = (centroid1.x + centroid2.x) / 2
        
        dist_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles=tiles
        )
        
        # Add satellite imagery if selected
        if distance_map_view == "Satellite":
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(dist_map)
        
        # Add markers for both districts
        folium.Marker(
            location=[centroid1.y, centroid1.x],
            popup=f"<b>{district1}</b><br>Crime: {int(dist1_data['crime_count']):,}<br>Safety: {dist1_data['safety_level']}",
            tooltip=district1,
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(dist_map)
        
        folium.Marker(
            location=[centroid2.y, centroid2.x],
            popup=f"<b>{district2}</b><br>Crime: {int(dist2_data['crime_count']):,}<br>Safety: {dist2_data['safety_level']}",
            tooltip=district2,
            icon=folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
        ).add_to(dist_map)
        
        # Draw line between districts
        folium.PolyLine(
            locations=[
                [centroid1.y, centroid1.x],
                [centroid2.y, centroid2.x]
            ],
            color='purple',
            weight=3,
            opacity=0.7,
            popup=f"Distance: {distance_km:.2f} km"
        ).add_to(dist_map)
        
        # Add fullscreen
        Fullscreen(
            position='topright',
            title='Enter fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(dist_map)
        
        st_folium(dist_map, width=1200, height=500, returned_objects=[])

# ---------------------------
# POLICE STATION DATA
# ---------------------------
# Sample police station data - can be extended with real data from data.gov.in or other sources
POLICE_STATIONS = {
    "Bangalore Urban": [
        {"name": "Cubbon Park Police Station", "address": "Kumara Krupa Rd, Bengaluru - 560001", "phone": "080-22942222"},
        {"name": "Koramangala Police Station", "address": "80 Feet Rd, Koramangala, Bengaluru - 560095", "phone": "080-25533506"},
        {"name": "Whitefield Police Station", "address": "Whitefield Main Rd, Bengaluru - 560066", "phone": "080-28452522"},
    ],
    "Mumbai": [
        {"name": "Colaba Police Station", "address": "Arthur Bunder Rd, Colaba, Mumbai - 400005", "phone": "022-22020111"},
        {"name": "Bandra Police Station", "address": "Bandra West, Mumbai - 400050", "phone": "022-26422222"},
        {"name": "Andheri Police Station", "address": "Andheri West, Mumbai - 400058", "phone": "022-26392222"},
    ],
    "New Delhi": [
        {"name": "Connaught Place Police Station", "address": "Parliament Street, New Delhi - 110001", "phone": "011-23742740"},
        {"name": "Saket Police Station", "address": "Saket District Centre, New Delhi - 110017", "phone": "011-26854930"},
        {"name": "Vasant Vihar Police Station", "address": "Vasant Vihar, New Delhi - 110057", "phone": "011-26142450"},
    ],
    "Kolkata": [
        {"name": "Park Street Police Station", "address": "Park Street, Kolkata - 700016", "phone": "033-22297750"},
        {"name": "New Market Police Station", "address": "Lindsay Street, Kolkata - 700087", "phone": "033-22487524"},
        {"name": "Alipore Police Station", "address": "Alipore Rd, Kolkata - 700027", "phone": "033-24791010"},
    ],
    "Chennai": [
        {"name": "Egmore Police Station", "address": "Gandhi Irwin Rd, Egmore, Chennai - 600008", "phone": "044-28190400"},
        {"name": "T Nagar Police Station", "address": "South Usman Rd, T Nagar, Chennai - 600017", "phone": "044-24345155"},
        {"name": "Mylapore Police Station", "address": "Luz Church Rd, Mylapore, Chennai - 600004", "phone": "044-24981234"},
    ],
    # Add default for districts without specific data
    "_default": [
        {"name": "District Police Station", "address": "Contact local authorities for exact location", "phone": "100 (Emergency)"},
    ]
}

def get_police_stations(district_name):
    """Get police stations for a district, with fallback to default"""
    # Try exact match
    if district_name in POLICE_STATIONS:
        return POLICE_STATIONS[district_name]
    
    # Try partial match
    for key in POLICE_STATIONS.keys():
        if key.lower() in district_name.lower() or district_name.lower() in key.lower():
            return POLICE_STATIONS[key]
    
    # Return default
    return POLICE_STATIONS["_default"]

# ---------------------------
# LOCATION SEARCH - WITH DROPDOWN
# ---------------------------
st.markdown("---")
st.subheader("🔍 Search Location Safety")

# Create tabs for different search methods
search_tab1, search_tab2 = st.tabs(["Select District", "Enter Location"])

with search_tab1:
    st.write("**Select a district to view safety information and nearby police stations:**")
    
    selected_district = st.selectbox(
        "Choose District:",
        options=sorted(merged[name_col].unique()),
        key="district_selector"
    )
    
    search_by_dropdown = st.button("🔍 View District Info", type="primary", key="dropdown_search")
    
    if search_by_dropdown and selected_district:
        # Get district data
        district_row = merged[merged[name_col] == selected_district].iloc[0]
        
        # Get centroid for location
        centroid = district_row.geometry.centroid
        lat, lon = centroid.y, centroid.x
        
        st.success(f"📍 District Center: {lat:.4f}, {lon:.4f}")
        
        # Display district info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("District", selected_district)
        with col2:
            safety = district_row['safety_level']
            color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
            st.metric("Safety Level", f"{safety} {color.get(safety, '🟢')}")
        with col3:
            st.metric("Crime Count", int(district_row['crime_count']))
        
        # Police Station Information
        st.markdown("---")
        st.subheader("🚔 Nearby Police Stations")
        
        police_stations = get_police_stations(selected_district)
        
        for idx, station in enumerate(police_stations, 1):
            with st.expander(f"**{idx}. {station['name']}**", expanded=(idx == 1)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"📍 **Address:** {station['address']}")
                    st.write(f"📞 **Phone:** {station['phone']}")
                with col2:
                    st.write("**Quick Actions:**")
                    st.write("☎️ Emergency: 100")
                    st.write("🚑 Ambulance: 102")
                    st.write("🚒 Fire: 101")
        
        # Map visualization
        st.subheader("📍 District Location Map")
        
        district_map_view = st.radio(
            "Select Map View:",
            ["Street Map", "Dark Mode", "Satellite"],
            horizontal=True,
            key="district_map_view"
        )
        
        # Define tile layers
        if district_map_view == "Dark Mode":
            tiles = "CartoDB dark_matter"
        elif district_map_view == "Satellite":
            tiles = None
        else:
            tiles = "OpenStreetMap"
        
        district_map = folium.Map(
            location=[lat, lon],
            zoom_start=11,
            tiles=tiles
        )
        
        # Add satellite imagery if selected
        if district_map_view == "Satellite":
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(district_map)
        
        # Add fullscreen button
        Fullscreen(
            position='topright',
            title='Enter fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(district_map)
        
        # Add district boundary
        def style_district(feature):
            safety = feature['properties'].get('safety_level', 'Low')
            colors = {'Low': '#4caf50', 'Medium': '#ffc107', 'High': '#f44336'}
            return {
                'fillColor': colors.get(safety, '#4caf50'),
                'color': '#ffffff',
                'weight': 2,
                'fillOpacity': 0.5
            }
        
        district_geojson = merged[merged[name_col] == selected_district].copy()
        if district_geojson.crs is None:
            district_geojson.set_crs("EPSG:4326", inplace=True)
        elif district_geojson.crs.to_string() != "EPSG:4326":
            district_geojson = district_geojson.to_crs("EPSG:4326")
        
        folium.GeoJson(
            district_geojson,
            style_function=style_district,
            tooltip=folium.GeoJsonTooltip(
                fields=[name_col, 'crime_count', 'safety_level'],
                aliases=['District:', 'Crime Count:', 'Safety:'],
                localize=True
            )
        ).add_to(district_map)
        
        # Add marker for district center
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{selected_district}</b><br>Crime: {int(district_row['crime_count']):,}<br>Safety: {district_row['safety_level']}",
            tooltip=f"{selected_district} Center",
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(district_map)
        
        st_folium(district_map, width=1000, height=500, returned_objects=[])
        
        st.info("💡 **Tip:** For emergencies, always call 100 for police assistance")

with search_tab2:
    st.write("**Enter coordinates or address to find safety information:**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        location_input = st.text_input(
            "Enter location (lat,lon format only):",
            placeholder="e.g., 12.9716,77.5946",
            help="Use lat,lon format. Address geocoding disabled due to API timeout."
        )
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 Search", type="primary", key="coord_search")
    
    if search_button and location_input:
        geolocator = Nominatim(user_agent="crime-map", timeout=10)
        
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
        
        point = Point(lon, lat)
        containing = merged[merged.contains(point)]
        
        if len(containing) > 0:
            district_row = containing.iloc[0]
        else:
            merged_copy = merged.copy()
            merged_copy['centroid'] = merged_copy.geometry.centroid
            merged_copy['dist'] = merged_copy['centroid'].apply(
                lambda c: geodesic((c.y, c.x), (lat, lon)).km
            )
            district_row = merged_copy.nsmallest(1, 'dist').iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("District", district_row[name_col])
        with col2:
            safety = district_row['safety_level']
            color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
            st.metric("Safety Level", f"{safety} {color.get(safety, '🟢')}")
        with col3:
            st.metric("Crime Count", int(district_row['crime_count']))
        
        st.subheader("📍 Local Area Map")
        
        # Map view selector for local map
        local_map_view = st.radio(
            "Select Local Map View:",
            ["Street Map", "Dark Mode", "Satellite"],
            horizontal=True,
            key="local_map_view"
        )
        
        # Define tile layers
        if local_map_view == "Dark Mode":
            tiles = "CartoDB dark_matter"
        elif local_map_view == "Satellite":
            tiles = None
        else:
            tiles = "OpenStreetMap"
        
        local_map = folium.Map(
            location=[lat, lon],
            zoom_start=12,
            tiles=tiles
        )
        
        # Add satellite imagery if selected
        if local_map_view == "Satellite":
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(local_map)
        
        # Add fullscreen button
        Fullscreen(
            position='topright',
            title='Enter fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(local_map)
        
        folium.Marker(
            location=[lat, lon],
            popup=f"📍 Your Location<br>{lat:.4f}, {lon:.4f}",
            tooltip="You are here",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(local_map)
        
        merged_nearby = merged.copy()
        merged_nearby['centroid'] = merged_nearby.geometry.centroid
        merged_nearby['dist_km'] = merged_nearby['centroid'].apply(
            lambda c: geodesic((c.y, c.x), (lat, lon)).km
        )
        nearby_filtered = merged_nearby[merged_nearby['dist_km'] <= 20].copy()
        
        # Keep only necessary columns for display and remove non-serializable columns
        display_columns = [name_col, 'crime_count', 'safety_level', 'dist_km', 'geometry']
        nearby = nearby_filtered[display_columns].copy()
        
        # Ensure CRS is set to WGS84 (EPSG:4326) for folium compatibility
        if nearby.crs is None:
            nearby.set_crs("EPSG:4326", inplace=True)
        elif nearby.crs.to_string() != "EPSG:4326":
            nearby = nearby.to_crs("EPSG:4326")
        
        def style_nearby(feature):
            safety = feature['properties'].get('safety_level', 'Low')
            colors = {'Low': '#4caf50', 'Medium': '#ffc107', 'High': '#f44336'}
            return {
                'fillColor': colors.get(safety, '#4caf50'),
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
        
        st.subheader("📋 Nearby Districts")
        nearby_display = nearby_filtered[[name_col, 'crime_count', 'safety_level', 'dist_km']].copy()
        nearby_display.columns = ['District', 'Crime Count', 'Safety Level', 'Distance (km)']
        nearby_display = nearby_display.sort_values('Distance (km)').head(10)
        st.dataframe(nearby_display, use_container_width=True)
        
        st.info("🚨 **Emergency Services**: Police: 100 | Ambulance: 102 | Fire: 101")

# Footer
st.markdown("---")
st.markdown("""
**Notes:**
- All districts now have crime data (synthetic low values generated for missing districts)
- Crime scores are relative rankings based on your dataset
- Safety levels: Low/Medium/High are determined by quantiles across all districts
- For emergencies, always call local emergency services
""")
