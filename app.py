# app.py - India Crime Heatmap with Debug
import streamlit as st

# Show loading progress
st.write("🔄 Starting application...")

try:
    import pandas as pd
    st.write("✅ Pandas loaded")
except Exception as e:
    st.error(f"❌ Pandas error: {e}")
    st.stop()

try:
    import geopandas as gpd
    st.write("✅ GeoPandas loaded")
except Exception as e:
    st.error(f"❌ GeoPandas error: {e}")
    st.stop()

try:
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
    st.write("✅ All imports successful")
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("🗺️ India Crime Heatmap - District Level Analysis")

# CONFIGURATION
DATA_FOLDER = "data"
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson",
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/India_Districts.geojson"
]

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

@st.cache_data(show_spinner=False)
def load_and_aggregate_csvs(data_folder):
    """Fast CSV loading with better error handling"""
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame(), []
    
    all_data = []
    
    for idx, f in enumerate(files):
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
    
    if not all_data:
        return pd.DataFrame(), files
    
    combined = pd.concat(all_data, ignore_index=True)
    aggregated = combined.groupby('district_norm', as_index=False)['crime_count'].sum()
    
    return aggregated, files

# Load data
st.write("🔄 Loading crime data...")
crime_agg, csv_files = load_and_aggregate_csvs(DATA_FOLDER)

if crime_agg.empty:
    st.warning(f"⚠️ No CSV files found in `{DATA_FOLDER}`. Using demo data instead.")
    # Create demo data
    crime_agg = pd.DataFrame({
        'district_norm': ['bangalore', 'mumbai', 'new delhi', 'chennai', 'kolkata'],
        'crime_count': [5000, 4500, 4000, 3500, 3000]
    })
    csv_files = ['demo_data']

st.success(f"✅ Loaded {len(csv_files)} CSV files, found {len(crime_agg)} districts")

@st.cache_data(show_spinner=False)
def load_geojson():
    """Load GeoJSON with multiple fallback sources"""
    errors = []
    
    for idx, url in enumerate(GEOJSON_URLS):
        try:
            st.write(f"🔄 Trying GeoJSON source {idx + 1}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            gj = response.json()
            gdf = gpd.GeoDataFrame.from_features(gj['features'])
            st.write(f"✅ Loaded from source {idx + 1}")
            return gdf
        except Exception as e:
            errors.append(f"Source {idx + 1}: {str(e)}")
            st.write(f"❌ Source {idx + 1} failed: {str(e)[:100]}")
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

st.write("🔄 Loading map boundaries...")
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
    st.write("Available columns:", gdf_districts.columns.tolist())
    st.stop()

st.write(f"✅ Using column: {name_col}")

gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# Merge data with synthetic data generation
merged = gdf_districts.merge(
    crime_agg[['district_norm', 'crime_count']], 
    on='district_norm', 
    how='left'
)

# Generate synthetic low crime data for missing districts
districts_without_data = merged['crime_count'].isna()
num_missing = districts_without_data.sum()

if num_missing > 0:
    st.write(f"ℹ️ Generating synthetic data for {num_missing} districts...")
    actual_crimes = crime_agg['crime_count'].values
    low_crime_threshold = np.percentile(actual_crimes, 10)
    
    min_val = 50
    max_val = int(max(500, low_crime_threshold * 0.5))
    
    synthetic_values = np.random.choice(
        range(min_val, max_val + 1), 
        size=num_missing, 
        replace=False if num_missing <= (max_val - min_val + 1) else True
    ).astype(int)
    
    merged.loc[districts_without_data, 'crime_count'] = synthetic_values
    merged.loc[districts_without_data, 'synthetic'] = True
    merged['synthetic'] = merged['synthetic'].fillna(False)

merged['crime_count'] = merged['crime_count'].fillna(1)
merged['crime_count'] = merged['crime_count'].replace(0, 1)
merged['crime_count'] = merged['crime_count'].astype(int)

# Calculate safety levels
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

st.write("✅ Data processing complete!")

# MAIN MAP
st.subheader("🗺️ National Crime Heatmap")

map_view = st.radio(
    "Select Map View:",
    ["Street Map", "Dark Mode", "Satellite"],
    horizontal=True
)

def create_main_map(merged_json, name_col, view_type="Street Map"):
    """Create optimized choropleth map"""
    
    if view_type == "Dark Mode":
        tiles = "CartoDB dark_matter"
    elif view_type == "Satellite":
        tiles = None
    else:
        tiles = "OpenStreetMap"
    
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles=tiles,
        prefer_canvas=True
    )
    
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
        )
    ).add_to(m)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; padding:10px; border:2px solid grey; border-radius:5px;">
    <p><strong>Crime Risk Level</strong></p>
    <p><span style="color:#4caf50;">⬤</span> Low Risk</p>
    <p><span style="color:#ffc107;">⬤</span> Medium Risk</p>
    <p><span style="color:#f44336;">⬤</span> High Risk</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    Fullscreen(
        position='topright',
        title='Enter fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    return m

st.write("🔄 Rendering map...")
merged_json = json.loads(merged.to_json())
main_map = create_main_map(merged_json, name_col, map_view)
st_folium(main_map, width=1200, height=600, returned_objects=[])

st.success("✅ Application loaded successfully!")

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
    height=400
)

st.markdown("---")
st.markdown("""
**Notes:**
- Application loaded successfully
- All districts have crime data (synthetic values for missing data)
- Safety levels: Low/Medium/High based on quantiles
- Emergency: Police 100 | Ambulance 102 | Fire 101
""")
