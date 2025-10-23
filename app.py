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
warnings.filterwarnings("ignore")

# Import folium plugins
try:
    from folium.plugins import Fullscreen, MeasureControl, MarkerCluster
except ImportError:
    # Fallback if plugins not available
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

def get_region_specific_contacts(district_name, state_name=None):
    """Get emergency contacts specific to a region"""
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
# UTIL: normalize district strings
# ---------------------------
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    
    # Handle common variations in Indian district names
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
    
    # keep alnum and spaces
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s

# ---------------------------
# Load & aggregate all CSVs in data/
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
        except Exception:
            try:
                df = pd.read_csv(f, encoding="latin1", low_memory=False)
            except Exception as e:
                failed.append((f, str(e)))
                continue

        # locate district-like column
        cols = [c for c in df.columns]
        district_col = None
        for c in cols:
            lc = c.lower()
            if "district" in lc or "district name" in lc or ("name" == lc and len(cols)>1):
                district_col = c
                break
        if district_col is None:
            district_col = cols[0]  # fallback to first column

        # sum numeric columns per row as file-level crime count
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            # try to coerce other columns (like year columns)
            numeric_candidates = []
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
                    numeric_candidates.append(c)
                except:
                    pass
            numeric_cols = numeric_candidates

        if numeric_cols:
            # create per-row total
            df['_file_total'] = df[numeric_cols].apply(lambda row: pd.to_numeric(row, errors='coerce').fillna(0).sum(), axis=1)
        else:
            # as last resort, count 1 for each row (presence)
            df['_file_total'] = 1

        # keep district and _file_total
        small = df[[district_col, '_file_total']].copy()
        small.columns = ['district_raw', 'file_total']
        small['district_norm'] = small['district_raw'].apply(normalize_name)
        aggregated_rows.append(small)

    if not aggregated_rows:
        return pd.DataFrame(), files, failed

    all_small = pd.concat(aggregated_rows, ignore_index=True)
    # group by normalized district and sum
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
# Load GeoJSON (districts) - Try multiple sources
# ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson_from_url_or_upload(uploaded_file, urls):
    if uploaded_file is not None:
        gj = json.load(uploaded_file)
        gdf = gpd.GeoDataFrame.from_features(gj["features"])
        return gdf
    
    # Try multiple URLs
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            gj = r.json()
            gdf = gpd.GeoDataFrame.from_features(gj["features"])
            return gdf
        except Exception as e:
            continue
    
    raise Exception("All sources failed")

try:
    gdf_districts = load_geojson_from_url_or_upload(uploaded_geo, DEFAULT_GEOJSON_URLS)
except Exception as e:
    st.error("Could not load any remote GeoJSON sources. Please upload a district-level GeoJSON file.")
    st.write("You can download district boundaries from:")
    st.write("- https://github.com/geohacker/india/blob/master/district/india_district.geojson")
    st.write("- https://github.com/datta07/INDIAN-SHAPEFILES")
    st.stop()

# ---------------------------
# Identify district name column in geojson - Improved detection
# ---------------------------
name_col = None

# Check for hierarchical naming patterns (NAME_2 is usually district level in GADM data)
hierarchical_cols = ['NAME_2', 'NAME_3', 'DTNAME', 'name_2', 'district_name']
for c in hierarchical_cols:
    if c in gdf_districts.columns:
        # Check if this column has diverse values (not all same like "India")
        unique_vals = gdf_districts[c].nunique()
        if unique_vals > 10:  # Should have many different district names
            name_col = c
            break

if name_col is None:
    # Try common district name patterns
    possible_name_cols = ['NAME', 'name', 'district', 'District', 'DISTRICT', 'dtname']
    for c in possible_name_cols:
        if c in gdf_districts.columns:
            unique_vals = gdf_districts[c].nunique()
            if unique_vals > 10:
                name_col = c
                break

if name_col is None:
    # Last resort - find any column with diverse values that might be districts
    for c in gdf_districts.columns:
        if gdf_districts[c].dtype == 'object':  # String column
            unique_vals = gdf_districts[c].nunique()
            if unique_vals > 50:  # Likely to be district names if many unique values
                name_col = c
                break

if name_col is None:
    st.error("Could not identify a district-name column in the GeoJSON. Available columns: " + str(list(gdf_districts.columns)))
    st.write("Column details:")
    for col in gdf_districts.columns:
        if gdf_districts[col].dtype == 'object':
            unique_count = gdf_districts[col].nunique()
            sample_vals = gdf_districts[col].dropna().head(5).tolist()
            st.write(f"- **{col}**: {unique_count} unique values, samples: {sample_vals}")
    st.write("Please upload a geojson with a proper district name column.")
    st.stop()

# create normalized name column
gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# ---------------------------
# Merge crime data into GeoDataFrame
# ---------------------------
# Left merge on normalized name
merged = gdf_districts.merge(crime_agg[['district_norm','crime_total']], on='district_norm', how='left')
merged['crime_total'] = merged['crime_total'].fillna(0)

# Show matching stats
matched = (merged['crime_total'] > 0).sum()
total = len(merged)
missing = total - matched
st.info(f"District matching: {matched}/{total} districts have crime data. {missing} districts show 0 (no match found).")

if missing > 0.5 * total:
    st.warning("⚠️ Many districts are unmatched. Consider:")
    st.write("1. Check if your CSV district names match the GeoJSON names")
    st.write("2. Upload a different GeoJSON file that matches your data")
    
    # Show some examples of unmatched vs CSV names
    unmatched_sample = merged[merged['crime_total'] == 0][name_col].head(10).tolist()
    csv_sample = crime_agg['district_example'].head(10).tolist()
    
    # Show normalized versions for debugging
    unmatched_norm = merged[merged['crime_total'] == 0]['district_norm'].head(10).tolist()
    csv_norm = crime_agg['district_norm'].head(10).tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sample GeoJSON districts (unmatched):**")
        for orig, norm in zip(unmatched_sample, unmatched_norm):
            st.write(f"'{orig}' → '{norm}'")
    
    with col2:
        st.write("**Sample CSV districts:**") 
        for orig, norm in zip(csv_sample, csv_norm):
            st.write(f"'{orig}' → '{norm}'")
    
    # Show if there are any matches between the normalized names
    geo_norms = set(merged['district_norm'].tolist())
    csv_norms = set(crime_agg['district_norm'].tolist())
    common = geo_norms.intersection(csv_norms)
    st.write(f"**Common normalized names found:** {len(common)} out of {len(csv_norms)} CSV districts")
    if len(common) > 0 and len(common) < 20:
        st.write("Sample matches:", list(common)[:10])

# classify into Low/Medium/High using quantiles (only for non-zero values)
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
# Make color scale
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
# Draw national choropleth (folium) - Enhanced Google Maps style
# ---------------------------
st.subheader("🗺️ India — Interactive Crime Safety Map")

# Map configuration
col1, col2 = st.columns([1, 1])
with col1:
    show_pois = st.checkbox("Show Police Stations", value=True)
with col2:
    danger_threshold = st.slider("Danger Zone Threshold", 0.1, 1.0, 0.7, 0.1,
                                help="Adjust what constitutes a 'danger zone'")

# Create main map
m = folium.Map(
    location=[20.5937, 78.9629],  # Center of India
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Add layer control
folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite View'
).add_to(m)
folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

# Enhanced styling function with danger zones
def style_function(feature):
    val = feature['properties'].get('crime_total', 0)
    safety_level = feature['properties'].get('safety_level', 'No Data')
    
    # Determine if it's a danger zone based on threshold
    max_crime = merged['crime_total'].max()
    is_danger_zone = val > (max_crime * danger_threshold) if max_crime > 0 else False
    
    if safety_level == "No Data":
        color_fill = '#f0f0f0'  # Light gray
        border_color = '#cccccc'
        border_weight = 0.5
    elif is_danger_zone:
        color_fill = '#ff0000'  # Bright red for danger zones
        border_color = '#cc0000'
        border_weight = 2.0
    elif safety_level == "High":
        color_fill = '#ff6b6b'  # Red
        border_color = '#e55353'
        border_weight = 1.0
    elif safety_level == "Medium":
        color_fill = '#ffd93d'  # Yellow
        border_color = '#ffcc02'
        border_weight = 0.8
    else:  # Low
        color_fill = '#6bcf7f'  # Green
        border_color = '#51b364'
        border_weight = 0.6
    
    return {
        'fillColor': color_fill,
        'color': border_color,
        'weight': border_weight,
        'fillOpacity': 0.7,
        'opacity': 1
    }

# Enhanced tooltip with more information
def create_tooltip_html(row):
    district = row[name_col]
    crime_count = int(row['crime_total'])
    safety = row['safety_level']
    
    # Safety emoji
    emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "No Data": "⚫"}.get(safety, "⚫")
    
    # Determine danger status
    max_crime = merged['crime_total'].max()
    is_danger = crime_count > (max_crime * danger_threshold) if max_crime > 0 else False
    danger_text = "🚨 DANGER ZONE" if is_danger else ""
    
    html = f"""
    <div style="font-family: Arial; font-size: 12px; min-width: 200px;">
        <b style="font-size: 14px;">{district}</b><br>
        <hr style="margin: 5px 0;">
        Safety Level: <b>{safety} {emoji}</b><br>
        Crime Count: <b>{crime_count:,}</b><br>
        {f'<span style="color: red; font-weight: bold;">{danger_text}</span><br>' if is_danger else ''}
        <small style="color: #666;">Click for more details</small>
    </div>
    """
    return html

# Add district polygons with enhanced styling
district_layer = folium.FeatureGroup(name="🏛️ Districts")
for _, row in merged.iterrows():
    # Create popup with detailed information including emergency contacts
    region_contacts = get_region_specific_contacts(row[name_col])
    emergency_info = ""
    for service, number in list(region_contacts.items())[:3]:  # Show first 3 emergency numbers
        emergency_info += f"<tr><td>{service}:</td><td><b>{number}</b></td></tr>"
    
    popup_html = f"""
    <div style="font-family: Arial; max-width: 300px;">
        <h4 style="margin: 0 0 10px 0; color: #2c3e50;">{row[name_col]}</h4>
        <table style="width: 100%; font-size: 12px; margin-bottom: 10px;">
            <tr><td><b>Safety Level:</b></td><td>{row['safety_level']}</td></tr>
            <tr><td><b>Crime Count:</b></td><td>{int(row['crime_total']):,}</td></tr>
            <tr><td><b>Risk Category:</b></td><td>{'High Risk' if row['crime_total'] > (merged['crime_total'].max() * danger_threshold) else 'Moderate/Low Risk'}</td></tr>
        </table>
        <hr style="margin: 10px 0;">
        <h5 style="margin: 5px 0; color: #e74c3c;">🚨 Emergency Contacts:</h5>
        <table style="width: 100%; font-size: 11px;">
            {emergency_info}
        </table>
        <hr style="margin: 10px 0;">
        <small style="color: #7f8c8d;">Based on aggregated crime data from multiple sources</small>
    </div>
    """
    
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, row=row: style_function({'properties': row}),
        popup=folium.Popup(popup_html, max_width=350),
        tooltip=create_tooltip_html(row)
    ).add_to(district_layer)

district_layer.add_to(m)

# Add police stations layer if enabled
if show_pois:
    with st.spinner("Loading police stations across India..."):
        # Query major police stations in major cities
        major_cities = [
            {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
            {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
            {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
            {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
            {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
            {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
        ]
        
        police_layer = folium.FeatureGroup(name="🚔 Police Stations")
        
        # Add some sample police stations for demonstration
        sample_stations = [
            {"name": "Delhi Police Headquarters", "lat": 28.6289, "lon": 77.2065, "city": "Delhi"},
            {"name": "Mumbai Police Commissioner Office", "lat": 18.9220, "lon": 72.8347, "city": "Mumbai"},
            {"name": "Bangalore City Police", "lat": 12.9716, "lon": 77.5946, "city": "Bangalore"},
            {"name": "Chennai Police Station", "lat": 13.0827, "lon": 80.2707, "city": "Chennai"},
            {"name": "Kolkata Police HQ", "lat": 22.5726, "lon": 88.3639, "city": "Kolkata"},
        ]
        
        for station in sample_stations:
            popup_html = f"""
            <div style="font-family: Arial;">
                <h4 style="margin: 0; color: #c0392b;">🚔 {station['name']}</h4>
                <hr style="margin: 5px 0;">
                <b>Location:</b> {station['city']}<br>
                <b>Coordinates:</b> {station['lat']:.4f}, {station['lon']:.4f}<br>
                <small style="color: #7f8c8d;">Emergency: 100</small>
            </div>
            """
            
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"🚔 {station['name']}",
                icon=folium.Icon(color='red', icon='shield-alt', prefix='fa')
            ).add_to(police_layer)
        
        police_layer.add_to(m)

# Add legend
legend_html = f"""
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 180px; 
     background-color: white; border:2px solid grey; z-index:9999; 
     font-size:12px; padding: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.2);
     border-radius: 5px;">
  
<h4 style="margin: 0 0 10px 0; text-align: center;">🛡️ Safety Legend</h4>
<p style="margin: 5px 0;"><span style="color: #6bcf7f;">🟢</span> <b>Low Risk</b> - Safe areas</p>
<p style="margin: 5px 0;"><span style="color: #ffd93d;">🟡</span> <b>Medium Risk</b> - Moderate caution</p>
<p style="margin: 5px 0;"><span style="color: #ff6b6b;">🔴</span> <b>High Risk</b> - Exercise caution</p>
<p style="margin: 5px 0;"><span style="color: #ff0000;">🚨</span> <b>Danger Zone</b> - High alert</p>
<p style="margin: 5px 0;"><span style="color: #f0f0f0;">⚫</span> <b>No Data</b> - Insufficient info</p>
<hr style="margin: 8px 0;">
<p style="margin: 5px 0; font-size: 10px;"><span style="color: red;">🚔</span> Police Stations</p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl().add_to(m)

# Add fullscreen button
try:
    from folium.plugins import Fullscreen
    Fullscreen().add_to(m)
except ImportError:
    pass

# Add measure tool
try:
    from folium.plugins import MeasureControl
    MeasureControl().add_to(m)
except ImportError:
    pass

# Display the main map
st_data = st_folium(m, width=1200, height=700)

# ---------------------------
# Sidebar: location input
# ---------------------------
st.sidebar.header("Find safety near your location")
loc_input = st.sidebar.text_input("Provide me info about the place you are in exactly", placeholder="Type an address or 'lat,lon'")

# helper: geocode or parse lat,lon
geolocator = Nominatim(user_agent="india-crime-heatmap-app", timeout=10)

def parse_or_geocode(s):
    s = s.strip()
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 2:
            try:
                lat = float(parts[0]); lon = float(parts[1])
                return lat, lon, None
            except:
                pass
    # else geocode
    try:
        loc = geolocator.geocode(s + ", India")  # Add India for better results
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except Exception as e:
        return None
    return None

# helper: find containing district polygon
def find_district_for_point(point_latlon, merged_gdf):
    lat, lon = point_latlon
    pt = Point(lon, lat)
    # check containment
    contains = merged_gdf[merged_gdf.contains(pt)]
    if len(contains) > 0:
        return contains.iloc[0]
    # else find nearest by centroid distance
    merged_gdf = merged_gdf.copy()
    merged_gdf['centroid'] = merged_gdf.geometry.centroid
    distances = merged_gdf['centroid'].apply(lambda c: geodesic((c.y, c.x), (lat, lon)).km)
    merged_gdf['dist_km_to_point'] = distances
    nearest = merged_gdf.loc[merged_gdf['dist_km_to_point'].idxmin()]
    return nearest

# helper: find districts within radius_km
def districts_within_radius(point_latlon, gdf, radius_km=10):
    lat, lon = point_latlon
    gdf = gdf.copy()
    gdf['centroid'] = gdf.geometry.centroid
    gdf['dist_km'] = gdf['centroid'].apply(lambda c: geodesic((c.y,c.x),(lat,lon)).km)
    near = gdf[gdf['dist_km'] <= radius_km].copy()
    return near.sort_values('dist_km')

# helper: Overpass query for police/residential/shelters within radius
def overpass_query_pois(lat, lon, radius_m=5000):
    overpass_url = "https://overpass-api.de/api/interpreter"
    # search police nodes/ways, residential ways, and shelter/helpline-like amenities
    query = f"""
    [out:json][timeout:40];
    (
      node["amenity"="police"](around:{radius_m},{lat},{lon});
      node["amenity"="police_station"](around:{radius_m},{lat},{lon});
      way["amenity"="police"](around:{radius_m},{lat},{lon});
      node["healthcare"~"hospital|clinic|health_centre"](around:{radius_m},{lat},{lon});
      node["building"="residential"](around:{radius_m},{lat},{lon});
      way["landuse"="residential"](around:{radius_m},{lat},{lon});
      node["amenity"="shelter"](around:{radius_m},{lat},{lon});
    );
    out center 50;
    """
    try:
        res = requests.get(overpass_url, params={'data': query}, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get('elements', [])
    except Exception as e:
        st.warning(f"Overpass API error: {str(e)}")
        return []

# On submit:
if loc_input:
    with st.spinner("Geocoding and analyzing..."):
        p = parse_or_geocode(loc_input)
        if p is None:
            st.error("Could not geocode or parse the input. Try 'City, State' or 'lat,lon' (e.g. 28.7041,77.1025).")
        else:
            lat, lon, resolved = p[0], p[1], (p[2] if len(p)>2 else None)
            st.sidebar.markdown(f"**Resolved:** {resolved if resolved else ''} ({lat:.5f}, {lon:.5f})")
            
            # find district
            nearest_district_row = find_district_for_point((lat, lon), merged.copy())
            st.write("### Location analysis")
            try:
                district_name = nearest_district_row[name_col]
                safety = nearest_district_row['safety_level']
                crime_count = int(nearest_district_row['crime_total'])
                
                if safety == "No Data":
                    st.warning(f"You are in/near **{district_name}** — **No crime data available** for this district.")
                else:
                    color = "🟢" if safety == "Low" else ("🟡" if safety == "Medium" else "🔴")
                    st.success(f"You are in/near **{district_name}** — Safety Level: **{safety}** {color} (crime count: {crime_count})")
                
                # Display location-specific emergency contacts
                st.subheader("📞 Emergency Contacts for Your Area")
                region_contacts = get_region_specific_contacts(district_name)
                
                # Create emergency contact cards
                emergency_cols = st.columns(3)
                contact_items = list(region_contacts.items())
                
                for i, (service, number) in enumerate(contact_items[:9]):  # Show up to 9 contacts
                    with emergency_cols[i % 3]:
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #dc3545; margin-bottom: 10px;">
                            <h6 style="margin: 0; color: #dc3545;">{service}</h6>
                            <h4 style="margin: 5px 0 0 0; color: #212529;">{number}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Quick dial buttons for essential services
                st.markdown("### 🚨 Quick Emergency Actions")
                emergency_actions = st.columns(4)
                
                with emergency_actions[0]:
                    if st.button("🚔 Call Police (100)", type="primary", key="quick_police"):
                        st.success("Emergency Number: 100\n\nDial immediately for police assistance")
                
                with emergency_actions[1]:
                    if st.button("🚑 Call Ambulance (102)", type="secondary", key="quick_ambulance"):
                        st.success("Emergency Number: 102\n\nDial for medical emergency")
                
                with emergency_actions[2]:
                    if st.button("🔥 Fire Service (101)", type="secondary", key="quick_fire"):
                        st.success("Emergency Number: 101\n\nDial for fire emergency")
                
                with emergency_actions[3]:
                    if st.button("👩 Women Helpline (1091)", type="secondary", key="quick_women"):
                        st.success("Emergency Number: 1091\n\nDial for women's safety assistance")
                        
            except Exception:
                st.info("Could not map to a district polygon; showing nearest districts.")

            # find surrounding districts within radius
            nearby_districts = districts_within_radius((lat,lon), merged.copy(), radius_km=10)
            st.write(f"Districts within 10 km: {len(nearby_districts)}")
            if not nearby_districts.empty:
                nd_display = nearby_districts[[name_col,'crime_total','safety_level','dist_km']].rename(columns={name_col:'District','crime_total':'Crime Count','safety_level':'Safety','dist_km':'Distance_km'})
                st.dataframe(nd_display.head(20))

            # Overpass POIs
            with st.spinner("Fetching nearby points of interest..."):
                elements = overpass_query_pois(lat, lon, radius_m=NEAREST_RADIUS_KM*1000)
            st.write(f"Found {len(elements)} nearby POI elements (police/residential/shelters) within {NEAREST_RADIUS_KM} km via Overpass API.")

            # Build a focused local area map
            st.subheader("🔍 Local Safety Analysis Map")
            
            # Local map controls
            local_col1, local_col2 = st.columns(2)
            with local_col1:
                local_map_style = st.selectbox("Local Map Style:", ["Street", "Satellite", "Hybrid"], key="local_style")
            with local_col2:
                search_radius = st.slider("POI Search Radius (km):", 1, 20, 5, key="search_radius")
            
            # Configure local map tiles
            if local_map_style == "Street":
                local_tiles = "OpenStreetMap"
            elif local_map_style == "Satellite":
                local_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            else:  # Hybrid
                local_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            
            m_local = folium.Map(
                location=[lat, lon], 
                zoom_start=12, 
                tiles=local_tiles,
                attr="Esri" if local_map_style != "Street" else "OpenStreetMap"
            )
            
            # Add different tile layers
            folium.TileLayer('OpenStreetMap', name='Street View').add_to(m_local)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite View'
            ).add_to(m_local)
            
            # Draw nearby districts with enhanced styling
            for _, row in nearby_districts.iterrows():
                try:
                    geom = row.geometry
                    lvl = row['safety_level']
                    district_name = row[name_col]
                    crime_count = int(row['crime_total'])
                    
                    if lvl == "No Data":
                        color = "#f0f0f0"
                        border_color = "#cccccc"
                    elif lvl == "High":
                        color = "#ff6b6b"
                        border_color = "#e55353"
                    elif lvl == "Medium":
                        color = "#ffd93d"
                        border_color = "#ffcc02"
                    else:  # Low
                        color = "#6bcf7f"
                        border_color = "#51b364"
                    
                    # Enhanced popup for districts with emergency contacts
                    district_contacts = get_region_specific_contacts(district_name)
                    emergency_list = ""
                    for service, number in list(district_contacts.items())[:5]:
                        emergency_list += f"<tr><td>{service}:</td><td><b>{number}</b></td></tr>"
                    
                    popup_html = f"""
                    <div style="font-family: Arial; max-width: 280px;">
                        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">{district_name}</h4>
                        <table style="width: 100%; font-size: 11px; margin-bottom: 10px;">
                            <tr><td><b>Safety Level:</b></td><td>{lvl}</td></tr>
                            <tr><td><b>Crime Count:</b></td><td>{crime_count:,}</td></tr>
                            <tr><td><b>Distance:</b></td><td>{row['dist_km']:.1f} km</td></tr>
                        </table>
                        <hr style="margin: 8px 0;">
                        <h6 style="margin: 5px 0; color: #e74c3c;">Emergency Contacts:</h6>
                        <table style="width: 100%; font-size: 10px;">
                            {emergency_list}
                        </table>
                    </div>
                    """
                    
                    folium.GeoJson(
                        data=geom.__geo_interface__, 
                        style_function=lambda feat, col=color, border=border_color: {
                            'fillColor': col, 
                            'color': border, 
                            'weight': 1.5, 
                            'fillOpacity': 0.6,
                            'opacity': 1
                        },
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{district_name} - {lvl}"
                    ).add_to(m_local)
                except Exception:
                    pass

            # Add user location with enhanced marker
            folium.CircleMarker(
                location=[lat, lon], 
                radius=12, 
                color="blue", 
                fill=True, 
                fill_opacity=0.9, 
                popup=folium.Popup(f"<b>📍 Your Location</b><br>{lat:.5f}, {lon:.5f}<br><hr><b>Emergency: 100 (Police)</b>", max_width=200),
                tooltip="📍 You are here"
            ).add_to(m_local)
            
            # Add search radius circle
            folium.Circle(
                location=[lat, lon],
                radius=search_radius * 1000,  # Convert km to meters
                color="blue",
                fill=False,
                weight=2,
                opacity=0.5,
                popup=f"Search Radius: {search_radius} km"
            ).add_to(m_local)

            # Enhanced POI markers with clustering
            try:
                marker_cluster = MarkerCluster(name="📍 Points of Interest").add_to(m_local)
            except:
                marker_cluster = m_local  # Fallback if clustering not available
            
            # Re-fetch POIs with updated radius
            elements = overpass_query_pois(lat, lon, radius_m=search_radius*1000)
            
            poi_categories = {
                'police': {'count': 0, 'icon': 'shield-alt', 'color': 'red', 'prefix': 'fa'},
                'hospital': {'count': 0, 'icon': 'plus', 'color': 'blue', 'prefix': 'fa'},
                'residential': {'count': 0, 'icon': 'home', 'color': 'green', 'prefix': 'fa'},
                'shelter': {'count': 0, 'icon': 'bed', 'color': 'purple', 'prefix': 'fa'}
            }
            
            for el in elements:
                # Get coordinates
                if el.get('type') == 'node' and 'lat' in el and 'lon' in el:
                    el_lat, el_lon = el['lat'], el['lon']
                elif 'center' in el:
                    el_lat, el_lon = el['center']['lat'], el['center']['lon']
                else:
                    continue
                
                tags = el.get('tags', {})
                name = tags.get('name', tags.get('operator', 'POI'))
                distance = geodesic((lat, lon), (el_lat, el_lon)).km
                
                # Categorize POI
                amenity = tags.get('amenity', '')
                building = tags.get('building', '')
                landuse = tags.get('landuse', '')
                healthcare = tags.get('healthcare', '')
                
                if amenity in ('police', 'police_station'):
                    category = 'police'
                    icon_config = poi_categories['police']
                elif amenity == 'shelter' or 'shelter' in name.lower():
                    category = 'shelter'
                    icon_config = poi_categories['shelter']
                elif healthcare or amenity in ('hospital', 'clinic'):
                    category = 'hospital'
                    icon_config = poi_categories['hospital']
                elif building == 'residential' or landuse == 'residential':
                    category = 'residential'
                    icon_config = poi_categories['residential']
                else:
                    continue  # Skip unknown POIs
                
                poi_categories[category]['count'] += 1
                
                # Create detailed popup with emergency info
                popup_html = f"""
                <div style="font-family: Arial; max-width: 250px;">
                    <h4 style="margin: 0 0 8px 0; color: #2c3e50;">{name}</h4>
                    <hr style="margin: 5px 0;">
                    <b>Type:</b> {category.title()}<br>
                    <b>Distance:</b> {distance:.2f} km<br>
                    <b>Coordinates:</b> {el_lat:.4f}, {el_lon:.4f}<br>
                    {f"<b>Phone:</b> {tags.get('phone', 'N/A')}<br>" if tags.get('phone') else ''}
                    <hr style="margin: 5px 0;">
                    <b>Emergency Numbers:</b><br>
                    Police: <b>100</b> | Fire: <b>101</b> | Medical: <b>102</b>
                    <hr style="margin: 5px 0;">
                    <small style="color: #7f8c8d;">Source: OpenStreetMap</small>
                </div>
                """
                
                folium.Marker(
                    location=[el_lat, el_lon],
                    popup=folium.Popup(popup_html, max_width=270),
                    tooltip=f"{icon_config['color'].title()} {category.title()}: {name}",
                    icon=folium.Icon(
                        color=icon_config['color'], 
                        icon=icon_config['icon'], 
                        prefix=icon_config['prefix']
                    )
                ).add_to(marker_cluster)

            # Add layer control to local map
            folium.LayerControl().add_to(m_local)
            
            # Add fullscreen button
            try:
                Fullscreen().add_to(m_local)
            except:
                pass
            
            # Display local map
            st_folium(m_local, width=1000, height=600)
            
            # POI Summary Statistics
            st.subheader("📊 Nearby Points of Interest Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚔 Police Stations", poi_categories['police']['count'])
            with col2:
                st.metric("🏥 Healthcare", poi_categories['hospital']['count'])
            with col3:
                st.metric("🏠 Residential Areas", poi_categories['residential']['count'])
            with col4:
                st.metric("🏨 Shelters", poi_categories['shelter']['count'])

            # List nearest police stations (text)
            st.subheader("🚓 Nearest Police Stations")
            nearest_police = []
            for el in elements:
                tags = el.get('tags', {})
                if tags.get('amenity') in ('police','police_station'):
                    if el.get('type') == 'node':
                        latp, lonp = el['lat'], el['lon']
                    elif 'center' in el:
                        latp, lonp = el['center']['lat'], el['center']['lon']
                    else:
                        continue
                    name = tags.get('name', 'Police Station')
                    distance = geodesic((lat, lon), (latp, lonp)).km
                    phone = tags.get('phone', 'Call 100')
                    nearest_police.append((name, distance, phone))

            if nearest_police:
                nearest_police.sort(key=lambda x: x[1])
                for name, dist, phone in nearest_police[:10]:  # show top 10 closest
                    st.write(f"• **{name}** — {dist:.2f} km away | Contact: {phone}")
            else:
                st.write("No nearby police stations found in the area.")
                st.info("In emergency, dial **100** for police assistance anywhere in India.")

# Add emergency preparedness tips
st.markdown("---")
st.subheader("🛡️ Safety & Emergency Preparedness Tips")

safety_cols = st.columns(2)

with safety_cols[0]:
    st.markdown("""
    **🚨 In Case of Emergency:**
    - **Police Emergency:** Dial **100**
    - **Medical Emergency:** Dial **102** 
    - **Fire Emergency:** Dial **101**
    - Stay calm and provide clear location details
    - Keep important contacts saved in your phone
    - Share your location with trusted contacts
    """)

with safety_cols[1]:
    st.markdown("""
    **🛡️ Safety Best Practices:**
    - Be aware of your surroundings
    - Avoid isolated areas, especially at night
    - Keep emergency contacts easily accessible
    - Inform someone about your travel plans
    - Trust your instincts if something feels wrong
    - Use well-lit, populated routes
    """)

st.markdown("---")
st.markdown(
    "**Notes:** \n"
    "1. District-name mismatches are common between CSVs and GeoJSON files. If many districts show zero, check name variations.\n"
    "2. Overpass/Nominatim are free public services and may rate-limit or be slow.\n"
    "3. For production use, consider paid geocoding/mapping services.\n"
    "4. Crime scores are relative - 'Low/Medium/High' are based on quantiles within your dataset.\n"
    "5. Emergency numbers are for India. Always verify local emergency contacts.\n"
    "6. In case of immediate danger, prioritize your safety and call emergency services."
        )
