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

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — district-level (green → yellow → red)")
st.markdown(
    "This app aggregates all CSVs in `data/` (district-wise crime tables), "
    "builds a district-level crime score, plots a choropleth for India, and "
    "lets you check safety around a supplied place (address or `lat,lon`)."
)

# ---------------------------
# PARAMETERS
# ---------------------------
DEFAULT_GEOJSON_URL = "https://raw.githubusercontent.com/datameet/maps/master/Districts/India_districts.geojson"
DATA_FOLDER = "data"
NEAREST_RADIUS_KM = 5  # radius for nearest POIs

# ---------------------------
# UTIL: normalize district strings
# ---------------------------
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
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
        return pd.DataFrame(), files

    all_small = pd.concat(aggregated_rows, ignore_index=True)
    # group by normalized district and sum
    agg = all_small.groupby('district_norm', as_index=False).agg({
        'file_total': 'sum',
        'district_raw': lambda x: x.iloc[0]
    }).rename(columns={'file_total':'crime_total', 'district_raw':'district_example'})

    # Keep original raw example also
    return agg, files, failed

crime_agg, csv_files, failed_reads = load_and_aggregate_csvs(DATA_FOLDER)
if crime_agg.empty:
    st.error(f"No CSVs found or failed to parse files in `{DATA_FOLDER}`. Place your 57 CSVs there.")
    if failed_reads:
        st.write("Failed reads (examples):")
        st.write(failed_reads[:5])
    st.stop()
st.success(f"Loaded and aggregated {len(csv_files)} CSV(s) from `{DATA_FOLDER}`; found {len(crime_agg)} distinct normalized districts.")

st.sidebar.header("Data & GeoJSON options")
st.sidebar.write(f"Detected {len(csv_files)} CSV files in `{DATA_FOLDER}`.")
geo_choice = st.sidebar.radio("GeoJSON source:", ("Download default district GeoJSON", "Upload my geojson"))

uploaded_geo = None
if geo_choice == "Upload my geojson":
    uploaded_geo = st.sidebar.file_uploader("Upload India districts GeoJSON", type=["json","geojson"])
else:
    st.sidebar.write("Default: will try to download district-level geojson from a community repo.")

# ---------------------------
# Load GeoJSON (districts)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson_from_url_or_upload(uploaded_file, url):
    if uploaded_file is not None:
        gj = json.load(uploaded_file)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        gj = r.json()
    gdf = gpd.GeoDataFrame.from_features(gj["features"])
    return gdf

try:
    gdf_districts = load_geojson_from_url_or_upload(uploaded_geo, DEFAULT_GEOJSON_URL)
except Exception as e:
    st.error("Could not load remote GeoJSON. Please upload a district-level GeoJSON file. Error: " + str(e))
    st.stop()

st.write("Sample GeoJSON properties columns detected:", list(gdf_districts.columns)[:10])
st.write("Preview GeoJSON (first 3 rows):")
st.dataframe(gdf_districts.head(3))

# ---------------------------
# Identify district name column in geojson
# ---------------------------
name_col = None
for c in gdf_districts.columns:
    lc = c.lower()
    if "dist" in lc or "name" in lc:
        name_col = c
        break
if name_col is None:
    st.error("Could not identify a district-name column in the GeoJSON. Please upload a geojson with a 'district' or 'name' attribute.")
    st.stop()

# create normalized name column
gdf_districts['district_norm'] = gdf_districts[name_col].apply(normalize_name)

# ---------------------------
# Merge crime data into GeoDataFrame
# ---------------------------
# Left merge on normalized name
merged = gdf_districts.merge(crime_agg[['district_norm','crime_total']], on='district_norm', how='left')
merged['crime_total'] = merged['crime_total'].fillna(0)

# If many NaNs, display hint
missing = merged['crime_total'].isna().sum()
if missing > 0:
    st.info(f"{missing} districts had no matched crime_total → treated as 0. Consider mapping name variants if mismatches are large.")

# classify into Low/Medium/High using quantiles
q1 = merged['crime_total'].quantile(0.33)
q2 = merged['crime_total'].quantile(0.66)

def classify_val(n):
    if n <= q1:
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
colormap = StepColormap(
    colors=["green","yellow","red"],
    index=[vmin, q1, q2, vmax],
    vmin=vmin, vmax=vmax
)
colormap.caption = "Crime count (green = low → red = high)"

# ---------------------------
# Draw national choropleth (folium)
# ---------------------------
st.subheader("India — District-level crime heatmap")
m = folium.Map(location=[22.0,80.0], zoom_start=5, tiles="cartodbpositron")

def style_function(feature):
    val = feature['properties'].get('crime_total', 0)
    return {
        'fillColor': colormap(val),
        'color': 'black',
        'weight': 0.4,
        'fillOpacity': 0.7
    }

tooltip_fields = [name_col, 'crime_total', 'safety_level']
tooltip_aliases = ["District","Crime Count","Safety Level"]
folium.GeoJson(
    merged.to_json(),
    name="Districts",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True)
).add_to(m)

colormap.add_to(m)
st_data = st_folium(m, width=1000, height=650)

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
        loc = geolocator.geocode(s)
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except Exception as e:
        return None
    return None

# helper: find containing district polygon
def find_district_for_point(point_latlon, merged_gdf):
    lat, lon = point_latlon
    pt = Point(lon, lat)
    # ensure GeoSeries geometry is shapely and in lat/lon (assume EPSG:4326)
    # check containment (works with shapely)
    contains = merged_gdf[merged_gdf.contains(pt)]
    if len(contains) > 0:
        return contains.iloc[0]
    # else find nearest by centroid distance
    merged_gdf['centroid'] = merged_gdf.geometry.centroid
    distances = merged_gdf['centroid'].apply(lambda c: geodesic((c.y, c.x), (lat, lon)).km)
    merged_gdf['dist_km_to_point'] = distances
    nearest = merged_gdf.loc[merged_gdf['dist_km_to_point'].idxmin()]
    return nearest

# helper: find districts within radius_km
def districts_within_radius(point_latlon, gdf, radius_km=10):
    lat, lon = point_latlon
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
                st.success(f"You are in/near **{district_name}** — Safety Level: **{safety}** (crime count: {crime_count})")
            except Exception:
                st.info("Could not map to a district polygon; showing nearest districts.")

            # find surrounding districts within radius
            nearby_districts = districts_within_radius((lat,lon), merged.copy(), radius_km=10)
            st.write(f"Districts within 10 km: {len(nearby_districts)}")
            if not nearby_districts.empty:
                nd_display = nearby_districts[[name_col,'crime_total','safety_level','dist_km']].rename(columns={name_col:'District','crime_total':'Crime Count','safety_level':'Safety','dist_km':'Distance_km'})
                st.dataframe(nd_display.head(20))

            # Overpass POIs
            elements = overpass_query_pois(lat, lon, radius_m=NEAREST_RADIUS_KM*1000)
            st.write(f"Found {len(elements)} nearby POI elements (police/residential/shelters) within {NEAREST_RADIUS_KM} km via Overpass API.")

            # Build a focused map
            m_local = folium.Map(location=[lat, lon], zoom_start=13, tiles="cartodbpositron")
            # draw nearby districts colored
            for _, row in nearby_districts.iterrows():
                try:
                    geom = row.geometry
                    lvl = row['safety_level']
                    color = "green" if lvl=="Low" else ("yellow" if lvl=="Medium" else "red")
                    folium.GeoJson(data=geom.__geo_interface__, style_function=lambda feat, col=color: {
                        'fillColor': col, 'color':'black', 'weight':0.6, 'fillOpacity':0.4
                    }).add_to(m_local)
                except Exception:
                    pass

            # add user marker
            folium.CircleMarker(location=[lat,lon], radius=6, color="blue", fill=True, fill_opacity=1, popup="You are here").add_to(m_local)

            # Add POI markers
            poi_count = 0
            for el in elements:
                # obtain coordinates
                if el.get('type') == 'node' and 'lat' in el and 'lon' in el:
                    el_lat, el_lon = el['lat'], el['lon']
                elif 'center' in el:
                    el_lat, el_lon = el['center']['lat'], el['center']['lon']
                else:
                    continue
                tags = el.get('tags', {})
                name = tags.get('name', tags.get('operator', 'POI'))
                # choose icon by tag
                if tags.get('amenity') in ('police','police_station'):
                    icon = folium.Icon(color='red', icon='shield')
                elif tags.get('amenity') in ('shelter',):
                    icon = folium.Icon(color='purple', icon='info-sign')
                elif tags.get('building') == 'residential' or tags.get('landuse') == 'residential':
                    icon = folium.Icon(color='green', icon='home')
                else:
                    icon = folium.Icon(color='gray', icon='circle')
                folium.Marker(location=[el_lat, el_lon], popup=f"{name}", tooltip=str(tags.get('amenity', tags.get('building',''))), icon=icon).add_to(m_local)
                poi_count += 1

            st.subheader("Local map — colored surrounding districts + nearby POIs")
            st_folium(m_local, width=950, height=600)

            # list nearest police stations (text)
            police_items = []
            for el in elements:
                tags = el.get('tags', {})
                if tags.get('amenity') in ('police','police_station'):
                    if el.get('type')=='node':
                        latp, lonp = el['lat'], el['lon']
                    elif 'center' in el:
                        latp, lonp = el['center']['lat'], el['center']['lon']
                    else:
                        continue
                    name = tags.get('name','Police Station')
                    dkm = geodesic((lat,lon),(latp,lonp)).km
                    police_items.append((name, dkm, latp, lonp))
            if police_items:
                st.write("Nearest police stations (sorted by distance):")
                police_items_sorted = sorted(police_items, key=lambda x: x[1])
                for name, dkm, plat, plon in police_items_sorted[:10]:
                    st.write(f"- **{name}** — {dkm:.2f} km — ({plat:.5f}, {plon:.5f})")
            else:
                st.write("No police stations found within the Overpass radius.")

st.markdown("---")
st.markdown(
    "Notes: 1) District-name mismatches are common; if many districts show zero, upload a GeoJSON which matches your CSV names or adjust names. "
    "2) Overpass/Nominatim are public services and may rate-limit. For production use consider a paid geocoding / tile service."
)
