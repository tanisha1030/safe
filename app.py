# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import glob, os, io, json, requests, random
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="India Crime Heatmap (Safesomes)", layout="wide")
DATA_FOLDER = "data"
POLICE_JSON = "police_stations.json"
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
]

# ---------------------------
# Utilities
# ---------------------------
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    for old, new in {
        'commr': 'commissioner',
        'commissionerate': 'commissioner',
        'dist': 'district',
        'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
        'city': '', 'rural': ''
    }.items():
        s = s.replace(old, new)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())

def load_police_json(path=POLICE_JSON):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

police_db = load_police_json()

# ---------------------------
# Load & aggregate CSVs
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv_data(data_folder):
    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not files:
        return pd.DataFrame()
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except:
            try:
                df = pd.read_csv(f, encoding="latin1", low_memory=False)
            except:
                continue
        district_col = None
        for c in df.columns:
            if "district" in c.lower():
                district_col = c
                break
        if district_col is None:
            district_col = df.columns[0]
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df["_total"] = df[numeric_cols].sum(axis=1, numeric_only=True)
        else:
            df["_total"] = 1
        small = df[[district_col, "_total"]].copy()
        small.columns = ["district_raw", "total"]
        small["district_norm"] = small["district_raw"].apply(normalize_name)
        all_data.append(small)
    if not all_data:
        return pd.DataFrame()
    combined = pd.concat(all_data)
    return combined.groupby("district_norm", as_index=False).agg({"total": "sum"})

crime_agg = load_csv_data(DATA_FOLDER)

# ---------------------------
# Load GeoJSON
# ---------------------------
@st.cache_data(show_spinner=False)
def load_geojson():
    for url in DEFAULT_GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            gj = r.json()
            gdf = gpd.GeoDataFrame.from_features(gj["features"])
            return gdf
        except:
            continue
    st.error("Failed to load any GeoJSON file.")
    st.stop()

geo = load_geojson()

# Determine district column
name_col = None
for c in geo.columns:
    if "name" in c.lower() or "district" in c.lower():
        name_col = c
        break
if not name_col:
    name_col = geo.columns[0]

geo["district_norm"] = geo[name_col].apply(normalize_name)

# Merge with data
merged = geo.merge(crime_agg, on="district_norm", how="left")

# ---------------------------
# Synthetic data generation (no JSON file)
# ---------------------------
for i, row in merged.iterrows():
    if pd.isna(row["total"]) or row["total"] == 0:
        merged.at[i, "total"] = random.randint(3000, 5000)
        merged.at[i, "source"] = "synthetic"
    else:
        merged.at[i, "source"] = "csv"

# ---------------------------
# Map rendering
# ---------------------------
st.title("🗺️ India Crime Heatmap (Safesomes)")
st.markdown("Green regions are **synthetic (safe)** — generated where no CSV data exists.")

m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="CartoDB positron")

def color_for_row(row):
    if row["source"] == "synthetic":
        return "#4CAF50"  # green for synthetic
    total = row["total"]
    if total < merged["total"].quantile(0.33):
        return "#81C784"  # low
    elif total < merged["total"].quantile(0.66):
        return "#FFF176"  # medium
    else:
        return "#E57373"  # high

for _, r in merged.iterrows():
    color = color_for_row(r)
    tooltip = f"{r[name_col]} — Crime Index: {int(r['total'])} ({r['source']})"
    folium.GeoJson(
        r.geometry.__geo_interface__,
        style_function=lambda x, clr=color: {
            "fillColor": clr, "color": "black", "weight": 0.4, "fillOpacity": 0.7
        },
        tooltip=tooltip
    ).add_to(m)

st_folium(m, width=1100, height=650)

# ---------------------------
# Safesomes Search
# ---------------------------
st.sidebar.header("🔍 Safesomes — Search")
query = st.sidebar.text_input("Search by district or police station")
btn = st.sidebar.button("Search")

def safesomes_search(q):
    q = q.lower().strip()
    results = []
    for _, r in merged.iterrows():
        if q in str(r[name_col]).lower():
            results.append({
                "type": "district",
                "name": r[name_col],
                "crime": int(r["total"]),
                "source": r["source"]
            })
    for d, stations in police_db.items():
        for stn in stations:
            if q in stn["name"].lower() or q in stn["address"].lower():
                results.append({
                    "type": "station",
                    "district": d,
                    "name": stn["name"],
                    "address": stn["address"],
                    "phone": stn["phone"]
                })
    return results

if btn and query:
    res = safesomes_search(query)
    if not res:
        st.sidebar.info("No match found.")
    else:
        for r in res:
            if r["type"] == "district":
                st.sidebar.markdown(f"**District:** {r['name']}  \nCrime Index: `{r['crime']}`  \nSource: `{r['source']}`")
            else:
                st.sidebar.markdown(f"**Station:** {r['name']}  \n📍 {r['address']}  \n📞 {r['phone']}")

# ---------------------------
# Optional: Location safety check
# ---------------------------
st.sidebar.header("📍 Check Safety by Location")
loc_in = st.sidebar.text_input("Enter address or lat,lon")
if st.sidebar.button("Find Safety"):
    geo_locator = Nominatim(user_agent="safesomes-locator")
    try:
        if "," in loc_in:
            lat, lon = [float(x) for x in loc_in.split(",")]
        else:
            loc = geo_locator.geocode(loc_in + ", India")
            lat, lon = loc.latitude, loc.longitude
        pt = Point(lon, lat)
        contains = merged[merged.contains(pt)]
        if len(contains) > 0:
            r = contains.iloc[0]
        else:
            merged["centroid"] = merged.geometry.centroid
            merged["dist_km"] = merged["centroid"].apply(lambda c: geodesic((c.y, c.x), (lat, lon)).km)
            r = merged.loc[merged["dist_km"].idxmin()]
        st.sidebar.success(f"District: {r[name_col]}  \nSafety Level: `{r['source']}`  \nCrime Index: `{int(r['total'])}`")
    except Exception as e:
        st.sidebar.error(f"Could not find location: {e}")
