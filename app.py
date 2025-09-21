import os
import pandas as pd
import json
import folium
from folium.plugins import MarkerCluster
from rapidfuzz import process, fuzz

# ==========================
# CONFIGURATION
# ==========================
DATA_DIR = "data"  # Folder with CSV files
GEOJSON_FILE = "india_district.geojson"
OUTPUT_JSON = "district_crime_scores.json"

# Manual district mapping (add as needed)
manual_map = {
    "Bengaluru": "Bangalore",
    "Thiruvananthapuram": "Trivandrum",
    # Add other known variations here
}

# ==========================
# LOAD AND AGGREGATE CSVs
# ==========================
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
dfs = []
for f in all_files:
    df = pd.read_csv(f)
    dfs.append(df)
crime_data = pd.concat(dfs, ignore_index=True)

# Assuming CSV has 'District' and 'Crime_Count' columns
crime_data['District'] = crime_data['District'].replace(manual_map)
district_crime = crime_data.groupby('District')['Crime_Count'].sum().reset_index()

# ==========================
# LOAD GEOJSON
# ==========================
with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
    geojson = json.load(f)

geo_districts = [feature['properties']['NAME_2'] for feature in geojson['features']]

# ==========================
# DISTRICT MATCHING USING FUZZY MATCH
# ==========================
matched = {}
unmatched = []

for district in district_crime['District']:
    if district in geo_districts:
        matched[district] = district
    else:
        # Fuzzy match with threshold 80
        match, score, _ = process.extractOne(district, geo_districts, scorer=fuzz.token_sort_ratio)
        if score >= 80:
            matched[district] = match
        else:
            unmatched.append(district)

print(f"Matched districts: {len(matched)}")
print(f"Unmatched districts ({len(unmatched)}): {unmatched}")

# Apply matched names
district_crime['Geo_District'] = district_crime['District'].map(matched)

# Save precomputed scores
district_crime.to_json(OUTPUT_JSON, orient='records', indent=2)

# ==========================
# BUILD CHOROPLETH MAP
# ==========================
# Compute quantiles for coloring
district_crime['Crime_Score'] = district_crime['Crime_Count']
quantiles = district_crime['Crime_Score'].quantile([0.33, 0.66]).tolist()

def get_color(score):
    if score <= quantiles[0]:
        return 'green'
    elif score <= quantiles[1]:
        return 'yellow'
    else:
        return 'red'

# Initialize map
m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles='Stamen Terrain')

# Add choropleth
for feature in geojson['features']:
    district_name = feature['properties']['NAME_2']
    crime_row = district_crime[district_crime['Geo_District'] == district_name]
    if not crime_row.empty:
        score = int(crime_row['Crime_Score'])
        folium.GeoJson(
            feature,
            style_function=lambda x, s=score: {
                'fillColor': get_color(s),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.6
            },
            tooltip=folium.Tooltip(f"{district_name}<br>Crime Score: {score}")
        ).add_to(m)

# ==========================
# OPTIONAL: POINT MARKERS
# ==========================
marker_cluster = MarkerCluster().add_to(m)
# Example: using centroids from GeoJSON
for feature in geojson['features']:
    district_name = feature['properties']['NAME_2']
    crime_row = district_crime[district_crime['Geo_District'] == district_name]
    if not crime_row.empty:
        score = int(crime_row['Crime_Score'])
        # Compute centroid of polygon
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            coords = geom['coordinates'][0]
        elif geom['type'] == 'MultiPolygon':
            coords = geom['coordinates'][0][0]
        lon = sum([c[0] for c in coords])/len(coords)
        lat = sum([c[1] for c in coords])/len(coords)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=get_color(score),
            fill=True,
            fill_opacity=0.7,
            popup=f"{district_name}: {score}"
        ).add_to(marker_cluster)

# Save map
m.save("india_crime_map.html")
print("Map saved to india_crime_map.html")
