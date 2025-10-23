# safe_fast_fixed.py — fully working fast version
import streamlit as st, pandas as pd, geopandas as gpd, glob, os, json, requests
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MeasureControl
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — district-level (green → yellow → red)")

# -------------------------------------------
# Utility functions
# -------------------------------------------
def normalize_name(s):
    if pd.isna(s): return ""
    s=str(s).lower()
    for a,b in {'dist':'district','city':'','rural':''}.items(): s=s.replace(a,b)
    s="".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())

@st.cache_data(show_spinner=False)
def load_csvs(folder="data"):
    files=sorted(glob.glob(os.path.join(folder,"*.csv"))); rows=[]
    for f in files:
        try: df=pd.read_csv(f,low_memory=False)
        except: df=pd.read_csv(f,encoding="latin1",low_memory=False)
        dcol=next((c for c in df.columns if "district" in c.lower()),df.columns[0])
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums: continue
        df["_total"]=df[nums].sum(axis=1,numeric_only=True)
        df["district_norm"]=df[dcol].apply(normalize_name)
        rows.append(df[["district_norm","_total"]])
    if not rows: return pd.DataFrame()
    agg=pd.concat(rows).groupby("district_norm",as_index=False).sum()
    agg.rename(columns={"_total":"crime_total"},inplace=True)
    return agg

@st.cache_data(show_spinner=False)
def load_geojson():
    urls=[
        "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
        "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
    ]
    for u in urls:
        try:
            r=requests.get(u,timeout=25); r.raise_for_status()
            gdf=gpd.GeoDataFrame.from_features(r.json()["features"])
            if gdf.crs is None: gdf.set_crs("EPSG:4326",inplace=True)
            else: gdf=gdf.to_crs("EPSG:4326")
            gdf=gdf[gdf.geometry.notnull()]
            return gdf
        except Exception as e:
            st.write(f"⚠️ GeoJSON load failed from {u[:40]}...: {e}")
            continue
    st.error("Failed to load India district GeoJSON.")
    st.stop()

# -------------------------------------------
# Load data
# -------------------------------------------
data=load_csvs("data")
if data.empty:
    st.error("No CSVs found in 'data/' folder. Please add crime CSVs.")
    st.stop()

gdf=load_geojson()
name_col=next((c for c in gdf.columns if "NAME_2" in c or "district" in c.lower()),gdf.columns[0])
gdf["district_norm"]=gdf[name_col].apply(normalize_name)

merged=gdf.merge(data,on="district_norm",how="left").fillna({"crime_total":0})
nonzero=merged.loc[merged.crime_total>0,"crime_total"]
if nonzero.empty:
    st.error("All districts have zero crime data — please check CSVs.")
    st.stop()
q1,q2=nonzero.quantile(0.33),nonzero.quantile(0.66)
def classify(v): return "No Data" if v==0 else "Low" if v<=q1 else "Medium" if v<=q2 else "High"
merged["safety"]=merged["crime_total"].apply(classify)

# -------------------------------------------
# Draw the map
# -------------------------------------------
m=folium.Map(location=[20.6,78.9],zoom_start=5,tiles="cartodbpositron")

vmin,vmax=merged.crime_total.min(),merged.crime_total.max()
colormap=StepColormap(colors=["#e0e0e0","#6bcf7f","#ffd93d","#ff6b6b"],
                      index=[vmin,0.1,q1,q2,vmax],vmin=vmin,vmax=vmax)
colormap.caption="Crime count (Gray = No data, Green → Red)"

def style_fn(f):
    v=f["properties"]["crime_total"]
    if v==0:return{"fillColor":"#e0e0e0","color":"#999","weight":0.3}
    elif v<=q1:return{"fillColor":"#6bcf7f","color":"#4fa860","weight":0.4}
    elif v<=q2:return{"fillColor":"#ffd93d","color":"#e6c300","weight":0.5}
    else:return{"fillColor":"#ff6b6b","color":"#d35b5b","weight":0.6}

tooltip=folium.GeoJsonTooltip(fields=[name_col,"crime_total","safety"],
                              aliases=["District","Crimes","Safety"],
                              localize=True)
folium.GeoJson(
    merged.to_json(),
    name="Crime Heatmap",
    style_function=style_fn,
    tooltip=tooltip
).add_to(m)
m.add_child(colormap)
folium.LayerControl().add_to(m)
Fullscreen().add_to(m)
MeasureControl().add_to(m)

st_folium(m,width=1150,height=650)
st.info("✅ Heatmap loaded successfully. Use zoom & layer controls to explore.")
