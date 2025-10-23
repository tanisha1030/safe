# safe_fast.py  (optimized)
import streamlit as st, pandas as pd, geopandas as gpd, glob, os, json, requests
from branca.colormap import StepColormap
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MeasureControl
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Crime Heatmap", layout="wide")
st.title("India Crime Heatmap — district-level (green → yellow → red)")

def normalize_name(s):
    if pd.isna(s): return ""
    s=str(s).lower()
    for a,b in {'dist':'district','city':'','rural':''}.items(): s=s.replace(a,b)
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s).split())

@st.cache_data(show_spinner=False)
def load_csvs(folder):
    files=sorted(glob.glob(os.path.join(folder,"*.csv"))); rows=[]
    for f in files:
        try: df=pd.read_csv(f,low_memory=False)
        except: df=pd.read_csv(f,encoding="latin1",low_memory=False)
        dcol=next((c for c in df.columns if "district" in c.lower()),df.columns[0])
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums: continue
        df["_t"]=df[nums].sum(axis=1,numeric_only=True)
        small=df[[dcol,"_t"]].rename(columns={dcol:"district_raw","_t":"crime_total"})
        small["district_norm"]=small["district_raw"].apply(normalize_name)
        rows.append(small)
    if not rows: return pd.DataFrame()
    agg=pd.concat(rows).groupby("district_norm",as_index=False).agg({"crime_total":"sum"})
    return agg

@st.cache_data(show_spinner=False)
def load_geojson():
    for u in [
        "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
        "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"]:
        try:
            r=requests.get(u,timeout=20); r.raise_for_status()
            return gpd.GeoDataFrame.from_features(r.json()["features"])
        except: continue
    st.stop()

data=load_csvs("data")
if data.empty: st.error("No CSVs found in data/"); st.stop()
gdf=load_geojson()
name_col=next((c for c in gdf.columns if "NAME_2" in c or "district" in c.lower()),gdf.columns[0])
gdf["district_norm"]=gdf[name_col].apply(normalize_name)
merged=gdf.merge(data,on="district_norm",how="left").fillna({"crime_total":0})

nz=merged.loc[merged.crime_total>0,"crime_total"]
q1,q2=nz.quantile(0.33),nz.quantile(0.66)
def cls(v): return "No Data" if v==0 else "Low" if v<=q1 else "Medium" if v<=q2 else "High"
merged["safety"]=merged["crime_total"].apply(cls)

m=folium.Map(location=[20.6,78.9],zoom_start=5,tiles="cartodbpositron")
vmin,vmax=merged.crime_total.min(),merged.crime_total.max()
cm=StepColormap(colors=["lightgray","green","yellow","red"],index=[vmin,0.1,q1,q2,vmax],vmin=vmin,vmax=vmax)
def style(f):
    v=f["properties"]["crime_total"]
    if v==0:return{"fillColor":"#eee","color":"#bbb","weight":0.3}
    return{"fillColor":"#6bcf7f" if v<=q1 else "#ffd93d" if v<=q2 else "#ff6b6b",
           "color":"#666","weight":0.4}
folium.GeoJson(merged.to_json(),name="Districts",style_function=style,
    tooltip=folium.GeoJsonTooltip(fields=[name_col,"crime_total","safety"],
                                  aliases=["District","Crimes","Safety"])
).add_to(m)
cm.caption="Crime count (Gray=no data, Green→Red)"
m.add_child(cm); folium.LayerControl().add_to(m)
Fullscreen().add_to(m); MeasureControl().add_to(m)
st_folium(m,width=1100,height=650)
