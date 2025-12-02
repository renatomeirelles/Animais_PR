import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from esda import Moran_Local, Moran_Local_BV
import libpysal

# ---------------- Config ----------------
st.set_page_config(page_title="Animais PR", layout="wide")
st.title("Mapas e Clusters LISA — Paraná")

# ---------------- Caminhos ----------------
excel_path = "data/resultado_campanha_de_atualizacao_cadastral.xlsx"
shp_path   = "data/PR_Municipios_2024.shp"

# ---------------- Carregar dados ----------------
df = pd.read_excel(excel_path, dtype={'CD_MUN': str})
gdf = gpd.read_file(shp_path)
gdf['CD_MUN'] = gdf['CD_MUN'].astype(str)
g = gdf.merge(df, on='CD_MUN', how='left')

# ---------------- Variáveis ----------------
animal_vars = ['Bovinos','Galinaceos','Ovinos','Suinos','Equinos','Caprinos','Muar','Total']

# Paleta para clusters
cluster_color_map = {
    'HH': '#ca0020',
    'LL': '#0571b0',
    'HL': '#f4a582',
    'LH': '#92c5de',
    'Não sig.': '#d9d9d9'
}

# ---------------- Pesos espaciais ----------------
w = libpysal.weights.Queen.from_dataframe(g)
w.transform = 'r'

# ---------------- Funções com cache ----------------
@st.cache_data
def calcula_clusters_univariados(g, w, animal_vars):
    results = {}
    for var in animal_vars:
        x = g[var].fillna(0).values
        lisa = Moran_Local(x, w)
        sig = lisa.p_sim < 0.05
        labels_map = {1:'HH',2:'LH',3:'LL',4:'HL'}
        cluster = np.array(['Não sig.']*len(x),dtype=object)
        cluster[sig] = [labels_map[c] for c in lisa.q[sig]]
        g[f'cluster_{var.lower()}'] = cluster
        results[var] = lisa
    return g, results

@st.cache_data
def calcula_clusters_bivariados(g, w, pairs):
    results = {}
    for var_x,var_y in pairs:
        x = g[var_x].fillna(0).values
        y = g[var_y].fillna(0).values
        lisa_bv = Moran_Local_BV(x,y,w)
        sig = lisa_bv.p_sim < 0.05
        labels_map = {1:'HH',2:'LH',3:'LL',4:'HL'}
        cluster = np.array(['Não sig.']*len(x),dtype=object)
        cluster[sig] = [labels_map[c] for c in lisa_bv.q[sig]]
        col_name = f'cluster_{var_x.lower()}_{var_y.lower()}'
        g[col_name] = cluster
        results[(var_x,var_y)] = lisa_bv
    return g, results

# ---------------- Pares bivariados ----------------
pairs = [('Bovinos','Equinos'),('Bovinos','Muar'),('Suinos','Galinaceos'),('Ovinos','Caprinos')]

# Calcula clusters
g, moran_uni = calcula_clusters_univariados(g,w,animal_vars)
g, moran_bi  = calcula_clusters_bivariados(g,w,pairs)

# ---------------- Funções de mapa ----------------
def center_from_geoms(geo):
    c = geo.geometry.unary_union.centroid
    return [c.y,c.x]

def render_cluster_map(cluster_col,label):
    m = folium.Map(location=center_from_geoms(g),zoom_start=7,tiles='CartoDB positron')
    folium.GeoJson(
        g[['NM_MUN',cluster_col,'geometry']].to_json(),
        style_function=lambda f:{
            'fillColor': cluster_color_map.get(f['properties'][cluster_col],'#d9d9d9'),
            'color':'black','weight':0.3,'fillOpacity':0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NM_MUN',cluster_col],
            aliases=['Município',label],
            localize=True
        )
    ).add_to(m)
    return m

# ---------------- Interface ----------------
st.sidebar.header("Configurações")
mode = st.sidebar.selectbox("Selecione a visualização:",
    ["Clusters LISA (univariado)","Clusters LISA (bivariado)"])

if mode=="Clusters LISA (univariado)":
    var = st.sidebar.selectbox("Variável:",animal_vars)
    m = render_cluster_map(f'cluster_{var.lower()}',f'Cluster {var}')
    col1,col2 = st.columns([4,1])
    with col1:
        st_folium(m,width=850,height=600,returned_objects=[])
    with col2:
        st.markdown("### Legenda")
        for label,desc in {
            'HH':'Alto entre altos',
            'LL':'Baixo entre baixos',
            'HL':'Alto entre baixos',
            'LH':'Baixo entre altos',
            'Não sig.':'Não significativo'
        }.items():
            st.markdown(f"<div style='display:flex;align-items:center'><div style='width:14px;height:14px;background:{cluster_color_map[label]};margin-right:6px;border:1px solid #888'></div>{label} ({desc})</div>",unsafe_allow_html=True)

else:
    pair_label = st.sidebar.selectbox("Par bivariado:",[f"{x} vs {y}" for x,y in pairs])
    var_x,var_y = pair_label.split(" vs ")
    m = render_cluster_map(f'cluster_{var_x.lower()}_{var_y.lower()}',f'Cluster {var_x} vs {var_y}')
    col1,col2 = st.columns([4,1])
    with col1:
        st_folium(m,width=850,height=600,returned_objects=[])
    with col2:
        st.markdown(f"### Legenda {var_x} vs {var_y}")
        for label,desc in {
            'HH':'Alto entre altos',
            'LL':'Baixo entre baixos',
            'HL':'Alto entre baixos',
            'LH':'Baixo entre altos',
            'Não sig.':'Não significativo'
        }.items():
            st.markdown(f"<div style='display:flex;align-items:center'><div style='width:14px;height:14px;background:{cluster_color_map[label]};margin-right:6px;border:1px solid #888'></div>{label} ({desc})</div>",unsafe_allow_html=True)
