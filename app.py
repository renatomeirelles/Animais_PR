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

# ---------------- Cache de dados ----------------
@st.cache_data
def load_data():
    df = pd.read_excel("data/resultado_campanha_de_atualizacao_cadastral.xlsx", dtype={'CD_MUN': str})
    gdf = gpd.read_file("data/PR_Municipios_2024.shp")
    gdf['CD_MUN'] = gdf['CD_MUN'].astype(str)
    g = gdf.merge(df, on='CD_MUN', how='left')
    return g

g = load_data()

# ---------------- Variáveis ----------------
animal_vars = ['Bovinos','Galinaceos','Ovinos','Suinos','Equinos','Caprinos','Muar','Total']
pairs = [('Bovinos','Equinos'),('Bovinos','Muar'),('Suinos','Galinaceos'),('Ovinos','Caprinos')]

# Paleta para clusters
cluster_color_map = {
    'HH': '#ca0020',
    'LL': '#0571b0',
    'HL': '#f4a582',
    'LH': '#92c5de',
    'Não sig.': '#d9d9d9'
}

# Paleta para valores absolutos
absolute_colors = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026','#7f0000']

# ---------------- Pesos espaciais ----------------
w = libpysal.weights.Queen.from_dataframe(g)
w.transform = 'r'

# ---------------- Funções de cálculo ----------------
def calcula_cluster_univariado(df, var):
    x = df[var].fillna(0).values
    lisa = Moran_Local(x, w)
    sig = lisa.p_sim < 0.05
    labels_map = {1:'HH',2:'LH',3:'LL',4:'HL'}
    cluster = np.array(['Não sig.']*len(x),dtype=object)
    cluster[sig] = [labels_map[c] for c in lisa.q[sig]]
    df[f'cluster_{var.lower()}'] = cluster
    return df, f'cluster_{var.lower()}'

def calcula_cluster_bivariado(df, var_x, var_y):
    x = df[var_x].fillna(0).values
    y = df[var_y].fillna(0).values
    lisa_bv = Moran_Local_BV(x,y,w)
    sig = lisa_bv.p_sim < 0.05
    labels_map = {1:'HH',2:'LH',3:'LL',4:'HL'}
    cluster = np.array(['Não sig.']*len(x),dtype=object)
    cluster[sig] = [labels_map[c] for c in lisa_bv.q[sig]]
    col_name = f'cluster_{var_x.lower()}_{var_y.lower()}'
    df[col_name] = cluster
    return df, col_name

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

def render_absolute_map(var):
    # cria faixas por quantis
    s = g[var].fillna(0).sort_values()
    categorias = pd.qcut(g[var].rank(method='first'), q=6, labels=False)
    g[f'faixa_{var.lower()}'] = categorias
    m = folium.Map(location=center_from_geoms(g),zoom_start=7,tiles='CartoDB positron')
    folium.GeoJson(
        g[['NM_MUN',var,f'faixa_{var.lower()}','geometry']].to_json(),
        style_function=lambda f:{
            'fillColor': absolute_colors[f['properties'][f'faixa_{var.lower()}']],
            'color':'black','weight':0.3,'fillOpacity':0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NM_MUN',var,f'faixa_{var.lower()}'],
            aliases=['Município',var,'Faixa (0-5)'],
            localize=True
        )
    ).add_to(m)
    return m

def legenda(label):
    st.markdown(f"### Legenda {label}")
    for l,desc in {
        'HH':'Alto entre altos',
        'LL':'Baixo entre baixos',
        'HL':'Alto entre baixos',
        'LH':'Baixo entre altos',
        'Não sig.':'Não significativo'
    }.items():
        st.markdown(
            f"<div style='display:flex;align-items:center'>"
            f"<div style='width:14px;height:14px;background:{cluster_color_map[l]};margin-right:6px;border:1px solid #888'></div>"
            f"{l} ({desc})</div>",
            unsafe_allow_html=True
        )

# ---------------- Interface ----------------
st.sidebar.header("Configurações")
mode = st.sidebar.selectbox("Selecione a visualização:",
    ["Valores absolutos","Clusters LISA (univariado)","Clusters LISA (bivariado)"])

if mode=="Valores absolutos":
    var = st.sidebar.selectbox("Variável:",animal_vars)
    m = render_absolute_map(var)
    st_folium(m,width=850,height=600,returned_objects=[])

elif mode=="Clusters LISA (univariado)":
    var = st.sidebar.selectbox("Variável:",animal_vars)
    g, cluster_col = calcula_cluster_univariado(g,var)
    m = render_cluster_map(cluster_col,f'Cluster {var}')
    col1,col2 = st.columns([4,1])
    with col1:
        st_folium(m,width=850,height=600,returned_objects=[])
    with col2:
        legenda(var)

else:
    pair_label = st.sidebar.selectbox("Par bivariado:",[f"{x} vs {y}" for x,y in pairs])
    var_x,var_y = pair_label.split(" vs ")
    g, cluster_col = calcula_cluster_bivariado(g,var_x,var_y)
    m = render_cluster_map(cluster_col,f'Cluster {var_x} vs {var_y}')
    col1,col2 = st.columns([4,1])
    with col1:
        st_folium(m,width=850,height=600,returned_objects=[])
    with col2:
        legenda(f"{var_x} vs {var_y}")
