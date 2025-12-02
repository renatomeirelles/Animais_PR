import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from esda import Moran, Moran_Local
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

# ---------------- Menu lateral ----------------
opcao = st.sidebar.selectbox(
    "Selecione a visualização:",
    ["Valores absolutos", "Clusters LISA (univariado)"]
)

# ---------------- Valores absolutos ----------------
if opcao == "Valores absolutos":
    var = st.sidebar.selectbox("Escolha a variável:", df.columns[2:])
    m = folium.Map(location=[-24, -51], zoom_start=6)
    folium.Choropleth(
        geo_data=g,
        data=g,
        columns=['CD_MUN', var],
        key_on='feature.properties.CD_MUN',
        fill_color='YlGnBu',
        legend_name=var
    ).add_to(m)
    st_folium(m, width=700, height=500)

# ---------------- Clusters LISA ----------------
elif opcao == "Clusters LISA (univariado)":
    var = st.sidebar.selectbox("Escolha a variável:", df.columns[2:])
    w = libpysal.weights.Queen.from_dataframe(g)
    w.transform = 'r'
    y = g[var].fillna(0).values
    moran_loc = Moran_Local(y, w)

    g['cluster'] = moran_loc.q
    m = folium.Map(location=[-24, -51], zoom_start=6)
    folium.Choropleth(
        geo_data=g,
        data=g,
        columns=['CD_MUN', 'cluster'],
        key_on='feature.properties.CD_MUN',
        fill_color='Set1',
        legend_name="Cluster LISA"
    ).add_to(m)
    st_folium(m, width=700, height=500)
