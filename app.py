import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import st_folium
import libpysal
from esda import Moran, Moran_Local, Moran_BV, Moran_Local_BV

# ---------------- Config ----------------
st.set_page_config(page_title="Animais PR", layout="wide")
st.title("Mapas e Clusters LISA — Paraná")

# ---------------- Caminhos relativos ----------------
# Os arquivos devem estar dentro da pasta "data" junto do app.py
excel_path = "data/resultado_campanha_de_atualizacao_cadastral.xlsx"
shp_path   = "data/PR_Municipios_2024.shp"

# ---------------- Função para limpar números ----------------
def limpa_numero(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace('.', '').replace(',', '')
    if s.isdigit():
        return int(s)
    return pd.to_numeric(s, errors='coerce')

# ---------------- Carregar dados ----------------
df = pd.read_excel(excel_path, dtype={'CD_MUN': str})

animal_vars = ['Bovinos','Galinaceos','Ovinos','Suinos','Equinos','Caprinos','Muar']
for col in animal_vars:
    df[col] = df[col].apply(limpa_numero).astype('Int64')

df['Total'] = df['Total'].apply(limpa_numero).astype('Int64')

df['VBP_MUN'] = (
    df['VBP_MUN'].astype(str)
      .str.replace('[R$ ]', '', regex=True)
      .str.replace('.', '')
      .str.replace(',', '.')
      .astype(float)
)

df['PIB_MUN (RS 1.000)'] = (
    df['PIB_MUN (RS 1.000)'].astype(str)
      .str.replace('[R$ ]', '', regex=True)
      .str.replace('.', '')
      .str.replace(',', '.')
      .astype(float)
)

# ---------------- Ler shapefile e juntar ----------------
gdf = gpd.read_file(shp_path)
gdf['CD_MUN'] = gdf['CD_MUN'].astype(str)
g = gdf.merge(df, on='CD_MUN', how='left')

# ---------------- Visualização inicial ----------------
st.write("Prévia dos dados carregados:")
st.dataframe(g[['NM_MUN'] + animal_vars + ['Total','VBP_MUN','PIB_MUN (RS 1.000)']].head())
