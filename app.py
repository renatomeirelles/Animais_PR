import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from esda import Moran, Moran_Local, Moran_BV, Moran_Local_BV
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

# Padroniza nomes e junta
g = gdf.merge(df, on='CD_MUN', how='left')

# ---------------- Variáveis e paletas ----------------
animal_vars = ['Bovinos', 'Galinaceos', 'Ovinos', 'Suinos', 'Equinos', 'Caprinos', 'Muar', 'Total']

# Paleta para valores absolutos (6 classes: do claro ao escuro)
abs_colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026', '#7f0000']  # YlOrRd aprofundado

# Paleta para clusters LISA (fixa e consistente)
cluster_color_map = {
    'HH': '#ca0020',       # Alto entre altos
    'LL': '#0571b0',       # Baixo entre baixos
    'HL': '#f4a582',       # Alto entre baixos
    'LH': '#92c5de',       # Baixo entre altos
    'Não sig.': '#d9d9d9'  # Não significativo
}

# ---------------- Faixas para valores absolutos ----------------
faixas_info = {}
for var in animal_vars:
    s = g[var].fillna(0).sort_values()
    # cria 6 quantis (q=6) rotulados 0–5
    categorias = pd.qcut(s.rank(method='first'), q=6, labels=False)
    # realoca no g pela ordem original
    g[f'faixa_{var.lower()}'] = pd.qcut(g[var].rank(method='first'), q=6, labels=False)
    # salva min/max de cada faixa para legendas
    tamanhos = [len(s)//6]*5 + [len(s) - (len(s)//6)*5]
    bins = []
    start = 0
    for t in tamanhos:
        end = start + t
        grupo = s.iloc[start:end]
        bins.append((grupo.min(), grupo.max()))
        start = end
    faixas_info[var] = bins

# ---------------- Pesos espaciais ----------------
w = libpysal.weights.Queen.from_dataframe(g)
w.transform = 'r'

# ---------------- Resultados LISA univariados ----------------
moran_results = {}
for var in animal_vars:
    x = g[var].fillna(0).values
    lisa = Moran_Local(x, w)
    sig = lisa.p_sim < 0.05
    labels_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    cluster = np.array(['Não sig.'] * len(x), dtype=object)
    cluster[sig] = [labels_map[c] for c in lisa.q[sig]]
    g[f'cluster_{var.lower()}'] = cluster
    moran_results[var] = {'lisa': lisa}

# ---------------- Pares bivariados ----------------
pairs = [
    ('Bovinos', 'Equinos'),
    ('Bovinos', 'Muar'),
    ('Suinos', 'Galinaceos'),
    ('Ovinos', 'Caprinos'),
    ('Muar', 'Equinos'),
]

bv_results = {}
for var_x, var_y in pairs:
    x = g[var_x].fillna(0).values
    y = g[var_y].fillna(0).values
    lisa_bv = Moran_Local_BV(x, y, w)
    sig = lisa_bv.p_sim < 0.05
    labels_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    cluster = np.array(['Não sig.'] * len(x), dtype=object)
    cluster[sig] = [labels_map[c] for c in lisa_bv.q[sig]]
    col_name = f'cluster_{var_x.lower()}_{var_y.lower()}'
    g[col_name] = cluster
    bv_results[(var_x, var_y)] = {'lisa_bv': lisa_bv, 'cluster_col': col_name}

# ---------------- Funções de mapa ----------------
def center_from_geoms(geo):
    c = geo.geometry.unary_union.centroid
    return [c.y, c.x]

def render_absolute_map(var_key):
    m = folium.Map(location=center_from_geoms(g), zoom_start=7, tiles='CartoDB positron')
    faixa_col = f'faixa_{var_key.lower()}'
    bins = faixas_info[var_key]
    # GeoJson com estilo por faixa
    folium.GeoJson(
        g[['NM_MUN', var_key, faixa_col, 'geometry']].to_json(),
        style_function=lambda f: {
            'fillColor': abs_colors[int(f['properties'][faixa_col])] if f['properties'][faixa_col] is not None else '#cccccc',
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NM_MUN', var_key, faixa_col],
            aliases=['Município', var_key, 'Faixa (0–5)'],
            localize=True
        )
    ).add_to(m)
    # Legenda manual (faixas)
    legend_html = "<div style='position: fixed; bottom: 20px; right: 20px; background: white; padding: 10px; border: 1px solid #bbb; font-size: 12px'>"
    legend_html += f"<b>{var_key} (faixas)</b><br>"
    for i, (vmin, vmax) in enumerate(bins):
        legend_html += f"<div style='display:flex;align-items:center;margin:2px 0'><div style='width:14px;height:14px;background:{abs_colors[i]};margin-right:6px;border:1px solid #888'></div>{int(vmin)} – {int(vmax)}</div>"
    legend_html += "</div>"
    folium.Marker(location=center_from_geoms(g), icon=folium.DivIcon(html=legend_html)).add_to(m)
    return m

def render_cluster_map_uni(var_key):
    m = folium.Map(location=center_from_geoms(g), zoom_start=7, tiles='CartoDB positron')
    cluster_col = f'cluster_{var_key.lower()}'
    folium.GeoJson(
        g[['NM_MUN', cluster_col, 'geometry']].to_json(),
        style_function=lambda f: {
            'fillColor': cluster_color_map.get(f['properties'][cluster_col], '#d9d9d9'),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NM_MUN', cluster_col],
            aliases=['Município', 'Cluster LISA'],
            localize=True
        )
    ).add_to(m)
    # Legenda de clusters
    legend_html = "<div style='position: fixed; bottom: 20px; right: 20px; background: white; padding: 10px; border: 1px solid #bbb; font-size: 12px'>"
    legend_html += "<b>Clusters LISA</b><br>"
    for label in ['HH', 'LL', 'HL', 'LH', 'Não sig.']:
        color = cluster_color_map[label]
        desc = {
            'HH': 'Alto entre altos',
            'LL': 'Baixo entre baixos',
            'HL': 'Alto entre baixos',
            'LH': 'Baixo entre altos',
            'Não sig.': 'Não significativo'
        }[label]
        legend_html += f"<div style='display:flex;align-items:center;margin:2px 0'><div style='width:14px;height:14px;background:{color};margin-right:6px;border:1px solid #888'></div>{label} ({desc})</div>"
    legend_html += "</div>"
    folium.Marker(location=center_from_geoms(g), icon=folium.DivIcon(html=legend_html)).add_to(m)
    return m

def render_cluster_map_bv(var_x, var_y):
    m = folium.Map(location=center_from_geoms(g), zoom_start=7, tiles='CartoDB positron')
    cluster_col = f'cluster_{var_x.lower()}_{var_y.lower()}'
    folium.GeoJson(
        g[['NM_MUN', cluster_col, 'geometry']].to_json(),
        style_function=lambda f: {
            'fillColor': cluster_color_map.get(f['properties'][cluster_col], '#d9d9d9'),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NM_MUN', cluster_col],
            aliases=['Município', f'Cluster {var_x} vs {var_y}'],
            localize=True
        )
    ).add_to(m)
    # Legenda de clusters
    legend_html = "<div style='position: fixed; bottom: 20px; right: 20px; background: white; padding: 10px; border: 1px solid #bbb; font-size: 12px'>"
    legend_html += f"<b>Clusters LISA — {var_x} vs {var_y}</b><br>"
    for label in ['HH', 'LL', 'HL', 'LH', 'Não sig.']:
        color = cluster_color_map[label]
        desc = {
            'HH': 'Alto entre altos',
            'LL': 'Baixo entre baixos',
            'HL': 'Alto entre baixos',
            'LH': 'Baixo entre altos',
            'Não sig.': 'Não significativo'
        }[label]
        legend_html += f"<div style='display:flex;align-items:center;margin:2px 0'><div style='width:14px;height:14px;background:{color};margin-right:6px;border:1px solid #888'></div>{label} ({desc})</div>"
    legend_html += "</div>"
    folium.Marker(location=center_from_geoms(g), icon=folium.DivIcon(html=legend_html)).add_to(m)
    return m

# ---------------- Interface ----------------
st.sidebar.header("Configurações")
mode = st.sidebar.selectbox(
    "Selecione a visualização:",
    ["Valores absolutos", "Clusters LISA (univariado)", "Clusters LISA (bivariado)"]
)

if mode == "Valores absolutos":
    var = st.sidebar.selectbox("Variável:", animal_vars, index=animal_vars.index('Total') if 'Total' in animal_vars else 0)
    m = render_absolute_map(var)
    st_folium(m, width=850, height=600)

elif mode == "Clusters LISA (univariado)":
    var = st.sidebar.selectbox("Variável:", animal_vars)
    m = render_cluster_map_uni(var)
    st_folium(m, width=850, height=600)

else:
    pair_label = st.sidebar.selectbox("Par bivariado:", [f"{x} vs {y}" for x, y in pairs])
    var_x, var_y = pair_label.split(" vs ")
    m = render_cluster_map_bv(var_x, var_y)
    st_folium(m, width=850, height=600)
