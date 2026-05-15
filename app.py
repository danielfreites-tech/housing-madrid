import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Constants ──────────────────────────────────────────────────────────────────

HOUSE_TYPES = [
    "HouseType 1: Pisos",
    "HouseType 2: Casa o chalet",
    "HouseType 4: Dúplex",
    "HouseType 5: Áticos",
]

FLOOR_OPTIONS = [
    "Bajo", "Entreplanta", "Semi-sótano", "Sótano",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

FEATURES = [
    "sq_mt_built", "sq_mt_useful", "n_rooms", "n_bathrooms", "floor",
    "subtitle", "has_lift", "has_ac", "has_parking", "built_year",
    "is_new_development", "house_type_id", "is_renewal_needed",
    "is_exterior", "has_terrace", "has_balcony",
    "sq_mt_built_missing", "sq_mt_useful_missing", "n_bathrooms_missing",
    "floor_missing", "has_lift_missing", "has_ac_missing",
    "built_year_missing", "is_new_development_missing", "house_type_id_missing",
]

CAT_FEATURES = ["subtitle", "house_type_id", "floor"]

# Median price per m² by neighborhood (€/m², based on ~2024 market data)
NEIGHBORHOOD_MEDIANS = {
    "Recoletos": 8500,
    "Barrio de Salamanca": 7500,
    "El Viso": 7200,
    "Almagro": 7000,
    "Castellana": 7000,
    "Lista": 6500,
    "Bernabéu-Hispanoamérica": 6800,
    "Goya": 6400,
    "Ibiza": 6200,
    "Conde Orgaz-Piovera": 6000,
    "Jerónimos": 6800,
    "Retiro": 5800,
    "Chamartín": 5800,
    "Nueva España": 5500,
    "Aravaca": 5500,
    "Chamberí": 5500,
    "Moncloa": 5200,
    "Ciudad Universitaria": 5000,
    "Argüelles": 5000,
    "Chueca-Justicia": 5000,
    "Niño Jesús": 5000,
    "Palacio": 4800,
    "Fuencarral": 4800,
    "Malasaña": 4800,
    "Centro": 4600,
    "Sol": 5000,
    "Cortes": 5000,
    "Universidad": 4500,
    "Hortaleza": 4500,
    "Castilla": 4500,
    "Cuzco-Castillejos": 4200,
    "Montecarmelo": 4000,
    "Prosperidad": 4000,
    "Guindalera": 4000,
    "Las Tablas": 3800,
    "Buenavista": 3800,
    "Ventas": 3800,
    "Lavapiés": 3800,
    "Pacífico": 3400,
    "Adelfas": 3600,
    "Estrella": 3600,
    "Sanchinarro": 3500,
    "Pueblo Nuevo": 3500,
    "Tetuán": 3500,
    "Embajadores": 3600,
    "Fuencarral-El Pardo": 3500,
    "Acacias": 3500,
    "Hortaleza Norte": 3500,
    "Concepción": 3600,
    "San Pascual": 3600,
    "Valdebebas": 3200,
    "Canillas": 3200,
    "San Juan Bautista": 3400,
    "Palomas": 3400,
    "Arganzuela": 3200,
    "Delicias": 3000,
    "Palos de Moguer": 3200,
    "Latina": 3000,
    "Vista Alegre": 2400,
    "Aluche": 2500,
    "Usera": 2500,
    "Atalaya": 3000,
    "Costillares": 3200,
    "Barajas": 2800,
    "Canillejas": 2800,
    "Alameda de Osuna": 3200,
    "Moratalaz": 2600,
    "Carabanchel": 2300,
    "Comillas": 2200,
    "Opañel": 2200,
    "San Isidro": 2200,
    "Campamento": 2400,
    "Cuatro Vientos": 2200,
    "Vallecas": 2000,
    "Puente de Vallecas": 2100,
    "Pradolongo": 2100,
    "San Blas": 2200,
    "Entrevías": 1900,
    "Vicálvaro": 2000,
    "Villaverde": 1900,
    "Butarque": 1900,
    "Los Rosales": 1900,
}

FEATURE_IMPORTANCE_TOP = {
    "sq_mt_built": 30.4,
    "subtitle": 29.7,
    "n_bathrooms": 11.4,
    "has_lift": 4.5,
    "floor": 4.1,
    "house_type_id": 4.0,
    "built_year": 3.5,
}

FEATURE_LABELS = {
    "sq_mt_built": "Superficie construida",
    "subtitle": "Barrio",
    "n_bathrooms": "Número de baños",
    "has_lift": "Ascensor",
    "floor": "Planta",
    "house_type_id": "Tipo de inmueble",
    "built_year": "Año de construcción",
}

# ── Loaders ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("model/artifacts/model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_neighborhoods():
    with open("model/artifacts/neighborhoods.json") as f:
        return json.load(f)

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_eur(value: float) -> str:
    return f"{value:,.0f} €".replace(",", ".")

def build_row(inputs: dict) -> pd.DataFrame:
    row = {f: np.nan for f in FEATURES}
    for cat in CAT_FEATURES:
        row[cat] = "missing"
    row.update(inputs)
    for flag in [
        "sq_mt_built_missing", "sq_mt_useful_missing", "n_bathrooms_missing",
        "floor_missing", "has_lift_missing", "has_ac_missing",
        "built_year_missing", "is_new_development_missing", "house_type_id_missing",
    ]:
        base = flag.replace("_missing", "")
        val = inputs.get(base)
        row[flag] = 1.0 if (val is None or val == "missing") else 0.0
    return pd.DataFrame([row], columns=FEATURES)

def render_step_indicator(current: int):
    steps = [("1", "Ubicación"), ("2", "Tipo"), ("3", "Equipamiento")]
    parts = []
    for i, (num, label) in enumerate(steps):
        if i < current:
            state, icon = "done", "✓"
        elif i == current:
            state, icon = "active", num
        else:
            state, icon = "", num
        parts.append(f'<div class="pj-wizard-item"><div class="pj-wizard-dot {state}">{icon}</div><div class="pj-wizard-label {state}">{label}</div></div>')
        if i < len(steps) - 1:
            parts.append('<div class="pj-wizard-connector"></div>')
    html = '<div class="pj-wizard-bar">' + "".join(parts) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_feature_chart():
    labels = list(FEATURE_LABELS.values())
    values = [FEATURE_IMPORTANCE_TOP[k] for k in FEATURE_LABELS]
    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color="#b5451a",
        marker_opacity=0.85,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=4, b=0),
        height=210,
        font=dict(family="DM Sans, sans-serif", size=12, color="#2a2820"),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(size=12), tickcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Design system ──────────────────────────────────────────────────────────────

DESIGN_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Playfair+Display:ital,wght@0,900;1,400;1,700;1,900&display=swap" rel="stylesheet">

<style>
:root {
  --cream:       #f6f1e9;
  --warm-white:  #faf9f5;
  --charcoal:    #2a2820;
  --mid:         #6b6860;
  --light-mid:   #a8a5a0;
  --border:      #e2ddd6;
  --brick:       #b5451a;
  --brick-light: #f7ece7;
  --brick-dark:  #8c3312;
}

/* ─── Base ─── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--charcoal) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
  max-width: 860px !important;
  padding: 0 24px 80px 24px !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
  background: var(--warm-white) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 28px 20px 40px !important;
}

.pj-sb-logo {
  font-family: 'Playfair Display', serif;
  font-size: 18px; font-weight: 900;
  color: var(--charcoal); margin-bottom: 28px;
  padding-bottom: 16px; border-bottom: 1px solid var(--border);
}
.pj-sb-logo span { color: var(--brick); }

.pj-sb-section { margin-bottom: 28px; }
.pj-sb-title {
  font-size: 10px; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--light-mid); margin-bottom: 12px;
}

.pj-accuracy-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.pj-accuracy-table th {
  background: var(--cream); padding: 6px 8px;
  text-align: left; font-weight: 600;
  color: var(--mid); font-size: 11px;
  border-bottom: 1px solid var(--border);
}
.pj-accuracy-table td {
  padding: 6px 8px; border-bottom: 1px solid var(--border);
  color: var(--charcoal);
}
.pj-accuracy-table tr:last-child td { font-weight: 600; border-bottom: none; }

.pj-how-item {
  display: flex; align-items: flex-start; gap: 10px;
  margin-bottom: 12px;
}
.pj-how-num {
  width: 22px; height: 22px; border-radius: 50%;
  background: var(--brick); color: #fff;
  font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.pj-how-text { font-size: 12px; color: var(--mid); line-height: 1.5; }

.pj-ficha {
  background: var(--cream); border-radius: 12px;
  padding: 14px 16px;
}
.pj-ficha-row {
  display: flex; justify-content: space-between;
  font-size: 12px; padding: 4px 0;
  border-bottom: 1px solid var(--border);
}
.pj-ficha-row:last-child { border-bottom: none; }
.pj-ficha-key { color: var(--mid); }
.pj-ficha-val { font-weight: 600; color: var(--charcoal); }

/* ─── Logo / Nav ─── */
.pj-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 28px 0 0 0; margin-bottom: 40px;
}
.pj-logo {
  font-family: 'Playfair Display', serif;
  font-size: 26px; font-weight: 900; color: var(--charcoal);
  letter-spacing: -0.5px;
}
.pj-logo span { color: var(--brick); }
.pj-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--brick-light); color: var(--brick-dark);
  border-radius: 100px; padding: 6px 16px;
  font-size: 13px; font-weight: 500;
}
.pj-badge::before {
  content: ''; width: 6px; height: 6px;
  background: var(--brick); border-radius: 50%;
}

/* ─── Hero ─── */
.pj-hero-img {
  position: relative;
  border-radius: 24px; overflow: hidden;
  margin-bottom: 32px; min-height: 300px;
  background: linear-gradient(135deg, #b5451a 0%, #2a2820 100%);
  background-image: url('https://images.unsplash.com/photo-1539037116277-4db20889f2d4?w=1400&q=80&fm=webp');
  background-size: cover; background-position: center 40%;
}
.pj-hero-overlay {
  position: absolute; inset: 0;
  background: linear-gradient(to right, rgba(42,40,32,0.85) 0%, rgba(42,40,32,0.30) 100%);
}
.pj-hero-content {
  position: relative; z-index: 2;
  padding: 52px 52px 48px;
}
.pj-hero-content h1 {
  font-family: 'Playfair Display', serif;
  font-size: clamp(36px, 5vw, 56px);
  font-weight: 900; line-height: 1.05;
  letter-spacing: -1.5px; color: #fff;
  margin-bottom: 14px;
}
.pj-hero-content h1 em {
  font-style: italic; color: #f5b89a;
}
.pj-hero-content p {
  font-size: 16px; color: rgba(255,255,255,0.80);
  line-height: 1.65; max-width: 480px; margin: 0;
}

/* ─── Stat strip ─── */
.pj-stats {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 1px; background: var(--border);
  border-radius: 16px; overflow: hidden;
  margin-bottom: 40px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.pj-stat {
  background: var(--warm-white);
  padding: 20px 24px; text-align: center;
}
.pj-stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 30px; font-weight: 900;
  color: var(--brick); line-height: 1;
}
.pj-stat-lbl {
  font-size: 12px; color: var(--mid);
  margin-top: 5px; line-height: 1.4;
}

/* ─── Section divider ─── */
.pj-divider {
  display: flex; align-items: center; gap: 14px;
  margin: 8px 0 28px;
}
.pj-divider::before, .pj-divider::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}
.pj-divider-icon { font-size: 17px; }
.pj-divider-label {
  font-size: 11px; font-weight: 600; color: var(--mid);
  letter-spacing: 0.07em; text-transform: uppercase; white-space: nowrap;
}

/* ─── Wizard bar ─── */
.pj-wizard-bar {
  display: flex; align-items: center;
  margin-bottom: 28px;
}
.pj-wizard-item {
  display: flex; flex-direction: column; align-items: center;
  flex-shrink: 0;
}
.pj-wizard-connector {
  flex: 1; height: 2px; background: var(--border);
  margin: 0 8px; margin-bottom: 20px;
  transition: background 0.3s;
}
.pj-wizard-dot {
  width: 34px; height: 34px; border-radius: 50%;
  background: var(--border); color: var(--mid);
  font-size: 13px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.2s;
}
.pj-wizard-dot.active {
  background: var(--brick); color: #fff;
  box-shadow: 0 0 0 5px var(--brick-light);
}
.pj-wizard-dot.done {
  background: var(--brick-dark); color: #fff;
  font-size: 14px;
}
.pj-wizard-label {
  font-size: 11px; color: var(--light-mid);
  margin-top: 7px; font-weight: 500; text-align: center;
  white-space: nowrap;
}
.pj-wizard-label.active { color: var(--brick); font-weight: 700; }
.pj-wizard-label.done { color: var(--brick-dark); }

/* ─── Form card ─── */
.pj-card {
  background: var(--warm-white);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 36px 40px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  transition: box-shadow 0.2s;
}
.pj-card-title {
  font-family: 'Playfair Display', serif;
  font-size: 20px; font-weight: 700;
  color: var(--charcoal); margin-bottom: 6px;
}
.pj-card-sub {
  font-size: 14px; color: var(--mid);
  margin-bottom: 28px; line-height: 1.5;
}
.pj-section-label {
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--light-mid); margin: 24px 0 12px 0;
}

/* ─── Streamlit widget overrides ─── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
  font-size: 14px !important; font-weight: 500 !important;
  color: var(--charcoal) !important;
}
div[data-baseweb="select"] > div {
  background: #fff !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-size: 14px !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--brick) !important;
  box-shadow: 0 0 0 3px rgba(181,69,26,0.12) !important;
}
input[type="number"] {
  background: #fff !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-size: 14px !important;
}
input[type="number"]:focus {
  border-color: var(--brick) !important;
  box-shadow: 0 0 0 3px rgba(181,69,26,0.12) !important;
}

/* ─── Radio buttons ─── */
[data-testid="stRadio"] > div { gap: 6px !important; }
[data-testid="stRadio"] div[role="radiogroup"] {
  display: flex; flex-direction: row; gap: 8px !important; flex-wrap: wrap;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
  background: #fff; border: 1px solid var(--border) !important;
  border-radius: 8px; padding: 6px 14px !important;
  font-size: 13px !important; cursor: pointer;
  transition: all 0.15s;
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--brick-light) !important;
  border-color: var(--brick) !important;
  color: var(--brick-dark) !important;
}

/* ─── Primary button ─── */
[data-testid="stButton"] > button[kind="primary"] {
  background: var(--brick) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 100px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 15px !important; font-weight: 600 !important;
  padding: 12px 36px !important;
  height: auto !important;
  transition: background 0.2s, transform 0.15s !important;
  box-shadow: 0 4px 14px rgba(181,69,26,0.30) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  background: var(--brick-dark) !important;
  transform: translateY(-1px) !important;
}

/* ─── Secondary button ─── */
[data-testid="stButton"] > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--charcoal) !important;
  border-radius: 100px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important; font-weight: 500 !important;
  padding: 10px 24px !important;
  height: auto !important;
  transition: border-color 0.15s, background 0.15s !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
  border-color: var(--charcoal) !important;
  background: rgba(42,40,32,0.04) !important;
}

/* ─── Result panel ─── */
.pj-result {
  background: var(--warm-white);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 36px 40px;
  margin-top: 32px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.pj-result-header {
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 28px;
}
.pj-result-icon {
  width: 52px; height: 52px; border-radius: 16px;
  background: var(--brick-light);
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; flex-shrink: 0;
}
.pj-result-title {
  font-family: 'Playfair Display', serif;
  font-size: 22px; font-weight: 900; color: var(--charcoal);
}
.pj-result-sub { font-size: 14px; color: var(--mid); margin-top: 2px; }

.pj-price-block {
  background: var(--cream);
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 20px;
  text-align: center;
}
.pj-price-label {
  font-size: 12px; font-weight: 600;
  color: var(--mid); letter-spacing: 0.05em;
  text-transform: uppercase; margin-bottom: 12px;
}
.pj-price-main {
  font-family: 'Playfair Display', serif;
  font-size: clamp(44px, 6vw, 66px);
  font-weight: 900; color: var(--charcoal);
  letter-spacing: -2px; line-height: 1;
  text-shadow: 0 2px 8px rgba(42,40,32,0.08);
}
.pj-price-currency {
  font-size: 0.45em; vertical-align: super;
  color: var(--brick); margin-right: 6px;
}
.pj-price-range {
  font-size: 14px; color: var(--mid);
  margin-top: 12px; line-height: 1.6;
}
.pj-price-range strong { color: var(--charcoal); }

/* ─── Gauge ─── */
.pj-gauge-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 12px 0 4px;
}
.pj-gauge-sub {
  font-size: 12px; color: var(--mid); margin-top: 4px;
  text-align: center; line-height: 1.4;
}

/* ─── Metrics ─── */
.pj-metrics {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 12px; margin-bottom: 20px;
}
.pj-metric {
  background: var(--cream); border-radius: 14px;
  padding: 16px 20px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
}
.pj-metric-label { font-size: 12px; color: var(--mid); margin-bottom: 4px; }
.pj-metric-value { font-size: 20px; font-weight: 700; color: var(--charcoal); }

/* ─── Comparable ─── */
.pj-comparable {
  background: var(--cream); border-radius: 14px;
  padding: 16px 20px; margin-bottom: 16px;
  font-size: 14px; line-height: 1.75;
}
.pj-comparable-label {
  font-size: 10px; font-weight: 700; color: var(--mid);
  text-transform: uppercase; letter-spacing: 0.07em;
  margin-bottom: 8px;
}
.pj-comparable-delta-up { color: #2e7d32; font-weight: 600; }
.pj-comparable-delta-down { color: var(--brick-dark); font-weight: 600; }

/* ─── Feature chart title ─── */
.pj-chart-title {
  font-size: 10px; font-weight: 700; color: var(--mid);
  text-transform: uppercase; letter-spacing: 0.07em;
  margin-bottom: 4px;
}

/* ─── CTA ─── */
.pj-cta {
  display: flex; align-items: center; gap: 16px;
  background: var(--brick); border-radius: 16px;
  padding: 20px 24px; margin-top: 24px; color: #fff;
}
.pj-cta-icon { font-size: 28px; flex-shrink: 0; }
.pj-cta-body { flex: 1; }
.pj-cta-title {
  font-family: 'Playfair Display', serif;
  font-size: 16px; font-weight: 700; line-height: 1.3;
}
.pj-cta-sub { font-size: 13px; opacity: 0.85; margin-top: 2px; }
.pj-cta-btn {
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.40);
  border-radius: 100px; padding: 9px 22px;
  font-size: 13px; font-weight: 600; white-space: nowrap; flex-shrink: 0;
}

/* ─── Disclaimer ─── */
.pj-disclaimer {
  font-size: 13px; color: var(--light-mid);
  line-height: 1.65; padding-top: 20px;
  border-top: 1px solid var(--border);
  margin-top: 24px;
}

/* ─── Footer ─── */
.pj-footer {
  border-top: 1px solid var(--border);
  margin-top: 64px; padding: 36px 0 52px;
  text-align: center;
}
.pj-footer-logo {
  font-family: 'Playfair Display', serif;
  font-size: 20px; font-weight: 900; color: var(--charcoal);
  margin-bottom: 10px;
}
.pj-footer-logo span { color: var(--brick); }
.pj-footer-text {
  font-size: 12px; color: var(--light-mid);
  line-height: 1.7; margin-bottom: 14px;
}
.pj-footer-badges { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.pj-version-badge {
  background: var(--border); color: var(--mid);
  border-radius: 100px; padding: 3px 12px;
  font-size: 11px; font-weight: 500;
}

/* ─── Warning / info overrides ─── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  font-size: 14px !important;
}

hr { border-color: var(--border) !important; }
</style>
"""

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PrecioJusto · Madrid",
    page_icon="🏠",
    layout="centered",
)
st.markdown(DESIGN_CSS, unsafe_allow_html=True)

model = load_model()
neighborhoods = load_neighborhoods()

# ── Session state init ─────────────────────────────────────────────────────────

if "step" not in st.session_state:
    st.session_state.step = 0
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="pj-sb-logo">Precio<span>Justo</span></div>', unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="pj-sb-section">
      <div class="pj-sb-title">Cómo funciona</div>
      <div class="pj-how-item">
        <div class="pj-how-num">1</div>
        <div class="pj-how-text">Introduce los datos de tu vivienda en el formulario</div>
      </div>
      <div class="pj-how-item">
        <div class="pj-how-num">2</div>
        <div class="pj-how-text">El modelo analiza 25 variables clave del inmueble</div>
      </div>
      <div class="pj-how-item">
        <div class="pj-how-num">3</div>
        <div class="pj-how-text">Obtienes el precio estimado con rango de confianza</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Accuracy by price band
    st.markdown("""
    <div class="pj-sb-section">
      <div class="pj-sb-title">Precisión por rango de precio</div>
      <table class="pj-accuracy-table">
        <thead>
          <tr><th>Rango</th><th>Precisión ±20%</th></tr>
        </thead>
        <tbody>
          <tr><td>≤ 200.000 €</td><td>77%</td></tr>
          <tr><td>200k – 400k €</td><td>78%</td></tr>
          <tr><td>400k – 700k €</td><td>79%</td></tr>
          <tr><td>700k – 1M €</td><td>71%</td></tr>
          <tr><td>1M – 2M €</td><td>63%</td></tr>
          <tr><td>&gt; 2M €</td><td>58%</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Model card
    st.markdown("""
    <div class="pj-sb-section">
      <div class="pj-sb-title">Ficha técnica</div>
      <div class="pj-ficha">
        <div class="pj-ficha-row">
          <span class="pj-ficha-key">Algoritmo</span>
          <span class="pj-ficha-val">CatBoost</span>
        </div>
        <div class="pj-ficha-row">
          <span class="pj-ficha-key">R² (test)</span>
          <span class="pj-ficha-val">0,897</span>
        </div>
        <div class="pj-ficha-row">
          <span class="pj-ficha-key">Viviendas entrenadas</span>
          <span class="pj-ficha-val">21.542</span>
        </div>
        <div class="pj-ficha-row">
          <span class="pj-ficha-key">Error mediano</span>
          <span class="pj-ficha-val">11,1%</span>
        </div>
        <div class="pj-ficha-row">
          <span class="pj-ficha-key">Variables</span>
          <span class="pj-ficha-val">25</span>
        </div>
        <div class="pj-ficha-row">
          <span class="pj-ficha-key">Datos</span>
          <span class="pj-ficha-val">~2024</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Nav ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pj-nav">
  <div class="pj-logo">Precio<span>Justo</span></div>
  <div class="pj-badge">Madrid · 2025</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pj-hero-img">
  <div class="pj-hero-overlay"></div>
  <div class="pj-hero-content">
    <h1>¿Cuánto vale<br>tu <em>vivienda</em>?</h1>
    <p>Estimación inteligente del precio de mercado basada en un modelo de IA entrenado con más de 21.000 viviendas reales en Madrid.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats strip
st.markdown("""
<div class="pj-stats">
  <div class="pj-stat">
    <div class="pj-stat-num">21.542</div>
    <div class="pj-stat-lbl">Viviendas analizadas</div>
  </div>
  <div class="pj-stat">
    <div class="pj-stat-num">88,9%</div>
    <div class="pj-stat-lbl">Precisión mediana</div>
  </div>
  <div class="pj-stat">
    <div class="pj-stat-num">74,6%</div>
    <div class="pj-stat-lbl">Casos dentro de ±20%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Section divider ────────────────────────────────────────────────────────────

st.markdown("""
<div class="pj-divider">
  <span class="pj-divider-icon">🏠</span>
  <span class="pj-divider-label">Datos de la vivienda</span>
</div>
""", unsafe_allow_html=True)

# ── Multi-step form ────────────────────────────────────────────────────────────

step = st.session_state.step
fd = st.session_state.form_data  # shorthand

render_step_indicator(step)

# ── Step 0: Ubicación y superficie ────────────────────────────────────────────

if step == 0:
    st.markdown('<div class="pj-card">', unsafe_allow_html=True)
    st.markdown('<div class="pj-card-title">Ubicación y superficie</div>', unsafe_allow_html=True)
    st.markdown('<div class="pj-card-sub">El barrio y la superficie son los factores más determinantes del precio.</div>', unsafe_allow_html=True)

    saved_subtitle = fd.get("subtitle")
    subtitle_idx = neighborhoods.index(saved_subtitle) if saved_subtitle in neighborhoods else 0
    subtitle = st.selectbox(
        "Barrio *",
        neighborhoods,
        index=subtitle_idx,
        help="El barrio explica el 30% del precio. Selecciona el más cercano al inmueble.",
    )

    col1, col2 = st.columns(2)
    with col1:
        sq_mt_built_input = st.number_input(
            "Superficie construida (m²)",
            min_value=10.0, max_value=2000.0,
            value=float(fd["sq_mt_built"]) if fd.get("sq_mt_built") else None,
            placeholder="Ej. 85",
            help="Superficie total incluyendo paredes. Aparece en la escritura o en el anuncio.",
        )
        n_rooms = st.number_input(
            "Habitaciones *",
            min_value=0, max_value=20,
            value=int(fd.get("n_rooms", 3)),
            help="Número total de dormitorios (sin contar salón ni cocina).",
        )
    with col2:
        sq_mt_useful_input = st.number_input(
            "Superficie útil (m²)",
            min_value=10.0, max_value=2000.0,
            value=float(fd["sq_mt_useful"]) if fd.get("sq_mt_useful") else None,
            placeholder="Ej. 70",
            help="Superficie pisable real (excluye paredes y espacios no habitables).",
        )
        n_bathrooms_input = st.number_input(
            "Baños",
            min_value=0, max_value=10,
            value=int(fd["n_bathrooms"]) if fd.get("n_bathrooms") else None,
            placeholder="Ej. 2",
            help="Incluye aseos completos y/o cuartos de baño.",
        )

    floor_opts = ["No sé / No disponible"] + FLOOR_OPTIONS
    saved_floor = fd.get("floor")
    floor_idx = floor_opts.index(saved_floor) if saved_floor in floor_opts else 0
    floor_option = st.selectbox(
        "Planta",
        floor_opts,
        index=floor_idx,
        help="La planta influye en precio. Los pisos altos y bajos tienen valoraciones distintas.",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Siguiente →", type="primary", use_container_width=False):
        if not subtitle:
            st.warning("Selecciona un barrio para continuar.")
        else:
            st.session_state.form_data.update({
                "subtitle": subtitle,
                "sq_mt_built": float(sq_mt_built_input) if sq_mt_built_input else None,
                "sq_mt_useful": float(sq_mt_useful_input) if sq_mt_useful_input else None,
                "n_rooms": float(n_rooms),
                "n_bathrooms": float(n_bathrooms_input) if n_bathrooms_input else None,
                "floor": floor_option if floor_option != "No sé / No disponible" else None,
            })
            st.session_state.step = 1
            st.rerun()

# ── Step 1: Tipo y estado ──────────────────────────────────────────────────────

elif step == 1:
    st.markdown('<div class="pj-card">', unsafe_allow_html=True)
    st.markdown('<div class="pj-card-title">Tipo y estado del inmueble</div>', unsafe_allow_html=True)
    st.markdown('<div class="pj-card-sub">El tipo de propiedad y su estado afectan directamente al valor de mercado.</div>', unsafe_allow_html=True)

    house_type_opts = ["No sé / No disponible"] + HOUSE_TYPES
    saved_ht = fd.get("house_type_id")
    ht_idx = house_type_opts.index(saved_ht) if saved_ht in house_type_opts else 0

    col3, col4 = st.columns(2)
    with col3:
        house_type_option = st.selectbox(
            "Tipo de inmueble",
            house_type_opts,
            index=ht_idx,
            help="Los áticos y dúplex tienen valoraciones específicas distintas a los pisos estándar.",
        )
        built_year_input = st.number_input(
            "Año de construcción",
            min_value=1800, max_value=2025,
            value=int(fd["built_year"]) if fd.get("built_year") else None,
            placeholder="Ej. 1990",
            help="Las viviendas de construcción más reciente suelen tener mejores eficiencias.",
        )
    with col4:
        st.markdown('<div class="pj-section-label">Características</div>', unsafe_allow_html=True)
        is_renewal_needed = st.checkbox("Necesita reforma", value=bool(fd.get("is_renewal_needed", False)))
        is_exterior = st.checkbox("Es exterior", value=bool(fd.get("is_exterior", False)))
        has_terrace = st.checkbox("Tiene terraza", value=bool(fd.get("has_terrace", False)))
        has_balcony = st.checkbox("Tiene balcón", value=bool(fd.get("has_balcony", False)))

    st.markdown('</div>', unsafe_allow_html=True)

    col_back, col_next, _ = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Atrás", type="secondary"):
            st.session_state.step = 0
            st.rerun()
    with col_next:
        if st.button("Siguiente →", type="primary"):
            st.session_state.form_data.update({
                "house_type_id": house_type_option if house_type_option != "No sé / No disponible" else None,
                "built_year": float(built_year_input) if built_year_input else None,
                "is_renewal_needed": 1.0 if is_renewal_needed else 0.0,
                "is_exterior": 1.0 if is_exterior else 0.0,
                "has_terrace": 1.0 if has_terrace else 0.0,
                "has_balcony": 1.0 if has_balcony else 0.0,
            })
            st.session_state.step = 2
            st.rerun()

# ── Step 2: Equipamiento ───────────────────────────────────────────────────────

elif step == 2:
    st.markdown('<div class="pj-card">', unsafe_allow_html=True)
    st.markdown('<div class="pj-card-title">Equipamiento</div>', unsafe_allow_html=True)
    st.markdown('<div class="pj-card-sub">Si no sabes algún dato, selecciona "No sé" — el modelo lo imputará automáticamente.</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        has_parking = st.checkbox("Garaje incluido", value=bool(fd.get("has_parking", False)))

        saved_lift = fd.get("_lift_radio", "No sé")
        lift_val = st.radio(
            "Ascensor",
            ["Sí", "No", "No sé"],
            index=["Sí", "No", "No sé"].index(saved_lift),
            horizontal=True,
            help="El ascensor es el 4.º factor de importancia en el modelo.",
        )

        saved_ac = fd.get("_ac_radio", "No sé")
        ac_val = st.radio(
            "Aire acondicionado",
            ["Sí", "No", "No sé"],
            index=["Sí", "No", "No sé"].index(saved_ac),
            horizontal=True,
        )

    with col6:
        saved_newdev = fd.get("_newdev_radio", "No sé")
        newdev_val = st.radio(
            "Obra nueva",
            ["Sí", "No", "No sé"],
            index=["Sí", "No", "No sé"].index(saved_newdev),
            horizontal=True,
            help="Las viviendas de obra nueva tienen características fiscales y de mercado diferentes.",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    col_back2, col_est, _ = st.columns([1, 2, 1])
    with col_back2:
        if st.button("← Atrás", type="secondary"):
            st.session_state.form_data.update({
                "_lift_radio": lift_val,
                "_ac_radio": ac_val,
                "_newdev_radio": newdev_val,
                "has_parking": 1.0 if has_parking else 0.0,
            })
            st.session_state.step = 1
            st.rerun()
    with col_est:
        if st.button("Estimar precio  ✓", type="primary", use_container_width=True):
            st.session_state.form_data.update({
                "has_parking": 1.0 if has_parking else 0.0,
                "has_lift": 1.0 if lift_val == "Sí" else (0.0 if lift_val == "No" else None),
                "has_lift_missing_override": lift_val == "No sé",
                "has_ac": 1.0 if ac_val == "Sí" else (0.0 if ac_val == "No" else None),
                "has_ac_missing_override": ac_val == "No sé",
                "is_new_development": 1.0 if newdev_val == "Sí" else (0.0 if newdev_val == "No" else None),
                "is_new_development_missing_override": newdev_val == "No sé",
                "_lift_radio": lift_val,
                "_ac_radio": ac_val,
                "_newdev_radio": newdev_val,
            })
            st.session_state.predicted = True
            st.rerun()

# ── Result ─────────────────────────────────────────────────────────────────────

if st.session_state.predicted:
    data = st.session_state.form_data

    inputs = {
        "subtitle": data.get("subtitle", "missing"),
        "sq_mt_built": data.get("sq_mt_built"),
        "sq_mt_useful": data.get("sq_mt_useful"),
        "n_rooms": data.get("n_rooms", 3.0),
        "n_bathrooms": data.get("n_bathrooms"),
        "floor": data.get("floor"),
        "built_year": data.get("built_year"),
        "house_type_id": data.get("house_type_id"),
        "has_parking": data.get("has_parking", 0.0),
        "has_lift": data.get("has_lift"),
        "has_ac": data.get("has_ac"),
        "is_new_development": data.get("is_new_development"),
        "is_renewal_needed": data.get("is_renewal_needed", 0.0),
        "is_exterior": data.get("is_exterior", 0.0),
        "has_terrace": data.get("has_terrace", 0.0),
        "has_balcony": data.get("has_balcony", 0.0),
    }
    for cat in CAT_FEATURES:
        if inputs.get(cat) is None:
            inputs[cat] = "missing"

    df = build_row(inputs)

    with st.spinner("Analizando tu vivienda…"):
        result = model.predict_interval(df, lower="p10", upper="p90")

    lo, mid_price, hi = float(result[0, 0]), float(result[0, 1]), float(result[0, 2])

    sq = data.get("sq_mt_built") or data.get("sq_mt_useful")
    ppm_str = f"{int(mid_price / sq):,} €/m²".replace(",", ".") if sq else "—"

    subtitle = data.get("subtitle", "")

    # Main result card
    st.markdown(f"""
<div class="pj-result">
  <div class="pj-result-header">
    <div class="pj-result-icon">🏡</div>
    <div>
      <div class="pj-result-title">Resultado del análisis</div>
      <div class="pj-result-sub">Estimación de mercado · {subtitle}</div>
    </div>
  </div>

  <div class="pj-price-block">
    <div class="pj-price-label">Precio estimado de mercado</div>
    <div class="pj-price-main">
      <span class="pj-price-currency">€</span>{fmt_eur(mid_price).replace(' €', '')}
    </div>
    <div class="pj-price-range">
      Rango orientativo (80%): <strong>{fmt_eur(lo)}</strong> — <strong>{fmt_eur(hi)}</strong>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Metrics row
    st.markdown(f"""
<div class="pj-result" style="margin-top:12px; padding: 24px 40px;">
  <div class="pj-metrics">
    <div class="pj-metric">
      <div class="pj-metric-label">Precio / m²</div>
      <div class="pj-metric-value">{ppm_str}</div>
    </div>
    <div class="pj-metric">
      <div class="pj-metric-label">Precisión global del modelo</div>
      <div class="pj-metric-value">75% casos ±20%</div>
    </div>
  </div>
""", unsafe_allow_html=True)

    # Gauge SVG
    r2_pct = 89.7
    track_len = 314.16
    filled_len = track_len * (r2_pct / 100)
    offset = track_len - filled_len
    st.markdown(f"""
  <div class="pj-gauge-wrap">
    <svg viewBox="0 0 240 130" width="220">
      <path d="M 20 120 A 100 100 0 0 1 220 120"
            fill="none" stroke="#e2ddd6" stroke-width="14" stroke-linecap="round"/>
      <path d="M 20 120 A 100 100 0 0 1 220 120"
            fill="none" stroke="#b5451a" stroke-width="14" stroke-linecap="round"
            stroke-dasharray="{track_len:.1f}"
            stroke-dashoffset="{offset:.1f}"/>
      <text x="120" y="104" text-anchor="middle"
            font-family="Playfair Display, serif"
            font-size="28" font-weight="900" fill="#2a2820">R²&nbsp;0,90</text>
      <text x="120" y="122" text-anchor="middle"
            font-family="DM Sans, sans-serif"
            font-size="11" fill="#6b6860">Varianza explicada por el modelo</text>
    </svg>
    <div class="pj-gauge-sub">El modelo explica el 89,7% de la variabilidad de precios</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Comparable with neighborhood
    median_sqm = NEIGHBORHOOD_MEDIANS.get(subtitle)
    if median_sqm and sq:
        user_sqm = int(mid_price / sq)
        delta = user_sqm - median_sqm
        delta_str = f"+{abs(delta):,}".replace(",", ".")
        delta_class = "pj-comparable-delta-up" if delta > 0 else "pj-comparable-delta-down"
        direction = "por encima" if delta > 0 else "por debajo"
        if delta < 0:
            delta_str = f"−{abs(delta):,}".replace(",", ".")
        st.markdown(f"""
<div class="pj-comparable">
  <div class="pj-comparable-label">Comparativa con el barrio</div>
  Precio medio en <strong>{subtitle}</strong>: <strong>{median_sqm:,} €/m²</strong>
  &nbsp;·&nbsp;
  Tu vivienda: <strong>{user_sqm:,} €/m²</strong>
  &nbsp;
  <span class="{delta_class}">({delta_str} €/m² {direction} de la media)</span>
</div>
""".replace(",", "."), unsafe_allow_html=True)

    # Feature importance chart
    st.markdown('<div class="pj-result" style="margin-top:12px; padding: 24px 40px 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="pj-chart-title">Variables más influyentes en el precio</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    render_feature_chart()

    # Disclaimer
    st.markdown("""
<div class="pj-result" style="margin-top:12px; padding: 24px 40px;">
  <div class="pj-disclaimer">
    Esta estimación es orientativa y no equivale a una tasación oficial (RD 775/1997). El modelo
    tiene una precisión del ±20% en el 75% de los casos. La fiabilidad es menor en viviendas de
    lujo (&gt; 1M €) y en barrios con pocas transacciones. Datos de mercado aproximados a 2024.
  </div>
</div>
""", unsafe_allow_html=True)

    # CTA
    st.markdown("""
<div class="pj-cta">
  <div class="pj-cta-icon">🤝</div>
  <div class="pj-cta-body">
    <div class="pj-cta-title">¿Quieres comprar o vender en Madrid?</div>
    <div class="pj-cta-sub">Contacta con un agente especializado para una valoración oficial</div>
  </div>
  <div class="pj-cta-btn">Contactar</div>
</div>
""", unsafe_allow_html=True)

    # Reset button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Nueva estimación", type="secondary"):
        st.session_state.step = 0
        st.session_state.form_data = {}
        st.session_state.predicted = False
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pj-footer">
  <div class="pj-footer-logo">Precio<span>Justo</span></div>
  <div class="pj-footer-text">
    Datos de viviendas en venta en Madrid · Modelo entrenado con datos de mercado ~2024<br>
    Esta herramienta es orientativa. No equivale a una tasación oficial (RD 775/1997).
  </div>
  <div class="pj-footer-badges">
    <span class="pj-version-badge">v1.0.0</span>
    <span class="pj-version-badge">CatBoost</span>
    <span class="pj-version-badge">Python 3.13</span>
    <span class="pj-version-badge">R² 0,897</span>
  </div>
</div>
""", unsafe_allow_html=True)
