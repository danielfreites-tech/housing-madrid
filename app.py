import sys
import os

# training_utils.py lives in model/ — must be on sys.path before unpickling model.pkl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st

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

# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PrecioJusto Madrid", page_icon="🏠", layout="centered")

st.title("🏠 PrecioJusto Madrid")
st.caption("Estimación orientativa del precio de compra de una vivienda en Madrid, impulsada por un modelo de IA entrenado con datos reales.")

model = load_model()
neighborhoods = load_neighborhoods()

st.divider()
st.subheader("Datos de la vivienda")

col1, col2 = st.columns(2)

with col1:
    subtitle = st.selectbox("Barrio *", neighborhoods)

    sq_mt_built_input = st.number_input(
        "Superficie construida (m²)", min_value=10, max_value=2000, value=None,
        placeholder="Ej. 85", help="Déjalo vacío si no lo sabes"
    )

    sq_mt_useful_input = st.number_input(
        "Superficie útil (m²)", min_value=10, max_value=2000, value=None,
        placeholder="Ej. 70", help="Déjalo vacío si no lo sabes"
    )

    n_rooms = st.number_input("Número de habitaciones *", min_value=0, max_value=20, value=3)

    n_bathrooms_input = st.number_input(
        "Número de baños", min_value=0, max_value=10, value=None,
        placeholder="Ej. 2", help="Déjalo vacío si no lo sabes"
    )

    floor_option = st.selectbox(
        "Planta", ["No sé / No disponible"] + FLOOR_OPTIONS
    )

    built_year_input = st.number_input(
        "Año de construcción", min_value=1800, max_value=2025, value=None,
        placeholder="Ej. 1990", help="Déjalo vacío si no lo sabes"
    )

with col2:
    house_type_option = st.selectbox(
        "Tipo de inmueble", ["No sé / No disponible"] + HOUSE_TYPES
    )

    st.markdown("**Características**")
    has_parking = st.checkbox("Garaje incluido")

    lift_unknown = st.checkbox("Ascensor — no sé", value=False)
    has_lift = st.checkbox("Tiene ascensor", disabled=lift_unknown)

    ac_unknown = st.checkbox("Aire acondicionado — no sé", value=False)
    has_ac = st.checkbox("Tiene aire acondicionado", disabled=ac_unknown)

    new_dev_unknown = st.checkbox("Obra nueva — no sé", value=False)
    is_new_development = st.checkbox("Es obra nueva", disabled=new_dev_unknown)

    is_renewal_needed = st.checkbox("Necesita reforma")
    is_exterior = st.checkbox("Es exterior")
    has_terrace = st.checkbox("Tiene terraza")
    has_balcony = st.checkbox("Tiene balcón")

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────

if st.button("Estimar precio", type="primary", use_container_width=True):
    inputs = {
        "subtitle": subtitle,
        "sq_mt_built": float(sq_mt_built_input) if sq_mt_built_input is not None else None,
        "sq_mt_useful": float(sq_mt_useful_input) if sq_mt_useful_input is not None else None,
        "n_rooms": float(n_rooms),
        "n_bathrooms": float(n_bathrooms_input) if n_bathrooms_input is not None else None,
        "floor": floor_option if floor_option != "No sé / No disponible" else None,
        "built_year": float(built_year_input) if built_year_input is not None else None,
        "house_type_id": house_type_option if house_type_option != "No sé / No disponible" else None,
        "has_parking": 1.0 if has_parking else 0.0,
        "has_lift": None if lift_unknown else (1.0 if has_lift else 0.0),
        "has_ac": None if ac_unknown else (1.0 if has_ac else 0.0),
        "is_new_development": None if new_dev_unknown else (1.0 if is_new_development else 0.0),
        "is_renewal_needed": 1.0 if is_renewal_needed else 0.0,
        "is_exterior": 1.0 if is_exterior else 0.0,
        "has_terrace": 1.0 if has_terrace else 0.0,
        "has_balcony": 1.0 if has_balcony else 0.0,
    }

    # Replace None categorical values with "missing" string for CatBoost
    for cat in CAT_FEATURES:
        if inputs.get(cat) is None:
            inputs[cat] = "missing"

    df = build_row(inputs)

    result = model.predict_interval(df, lower="p10", upper="p90")
    lo, mid, hi = float(result[0, 0]), float(result[0, 1]), float(result[0, 2])

    st.divider()
    st.subheader("Estimación de precio")

    st.metric(label="Precio estimado", value=fmt_eur(mid))

    st.info(f"**Rango orientativo (80%):** {fmt_eur(lo)} — {fmt_eur(hi)}")

    st.caption(
        "Esta estimación es orientativa y no equivale a una tasación oficial. "
        "El modelo tiene una precisión de ±20% en el 75% de los casos. "
        "La fiabilidad es menor en viviendas de lujo (>1M €) y algunos barrios."
    )
