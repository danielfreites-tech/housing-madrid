# PrecioJusto · Estimador de precio de vivienda en Madrid

Aplicación web que estima el precio de compra de una vivienda en Madrid mediante un modelo de Machine Learning entrenado con más de 21.000 viviendas reales. El usuario introduce los datos del inmueble en un formulario guiado y obtiene un precio estimado con intervalo de confianza.

---

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

| Capa        | Tecnología                           |
|-------------|--------------------------------------|
| Aplicación  | Streamlit (Python 3.13)              |
| Modelo IA   | CatBoost                             |
| Visualización | Plotly                             |
| Dataset     | houses_Madrid.csv — 21.742 viviendas |

---

## Rendimiento del modelo (test set, n = 4.309)

| Métrica       | Valor      |
|---------------|------------|
| R²            | 0,897      |
| MAE           | 104.274 €  |
| Error mediano | 38.054 €   |
| Casos ±20%    | 74,6 %     |

> CatBoost seleccionado por su manejo nativo de categóricas y nulos, y mayor cobertura ±20% frente a LightGBM, XGBoost y Random Forest.

---

## Estructura del proyecto

```
├── app.py                   # Aplicación Streamlit
├── requirements.txt
├── data/                    # Dataset (no versionado por tamaño)
└── model/
    ├── notebooks/           # 01_eda_model_selection · 02_training_final
    ├── artifacts/           # model.pkl, metrics_report.json, neighborhoods.json, ...
    └── training_utils.py    # Wrapper PricePredictionModel
```

---

## Dataset

`data/houses_Madrid.csv` — 21.742 registros de viviendas en venta en Madrid. No incluido en el repositorio por tamaño. El modelo usa 25 features: 16 variables base + 9 indicadores de valores nulos.

---

> Esta herramienta es orientativa y no equivale a una tasación oficial (RD 775/1997).
