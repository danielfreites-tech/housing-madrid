# housing-madrid

Aplicación web full-stack que predice el precio de compra de una vivienda en Madrid.  
El usuario introduce los datos de su vivienda, el backend los pasa al modelo de ML entrenado y devuelve el precio estimado con un intervalo de confianza.

---

## Stack tecnológico

| Capa      | Tecnología                              |
|-----------|-----------------------------------------|
| Frontend  | React                                   |
| Backend   | FastAPI (Python 3.13)                   |
| Modelo IA | CatBoost (principal)                    |
| Dataset   | houses_Madrid.csv — 21 742 viviendas    |

---

## Estado del proyecto

| Fase         | Estado       |
|--------------|--------------|
| Modelo / EDA | ✅ Completado |
| Backend API  | 🔜 Pendiente  |
| Frontend     | 🔜 Pendiente  |

---

## Modelo

### Dataset
- **Fuente:** `houses_Madrid.csv` — 21 742 registros de venta en Madrid
- **Preprocesamiento:** eliminación de 200 outliers extremos, flags de valores nulos como features
- **Split:** 72% train · 8% validación · 20% test

### Comparativa de modelos (test set, n = 4 309)

| Modelo        | R²     | MAE (€)   | MedAE (€) | ≤20% error |
|---------------|--------|-----------|-----------|------------|
| **CatBoost**  | 0.897  | 104 274   | 38 054    | 74.6 %     |
| LightGBM      | 0.897  | 106 644   | 40 026    | 73.3 %     |
| XGBoost       | 0.887  | 116 779   | 47 745    | 65.9 %     |
| Random Forest | 0.877  | 123 109   | 53 073    | 63.7 %     |
| Dummy median  | −0.14  | 418 989   | 214 000   | 14.9 %     |

> Criterio de selección: mayor cobertura ≤20% de error, desempate por menor MAE en euros.

### Features del modelo final (25)

**Base (16):** `sq_mt_built`, `sq_mt_useful`, `n_rooms`, `n_bathrooms`, `floor`, `subtitle` (barrio), `has_lift`, `has_ac`, `has_parking`, `built_year`, `is_new_development`, `house_type_id`, `is_renewal_needed`, `is_exterior`, `has_terrace`, `has_balcony`

**Missing indicators (9):** flags binarios para campos con nulos frecuentes (`sq_mt_built_missing`, `sq_mt_useful_missing`, `n_bathrooms_missing`, `floor_missing`, `has_lift_missing`, `has_ac_missing`, `built_year_missing`, `is_new_development_missing`, `house_type_id_missing`)


### Artefactos exportados (`model/artifacts/`)

| Archivo                    | Descripción                                          |
|----------------------------|------------------------------------------------------|
| `model.cbm`                | Modelo CatBoost serializado (no versionado en git)   |
| `feature_columns.json`     | Lista de features y categóricas que espera el modelo |
| `metrics_report.json`      | Métricas completas train/val/test + segmentos        |
| `model_selection_summary.json` | Comparativa de todos los modelos evaluados       |
| `neighborhoods.json`       | Mapeo de IDs de barrio a nombre legible              |

---

## Estructura del repositorio

```
├── data/                    # Dataset (CSV no versionado)
├── model/
│   ├── notebooks/           # 01_eda_model_selection.ipynb · 02_training_final.ipynb
│   ├── artifacts/           # JSON de métricas y features (model.cbm excluido)
│   └── training_utils.py    # PricePredictionModel wrapper
├── backend/                 # FastAPI — pendiente
│   └── app/
└── frontend/                # React — pendiente
    └── src/
```

---



> El dataset `data/houses_Madrid.csv` y el binario `model/artifacts/model.cbm` están excluidos del repositorio por tamaño.
