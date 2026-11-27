# Earthquake Aftershock Forecasting

A Streamlit dashboard for exploring aftershock patterns from the official **USGS Earthquake Catalog**. It combines the original neural-network comparison view with the full app-first workspace: live CSV feeds, a bundled Random Forest estimator, a transparent baseline, a spatial probability map, catalog evidence, exports, and local relief tools.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, usually <http://localhost:8501>.

## Features

- **Live USGS CSV** — past day, past week, or past month; bundled CSV fallback and local `.csv` import
- **Engine selector** — bundled Random Forest v1.0 or transparent catalog-calibrated baseline v0.2
- **Scenario controls** — mainshock from catalog, magnitude, depth, elapsed hours, forecast window, and region context
- **Spatial probability surface** — 40 × 25 grid over a 3 km cell layout with observed events and inferred trend
- **Priority zones** — top sectors ranked by modelled `P(≥1 M3+ event)`
- **Expected daily rate** — Omori-type decay chart
- **Catalog evidence** — observed USGS rows around the selected mainshock
- **Exports** — map PNG, grid CSV, observed CSV
- **Method & limits** — model details and responsible-use notes
- **Relief & recovery** — official links, local-only checklist, printable contact card
- **Model comparison** — the original 118-earthquake neural-network vs baseline view

## Model notes

The default engine is a compact Random Forest exported to `assets/ml-model.json` and evaluated in Python. It predicts the expected count of `M≥3` aftershocks in a chosen window from magnitude, depth, elapsed time, Omori decay, early observed counts, and inferred trend. A spatial kernel distributes that expected count over the grid.

The transparent baseline uses an Omori–Utsu time decay, an anisotropic spatial kernel, and a productivity scale from observed `M≥3` counts in the CSV.

A time-ordered 80/20 holdout on one USGS monthly catalog reports:

| Metric | Random Forest | Baseline |
| --- | ---: | ---: |
| MAE, M≥3 count | 0.3789 | 8.4350 |
| Brier score | 0.1258 | 0.7641 |

This is a same-split prototype comparison only. It is not an accuracy claim or operational validation.

## Original dataset

The model-comparison view uses the original 118 mainshock records under `data/processed/`, with fault direction estimated from the aftershock cloud and fault length approximated from magnitude. Stress values are simplified estimates, not full Coulomb calculations.

## Responsible use

This tool must not be used for emergency response, evacuation orders, or public-safety decisions. The relief section links to official guidance and stores a checklist only in the browser session; it never submits data to an agency.
