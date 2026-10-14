# UAC Program: Care Load & Placement Demand Forecasting

## Run the dashboard
    pip install -r requirements.txt
    streamlit run app.py
The dashboard reads the precomputed backtest and forecast files in `artifacts/` (already included).

## Reproduce everything
    python pipeline.py        # prep, features, walk-forward backtest, KPIs, final forecasts (~10 min, 1 CPU)
    python make_figures.py    # paper figures + EDA statistics
    python paper_data.py && node build_docs.js   # paper + executive summary (needs `npm i docx`)

## Files
- `data/` source data converted from the supplied .xlsx to CSV
- `pipeline.py` all modelling code; `app.py` Streamlit dashboard
- `artifacts/` cleaned data, decomposition, backtest (every forecast vs actual), metrics, KPIs, calibration, final forecasts
- To refresh with new data: replace the CSV (same columns), rerun `pipeline.py`.

## Key assumptions
Missing days are interpolated and excluded from scoring. Surge = >=5% rise in care load within 28 days.
Capacity threshold is user-set (default 115% of the latest care load). Scenarios are flow-balance arithmetic, not causal estimates.
