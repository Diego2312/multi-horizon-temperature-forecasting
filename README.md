# Multi-Horizon Temperature Forecasting: A Case Study with Rome Ciampino Data


## Overview
This project is a **methodological exploration** of multi-horizon time series forecasting for daily mean temperature.  
Using 30 years of data from the **Rome Ciampino** meteorological station (NOAA GHCN-Daily), I investigate:

- How forecast skill changes as prediction horizons increase (1, 3, 7, 14, 30 days ahead).
- How feature importance shifts from recent lags to seasonal patterns for longer horizons.
- The differences between **statistical baselines** and **machine learning models** in a multi-horizon setting.

The goal is to **demonstrate the end-to-end process** of multi-horizon forecasting, from time series exploration to model training, evaluation, and interpretation.

## Key Insights Explored
- **Forecast skill decay:** Error metrics (MAE, RMSE, MASE) increase with horizon length.
- **Feature importance shift:** XGBoost feature importance reveals lag features dominate short-term forecasts, while seasonal features gain importance for longer horizons.
- **Model comparisons:** Statistical models (persistence, seasonal-naive, ARIMA/SARIMA) vs. ML models (XGBoost with engineered time features).

## Project Structure

├── data/                 # Raw data
│   ├── README.md          # Instructions to download/process
├── src/
│   ├── analysis/          # EDA, stationarity tests, STL decomposition and baselines 
│   ├── models/            # ML and statistical forecasting script
│   ├── evaluation/        # Plotting and metrics aggregation
├── reports/               # Generated metrics and plots
├── configs/               # Model and feature configuration files
├── requirements.txt
├── README.md

## Data
- Source: NOAA GHCN daily (Rome Ciampino station). See `data/README.md` for how to fetch/process.
- Target: Daily mean temperature (TAVG), °C.

## Methods
1. **Time Series Analysis** – rolling statistics, stationarity tests (ADF), STL decomposition, ACF/PACF analysis.
2. **Baselines** – persistence and seasonal-naive.
3. **Statistical Models** – ARIMA/SARIMA.
4. **Machine Learning** – XGBoost with lagged features, rolling statistics, and seasonal encodings.
5. **Evaluation** – rolling-origin cross-validation, per-horizon metrics, and feature importance analysis.


## Get started
```bash
pip install -r requirements.txt
```

## Analysis
```bash
python -m src.analysis.explore_ts --data_path data/Raw_dataset.csv --start_year 1990 --end_year 2020
```

## Baselines
```bash
python -m src.analysis.baselines --data_path data/Raw_dataset.csv --horizons 1 3 7 14 30
```

## Statistical Model
```bash
python -m src.analysis.stat_models --data_path data/Raw_dataset.csv --horizons 1 3 7 14 30 --order 2 0 2
```

## XGBoost + Feature Importance
```bash
python -m src.models.run_xgb_multi_horizon --data_path data/Raw_dataset.csv --horizons 1 3 7 14 30
```

## Plotting
```bash
python -m src.evaluation.plot_results
python -m src.evaluation.plot_feature_importance
```

## Results
See `reports/` for generated figures: metric vs horizon and feature importance vs horizon.

- Short horizons (H ≤ 7) rely heavily on recent lag features.
- Longer horizons (H ≥ 14) see seasonal features (sin/cos day-of-year) gain importance.

## Reproducibility
- Python 3.11
- Deterministic seeds where applicable.
- See `configs/` for experiment knobs.

## Limitations & Next Steps
- Add second station for generalization.
- Try LightGBM/linear baselines.
- Calibrate prediction intervals.
- Hyperparameter optimization per horizon.
- Model explainability via SHAP to complement feature importance.
- Incorporate exogenous predictors (humidity, wind, pressure).
