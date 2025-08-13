import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.cv import rolling_origin_splits

def preprocess_data(df, start_year=1990, end_year=2020):
    df = df[['DATE','TAVG']].copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[df['DATE'].dt.year.between(start_year, end_year)].copy()
    if df['TAVG'].abs().median() > 80:
        df['TAVG'] = df['TAVG'] / 10.0
    df = df.sort_values('DATE').reset_index(drop=True)
    df['TAVG'] = df['TAVG'].interpolate('linear')
    return df

def seasonal_naive(series, horizon):
    return series.shift(365)

def persistence(series, horizon):
    return series.shift(1)

def evaluate_baselines(df, horizons, out_path, n_splits=5):
    s = df['TAVG'].reset_index(drop=True)
    n = len(s)
    results = {}
    for h in horizons:
        results[str(h)] = {"persistence": {"folds":[]}, "seasonal": {"folds": []}}
        y = s.shift(-h)
        for fold, (tr, te) in enumerate(rolling_origin_splits(n - h, n_splits=n_splits, test_size=max(30, 7*h))):
            te_idx = te
            y_te = y.iloc[te_idx].dropna()
            if y_te.empty:
                continue
            pred_persist = persistence(s, h).iloc[te_idx].reindex(y_te.index)
            pred_season = seasonal_naive(s, h).iloc[te_idx].reindex(y_te.index)
            mask_p = (~pred_persist.isna()) & (~y_te.isna())
            mask_s = (~pred_season.isna()) & (~y_te.isna())
            if mask_p.any():
                mae = mean_absolute_error(y_te[mask_p], pred_persist[mask_p])
                rmse = np.sqrt(mean_squared_error(y_te[mask_p], pred_persist[mask_p]))
                results[str(h)]["persistence"]["folds"].append({"mae": float(mae), "rmse": float(rmse)})
            if mask_s.any():
                mae = mean_absolute_error(y_te[mask_s], pred_season[mask_s])
                rmse = np.sqrt(mean_squared_error(y_te[mask_s], pred_season[mask_s]))
                results[str(h)]["seasonal"]["folds"].append({"mae": float(mae), "rmse": float(rmse)})
        for name in ("persistence","seasonal"):
            folds = results[str(h)][name]["folds"]
            if folds:
                results[str(h)][name]["mean_mae"] = float(np.mean([f["mae"] for f in folds]))
                results[str(h)][name]["mean_rmse"] = float(np.mean([f["rmse"] for f in folds]))
            else:
                results[str(h)][name]["mean_mae"] = None
                results[str(h)][name]["mean_rmse"] = None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Saved baselines to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--horizons', nargs='+', default=[1,3,7,14,30], type=int)
    ap.add_argument('--start_year', type=int, default=1990)
    ap.add_argument('--end_year', type=int, default=2020)
    ap.add_argument('--out_path', default='reports/metrics/baselines.json')
    args = ap.parse_args()
    df = pd.read_csv(args.data_path)
    df = preprocess_data(df, args.start_year, args.end_year)
    evaluate_baselines(df, args.horizons, args.out_path)
if __name__ == '__main__':
    main()
