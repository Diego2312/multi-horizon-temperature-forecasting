import json, argparse
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from src.utils.ts_features import add_time_features
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

def train_direct(df_feat, date_col, target_col, horizons, results_path, fi_path):
    feature_cols = [c for c in df_feat.columns if c not in (date_col, target_col)]
    X_all = df_feat[feature_cols].astype('float32').copy()
    y_all = df_feat[target_col].astype('float32').copy()
    metrics = {str(h): {"folds": []} for h in horizons}
    feature_importances = {}
    for h in horizons:
        y_shift = y_all.shift(-h).rename('target_h')
        aligned = pd.concat([X_all, y_shift], axis=1).dropna()
        X_h = aligned[feature_cols].reset_index(drop=True)
        y_h = aligned['target_h'].to_numpy()
        n = len(aligned)
        gains_accum = {f: [] for f in feature_cols}
        for fold, (tr, te) in enumerate(rolling_origin_splits(n, n_splits=5, test_size=max(30, 7*h))):
            model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)
            model.fit(X_h.iloc[tr], y_h[tr])
            pred = model.predict(X_h.iloc[te])
            mae = mean_absolute_error(y_h[te], pred)
            rmse = np.sqrt(mean_squared_error(y_h[te], pred))
            metrics[str(h)]["folds"].append({"mae": float(mae), "rmse": float(rmse)})
            gain_map = model.get_booster().get_score(importance_type='gain')
            for f in feature_cols:
                gains_accum[f].append(float(gain_map.get(f, 0.0)))
        folds = metrics[str(h)]["folds"]
        metrics[str(h)]["mean_mae"] = float(np.mean([f["mae"] for f in folds])) if folds else None
        metrics[str(h)]["mean_rmse"] = float(np.mean([f["rmse"] for f in folds])) if folds else None
        feature_importances[str(h)] = {f: float(np.mean(v)) if v else 0.0 for f, v in gains_accum.items()}
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(metrics, indent=2))
    Path(fi_path).parent.mkdir(parents=True, exist_ok=True)
    Path(fi_path).write_text(json.dumps(feature_importances, indent=2))
    print(f"Saved results to {results_path}")
    print(f"Saved feature importances to {fi_path}")

def main(args):
    df = pd.read_csv(args.data_path)
    assert 'DATE' in df.columns and 'TAVG' in df.columns
    df = preprocess_data(df, args.start_year, args.end_year)
    date_col = 'DATE'; target_col = 'TAVG'
    df_feat = add_time_features(df, date_col, target_col, max_lag=14)
    horizons = [int(h) for h in args.horizons]
    train_direct(df_feat, date_col, target_col, horizons, args.results_path, args.fi_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--horizons", nargs="+", default=[1,3,7,14,30])
    p.add_argument("--results_path", type=str, default="reports/metrics/xgb_results.json")
    p.add_argument("--fi_path", type=str, default="reports/metrics/xgb_feature_importance.json")
    p.add_argument("--start_year", type=int, default=1990)
    p.add_argument("--end_year", type=int, default=2020)
    args = p.parse_args()
    main(args)
