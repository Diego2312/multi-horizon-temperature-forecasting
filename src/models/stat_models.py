# src/analysis/stat_models.py

import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional seasonal support
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def preprocess_data(df, start_year=1990, end_year=2020):
    df = df[['DATE','TAVG']].copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[df['DATE'].dt.year.between(start_year, end_year)].copy()
    # unit fix
    if df['TAVG'].abs().median() > 80:
        df['TAVG'] = df['TAVG'] / 10.0
    df = df.sort_values('DATE').reset_index(drop=True)
    # interpolate small gaps
    df['TAVG'] = df['TAVG'].interpolate('linear')
    return df


def single_split_indices(n, horizon, test_size):
    """
    For y = s.shift(-horizon), the last 'test_size' valid points are indices [n - test_size - horizon : n - horizon).
    We train on everything before that window.
    """
    test_start = max(0, n - test_size - horizon)
    test_end = max(0, n - horizon)
    train_end = test_start
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, test_end)
    return train_idx, test_idx


def fit_forecast(train_series, steps, order, seasonal_order=None):
    """
    Fit ARIMA or SARIMAX depending on seasonal_order. Enforce stationarity/invertibility for stability.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")
        warnings.filterwarnings("ignore", category=UserWarning)

        if seasonal_order is not None:
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            fit = model.fit(disp=False)
        else:
            model = ARIMA(
                train_series,
                order=order,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            fit = model.fit(method_kwargs={"warn_convergence": False})

    # Forecast 'steps' steps ahead
    fc = fit.forecast(steps=steps)
    return fc


def eval_stat_models_single(df, horizons, out_path, order=(2,1,2),
                            seasonal_order=None, base_test_size=None):
    """
    Single hold-out per horizon:
      - test_size = base_test_size if provided else max(30, 7*h)
      - train on everything before the test window
      - evaluate on the test window for y = s.shift(-h)
    """
    s = df['TAVG'].reset_index(drop=True).astype('float32')
    n = len(s)
    results = {}

    for h in horizons:

        print(f'Training stat model horizon {h}')

        test_size = int(base_test_size) if base_test_size is not None else max(30, 7*h)

        # Build aligned target
        y = s.shift(-h)

        # Build indices that respect the shift
        tr, te = single_split_indices(n, h, test_size)

        # If not enough data, skip
        if len(tr) < max(order[0], order[2]) + 2 or len(te) == 0:
            results[str(h)] = {"mean_mae": None, "mean_rmse": None, "info": "Insufficient data for this h"}
            continue

        train_series = s.iloc[tr]
        # Forecast len(te) + h, then take the h-step ahead aligned with te:
        # Simpler: directly forecast len(te) steps from the end of train, then compare with y on te.
        try:
            pred = fit_forecast(train_series, steps=len(te), order=order, seasonal_order=seasonal_order)
        except Exception:
            # Fallback simpler order if needed
            try:
                pred = fit_forecast(train_series, steps=len(te), order=(1,1,1), seasonal_order=seasonal_order)
            except Exception as e2:
                results[str(h)] = {"mean_mae": None, "mean_rmse": None, "info": f"Fit failed: {e2}"}
                continue

        # Align prediction with y on test indices
        y_true = y.iloc[te]
        # Drop any NA alignment (e.g., at the series tail)
        mask = (~y_true.isna()) & (~pd.Series(pred).isna())
        if mask.any():
            y_al = y_true[mask].to_numpy()
            p_al = np.asarray(pred)[mask.values]
            mae = mean_absolute_error(y_al, p_al)
            rmse = np.sqrt(mean_squared_error(y_al, p_al))
        else:
            mae, rmse = None, None

        results[str(h)] = {
            "mean_mae": float(mae) if mae is not None else None,
            "mean_rmse": float(rmse) if rmse is not None else None,
            "test_size": int(test_size),
            "order": list(order),
            "seasonal_order": list(seasonal_order) if seasonal_order is not None else None,
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Saved ARIMA/SARIMAX single-split metrics to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--horizons', nargs='+', default=[1,3,7,14,30], type=int)
    ap.add_argument('--start_year', type=int, default=1990)
    ap.add_argument('--end_year', type=int, default=2020)
    # ARIMA(p,d,q)
    ap.add_argument('--order', nargs=3, type=int, default=(2,1,2))
    # Optional seasonal (P,D,Q,s), e.g., 1 0 1 365; if omitted, uses plain ARIMA
    ap.add_argument('--seasonal_order', nargs=4, type=int, default=None)
    # Override test size; if not set, uses max(30, 7*h)
    ap.add_argument('--test_size', type=int, default=None)
    ap.add_argument('--out_path', default='reports/metrics/arima.json')
    args = ap.parse_args()

    df = pd.read_csv(args.data_path)
    df = preprocess_data(df, args.start_year, args.end_year)

    seasonal_order = tuple(args.seasonal_order) if args.seasonal_order is not None else None
    eval_stat_models_single(
        df,
        horizons=args.horizons,
        out_path=args.out_path,
        order=tuple(args.order),
        seasonal_order=seasonal_order,
        base_test_size=args.test_size
    )


if __name__ == '__main__':
    main()
