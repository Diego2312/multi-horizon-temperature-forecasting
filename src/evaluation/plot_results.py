import json, argparse
from pathlib import Path
import matplotlib.pyplot as plt

def safe_load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

def extract_curve(d, key):
    xs, ys = [], []
    for h_str, vals in d.items():
        h = int(h_str)
        m = vals.get(key)
        if m is not None:
            xs.append(h); ys.append(m)
    xs, ys = zip(*sorted(zip(xs, ys))) if xs else ([], [])
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xgb_metrics', default='reports/metrics/xgb_results.json')
    ap.add_argument('--baseline_metrics', default='reports/metrics/baselines.json')
    ap.add_argument('--arima_metrics', default='reports/metrics/arima.json')
    ap.add_argument('--outdir', default='reports/plots')
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    xgb = safe_load(args.xgb_metrics)
    base = safe_load(args.baseline_metrics)
    arim = safe_load(args.arima_metrics)
    plt.figure()
    if xgb: 
        x, y = extract_curve(xgb, 'mean_rmse'); plt.plot(x, y, marker='o', label='XGBoost')
    if base:
        xb, ybp = extract_curve({k:v['persistence'] for k,v in base.items()}, 'mean_rmse')
        xs, ybs = extract_curve({k:v['seasonal'] for k,v in base.items()}, 'mean_rmse')
        if xb: plt.plot(xb, ybp, marker='o', label='Persistence')
        if xs: plt.plot(xs, ybs, marker='o', label='Seasonal-naive')
    if arim:
        xa, ya = extract_curve(arim, 'mean_rmse')
        if xa: plt.plot(xa, ya, marker='o', label='ARIMA')
    plt.xlabel('Horizon (days)'); plt.ylabel('RMSE'); plt.title('RMSE vs Horizon'); plt.grid(True); plt.legend()
    plt.savefig(outdir / 'rmse_vs_horizon.png', bbox_inches='tight'); plt.close()
    plt.figure()
    if xgb: 
        x, y = extract_curve(xgb, 'mean_mae'); plt.plot(x, y, marker='o', label='XGBoost')
    if base:
        xb, ybp = extract_curve({k:v['persistence'] for k,v in base.items()}, 'mean_mae')
        xs, ybs = extract_curve({k:v['seasonal'] for k,v in base.items()}, 'mean_mae')
        if xb: plt.plot(xb, ybp, marker='o', label='Persistence')
        if xs: plt.plot(xs, ybs, marker='o', label='Seasonal-naive')
    if arim:
        xa, ya = extract_curve(arim, 'mean_mae')
        if xa: plt.plot(xa, ya, marker='o', label='ARIMA')
    plt.xlabel('Horizon (days)'); plt.ylabel('MAE'); plt.title('MAE vs Horizon'); plt.grid(True); plt.legend()
    plt.savefig(outdir / 'mae_vs_horizon.png', bbox_inches='tight'); plt.close()
    print(f"Saved plots to {outdir}")
if __name__ == '__main__':
    main()
