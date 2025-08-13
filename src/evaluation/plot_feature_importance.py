import json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fi_path', default='reports/metrics/xgb_feature_importance.json')
    ap.add_argument('--outdir', default='reports/plots')
    ap.add_argument('--top_k', type=int, default=15)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fi = json.loads(Path(args.fi_path).read_text())
    horizons = sorted([int(h) for h in fi.keys()])
    all_feats = sorted({f for h in fi for f in fi[h]})
    avg_gain = {f: np.mean([fi[str(h)].get(f,0.0) for h in horizons]) for f in all_feats}
    top_feats = [f for f,_ in sorted(avg_gain.items(), key=lambda x: x[1], reverse=True)[:args.top_k]]
    M = np.array([[fi[str(h)].get(f, 0.0) for f in top_feats] for h in horizons])
    plt.figure(figsize=(max(8, len(top_feats)*0.5), 5))
    plt.imshow(M, aspect='auto', interpolation='nearest'); plt.colorbar(label='XGB gain')
    plt.yticks(range(len(horizons)), horizons); plt.xticks(range(len(top_feats)), top_feats, rotation=45, ha='right')
    plt.xlabel('Feature'); plt.ylabel('Horizon (days)'); plt.title('XGBoost Feature Importance (gain) across horizons'); plt.tight_layout()
    plt.savefig(outdir / 'feature_importance_heatmap.png', bbox_inches='tight'); plt.close()
    def is_lag(f): return f.startswith('lag_')
    def is_roll(f): return f.startswith('rollmean_')
    def is_seasonal(f): return f in ('sin_doy','cos_doy')
    grouped = {'lag': [], 'roll': [], 'seasonal': []}
    for h in horizons:
        d = fi[str(h)]
        grouped['lag'].append(sum(v for f,v in d.items() if is_lag(f)))
        grouped['roll'].append(sum(v for f,v in d.items() if is_roll(f)))
        grouped['seasonal'].append(sum(v for f,v in d.items() if is_seasonal(f)))
    plt.figure(figsize=(10,5))
    x = horizons
    plt.stackplot(x, grouped['lag'], grouped['roll'], grouped['seasonal'], labels=['lag','roll','seasonal'])
    plt.xlabel('Horizon (days)'); plt.ylabel('Total gain'); plt.title('Grouped feature importance vs horizon'); plt.legend(loc='upper right'); plt.tight_layout()
    plt.savefig(outdir / 'feature_importance_grouped.png', bbox_inches='tight'); plt.close()
    print(f"Saved feature-importance plots to {outdir}")
if __name__ == '__main__':
    main()
