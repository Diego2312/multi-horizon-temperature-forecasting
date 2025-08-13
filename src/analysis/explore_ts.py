import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf

def preprocess_data(df, start_year=1990, end_year=2020):
    df = df[['DATE','TAVG']].copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[df['DATE'].dt.year.between(start_year, end_year)].copy()
    if df['TAVG'].abs().median() > 80:
        df['TAVG'] = df['TAVG'] / 10.0
    df = df.sort_values('DATE').reset_index(drop=True)
    df['TAVG'] = df['TAVG'].interpolate('linear')
    return df

def plot_rolling_stats(df, outdir):
    s = df['TAVG']
    roll_mean = s.rolling(365, min_periods=1).mean()
    roll_std = s.rolling(365, min_periods=1).std()
    plt.figure(figsize=(12,4)); plt.plot(df['DATE'], s, label='TAVG'); plt.plot(df['DATE'], roll_mean, label='Rolling mean (365d)'); plt.title('Rolling Mean (365d)'); plt.tight_layout(); plt.legend(); plt.savefig(outdir / "rolling_mean_365d.png", bbox_inches='tight'); plt.close()
    plt.figure(figsize=(12,4)); plt.plot(df['DATE'], roll_std, label='Rolling std (365d)'); plt.title('Rolling Standard Deviation (365d)'); plt.tight_layout(); plt.legend(); plt.savefig(outdir / "rolling_std_365d.png", bbox_inches='tight'); plt.close()

def plot_stl(df, outdir):
    s = df.set_index('DATE')['TAVG']
    stl = STL(s, period=365, robust=True)
    res = stl.fit()
    fig = res.plot(); fig.set_size_inches(12,8); fig.suptitle('STL decomposition (period=365)', y=0.98)
    fig.savefig(outdir / "stl_decomposition.png", bbox_inches='tight'); plt.close(fig)

def run_adf(df, outdir):
    s = df['TAVG']
    stat, pval, *_ = adfuller(s.dropna())
    Path(outdir / "adf_result.txt").write_text(f"ADF statistic: {stat:.4f}\np-value: {pval:.4g}\n", encoding='utf-8')

def plot_acf_pacf(df, outdir, nlags=365):

    s = df['TAVG'].dropna()

    # keep nlags valid for the available series length
    nlags = int(min(nlags, max(1, len(s) - 1)))

    nlags_pacf = 30

    ac = acf(s, nlags=nlags, fft=True)
    pc = pacf(s, nlags=nlags_pacf, method='yw')

    # ACF
    x = np.arange(len(ac))
    conf_int_acf = 1.96 / np.sqrt(len(s))
    plt.figure(figsize=(12, 4))
    plt.vlines(x, 0, ac, linewidth=1)
    plt.plot(x, ac, 'o', ms=3)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(conf_int_acf, color='red', linestyle='--', linewidth=1, label='95% CI')
    plt.axhline(-conf_int_acf, color='red', linestyle='--', linewidth=1)
    plt.title(f'ACF (nlags={nlags})')
    plt.tight_layout()
    plt.legend()
    plt.savefig(outdir / f"acf_{nlags}.png", bbox_inches='tight')
    plt.close()

    # PACF
    x = np.arange(len(pc))
    conf_int_pacf = 1.96 / np.sqrt(len(s))
    plt.figure(figsize=(12, 4))
    plt.vlines(x, 0, pc, linewidth=1)
    plt.plot(x, pc, 'o', ms=3)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(conf_int_pacf, color='red', linestyle='--', linewidth=1, label='95% CI')
    plt.axhline(-conf_int_pacf, color='red', linestyle='--', linewidth=1)
    plt.title(f'PACF (nlags={nlags_pacf})')
    plt.tight_layout()
    plt.legend()
    plt.savefig(outdir / f"pacf_{nlags_pacf}.png", bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--start_year', type=int, default=1990)
    ap.add_argument('--end_year', type=int, default=2020)
    ap.add_argument('--outdir', default='reports/analysis')
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data_path)
    assert 'DATE' in df.columns and 'TAVG' in df.columns
    df = preprocess_data(df, args.start_year, args.end_year)
    df.head(10).to_csv(outdir / "data_preview.csv", index=False)
    plot_rolling_stats(df, outdir); plot_stl(df, outdir); run_adf(df, outdir); plot_acf_pacf(df, outdir)
    print(f"Saved time series analysis to {outdir}")
if __name__ == '__main__':
    main()
