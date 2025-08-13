import numpy as np
import pandas as pd


def add_time_features(df, date_col, target_col, max_lag=14, roll_windows=(7,14)):
    df = df.copy()
    
    #Ensure date_col is index and datetime format
    if date_col == df.index.name:
        df.index = pd.to_datetime(df.index)
    else:
        df[date_col] = pd.to_datetime(df[date_col])

    #Ensure chronological order
    df.sort_values(date_col, inplace=True)

    #Sin/cos day-of-year feature
    doy = df[date_col].dt.dayofyear
    df['sin_doy'] = np.sin(2*np.pi*doy/365.25)
    df['cos_doy'] = np.cos(2*np.pi*doy/365.25)
    
    #Lags features
    for l in range(1, max_lag+1):
        df[f'lag_{l}'] = df[target_col].shift(l)
    
    #Rolling means features
    for w in roll_windows:
        df[f'rollmean_{w}'] = df[target_col].shift(1).rolling(w).mean()
    
    return df