import pandas as pd
import numpy as np

def analyze_vrp():
    try:
        df_dvol = pd.read_csv("btc_dvol_history.csv")
        df_rv = pd.read_csv("btc_realized_vol_history.csv")
    except FileNotFoundError:
        print("Run `python scripts/fetch_deribit_history.py` first to generate CSVs.")
        return

    # Merge on timestamp
    df_dvol['datetime'] = pd.to_datetime(df_dvol['datetime'])
    df_rv['datetime'] = pd.to_datetime(df_rv['datetime'])

    merged = pd.merge_asof(
        df_dvol.sort_values('datetime'),
        df_rv.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )

    merged['vrp'] = merged['close'] - merged['realized_vol']

    print("=" * 60)
    print("=== MILESTONE M6: VARIANCE RISK PREMIUM (VRP) SUMMARY ===")
    print("=" * 60)
    print(f"Mean Implied Vol (DVOL):   {merged['close'].mean():.2f}%")
    print(f"Mean Realized Vol (RV):    {merged['realized_vol'].mean():.2f}%")
    print(f"Mean VRP (DVOL - RV):      {merged['vrp'].mean():+.2f}%")
    print(f"VRP Positive Fraction:     {(merged['vrp'] > 0).mean():.2%}")
    print(f"Max VRP Premium:           {merged['vrp'].max():+.2f}%")
    print(f"Max VRP Inversion:         {merged['vrp'].min():+.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    analyze_vrp()