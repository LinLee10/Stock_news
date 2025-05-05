# forecast_utils.py

import os
import logging
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────
LOG_DIR          = "logs"
PLOTS_DIR        = "plots"
FORECAST_HORIZON = 3  # forecast 3 days ahead

# ─── Logging setup ────────────────────────────────────────────────
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "forecast.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())

# ─── Forecast plotting ────────────────────────────────────────────
def plot_forecast(ticker, df_hist, preds, current_date, model=None):
    """
    Save a simple 3-day forecast line (orange dashed) for `ticker`.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))

    # Build forecast dates
    dates = [current_date + timedelta(days=i) for i in range(1, FORECAST_HORIZON+1)]
    # Plot forecast only
    ax.plot(dates, preds, '--o', color='orange', linewidth=2)

    # Optional confidence bands
    if model is not None and not df_hist.empty:
        feature_cols = [c for c in df_hist.columns if c not in ('Date','AdjClose','Return')]
        resids = model.predict(df_hist[feature_cols]) - df_hist['AdjClose']
        std = np.std(resids)
        lower = [p - 1.96*std for p in preds]
        upper = [p + 1.96*std for p in preds]
        ax.fill_between(dates, lower, upper, color='orange', alpha=0.3)

    ax.set_title(f"{ticker} {FORECAST_HORIZON}-Day Forecast", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=45)

    out_path = os.path.join(PLOTS_DIR, f"{ticker}_{current_date.strftime('%Y%m%d')}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

    logging.info(f"Saved forecast plot to {out_path}")
    return out_path
