# charts.py

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def make_insight(ticker: str,
                 price_df: pd.DataFrame,
                 forecast_df: pd.DataFrame,
                 sentiment_list) -> str:
    """
    Two-sentence insight:
      1) 10-day % change
      2) 90-day avg sentiment tone
    """
    if price_df.empty:
        return ""
    hist10 = price_df.tail(10)
    start = float(hist10["Close"].iloc[0])
    end   = float(hist10["Close"].iloc[-1])
    pct   = (end - start) / start
    trend = "up" if pct >= 0 else "down"

    # sentiment
    scores = []
    for v in sentiment_list:
        if isinstance(v, dict):
            scores.append(float(v.get("Score", v.get("score", 0.0))))
        elif isinstance(v, (int, float)):
            scores.append(float(v))
    avg_s = float(np.mean(scores)) if scores else 0.0
    tone  = "positive" if avg_s >= 0 else "negative"

    s1 = f"{ticker} has gone {trend} {abs(pct)*100:.1f}% over the past 10 days."
    s2 = f"Average news sentiment was {tone} ({avg_s:+.2f})."
    return f"{s1} {s2}"

def plot_stock(ax,
               ticker: str,
               price_df: pd.DataFrame,
               forecast_df: pd.DataFrame,
               sentiment_list):
    """
    Plot on ax:
      • next 3 days forecast (dashed orange)
      • extend x-axis to include forecast dates
      • add insight text
    """
    # Forecast only (Historical data not shown for clarity)
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        fc = forecast_df.copy()
        fc["Date"] = pd.to_datetime(fc["Date"])
        ax.plot(
            fc["Date"], fc["Forecast_Close"],
            "--o", color="orange", linewidth=2
        )
        all_dates = list(fc["Date"])
    else:
        all_dates = []

    if all_dates:
        ax.set_xlim(min(all_dates), max(all_dates) + pd.Timedelta(days=3))

    # format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax.set_title(ticker, fontsize=10)

    # insight
    text = make_insight(ticker, price_df, forecast_df, sentiment_list)
    if text:
        ax.text(
            0.5, -0.2, text,
            transform=ax.transAxes,
            ha="center", va="top",
            wrap=True, fontsize=7
        )

def create_collage(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    forecast_data: dict[str, pd.DataFrame],
    sentiment_map: dict[str, list],
    title: str,
    save_path: str
) -> str:
    """
    Build and save a 2-column collage of subplots for each ticker.
    Returns the file path (save_path).
    """
    rows = math.ceil(len(tickers) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 4), squeeze=False)
    axes_flat = axes.flatten()

    for ax, tic in zip(axes_flat, tickers):
        hist_df = price_data.get(tic, pd.DataFrame())
        f_df    = forecast_data.get(tic, pd.DataFrame())
        s_list  = sentiment_map.get(tic, [])
        plot_stock(ax, tic, hist_df, f_df, s_list)

    # remove any unused axes
    for ax in axes_flat[len(tickers):]:
        fig.delaxes(ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
