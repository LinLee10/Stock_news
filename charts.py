import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def make_insight(ticker: str,
                 price_df: pd.DataFrame,
                 forecast_df: pd.DataFrame,
                 sentiment_list) -> str:
    if price_df.empty:
        return ""
    hist10 = price_df.tail(10)
    start = float(hist10["Close"].iloc[0])
    end   = float(hist10["Close"].iloc[-1])
    pct   = (end - start) / start
    trend = "up" if pct >= 0 else "down"

    scores = []
    for v in sentiment_list:
        try:
            scores.append(float(v))
        except:
            pass
    avg_s = float(np.mean(scores)) if scores else 0.0
    tone  = "positive" if avg_s >= 0 else "negative"

    s1 = f"{ticker} has gone {trend} {abs(pct)*100:.1f}% over the past 10 days."
    s2 = f"Average news sentiment was {tone} ({avg_s:+.2f})."
    return f"{s1}\n{s2}"

def plot_stock(ax,
               ticker: str,
               price_df: pd.DataFrame,
               forecast_df: pd.DataFrame,
               sentiment_list):
    # historical (last 10 days)
    hist = price_df.tail(10).copy()
    if not hist.empty:
        hist["Date"] = pd.to_datetime(hist["Date"])
        ax.plot(
            hist["Date"], hist["Close"],
            label="Historical (10d)", linewidth=1.5, color="blue"
        )

    # forecast (next 3 days)
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        fc = forecast_df.copy()
        fc["Date"] = pd.to_datetime(fc["Date"])
        ax.plot(
            fc["Date"], fc["Forecast_Close"],
            "--o", label="Forecast (3d)", color="orange", linewidth=1.2
        )
        all_dates = list(hist["Date"]) + list(fc["Date"])
    else:
        all_dates = list(hist["Date"])

    # extend x-axis a bit beyond data
    if all_dates:
        ax.set_xlim(min(all_dates), max(all_dates) + pd.Timedelta(days=3))

    # sentiment overlay on secondary axis
    if isinstance(sentiment_list, pd.Series):
        s = sentiment_list.dropna()
    else:
        try:
            s = pd.Series(sentiment_list, index=hist["Date"])
        except:
            s = pd.Series(sentiment_list)
    if not s.empty and hasattr(s.index, 'dtype') and np.issubdtype(s.index.dtype, np.datetime64):
        ax2 = ax.twinx()
        ax2.plot(
            s.index, s.values,
            "-", label="Sentiment", color="green", linewidth=1.0, alpha=0.7
        )
        ax2.set_ylabel("Sentiment", fontsize=8)
    else:
        ax2 = None

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title(ticker, fontsize=10)
    ax.set_ylabel("Price", fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2; labels += l2
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper left")

def create_collage(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    forecast_data: dict[str, pd.DataFrame],
    sentiment_map: dict[str, list],
    title: str,
    save_path: str
) -> str:
    rows = math.ceil(len(tickers) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 4), squeeze=False)
    axes_flat = axes.flatten()

    for ax, tic in zip(axes_flat, tickers):
        hist_df = price_data.get(tic, pd.DataFrame(columns=["Date","Close"]))
        f_df    = forecast_data.get(tic, pd.DataFrame())
        s_list  = sentiment_map.get(tic, [])
        plot_stock(ax, tic, hist_df, f_df, s_list)

    # remove unused subplots
    for ax in axes_flat[len(tickers):]:
        fig.delaxes(ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
