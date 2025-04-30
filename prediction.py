import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
import yfinance as yf
import time
import pandas as pd
import logging
import requests
import io
import random


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


ALPHAVANTAGE_API_KEY = os.getenv("7CNI4DS6B0KEYSAM")

_bulk_price_cache: dict[str, pd.DataFrame] = {}

def fetch_price_history_bulk(tickers: list[str], period: str = "90d") -> dict[str, pd.DataFrame]:
    """
    Bulk download price history for a list of tickers via yfinance.download,
    with exponential‐backoff retries and local caching per ticker.
    """
    logger = logging.getLogger("prediction")
    cache_dir = "data/yf_bulk_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # determine which tickers need fresh download (cache < 24h)
    to_download = []
    now = time.time()
    for tic in tickers:
        path = f"{cache_dir}/{tic}_{period}.csv"
        if os.path.exists(path) and now - os.path.getmtime(path) < 86400:
            _bulk_price_cache[tic] = pd.read_csv(path, parse_dates=["Date"])
        else:
            to_download.append(tic)

    # bulk download in one shot
    if to_download:
        wait = 1
        for attempt in range(5):
            try:
                df_all = yf.download(
                    to_download,
                    period=period,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False
                )
                # split out per‐ticker frames
                for tic in to_download:
                    if tic in df_all and not df_all[tic].empty:
                        df = df_all[tic].reset_index()[["Date", "Close"]].rename(columns={"Close": "Stock_Close"})
                    else:
                        logger.warning(f"[bulk] no data for {tic}")
                        df = pd.DataFrame(columns=["Date", "Stock_Close"])
                    # cache to disk & memory
                    path = f"{cache_dir}/{tic}_{period}.csv"
                    df.to_csv(path, index=False)
                    _bulk_price_cache[tic] = df
                break
            except Exception as e:
                logger.warning(f"[bulk] download failed: {e} → retry in {wait}s")
                time.sleep(wait + random.uniform(0, 0.3 * wait))
                wait = min(wait * 2, 16)

    return _bulk_price_cache

def load_bulk_price_data(tickers: list[str], period: str = "90d"):
    """Populate the global cache once, at startup."""
    fetch_price_history_bulk(tickers, period)

def fetch_price_history(ticker: str, period: str = "90d") -> pd.DataFrame:
    """
    Wrapper that first looks in the bulk cache, then falls back to a quick yf.Ticker call.
    """
    # 1) In‐memory bulk cache?
    if ticker in _bulk_price_cache:
        return _bulk_price_cache[ticker]

    # 2) Fallback one‐off fetch (rarely used)
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Stock_Close"])
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Close": "Stock_Close"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df




def train_and_forecast(
    ticker: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    peer_df: pd.DataFrame,
    sentiment_series: pd.Series | None = None,
    half_life: float = 7.0
) -> tuple[pd.DataFrame, float, bool]:
    """
    Train RF & XGB on 90d history, forecast 3d ahead,
    compute confidence & red-flag. Returns:
      - forecast_df: DataFrame with ['Date','Forecast_Close']
      - confidence: float between 0 and 1
      - red_flag: bool alert if strong negative signal
    """
    try:
        # 1) Merge price history & compute returns
        df = price_df.rename(columns={"Close": "Stock_Close"}).copy()
        df = df.merge(
            market_df.rename(columns={"Close": "Market_Close"})[["Date", "Market_Close"]],
            on="Date", how="inner"
        )
        df = df.merge(
            peer_df.rename(columns={"Close": "Peer_Close"})[["Date", "Peer_Close"]],
            on="Date", how="inner"
        )
        df["Return"]        = df["Stock_Close"].pct_change()
        df["Market_Return"] = df["Market_Close"].pct_change()
        df["Peer_Return"]   = df["Peer_Close"].pct_change()
        df.dropna(inplace=True)

        # 2) Volatility feature: 5-day rolling std of returns
        df["Volatility"] = df["Return"].rolling(window=5, min_periods=1).std().fillna(0.0)

        # 3) Sentiment EWMA
        if sentiment_series is not None and hasattr(sentiment_series, "index") and not sentiment_series.empty:
            sent = sentiment_series.copy()
            sent.index = pd.to_datetime(sent.index)
            sent_daily = sent.groupby(sent.index).mean()
            dates = pd.to_datetime(df["Date"])
            sent_daily = sent_daily.reindex(dates, method="ffill").fillna(0.0)
            df["Sentiment"] = sent_daily.ewm(halflife=half_life).mean().values
        else:
            df["Sentiment"] = 0.0

        # 4) Prepare features & target
        features = ["Market_Return", "Peer_Return", "Sentiment", "Volatility"]
        X = df[features].values
        y = df["Return"].values
        if len(y) < 2:
            return pd.DataFrame(columns=["Date", "Forecast_Close"]), 0.0, False

        # 5) Train ensemble models
        rf  = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        rf.fit(X, y)
        xgb.fit(X, y)

        # 6) In-sample predictions & rebuild price path
        preds_rf  = rf.predict(X)
        preds_xgb = xgb.predict(X)
        df["Predicted_Return"] = (preds_rf + preds_xgb) / 2
        df["Predicted_Close"]  = df["Stock_Close"].iloc[0]
        for i in range(1, len(df)):
            df.at[i, "Predicted_Close"] = df.at[i-1, "Predicted_Close"] * (1 + df.at[i, "Predicted_Return"])

        # 7) 3-day recursive forecast
        last = df.iloc[-1]
        base_price = last["Predicted_Close"]
        date0 = pd.to_datetime(last["Date"])
        forecasts = []
        ensemble_preds = []
        for i in range(1, 4):
            feat = np.array([[ 
                last["Market_Return"],
                last["Peer_Return"],
                last["Sentiment"],
                last["Volatility"]
            ]])
            prf = rf.predict(feat)[0]
            pxg = xgb.predict(feat)[0]
            p   = (prf + pxg) / 2
            ensemble_preds.append(p)
            base_price *= (1 + p)
            forecasts.append({
                "Date": (date0 + timedelta(days=i)).strftime("%Y-%m-%d"),
                "Forecast_Close": round(base_price, 2)
            })
        forecast_df = pd.DataFrame(forecasts)

        # 8) Confidence meter
        eps = 1e-6
        ens_std  = float(np.std(ensemble_preds))
        vol_last = float(last["Volatility"])
        confidence = 1.0 / (1.0 + ens_std * vol_last + eps)

        # 9) Red-flag alert
        red_flag = (ensemble_preds[0] <= -0.05) and (last["Sentiment"] <= -0.3)

        # 10) Log big misses >5%
        miss_idx = np.where(np.abs(df["Predicted_Return"] - df["Return"]) > 0.05)[0]
        if len(miss_idx):
            miss_df = df.iloc[miss_idx][["Date", "Return", "Predicted_Return"]].copy()
            miss_df.insert(0, "Ticker", ticker)
            try:
                old = pd.read_csv("forecast_misses.csv")
                pd.concat([old, miss_df]).to_csv("forecast_misses.csv", index=False)
            except FileNotFoundError:
                miss_df.to_csv("forecast_misses.csv", index=False)

        return forecast_df, float(confidence), bool(red_flag)

    except Exception as ex:
        print(f"[prediction] train_and_forecast error for {ticker}: {ex}")
        return pd.DataFrame(columns=["Date", "Forecast_Close"]), 0.0, False

# ─── Forecast Logging Utilities ──────────────────────────────────────────────

def log_forecast_entry(pred_date, target_date, ticker, predicted_price,
                       actual_price=None, file='forecasts.csv'):
    """
    Append one 3-day forecast to forecasts.csv.
    Columns: pred_date, target_date, ticker, predicted_price, actual_price
    """
    entry = {
        'pred_date': pd.to_datetime(pred_date).date(),
        'target_date': pd.to_datetime(target_date).date(),
        'ticker': ticker,
        'predicted_price': predicted_price,
        'actual_price': actual_price if actual_price is not None else np.nan
    }
    df = pd.DataFrame([entry])
    header = not os.path.isfile(file)
    try:
        df.to_csv(file, mode='a', header=header, index=False)
    except Exception:
        logging.exception(f"Failed to log forecast for {ticker} on {target_date}")

def update_actual_prices(file='forecasts.csv'):
    """
    Fill in missing actual_price for any completed forecasts,
    pulling true closes from data/{ticker}.csv.
    """
    try:
        forecasts = pd.read_csv(file, parse_dates=['pred_date','target_date'])
    except FileNotFoundError:
        logging.error(f"{file} not found.")
        return

    updated = False
    today = pd.Timestamp.now().normalize()

    for idx, row in forecasts.iterrows():
        if pd.isna(row['actual_price']):
            tgt = row['target_date'].date()
            if pd.Timestamp(tgt) < today:
                ticker = row['ticker']
                try:
                    stock_df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'])
                    stock_df.set_index('Date', inplace=True)
                    actual = stock_df.loc[pd.Timestamp(tgt), 'Close']
                    forecasts.at[idx, 'actual_price'] = actual
                    updated = True
                except Exception:
                    logging.exception(f"Could not fetch actual for {ticker} on {tgt}")

    if updated:
        try:
            forecasts.to_csv(file, index=False)
        except Exception:
            logging.exception(f"Failed to write back updated {file}")

def train_predict_stock(ticker: str, sentiment_series: pd.Series | None = None):
    """
    Wrapper for main.py:
      1) Fetch price, market, and peer data (always providing the expected column names)
      2) Call train_and_forecast(...)
      3) Return the dict main.py expects
    """
    # 1) Fetch 90-day histories
    price_df  = fetch_price_history(ticker,   period="90d")
    market_df = fetch_price_history("^GSPC",   period="90d")
    peer_df   = fetch_price_history(ticker,   period="90d")  # using same ticker as fallback

    # 1a) Ensure correct column names for merging inside train_and_forecast
    # price_df -> Stock_Close
    if "Close" in price_df.columns:
        price_df = price_df.rename(columns={"Close": "Stock_Close"})
    elif "Stock_Close" not in price_df.columns:
        raise KeyError(f"price_df missing Close/Stock_Close for {ticker}")

    # market_df -> Market_Close
    if "Close" in market_df.columns:
        market_df = market_df.rename(columns={"Close": "Market_Close"})
    elif "Stock_Close" in market_df.columns:
        market_df = market_df.rename(columns={"Stock_Close": "Market_Close"})
    else:
        raise KeyError("market_df missing Close/Stock_Close for ^GSPC")

    # peer_df -> Peer_Close
    if "Close" in peer_df.columns:
        peer_df = peer_df.rename(columns={"Close": "Peer_Close"})
    elif "Stock_Close" in peer_df.columns:
        peer_df = peer_df.rename(columns={"Stock_Close": "Peer_Close"})
    else:
        raise KeyError(f"peer_df missing Close/Stock_Close for {ticker}")

    # 2) Run your core forecasting routine
    forecast_df, confidence, red_flag = train_and_forecast(
        ticker,
        price_df,
        market_df,
        peer_df,
        sentiment_series
    )

    # 3) Prepare history DataFrame for plotting (rename to AdjClose)
    if "Stock_Close" in price_df.columns:
        history = price_df.rename(columns={"Stock_Close": "AdjClose"})[["Date", "AdjClose"]]
    else:
        history = price_df.rename(columns={"Close": "AdjClose"})[["Date", "AdjClose"]]

    # 4) Package output
    return {
        "dates":       [pd.to_datetime(d) for d in forecast_df["Date"]],
        "history":     history,
        "predictions": [float(x) for x in forecast_df["Forecast_Close"]],
        "confidence":  confidence,
        "red_flag":    red_flag
    }






# ─── Accuracy Metrics Utilities ──────────────────────────────────────────────

def calculate_accuracy(file='forecasts.csv'):
    """
    Compute and print:
      • Overall MAPE & Directional Accuracy (MDA)
      • MAPE & MDA for the last 7 days
      • MAPE & MDA for the last 30 days
    """
    try:
        df = pd.read_csv(file, parse_dates=['pred_date','target_date'])
    except FileNotFoundError:
        print(f"{file} not found.")
        return

    df = df.dropna(subset=['actual_price'])
    if df.empty:
        print("No completed forecasts to evaluate.")
        return

    # Overall MAPE
    df['error_pct'] = (df['actual_price'] - df['predicted_price']).abs() / df['actual_price'] * 100
    mape_all = df['error_pct'].mean()

    # Overall MDA
    df = df.sort_values(['ticker','target_date'])
    df['prev_actual'] = df.groupby('ticker')['actual_price'].shift(1)
    df['actual_dir'] = np.sign(df['actual_price'] - df['prev_actual'])
    df['pred_dir']   = np.sign(df['predicted_price'] - df['prev_actual'])
    valid = df['prev_actual'].notna()
    mda_all = (df.loc[valid,'actual_dir'] == df.loc[valid,'pred_dir']).mean() * 100 if valid.any() else float('nan')

    print(f"Overall ➞ MAPE: {mape_all:.2f}%, MDA: {mda_all:.2f}%")

    # Helper for periods
    def period_stats(df_sub, days):
        if df_sub.empty:
            print(f"No forecasts in last {days} days.")
            return
        df_sub['error_pct'] = (df_sub['actual_price'] - df_sub['predicted_price']).abs() / df_sub['actual_price'] * 100
        mape = df_sub['error_pct'].mean()
        df_sub['prev_actual'] = df_sub.groupby('ticker')['actual_price'].shift(1)
        df_sub['actual_dir'] = np.sign(df_sub['actual_price'] - df_sub['prev_actual'])
        df_sub['pred_dir']   = np.sign(df_sub['predicted_price'] - df_sub['prev_actual'])
        valid = df_sub['prev_actual'].notna()
        mda = (df_sub.loc[valid,'actual_dir'] == df_sub.loc[valid,'pred_dir']).mean() * 100 if valid.any() else float('nan')
        print(f"Last {days}d ➞ MAPE: {mape:.2f}%, MDA: {mda:.2f}%")

    today = pd.Timestamp.now().normalize()
    for days in (7, 30):
        cutoff = today - pd.Timedelta(days=days)
        period_stats(df[df['target_date'] >= cutoff], days)
