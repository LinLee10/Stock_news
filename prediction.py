import os
import time
import random
import logging
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Alpha Vantage key
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# In-memory bulk cache
_bulk_price_cache: dict[str, pd.DataFrame] = {}

def fetch_price_history_bulk(tickers: list[str], period: str = "90d") -> dict[str, pd.DataFrame]:
    logger = logging.getLogger("prediction")
    cache_dir = "data/yf_bulk_cache"
    os.makedirs(cache_dir, exist_ok=True)

    to_download = []
    now = time.time()
    for tic in tickers:
        path = f"{cache_dir}/{tic}_{period}.csv"
        if os.path.exists(path) and now - os.path.getmtime(path) < 86400:
            _bulk_price_cache[tic] = pd.read_csv(path, parse_dates=["Date"])
        else:
            to_download.append(tic)

    if to_download:
        wait = 1
        for _ in range(5):
            try:
                df_all = yf.download(
                    to_download,
                    period=period,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False
                )
                for tic in to_download:
                    frame = df_all.get(tic, pd.DataFrame())
                    if not frame.empty:
                        df = (
                            frame
                            .reset_index()[["Date", "Close"]]
                            .rename(columns={"Close": "Stock_Close"})
                        )
                    else:
                        logger.warning(f"[bulk] no data for {tic}")
                        df = pd.DataFrame(columns=["Date", "Stock_Close"])

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
    fetch_price_history_bulk(tickers, period)

def fetch_price_history(ticker: str, period: str = "90d") -> pd.DataFrame:
    if ticker in _bulk_price_cache:
        return _bulk_price_cache[ticker]

    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Stock_Close"])

    df = (
        df
        .reset_index()[["Date", "Close"]]
        .rename(columns={"Close": "Stock_Close"})
    )
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
    try:
        # Merge and compute returns
        df = price_df.rename(columns={"Stock_Close": "Stock_Close"}).copy()
        df = df.merge(
            market_df.rename(columns={"Stock_Close": "Market_Close"})[["Date", "Market_Close"]],
            on="Date"
        ).merge(
            peer_df.rename(columns={"Stock_Close": "Peer_Close"})[["Date", "Peer_Close"]],
            on="Date"
        )
        df["Return"] = df["Stock_Close"].pct_change()
        df["Market_Return"] = df["Market_Close"].pct_change()
        df["Peer_Return"] = df["Peer_Close"].pct_change()
        df.dropna(inplace=True)

        # Features
        df["Volatility"] = df["Return"].rolling(5, min_periods=1).std().fillna(0)
        if sentiment_series is not None and not sentiment_series.empty:
            sent = pd.to_datetime(sentiment_series.index.to_series())
            sent = sentiment_series.groupby(sent).mean()
            sent = sent.reindex(pd.to_datetime(df["Date"]), method="ffill").fillna(0)
            df["Sentiment"] = sent.ewm(halflife=half_life).mean().values
        else:
            df["Sentiment"] = 0.0

        features = ["Market_Return", "Peer_Return", "Sentiment", "Volatility"]
        X, y = df[features].values, df["Return"].values
        if len(y) < 2:
            return pd.DataFrame(columns=["Date", "Forecast_Close"]), 0.0, False

        # Train models
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        rf.fit(X, y)
        xgb.fit(X, y)

        # In-sample predictions
        preds = (rf.predict(X) + xgb.predict(X)) / 2
        df["Predicted_Return"] = preds
        df["Predicted_Close"] = df["Stock_Close"].iloc[0]
        for i in range(1, len(df)):
            df.at[i, "Predicted_Close"] = df.at[i-1, "Predicted_Close"] * (1 + df.at[i, "Predicted_Return"])

        # 3-day forecast
        last = df.iloc[-1]
        base = last["Predicted_Close"]
        today = pd.to_datetime(last["Date"])
        forecasts, ensemble = [], []
        for i in range(1, 4):
            feat = [[last["Market_Return"], last["Peer_Return"], last["Sentiment"], last["Volatility"]]]
            p = np.mean([rf.predict(feat)[0], xgb.predict(feat)[0]])
            ensemble.append(p)
            base *= (1 + p)
            forecasts.append({
                "Date": (today + timedelta(days=i)).strftime("%Y-%m-%d"),
                "Forecast_Close": round(base, 2)
            })
        forecast_df = pd.DataFrame(forecasts)

        # Confidence & red flag
        eps = 1e-6
        confidence = 1 / (1 + np.std(ensemble) * last["Volatility"] + eps)
        red_flag = ensemble[0] <= -0.05 and last["Sentiment"] <= -0.3

        return forecast_df, float(confidence), bool(red_flag)

    except Exception as e:
        logging.error(f"train_and_forecast error for {ticker}: {e}")
        return pd.DataFrame(columns=["Date", "Forecast_Close"]), 0.0, False

def log_forecast_entry(pred_date, target_date, ticker, predicted_price, actual_price=None, file="forecasts.csv"):
    entry = {
        "pred_date": pd.to_datetime(pred_date).date(),
        "target_date": pd.to_datetime(target_date).date(),
        "ticker": ticker,
        "predicted_price": predicted_price,
        "actual_price": actual_price or np.nan
    }
    df = pd.DataFrame([entry])
    header = not os.path.isfile(file)
    df.to_csv(file, mode="a", header=header, index=False)

def train_predict_stock(ticker: str, sentiment_series: pd.Series | None = None):
    price_df  = fetch_price_history(ticker)
    market_df = fetch_price_history("^GSPC")
    peer_df   = fetch_price_history(ticker)

    # normalize price_df → Stock_Close
    if "Close" in price_df.columns:
        price_df = price_df.rename(columns={"Close": "Stock_Close"})
    elif "Stock_Close" not in price_df.columns:
        raise KeyError(f"Stock_Close missing for {ticker}")

    # normalize market_df → Market_Close
    if "Close" in market_df.columns:
        market_df = market_df.rename(columns={"Close": "Market_Close"})
    elif "Stock_Close" in market_df.columns:
        market_df = market_df.rename(columns={"Stock_Close": "Market_Close"})
    else:
        raise KeyError("Market_Close missing for ^GSPC")

    # normalize peer_df → Peer_Close
    if "Close" in peer_df.columns:
        peer_df = peer_df.rename(columns={"Close": "Peer_Close"})
    elif "Stock_Close" in peer_df.columns:
        peer_df = peer_df.rename(columns={"Stock_Close": "Peer_Close"})
    else:
        raise KeyError(f"Peer_Close missing for {ticker}")

    forecast_df, confidence, red_flag = train_and_forecast(
        ticker, price_df, market_df, peer_df, sentiment_series
    )

    # prepare history for plotting
    history = price_df.rename(columns={"Stock_Close": "AdjClose"})[["Date", "AdjClose"]]

    return {
        "dates":       [pd.to_datetime(d) for d in forecast_df["Date"]],
        "history":     history,
        "predictions": [float(x) for x in forecast_df["Forecast_Close"]],
        "confidence":  confidence,
        "red_flag":    red_flag
    }
