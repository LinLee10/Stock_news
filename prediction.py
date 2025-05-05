# prediction.py

import os
import time
import random
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ─── New imports for LSTM ─────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional

# ─── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger("prediction")

# ─── Alpha Vantage key (required) ─────────────────────────────────────────
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHA_VANTAGE_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    logger.error("Alpha Vantage API key missing! Set ALPHA_VANTAGE_KEY or ALPHAVANTAGE_API_KEY in config/secrets.env and retry.")
    raise RuntimeError("ALPHA_VANTAGE_KEY is required for price fetches")

# ─── In-memory bulk cache ───────────────────────────────────────────────────
_bulk_price_cache: dict[str, pd.DataFrame] = {}

def fetch_price_history_bulk(tickers: List[str], period: str = "90d") -> dict[str, pd.DataFrame]:
    """
    Bulk-fetch via Alpha Vantage TIME_SERIES_DAILY_ADJUSTED (compact).
    Caches per-ticker CSV under data/av_bulk_cache, with 1-day freshness.
    """
    cache_dir = "data/av_bulk_cache"
    os.makedirs(cache_dir, exist_ok=True)
    now = time.time()
    to_download = []

    # check cache freshness
    for tic in tickers:
        path = f"{cache_dir}/{tic}_{period}.csv"
        if os.path.exists(path) and now - os.path.getmtime(path) < 86400:
            logger.debug(f"Loading {tic} from AV bulk cache")
            _bulk_price_cache[tic] = pd.read_csv(path, parse_dates=["Date"])
        else:
            to_download.append(tic)

    # fetch missing
    for tic in to_download:
        try:
            logger.info(f"AV downloading: {tic}")
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": tic,
                "apikey": API_KEY,
                "outputsize": "compact"
            }
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=(3, 5)  # 3s connect, 5s read
            )
            resp.raise_for_status()
            raw = resp.json().get("Time Series (Daily)", {})
            if not raw:
                logger.warning(f"[AV bulk] no data for {tic}")
                df = pd.DataFrame(columns=["Date", "Stock_Close"])
            else:
                df = (
                    pd.DataFrame.from_dict(raw, orient="index")
                      .rename(columns={"5. adjusted close": "Stock_Close"})
                      .loc[:, ["Stock_Close"]]
                      .reset_index()
                      .rename(columns={"index": "Date"})
                )
                df["Date"] = pd.to_datetime(df["Date"])
                df["Stock_Close"] = df["Stock_Close"].astype(float)
                df = df.sort_values("Date").reset_index(drop=True)

            # persist
            path = f"{cache_dir}/{tic}_{period}.csv"
            df.to_csv(path, index=False)
            _bulk_price_cache[tic] = df
            logger.debug(f"Cached AV bulk {tic} → {path}")

            # abide free-tier rate limit (5 calls/min → ~12s per)
            time.sleep(12.0 + random.uniform(0, 1.0))

        except Exception as e:
            logger.warning(f"[AV bulk] download failed for {tic}: {e}")

    return _bulk_price_cache

def load_bulk_price_data(tickers: List[str], period: str = "90d"):
    fetch_price_history_bulk(tickers, period)

def fetch_price_history(ticker: str, period: str = "90d") -> pd.DataFrame:
    """
    First try in-memory bulk cache, else fallback to single AV fetch.
    Always returns a DataFrame with columns ["Date","Stock_Close"], filtered to `period`.
    """
    if ticker in _bulk_price_cache:
        logger.debug(f"Using in-memory cache for {ticker}")
        df = _bulk_price_cache[ticker]
    else:
        logger.debug(f"Fetching single history (AV) for {ticker}")
        fetch_price_history_bulk([ticker], period)
        df = _bulk_price_cache.get(ticker, pd.DataFrame(columns=["Date","Stock_Close"]))

    # filter to last N days if period ends with 'd'
    try:
        days = int(period.rstrip("d"))
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
        df = df[df["Date"] >= cutoff].reset_index(drop=True)
    except Exception:
        pass

    return df

def train_and_forecast(
    ticker: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    peer_df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    half_life: float = 7.0
) -> Tuple[pd.DataFrame, float, bool]:
    """
    1) Merge histories & compute returns
    2) Build features (market, peer, volatility, sentiment)
    3) Train RF & XGB
    4) In-sample predict & roll forward for 3-day forecast
    5) Compute confidence & red_flag
    """
    try:
        logger.debug(f"Preparing data for {ticker} (tree models)")
        df = price_df.rename(columns={"Stock_Close":"Stock_Close"}).copy()
        df = df.merge(
            market_df.rename(columns={"Stock_Close":"Market_Close"})[["Date","Market_Close"]],
            on="Date", how="inner"
        ).merge(
            peer_df.rename(columns={"Stock_Close":"Peer_Close"})[["Date","Peer_Close"]],
            on="Date", how="inner"
        )
        df["Return"]        = df["Stock_Close"].pct_change()
        df["Market_Return"] = df["Market_Close"].pct_change()
        df["Peer_Return"]   = df["Peer_Close"].pct_change()
        df.dropna(inplace=True)

        df["Volatility"] = df["Return"].rolling(window=5, min_periods=1).std().fillna(0.0)

        if sentiment_series is not None and not sentiment_series.empty:
            logger.debug("Applying sentiment EWMA")
            sent = sentiment_series.copy(); sent.index = pd.to_datetime(sent.index)
            sent_daily = sent.groupby(sent.index).mean()
            dates = pd.to_datetime(df["Date"])
            sent_daily = sent_daily.reindex(dates, method="ffill").fillna(0.0)
            df["Sentiment"] = sent_daily.ewm(halflife=half_life).mean().values
        else:
            df["Sentiment"] = 0.0

        features = ["Market_Return","Peer_Return","Sentiment","Volatility"]
        X = df[features].values
        y = df["Return"].values
        if len(y) < 2:
            logger.warning(f"Not enough data to train for {ticker}")
            return pd.DataFrame(columns=["Date","Forecast_Close"]), 0.0, False

        logger.info(f"Training RF+XGB for {ticker}")
        rf  = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        rf.fit(X, y)
        xgb.fit(X, y)

        preds_rf  = rf.predict(X)
        preds_xgb = xgb.predict(X)
        df["Predicted_Return"] = (preds_rf + preds_xgb) / 2
        df["Predicted_Close"] = df["Stock_Close"].iloc[0]
        for i in range(1, len(df)):
            df.at[i, "Predicted_Close"] = df.at[i-1, "Predicted_Close"] * (1 + df.at[i, "Predicted_Return"])

        last = df.iloc[-1]
        base_price = last["Predicted_Close"]
        last_date  = pd.to_datetime(last["Date"])
        forecasts, ensemble_preds = [], []
        for d in range(1, 4):
            feat = [[ last["Market_Return"], last["Peer_Return"], last["Sentiment"], last["Volatility"] ]]
            prf = rf.predict(feat)[0]
            pxg = xgb.predict(feat)[0]
            p   = (prf + pxg) / 2
            ensemble_preds.append(p)
            base_price *= (1 + p)
            forecasts.append({
                "Date": (last_date + timedelta(days=d)).strftime("%Y-%m-%d"),
                "Forecast_Close": round(base_price, 2)
            })

        forecast_df = pd.DataFrame(forecasts)

        eps = 1e-6
        confidence = 1.0 / (1.0 + np.std(ensemble_preds) * last["Volatility"] + eps)
        red_flag   = (ensemble_preds[0] <= -0.05) and (last["Sentiment"] <= -0.3)

        logger.info(f"{ticker} (tree) → confidence={confidence:.3f}, red_flag={red_flag}")
        return forecast_df, float(confidence), bool(red_flag)

    except Exception as e:
        logger.exception(f"train_and_forecast error for {ticker}")
        return pd.DataFrame(columns=["Date","Forecast_Close"]), 0.0, False

def log_forecast_entry(
    pred_date, target_date, ticker, predicted_price, actual_price=None,
    file: str="forecasts.csv"
):
    entry = {
        "pred_date":       pd.to_datetime(pred_date).date(),
        "target_date":     pd.to_datetime(target_date).date(),
        "ticker":          ticker,
        "predicted_price": predicted_price,
        "actual_price":    actual_price if actual_price is not None else np.nan
    }
    df = pd.DataFrame([entry])
    header = not os.path.isfile(file)
    df.to_csv(file, mode="a", header=header, index=False)
    logger.debug(f"Logged forecast for {ticker} on {target_date}")

def prepare_lstm_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 20
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Turn features+returns DataFrame into (samples, window_size, n_features) 
    and 1-step returns target. Scales features via StandardScaler.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").asfreq("B").ffill().reset_index()
    if "Return" not in df.columns:
        df["Return"] = df["Stock_Close"].pct_change().fillna(0.0)

    data = df[feature_cols + ["Return"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, :])
        y.append(scaled[i, -1])
    return np.array(X), np.array(y), scaler

def build_and_train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    n_features: int,
    val_split: float = 0.2
) -> Sequential:
    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(window_size, n_features), dropout=0.2, recurrent_dropout=0.2),
        LSTM(50, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="huber")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[es], verbose=0)
    return model

def forecast_lstm(
    model: Sequential,
    last_window: np.ndarray,
    steps: int = 3,
    scaler: StandardScaler = None,
    last_price: float = None
) -> List[float]:
    preds = []
    window = last_window.copy()
    for _ in range(steps):
        ret_scaled = model.predict(window[np.newaxis, :, :])[0, 0]
        dummy = np.zeros((1, window.shape[1]))
        dummy[0, -1] = ret_scaled
        inv = scaler.inverse_transform(dummy)[0, -1]
        last_price = last_price * (1 + inv)
        preds.append(last_price)
        next_row = window[-1].copy()
        next_row[-1] = ret_scaled
        window = np.vstack([window[1:], next_row])
    return preds

# ─── Combined wrapper ──────────────────────────────────────────────────────
def train_predict_stock(ticker: str, sentiment_series: Optional[pd.Series] = None):
    """
    Wrapper for main.py:
      • fetch histories (price, market, peer)
      • tree-based RF+XGB forecast & LSTM forecast
      • ensemble & confidence
      • package as dict for plotting & email
    """
    logger.debug(f"=== train_predict_stock: {ticker} ===")
    price_df  = fetch_price_history(ticker)
    market_df = fetch_price_history("^GSPC")
    peer_df   = fetch_price_history(ticker)

    # Tree-based forecast
    tree_df, conf_tree, red_tree = train_and_forecast(
        ticker, price_df, market_df, peer_df, sentiment_series
    )
    tree_preds = list(tree_df["Forecast_Close"])

    # Merge for LSTM + ensemble
    merged = price_df.rename(columns={"Stock_Close":"Stock_Close"}).merge(
        market_df.rename(columns={"Stock_Close":"Market_Close"})[["Date","Market_Close"]],
        on="Date", how="inner"
    ).merge(
        peer_df.rename(columns={"Stock_Close":"Peer_Close"})[["Date","Peer_Close"]],
        on="Date", how="inner"
    )
    merged["Return"]        = merged["Stock_Close"].pct_change().fillna(0.0)
    merged["Market_Return"] = merged["Market_Close"].pct_change().fillna(0.0)
    merged["Peer_Return"]   = merged["Peer_Close"].pct_change().fillna(0.0)
    merged["Volatility"]    = merged["Return"].rolling(5, min_periods=1).std().fillna(0.0)
    if sentiment_series is not None and not sentiment_series.empty:
        sent = sentiment_series.copy(); sent.index = pd.to_datetime(sent.index)
        sent_daily = sent.groupby(sent.index).mean().reindex(pd.to_datetime(merged["Date"]), method="ffill").fillna(0.0)
        merged["Sentiment"] = sent_daily.ewm(halflife=7.0).mean().values
    else:
        merged["Sentiment"] = 0.0

    # --- Guard against no merged data ---------------------------------------
    if merged.empty:
        import pandas as _pd
        logger.warning(f"No merged history for {ticker}, skipping LSTM/ensemble")
        dates   = []
        history = _pd.DataFrame(columns=["Date", "Stock_Close"])
        return {
            "dates":       dates,
            "history":     history,
            "predictions": tree_preds,
            "confidence":  conf_tree,
            "red_flag":    red_tree,
        }

    feature_cols = ["Market_Return","Peer_Return","Volatility","Sentiment"]
    try:
        X, y, scaler = prepare_lstm_data(
            merged[["Date","Stock_Close"] + feature_cols + ["Return"]],
            feature_cols,
            window_size=20
        )
        if len(X) >= 50:
            model = build_and_train_lstm(X, y, window_size=20, n_features=X.shape[2])
            last_scaled_window = X[-1]
            last_price = merged["Stock_Close"].iloc[-1]
            lstm_preds = forecast_lstm(model, last_scaled_window, steps=3, scaler=scaler, last_price=last_price)
            conf_lstm = 1.0
        else:
            logger.warning(f"Not enough data for LSTM on {ticker}, skipping LSTM path.")
            lstm_preds = tree_preds.copy()
            conf_lstm = conf_tree
    except Exception:
        logger.exception(f"LSTM pipeline failed for {ticker}, falling back")
        lstm_preds = tree_preds.copy()
        conf_lstm = conf_tree

    # Ensemble final predictions
    final_preds = [
        round(np.mean([t, l]), 2)
        for t, l in zip(tree_preds, lstm_preds)
    ]
    stds       = np.std(np.vstack([tree_preds, lstm_preds]), axis=0)
    avg_vol    = float(merged["Volatility"].iloc[-1]) + 1e-6
    confidence = float(1.0 / (1.0 + np.mean(stds) * avg_vol))
    red_flag   = red_tree

    dates   = [pd.to_datetime(r["Date"]) for r in tree_df.to_dict("records")]
    history = price_df.rename(columns={"Stock_Close":"AdjClose"})[["Date","AdjClose"]]

    return {
        "dates":       dates,
        "history":     history,
        "predictions": final_preds,
        "confidence":  confidence,
        "red_flag":    red_flag
    }
