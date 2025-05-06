import os
import time
import random
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger("prediction")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(message)s")

# ─── Load Alpha Vantage key ───────────────────────────────────────────────────
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
if not API_KEY:
    logger.error("Missing ALPHA_VANTAGE_KEY")
    raise RuntimeError("Set ALPHA_VANTAGE_KEY in config/secrets.env")

# ─── In-memory cache ───────────────────────────────────────────────────────────
_bulk_price_cache: Dict[str, pd.DataFrame] = {}

def fetch_price_history_bulk(
    tickers: List[str], period: str = "90d"
) -> Dict[str, pd.DataFrame]:
    global _bulk_price_cache
    cache_dir = "data/av_bulk_cache"
    os.makedirs(cache_dir, exist_ok=True)
    now = time.time()
    to_dl = []

    # load existing or mark for download
    for tic in tickers:
        path = f"{cache_dir}/{tic}_{period}.csv"
        if os.path.exists(path) and now - os.path.getmtime(path) < 86400:
            try:
                df = pd.read_csv(path)
                if "Date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "Date"})
                df["Date"] = pd.to_datetime(df["Date"])
                if "Stock_Close" not in df.columns:
                    df = df.rename(columns={df.columns[1]: "Stock_Close"})
                df = df[["Date", "Stock_Close"]]
                _bulk_price_cache[tic] = df.sort_values("Date").reset_index(drop=True)
                continue
            except Exception:
                pass
        to_dl.append(tic)

    # download fresh
    for tic in to_dl:
        try:
            logger.info(f"AV downloading: {tic}")
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function":   "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol":     tic,
                    "apikey":     API_KEY,
                    "outputsize": "compact"
                },
                timeout=(3, 5)
            )
            resp.raise_for_status()
            raw = resp.json().get("Time Series (Daily)", {})
            if not raw:
                logger.warning(f"[AV bulk] no data for {tic}")
                df = pd.DataFrame(columns=["Date","Stock_Close"])
            else:
                df = (
                    pd.DataFrame.from_dict(raw, orient="index")
                      .rename(columns={"5. adjusted close":"Stock_Close"})
                      .loc[:, ["Stock_Close"]]
                      .reset_index()
                      .rename(columns={"index":"Date"})
                )
                df["Date"]        = pd.to_datetime(df["Date"])
                df["Stock_Close"] = df["Stock_Close"].astype(float)
                df.sort_values("Date", inplace=True)
            df.to_csv(path, index=False)
            _bulk_price_cache[tic] = df
            time.sleep(12 + random.random())
        except Exception as e:
            logger.warning(f"[AV bulk] download failed for {tic}: {e}")

    return _bulk_price_cache

def fetch_price_history(ticker: str, period: str = "90d") -> pd.DataFrame:
    if ticker not in _bulk_price_cache:
        fetch_price_history_bulk([ticker], period)
    df = _bulk_price_cache.get(ticker, pd.DataFrame(columns=["Date","Stock_Close"]))
    try:
        days  = int(period.rstrip("d"))
        cut   = pd.Timestamp.today() - pd.Timedelta(days=days)
        df    = df[df["Date"] >= cut].reset_index(drop=True)
    except:
        pass
    return df.rename(columns={"Stock_Close":"Close"})[["Date","Close"]]

def train_and_forecast(
    ticker: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    peer_df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, float, bool]:
    try:
        df = (
            price_df
            .merge(market_df.rename(columns={"Close":"Market_Close"}), on="Date")
            .merge(peer_df.rename(columns={"Close":"Peer_Close"}), on="Date")
        )
        df["Return"]     = df["Close"].pct_change()
        df["MR"]         = df["Market_Close"].pct_change()
        df["PR"]         = df["Peer_Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(5, min_periods=1).std().fillna(0)

        if sentiment_series is not None and not sentiment_series.empty:
            sent      = sentiment_series.copy(); sent.index = pd.to_datetime(sent.index)
            daily     = sent.groupby(sent.index).mean().reindex(df["Date"], method="ffill").fillna(0)
            df["Sentiment"] = daily.ewm(halflife=7).mean().values
        else:
            df["Sentiment"] = 0.0

        df.dropna(inplace=True)
        if len(df) < 2:
            return pd.DataFrame(columns=["Date","Forecast_Close"]), 0.0, False

        X = df[["MR","PR","Sentiment","Volatility"]].values
        y = df["Return"].values

        rf  = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        rf.fit(X, y)
        xgb.fit(X, y)

        preds_rf  = rf.predict(X)
        preds_xgb = xgb.predict(X)
        df["EnsembleRet"] = (preds_rf + preds_xgb) / 2
        df["Pred_Close"]  = df["Close"].iloc[0]
        for i in range(1, len(df)):
            df.at[i,"Pred_Close"] = df.at[i-1,"Pred_Close"] * (1 + df.at[i,"EnsembleRet"])

        last   = df.iloc[-1]
        base   = last["Pred_Close"]
        date0  = last["Date"]
        futs   = []
        vols   = []
        for h in range(1,4):
            feat = [[last["MR"], last["PR"], last["Sentiment"], last["Volatility"]]]
            e    = (rf.predict(feat)[0] + xgb.predict(feat)[0]) / 2
            base *= (1 + e)
            futs.append({
                "Date":           (date0 + timedelta(days=h)).strftime("%Y-%m-%d"),
                "Forecast_Close": round(base,2)
            })
            vols.append(e)

        fc_df = pd.DataFrame(futs)
        eps   = 1e-6
        conf  = 1.0 / (1.0 + np.std(vols) * (last["Volatility"] + eps))
        red   = (vols[0] <= -0.05) and (last["Sentiment"] <= -0.3)
        return fc_df, float(conf), bool(red)

    except Exception as e:
        logger.exception(f"train_and_forecast err for {ticker}")
        return pd.DataFrame(columns=["Date","Forecast_Close"]), 0.0, False

def prepare_lstm_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 20
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    df = df.set_index("Date").asfreq("B").ffill().reset_index()
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change().fillna(0)
    data   = df[feature_cols + ["Return"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    X,y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, :])
        y.append(scaled[i, -1])
    return np.array(X), np.array(y), scaler

def build_and_train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    n_features: int
) -> Sequential:
    split      = int(len(X) * 0.7)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(window_size,n_features), dropout=0.2, recurrent_dropout=0.2),
        LSTM(50, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=50, batch_size=32, callbacks=[es], verbose=0)
    return model

def forecast_lstm(
    model: Sequential,
    last_window: np.ndarray,
    steps: int,
    scaler: StandardScaler,
    last_price: float
) -> List[float]:
    preds  = []
    window = last_window.copy()
    for _ in range(steps):
        r_s = model.predict(window[np.newaxis,:,:])[0,0]
        dummy = np.zeros((1, window.shape[1]))
        dummy[0,-1] = r_s
        ret = scaler.inverse_transform(dummy)[0,-1]
        last_price *= (1 + ret)
        preds.append(round(last_price,2))
        row = window[-1].copy(); row[-1] = r_s
        window = np.vstack([window[1:], row])
    return preds

def train_predict_stock(
    ticker: str,
    sentiment_series: Optional[pd.Series] = None
) -> dict:
    logger.debug(f"=== train_predict_stock: {ticker} ===")
    price_df = fetch_price_history(ticker)
    mkt_df   = fetch_price_history("^GSPC")
    peer_df  = fetch_price_history(ticker)

    tree_df, c_tree, r_tree = train_and_forecast(ticker, price_df, mkt_df, peer_df, sentiment_series)
    tree_preds = list(tree_df["Forecast_Close"])

    merged = (
        price_df
        .merge(mkt_df.rename(columns={"Close":"Market_Close"}), on="Date")
        .merge(peer_df.rename(columns={"Close":"Peer_Close"}), on="Date")
    )
    merged["Return"]     = merged["Close"].pct_change().fillna(0)
    merged["MR"]         = merged["Market_Close"].pct_change().fillna(0)
    merged["PR"]         = merged["Peer_Close"].pct_change().fillna(0)
    merged["Volatility"] = merged["Return"].rolling(5,min_periods=1).std().fillna(0)
    if sentiment_series is not None and not sentiment_series.empty:
        sent  = sentiment_series.copy(); sent.index = pd.to_datetime(sent.index)
        daily = sent.groupby(sent.index).mean().reindex(merged["Date"], method="ffill").fillna(0)
        merged["Sentiment"] = daily.ewm(halflife=7).mean().values
    else:
        merged["Sentiment"] = 0.0

    if merged.empty:
        logger.warning(f"No merged history for {ticker}, skipping LSTM")
        return {
            "dates":       [],
            "history":     price_df,
            "predictions": tree_preds,
            "confidence":  c_tree, 
            "red_flag":    r_tree
        }

    fcols = ["MR","PR","Volatility","Sentiment"]
    try:
        X, y, sc = prepare_lstm_data(merged[["Date","Close"] + fcols + ["Return"]], fcols, window_size=20)
        if len(X) >= 50:
            model      = build_and_train_lstm(X, y, 20, X.shape[2])
            lstm_preds = forecast_lstm(model, X[-1], 3, sc, merged["Close"].iloc[-1])
            c_lstm     = 1.0
        else:
            lstm_preds, c_lstm = tree_preds.copy(), c_tree
    except:
        lstm_preds, c_lstm = tree_preds.copy(), c_tree

    final_preds = [round((t + l)/2,2) for t, l in zip(tree_preds, lstm_preds)]
    stds        = np.std(np.vstack([tree_preds, lstm_preds]), axis=0)
    avg_vol     = merged["Volatility"].iloc[-1] + 1e-6
    conf        = float(1.0 / (1.0 + np.mean(stds) * avg_vol))

    dates   = [pd.to_datetime(r["Date"]) for r in tree_df.to_dict("records")]
    history = price_df[["Date","Close"]]

    return {
        "dates":       dates,
        "history":     history,
        "predictions": final_preds,
        "confidence":  conf,
        "red_flag":    r_tree
    }
