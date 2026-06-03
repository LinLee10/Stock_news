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
# TensorFlow and Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict

# BEGIN F01 - Multisource price data integration
import asyncio
from config.feature_flags import is_multisource_prices_enabled
# END F01

# BEGIN YF_INTEGRATION_READ - YFinance price data integration
from config.feature_flags import is_yf_prices_enabled
# END YF_INTEGRATION_READ

# BEGIN F12 - Forecasting extensions
from config.feature_flags import is_alt_forecasts_enabled, is_timegpt_stub_enabled
# END F12

logger = logging.getLogger("prediction")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

# ─── Load Alpha Vantage API Key ───────────────────────────────────────────────
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
if not API_KEY:
    logger.error("Missing ALPHA_VANTAGE_KEY")
    raise RuntimeError("Set ALPHA_VANTAGE_KEY in config/secrets.env")

# ─── In-memory cache for price data ───────────────────────────────────────────
_bulk_price_cache: Dict[str, pd.DataFrame] = {}

# ─── Daily API call counter ───────────────────────────────────────────────────
_daily_api_calls = 0
_last_api_reset = None

# ─── Available tickers in free Alpha Vantage API ─────────────────────────────
# These are common tickers that work with the free API
AVAILABLE_TICKERS = {
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "NFLX", "CRM", 
    "ADBE", "PYPL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU", "NVDA", "TSLA"
}
# ─── Sector-based peer mapping for peer correlation feature ───────────────────
# Maps each ticker to a peer stock in the same sector for meaningful correlation
TICKER_PEER_MAP: Dict[str, str] = {
    # Technology - Semiconductors
    "NVDA": "AMD",
    "AMD": "NVDA",
    "INTC": "AMD",
    "QCOM": "AVGO",
    "AVGO": "QCOM",
    "TXN": "ADI",
    "MU": "NVDA",
    "MRVL": "NVDA",
    "ADI": "TXN",
    # Technology - Software/Cloud
    "MSFT": "GOOGL",
    "GOOGL": "MSFT",
    "AAPL": "MSFT",
    "META": "GOOGL",
    "NFLX": "META",
    "CRM": "ADBE",
    "ADBE": "CRM",
    # Technology - EVs/Auto
    "TSLA": "RIVN",
    "RIVN": "TSLA",
    # Healthcare/Pharma
    "PFE": "LLY",
    "LLY": "PFE",
    # Defense/Aerospace
    "RTX": "LMT",
    "LMT": "RTX",
    # Energy
    "UUUU": "CCJ",
    "CCJ": "UUUU",
    # Fintech
    "PYPL": "SQ",
    "SQ": "PYPL",
    # Data/Analytics
    "PLTR": "SNOW",
    "SNOW": "PLTR",
    "SRAD": "DKNG",
    "DKNG": "SRAD",
    # E-commerce
    "AMZN": "GOOGL",
}

def get_peer_ticker(ticker: str) -> str:
    """
    Get a peer ticker for the given stock based on sector/industry.
    Falls back to S&P 500 (^GSPC) if no specific peer is mapped.
    """
    return TICKER_PEER_MAP.get(ticker, "^GSPC")


def _check_daily_limit() -> bool:
    """
    Check if we've hit the daily API limit (25 calls per day).
    Returns True if we can make another call, False if limit reached.
    """
    global _daily_api_calls, _last_api_reset
    
    now = time.time()
    today = time.strftime("%Y-%m-%d")
    
    # Reset counter if it's a new day
    if _last_api_reset != today:
        _daily_api_calls = 0
        _last_api_reset = today
    
    # Check if we've hit the limit
    if _daily_api_calls >= 25:
        logger.warning(f"Daily API limit reached (25 calls). Using cached data only.")
        return False
    
    return True

def _increment_api_counter():
    """Increment the daily API call counter."""
    global _daily_api_calls
    _daily_api_calls += 1
    logger.info(f"API call {_daily_api_calls}/25 for today")

def fetch_price_history_bulk(tickers: List[str], period: str = "90d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical adjusted close prices for multiple tickers using Alpha Vantage.
    Caches results on disk and in memory. `period` (e.g. "90d") is used for cache file naming 
    and filtering (last N days). Respects 25-call daily limit.
    """
    global _bulk_price_cache
    
    # BEGIN YF_INTEGRATION_READ - Try yfinance cache first when enabled
    if is_yf_prices_enabled():
        try:
            from services.multi_source_data_manager import data_manager
            
            logger.info(f"Checking yfinance cache for {len(tickers)} tickers")
            
            # Check for cached yfinance data first
            cached_data = data_manager.get_yf_cached_data(tickers)
            
            if cached_data:
                logger.info(f"Found yfinance cache data for {len(cached_data)} tickers")
                
                # Convert yfinance format to expected format
                result = {}
                for ticker, df in cached_data.items():
                    if df is not None and not df.empty:
                        # YFinance data has columns: Date, Open, High, Low, Close, Volume, Adj Close, Symbol
                        # Convert to expected format (Date, Stock_Close)
                        if 'Adj Close' in df.columns:
                            price_col = 'Adj Close'
                        elif 'Close' in df.columns:
                            price_col = 'Close'
                        else:
                            logger.warning(f"No price column found in yfinance data for {ticker}")
                            continue
                            
                        converted_df = pd.DataFrame({
                            "Date": pd.to_datetime(df["Date"]),
                            "Stock_Close": df[price_col]
                        })
                        
                        # Filter to requested period
                        try:
                            days = int(period.replace('d', '')) if period.endswith('d') else 90
                            cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
                            converted_df = converted_df[converted_df["Date"] >= cutoff]
                        except Exception as e:
                            logger.warning(f"Error filtering yfinance data for {ticker}: {e}")
                        
                        converted_df = converted_df.sort_values("Date").reset_index(drop=True)
                        result[ticker] = converted_df
                        
                        # Update legacy cache for consistency
                        _bulk_price_cache[ticker] = converted_df
                
                # If we got data for all requested tickers, return it
                if len(result) == len(tickers):
                    logger.info(f"YFinance cache provided all {len(result)} tickers")
                    return result
                elif result:
                    logger.info(f"YFinance cache provided {len(result)}/{len(tickers)} tickers, falling back for rest")
                    # For partial cache hits, we'd need to merge with other sources
                    # For now, fall through to other providers for simplicity
                else:
                    logger.info("No yfinance cache data found, falling back to other sources")
                    
        except Exception as e:
            logger.warning(f"YFinance cache check failed: {e}, falling back to other sources")
    # END YF_INTEGRATION_READ
    
    # BEGIN F01 - Use multisource data manager when enabled
    if is_multisource_prices_enabled():
        try:
            from services.multi_source_data_manager import data_manager
            
            # Convert period string to days (e.g. "90d" -> 90)
            lookback_days = int(period.replace('d', '')) if period.endswith('d') else 90
            
            logger.info(f"Using multisource price provider for {len(tickers)} tickers")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                price_data = loop.run_until_complete(
                    data_manager.get_price_data(tickers, lookback_days)
                )
                
                # Convert to expected format for backward compatibility
                result = {}
                for ticker, df in price_data.items():
                    if df is not None and not df.empty:
                        # Convert to expected columns (Date, Stock_Close)
                        converted_df = pd.DataFrame({
                            "Date": df.index,
                            "Stock_Close": df['close'] if 'close' in df.columns else df['adjusted_close']
                        })
                        converted_df = converted_df.sort_values("Date").reset_index(drop=True)
                        result[ticker] = converted_df
                        
                        # Also update legacy cache
                        _bulk_price_cache[ticker] = converted_df
                
                logger.info(f"Multisource fetch completed: {len(result)}/{len(tickers)} successful")
                return result
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Multisource price fetch failed: {e}")
            logger.info("Falling back to legacy Alpha Vantage implementation")
            # Fall through to legacy implementation
    # END F01
    
    # Legacy Alpha Vantage implementation continues here
    cache_dir = "data/av_bulk_cache"
    os.makedirs(cache_dir, exist_ok=True)
    now = time.time()
    to_download = []  # list of tickers we need to fetch from API

    # First pass: check cache and load what we can
    # Use a longer cache validity period (7 days) since API is rate-limited
    cache_max_age = 604800  # 7 days in seconds
    for tic in tickers:
        path = f"{cache_dir}/{tic}_{period}.csv"
        if os.path.exists(path) and now - os.path.getmtime(path) < cache_max_age:
            # Recent cache exists, load it
            try:
                df = pd.read_csv(path)
                # Check if cache has actual data (not just headers)
                if len(df) <= 1:
                    logger.warning(f"Cache file for {tic} is empty or has only headers, will re-download")
                    to_download.append(tic)
                    continue
                    
                # Ensure expected columns
                if "Date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "Date"})
                df["Date"] = pd.to_datetime(df["Date"])
                if "Stock_Close" not in df.columns:
                    # If cached from old source, second column might be close price
                    df = df.rename(columns={df.columns[1]: "Stock_Close"})
                df = df[["Date", "Stock_Close"]]
                
                # Check if data is recent enough (within last 7 days)
                latest_date = df["Date"].max()
                days_since_latest = (pd.Timestamp.now() - latest_date).days
                if days_since_latest > 7:
                    logger.warning(f"Cache for {tic} is outdated (latest: {latest_date.date()}, {days_since_latest} days old), will refresh")
                    to_download.append(tic)
                    continue
                    
                # Sort by date and reset index
                df = df.sort_values("Date").reset_index(drop=True)
                _bulk_price_cache[tic] = df
                logger.info(f"Loaded cached data for {tic} (latest: {latest_date.date()})")
                continue  # use cache, skip download
            except Exception as e:
                logger.warning(f"Error loading cache for {tic}: {e}")
                # If cache read fails, mark for re-download
        to_download.append(tic)

    # Second pass: download data for tickers without fresh cache (respecting daily limit)
    for tic in to_download:
        # Check daily limit before making API call
        if not _check_daily_limit():
            logger.warning(f"Skipping {tic} due to daily API limit. Using old cache if available.")
            # Try to load old cache - any age is better than nothing
            cache_path = f"{cache_dir}/{tic}_{period}.csv"
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path)
                    if len(df) > 1:  # Has data beyond header
                        if "Date" not in df.columns:
                            df = df.rename(columns={df.columns[0]: "Date"})
                        df["Date"] = pd.to_datetime(df["Date"])
                        if "Stock_Close" not in df.columns:
                            df = df.rename(columns={df.columns[1]: "Stock_Close"})
                        df = df[["Date", "Stock_Close"]]
                        df = df.sort_values("Date").reset_index(drop=True)
                        _bulk_price_cache[tic] = df
                        latest = df["Date"].max().date()
                        logger.info(f"Using old cached data for {tic} (latest: {latest}, daily limit reached)")
                        continue
                except Exception as e:
                    logger.warning(f"Error loading old cached data for {tic}: {e}")
            
            # Create empty DataFrame if no cache available
            _bulk_price_cache[tic] = pd.DataFrame(columns=["Date", "Stock_Close"])
            continue

        try:
            logger.info(f"Fetching {tic} from Alpha Vantage...")
            _increment_api_counter()
            
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function":   "TIME_SERIES_DAILY",
                    "symbol":     tic,
                    "apikey":     API_KEY,
                    "outputsize": "full"
                },
                timeout=(3, 10)  # short connect timeout, longer read timeout
            )
            resp.raise_for_status()
            raw_timeseries = resp.json().get("Time Series (Daily)", {})
            
            # Check for rate limit or other API issues
            if not raw_timeseries:
                info_msg = resp.json().get("Information", "")
                error_msg = resp.json().get("Error Message", "")
                note_msg = resp.json().get("Note", "")
                
                # Check for rate limit OR premium feature restriction
                is_rate_limited = "rate limit" in info_msg.lower() or "rate limit" in note_msg.lower()
                is_premium_only = "premium" in info_msg.lower() or "premium" in note_msg.lower()
                
                if is_rate_limited or is_premium_only:
                    limit_reason = "rate limit" if is_rate_limited else "premium feature restriction"
                    logger.warning(f"API {limit_reason} for {tic}. Using cached data if available.")
                    # Try to load from cache even if old
                    cache_path = f"{cache_dir}/{tic}_{period}.csv"
                    if os.path.exists(cache_path):
                        try:
                            df = pd.read_csv(cache_path)
                            if len(df) > 1:  # Has data beyond header
                                df["Date"] = pd.to_datetime(df["Date"])
                                df = df[["Date", "Stock_Close"]]
                                df = df.sort_values("Date").reset_index(drop=True)
                                _bulk_price_cache[tic] = df
                                logger.info(f"Using cached data for {tic} (rate limit reached)")
                                continue
                        except Exception as e:
                            logger.warning(f"Error loading cached data for {tic}: {e}")
                    
                    # If no cache available, create empty DataFrame but don't save it
                    df = pd.DataFrame(columns=["Date", "Stock_Close"])
                    logger.warning(f"No cached data available for {tic}, API limit reached - skipping")
                    _bulk_price_cache[tic] = df
                    continue  # Skip saving empty file
                else:
                    logger.warning(f"No price data returned for {tic}: {info_msg} {error_msg} {note_msg}")
                    df = pd.DataFrame(columns=["Date", "Stock_Close"])
                    _bulk_price_cache[tic] = df
                    continue  # Skip saving empty file
            else:
                # Convert JSON time series to DataFrame
                df = pd.DataFrame.from_dict(raw_timeseries, orient="index")
                # Use close price (not adjusted close) for free endpoint
                if "4. close" in df.columns:
                    df = df.rename(columns={"4. close": "Stock_Close"})
                elif "5. adjusted close" in df.columns:
                    df = df.rename(columns={"5. adjusted close": "Stock_Close"})
                else:
                    logger.warning(f"Unexpected column structure for {tic}: {df.columns.tolist()}")
                    df = pd.DataFrame(columns=["Date", "Stock_Close"])
                
                if "Stock_Close" in df.columns:
                    df = df[["Stock_Close"]].reset_index().rename(columns={"index": "Date"})
                    df["Date"] = pd.to_datetime(df["Date"])
                    df["Stock_Close"] = df["Stock_Close"].astype(float)
                    df.sort_values("Date", inplace=True)
                else:
                    df = pd.DataFrame(columns=["Date", "Stock_Close"])
            
            # Only save to cache if we have actual data
            if len(df) > 0 and "Stock_Close" in df.columns:
                df.to_csv(f"{cache_dir}/{tic}_{period}.csv", index=False)
                logger.info(f"Saved {len(df)} data points for {tic} to cache")
            else:
                logger.warning(f"No data to save for {tic}, skipping cache write")
                
            _bulk_price_cache[tic] = df.reset_index(drop=True)
            # Rate limit: wait 12+ seconds after each API call
            time.sleep(12 + random.random())
            
        except Exception as e:
            logger.warning(f"Failed to download data for {tic}: {e}")
            # Try to load from cache if available
            cache_path = f"{cache_dir}/{tic}_{period}.csv"
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path)
                    if len(df) > 1:  # Has data beyond header
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df[["Date", "Stock_Close"]]
                        df = df.sort_values("Date").reset_index(drop=True)
                        _bulk_price_cache[tic] = df
                        logger.info(f"Using cached data for {tic} after download failure")
                        continue
                except Exception as cache_e:
                    logger.warning(f"Error loading cached data for {tic}: {cache_e}")
            
            _bulk_price_cache[tic] = pd.DataFrame(columns=["Date", "Stock_Close"])
    
    return _bulk_price_cache

def fetch_price_history(ticker: str, period: str = "90d") -> pd.DataFrame:
    """
    Retrieve historical price DataFrame (Date, Close) for a single ticker.
    Uses cache or triggers download via fetch_price_history_bulk.
    """
    if ticker not in _bulk_price_cache:
        fetch_price_history_bulk([ticker], period)
    df = _bulk_price_cache.get(ticker, pd.DataFrame(columns=["Date", "Stock_Close"]))
    # Filter to the last N days specified by period
    try:
        days = int(period.rstrip("d"))
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
        filtered_df = df[df["Date"] >= cutoff].reset_index(drop=True)
        
        # If no data in the recent period, use the most recent N days of available data
        if len(filtered_df) < 10 and len(df) >= 10:
            logger.warning(f"No recent data for {ticker} (cutoff: {cutoff.date()}), using last {days} rows of cached data")
            df = df.tail(days).reset_index(drop=True)
        else:
            df = filtered_df
    except Exception as e:
        logger.warning(f"Error filtering price history for {ticker}: {e}")
    # Rename Stock_Close to Close for consistency
    out_df = df.copy()
    if "Stock_Close" in out_df.columns:
        out_df = out_df.rename(columns={"Stock_Close": "Close"})
    # Ensure correct columns and types
    if "Date" in out_df.columns and "Close" in out_df.columns:
        out_df = out_df[["Date", "Close"]]
        out_df["Date"] = pd.to_datetime(out_df["Date"])
    else:
        # If data is missing, return empty DataFrame with expected columns
        out_df = pd.DataFrame(columns=["Date", "Close"])
    return out_df

def train_and_forecast(
    ticker: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    peer_df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, float, bool]:
    """
    Train ensemble (RandomForest + XGBoost) models on historical data and forecast 3 days.
    Returns a DataFrame of forecast prices, a confidence score, and a red_flag indicator.
    """
    try:
        # Merge price, market, and peer data on Date
        df = price_df.merge(
            market_df.rename(columns={"Close": "Market_Close"}),
            on="Date", how="outer"
        ).merge(
            peer_df.rename(columns={"Close": "Peer_Close"}),
            on="Date", how="outer"
        )
        df.sort_values("Date", inplace=True)
        df = df.copy()  # avoid SettingWithCopy warnings
        # Forward-fill to propagate last known values
        df["Close"]        = df["Close"].ffill()
        df["Market_Close"] = df["Market_Close"].ffill()
        df["Peer_Close"]   = df["Peer_Close"].ffill()
        # Calculate features: daily returns and volatility
        df["Return"]     = df["Close"].pct_change()
        df["MR"]         = df["Market_Close"].pct_change()
        df["PR"]         = df["Peer_Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(5, min_periods=1).std().fillna(0)
        # Integrate sentiment data
        if sentiment_series is not None and not sentiment_series.empty:
            # Align sentiment by date, forward-fill and smooth
            sent = sentiment_series.copy()
            sent.index = pd.to_datetime(sent.index)
            daily = sent.groupby(sent.index).mean()  # average if multiple entries same day
            daily = daily.reindex(df["Date"], method="ffill").fillna(0)
            # Exponential moving average on sentiment to decay old news influence
            df["Sentiment"] = daily.ewm(halflife=7).mean().values
        else:
            df["Sentiment"] = 0.0
        df.dropna(inplace=True)
        if len(df) < 2:
            # Not enough data to train
            return pd.DataFrame(columns=["Date", "Forecast_Close"]), 0.0, False

        # Prepare features (X) and target (y) for training
        features = ["MR", "PR", "Sentiment", "Volatility"]
        X = df[features].values
        y = df["Return"].values

        # Train ensemble models (Random Forest and XGBoost)
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        xgb_model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        rf_model.fit(X, y)
        xgb_model.fit(X, y)

        # In-sample predictions (not really used directly for output, but for forecasting base)
        preds_rf  = rf_model.predict(X)
        preds_xgb = xgb_model.predict(X)
        df["EnsembleRet"] = (preds_rf + preds_xgb) / 2  # ensemble predicted return
        # Reconstruct predicted Close prices from returns
        df["Pred_Close"] = df["Close"].iloc[0]
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc("Pred_Close")] = df.iloc[i-1]["Pred_Close"] * (1 + df.iloc[i]["EnsembleRet"])

        # Use the last known data point to forecast forward
        last_row = df.iloc[-1]
        last_date = last_row["Date"]
        base_price = last_row["Pred_Close"]
        future_preds = []
        future_rets = []
        for h in range(1, 4):  # forecast 3 days ahead
            features_arr = [[last_row["MR"], last_row["PR"], last_row["Sentiment"], last_row["Volatility"]]]
            # Predict return for next step using the ensemble
            ret_pred = (rf_model.predict(features_arr)[0] + xgb_model.predict(features_arr)[0]) / 2
            base_price *= (1 + ret_pred)
            future_date = (last_date + timedelta(days=h)).strftime("%Y-%m-%d")
            future_preds.append({"Date": future_date, "Forecast_Close": round(base_price, 2)})
            future_rets.append(ret_pred)
        forecast_df = pd.DataFrame(future_preds)
        # Confidence: higher if predictions are consistent and volatility is low
        eps = 1e-6
        confidence = 1.0 / (1.0 + np.std(future_rets) * (last_row["Volatility"] + eps))
        # Red flag: trigger if immediate drop is big and sentiment is very negative
        red_flag = bool(future_rets[0] <= -0.05 and last_row["Sentiment"] <= -0.3)
        return forecast_df, float(confidence), red_flag

    except Exception as e:
        logger.exception(f"train_and_forecast error for {ticker}: {e}")
        return pd.DataFrame(columns=["Date", "Forecast_Close"]), 0.0, False

def prepare_lstm_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 20
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare sequences of features and targets for LSTM training.
    Returns (X, y, scaler) where X is 3D array for LSTM and y is target returns.
    """
    # Ensure the DataFrame has a continuous index (business days) and fill missing with last value
    df = df.set_index("Date").asfreq("B").ffill().reset_index()
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change().fillna(0)
    data = df[feature_cols + ["Return"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    X_seq, y_seq = [], []
    for i in range(window_size, len(scaled)):
        X_seq.append(scaled[i-window_size:i, :])
        y_seq.append(scaled[i, -1])  # next-day return
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    return X_seq, y_seq, scaler

def build_and_train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    n_features: int
) -> Sequential:
    """
    Build and train an LSTM model for given feature sequences X and target y.
    Uses early stopping and returns the trained model.
    """
    # Split into training and validation sets
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    # Define LSTM model architecture
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(window_size, n_features), dropout=0.2, recurrent_dropout=0.2),
        LSTM(50, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    # Train with early stopping
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[es], verbose=0)
    return model

def forecast_lstm(
    model: Sequential,
    last_window: np.ndarray,
    steps: int,
    scaler: StandardScaler,
    last_close: float
) -> List[float]:
    """
    Use the trained LSTM model to forecast a number of steps ahead.
    Returns a list of predicted prices for the given number of steps.
    """
    preds = []
    window = last_window.copy()
    current_price = last_close
    for _ in range(steps):
        # Predict next-day return (scaled) using the last window
        scaled_ret_pred = model.predict(window[np.newaxis, :, :])[0, 0]
        # Inverse transform the predicted return to original scale
        dummy = np.zeros((1, window.shape[1]))  # dummy array to inverse transform
        dummy[0, -1] = scaled_ret_pred
        ret_pred = scaler.inverse_transform(dummy)[0, -1]
        # Update price
        current_price *= (1 + ret_pred)
        preds.append(round(current_price, 2))
        # Slide the window: drop the oldest entry, append the new predicted return
        new_row = window[-1].copy()
        new_row[-1] = scaled_ret_pred  # insert the predicted return in place of actual
        window = np.vstack([window[1:], new_row])
    return preds

def train_predict_stock(
    ticker: str,
    sentiment_series: Optional[pd.Series] = None
) -> dict:
    """
    Fetch historical data and train models to predict the given stock's near-term price.
    Returns a dictionary with keys: dates, history, predictions, confidence, red_flag.
    """
    logger.debug(f"=== train_predict_stock: {ticker} ===")
    # 1. Fetch historical price data for stock, market index (S&P 500), and peer
    price_df = fetch_price_history(ticker)
    market_df = fetch_price_history("^GSPC")    # using S&P 500 Index (^GSPC) as market indicator
    peer_df = fetch_price_history(get_peer_ticker(ticker))  # use sector peer for meaningful correlation
    if price_df.empty or len(price_df) < 10:
        data_points = len(price_df) if not price_df.empty else 0
        logger.warning(f"Insufficient historical data for {ticker} ({data_points} data points), need at least 10 for forecast.")
        return {
            "dates":       [],
            "history":     price_df,
            "predictions": [],
            "confidence":  0.0,
            "red_flag":    False
        }
    
    # If market data is empty, use the stock's own data as market proxy
    if market_df.empty:
        logger.warning(f"No market data available for {ticker}, using stock data as market proxy.")
        market_df = price_df.copy()
    # 2. Use ensemble tree models to forecast
    tree_df, conf_tree, red_tree = train_and_forecast(ticker, price_df, market_df, peer_df, sentiment_series)
    tree_preds = tree_df["Forecast_Close"].tolist()
    # 3. Prepare data for LSTM model
    merged = price_df.merge(
        market_df.rename(columns={"Close": "Market_Close"}), on="Date", how="outer"
    ).merge(
        peer_df.rename(columns={"Close": "Peer_Close"}), on="Date", how="outer"
    )
    merged.sort_values("Date", inplace=True)
    # Forward-fill missing values
    merged = merged.copy()
    merged["Close"]        = merged["Close"].ffill()
    merged["Market_Close"] = merged["Market_Close"].ffill()
    merged["Peer_Close"]   = merged["Peer_Close"].ffill()
    # Compute returns and features
    merged["Return"]     = merged["Close"].pct_change().fillna(0)
    merged["MR"]         = merged["Market_Close"].pct_change().fillna(0)
    merged["PR"]         = merged["Peer_Close"].pct_change().fillna(0)
    merged["Volatility"] = merged["Return"].rolling(5, min_periods=1).std().fillna(0)
    if sentiment_series is not None and not sentiment_series.empty:
        sent = sentiment_series.copy()
        sent.index = pd.to_datetime(sent.index)
        daily_sent = sent.groupby(sent.index).mean()
        daily_sent = daily_sent.reindex(merged["Date"], method="ffill").fillna(0)
        merged["Sentiment"] = daily_sent.ewm(halflife=7).mean().values
    else:
        merged["Sentiment"] = 0.0
    if merged.dropna().empty:
        # No overlapping data to train LSTM
        logger.warning(f"No merged history for {ticker} (possibly missing data), skipping LSTM.")
        return {
            "dates":       tree_df["Date"].tolist(),
            "history":     price_df,
            "predictions": tree_preds,
            "confidence":  conf_tree,
            "red_flag":    red_tree
        }
    # 4. Train LSTM model if we have enough data
    final_preds = tree_preds[:]  # start with tree predictions as base
    try:
        feature_cols = ["MR", "PR", "Volatility", "Sentiment"]
        X_seq, y_seq, scaler = prepare_lstm_data(merged[["Date", "Close"] + feature_cols + ["Return"]], feature_cols, window_size=20)
        if len(X_seq) >= 50:
            lstm_model = build_and_train_lstm(X_seq, y_seq, window_size=20, n_features=X_seq.shape[2])
            lstm_preds = forecast_lstm(lstm_model, X_seq[-1], steps=3, scaler=scaler, last_close=merged["Close"].iloc[-1])
            # Combine tree and LSTM forecasts by averaging
            final_preds = [round((t + l) / 2, 2) for t, l in zip(tree_preds, lstm_preds)]
            # Optionally, we could adjust confidence using LSTM, but we will recompute overall confidence below
        else:
            lstm_preds = tree_preds  # not enough data for LSTM, use tree results
    except Exception as e:
        logger.error(f"LSTM model error for {ticker}: {e}")
        lstm_preds = tree_preds
    # 5. Compute overall confidence as average agreement between models scaled by volatility
    preds_matrix = np.vstack([tree_preds, lstm_preds]) if 'lstm_preds' in locals() else np.vstack([tree_preds, tree_preds])
    std_dev = np.mean(np.std(preds_matrix, axis=0))
    latest_vol = merged["Volatility"].iloc[-1] if not merged.empty else 0
    confidence = float(1.0 / (1.0 + std_dev * (latest_vol + 1e-6)))
    # 6. BEGIN F12 - Run baseline forecasts and comparison
    baseline_results = {}
    timegpt_result = None
    comparison_data = []
    
    if is_alt_forecasts_enabled() or is_timegpt_stub_enabled():
        try:
            # Prepare price data for baseline forecasters
            price_data = []
            for _, row in price_df.iterrows():
                price_data.append({
                    'date': row['Date'],
                    'close': row['Close']
                })
            
            prices_by_symbol = {ticker: price_data}
            
            # Run baseline forecasts (ARIMA/Prophet) if enabled
            if is_alt_forecasts_enabled():
                try:
                    from services.forecasting.baselines import run_baseline_forecasts
                    baseline_results = run_baseline_forecasts(prices_by_symbol, forecast_horizon=3)
                    logger.info(f"Baseline forecasts completed for {ticker}")
                except Exception as e:
                    logger.error(f"Baseline forecast error for {ticker}: {e}")
            
            # Run TimeGPT forecast if enabled
            if is_timegpt_stub_enabled():
                try:
                    from services.forecasting.timegpt_stub import run_timegpt_forecast
                    timegpt_results = run_timegpt_forecast(prices_by_symbol, forecast_horizon=3)
                    timegpt_result = timegpt_results.get(ticker)
                    logger.info(f"TimeGPT forecast completed for {ticker}")
                except Exception as e:
                    logger.error(f"TimeGPT forecast error for {ticker}: {e}")
            
            # Generate comparison data
            comparison_data = _generate_forecast_comparison(
                ticker, final_preds, confidence, baseline_results.get(ticker, []), timegpt_result
            )
            
            # Save comparison table
            if comparison_data:
                _save_forecast_comparison_table(comparison_data)
                
        except Exception as e:
            logger.error(f"F12 forecasting extensions error for {ticker}: {e}")
    # END F12
    
    # 7. Return result dict (AC2: existing predictions unchanged when flags off)
    result = {
        "dates":       [pd.to_datetime(d) for d in tree_df["Date"].tolist()],
        "history":     price_df[["Date", "Close"]],
        "predictions": final_preds,
        "confidence":  confidence,
        "red_flag":    bool(red_tree)
    }
    
    # BEGIN F12 - Add baseline results to output when enabled
    if is_alt_forecasts_enabled() and baseline_results:
        result["baseline_forecasts"] = baseline_results.get(ticker, [])
    
    if is_timegpt_stub_enabled() and timegpt_result:
        result["timegpt_forecast"] = timegpt_result
        
    if comparison_data:
        result["forecast_comparison"] = comparison_data
    # END F12
    
    return result


# BEGIN F12 - Helper functions for forecasting extensions
def _generate_forecast_comparison(
    ticker: str, 
    rf_xgb_preds: List[float], 
    rf_xgb_confidence: float,
    baseline_results: List = None,
    timegpt_result = None
) -> List[Dict]:
    """
    Generate comparison data for different forecasting models
    
    Args:
        ticker: Stock symbol
        rf_xgb_preds: RF/XGB ensemble predictions
        rf_xgb_confidence: RF/XGB confidence score
        baseline_results: List of baseline forecast results
        timegpt_result: TimeGPT forecast result
        
    Returns:
        List of comparison dictionaries
    """
    comparison_data = []
    
    # Add RF/XGB ensemble (existing model)
    for i, pred in enumerate(rf_xgb_preds):
        comparison_data.append({
            'symbol': ticker,
            'model': 'RF_XGB_Ensemble',
            'forecast_day': i + 1,
            'prediction': pred,
            'confidence': rf_xgb_confidence,
            'mae': None,  # Not available for existing model
            'rmse': None,
            'runtime_ms': None,
            'timestamp': pd.Timestamp.now()
        })
    
    # Add baseline results if available
    if baseline_results:
        for result in baseline_results:
            model_name = result.model_name
            if result.success and result.predictions:
                for i, pred in enumerate(result.predictions):
                    comparison_data.append({
                        'symbol': ticker,
                        'model': model_name,
                        'forecast_day': i + 1,
                        'prediction': pred,
                        'confidence': None,  # Baseline models don't use confidence scores
                        'mae': result.mae,
                        'rmse': result.rmse,
                        'runtime_ms': result.runtime_ms,
                        'timestamp': pd.Timestamp.now()
                    })
            else:
                # Add failed result
                comparison_data.append({
                    'symbol': ticker,
                    'model': model_name,
                    'forecast_day': 1,
                    'prediction': None,
                    'confidence': None,
                    'mae': None,
                    'rmse': None,
                    'runtime_ms': result.runtime_ms,
                    'error': result.error_message,
                    'timestamp': pd.Timestamp.now()
                })
    
    # Add TimeGPT result if available
    if timegpt_result:
        if timegpt_result.success and timegpt_result.predictions:
            for i, pred in enumerate(timegpt_result.predictions):
                comparison_data.append({
                    'symbol': ticker,
                    'model': 'TimeGPT_Stub',
                    'forecast_day': i + 1,
                    'prediction': pred,
                    'confidence': None,
                    'mae': timegpt_result.mae,
                    'rmse': timegpt_result.rmse,
                    'runtime_ms': timegpt_result.runtime_ms,
                    'api_credits_used': timegpt_result.api_credits_used,
                    'timestamp': pd.Timestamp.now()
                })
        else:
            # Add failed result
            comparison_data.append({
                'symbol': ticker,
                'model': 'TimeGPT_Stub',
                'forecast_day': 1,
                'prediction': None,
                'confidence': None,
                'mae': None,
                'rmse': None,
                'runtime_ms': timegpt_result.runtime_ms,
                'error': timegpt_result.error_message,
                'timestamp': pd.Timestamp.now()
            })
    
    return comparison_data


def _save_forecast_comparison_table(comparison_data: List[Dict]):
    """
    Save forecast comparison data to CSV file (AC1)
    
    Args:
        comparison_data: List of comparison dictionaries
    """
    if not comparison_data:
        return
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        comparison_file = 'data/forecast_comparison.csv'
        
        # Convert to DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Check if file exists
        if os.path.exists(comparison_file):
            # Append to existing file
            existing_df = pd.read_csv(comparison_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # Keep only last 1000 records to prevent file from growing too large
            if len(combined_df) > 1000:
                combined_df = combined_df.tail(1000)
        else:
            combined_df = df
        
        # Save to CSV
        combined_df.to_csv(comparison_file, index=False)
        
        logger.info(f"Forecast comparison saved to {comparison_file}: {len(df)} new records")
        
    except Exception as e:
        logger.error(f"Error saving forecast comparison: {e}")


def get_forecast_comparison_summary() -> Dict:
    """
    Get summary of forecast comparison results
    
    Returns:
        Dictionary with comparison summary statistics
    """
    try:
        comparison_file = 'data/forecast_comparison.csv'
        
        if not os.path.exists(comparison_file):
            return {"error": "No comparison data available"}
        
        df = pd.read_csv(comparison_file)
        
        if df.empty:
            return {"error": "Empty comparison data"}
        
        # Calculate summary statistics
        summary = {
            'total_forecasts': len(df),
            'unique_symbols': df['symbol'].nunique(),
            'unique_models': df['model'].nunique(),
            'models': df['model'].unique().tolist(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        # Model performance summary
        model_performance = {}
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Only include successful predictions
            successful_df = model_df[model_df['prediction'].notna()]
            
            if not successful_df.empty:
                model_performance[model] = {
                    'total_forecasts': len(model_df),
                    'successful_forecasts': len(successful_df),
                    'success_rate': len(successful_df) / len(model_df),
                    'avg_mae': successful_df['mae'].mean() if 'mae' in successful_df.columns else None,
                    'avg_rmse': successful_df['rmse'].mean() if 'rmse' in successful_df.columns else None,
                    'avg_runtime_ms': successful_df['runtime_ms'].mean() if 'runtime_ms' in successful_df.columns else None
                }
        
        summary['model_performance'] = model_performance
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting forecast comparison summary: {e}")
        return {"error": str(e)}
# END F12
