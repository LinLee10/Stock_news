# forecast_utils.py

import os
import logging
from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd

import logging

LGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception as e:
    logging.warning(f"LightGBM import failed ({e}), falling back to sklearn HistGradientBoostingRegressor")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

TRAIN_WINDOW_DAYS     = 180
SENTIMENT_WINDOW_DAYS = 90
FORECAST_HORIZON      = 3
TIMEZONE              = 'US/Eastern'

LOG_DIR       = "logs"
HISTORY_CSV   = os.path.join(LOG_DIR, "forecast_history.csv")
PLOTS_DIR     = "plots"
PRICE_DIR     = "data"
SENT_DIR      = "data"

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "forecast.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def compute_RSI(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def load_price_data(ticker):
    path = os.path.join(PRICE_DIR, f"prices_{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df.rename(columns={"date":"Date", "adj_close":"AdjClose"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    return df

def load_sentiment_data(ticker):
    path = os.path.join(SENT_DIR, f"sentiment_{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df.rename(columns={"date":"Date","sentiment":"Sentiment"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    return df.groupby('Date', as_index=False).Sentiment.mean()

def prepare_features(ticker):
    price     = load_price_data(ticker)
    sentiment = load_sentiment_data(ticker)
    df = pd.merge(price, sentiment, on='Date', how='left')
    df['Sentiment'].fillna(method='ffill', inplace=True)
    df['Return'] = df['AdjClose'].pct_change()
    df['RSI14']  = compute_RSI(df['AdjClose'], window=14)
    df['Vol30']  = df['Return'].rolling(30).std()
    for lag in (1,3,7):
        df[f'Sent_lag{lag}'] = df['Sentiment'].shift(lag)
    df.dropna(inplace=True)
    return df

def compute_metrics(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    if preds.shape != actuals.shape or len(preds)==0:
        return {}
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mask = actuals != 0
    mape = (np.abs((actuals[mask] - preds[mask]) / actuals[mask]) * 100).mean() if mask.any() else np.nan
    dir_pred = np.sign(preds[1:] - preds[:-1])
    dir_act  = np.sign(actuals[1:] - actuals[:-1])
    dir_acc  = (dir_pred == dir_act).mean() * 100 if len(dir_pred)>0 else np.nan
    return {"RMSE":rmse, "MAE":mae, "MAPE":mape, "DirAcc":dir_acc}

def update_history_with_actuals(ticker, price_df):
    if not os.path.exists(HISTORY_CSV): return
    hist = pd.read_csv(HISTORY_CSV, parse_dates=["ForecastDate"])
    mask = hist.Ticker == ticker
    to_fill = hist[mask & hist['Actual_1'].isna()]
    updates = False
    price_df['DateOnly'] = price_df['Date'].dt.tz_localize(None).dt.date
    for idx, row in to_fill.iterrows():
        fc_date = row.ForecastDate
        preds = [row[f'Pred_{i}'] for i in range(1, FORECAST_HORIZON+1)]
        actuals = []
        for i in range(1, FORECAST_HORIZON+1):
            d = (fc_date + timedelta(days=i)).date()
            match = price_df.loc[price_df['DateOnly'] == d, 'AdjClose']
            actuals.append(match.iloc[0] if not match.empty else np.nan)
        if all(~np.isnan(actuals)):
            for i, a in enumerate(actuals, start=1):
                hist.at[idx, f'Actual_{i}'] = a
            for k, v in compute_metrics(preds, actuals).items():
                hist.at[idx, k] = v
            updates = True
    if updates:
        hist.to_csv(HISTORY_CSV, index=False)
        logging.info(f"Updated history with actuals for {ticker}")

def log_new_forecast(ticker, forecast_date, preds):
    os.makedirs(LOG_DIR, exist_ok=True)
    row = {"ForecastDate": forecast_date, "Ticker": ticker}
    for i, p in enumerate(preds, start=1):
        row[f"Pred_{i}"] = p
        row[f"Actual_{i}"] = np.nan
    row.update({k: np.nan for k in ("RMSE", "MAE", "MAPE", "DirAcc")})
    df_row = pd.DataFrame([row])
    header = not os.path.exists(HISTORY_CSV)
    df_row.to_csv(HISTORY_CSV, mode='a', header=header, index=False)
    logging.info(f"Logged new forecast for {ticker} on {forecast_date.date()}")

def train_and_forecast(ticker, current_date):
    df = prepare_features(ticker)
    window_start = current_date - timedelta(days=TRAIN_WINDOW_DAYS)
    df_train = df[(df['Date'] < current_date) & (df['Date'] >= window_start)].copy()
    assert df_train.shape[0] >= SENTIMENT_WINDOW_DAYS, f"Not enough data for {ticker}"
    feature_cols = [c for c in df_train.columns if c not in ('Date','AdjClose','Return')]
    X = df_train[feature_cols]
    y = df_train['AdjClose'].shift(-FORECAST_HORIZON).dropna()
    X = X.iloc[:-FORECAST_HORIZON]
    if LGB_AVAILABLE:
        model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200)
    else:
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31)
    model.fit(X, y)
    X_fc = df.tail(FORECAST_HORIZON)[feature_cols]
    preds = model.predict(X_fc)
    return df, model, preds

def plot_forecast(ticker, df_hist, preds, current_date, model=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df_hist['Date'], df_hist['AdjClose'], label='Historical')
    future_dates = [current_date + timedelta(days=i) for i in range(1, FORECAST_HORIZON+1)]
    ax.plot(future_dates, preds, '--', color='orange', label='Forecast')
    if model is not None:
        feature_cols = [c for c in df_hist.columns if c not in ('Date','AdjClose','Return')]
        resids = model.predict(df_hist[feature_cols]) - df_hist['AdjClose']
        std = np.std(resids)
        lower, upper = preds - 1.96*std, preds + 1.96*std
        ax.fill_between(future_dates, lower, upper, color='orange', alpha=0.3)
    ax.set_title(f"{ticker} Price & {FORECAST_HORIZON}-Day Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    ax.legend()
    out_path = os.path.join(PLOTS_DIR, f"{ticker}_{current_date.strftime('%Y%m%d')}.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved plot to {out_path}")
