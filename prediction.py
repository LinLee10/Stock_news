import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def fetch_price_history(ticker: str, period: str = "90d") -> pd.DataFrame:
    """
    Fetch daily Close prices for `ticker` over `period`.
    Returns ['Date','Close'] as tz-naive DataFrame.
    """
    try:
        hist = yf.Ticker(ticker).history(
            period=period,
            interval="1d",
            auto_adjust=True
        )
    except Exception as ex:
        print(f"[prediction] yfinance error {ticker}: {ex}")
        return pd.DataFrame(columns=["Date", "Close"])

    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.DataFrame(columns=["Date", "Close"])

    df = hist.reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df[df["Close"].notnull()]


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
