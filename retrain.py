#!/usr/bin/env python3
import os
import pandas as pd
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import schedule
import time

# ─── Logging Setup ─────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ─── Model Retraining Function ─────────────────────────
def retrain_models():
    """
    Retrain RF + XGB on each ticker’s full history and overwrite model files.
    """
    tickers = ['AAPL', 'MSFT']  # ← add all your tickers here
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'])
            df['dayofyear'] = df['Date'].dt.dayofyear

            X = df[['dayofyear']]
            y = df['Close']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            rf  = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            rf .fit(X_train, y_train)
            xgb.fit(X_train, y_train)

            joblib.dump(rf , os.path.join(model_dir, f"rf_{ticker}.pkl"))
            joblib.dump(xgb, os.path.join(model_dir, f"xgb_{ticker}.pkl"))

            logging.info(f"Retrained models for {ticker}")
        except Exception:
            logging.exception(f"Failed retraining for {ticker}")

    # Optional: after retrain, update any actual-price fillings
    try:
        from prediction import update_actual_prices
        update_actual_prices()
    except Exception:
        logging.exception("Failed to update actual prices after retraining")

# ─── Scheduler Setup ────────────────────────────────────
schedule.every().monday.at("00:00").do(retrain_models)
logging.info("Scheduled weekly retraining every Monday at 00:00")

if __name__ == "__main__":
    # Run once at startup
    retrain_models()
    # Then enter the loop
    while True:
        schedule.run_pending()
        time.sleep(60)
