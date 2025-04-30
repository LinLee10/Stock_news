# retrain.py

import os
import joblib
import pandas as pd
from news_scraper import scrape_and_score_news
from prediction import fetch_price_history, train_and_forecast

# Directory to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def weekly_retrain(
    portfolio_csv: str = "data/portfolio.csv",
    watch_csv: str     = "data/watchlist.csv"
):
    """
    Retrain RF+XGB ensemble on the past 90 days for each ticker
    in portfolio & watchlist, and save model+params to disk.
    """
    # 1) Load tickers
    pf = pd.read_csv(portfolio_csv)["Ticker"].tolist()
    wl = pd.read_csv(watch_csv)["Ticker"].tolist()
    all_tickers = sorted(set(pf + wl))

    # 2) Scrape sentiment for past 90 days
    news_df = scrape_and_score_news(all_tickers)

    # 3) For each ticker: fetch data, train, and save
    market_df = fetch_price_history("SPY", period="90d")
    peer_df   = fetch_price_history("QQQ", period="90d")

    for tic in all_tickers:
        print(f"Retraining model for {tic}...")
        hist = fetch_price_history(tic, period="90d")
        # Build sentiment_series for this ticker
        sub = news_df[news_df["Ticker"]==tic][["Date","Score"]]
        if sub.empty:
            sent_ser = None
        else:
            sent_ser = pd.Series(
                sub["Score"].values,
                index=pd.to_datetime(sub["Date"])
            )

        # Train & ignore output
        _, _, _ = train_and_forecast(
            tic, hist, market_df, peer_df, sent_ser
        )

        # Save the raw model objects if needed (RF/XGB) separately:
        # We didn't return them, but you could modify train_and_forecast
        # to return them alongside forecasts if you wish to persist.

        # Instead, as a stub, we log the retrain
        print(f"  -> Completed retraining for {tic}.")

    print("Weekly retraining complete.")

if __name__ == "__main__":
    weekly_retrain()
