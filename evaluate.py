import os
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prediction import train_predict_stock

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def evaluate_3day_forecast(tickers: list[str],
                           price_fetcher,
                           sentiment_fetcher,
                           backtest_days: int = 90):
    """
    Perform a rolling backtest over the last `backtest_days` for each ticker:
      - At each step t, train on data up to (today - 3 days),
      - Forecast 3 days ahead,
      - Compare predicted vs actual closes for those 3 days.
    Compute MAE and RMSE across all tickers/dates, and track performance by confidence bucket.
    """
    results = []
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=backtest_days + 10)  # extra for warm-up

    for tic in tickers:
        try:
            # Fetch full historical up to today
            full_price = price_fetcher(tic)
            full_price['Date'] = pd.to_datetime(full_price['Date'])
            mask = (full_price['Date'] >= start_date) & (full_price['Date'] <= end_date)
            hist = full_price.loc[mask].reset_index(drop=True)
            if len(hist) < 15:
                logger.warning(f"{tic}: not enough history for backtest ({len(hist)} days)")
                continue

            # Rolling window
            for cutoff in hist['Date'].iloc[:-3]:
                # Prepare training slice: all dates <= cutoff
                train_hist = hist[hist['Date'] <= cutoff]
                # Fetch 30-day sentiment up to cutoff
                sent_series = sentiment_fetcher(tic, days=90)
                sent_series = pd.Series(sent_series).loc[:cutoff]

                # Forecast 3 days
                out = train_predict_stock(tic, sent_series)
                preds = out['predictions']  # list of 3 floats
                conf = out['confidence']
                red = out['red_flag']

                # Actual next 3-day closes
                future = hist[hist['Date'] > cutoff].head(3)['Close'].tolist()
                if len(future) < 3:
                    break  # not enough future data to evaluate

                # Error metrics
                mae = mean_absolute_error(future, preds)
                rmse = np.sqrt(mean_squared_error(future, preds))

                results.append({
                    'Ticker': tic,
                    'Cutoff': cutoff,
                    'Confidence': conf,
                    'RedFlag': red,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Error1': abs(future[0] - preds[0]),
                    'Error3': abs(future[-1] - preds[-1])
                })
        except Exception as e:
            logger.error(f"Error evaluating {tic}: {e}")

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("No backtest results generated.")
        return df

    # Aggregate overall
    overall_mae = df['MAE'].mean()
    overall_rmse = df['RMSE'].mean()
    logger.info(f"Overall 3-day MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}")

    # Performance by confidence decile
    df['ConfDecile'] = pd.qcut(df['Confidence'], 10, labels=False, duplicates='drop')
    perf_by_conf = df.groupby('ConfDecile')[['MAE','RMSE']].mean().reset_index()
    logger.info("Performance by Confidence Decile:\n" + perf_by_conf.to_string(index=False))

    return df

if __name__ == "__main__":
    import yaml
    # Example usage: load tickers from config
    with open('data/watchlist.csv') as f:
        tickers = pd.read_csv(f)['Ticker'].dropna().tolist()
    df_results = evaluate_3day_forecast(tickers, 
                                       price_fetcher=lambda t: __import__('prediction').fetch_price_history(t),
                                       sentiment_fetcher=lambda t, days: __import__('news_scraper').scrape_headlines([t], days)[t]['daily_sentiment'])
    df_results.to_csv('backtest_results.csv', index=False)
    logging.info("Saved backtest_results.csv")
