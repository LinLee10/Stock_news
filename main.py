# main.py

import logging
from datetime import datetime
import pytz

from forecast_utils import (
    setup_logging,
    load_price_data,
    update_history_with_actuals,
    train_and_forecast,
    log_new_forecast,
    plot_forecast
)

def main():
    setup_logging()
    logging.info("=== Starting daily forecast run ===")
    tz = pytz.timezone('US/Eastern')
    current_date = datetime.now(tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    tickers = ["NVMI", "RTX", "UUUU", "PFE", "SRAD", "MRVL", "ADI", "LLY"]

    for ticker in tickers:
        try:
            price_df = load_price_data(ticker)
            update_history_with_actuals(ticker, price_df)

            df_hist, model, preds = train_and_forecast(ticker, current_date)

            log_new_forecast(ticker, current_date, preds)
            plot_forecast(ticker, df_hist, preds, current_date, model=model)

        except AssertionError as ae:
            logging.error(f"{ticker}: {ae}")
        except Exception as e:
            logging.exception(f"Unexpected error for {ticker}: {e}")

    logging.info("=== Forecast run complete ===")

if __name__ == "__main__":
    main()
