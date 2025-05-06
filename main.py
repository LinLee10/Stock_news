import logging
import pandas as pd

from news_scraper import scrape_headlines
from prediction import train_predict_stock
from charts import create_collage
from email_report import send_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tickers(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError(f"{path} must contain a 'Ticker' column")
    return df["Ticker"].astype(str).tolist()

WATCHLIST = load_tickers("data/watchlist.csv")
PORTFOLIO = load_tickers("data/portfolio.csv")

def main():
    # 1. Scrape headlines & sentiment (7-day window)
    all_tickers = WATCHLIST + PORTFOLIO
    head_info   = scrape_headlines(all_tickers)

    # 2. Forecast & gather price history
    preds = {}
    price_data, forecast_data, sentiment_map = {}, {}, {}
    for t in all_tickers:
        daily_sent  = head_info[t]["daily_sentiment"]
        sent_series = pd.Series(daily_sent)
        out = train_predict_stock(t, sent_series)

        preds[t] = {
            "predictions": out["predictions"],
            "confidence":  out["confidence"],
            "red_flag":    out["red_flag"]
        }
        price_data[t]    = out["history"]
        forecast_data[t] = pd.DataFrame({
            "Date":           [d.strftime("%Y-%m-%d") for d in out["dates"]],
            "Forecast_Close": out["predictions"],
        })
        sentiment_map[t] = daily_sent

    # 3. Create and save collages
    collage_path = "portfolio_collage.png"
    create_collage(
        PORTFOLIO,
        price_data,
        forecast_data,
        sentiment_map,
        "Portfolio Forecasts",
        collage_path
    )
    logger.info(f"Saved collage to {collage_path}")

    watch_collage = "watchlist_collage.png"
    create_collage(
        WATCHLIST,
        price_data,
        forecast_data,
        sentiment_map,
        "Watchlist Forecasts",
        watch_collage
    )
    logger.info(f"Saved collage to {watch_collage}")

    # 4. Generate & send report (HTML output)
    send_report(
        WATCHLIST,
        PORTFOLIO,
        head_info,
        preds,
        collage_path=collage_path,
        out_path="report.html"
    )

if __name__ == "__main__":
    main()
