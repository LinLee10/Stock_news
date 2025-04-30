# main.py

import pandas as pd
from prediction import train_predict_stock, log_forecast_entry
from forecast_utils import plot_forecast
from charts import create_collage
from email_report import send_report
from news_scraper import scrape_and_score_news

def generate_report():
    # Your tickers
    portfolio = ['AAPL', 'MSFT', 'NVDA', 'PFE', 'SRAD', 'MRVL', 'ADI', 'LLY', 'RTX', 'UUUU']
    watchlist = ['GOOGL', 'TSLA', 'PLTR']

    from prediction import load_bulk_price_data
    all_tickers = list(set(portfolio + watchlist + ["^GSPC"]))
    load_bulk_price_data(all_tickers, period="90d")

    # Scrape & save sentiment history
    all_tickers = list(set(portfolio + watchlist))
    news_df = scrape_and_score_news(all_tickers)
    news_df["Date"] = pd.to_datetime(news_df["Date"])
    news_df.to_csv("sentiment_history.csv", index=False)

    # Cutoffs for 30-day / 7-day summaries
    cutoff30 = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    cutoff7  = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
    df30 = news_df[news_df["Date"] >= cutoff30]
    df7  = news_df[news_df["Date"] >= cutoff7]

    # Helper to build sentiment tables
    def summarize_sentiment(df):
        summary = []
        for tic in df["Ticker"].unique():
            sub = df[df["Ticker"] == tic]
            avg = round(sub["Score"].mean(), 2)
            pos = (sub["Score"] > 0.05).sum()
            neg = (sub["Score"] < -0.05).sum()
            neu = len(sub) - pos - neg
            summary.append([tic, avg, pos, neg, neu, len(sub)])
        return pd.DataFrame(
            summary,
            columns=[
                "Ticker",
                "Avg_Sentiment",
                "Count_Positive",
                "Count_Negative",
                "Count_Neutral",
                "Total_Headlines",
            ],
        )

    summary30 = summarize_sentiment(df30)
    summary7  = summarize_sentiment(df7) \
                   .sort_values("Total_Headlines", ascending=False) \
                   .head(10)
    summary7.to_csv("daily_mentions.csv", index=False)

    # ─── Portfolio Loop ────────────────────────────────────────────────
    portfolio_results = []
    portfolio_images  = []

    for ticker in portfolio:
        sentiment_series = df30[df30["Ticker"] == ticker] \
                              .set_index("Date")["Score"]
        result = train_predict_stock(ticker, sentiment_series)
        portfolio_results.append(result)

        pred_date = pd.Timestamp.now().normalize()
        for date, pred in zip(result["dates"], result["predictions"]):
            log_forecast_entry(
                pred_date       = pred_date,
                target_date     = date,
                ticker          = ticker,
                predicted_price = pred
            )

        if result.get("dates") and result.get("history") and result.get("predictions"):
            history_df   = result["history"]
            current_date = pd.to_datetime(history_df["Date"]).max()
            img_path = plot_forecast(
                ticker,
                history_df,
                result["predictions"],
                current_date
            )
            portfolio_images.append(img_path)

    # ─── Watchlist Loop ───────────────────────────────────────────────
    watchlist_results = []
    watchlist_images  = []

    for ticker in watchlist:
        sentiment_series = df30[df30["Ticker"] == ticker] \
                              .set_index("Date")["Score"]
        result = train_predict_stock(ticker, sentiment_series)
        watchlist_results.append(result)

        pred_date = pd.Timestamp.now().normalize()
        for date, pred in zip(result["dates"], result["predictions"]):
            log_forecast_entry(
                pred_date       = pred_date,
                target_date     = date,
                ticker          = ticker,
                predicted_price = pred
            )

        if result.get("dates") and result.get("history") and result.get("predictions"):
            history_df   = result["history"]
            current_date = pd.to_datetime(history_df["Date"]).max()
            img_path = plot_forecast(
                ticker,
                history_df,
                result["predictions"],
                current_date
            )
            watchlist_images.append(img_path)

    # ─── Build Collages ────────────────────────────────────────────────
    price_data_p = {r["ticker"]: r["history"] for r in portfolio_results}
    forecast_data_p = {
        r["ticker"]: pd.DataFrame({
            "Date": r["dates"],
            "Forecast_Close": r["predictions"]
        })
        for r in portfolio_results
    }
    sentiment_map_p = {
        tic: df30[df30["Ticker"] == tic].set_index("Date")["Score"].tolist()
        for tic in portfolio
    }
    collage_portfolio = create_collage(
        portfolio,
        price_data_p,
        forecast_data_p,
        sentiment_map_p,
        "Portfolio Forecasts",
        "portfolio_collage.png"
    )

    price_data_w = {r["ticker"]: r["history"] for r in watchlist_results}
    forecast_data_w = {
        r["ticker"]: pd.DataFrame({
            "Date": r["dates"],
            "Forecast_Close": r["predictions"]
        })
        for r in watchlist_results
    }
    sentiment_map_w = {
        tic: df30[df30["Ticker"] == tic].set_index("Date")["Score"].tolist()
        for tic in watchlist
    }
    collage_watchlist = create_collage(
        watchlist,
        price_data_w,
        forecast_data_w,
        sentiment_map_w,
        "Watchlist Forecasts",
        "watchlist_collage.png"
    )

    attachments = [collage_portfolio, collage_watchlist]
    send_report(
        portfolio_results,
        watchlist_results,
        summary7,
        summary30,
        attachments
    )

if __name__ == "__main__":
    generate_report()
