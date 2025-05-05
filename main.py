# main.py

import pandas as pd
from prediction import train_predict_stock, log_forecast_entry
from forecast_utils import plot_forecast
from charts import create_collage
from email_report import send_report
from news_scraper import scrape_and_score_news

def generate_report():
    # ─── Load tickers from CSV ───────────────────────────────────────
    portfolio = pd.read_csv("data/portfolio.csv")["Ticker"].tolist()
    watchlist = pd.read_csv("data/watchlist.csv")["Ticker"].tolist()

    # Preload price data (incl. S&P as market)
    from prediction import load_bulk_price_data
    all_tickers = list(set(portfolio + watchlist + ["^GSPC"]))
    load_bulk_price_data(all_tickers, period="90d")

    # ─── Scrape & prepare news DataFrame ────────────────────────────
    news_df = scrape_and_score_news(portfolio + watchlist)
    news_df["Date"] = pd.to_datetime(news_df["Date"])
    news_df.to_csv("sentiment_history.csv", index=False)

    # ─── Build 7-day & 30-day sentiment tables ───────────────────────
    cutoff7  = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
    cutoff30 = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    df7  = news_df[news_df["Date"] >= cutoff7]
    df30 = news_df[news_df["Date"] >= cutoff30]

    def summarize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tic in df["Ticker"].unique():
            sub = df[df["Ticker"] == tic]
            avg = round(sub["Score"].mean(), 2)
            pos = (sub["Score"] > 0.05).sum()
            neg = (sub["Score"] < -0.05).sum()
            neu = len(sub) - pos - neg
            rows.append([tic, avg, pos, neg, neu, len(sub)])
        return pd.DataFrame(rows, columns=[
            "Ticker","Avg_Sentiment","Count_Positive",
            "Count_Negative","Count_Neutral","Total_Headlines"
        ])

    summary7  = (summarize_sentiment(df7)
                 .sort_values("Total_Headlines", ascending=False)
                 .head(10))
    summary30 = summarize_sentiment(df30)
    summary7.to_csv("daily_mentions.csv", index=False)

    # ─── Top-10 Headlines for 7-Day Leaders ────────────────────────
    top10 = summary7["Ticker"].tolist()
    top_headlines: dict[str, list[tuple[str,str]]] = {}

    for tic in top10:
        sub = (news_df[news_df["Ticker"] == tic]
               .sort_values("Date", ascending=False))

        # Pick out headline/text column and URL column
        headline_col = next(
            (c for c in sub.columns
             if "headline" in c.lower() or "title" in c.lower()),
            None
        )
        link_col = next(
            (c for c in sub.columns
             if "url" in c.lower() or "link" in c.lower()),
            None
        )

        items: list[tuple[str,str]] = []
        if headline_col and link_col:
            for _, row in sub.iterrows():
                if len(items) >= 3:
                    break
                h = row[headline_col]
                u = row[link_col]
                if pd.notna(h) and pd.notna(u):
                    items.append((h, u))

        top_headlines[tic] = items

    # ─── Portfolio Loop ─────────────────────────────────────────────
    portfolio_results = []
    portfolio_images  = []

    for ticker in portfolio:
        sentiment_series = df30[df30["Ticker"] == ticker].set_index("Date")["Score"]
        result = train_predict_stock(ticker, sentiment_series)
        result["ticker"] = ticker
        portfolio_results.append(result)

        # Log forecasts
        pred_date = pd.Timestamp.now().normalize()
        for date, pred in zip(result["dates"], result["predictions"]):
            log_forecast_entry(
                pred_date       = pred_date,
                target_date     = date,
                ticker          = ticker,
                predicted_price = pred
            )

        # Plot if available
        if result.get("dates") and result.get("history") and result.get("predictions"):
            history_df   = result["history"]
            current_date = pd.to_datetime(history_df["Date"]).max()
            img = plot_forecast(
                ticker,
                history_df,
                result["predictions"],
                current_date
            )
            portfolio_images.append(img)

    # ─── Watchlist Loop ────────────────────────────────────────────
    watchlist_results = []
    watchlist_images  = []

    for ticker in watchlist:
        sentiment_series = df30[df30["Ticker"] == ticker].set_index("Date")["Score"]
        result = train_predict_stock(ticker, sentiment_series)
        result["ticker"] = ticker
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
            img = plot_forecast(
                ticker,
                history_df,
                result["predictions"],
                current_date
            )
            watchlist_images.append(img)

    # ─── Build Collages ────────────────────────────────────────────
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

    # ─── Send the email ─────────────────────────────────────────────
    attachments = [collage_portfolio, collage_watchlist]
    send_report(
        portfolio_results,
        watchlist_results,
        summary7,
        summary30,
        attachments,
        top_headlines
    )

if __name__ == "__main__":
    generate_report()
