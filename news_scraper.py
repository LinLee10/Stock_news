# news_scraper.py

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ——— Sentiment pipelines ———
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    top_k = 1
)
vader = SentimentIntensityAnalyzer()

# ——— Helper: parse RSS feed at `url` for headlines containing `ticker` ———
def scrape_generic_rss(ticker: str, url: str) -> list[tuple[datetime.date, str]]:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            if ticker.upper() not in title.upper():
                continue
            pub = item.findtext("pubDate")
            date = datetime.utcnow().date()
            if pub:
                try:
                    date = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").date()
                except:
                    pass
            items.append((date, title))
        return items
    except Exception:
        return []

# ——— Original helpers ———
def scrape_google_rss(ticker: str) -> list[tuple[datetime.date, str]]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    return scrape_generic_rss(ticker, url)

def scrape_yahoo_news(ticker: str) -> list[tuple[datetime.date, str]]:
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        items = []
        for art in news:
            title = art.get("title", "").strip()
            if not title or ticker.upper() not in title.upper():
                continue
            epoch = art.get("providerPublishTime")
            date = datetime.utcnow().date()
            if epoch:
                try:
                    date = datetime.utcfromtimestamp(epoch).date()
                except:
                    pass
            items.append((date, title))
        return items
    except Exception:
        return []

def scrape_benzinga_rss(ticker: str) -> list[tuple[datetime.date, str]]:
    url = f"https://www.benzinga.com/rss/{ticker.lower()}.xml"
    return scrape_generic_rss(ticker, url)

# ——— Top-10 Investor RSS feeds ———
INVESTOR_FEEDS = [
    "https://feeds.bloomberg.com/markets/news.rss",                   # Bloomberg
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",                 # WSJ Markets
    "https://feeds.reuters.com/reuters/businessNews",                # Reuters
    "https://www.cnbc.com/id/100003114/device/rss/rss.xml",          # CNBC
    "https://feeds.marketwatch.com/marketwatch/topstories/",         # MarketWatch
    "https://www.fool.com/feeds/stock-news-analysis.aspx",           # Motley Fool
    "https://seekingalpha.com/symbol/{ticker}.xml",                  # Seeking Alpha
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",   # Yahoo Finance RSS
    "https://www.investors.com/rss.ashx",                            # Investor's Business Daily
    "https://www.barrons.com/xml/rss/3_7031.xml",                    # Barron's
]

def get_news_headlines(ticker: str) -> list[tuple[datetime.date, str]]:
    """
    Aggregate headlines from:
      • Google RSS
      • Yahoo Finance (yfinance)
      • Benzinga RSS
      • Top-10 investor feeds (filtered by ticker)
    Returns list of (date, title), deduped by title.
    """
    raw = []
    raw += scrape_google_rss(ticker)
    raw += scrape_yahoo_news(ticker)
    raw += scrape_benzinga_rss(ticker)
    for feed in INVESTOR_FEEDS:
        url = feed.format(ticker=ticker)
        raw += scrape_generic_rss(ticker, url)

    # dedupe by title string
    seen = set()
    out = []
    for date, title in raw:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            out.append((date, title))
    return out

def analyze_headlines_sentiment(
    rows: list[tuple[datetime.date, str]]
) -> pd.DataFrame:
    """
    Batch-score headlines with FinBERT, fallback to VADER on error.
    Returns DataFrame: [Date, Headline, Label, Score].
    """
    if not rows:
        return pd.DataFrame(columns=["Date","Headline","Label","Score"])

    dates, texts = zip(*rows)
    try:
        results = finbert(list(texts), truncation=True, padding=False)
        recs = []
        for dt, txt, res in zip(dates, texts, results):
            lbl = res["label"].lower()
            scr = res["score"]
            val = scr if lbl=="positive" else -scr if lbl=="negative" else 0.0
            recs.append({
                "Date":     dt,
                "Headline": txt,
                "Label":    lbl,
                "Score":    val
            })
        return pd.DataFrame(recs)
    except Exception:
        recs = []
        for dt, txt in rows:
            vs = vader.polarity_scores(txt)
            comp = vs["compound"]
            lbl = "positive" if comp>0.05 else "negative" if comp< -0.05 else "neutral"
            recs.append({
                "Date":     dt,
                "Headline": txt,
                "Label":    lbl,
                "Score":    float(comp)
            })
        return pd.DataFrame(recs)

def scrape_and_score_news(tickers: list[str]) -> pd.DataFrame:
    """
    Main entrypoint. For each ticker:
      • scrape headlines
      • analyze sentiment
      • tag Ticker column
    Returns DataFrame [Date,Ticker,Headline,Label,Score].
    """
    all_dfs = []
    for tic in tickers:
        rows = get_news_headlines(tic)
        if not rows:
            continue
        df = analyze_headlines_sentiment(rows)
        df["Ticker"] = tic
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["Date","Ticker","Headline","Label","Score"])
    return pd.concat(all_dfs, ignore_index=True)
