# news_scraper.py

import os
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ——— RSS scraping helpers ———
def scrape_generic_rss(ticker: str, url: str) -> list[tuple[datetime.date, str, str]]:
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
            link = item.findtext("link", "").strip()
            items.append((date, title, link))
        return items
    except Exception:
        return []

def scrape_google_rss(ticker: str) -> list[tuple[datetime.date, str, str]]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    return scrape_generic_rss(ticker, url)

def scrape_yahoo_news(ticker: str) -> list[tuple[datetime.date, str, str]]:
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
            # Yahoo news may not provide link via API; skip or set empty
            items.append((date, title, ""))
        return items
    except Exception:
        return []

def scrape_benzinga_rss(ticker: str) -> list[tuple[datetime.date, str, str]]:
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

def get_news_headlines(ticker: str) -> list[tuple[datetime.date, str, str]]:
    """
    Aggregate headlines from:
      • Google RSS
      • Yahoo Finance (yfinance)
      • Benzinga RSS
      • Top investor RSS feeds (filtered by ticker)
    Returns list of (date, title, url), deduped by title.
    """
    raw = []
    raw += scrape_google_rss(ticker)
    raw += scrape_yahoo_news(ticker)
    raw += scrape_benzinga_rss(ticker)
    for feed in INVESTOR_FEEDS:
        url = feed.format(ticker=ticker)
        raw += scrape_generic_rss(ticker, url)

    # convert to triples with link
    raw2 = []
    for entry in raw:
        if len(entry) == 3:
            raw2.append(entry)
        else:
            d, h = entry
            raw2.append((d, h, ""))

    # dedupe by title string
    seen = set()
    out = []
    for date, title, link in raw2:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            out.append((date, title, link))
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
            comp = float(res["score"])
            recs.append({
                "Date":     dt,
                "Headline": txt,
                "Label":    lbl,
                "Score":    comp
            })
        return pd.DataFrame(recs)
    except Exception:
        analyzer = SentimentIntensityAnalyzer()
        recs = []
        for dt, txt in zip(dates, texts):
            vs = analyzer.polarity_scores(txt)
            comp = vs["compound"]
            if comp > 0.05:
                lbl = "positive"
            elif comp < -0.05:
                lbl = "negative"
            else:
                lbl = "neutral"
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
      • scrape headlines (with URLs)
      • analyze sentiment
      • tag Ticker column
    Returns DataFrame [Date,Ticker,Headline,Label,Score,URL].
    """
    all_dfs = []
    for tic in tickers:
        rows = get_news_headlines(tic)
        if not rows:
            continue
        # Prepare text-only for sentiment analysis
        pairs = [(d, h) for (d, h, lnk) in rows]
        df = analyze_headlines_sentiment(pairs)
        df["Ticker"] = tic
        df["URL"] = [lnk for (d, h, lnk) in rows]
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["Date","Ticker","Headline","Label","Score","URL"])
    return pd.concat(all_dfs, ignore_index=True)
