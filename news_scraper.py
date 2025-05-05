import os
import logging
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import requests
import feedparser
from dotenv import load_dotenv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter, Retry

# ─── Load environment variables ─────────────────────────────────────────────
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHA_VANTAGE_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    raise RuntimeError("ALPHA_VANTAGE_KEY or ALPHAVANTAGE_API_KEY must be set in config/secrets.env")

# ─── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger("news_scraper")

# ─── Sentiment pipeline (FinBERT) with fallback to VADER ────────────────────
try:
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )
except Exception as e:
    logger.warning(f"[FinBERT] load failed, will fallback to VADER: {e}")
    finbert = None

# ─── HTTP session with retries & headers ───────────────────────────────────
SESSION = requests.Session()
RETRY_STRAT = Retry(
    total=3,
    connect=0,           # no retries on connection errors (DNS, TCP)
    read=3,              # you can still retry if the server hangs mid-download
    backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"]
)

SESSION.mount("https://", HTTPAdapter(max_retries=RETRY_STRAT))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15"
    )
}

# ─── RSS scraping helpers ──────────────────────────────────────────────────
def scrape_generic_rss(ticker: str, url: str) -> List[Tuple[datetime.date, str, str]]:
    """
    Fetch and parse RSS. Connect timeout 3s, read timeout 5s.
    On any error or non-200, return empty list.
    """
    try:
        resp = SESSION.get(url, timeout=(3, 5), headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"[RSS] {ticker}: cannot fetch {url}: {e}")
        return []

    feed = feedparser.parse(resp.content)
    items: List[Tuple[datetime.date, str, str]] = []
    for entry in feed.entries:
        pub_date = datetime.utcnow().date()
        if getattr(entry, "published_parsed", None):
            try:
                pub_date = datetime(*entry.published_parsed[:6]).date()
            except Exception:
                pass
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        items.append((pub_date, title, link))
    return items

def scrape_google_rss(ticker: str) -> List[Tuple[datetime.date, str, str]]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    return scrape_generic_rss(ticker, url)

def fetch_av_news(ticker: str) -> List[Tuple[datetime.date, str, str]]:
    """
    Fetch latest news via Alpha Vantage NEWS_SENTIMENT endpoint.
    Returns list of (Date, Title, URL).
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": API_KEY
    }
    try:
        resp = SESSION.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=(3, 5),
            headers=HEADERS
        )
        resp.raise_for_status()
        feed = resp.json().get("feed", [])
    except Exception as e:
        logger.warning(f"[AV News] {ticker}: failed to fetch: {e}")
        return []

    items: List[Tuple[datetime.date, str, str]] = []
    for art in feed:
        try:
            dt = datetime.fromisoformat(art.get("time_published", "")[:10]).date()
        except Exception:
            dt = datetime.utcnow().date()
        title = art.get("title", "").strip()
        url = art.get("url", "").strip()
        if title:
            items.append((dt, title, url))
    return items

def scrape_benzinga_rss(ticker: str) -> List[Tuple[datetime.date, str, str]]:
    url = f"https://www.benzinga.com/rss/{ticker.lower()}.xml"
    return scrape_generic_rss(ticker, url)

# ─── Top-10 Investor RSS feeds ─────────────────────────────────────────────
INVESTOR_FEEDS = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.fool.com/feeds/stock-news-analysis.aspx",
    "https://seekingalpha.com/symbol/{ticker}.xml",
    "https://www.investors.com/rss.ashx",
    "https://www.barrons.com/xml/rss/3_7031.xml",
]

# ─── Aggregation & deduplication ──────────────────────────────────────────
def get_news_headlines(ticker: str) -> List[Tuple[datetime.date, str, str]]:
    """
    Aggregate headlines from:
      • Google RSS
      • Alpha Vantage NEWS_SENTIMENT
      • Benzinga RSS
      • Top investor RSS feeds
    Deduplicate by title and return list of (Date, Headline, URL).
    """
    raw: List[Tuple[datetime.date, str, str]] = []

    # core sources
    for fn in (scrape_google_rss, lambda t: fetch_av_news(t), scrape_benzinga_rss):
        try:
            raw += fn(ticker)
        except Exception as e:
            logger.warning(f"[get_news_headlines] {ticker}: {fn.__name__} failed: {e}")

    # investor feeds
    for feed in INVESTOR_FEEDS:
        url = feed.format(ticker=ticker)
        raw += scrape_generic_rss(ticker, url)

    # dedupe by title (case-insensitive)
    seen = set()
    deduped: List[Tuple[datetime.date, str, str]] = []
    for date, title, link in raw:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            deduped.append((date, title, link))

    return deduped

# ─── Sentiment analysis ────────────────────────────────────────────────────
def analyze_headlines_sentiment(rows: List[Tuple[datetime.date, str]]) -> pd.DataFrame:
    """
    Score headlines with FinBERT if available, else fallback to VADER.
    Returns DataFrame: Date, Headline, Label, Score.
    """
    if not rows:
        return pd.DataFrame(columns=["Date", "Headline", "Label", "Score"])

    dates, texts = zip(*rows)
    records = []

    # try FinBERT
    if finbert:
        try:
            results = finbert(list(texts), truncation=True, padding=False)
            for dt, txt, res in zip(dates, texts, results):
                label = res["label"].lower()
                score = float(res["score"])
                records.append({"Date": dt, "Headline": txt, "Label": label, "Score": score})
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning(f"[FinBERT] failed, falling back to VADER: {e}")

    # fallback VADER
    analyzer = SentimentIntensityAnalyzer()
    for dt, txt in zip(dates, texts):
        vs = analyzer.polarity_scores(txt)
        compound = vs["compound"]
        if compound > 0.05:
            label = "positive"
        elif compound < -0.05:
            label = "negative"
        else:
            label = "neutral"
        records.append({"Date": dt, "Headline": txt, "Label": label, "Score": float(compound)})

    return pd.DataFrame(records)

# ─── Main scrape & score ───────────────────────────────────────────────────
def scrape_and_score_news(tickers: List[str]) -> pd.DataFrame:
    """
    For each ticker:
      1) scrape headlines with URLs
      2) analyze sentiment
      3) tag with Ticker column
    Returns DataFrame: Date, Ticker, Headline, Label, Score, URL.
    """
    all_dfs = []
    for tic in tickers:
        rows = get_news_headlines(tic)
        if not rows:
            continue
        pairs = [(dt, h) for dt, h, _ in rows]
        df = analyze_headlines_sentiment(pairs)
        df["Ticker"] = tic
        df["URL"] = [link for _, _, link in rows]
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["Date", "Ticker", "Headline", "Label", "Score", "URL"])
    return pd.concat(all_dfs, ignore_index=True)
