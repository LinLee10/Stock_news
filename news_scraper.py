# news_scraper.py

import os
import time
import logging
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
from collections import defaultdict
from rapidfuzz import fuzz
from transformers import pipeline
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─── Load Alpha Vantage key ───────────────────────────────────────────────────
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
if not API_KEY:
    logger.error("Missing ALPHA_VANTAGE_KEY in config/secrets.env")
    raise RuntimeError("Set ALPHA_VANTAGE_KEY in config/secrets.env")

# ─── Mapping tickers to canonical company names ─────────────────────────────
TICKER_COMPANY = {
    "TSLA": "Tesla, Inc.",
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "NVDA": "NVIDIA Corporation",
    "RTX": "Raytheon Technologies Corporation",
    "UUUU": "Energy Fuels Inc.",
    "PFE": "Pfizer Inc.",
    "SRAD": "Sportradar Group AG",
    "MRVL": "Marvell Technology, Inc.",
    "ADI": "Analog Devices, Inc.",
    "LLY": "Eli Lilly and Company",
    # … add more tickers and company names as needed
}

# ─── FinBERT sentiment pipeline ───────────────────────────────────────────────
_sentiment_pipe = None
def get_sentiment_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
    return _sentiment_pipe

def fetch_google_rss(query: str, retries=3) -> str:
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}"
    for _ in range(retries):
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.text
        time.sleep(1)
    logger.warning(f"Failed to fetch Google RSS for {query}")
    return ""

def parse_rss(xml_text: str):
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        link  = item.findtext("link", "")
        pub   = item.findtext("pubDate", "")
        try:
            pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            pub_dt = datetime.utcnow()
        yield title, link, pub_dt

def fetch_av_news(ticker: str, retries=3):
    """
    Pulls Alpha Vantage News Sentiment feed for a ticker.
    Returns list of (title, url, date) tuples.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "apikey":    API_KEY
    }
    for _ in range(retries):
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json().get("feed", [])
            out = []
            for item in data:
                title = item.get("title", "")
                link  = item.get("url", "")
                ts    = item.get("time_published", "")
                try:
                    pub_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    pub_dt = datetime.utcnow()
                out.append((title, link, pub_dt))
            return out
        time.sleep(1)
    logger.warning(f"Failed to fetch AV news for {ticker}")
    return []

def smart_match(name: str, headline: str, thresh=75) -> bool:
    hl = headline.lower()
    nm = name.lower()
    if nm in hl:
        return True
    return fuzz.partial_ratio(nm, hl) >= thresh

def scrape_headlines(tickers: list[str], days: int = 10) -> dict[str, dict]:
    """
    Returns for each ticker over the past `days` calendar days:
      {
        'headlines':       [(title, link, date), ...],
        'count':           total_unique_matches,
        'daily_sentiment': {date: avg_score, ...},
        'count_positive':  number_of_positive,
        'count_negative':  number_of_negative,
        'count_neutral':   number_of_neutral
      }
    Combines Google RSS + Alpha Vantage feeds, dedupes overlaps by link.
    """
    today = datetime.utcnow().date()
    window_start = today - timedelta(days=days - 1)
    results = {}
    sent_pipe = get_sentiment_pipeline()

    for tic in tickers:
        comp = TICKER_COMPANY.get(tic, tic)
        # 1) fetch both sources
        google_xml = fetch_google_rss(comp)
        goog_raw   = list(parse_rss(google_xml))
        av_raw     = fetch_av_news(tic)

        # 2) filter by date window and smart match
        combined = []
        for title, link, pub_dt in goog_raw + av_raw:
            d = pub_dt.date()
            if window_start <= d <= today and smart_match(comp, title):
                combined.append((title, link, d))

        # 3) dedupe by link
        seen = set()
        matches = []
        for title, link, d in combined:
            if link and link not in seen:
                seen.add(link)
                matches.append((title, link, d))

        # 4) sentiment & counts (lowercase matching)
        daily = defaultdict(list)
        count_pos = count_neg = count_neu = 0
        if matches:
            texts = [t for t, _, _ in matches]
            batch = sent_pipe(texts, batch_size=16)
            scores = []
            for out in batch:
                lbl = out["label"].lower()
                val = out["score"]
                if lbl == "positive":
                    scores.append(val)
                    count_pos += 1
                elif lbl == "negative":
                    scores.append(-val)
                    count_neg += 1
                else:
                    scores.append(0.0)
                    count_neu += 1

            for (_, _, d), s in zip(matches, scores):
                daily[d].append(s)

        daily_mean = {d: sum(v)/len(v) for d, v in daily.items()} if daily else {}
        results[tic] = {
            "headlines":       matches,
            "count":           len(matches),
            "daily_sentiment": daily_mean,
            "count_positive":  count_pos,
            "count_negative":  count_neg,
            "count_neutral":   count_neu,
        }
        logger.info(f"[{tic}] {len(matches)} unique headlines from {window_start} → {today}")

    return results
