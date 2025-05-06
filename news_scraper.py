import time
import logging
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
from collections import defaultdict
from rapidfuzz import fuzz
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─── Mapping tickers to canonical company names ─────────────────────────────
TICKER_COMPANY = {
    "TSLA": "Tesla, Inc.",
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "NVDA": "NVIDIA Corporation",
    # … add the rest of your tickers here …
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
    logger.warning(f"Failed to fetch RSS for {query}")
    return ""

def parse_rss(xml_text: str):
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item"):
        title = item.findtext("title", default="")
        link  = item.findtext("link",  default="")
        pub   = item.findtext("pubDate", default="")
        try:
            pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            pub_dt = datetime.utcnow()
        yield title, link, pub_dt

def smart_match(name: str, headline: str, thresh=75) -> bool:
    hl = headline.lower()
    nm = name.lower()
    if nm in hl:
        return True
    return fuzz.partial_ratio(nm, hl) >= thresh

def scrape_headlines(tickers: list[str]) -> dict[str, dict]:
    """
    Returns for each ticker:
      {
        'headlines': [(title, link, date), ...],
        'count': int,
        'daily_sentiment': {date: mean_score, ...}
      }
    over the past 10 calendar days.
    """
    today = datetime.utcnow().date()
    window_start = today - timedelta(days=9)
    results = {}
    sent_pipe = get_sentiment_pipeline()

    for tic in tickers:
        comp    = TICKER_COMPANY.get(tic, tic)
        raw_xml = fetch_google_rss(comp)
        matches = []
        for title, link, pub_dt in parse_rss(raw_xml):
            d = pub_dt.date()
            if not (window_start <= d <= today):
                continue
            if smart_match(comp, title):
                matches.append((title, link, d))

        # Compute sentiment for each matched headline
        texts = [t for t, _, _ in matches]
        scores = []
        if texts:
            batch = sent_pipe(texts, batch_size=16)
            for out in batch:
                score = out["score"]
                if out["label"] != "POSITIVE":
                    score = -score
                scores.append(score)

        # Aggregate per-day mean
        daily = defaultdict(list)
        for ((_, _, d), s) in zip(matches, scores):
            daily[d].append(s)
        daily_mean = {d: sum(v)/len(v) for d, v in daily.items()}

        results[tic] = {
            "headlines":       matches,
            "count":           len(matches),
            "daily_sentiment": daily_mean
        }
        logger.info(f"[{tic}] {len(matches)} headlines from {window_start} → {today}")
    return results
