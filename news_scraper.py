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
    "RTX": "Raytheon Technologies Corporation",
    "UUUU": "Energy Fuels Inc.",
    "PFE": "Pfizer Inc.",
    "SRAD": "Sportradar Group AG",
    "MRVL": "Marvell Technology, Inc.",
    "ADI": "Analog Devices, Inc.",
    "LLY": "Eli Lilly and Company",
    "MSFT": "Microsoft Corporation",
    "PLTR": "Palantir Technologies Inc.",
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
        'headlines':       [(title, link, date), ...],
        'count':           total_matches,
        'daily_sentiment': {date: avg_score, ...},
        'count_positive':  number_of_positive,
        'count_negative':  number_of_negative,
        'count_neutral':   number_of_neutral
      }
    over the past 7 calendar days.
    """
    today = datetime.utcnow().date()
    window_start = today - timedelta(days=6)   # 7-day window
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
        daily = defaultdict(list)
        count_pos = count_neg = count_neu = 0
        if matches:
            texts = [t for t, _, _ in matches]
            batch = sent_pipe(texts, batch_size=16)
            labels = []
            scores = []
            for out in batch:
                label = out["label"]
                labels.append(label)
                score_val = out["score"]
                if label == "POSITIVE":
                    score = score_val
                elif label == "NEGATIVE":
                    score = -score_val
                else:
                    score = 0.0
                scores.append(score)
            count_pos = labels.count("POSITIVE")
            count_neg = labels.count("NEGATIVE")
            count_neu = labels.count("NEUTRAL")

            for ((_, _, d), s) in zip(matches, scores):
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
        logger.info(f"[{tic}] {len(matches)} headlines from {window_start} → {today}")
    return results
