import os
import time
import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
from rapidfuzz import fuzz
from transformers import pipeline
from dotenv import load_dotenv
from config.config import ALPHA_VANTAGE_KEY, NEWS_LOOKBACK_DAYS
from config.feature_flags import is_newsapi_ingestion_enabled, is_news_corroboration_enabled, is_async_io_enabled

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_date_window(lookback_days: int = 30, tz: str = "America/Los_Angeles"):
    """Get date window for news scraping with proper timezone handling.
    
    Args:
        lookback_days: Number of days to look back from today
        tz: Timezone string (default: America/Los_Angeles)
        
    Returns:
        tuple: (start_date_iso, end_date_iso) as ISO format strings
    """
    today = datetime.now(ZoneInfo(tz)).date()
    start = today - timedelta(days=lookback_days)
    return start.isoformat(), today.isoformat()

# ─── Load Alpha Vantage key ───────────────────────────────────────────────────
load_dotenv("config/secrets.env")
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
if not API_KEY:
    logger.error("Missing ALPHA_VANTAGE_KEY in config/secrets.env")
    raise RuntimeError("Set ALPHA_VANTAGE_KEY in config/secrets.env")

# ─── Mapping tickers to canonical company names ───────────────────────────────
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
    "PLTR": "Palantir Technologies Inc.",
    "RIVN": "Rivian Automotive, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "NFLX": "Netflix, Inc.",
    "CRM": "Salesforce, Inc.",
    "ADBE": "Adobe Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "INTC": "Intel Corporation",
    "QCOM": "Qualcomm Incorporated",
    "AVGO": "Broadcom Inc.",
    "TXN": "Texas Instruments Incorporated",
    "MU": "Micron Technology, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "WFC": "Wells Fargo & Company",
    "GS": "Goldman Sachs Group, Inc.",
    "MS": "Morgan Stanley",
    "C": "Citigroup Inc.",
    "USB": "U.S. Bancorp",
    "PNC": "PNC Financial Services Group, Inc.",
    "COF": "Capital One Financial Corporation",
    "AXP": "American Express Company",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Incorporated",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co., Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "DHR": "Danaher Corporation",
    "ABT": "Abbott Laboratories",
    "BMY": "Bristol-Myers Squibb Company",
    "PG": "Procter & Gamble Co.",
    "KO": "Coca-Cola Company",
    "PEP": "PepsiCo, Inc.",
    "WMT": "Walmart Inc.",
    "HD": "Home Depot, Inc.",
    "MCD": "McDonald's Corporation",
    "DIS": "Walt Disney Company",
    "NKE": "Nike, Inc.",
    "SBUX": "Starbucks Corporation",
    "TGT": "Target Corporation",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources, Inc.",
    "SLB": "Schlumberger Limited",
    "KMI": "Kinder Morgan, Inc.",
    "PSX": "Phillips 66",
    "VLO": "Valero Energy Corporation",
    "MPC": "Marathon Petroleum Corporation",
    "OXY": "Occidental Petroleum Corporation",
    "BA": "Boeing Company",
    "CAT": "Caterpillar Inc.",
    "GE": "General Electric Company",
    "MMM": "3M Company",
    "HON": "Honeywell International Inc.",
    "UPS": "United Parcel Service, Inc.",
    "FDX": "FedEx Corporation",
    "LMT": "Lockheed Martin Corporation",
    "NOC": "Northrop Grumman Corporation"
    # ... add more tickers and company names as needed
}

# ─── Alternative company names for better matching ─────────────────────────────
COMPANY_ALTERNATIVES = {
    "RTX": ["Raytheon", "Raytheon Technologies", "RTX Corporation", "RTX Corp"],
    "PFE": ["Pfizer", "Pfizer Inc"],
    "MRVL": ["Marvell", "Marvell Technology", "Marvell Tech"],
    "ADI": ["Analog Devices", "Analog Devices Inc"],
    "LLY": ["Eli Lilly", "Eli Lilly and Co", "Lilly"],
    "RIVN": ["Rivian", "Rivian Automotive", "Rivian Auto"],
    "PLTR": ["Palantir", "Palantir Technologies", "Palantir Tech"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc"],
    "META": ["Facebook", "Meta", "Meta Platforms"],
    "NVDA": ["NVIDIA", "Nvidia", "Nvidia Corp"],
    "TSLA": ["Tesla", "Tesla Inc", "Tesla Motors"],
    "AAPL": ["Apple", "Apple Inc", "Apple Computer"],
    "MSFT": ["Microsoft", "Microsoft Corp", "Microsoft Corporation"],
    "AMZN": ["Amazon", "Amazon.com", "Amazon Inc"],
    "NFLX": ["Netflix", "Netflix Inc"],
    "CRM": ["Salesforce", "Salesforce.com", "Salesforce Inc"],
    "ADBE": ["Adobe", "Adobe Inc", "Adobe Systems"],
    "PYPL": ["PayPal", "PayPal Holdings", "PayPal Inc"],
    "INTC": ["Intel", "Intel Corp", "Intel Corporation"],
    "AMD": ["Advanced Micro Devices", "AMD Inc"],
    "QCOM": ["Qualcomm", "Qualcomm Inc", "Qualcomm Incorporated"],
    "AVGO": ["Broadcom", "Broadcom Inc", "Broadcom Corporation"],
    "TXN": ["Texas Instruments", "TI", "Texas Instruments Inc"],
    "MU": ["Micron", "Micron Technology", "Micron Tech"],
    "JPM": ["JPMorgan", "JPMorgan Chase", "JP Morgan"],
    "BAC": ["Bank of America", "BofA", "Bank of America Corp"],
    "WFC": ["Wells Fargo", "Wells Fargo & Co", "Wells Fargo Bank"],
    "GS": ["Goldman Sachs", "Goldman Sachs Group", "Goldman"],
    "MS": ["Morgan Stanley", "Morgan Stanley Inc"],
    "C": ["Citigroup", "Citi", "Citigroup Inc"],
    "USB": ["U.S. Bancorp", "US Bank", "U.S. Bank"],
    "PNC": ["PNC Financial", "PNC Bank", "PNC Financial Services"],
    "COF": ["Capital One", "Capital One Financial", "Capital One Bank"],
    "AXP": ["American Express", "Amex", "American Express Co"],
    "JNJ": ["Johnson & Johnson", "J&J", "Johnson and Johnson"],
    "UNH": ["UnitedHealth", "UnitedHealth Group", "United Health"],
    "ABBV": ["AbbVie", "AbbVie Inc"],
    "MRK": ["Merck", "Merck & Co", "Merck Inc"],
    "TMO": ["Thermo Fisher", "Thermo Fisher Scientific", "Thermo"],
    "DHR": ["Danaher", "Danaher Corp", "Danaher Corporation"],
    "ABT": ["Abbott", "Abbott Laboratories", "Abbott Labs"],
    "BMY": ["Bristol-Myers", "Bristol-Myers Squibb", "BMS"],
    "PG": ["Procter & Gamble", "P&G", "Procter and Gamble"],
    "KO": ["Coca-Cola", "Coke", "Coca-Cola Co"],
    "PEP": ["PepsiCo", "Pepsi", "Pepsi-Cola"],
    "WMT": ["Walmart", "Wal-Mart", "Walmart Inc"],
    "HD": ["Home Depot", "The Home Depot", "Home Depot Inc"],
    "MCD": ["McDonald's", "McDonalds", "McDonald's Corp"],
    "DIS": ["Disney", "Walt Disney", "Walt Disney Co"],
    "NKE": ["Nike", "Nike Inc", "Nike Corporation"],
    "SBUX": ["Starbucks", "Starbucks Corp", "Starbucks Coffee"],
    "TGT": ["Target", "Target Corp", "Target Corporation"],
    "XOM": ["Exxon", "Exxon Mobil", "ExxonMobil"],
    "CVX": ["Chevron", "Chevron Corp", "Chevron Corporation"],
    "COP": ["ConocoPhillips", "Conoco", "Conoco Phillips"],
    "EOG": ["EOG Resources", "EOG", "EOG Inc"],
    "SLB": ["Schlumberger", "Schlumberger Ltd", "SLB"],
    "KMI": ["Kinder Morgan", "Kinder Morgan Inc", "KMI"],
    "PSX": ["Phillips 66", "Phillips", "Phillips 66 Co"],
    "VLO": ["Valero", "Valero Energy", "Valero Corp"],
    "MPC": ["Marathon Petroleum", "Marathon", "Marathon Petroleum Corp"],
    "OXY": ["Occidental", "Occidental Petroleum", "Oxy"],
    "BA": ["Boeing", "Boeing Co", "Boeing Company"],
    "CAT": ["Caterpillar", "Cat", "Caterpillar Inc"],
    "GE": ["General Electric", "GE", "General Electric Co"],
    "MMM": ["3M", "3M Company", "3M Co"],
    "HON": ["Honeywell", "Honeywell International", "Honeywell Inc"],
    "UPS": ["United Parcel Service", "UPS", "United Parcel"],
    "FDX": ["FedEx", "Federal Express", "FedEx Corp"],
    "LMT": ["Lockheed Martin", "Lockheed", "Lockheed Martin Corp"],
    "NOC": ["Northrop Grumman", "Northrop", "Northrop Grumman Corp"]
}

# ─── FinBERT sentiment pipeline (lazy initialization) ─────────────────────────
_sentiment_pipe = None
def get_sentiment_pipeline():
    """Load the FinBERT sentiment analysis model (lazy singleton)."""
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
    return _sentiment_pipe

# ─── Helper functions for news scraping ──────────────────────────────────────

def fetch_google_rss(query: str, retries: int = 3) -> str:
    """Fetch Google News RSS feed for the query. Return XML text or empty string on failure."""
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            logger.warning(f"RSS request error for query '{query}': {e}")
        time.sleep(1)
    logger.warning(f"Failed to fetch Google RSS for query: {query}")
    return ""

def parse_rss(xml_text: str):
    """Parse RSS XML text and yield (title, link, pub_datetime) for each item."""
    if not xml_text:
        return []
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item"):
        title = item.findtext("title", "") or ""
        link  = item.findtext("link", "") or ""
        pub   = item.findtext("pubDate", "") or ""
        try:
            # RSS pubDate example: "Tue, 20 Jun 2025 14:30:00 GMT"
            pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            pub_dt = datetime.utcnow()
        yield title, link, pub_dt

def fetch_av_news(ticker: str, retries: int = 3):
    """
    Fetch Alpha Vantage News Sentiment feed for the given ticker.
    Returns a list of (title, url, pub_datetime) tuples.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers":  ticker,
        "apikey":   API_KEY
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json().get("feed", [])
                articles = []
                for item in data:
                    title = item.get("title", "") or ""
                    link  = item.get("url", "") or ""
                    ts    = item.get("time_published", "")
                    try:
                        # Alpha Vantage time_published is ISO format, e.g. "2025-06-20T13:45:00Z"
                        pub_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        pub_dt = datetime.utcnow()
                    articles.append((title, link, pub_dt))
                return articles
        except Exception as e:
            logger.warning(f"AV news request error for {ticker}: {e}")
        time.sleep(1)
    logger.warning(f"Failed to fetch AV news for {ticker}")
    return []

def smart_match(name: str, headline: str, thresh: int = 75) -> bool:
    """
    Determine if the company name is relevant in the headline using fuzzy matching.
    Returns True if exact or partial match above threshold is found.
    """
    hl = headline.lower()
    nm = name.lower()
    
    # Check exact match first
    if nm in hl:
        return True
    
    # Check fuzzy match
    if fuzz.partial_ratio(nm, hl) >= thresh:
        return True
    
    return False

def smart_match_with_alternatives(ticker: str, headline: str, thresh: int = 75) -> bool:
    """
    Determine if the company name is relevant in the headline using fuzzy matching
    with alternative company names.
    Returns True if exact or partial match above threshold is found.
    """
    hl = headline.lower()
    
    # Get primary company name
    primary_name = TICKER_COMPANY.get(ticker, ticker).lower()
    
    # Check primary name first
    if primary_name in hl or fuzz.partial_ratio(primary_name, hl) >= thresh:
        return True
    
    # Check alternative names
    alternatives = COMPANY_ALTERNATIVES.get(ticker, [])
    for alt_name in alternatives:
        alt_lower = alt_name.lower()
        if alt_lower in hl or fuzz.partial_ratio(alt_lower, hl) >= thresh:
            return True
    
    # Check ticker symbol (case-insensitive)
    if ticker.lower() in hl:
        return True
    
    return False

# ─── Main function to scrape headlines and sentiments ─────────────────────────

def scrape_headlines(tickers: list[str], days: int = None) -> dict[str, dict]:
    """
    For each ticker in the list, scrape recent headlines (past `days` days)
    from Google News and Alpha Vantage, then compute sentiment scores.
    Returns a dictionary mapping each ticker to:
      {
        'headlines':       [(title, link, date), ...],
        'count':           total number of matched headlines,
        'daily_sentiment': {date: avg_sentiment_score, ...},
        'count_positive':  number of positive headlines,
        'count_negative':  number of negative headlines,
        'count_neutral':   number of neutral headlines
      }
    """
    # Use configured lookback days if not specified
    if days is None:
        days = NEWS_LOOKBACK_DAYS
    
    # Get proper date window using timezone-aware helper
    window_start_iso, today_iso = get_date_window(lookback_days=days, tz="America/Los_Angeles")
    today = datetime.fromisoformat(today_iso).date()
    window_start = datetime.fromisoformat(window_start_iso).date()
    results = {}
    sent_pipe = get_sentiment_pipeline()

    for tic in tickers:
        comp_name = TICKER_COMPANY.get(tic, tic)  # fallback to ticker if name not known
        
        # BEGIN F03 - Multisource news fetching with NewsAPI
        all_sources = []
        
        # 1. Fetch from existing sources (Google RSS + Alpha Vantage)
        google_xml = fetch_google_rss(comp_name)
        google_items = list(parse_rss(google_xml))
        av_items = fetch_av_news(tic)
        all_sources.extend(google_items + av_items)
        
        # 2. Fetch from NewsAPI if enabled
        if is_newsapi_ingestion_enabled():
            try:
                from integrations.newsapi_client import newsapi_client
                newsapi_result = newsapi_client.fetch([tic], days)
                
                if newsapi_result.success:
                    logger.info(f"F03: NewsAPI fetched {len(newsapi_result.articles)} articles for {tic}")
                    all_sources.extend(newsapi_result.articles)
                elif newsapi_result.fallback_triggered:
                    logger.warning(f"F03: NewsAPI fallback triggered for {tic}: {newsapi_result.error_message}")
                    # Continue with RSS/AV sources only
                    
            except Exception as e:
                logger.error(f"F03: NewsAPI integration failed for {tic}: {e}")
                # Continue with RSS/AV sources only
        
        # 3. Filter headlines that mention the company/ticker and are within date window
        combined = []
        for title, link, pub_dt in all_sources:
            pub_date = pub_dt.date()
            
            # Check if headline mentions the company/ticker
            if smart_match_with_alternatives(tic, title):
                # Filter headlines within the configured date window
                if window_start <= pub_date <= today:
                    combined.append((title, link, pub_dt))  # Keep datetime for deduplication
        
        # 4. Enhanced deduplication with F03 improvements
        if len(combined) > 1:
            try:
                from services.news_deduplicator import dedupe_headlines_simple
                unique_headlines = dedupe_headlines_simple(combined, threshold=0.75)
                logger.info(f"F03: Enhanced deduplication for {tic}: {len(combined)} → {len(unique_headlines)} headlines")
                
                # Convert back to date objects for compatibility
                unique_headlines = [(title, link, dt.date()) for title, link, dt in unique_headlines]
            except Exception as e:
                logger.warning(f"F03: Enhanced deduplication failed for {tic}, using simple deduplication: {e}")
                # Fallback to original simple deduplication
                seen_links = set()
                unique_headlines = []
                for title, link, dt in combined:
                    if link and link not in seen_links:
                        seen_links.add(link)
                        unique_headlines.append((title, link, dt.date()))
        else:
            # No need for deduplication with 0-1 headlines
            unique_headlines = [(title, link, dt.date()) for title, link, dt in combined]
        # END F03
        
        # 4. Sentiment analysis on unique headlines
        daily_scores = defaultdict(list)
        count_pos = count_neg = count_neu = 0
        if unique_headlines:
            texts = [title for title, _, _ in unique_headlines]
            try:
                batch_results = sent_pipe(texts, batch_size=16)
            except Exception as e:
                logger.error(f"Sentiment model inference error: {e}")
                batch_results = []
            for (title, link, d), result in zip(unique_headlines, batch_results):
                if not result:
                    continue
                label = result["label"].lower()
                score = result.get("score", 0)
                # Map labels to sentiment scores
                if label == "positive":
                    daily_scores[d].append(score)
                    count_pos += 1
                elif label == "negative":
                    daily_scores[d].append(-score)
                    count_neg += 1
                else:
                    # 'neutral' or any other label
                    daily_scores[d].append(0.0)
                    count_neu += 1
        
        # 5. Compute daily average sentiment scores
        daily_avg_sent = {date: (sum(vals)/len(vals) if vals else 0.0) 
                          for date, vals in daily_scores.items()}
        
        # 6. Store results for this ticker
        results[tic] = {
            "headlines":       unique_headlines,
            "count":           len(unique_headlines),
            "daily_sentiment": daily_avg_sent,
            "count_positive":  count_pos,
            "count_negative":  count_neg,
            "count_neutral":   count_neu,
        }
        logger.info(f"[{tic}] {len(unique_headlines)} headlines from {window_start} to {today} (tz=America/Los_Angeles, lookback={days}d)")
    
    # Apply news clustering and consensus scoring if enabled
    try:
        from services.news_clustering import NewsConsensusEnhancer
        enhancer = NewsConsensusEnhancer()
        results = enhancer.enhance_news_with_consensus(results)
    except ImportError:
        logger.debug("News clustering service not available, using basic sentiment")
    except Exception as e:
        logger.warning(f"Error applying news consensus enhancement: {e}")
    
    # BEGIN F03 - News corroboration enhancement
    if is_news_corroboration_enabled() and results:
        try:
            from services.news_corroboration import NewsCorroborationService
            corroboration_service = NewsCorroborationService()
            results = corroboration_service.corroborate_headlines(results)
            logger.info("F03: News corroboration enhancement applied")
        except Exception as e:
            logger.warning(f"F03: News corroboration failed: {e}")
            # Continue with uncorroborated results
    # END F03
    
    return results


# BEGIN F09 - Async I/O layer for news scraping
import asyncio
from services.retry_policies import (
    AsyncHTTPClient, make_resilient_requests, load_async_config,
    get_async_stats, AsyncRetryConfig
)


async def fetch_google_rss_async(query: str, config: dict = None) -> str:
    """
    F09: Async version of fetch_google_rss with resilient HTTP client
    """
    config = config or load_async_config()
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}"
    
    retry_config = AsyncRetryConfig(
        max_retries=config.get('max_retries', 3),
        base_delay=1.0,
        max_delay=config.get('backoff_max', 60)
    )
    
    try:
        async with AsyncHTTPClient(retry_config=retry_config, semaphore_limit=config.get('max_concurrency', 5)) as client:
            response = await client.get(url, timeout=30)
            
            if response['status'] == 200:
                return response['data']
            else:
                logger.warning(f"F09: RSS request failed for query '{query}': status {response['status']}")
                return ""
                
    except Exception as e:
        logger.warning(f"F09: RSS request error for query '{query}': {e}")
        return ""


async def fetch_av_news_async(ticker: str, config: dict = None) -> List[tuple]:
    """
    F09: Async version of fetch_av_news with resilient HTTP client
    Returns a list of (title, url, pub_datetime) tuples
    """
    config = config or load_async_config()
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": API_KEY
    }
    
    retry_config = AsyncRetryConfig(
        max_retries=config.get('max_retries', 3),
        base_delay=1.0,
        max_delay=config.get('backoff_max', 60)
    )
    
    try:
        async with AsyncHTTPClient(retry_config=retry_config, semaphore_limit=config.get('max_concurrency', 5)) as client:
            response = await client.get(url, params=params, timeout=30)
            
            if response['status'] == 200 and isinstance(response['data'], dict):
                data = response['data'].get("feed", [])
                articles = []
                for item in data:
                    title = item.get("title", "") or ""
                    link = item.get("url", "") or ""
                    ts = item.get("time_published", "")
                    try:
                        pub_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        pub_dt = datetime.utcnow()
                    articles.append((title, link, pub_dt))
                return articles
            else:
                logger.warning(f"F09: AV news request failed for {ticker}: status {response['status']}")
                return []
                
    except Exception as e:
        logger.warning(f"F09: AV news request error for {ticker}: {e}")
        return []


async def scrape_headlines_async(tickers: List[str], days: int = None) -> dict:
    """
    F09: Async version of scrape_headlines using asyncio gather with semaphore limits
    Implements resilient I/O with circuit breakers, exponential backoff, and caching
    
    AC1: Daily run resilient to transient failures; retries capped; failures logged but do not abort
    """
    if not is_async_io_enabled():
        logger.info("F09: Async I/O disabled, falling back to sync scraping")
        return scrape_headlines(tickers, days)
    
    logger.info(f"F09: Starting async headline scraping for {len(tickers)} tickers")
    
    # Load F09 configuration
    config = load_async_config()
    
    # Use configured lookback days if not specified
    if days is None:
        days = NEWS_LOOKBACK_DAYS
    
    # Get proper date window using timezone-aware helper
    window_start_iso, today_iso = get_date_window(lookback_days=days, tz="America/Los_Angeles")
    today = datetime.fromisoformat(today_iso).date()
    window_start = datetime.fromisoformat(window_start_iso).date()
    
    results = {}
    sent_pipe = get_sentiment_pipeline()
    
    # Prepare async tasks for all tickers
    async_tasks = []
    for ticker in tickers:
        task = _scrape_single_ticker_async(
            ticker, window_start, today, sent_pipe, config
        )
        async_tasks.append(task)
    
    # Execute with asyncio.gather and semaphore limits (core F09 pattern)
    try:
        logger.info(f"F09: Executing {len(async_tasks)} ticker scraping tasks with max_concurrency={config.get('max_concurrency', 5)}")
        ticker_results = await asyncio.gather(*async_tasks, return_exceptions=False)
        
        # Combine results
        for ticker, result in zip(tickers, ticker_results):
            if result and not result.get('failed', False):
                results[ticker] = result
            else:
                # AC1: failures logged but do not abort
                logger.warning(f"F09: Failed to scrape {ticker}, continuing with other tickers")
                results[ticker] = {
                    "headlines": [],
                    "count": 0,
                    "daily_sentiment": {},
                    "count_positive": 0,
                    "count_negative": 0,
                    "count_neutral": 0,
                }
    
    except Exception as e:
        # AC1: Daily run resilient to transient failures
        logger.error(f"F09: Async scraping failed, falling back to sync: {e}")
        return scrape_headlines(tickers, days)
    
    # Apply post-processing enhancements (same as sync version)
    try:
        from services.news_clustering import NewsConsensusEnhancer
        enhancer = NewsConsensusEnhancer()
        results = enhancer.enhance_news_with_consensus(results)
    except (ImportError, Exception) as e:
        logger.debug(f"F09: News clustering not available: {e}")
    
    # Apply news corroboration if enabled
    if is_news_corroboration_enabled() and results:
        try:
            from services.news_corroboration import NewsCorroborationService
            corroboration_service = NewsCorroborationService()
            results = corroboration_service.corroborate_headlines(results)
            logger.info("F09: News corroboration enhancement applied")
        except Exception as e:
            logger.warning(f"F09: News corroboration failed: {e}")
    
    # Log F09 performance stats
    async_stats = get_async_stats()
    logger.info(f"F09: Async scraping completed. Stats: {async_stats}")
    
    return results


async def _scrape_single_ticker_async(
    ticker: str, 
    window_start: datetime.date, 
    today: datetime.date, 
    sent_pipe, 
    config: dict
) -> dict:
    """
    F09: Async scraping for a single ticker with resilient I/O
    """
    try:
        comp_name = TICKER_COMPANY.get(ticker, ticker)
        
        # Gather news from multiple sources concurrently using F09 patterns
        # This replaces the sequential fetch_google_rss + fetch_av_news calls
        google_task = fetch_google_rss_async(comp_name, config)
        av_task = fetch_av_news_async(ticker, config)
        
        # Execute both requests concurrently with circuit breakers
        google_xml, av_items = await asyncio.gather(google_task, av_task, return_exceptions=False)
        
        # Parse Google RSS results
        google_items = list(parse_rss(google_xml)) if google_xml else []
        
        # Combine all sources
        all_sources = google_items + (av_items or [])
        
        # NewsAPI integration (if enabled)
        if is_newsapi_ingestion_enabled():
            try:
                from integrations.newsapi_client import newsapi_client
                # Note: This still uses sync API as newsapi_client is sync
                # In a full F09 implementation, this would be async too
                newsapi_result = newsapi_client.fetch([ticker], window_start, today)
                
                if newsapi_result.success:
                    logger.info(f"F09: NewsAPI fetched {len(newsapi_result.articles)} articles for {ticker}")
                    all_sources.extend(newsapi_result.articles)
                elif newsapi_result.fallback_triggered:
                    logger.warning(f"F09: NewsAPI fallback triggered for {ticker}: {newsapi_result.error_message}")
                    
            except Exception as e:
                logger.warning(f"F09: NewsAPI integration failed for {ticker}: {e}")
        
        # Filter headlines that mention the company/ticker and are within date window
        combined = []
        for title, link, pub_dt in all_sources:
            pub_date = pub_dt.date()
            
            if smart_match_with_alternatives(ticker, title):
                if window_start <= pub_date <= today:
                    combined.append((title, link, pub_dt))
        
        # Enhanced deduplication
        if len(combined) > 1:
            try:
                from services.news_deduplicator import dedupe_headlines_simple
                unique_headlines = dedupe_headlines_simple(combined, threshold=0.75)
                logger.info(f"F09: Deduplication for {ticker}: {len(combined)} → {len(unique_headlines)} headlines")
                unique_headlines = [(title, link, dt.date()) for title, link, dt in unique_headlines]
            except Exception as e:
                logger.warning(f"F09: Deduplication failed for {ticker}: {e}")
                # Fallback to simple deduplication
                seen_links = set()
                unique_headlines = []
                for title, link, dt in combined:
                    if link and link not in seen_links:
                        seen_links.add(link)
                        unique_headlines.append((title, link, dt.date()))
        else:
            unique_headlines = [(title, link, dt.date()) for title, link, dt in combined]
        
        # Sentiment analysis (still sync as FinBERT model is sync)
        daily_scores = defaultdict(list)
        count_pos = count_neg = count_neu = 0
        
        if unique_headlines:
            texts = [title for title, _, _ in unique_headlines]
            try:
                batch_results = sent_pipe(texts, batch_size=16)
            except Exception as e:
                logger.error(f"F09: Sentiment model inference error for {ticker}: {e}")
                batch_results = []
            
            for (title, link, d), result in zip(unique_headlines, batch_results):
                if not result:
                    continue
                label = result["label"].lower()
                score = result.get("score", 0)
                if label == "positive":
                    daily_scores[d].append(score)
                    count_pos += 1
                elif label == "negative":
                    daily_scores[d].append(-score)
                    count_neg += 1
                else:
                    daily_scores[d].append(0.0)
                    count_neu += 1
        
        # Compute daily average sentiment scores
        daily_avg_sent = {date: (sum(vals)/len(vals) if vals else 0.0) 
                          for date, vals in daily_scores.items()}
        
        # Log ticker completion
        logger.info(f"F09: [{ticker}] {len(unique_headlines)} headlines from {window_start} to {today}")
        
        return {
            "headlines": unique_headlines,
            "count": len(unique_headlines),
            "daily_sentiment": daily_avg_sent,
            "count_positive": count_pos,
            "count_negative": count_neg,
            "count_neutral": count_neu,
        }
        
    except Exception as e:
        logger.error(f"F09: Error scraping ticker {ticker}: {e}")
        return {
            'failed': True,
            'error': str(e),
            'ticker': ticker
        }


# Wrapper function that automatically chooses sync or async based on feature flag
def scrape_headlines_resilient(tickers: List[str], days: int = None) -> dict:
    """
    F09: Wrapper function that automatically chooses sync or async based on feature flag
    This provides backward compatibility while enabling F09 when the flag is on
    """
    if is_async_io_enabled():
        # Run async version
        return asyncio.run(scrape_headlines_async(tickers, days))
    else:
        # Run original sync version
        return scrape_headlines(tickers, days)

# END F09
