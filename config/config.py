import os
from pathlib import Path
from dotenv import load_dotenv

DOTENV_PATH = Path(__file__).parent / "secrets.env"
load_dotenv(DOTENV_PATH)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
if not ALPHA_VANTAGE_KEY:
    raise RuntimeError(f"Missing ALPHA_VANTAGE_KEY in {DOTENV_PATH}")

# API Configuration (for new endpoints)
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# Feature-specific configurations
SYMBOL_INTAKE_CONFIG = {
    "max_queue_size": int(os.getenv("INTAKE_MAX_QUEUE_SIZE", "100")),
    "job_timeout_seconds": int(os.getenv("INTAKE_JOB_TIMEOUT", "300")),
    "validation_strict": os.getenv("INTAKE_VALIDATION_STRICT", "true").lower() == "true"
}

NEWS_CORROBORATION_CONFIG = {
    "min_sources": int(os.getenv("NEWS_MIN_SOURCES", "2")),
    "confidence_threshold": float(os.getenv("NEWS_CONFIDENCE_THRESHOLD", "0.7"))
}

# News scraping configuration
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "30"))

# Multisource price data configuration
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "")
POLYGON_KEY = os.getenv("POLYGON_KEY", "")
EODHD_KEY = os.getenv("EODHD_KEY", "")

# News API configuration (F03)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# FinBERT configuration (F04)
FINBERT_LAMBDA = float(os.getenv("FINBERT_LAMBDA", "0.2"))
FINBERT_BARRIER_DAYS = int(os.getenv("FINBERT_BARRIER_DAYS", "29"))

# BEGIN F-YF-RATE-CONFIG
YF_CACHE_TTL_HOURS = int(os.getenv("YF_CACHE_TTL_HOURS", "24"))
YF_MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "2"))
YF_BACKOFF_BASE_SECONDS = float(os.getenv("YF_BACKOFF_BASE_SECONDS", "2"))
YF_REFRESH_WINDOW_UTC_HOUR = int(os.getenv("YF_REFRESH_WINDOW_UTC_HOUR", "2"))
YF_DAILY_KEY = os.getenv("YF_DAILY_KEY", "period=2y|interval=1d|auto_adjust=1")
YF_TEST_FAST_BACKOFF = os.getenv("YF_TEST_FAST_BACKOFF", "0")  # '1' in tests
# END F-YF-RATE-CONFIG

# Alpha Vantage batching configuration
AV_BATCH_SIZE = int(os.getenv("AV_BATCH_SIZE", "100"))
AV_DAILY_QUOTA = int(os.getenv("AV_DAILY_QUOTA", "25"))

# Price data source rate limits and configurations
PRICE_DATA_CONFIG = {
    "alpha_vantage": {
        "rate_limit": AV_DAILY_QUOTA,  # calls per day
        "rate_window": 86400,  # seconds (24 hours)
        "batch_size": AV_BATCH_SIZE,  # up to 100 symbols per batch call
        "timeout": 30
    },
    "yahoo": {
        "rate_limit": 2000,  # calls per hour (generous estimate)
        "rate_window": 3600,
        "batch_size": 10,
        "timeout": 15
    },
    "finnhub": {
        "rate_limit": 60,  # calls per minute
        "rate_window": 60,
        "batch_size": 5,
        "timeout": 20
    },
    "polygon": {
        "rate_limit": 5,  # calls per minute (free tier)
        "rate_window": 60,
        "batch_size": 1,
        "timeout": 20
    },
    "eodhd": {
        "rate_limit": 100000,  # calls per day (paid plan)
        "rate_window": 86400,
        "batch_size": 20,
        "timeout": 30
    }
}
