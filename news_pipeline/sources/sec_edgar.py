"""SEC EDGAR recent-submission metadata collector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from time import sleep
from typing import Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from news_pipeline.models import Article
from news_pipeline.tickers import TrackedTicker

from .source_registry import REGULATORY_OFFICIAL


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
MATERIAL_FORMS = {
    "8-K",
    "6-K",
    "10-Q",
    "10-K",
    "DEF 14A",
    "SC 13D",
    "SC 13G",
    "S-1",
    "424B2",
    "424B3",
    "424B4",
    "424B5",
}


@dataclass(frozen=True)
class SecCollectionAttempt:
    ticker: str
    status: str
    article_count: int
    error_class: str | None = None
    error_message: str | None = None


JsonFetcher = Callable[[str, str, float], Mapping[str, object]]
_TICKER_CIK_CACHE: dict[str, int] = {}


def collect_sec_edgar_articles(
    tracked_tickers: Sequence[TrackedTicker],
    *,
    run_date: str,
    user_agent: str,
    timeout_seconds: float,
    rate_limit_seconds: float = 0.12,
    max_filings_per_ticker: int = 5,
    lookback_days: int = 14,
    json_fetcher: JsonFetcher | None = None,
) -> tuple[list[Article], list[SecCollectionAttempt]]:
    fetch_json = json_fetcher or _fetch_json
    cik_by_ticker = dict(_TICKER_CIK_CACHE)
    if not cik_by_ticker:
        try:
            ticker_payload = fetch_json(SEC_TICKER_MAP_URL, user_agent, timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - source failure must not abort the run.
            return [], [
                SecCollectionAttempt(
                    ticker="*",
                    status="failure",
                    article_count=0,
                    error_class=type(exc).__name__,
                    error_message=_safe_message(exc),
                )
            ]
        cik_by_ticker = _cik_mapping(ticker_payload)
        _TICKER_CIK_CACHE.update(cik_by_ticker)
    articles: list[Article] = []
    attempts: list[SecCollectionAttempt] = []
    cutoff = date.fromisoformat(run_date) - timedelta(days=max(1, lookback_days))
    for ticker in tracked_tickers:
        cik = cik_by_ticker.get(ticker.symbol)
        if cik is None:
            attempts.append(SecCollectionAttempt(ticker.symbol, "missing_cik", 0))
            continue
        try:
            payload = fetch_json(
                SEC_SUBMISSIONS_URL.format(cik=cik),
                user_agent,
                timeout_seconds,
            )
            ticker_articles = _normalize_recent_filings(
                payload,
                ticker=ticker,
                cik=cik,
                cutoff=cutoff,
                max_filings=max_filings_per_ticker,
            )
            articles.extend(ticker_articles)
            attempts.append(
                SecCollectionAttempt(
                    ticker=ticker.symbol,
                    status="success",
                    article_count=len(ticker_articles),
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate each issuer.
            attempts.append(
                SecCollectionAttempt(
                    ticker=ticker.symbol,
                    status="failure",
                    article_count=0,
                    error_class=type(exc).__name__,
                    error_message=_safe_message(exc),
                )
            )
        if rate_limit_seconds > 0:
            sleep(rate_limit_seconds)
    return articles, attempts


def _cik_mapping(payload: Mapping[str, object]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in payload.values():
        if not isinstance(value, Mapping):
            continue
        ticker = str(value.get("ticker") or "").upper()
        cik = value.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = int(cik)
    return mapping


def _normalize_recent_filings(
    payload: Mapping[str, object],
    *,
    ticker: TrackedTicker,
    cik: int,
    cutoff: date,
    max_filings: int,
) -> list[Article]:
    filings = payload.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if not isinstance(recent, Mapping):
        return []
    forms = list(recent.get("form") or ())
    filing_dates = list(recent.get("filingDate") or ())
    accessions = list(recent.get("accessionNumber") or ())
    primary_documents = list(recent.get("primaryDocument") or ())
    descriptions = list(recent.get("primaryDocDescription") or ())
    articles: list[Article] = []
    row_count = min(len(forms), len(filing_dates), len(accessions), len(primary_documents))
    for index in range(row_count):
        form = str(forms[index] or "").upper()
        filed = str(filing_dates[index] or "")
        if form not in MATERIAL_FORMS:
            continue
        try:
            filing_date = date.fromisoformat(filed)
        except ValueError:
            continue
        if filing_date < cutoff:
            continue
        accession = str(accessions[index] or "")
        primary_document = str(primary_documents[index] or "")
        if not accession or not primary_document:
            continue
        accession_compact = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_compact}/{primary_document}"
        )
        description = (
            str(descriptions[index] or "")
            if index < len(descriptions)
            else ""
        )
        title = f"{ticker.company_name} files {form} with the SEC"
        articles.append(
            Article(
                canonical_url=url,
                title=title,
                published_at=f"{filed}T00:00:00Z",
                snippet=description or f"Official {form} filing for {ticker.company_name}.",
                metadata={
                    "provider": "sec_edgar",
                    "source_id": "sec_edgar",
                    "source_provider": "sec_edgar",
                    "publisher_name": "SEC EDGAR",
                    "source_name": "SEC EDGAR",
                    "source_family": REGULATORY_OFFICIAL,
                    "ticker": ticker.symbol,
                    "symbols": [ticker.symbol],
                    "company": ticker.company_name,
                    "event_type": "filing_or_sec_event",
                    "filing_form_type": form,
                    "accession_number": accession,
                    "ticker_match_confidence": 1.0,
                    "ticker_match_reason": "official_filing_for_configured_ticker",
                    "direct_source": True,
                    "extract_allowed": False,
                },
            )
        )
        if len(articles) >= max(1, max_filings):
            break
    return articles


def _fetch_json(url: str, user_agent: str, timeout_seconds: float) -> Mapping[str, object]:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _safe_message(error: Exception) -> str | None:
    message = str(error).strip()
    return message[:200] if message else None
