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
SEC_EVENT_TYPES = {
    "8-K": ("material_event", 100, "material current report"),
    "10-Q": ("quarterly_report", 92, "quarterly financial report"),
    "10-K": ("annual_report", 96, "annual financial report"),
    "6-K": ("foreign_issuer_report", 90, "foreign issuer report"),
    "SC 13D": ("ownership_event", 88, "beneficial ownership event"),
    "SC 13G": ("ownership_event", 84, "beneficial ownership disclosure"),
    "DEF 14A": ("proxy_event", 78, "definitive proxy statement"),
    "S-1": ("registration_event", 90, "securities registration statement"),
    "424B": ("offering_or_prospectus", 94, "offering or prospectus update"),
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
        event = classify_sec_form(form)
        if event is None:
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
        event_type, priority, summary_label = event
        title = f"{ticker.company_name} files {form} with the SEC"
        event_summary = (
            f"{ticker.company_name} filed a {summary_label} ({form})"
            f"{': ' + description if description else '.'}"
        )
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
                    "filing_event_type": event_type,
                    "official_event_priority": priority,
                    "sec_event_summary": event_summary,
                    "sec_event_basis": (
                        "form_type_and_primary_document_description"
                        if description
                        else "form_type"
                    ),
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


def classify_sec_form(
    form: str,
) -> tuple[str, int, str] | None:
    normalized = " ".join(str(form or "").upper().split())
    base = normalized.removesuffix("/A")
    aliases = {
        "13D": "SC 13D",
        "13G": "SC 13G",
        "SC13D": "SC 13D",
        "SC13G": "SC 13G",
        "DEF14A": "DEF 14A",
    }
    base = aliases.get(base, base)
    if base.startswith("424B"):
        return SEC_EVENT_TYPES["424B"]
    if base in SEC_EVENT_TYPES:
        return SEC_EVENT_TYPES[base]
    return None


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
