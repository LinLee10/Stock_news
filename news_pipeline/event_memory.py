"""Durable article, event, and sentiment memory records."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .article_fetch import ArticleFetchSummary
from .article_types import classify_article_type
from .dedup import DedupeCluster
from .models import Article
from .sentiment import analyze_sentiment
from .ticker_matching import assess_ticker_matches
from .tickers import load_tracked_tickers


@dataclass(frozen=True)
class EventMemoryRecord:
    article_id: str
    canonical_url: str
    published_at: str | None
    ticker: str
    company: str
    source_provider: str
    source_family: str
    article_type: str
    cluster_id: str
    ticker_match_confidence: float
    extraction_basis: str
    extraction_quality_grade: str
    internal_sentiment: float
    external_sentiment_provider: str | None
    external_sentiment: float | None
    event_type: str
    event_summary: str
    run_id: str
    run_date: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_event_memory_records(
    *,
    articles: Sequence[Article],
    clusters: Sequence[DedupeCluster],
    article_fetch_summary: ArticleFetchSummary,
    article_ids_by_url: Mapping[str, str],
    run_id: str,
    run_date: str,
) -> tuple[EventMemoryRecord, ...]:
    tracked = {ticker.symbol: ticker for ticker in load_tracked_tickers()}
    cluster_by_url = {
        article.canonical_url: cluster
        for cluster in clusters
        for article in cluster.articles
    }
    extraction_by_url = {
        record.canonical_url: record
        for record in article_fetch_summary.records
    }
    records: list[EventMemoryRecord] = []
    for article in articles:
        classification = classify_article_type(article)
        sentiment = analyze_sentiment(
            article_ids_by_url.get(article.canonical_url)
            or article.article_id
            or article.canonical_url,
            article.full_text or article.snippet or article.title,
            _article_basis(article),
        )
        extraction = extraction_by_url.get(article.canonical_url)
        cluster = cluster_by_url.get(article.canonical_url)
        for match in assess_ticker_matches(article):
            ticker = tracked.get(match.ticker)
            if ticker is None:
                continue
            external_provider, external_sentiment = (
                _external_sentiment_for_ticker(article, match.ticker)
            )
            records.append(
                EventMemoryRecord(
                    article_id=article_ids_by_url.get(article.canonical_url)
                    or article.article_id
                    or article.canonical_url,
                    canonical_url=article.canonical_url,
                    published_at=article.published_at,
                    ticker=match.ticker,
                    company=ticker.company_name,
                    source_provider=str(
                        article.metadata.get("source_provider")
                        or article.metadata.get("provider")
                        or "unknown"
                    ),
                    source_family=str(
                        article.metadata.get("source_family") or "unknown"
                    ),
                    article_type=classification.primary_type,
                    cluster_id=str(cluster.cluster_id if cluster else ""),
                    ticker_match_confidence=round(match.confidence, 4),
                    extraction_basis=(
                        extraction.extraction_basis
                        if extraction
                        else _article_basis(article)
                    ),
                    extraction_quality_grade=(
                        extraction.extraction_quality_grade
                        if extraction
                        else _fallback_quality_grade(article)
                    ),
                    internal_sentiment=round(sentiment.score, 4),
                    external_sentiment_provider=external_provider,
                    external_sentiment=external_sentiment,
                    event_type=str(
                        article.metadata.get("filing_event_type")
                        or article.metadata.get("event_type")
                        or classification.primary_type
                    ),
                    event_summary=str(
                        article.metadata.get("sec_event_summary")
                        or article.snippet
                        or article.title
                    ),
                    run_id=run_id,
                    run_date=run_date,
                )
            )
    return tuple(records)


def write_event_memory_artifacts(
    records: Sequence[EventMemoryRecord],
    *,
    output_dir: str | Path,
) -> tuple[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "event_memory_daily.json"
    csv_path = directory / "event_memory_daily.csv"
    rows = [record.as_dict() for record in records]
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fieldnames = tuple(EventMemoryRecord.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(json_path), str(csv_path)


def sec_event_candidate_rows(
    articles: Sequence[Article],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "article_id": article.article_id,
            "canonical_url": article.canonical_url,
            "published_at": article.published_at,
            "ticker": article.metadata.get("ticker"),
            "company": article.metadata.get("company"),
            "filing_form_type": article.metadata.get("filing_form_type"),
            "filing_event_type": article.metadata.get("filing_event_type"),
            "official_event_priority": article.metadata.get(
                "official_event_priority"
            ),
            "sec_event_summary": article.metadata.get("sec_event_summary"),
            "sec_event_basis": article.metadata.get("sec_event_basis"),
        }
        for article in articles
        if article.metadata.get("source_provider") == "sec_edgar"
    )


def _article_basis(article: Article) -> str:
    if article.full_text:
        return "full_text"
    if article.snippet:
        return "snippet"
    return "title"


def _fallback_quality_grade(article: Article) -> str:
    if article.full_text:
        return "usable_full_text"
    if article.snippet:
        return "snippet"
    return "title_only"


def _external_sentiment_for_ticker(
    article: Article,
    ticker: str,
) -> tuple[str | None, float | None]:
    provider = article.metadata.get("external_sentiment_provider")
    if provider == "alpha_vantage_news":
        for entry in article.metadata.get("ticker_sentiment") or ():
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("ticker") or "").upper() != ticker:
                continue
            return (
                "alpha_vantage_news",
                _optional_float(entry.get("ticker_sentiment_score")),
            )
    return (
        str(provider) if provider else None,
        _optional_float(article.metadata.get("external_sentiment")),
    )


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
