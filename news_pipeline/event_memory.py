"""Durable article, event, and sentiment memory records."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class PriorEventMemorySnapshot:
    available: bool
    run_id: str | None = None
    run_date: str | None = None
    records: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class EventMemoryComparison:
    prior_run_available: bool
    history_status: str
    prior_run_id: str | None
    prior_run_date: str | None
    new_events_since_prior_run: int
    repeated_events_from_prior_run: int
    sentiment_change_since_prior_run: Mapping[str, Mapping[str, float]]

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


def load_latest_prior_event_memory(
    *,
    runs_dir: str | Path,
    run_date: str,
) -> PriorEventMemorySnapshot:
    directory = Path(runs_dir)
    if not directory.is_dir():
        return PriorEventMemorySnapshot(available=False)
    candidates = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_dir() and child.name < run_date
        ),
        key=lambda child: child.name,
        reverse=True,
    )
    for candidate in candidates:
        database_path = candidate / "news_pipeline.sqlite3"
        if not database_path.is_file():
            continue
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                f"{database_path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            run = connection.execute(
                """
                SELECT run_id, run_date
                FROM runs
                WHERE status = 'completed'
                ORDER BY COALESCE(finished_at, started_at) DESC
                LIMIT 1
                """
            ).fetchone()
            if run is None:
                continue
            table = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'event_memory'
                """
            ).fetchone()
            records = (
                tuple(
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT *
                        FROM event_memory
                        WHERE run_id = ?
                        ORDER BY id ASC
                        """,
                        (str(run["run_id"]),),
                    ).fetchall()
                )
                if table
                else ()
            )
            return PriorEventMemorySnapshot(
                available=True,
                run_id=str(run["run_id"]),
                run_date=str(run["run_date"] or candidate.name),
                records=records,
            )
        except sqlite3.Error:
            continue
        finally:
            if connection is not None:
                connection.close()
    return PriorEventMemorySnapshot(available=False)


def alpha_vantage_selection_history(
    snapshot: PriorEventMemorySnapshot,
    *,
    minimum_articles_per_ticker: int,
) -> dict[str, object]:
    if not snapshot.available:
        return {
            "benchmark_coverage_counts": {},
            "google_dominated_tickers": [],
            "weak_coverage_tickers": [],
        }
    benchmark_counts: Counter[str] = Counter()
    google_counts: Counter[str] = Counter()
    direct_counts: Counter[str] = Counter()
    for record in snapshot.records:
        ticker = str(record.get("ticker") or "").upper()
        if not ticker:
            continue
        if record.get("external_sentiment_provider") == "alpha_vantage_news":
            benchmark_counts[ticker] += 1
        if record.get("source_family") == "google_news_backstop":
            google_counts[ticker] += 1
        else:
            direct_counts[ticker] += 1
    tickers = set(google_counts) | set(direct_counts) | set(benchmark_counts)
    minimum = max(1, int(minimum_articles_per_ticker))
    return {
        "benchmark_coverage_counts": dict(sorted(benchmark_counts.items())),
        "google_dominated_tickers": sorted(
            ticker
            for ticker in tickers
            if google_counts[ticker] > direct_counts[ticker]
        ),
        "weak_coverage_tickers": sorted(
            ticker
            for ticker in tickers
            if google_counts[ticker] + direct_counts[ticker] < minimum
        ),
    }


def compare_event_memory(
    current_records: Sequence[EventMemoryRecord],
    prior_snapshot: PriorEventMemorySnapshot,
) -> EventMemoryComparison:
    if not prior_snapshot.available:
        return EventMemoryComparison(
            prior_run_available=False,
            history_status="history_building",
            prior_run_id=None,
            prior_run_date=None,
            new_events_since_prior_run=0,
            repeated_events_from_prior_run=0,
            sentiment_change_since_prior_run={},
        )

    prior_keys = {
        _event_identity(record)
        for record in prior_snapshot.records
    }
    current_keys = {
        _event_identity(record.as_dict())
        for record in current_records
    }
    current_sentiment = _average_sentiment_by_ticker(
        record.as_dict() for record in current_records
    )
    prior_sentiment = _average_sentiment_by_ticker(prior_snapshot.records)
    changes = {}
    for ticker in sorted(set(current_sentiment) & set(prior_sentiment)):
        change = round(current_sentiment[ticker] - prior_sentiment[ticker], 4)
        if abs(change) < 0.0001:
            continue
        changes[ticker] = {
            "prior": round(prior_sentiment[ticker], 4),
            "current": round(current_sentiment[ticker], 4),
            "change": change,
        }
    return EventMemoryComparison(
        prior_run_available=True,
        history_status="compared_to_prior_run",
        prior_run_id=prior_snapshot.run_id,
        prior_run_date=prior_snapshot.run_date,
        new_events_since_prior_run=len(current_keys - prior_keys),
        repeated_events_from_prior_run=len(current_keys & prior_keys),
        sentiment_change_since_prior_run=changes,
    )


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


def _event_identity(record: Mapping[str, object]) -> tuple[str, str]:
    return (
        str(record.get("ticker") or "").upper(),
        str(record.get("canonical_url") or ""),
    )


def _average_sentiment_by_ticker(
    records: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        ticker = str(record.get("ticker") or "").upper()
        sentiment = _optional_float(record.get("internal_sentiment"))
        if ticker and sentiment is not None:
            grouped[ticker].append(sentiment)
    return {
        ticker: sum(values) / len(values)
        for ticker, values in grouped.items()
        if values
    }


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
