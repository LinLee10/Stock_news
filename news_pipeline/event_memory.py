"""Durable article, event, and sentiment memory records."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Mapping, Sequence

from .article_fetch import ArticleFetchSummary
from .article_types import classify_article_type
from .dedup import DedupeCluster
from .models import Article
from .sentiment import analyze_sentiment
from .ticker_matching import assess_ticker_matches
from .tickers import load_tracked_tickers


EVENT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_EVENT_MEMORY_LOOKBACK_DAYS = 3
EVENT_IDENTITY_METHODS = (
    "exact_url_repeat",
    "fuzzy_event_repeat",
    "likely_new_event",
)
EVENT_TITLE_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
EVENT_TITLE_FILLER_PHRASES = (
    "breaking news",
    "here is why",
    "heres why",
    "investors need to know",
    "stock market today",
    "stock update",
    "what happened",
    "what to know",
    "why it matters",
)
EVENT_TITLE_FILLER_WORDS = {
    "company",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "limited",
    "ltd",
    "plc",
    "nv",
    "news",
    "shares",
    "stock",
    "today",
    "update",
}
EVENT_TOKEN_ALIASES = {
    "announced": "announce",
    "announces": "announce",
    "announcement": "announce",
    "introduced": "launch",
    "introduces": "launch",
    "launched": "launch",
    "launches": "launch",
    "unveiled": "launch",
    "unveils": "launch",
    "expanded": "expand",
    "expands": "expand",
    "reported": "report",
    "reports": "report",
    "raised": "raise",
    "raises": "raise",
    "cutting": "cut",
    "cuts": "cut",
}


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
    event_title: str
    normalized_event_title: str
    published_date_bucket: str | None
    event_identity_fingerprint: str
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
    prior_runs: tuple[Mapping[str, object], ...] = ()
    lookback_days: int = DEFAULT_EVENT_MEMORY_LOOKBACK_DAYS


@dataclass(frozen=True)
class EventMemoryComparison:
    prior_run_available: bool
    history_status: str
    prior_run_id: str | None
    prior_run_date: str | None
    event_memory_lookback_days: int
    prior_runs_considered: tuple[Mapping[str, object], ...]
    prior_event_records_considered: int
    new_events_since_prior_run: int
    repeated_events_from_prior_run: int
    exact_repeated_events_from_prior_run: int
    fuzzy_repeated_events_from_prior_run: int
    event_identity_method_counts: Mapping[str, int]
    event_similarity_threshold: float
    event_identity_matches: tuple[Mapping[str, object], ...]
    sentiment_change_since_prior_run: Mapping[str, Mapping[str, float]]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "exact_url_repeat": self.exact_repeated_events_from_prior_run,
                "fuzzy_event_repeat": self.fuzzy_repeated_events_from_prior_run,
                "likely_new_event": self.new_events_since_prior_run,
            }
        )
        return payload


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
            normalized_title = normalize_event_title(
                article.title,
                ticker=match.ticker,
                company=ticker.company_name,
            )
            date_bucket = event_date_bucket(
                article.published_at,
                fallback_date=run_date,
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
                    event_title=article.title,
                    normalized_event_title=normalized_title,
                    published_date_bucket=date_bucket,
                    event_identity_fingerprint=event_identity_fingerprint(
                        ticker=match.ticker,
                        normalized_title=normalized_title,
                        event_type=str(
                            article.metadata.get("filing_event_type")
                            or article.metadata.get("event_type")
                            or classification.primary_type
                        ),
                        article_type=classification.primary_type,
                        company=ticker.company_name,
                        source_family=str(
                            article.metadata.get("source_family") or "unknown"
                        ),
                        published_date_bucket=date_bucket,
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
    lookback_days: int = DEFAULT_EVENT_MEMORY_LOOKBACK_DAYS,
) -> PriorEventMemorySnapshot:
    directory = Path(runs_dir)
    bounded_lookback = max(1, int(lookback_days))
    if not directory.is_dir():
        return PriorEventMemorySnapshot(
            available=False,
            lookback_days=bounded_lookback,
        )
    current_date = _parse_calendar_date(run_date)
    if current_date is None:
        return PriorEventMemorySnapshot(
            available=False,
            lookback_days=bounded_lookback,
        )
    candidates = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_dir()
            and (
                candidate_date := _parse_calendar_date(child.name)
            )
            is not None
            and 0 < (current_date - candidate_date).days <= bounded_lookback
        ),
        key=lambda child: child.name,
        reverse=True,
    )
    prior_runs: list[Mapping[str, object]] = []
    all_records: list[Mapping[str, object]] = []
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
            prior_run: dict[str, object] = {
                "run_id": str(run["run_id"]),
                "run_date": str(run["run_date"] or candidate.name),
            }
            tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "event_memory" in tables:
                columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(event_memory)"
                    ).fetchall()
                }
                event_title_expression = (
                    "NULLIF(em.event_title, '')"
                    if "event_title" in columns
                    else "NULL"
                )
                records = tuple(
                    {
                        **dict(row),
                        "_prior_run_id": str(prior_run["run_id"]),
                        "_prior_run_date": str(prior_run["run_date"]),
                        "_prior_memory_basis": "event_memory",
                    }
                    for row in connection.execute(
                        f"""
                        SELECT
                            em.*,
                            COALESCE(
                                {event_title_expression},
                                a.title,
                                em.event_summary
                            ) AS comparison_title
                        FROM event_memory AS em
                        LEFT JOIN articles AS a
                            ON a.article_id = em.article_id
                        WHERE em.run_id = ?
                        ORDER BY em.id ASC
                        """,
                        (str(run["run_id"]),),
                    ).fetchall()
                )
                memory_basis = "event_memory"
            elif {
                "run_articles",
                "ticker_mentions",
            }.issubset(tables):
                records = _load_legacy_prior_records(
                    connection,
                    run_id=str(prior_run["run_id"]),
                    run_date=str(prior_run["run_date"]),
                    tables=tables,
                )
                memory_basis = "legacy_run_tables"
            else:
                records = ()
                memory_basis = "unavailable"
            prior_run.update(
                {
                    "event_memory_basis": memory_basis,
                    "event_record_count": len(records),
                }
            )
            prior_runs.append(prior_run)
            all_records.extend(records)
        except sqlite3.Error:
            continue
        finally:
            if connection is not None:
                connection.close()
    if not prior_runs:
        return PriorEventMemorySnapshot(
            available=False,
            lookback_days=bounded_lookback,
        )
    return PriorEventMemorySnapshot(
        available=True,
        run_id=prior_runs[0]["run_id"],
        run_date=prior_runs[0]["run_date"],
        records=tuple(all_records),
        prior_runs=tuple(prior_runs),
        lookback_days=bounded_lookback,
    )


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
    *,
    lookback_days: int | None = None,
) -> EventMemoryComparison:
    bounded_lookback = max(
        1,
        int(
            prior_snapshot.lookback_days
            if lookback_days is None
            else lookback_days
        ),
    )
    empty_method_counts = {method: 0 for method in EVENT_IDENTITY_METHODS}
    if not prior_snapshot.available:
        return EventMemoryComparison(
            prior_run_available=False,
            history_status="history_building",
            prior_run_id=None,
            prior_run_date=None,
            event_memory_lookback_days=bounded_lookback,
            prior_runs_considered=(),
            prior_event_records_considered=0,
            new_events_since_prior_run=0,
            repeated_events_from_prior_run=0,
            exact_repeated_events_from_prior_run=0,
            fuzzy_repeated_events_from_prior_run=0,
            event_identity_method_counts=empty_method_counts,
            event_similarity_threshold=EVENT_SIMILARITY_THRESHOLD,
            event_identity_matches=(),
            sentiment_change_since_prior_run={},
        )

    prior_records = tuple(prior_snapshot.records)
    prior_runs = prior_snapshot.prior_runs or (
        (
            {
                "run_id": prior_snapshot.run_id or "",
                "run_date": prior_snapshot.run_date or "",
            },
        )
        if prior_snapshot.run_id or prior_snapshot.run_date
        else ()
    )
    current_rows = [record.as_dict() for record in current_records]
    matched_prior_indexes: set[int] = set()
    matches_by_current_index: dict[int, tuple[str, int, float]] = {}
    method_counts: Counter[str] = Counter()
    for current_index, current in enumerate(current_rows):
        exact_index = _exact_prior_match_index(
            current,
            prior_records,
            matched_prior_indexes,
        )
        if exact_index is not None:
            matched_prior_indexes.add(exact_index)
            matches_by_current_index[current_index] = (
                "exact_url_repeat",
                exact_index,
                1.0,
            )

    for current_index, current in enumerate(current_rows):
        if current_index in matches_by_current_index:
            continue
        fuzzy_index, similarity = _fuzzy_prior_match(
            current,
            prior_records,
            matched_prior_indexes,
            lookback_days=bounded_lookback,
        )
        if fuzzy_index is not None:
            matched_prior_indexes.add(fuzzy_index)
            matches_by_current_index[current_index] = (
                "fuzzy_event_repeat",
                fuzzy_index,
                similarity,
            )

    identity_matches: list[Mapping[str, object]] = []
    for current_index, current in enumerate(current_rows):
        match = matches_by_current_index.get(current_index)
        if match is None:
            category = "likely_new_event"
            prior = None
            similarity = 0.0
        else:
            category, prior_index, similarity = match
            prior = prior_records[prior_index]
        method_counts[category] += 1
        identity_matches.append(
            _identity_match_row(
                current,
                prior,
                category=category,
                similarity=similarity,
            )
        )

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
        event_memory_lookback_days=bounded_lookback,
        prior_runs_considered=tuple(prior_runs),
        prior_event_records_considered=len(prior_records),
        new_events_since_prior_run=method_counts["likely_new_event"],
        repeated_events_from_prior_run=(
            method_counts["exact_url_repeat"]
            + method_counts["fuzzy_event_repeat"]
        ),
        exact_repeated_events_from_prior_run=method_counts[
            "exact_url_repeat"
        ],
        fuzzy_repeated_events_from_prior_run=method_counts[
            "fuzzy_event_repeat"
        ],
        event_identity_method_counts={
            method: method_counts[method]
            for method in EVENT_IDENTITY_METHODS
        },
        event_similarity_threshold=EVENT_SIMILARITY_THRESHOLD,
        event_identity_matches=tuple(identity_matches),
        sentiment_change_since_prior_run=changes,
    )


def normalize_event_title(
    title: str,
    *,
    ticker: str = "",
    company: str = "",
) -> str:
    normalized = title.casefold()
    for phrase in EVENT_TITLE_FILLER_PHRASES:
        normalized = normalized.replace(phrase, " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    noise_tokens = {
        token
        for token in re.sub(
            r"[^a-z0-9]+",
            " ",
            f"{ticker} {company}".casefold(),
        ).split()
        if token
    }
    tokens = []
    for token in normalized.split():
        if (
            token in EVENT_TITLE_STOP_WORDS
            or token in EVENT_TITLE_FILLER_WORDS
            or token in noise_tokens
        ):
            continue
        tokens.append(EVENT_TOKEN_ALIASES.get(token, token))
    return " ".join(tokens)


def event_title_similarity(
    left: str,
    right: str,
    *,
    ticker: str = "",
    company: str = "",
) -> float:
    left_normalized = normalize_event_title(
        left,
        ticker=ticker,
        company=company,
    )
    right_normalized = normalize_event_title(
        right,
        ticker=ticker,
        company=company,
    )
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    sequence_score = SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    intersection = len(left_tokens & right_tokens)
    if not intersection:
        return round(sequence_score, 4)
    union_score = intersection / len(left_tokens | right_tokens)
    containment_score = intersection / min(len(left_tokens), len(right_tokens))
    token_score = (union_score + containment_score) / 2
    return round(max(sequence_score, token_score), 4)


def event_date_bucket(
    published_at: str | None,
    *,
    fallback_date: str | None = None,
) -> str | None:
    for value in (published_at, fallback_date):
        if not value:
            continue
        try:
            return datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            ).date().isoformat()
        except ValueError:
            match = re.match(r"^\d{4}-\d{2}-\d{2}", str(value))
            if match:
                return match.group(0)
    return None


def event_identity_fingerprint(
    *,
    ticker: str,
    normalized_title: str,
    event_type: str,
    article_type: str,
    company: str,
    source_family: str,
    published_date_bucket: str | None,
) -> str:
    material = "|".join(
        (
            ticker.strip().upper(),
            normalized_title,
            _normalize_identity_text(event_type),
            _normalize_identity_text(article_type),
            _normalize_identity_text(company),
            _normalize_identity_text(source_family),
            published_date_bucket or "",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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


def _exact_prior_match_index(
    current: Mapping[str, object],
    prior_records: Sequence[Mapping[str, object]],
    matched_prior_indexes: set[int],
) -> int | None:
    current_ticker = str(current.get("ticker") or "").upper()
    current_url = str(current.get("canonical_url") or "")
    if not current_ticker or not current_url:
        return None
    for index, prior in enumerate(prior_records):
        if index in matched_prior_indexes:
            continue
        if (
            str(prior.get("ticker") or "").upper() == current_ticker
            and str(prior.get("canonical_url") or "") == current_url
        ):
            return index
    return None


def _fuzzy_prior_match(
    current: Mapping[str, object],
    prior_records: Sequence[Mapping[str, object]],
    matched_prior_indexes: set[int],
    *,
    lookback_days: int,
) -> tuple[int | None, float]:
    current_ticker = str(current.get("ticker") or "").upper()
    current_title = _record_title(current)
    current_date = _record_date_bucket(current)
    if not current_ticker or not current_title or current_date is None:
        return None, 0.0

    current_company = str(current.get("company") or "")
    current_fingerprint = _record_fingerprint(current)
    candidates: list[tuple[float, int, int]] = []
    for index, prior in enumerate(prior_records):
        if index in matched_prior_indexes:
            continue
        if str(prior.get("ticker") or "").upper() != current_ticker:
            continue
        prior_date = _record_date_bucket(prior)
        if prior_date is None:
            continue
        if abs((current_date - prior_date).days) > lookback_days:
            continue

        prior_fingerprint = _record_fingerprint(prior)
        if current_fingerprint and current_fingerprint == prior_fingerprint:
            similarity = 1.0
        else:
            similarity = event_title_similarity(
                current_title,
                _record_title(prior),
                ticker=current_ticker,
                company=current_company or str(prior.get("company") or ""),
            )
        if similarity < EVENT_SIMILARITY_THRESHOLD:
            continue
        metadata_agreement = sum(
            _normalized_record_value(current, field)
            == _normalized_record_value(prior, field)
            for field in (
                "event_type",
                "article_type",
                "company",
                "source_family",
            )
        )
        candidates.append((similarity, metadata_agreement, index))

    if not candidates:
        return None, 0.0
    similarity, _metadata_agreement, index = max(
        candidates,
        key=lambda item: (item[0], item[1], -item[2]),
    )
    return index, round(similarity, 4)


def _identity_match_row(
    current: Mapping[str, object],
    prior: Mapping[str, object] | None,
    *,
    category: str,
    similarity: float,
) -> Mapping[str, object]:
    return {
        "category": category,
        "ticker": str(current.get("ticker") or "").upper(),
        "current_article_id": str(current.get("article_id") or ""),
        "current_canonical_url": str(current.get("canonical_url") or ""),
        "current_event_identity_fingerprint": _record_fingerprint(current),
        "prior_article_id": (
            str(prior.get("article_id") or "") if prior is not None else None
        ),
        "prior_canonical_url": (
            str(prior.get("canonical_url") or "")
            if prior is not None
            else None
        ),
        "prior_event_identity_fingerprint": (
            _record_fingerprint(prior) if prior is not None else None
        ),
        "prior_run_id": (
            str(prior.get("_prior_run_id") or prior.get("run_id") or "")
            if prior is not None
            else None
        ),
        "prior_run_date": (
            str(prior.get("_prior_run_date") or prior.get("run_date") or "")
            if prior is not None
            else None
        ),
        "title_similarity": round(similarity, 4),
    }


def _record_title(record: Mapping[str, object]) -> str:
    return str(
        record.get("event_title")
        or record.get("comparison_title")
        or record.get("title")
        or record.get("event_summary")
        or ""
    )


def _record_date_bucket(record: Mapping[str, object]):
    bucket = event_date_bucket(
        str(record.get("published_date_bucket") or "") or None,
        fallback_date=(
            str(record.get("published_at") or "")
            or str(record.get("run_date") or "")
            or None
        ),
    )
    if bucket is None:
        return None
    return datetime.fromisoformat(bucket).date()


def _record_fingerprint(record: Mapping[str, object]) -> str:
    existing = str(record.get("event_identity_fingerprint") or "")
    if existing:
        return existing
    ticker = str(record.get("ticker") or "").upper()
    company = str(record.get("company") or "")
    normalized_title = str(record.get("normalized_event_title") or "")
    if not normalized_title:
        normalized_title = normalize_event_title(
            _record_title(record),
            ticker=ticker,
            company=company,
        )
    return event_identity_fingerprint(
        ticker=ticker,
        normalized_title=normalized_title,
        event_type=str(record.get("event_type") or ""),
        article_type=str(record.get("article_type") or ""),
        company=company,
        source_family=str(record.get("source_family") or ""),
        published_date_bucket=(
            event_date_bucket(
                str(record.get("published_date_bucket") or "") or None,
                fallback_date=(
                    str(record.get("published_at") or "")
                    or str(record.get("run_date") or "")
                    or None
                ),
            )
        ),
    )


def _normalized_record_value(
    record: Mapping[str, object],
    field: str,
) -> str:
    return _normalize_identity_text(str(record.get(field) or ""))


def _normalize_identity_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()


def _parse_calendar_date(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        match = re.match(r"^\d{4}-\d{2}-\d{2}", value)
        if match:
            return datetime.fromisoformat(match.group(0)).date()
    return None


def _load_legacy_prior_records(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    run_date: str,
    tables: set[str],
) -> tuple[Mapping[str, object], ...]:
    source_provider_expression = (
        """
        COALESCE(
            (
                SELECT source.provider
                FROM article_sources AS source
                WHERE source.run_id = ra.run_id
                  AND source.article_id = ra.article_id
                ORDER BY source.id ASC
                LIMIT 1
            ),
            'unknown'
        )
        """
        if "article_sources" in tables
        else "'unknown'"
    )
    sentiment_expression = (
        """
        COALESCE(
            (
                SELECT AVG(sentiment.score)
                FROM sentiment_results AS sentiment
                WHERE sentiment.run_id = ra.run_id
                  AND sentiment.article_id = ra.article_id
            ),
            0.0
        )
        """
        if "sentiment_results" in tables
        else "0.0"
    )
    rows = connection.execute(
        f"""
        SELECT
            ra.article_id,
            ra.canonical_url,
            ra.published_at,
            mention.ticker,
            COALESCE(mention.company_name, mention.ticker) AS company,
            {source_provider_expression} AS source_provider,
            'legacy_run_memory' AS source_family,
            'unknown' AS article_type,
            mention.confidence AS ticker_match_confidence,
            mention.basis AS extraction_basis,
            'legacy_unknown' AS extraction_quality_grade,
            {sentiment_expression} AS internal_sentiment,
            NULL AS external_sentiment_provider,
            NULL AS external_sentiment,
            'unknown' AS event_type,
            ra.title AS event_title,
            '' AS normalized_event_title,
            substr(COALESCE(ra.published_at, ?), 1, 10)
                AS published_date_bucket,
            '' AS event_identity_fingerprint,
            COALESCE(ra.snippet, ra.title) AS event_summary,
            ra.run_id,
            ? AS run_date,
            ra.title AS comparison_title
        FROM run_articles AS ra
        JOIN ticker_mentions AS mention
          ON mention.run_id = ra.run_id
         AND mention.article_id = ra.article_id
        WHERE ra.run_id = ?
        ORDER BY ra.article_id, mention.ticker
        """,
        (run_date, run_date, run_id),
    ).fetchall()
    return tuple(
        {
            **dict(row),
            "_prior_run_id": run_id,
            "_prior_run_date": run_date,
            "_prior_memory_basis": "legacy_run_tables",
        }
        for row in rows
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
