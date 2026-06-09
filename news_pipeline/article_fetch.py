"""Opt-in capped article page fetching and extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urljoin, urlparse

from .article_types import ARTICLE_TYPE_SERIOUSNESS, classify_article_type
from .dedup import DedupeCluster
from .extract import extract_article, extract_url_metadata, extraction_dependency_status
from .models import Article
from .recency import article_recency
from .source_quality import (
    DEFAULT_MIN_SOURCE_QUALITY_TIER,
    TIER_4_EXCLUDE_BY_DEFAULT,
    assess_article_source,
    source_quality_sort_key,
)
from .ticker_matching import assess_ticker_matches
from .tickers import load_portfolio, match_tickers, ticker_lookup


DEFAULT_MAX_ARTICLE_FETCHES = 75
DEFAULT_MAX_FETCHES_PER_TICKER = 5
DEFAULT_FETCH_TIMEOUT_SECONDS = 8.0
DEFAULT_ARTICLE_FETCH_USER_AGENT = "StonkNewsPipeline/0.1 (+local dry-run article extraction)"
URL_CLASS_GOOGLE_NEWS_WRAPPER = "google_news_wrapper"
URL_CLASS_DIRECT_PUBLISHER = "direct_publisher_url"
URL_CLASS_UNSUPPORTED = "unsupported_or_unknown"
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


@dataclass(frozen=True)
class ArticleExtractionRecord:
    article_id: str | None
    canonical_url: str
    title: str
    extraction_status: str
    extraction_basis: str
    error_class: str | None
    failure_reasons: tuple[str, ...]
    url_classification: str
    requested_url: str
    fetch_url: str | None
    resolved_url: str | None
    resolution_status: str
    final_url: str | None
    latency_ms: int
    content_type: str | None
    content_length: int
    text_hash: str | None
    extracted_preview: str | None
    extractor: str | None
    extraction_method_used: str | None
    extraction_failure_reason: str | None
    fetched: bool
    tickers: tuple[str, ...]
    source_publisher: str = ""
    source_provider: str = ""
    queue_score: float = 0.0
    queue_reasons: tuple[str, ...] = ()
    status_code: int | None = None
    canonical_link: str | None = None
    og_url: str | None = None
    extractor_methods_tried: tuple[str, ...] = ()
    accepted_method: str | None = None
    accepted_text_length: int = 0
    extraction_quality_score: float = 0.0
    extraction_quality_grade: str = "title_only"
    extraction_quality_reasons: tuple[str, ...] = ()
    accepted_as_full_text: bool = False
    cache_hit: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "article_id": self.article_id,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "extraction_status": self.extraction_status,
            "extraction_basis": self.extraction_basis,
            "error_class": self.error_class,
            "failure_reasons": list(self.failure_reasons),
            "url_classification": self.url_classification,
            "requested_url": self.requested_url,
            "fetch_url": self.fetch_url,
            "resolved_url": self.resolved_url,
            "resolution_status": self.resolution_status,
            "final_url": self.final_url,
            "latency_ms": self.latency_ms,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "text_hash": self.text_hash,
            "extracted_preview": self.extracted_preview,
            "extractor": self.extractor,
            "extraction_method_used": self.extraction_method_used,
            "extraction_failure_reason": self.extraction_failure_reason,
            "fetched": self.fetched,
            "tickers": list(self.tickers),
            "source_publisher": self.source_publisher,
            "source_provider": self.source_provider,
            "queue_score": self.queue_score,
            "queue_reasons": list(self.queue_reasons),
            "status_code": self.status_code,
            "canonical_link": self.canonical_link,
            "og_url": self.og_url,
            "extractor_methods_tried": list(self.extractor_methods_tried),
            "accepted_method": self.accepted_method,
            "accepted_text_length": self.accepted_text_length,
            "extraction_quality_score": self.extraction_quality_score,
            "extraction_quality_grade": self.extraction_quality_grade,
            "extraction_quality_reasons": list(self.extraction_quality_reasons),
            "accepted_as_full_text": self.accepted_as_full_text,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class ExtractionQueueItem:
    cluster: DedupeCluster
    article: Article
    tickers: tuple[str, ...]
    primary_ticker: str | None
    score: float
    score_reasons: tuple[str, ...]
    eligible: bool
    skip_reason: str | None = None


@dataclass(frozen=True)
class ArticleFetchSummary:
    enabled: bool
    attempted_fetches: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    max_article_fetches: int = DEFAULT_MAX_ARTICLE_FETCHES
    max_fetches_per_ticker: int = DEFAULT_MAX_FETCHES_PER_TICKER
    fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS
    reason: str | None = None
    records: tuple[ArticleExtractionRecord, ...] = ()
    extractor_diagnostics: dict[str, bool] | None = None
    extraction_queue_size: int = 0
    extraction_selected_count: int = 0
    extraction_skipped_count: int = 0
    extraction_skipped_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def basis_counts(self) -> dict[str, int]:
        counts = {"full_text": 0, "snippet": 0, "title": 0}
        for record in self.records:
            if not (record.fetched or record.cache_hit):
                continue
            if record.extraction_basis in counts:
                counts[record.extraction_basis] += 1
        return counts

    @property
    def publisher_article_fetches(self) -> int:
        return sum(1 for record in self.records if record.fetched and record.fetch_url and not _is_google_news_url(record.fetch_url))

    @property
    def google_news_wrappers_skipped(self) -> int:
        return self.google_wrappers_unresolved

    @property
    def google_news_wrappers_resolved(self) -> int:
        return sum(1 for record in self.records if record.resolution_status.startswith("resolved_to_publisher"))

    @property
    def snippet_fallbacks(self) -> int:
        return sum(1 for record in self.records if "snippet_fallback" in record.failure_reasons)

    @property
    def title_fallbacks(self) -> int:
        return sum(1 for record in self.records if "title_fallback" in record.failure_reasons)

    @property
    def failure_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            for reason in record.failure_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    @property
    def extraction_method_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            for method in record.extractor_methods_tried:
                counts[method] = counts.get(method, 0) + 1
        return counts

    @property
    def extraction_method_success_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if not record.accepted_as_full_text or not record.accepted_method:
                continue
            counts[record.accepted_method] = counts.get(record.accepted_method, 0) + 1
        return counts

    @property
    def extraction_quality_grade_counts(self) -> dict[str, int]:
        counts = {
            "strong_full_text": 0,
            "usable_full_text": 0,
            "weak_text": 0,
            "snippet_only": 0,
            "title_only": 0,
            "blocked_or_shell": 0,
        }
        for record in self.records:
            grade = record.extraction_quality_grade
            counts[grade] = counts.get(grade, 0) + 1
        return counts

    @property
    def direct_publisher_candidates(self) -> int:
        return sum(1 for record in self.records if record.url_classification == URL_CLASS_DIRECT_PUBLISHER)

    @property
    def google_wrapper_candidates(self) -> int:
        return sum(1 for record in self.records if record.url_classification == URL_CLASS_GOOGLE_NEWS_WRAPPER)

    @property
    def google_wrappers_unresolved(self) -> int:
        return sum(
            1
            for record in self.records
            if "google_wrapper_unresolved" in record.failure_reasons
            or "google_news_unresolved" in record.failure_reasons
        )

    @property
    def publisher_profiles(self) -> tuple[dict[str, object], ...]:
        profiles: dict[tuple[str, str], dict[str, object]] = {}
        for record in self.records:
            publisher = record.source_publisher or "unknown"
            domain = urlparse(record.final_url or record.fetch_url or record.canonical_url).netloc.lower()
            key = (publisher, domain)
            profile = profiles.setdefault(
                key,
                {
                    "publisher": publisher,
                    "domain": domain,
                    "fetch_allowed": record.url_classification != URL_CLASS_UNSUPPORTED,
                    "paywall_likely": False,
                    "direct_url_available": record.url_classification == URL_CLASS_DIRECT_PUBLISHER,
                    "historical_success_count": 0,
                    "historical_failure_count": 0,
                    "last_failure_reason": None,
                    "current_run_attempts": 0,
                    "current_run_successes": 0,
                    "current_run_success_rate": 0.0,
                },
            )
            if record.fetched or record.cache_hit:
                profile["current_run_attempts"] = int(profile["current_run_attempts"]) + 1
            if record.accepted_as_full_text:
                profile["current_run_successes"] = int(profile["current_run_successes"]) + 1
            elif record.extraction_failure_reason:
                profile["last_failure_reason"] = record.extraction_failure_reason
            if "paywall_or_login" in record.failure_reasons:
                profile["paywall_likely"] = True
        for profile in profiles.values():
            attempts = int(profile["current_run_attempts"])
            profile["current_run_success_rate"] = (
                round(int(profile["current_run_successes"]) / attempts, 4) if attempts else 0.0
            )
        return tuple(profiles[key] for key in sorted(profiles))

    @property
    def top_unresolved_wrapper_publishers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.url_classification != URL_CLASS_GOOGLE_NEWS_WRAPPER:
                continue
            if record.resolution_status not in {"google_news_unresolved", "resolve_error"}:
                continue
            publisher = record.source_publisher or "unknown"
            counts[publisher] = counts.get(publisher, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10])

    @property
    def extraction_failure_reason(self) -> str | None:
        reasons = self.failure_reason_counts
        if not reasons:
            return None
        return sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]

    @property
    def extraction_success_rate(self) -> float:
        if not self.extraction_selected_count:
            return 0.0
        return round(self.successful_extractions / self.extraction_selected_count, 4)

    @property
    def extraction_success_rate_by_publisher(self) -> dict[str, float]:
        return _success_rates(self.records, "source_publisher")

    @property
    def extraction_success_rate_by_source_provider(self) -> dict[str, float]:
        return _success_rates(self.records, "source_provider")

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "attempted_fetches": self.attempted_fetches,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "max_article_fetches": self.max_article_fetches,
            "max_fetches_per_ticker": self.max_fetches_per_ticker,
            "fetch_timeout_seconds": self.fetch_timeout_seconds,
            "reason": self.reason,
            "sentiment_basis_counts": self.basis_counts,
            "publisher_article_fetches": self.publisher_article_fetches,
            "google_news_wrappers_skipped": self.google_news_wrappers_skipped,
            "google_news_wrappers_resolved": self.google_news_wrappers_resolved,
            "snippet_fallbacks": self.snippet_fallbacks,
            "title_fallbacks": self.title_fallbacks,
            "top_extraction_failure_reasons": self.failure_reason_counts,
            "extraction_method_counts": self.extraction_method_counts,
            "extractor_method_counts": self.extraction_method_counts,
            "extractor_method_success_counts": self.extraction_method_success_counts,
            "extraction_failure_reason": self.extraction_failure_reason,
            "extractor_diagnostics": self.extractor_diagnostics or extraction_dependency_status(),
            "extraction_queue_size": self.extraction_queue_size,
            "extraction_selected_count": self.extraction_selected_count,
            "extraction_skipped_count": self.extraction_skipped_count,
            "extraction_skipped_reasons": dict(self.extraction_skipped_reasons),
            "extraction_success_rate": self.extraction_success_rate,
            "extraction_success_rate_by_publisher": self.extraction_success_rate_by_publisher,
            "extraction_success_rate_by_source_provider": self.extraction_success_rate_by_source_provider,
            "full_text_basis_count": self.basis_counts["full_text"],
            "snippet_basis_count": self.basis_counts["snippet"],
            "title_basis_count": self.basis_counts["title"],
            "extraction_attempted_count": self.attempted_fetches,
            "extraction_budget_unused_count": max(0, self.max_article_fetches - self.extraction_selected_count),
            "direct_publisher_candidates": self.direct_publisher_candidates,
            "google_wrapper_candidates": self.google_wrapper_candidates,
            "google_wrappers_unresolved": self.google_wrappers_unresolved,
            "full_text_accepted_count": self.successful_extractions,
            "usable_full_text_count": self.extraction_quality_grade_counts.get("usable_full_text", 0),
            "weak_text_count": self.extraction_quality_grade_counts.get("weak_text", 0),
            "snippet_fallback_count": self.snippet_fallbacks,
            "title_fallback_count": self.title_fallbacks,
            "blocked_or_shell_count": self.extraction_quality_grade_counts.get("blocked_or_shell", 0),
            "extraction_quality_grade_counts": self.extraction_quality_grade_counts,
            "publisher_success_rates": self.extraction_success_rate_by_publisher,
            "publisher_profiles": list(self.publisher_profiles),
            "top_failure_reasons": self.failure_reason_counts,
            "top_unresolved_wrapper_publishers": self.top_unresolved_wrapper_publishers,
            "records": [record.as_dict() for record in self.records],
        }


def fetch_top_cluster_articles(
    clusters: tuple[DedupeCluster, ...] | list[DedupeCluster],
    *,
    run_date: str,
    max_article_fetches: int = DEFAULT_MAX_ARTICLE_FETCHES,
    max_fetches_per_ticker: int = DEFAULT_MAX_FETCHES_PER_TICKER,
    fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_ARTICLE_FETCH_USER_AGENT,
    include_low_quality_sources: bool = False,
    min_source_quality_tier: int = DEFAULT_MIN_SOURCE_QUALITY_TIER,
    cache_path: str | Path | None = None,
) -> tuple[dict[str, Article], ArticleFetchSummary]:
    """Fetch at most one article page for selected top event clusters."""
    max_article_fetches = max(0, int(max_article_fetches))
    max_fetches_per_ticker = max(1, int(max_fetches_per_ticker))
    fetch_timeout_seconds = max(0.1, float(fetch_timeout_seconds))
    enriched: dict[str, Article] = {}
    records: list[ArticleExtractionRecord] = []
    ticker_counts: dict[str, int] = {}
    seen_urls: set[str] = set()
    queue = build_extraction_queue(
        tuple(clusters),
        run_date=run_date,
        include_low_quality_sources=include_low_quality_sources,
        min_source_quality_tier=min_source_quality_tier,
    )
    queue_diagnostics: dict[str, object] = {
        "selected_count": 0,
        "skipped_reasons": {},
    }
    extraction_cache = _load_extraction_cache(cache_path)

    for article, tickers, requested_url, fetch_url, url_classification, resolution_status, resolution_record, queue_item in _iter_fetch_candidates(
        queue,
        max_article_fetches=max_article_fetches,
        max_fetches_per_ticker=max_fetches_per_ticker,
        fetch_timeout_seconds=fetch_timeout_seconds,
        user_agent=user_agent,
        ticker_counts=ticker_counts,
        seen_urls=seen_urls,
        queue_diagnostics=queue_diagnostics,
    ):
        if resolution_record is not None:
            records.append(resolution_record)
            continue
        cached = _find_cached_extraction(extraction_cache, requested_url, fetch_url)
        if cached is not None:
            enriched_article, record = _cached_extraction(
                article,
                tickers=tickers,
                requested_url=requested_url,
                fetch_url=fetch_url,
                url_classification=url_classification,
                resolution_status=resolution_status,
                queue_item=queue_item,
                cached=cached,
            )
        else:
            enriched_article, record = _fetch_and_extract(
                article,
                tickers=tickers,
                requested_url=requested_url,
                fetch_url=fetch_url,
                url_classification=url_classification,
                resolution_status=resolution_status,
                timeout_seconds=fetch_timeout_seconds,
                user_agent=user_agent,
                queue_item=queue_item,
            )
            _cache_extraction(extraction_cache, record, enriched_article)
        records.append(record)
        if enriched_article is not None:
            enriched[article.canonical_url] = enriched_article
    _write_extraction_cache(cache_path, extraction_cache)

    successful = sum(1 for record in records if record.accepted_as_full_text)
    fetched = sum(1 for record in records if record.fetched)
    failed = sum(1 for record in records if record.fetched and not record.accepted_as_full_text)
    skipped_reasons = dict(queue_diagnostics["skipped_reasons"])
    skipped_count = sum(int(count) for count in skipped_reasons.values())
    return enriched, ArticleFetchSummary(
        enabled=True,
        attempted_fetches=fetched,
        successful_extractions=successful,
        failed_extractions=failed,
        max_article_fetches=max_article_fetches,
        max_fetches_per_ticker=max_fetches_per_ticker,
        fetch_timeout_seconds=fetch_timeout_seconds,
        records=tuple(records),
        extractor_diagnostics=extraction_dependency_status(),
        extraction_queue_size=len(queue),
        extraction_selected_count=int(queue_diagnostics["selected_count"]),
        extraction_skipped_count=skipped_count,
        extraction_skipped_reasons=skipped_reasons,
    )


def disabled_article_fetch_summary(
    *,
    reason: str,
    max_article_fetches: int = DEFAULT_MAX_ARTICLE_FETCHES,
    max_fetches_per_ticker: int = DEFAULT_MAX_FETCHES_PER_TICKER,
    fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
) -> ArticleFetchSummary:
    return ArticleFetchSummary(
        enabled=False,
        max_article_fetches=max(0, int(max_article_fetches)),
        max_fetches_per_ticker=max(1, int(max_fetches_per_ticker)),
        fetch_timeout_seconds=max(0.1, float(fetch_timeout_seconds)),
        reason=reason,
        extractor_diagnostics=extraction_dependency_status(),
    )


def build_extraction_queue(
    clusters: tuple[DedupeCluster, ...] | list[DedupeCluster],
    *,
    run_date: str,
    include_low_quality_sources: bool,
    min_source_quality_tier: int,
) -> tuple[ExtractionQueueItem, ...]:
    portfolio_symbols = {ticker.symbol for ticker in load_portfolio()}
    queue: list[ExtractionQueueItem] = []
    for cluster in clusters:
        for article in cluster.articles:
            quality = assess_article_source(article)
            matches = assess_ticker_matches(article)
            usable_matches = tuple(match for match in matches if not match.related)
            tickers = tuple(sorted(match.ticker for match in matches))
            primary = next((match.ticker for match in matches if match.primary), cluster.primary_ticker)
            recency = article_recency(
                run_date=run_date,
                published_at=article.published_at,
                collected_at=article.created_at,
                archive_context=bool(article.metadata.get("archive_context")),
            )
            classification = classify_article_type(article)
            direct = classify_article_url(article.canonical_url) == URL_CLASS_DIRECT_PUBLISHER
            confidence = max((match.confidence for match in usable_matches), default=0.0)
            reasons = (
                f"ticker_confidence={confidence:.2f}",
                f"source_tier={quality.tier}",
                f"recency={recency.recency_bucket}",
                f"article_type={classification.primary_type}",
                f"direct_publisher={str(direct).lower()}",
                f"cluster_primary={str(article.canonical_url == cluster.canonical_article.canonical_url).lower()}",
                f"portfolio_relevance={str(bool(set(tickers) & portfolio_symbols)).lower()}",
            )
            score = (
                confidence * 40
                + {1: 30, 2: 22, 3: 8, 4: -100}.get(quality.tier, 12)
                + {
                    "today_signal": 20,
                    "recent_pulse": 14,
                    "weekly_trend": 8,
                    "background_context": 2,
                }.get(recency.recency_bucket, 0)
                + ARTICLE_TYPE_SERIOUSNESS.get(classification.primary_type, 0)
                + (12 if direct else 0)
                + (10 if article.canonical_url == cluster.canonical_article.canonical_url else 0)
                + (6 if set(tickers) & portfolio_symbols else 3)
                + min(8, cluster.publisher_diversity + cluster.source_diversity)
            )
            skip_reason = None
            if not usable_matches:
                skip_reason = "no_ticker_specific_match"
            elif not include_low_quality_sources and (
                quality.tier >= 3
                or quality.tier > min(min_source_quality_tier, 2)
            ):
                skip_reason = "source_quality_excluded"
            elif not _is_fetchable_url(article.canonical_url):
                skip_reason = "unsupported_or_unknown"
            queue.append(
                ExtractionQueueItem(
                    cluster=cluster,
                    article=article,
                    tickers=tickers,
                    primary_ticker=primary,
                    score=round(score, 2),
                    score_reasons=reasons,
                    eligible=skip_reason is None,
                    skip_reason=skip_reason,
                )
            )
    return tuple(
        sorted(
            queue,
            key=lambda item: (
                item.eligible,
                item.score,
                item.article.canonical_url == item.cluster.canonical_article.canonical_url,
                item.article.canonical_url,
            ),
            reverse=True,
        )
    )


def _iter_fetch_candidates(
    queue: tuple[ExtractionQueueItem, ...],
    *,
    max_article_fetches: int,
    max_fetches_per_ticker: int,
    fetch_timeout_seconds: float,
    user_agent: str,
    ticker_counts: dict[str, int],
    seen_urls: set[str],
    queue_diagnostics: dict[str, object],
):
    attempted_fetches = 0
    selected_clusters: set[str] = set()
    skipped_reasons = queue_diagnostics["skipped_reasons"]
    for item in queue:
        if not item.eligible:
            _increment(skipped_reasons, item.skip_reason or "ineligible")
            continue
        cluster_key = item.cluster.cluster_id or item.cluster.canonical_article.canonical_url
        if cluster_key in selected_clusters:
            _increment(skipped_reasons, "lower_ranked_cluster_candidate")
            continue
        if attempted_fetches >= max_article_fetches:
            _increment(skipped_reasons, "max_article_fetches")
            continue
        budget_ticker = item.primary_ticker or (item.tickers[0] if item.tickers else None)
        if budget_ticker and ticker_counts.get(budget_ticker, 0) >= max_fetches_per_ticker:
            _increment(skipped_reasons, "max_fetches_per_ticker")
            continue
        candidate_article = item.article
        if candidate_article.canonical_url in seen_urls:
            _increment(skipped_reasons, "duplicate_url")
            continue
        seen_urls.add(candidate_article.canonical_url)
        article = item.cluster.canonical_article
        requested_url = candidate_article.canonical_url
        url_classification = classify_article_url(requested_url)
        resolution_status = "direct_publisher_url"
        fetch_url = requested_url
        if url_classification == URL_CLASS_GOOGLE_NEWS_WRAPPER:
            resolved = _resolve_google_news_url(
                requested_url,
                timeout_seconds=fetch_timeout_seconds,
                user_agent=user_agent,
            )
            if resolved.resolved_url is None:
                yield article, item.tickers, requested_url, None, url_classification, resolved.status, _skipped_record(
                    article,
                    tickers=item.tickers,
                    reason=resolved.error_class or "google_news_unresolved",
                    url_classification=url_classification,
                    requested_url=requested_url,
                    resolution_status=resolved.status,
                    final_url=resolved.final_url,
                    latency_ms=resolved.latency_ms,
                    queue_item=item,
                    canonical_link=resolved.canonical_url,
                    og_url=resolved.og_url,
                    extraction_quality_grade="blocked_or_shell",
                    extraction_quality_reasons=("blocked_or_shell", "google_wrapper_unresolved"),
                ), item
                continue
            fetch_url = resolved.resolved_url
            resolution_status = resolved.status
        attempted_fetches += 1
        queue_diagnostics["selected_count"] = attempted_fetches
        selected_clusters.add(cluster_key)
        if budget_ticker:
            ticker_counts[budget_ticker] = ticker_counts.get(budget_ticker, 0) + 1
        yield article, item.tickers, requested_url, fetch_url, url_classification, resolution_status, None, item


def _cluster_rank(cluster: DedupeCluster, run_date: str) -> tuple[int, int, int, int, int]:
    bucket_rank = {
        "today_signal": 4,
        "recent_pulse": 3,
        "weekly_trend": 2,
        "background_context": 1,
    }
    infos = [
        article_recency(
            run_date=run_date,
            published_at=article.published_at,
            collected_at=article.created_at,
            archive_context=bool(article.metadata.get("archive_context")),
        )
        for article in cluster.articles
    ]
    best_bucket = max((bucket_rank.get(info.recency_bucket, 0) for info in infos), default=0)
    return (
        best_bucket,
        cluster.source_count,
        cluster.publisher_count,
        len(_cluster_tickers(cluster)),
        len(cluster.articles),
    )


def _cluster_tickers(cluster: DedupeCluster) -> tuple[str, ...]:
    text = " ".join(
        part
        for article in cluster.articles
        for part in (article.title, article.snippet or "")
        if part
    )
    return tuple(sorted({ticker.symbol for ticker in match_tickers(text)}))


def _is_fetchable_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _absolute_http_url(value: object, base_url: str | None) -> str | None:
    if not value:
        return None
    absolute = urljoin(base_url or "", str(value).strip())
    return absolute if _is_fetchable_url(absolute) else None


def classify_article_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return URL_CLASS_UNSUPPORTED
    if _is_google_news_wrapper_url(url):
        return URL_CLASS_GOOGLE_NEWS_WRAPPER
    if parsed.netloc:
        return URL_CLASS_DIRECT_PUBLISHER
    return URL_CLASS_UNSUPPORTED


def _fetch_and_extract(
    article: Article,
    *,
    tickers: tuple[str, ...],
    requested_url: str,
    fetch_url: str,
    url_classification: str,
    resolution_status: str,
    timeout_seconds: float,
    user_agent: str,
    queue_item: ExtractionQueueItem,
) -> tuple[Article | None, ArticleExtractionRecord]:
    started = monotonic()
    final_url = fetch_url
    content_type = None
    content_length = 0
    status_code = None
    try:
        request = Request(fetch_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
            final_url = getattr(response, "url", fetch_url)
            status_code = getattr(response, "status", None)
            if status_code is None and hasattr(response, "getcode"):
                status_code = response.getcode()
            content_type = response.headers.get("Content-Type") if getattr(response, "headers", None) else None
        content_length = len(raw_bytes)
        if _is_google_news_url(final_url):
            return None, _fetched_failure_record(
                article,
                tickers=tickers,
                reason="google_news_unresolved",
                url_classification=url_classification,
                requested_url=requested_url,
                fetch_url=fetch_url,
                resolved_url=fetch_url if fetch_url != article.canonical_url else None,
                resolution_status=resolution_status,
                final_url=final_url,
                latency_ms=int((monotonic() - started) * 1000),
                content_type=content_type,
                content_length=content_length,
                queue_item=queue_item,
                status_code=status_code,
                extraction_quality_grade="blocked_or_shell",
                extraction_quality_reasons=("blocked_or_shell", "google_wrapper_unresolved"),
            )
        if not _is_html_content_type(content_type):
            return None, _fetched_failure_record(
                article,
                tickers=tickers,
                reason="unsupported_content_type",
                url_classification=url_classification,
                requested_url=requested_url,
                fetch_url=fetch_url,
                resolved_url=fetch_url if fetch_url != article.canonical_url else None,
                resolution_status=resolution_status,
                final_url=final_url,
                latency_ms=int((monotonic() - started) * 1000),
                content_type=content_type,
                content_length=content_length,
                queue_item=queue_item,
                status_code=status_code,
            )
        raw_html = raw_bytes.decode("utf-8", errors="replace")
        result = extract_article(
            raw_html=raw_html,
            url=final_url,
            title=article.title,
            snippet=article.snippet,
            ticker_terms=_ticker_context_terms(tickers),
        )
        latency_ms = int((monotonic() - started) * 1000)
        text = result.main_text.strip()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        failure_reasons = _failure_reasons(
            result.extraction_basis,
            result.extraction_error,
            result.extraction_quality_reasons,
        )
        extraction_failure_reason = _primary_failure_reason(failure_reasons)
        record = ArticleExtractionRecord(
            article_id=article.article_id,
            canonical_url=article.canonical_url,
            title=article.title,
            extraction_status=result.extraction_status,
            extraction_basis=result.extraction_basis if result.extraction_basis != "failed" else _fallback_basis(article),
            error_class=None if result.extraction_basis == "full_text" else _primary_failure_reason(failure_reasons),
            failure_reasons=failure_reasons,
            url_classification=url_classification,
            requested_url=requested_url,
            fetch_url=fetch_url,
            resolved_url=fetch_url if fetch_url != article.canonical_url else None,
            resolution_status=resolution_status,
            final_url=final_url,
            latency_ms=latency_ms,
            content_type=content_type,
            content_length=content_length,
            text_hash=text_hash,
            extracted_preview=_preview(text),
            extractor=result.extractor,
            extraction_method_used=result.extraction_method_used,
            extraction_failure_reason=extraction_failure_reason,
            fetched=True,
            tickers=tickers,
            source_publisher=str(queue_item.article.metadata.get("source_name") or ""),
            source_provider=str(queue_item.article.metadata.get("provider") or ""),
            queue_score=queue_item.score,
            queue_reasons=queue_item.score_reasons,
            status_code=status_code,
            canonical_link=_absolute_http_url(result.canonical_url, final_url),
            og_url=_absolute_http_url(result.og_url, final_url),
            extractor_methods_tried=tuple(attempt.method_name for attempt in result.attempts),
            accepted_method=result.extraction_method_used if result.accepted_as_full_text else None,
            accepted_text_length=len(text) if result.accepted_as_full_text else 0,
            extraction_quality_score=result.extraction_quality_score,
            extraction_quality_grade=result.extraction_quality_grade,
            extraction_quality_reasons=result.extraction_quality_reasons,
            accepted_as_full_text=result.accepted_as_full_text,
        )
        if result.accepted_as_full_text and result.extraction_basis == "full_text" and text:
            return _with_full_text(article, text, record), record
        return None, record
    except Exception as exc:  # noqa: BLE001 - fetch failures must not fail dry runs.
        latency_ms = int((monotonic() - started) * 1000)
        blocked_reasons = (
            ("blocked_or_shell", "paywall_or_login")
            if isinstance(exc, HTTPError) and exc.code in {401, 403}
            else ()
        )
        failure_reasons = _normalize_reasons(
            ("fetch_error", type(exc).__name__, *blocked_reasons, _fallback_reason(article))
        )
        return None, ArticleExtractionRecord(
            article_id=article.article_id,
            canonical_url=article.canonical_url,
            title=article.title,
            extraction_status="failed",
            extraction_basis=_fallback_basis(article),
            error_class=type(exc).__name__,
            failure_reasons=failure_reasons,
            url_classification=url_classification,
            requested_url=requested_url,
            fetch_url=fetch_url,
            resolved_url=fetch_url if fetch_url != article.canonical_url else None,
            resolution_status=resolution_status,
            final_url=final_url,
            latency_ms=latency_ms,
            content_type=content_type,
            content_length=content_length,
            text_hash=None,
            extracted_preview=None,
            extractor=None,
            extraction_method_used=None,
            extraction_failure_reason=_primary_failure_reason(failure_reasons),
            fetched=True,
            tickers=tickers,
            source_publisher=str(queue_item.article.metadata.get("source_name") or ""),
            source_provider=str(queue_item.article.metadata.get("provider") or ""),
            queue_score=queue_item.score,
            queue_reasons=queue_item.score_reasons,
            status_code=exc.code if isinstance(exc, HTTPError) else status_code,
            extraction_quality_grade="blocked_or_shell" if isinstance(exc, HTTPError) and exc.code in {401, 403} else "title_only",
            extraction_quality_reasons=blocked_reasons or ("fetch_error",),
        )


def _with_full_text(article: Article, full_text: str, record: ArticleExtractionRecord) -> Article:
    metadata = {
        **article.metadata,
        "extraction_status": record.extraction_status,
        "extraction_basis": record.extraction_basis,
        "extraction_final_url": record.final_url,
        "extraction_text_hash": record.text_hash,
        "extraction_requested_url": record.requested_url,
        "extraction_fetch_url": record.fetch_url,
        "extraction_url_classification": record.url_classification,
        "extraction_quality_score": record.extraction_quality_score,
        "extraction_quality_grade": record.extraction_quality_grade,
        "extraction_method": record.accepted_method,
        "extraction_cache_hit": record.cache_hit,
    }
    return Article(
        canonical_url=article.canonical_url,
        title=article.title,
        article_id=article.article_id,
        published_at=article.published_at,
        full_text=full_text,
        snippet=article.snippet,
        metadata=metadata,
        created_at=article.created_at,
    )


def _ticker_context_terms(tickers: tuple[str, ...]) -> tuple[str, ...]:
    lookup = ticker_lookup()
    terms: list[str] = []
    for symbol in tickers:
        ticker = lookup.get(symbol)
        if ticker is None:
            terms.append(symbol)
            continue
        terms.extend(ticker.match_terms)
    return tuple(dict.fromkeys(terms))


def _fallback_basis(article: Article) -> str:
    if article.snippet and article.snippet.strip():
        return "snippet"
    return "title"


def _preview(text: str) -> str | None:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return None
    return collapsed[:280]


def _load_extraction_cache(cache_path: str | Path | None) -> dict[str, dict[str, object]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    return {
        str(key): dict(value)
        for key, value in entries.items()
        if isinstance(value, dict)
    }


def _write_extraction_cache(
    cache_path: str | Path | None,
    cache: dict[str, dict[str, object]],
) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "entries": cache}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _find_cached_extraction(
    cache: dict[str, dict[str, object]],
    requested_url: str,
    fetch_url: str,
) -> dict[str, object] | None:
    for key in (requested_url, fetch_url):
        if key in cache:
            return cache[key]
    return None


def _cache_extraction(
    cache: dict[str, dict[str, object]],
    record: ArticleExtractionRecord,
    enriched_article: Article | None,
) -> None:
    entry: dict[str, object] = {
        "canonical_url": record.canonical_link or record.canonical_url,
        "final_url": record.final_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "content_type": record.content_type,
        "status_code": record.status_code,
        "extractor_methods_tried": list(record.extractor_methods_tried),
        "accepted_method": record.accepted_method,
        "accepted_text_hash": record.text_hash,
        "accepted_text_length": record.accepted_text_length,
        "quality_score": record.extraction_quality_score,
        "quality_grade": record.extraction_quality_grade,
        "quality_reasons": list(record.extraction_quality_reasons),
        "success": record.accepted_as_full_text,
        "failure_reason": record.extraction_failure_reason,
        "accepted_text": enriched_article.full_text if enriched_article and enriched_article.full_text else None,
    }
    keys = (
        record.requested_url,
        record.fetch_url,
        record.final_url,
        record.canonical_link,
        record.og_url,
    )
    for key in keys:
        if key:
            cache[str(key)] = entry


def _cached_extraction(
    article: Article,
    *,
    tickers: tuple[str, ...],
    requested_url: str,
    fetch_url: str,
    url_classification: str,
    resolution_status: str,
    queue_item: ExtractionQueueItem,
    cached: dict[str, object],
) -> tuple[Article | None, ArticleExtractionRecord]:
    text = str(cached.get("accepted_text") or "").strip()
    success = bool(cached.get("success")) and bool(text)
    fallback_basis = _fallback_basis(article)
    quality_reasons = tuple(str(value) for value in cached.get("quality_reasons", ()) if value)
    failure_reasons = () if success else _normalize_reasons(
        (
            str(cached.get("failure_reason") or "cached_extraction_failure"),
            *quality_reasons,
            _fallback_reason(article),
        )
    )
    record = ArticleExtractionRecord(
        article_id=article.article_id,
        canonical_url=article.canonical_url,
        title=article.title,
        extraction_status="success" if success else "fallback",
        extraction_basis="full_text" if success else fallback_basis,
        error_class=None if success else _primary_failure_reason(failure_reasons),
        failure_reasons=failure_reasons,
        url_classification=url_classification,
        requested_url=requested_url,
        fetch_url=fetch_url,
        resolved_url=fetch_url if fetch_url != requested_url else None,
        resolution_status=f"{resolution_status}_cache_hit",
        final_url=str(cached.get("final_url") or fetch_url),
        latency_ms=0,
        content_type=str(cached.get("content_type") or "") or None,
        content_length=0,
        text_hash=str(cached.get("accepted_text_hash") or "") or None,
        extracted_preview=_preview(text),
        extractor=str(cached.get("accepted_method") or "cache"),
        extraction_method_used=str(cached.get("accepted_method") or "cache"),
        extraction_failure_reason=None if success else _primary_failure_reason(failure_reasons),
        fetched=False,
        tickers=tickers,
        source_publisher=str(queue_item.article.metadata.get("source_name") or ""),
        source_provider=str(queue_item.article.metadata.get("provider") or ""),
        queue_score=queue_item.score,
        queue_reasons=queue_item.score_reasons,
        status_code=int(cached["status_code"]) if cached.get("status_code") is not None else None,
        canonical_link=str(cached.get("canonical_url") or "") or None,
        extractor_methods_tried=tuple(str(value) for value in cached.get("extractor_methods_tried", ()) if value),
        accepted_method=str(cached.get("accepted_method") or "") or None,
        accepted_text_length=int(cached.get("accepted_text_length") or 0),
        extraction_quality_score=float(cached.get("quality_score") or 0.0),
        extraction_quality_grade=str(cached.get("quality_grade") or fallback_basis),
        extraction_quality_reasons=quality_reasons,
        accepted_as_full_text=success,
        cache_hit=True,
    )
    if success:
        return _with_full_text(article, text, record), record
    return None, record


@dataclass(frozen=True)
class _ResolvedUrl:
    resolved_url: str | None
    final_url: str | None
    status: str
    latency_ms: int
    error_class: str | None = None
    canonical_url: str | None = None
    og_url: str | None = None


def _cluster_fetch_articles(cluster: DedupeCluster) -> tuple[Article, ...]:
    articles = list(cluster.articles)
    articles.sort(key=lambda article: (_article_url_rank(article), source_quality_sort_key(article), article.title, article.canonical_url))
    return tuple(articles)


def _article_url_rank(article: Article) -> tuple[int, str]:
    url_classification = classify_article_url(article.canonical_url)
    quality = assess_article_source(article)
    source_name = str(article.metadata.get("source_name") or "").lower()
    provider = str(article.metadata.get("provider") or "").lower()
    preferred_source = any(name in source_name or name in provider for name in ("yahoo", "cnbc", "marketwatch"))
    if url_classification == URL_CLASS_DIRECT_PUBLISHER and preferred_source:
        return (0, f"{quality.tier}:{article.canonical_url}")
    if url_classification == URL_CLASS_DIRECT_PUBLISHER:
        return (1, f"{quality.tier}:{article.canonical_url}")
    if url_classification == URL_CLASS_GOOGLE_NEWS_WRAPPER:
        return (2, f"{quality.tier}:{article.canonical_url}")
    return (3, f"{quality.tier}:{article.canonical_url}")


def _is_google_news_wrapper_url(url: str) -> bool:
    parsed = urlparse(url)
    if not _is_google_news_url(url):
        return False
    path = parsed.path.rstrip("/")
    return path.startswith(("/rss/articles/", "/articles/", "/read/"))


def _is_google_news_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"news.google.com", "www.news.google.com"}


def _resolve_google_news_url(
    url: str,
    *,
    timeout_seconds: float,
    user_agent: str,
) -> _ResolvedUrl:
    started = monotonic()
    final_url = url
    try:
        request = Request(url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout_seconds) as response:
            final_url = getattr(response, "url", url)
            raw_html = response.read().decode("utf-8", errors="replace")
        latency_ms = int((monotonic() - started) * 1000)
        if final_url and not _is_google_news_url(final_url):
            return _ResolvedUrl(final_url, final_url, "resolved_to_publisher", latency_ms)
        metadata = extract_url_metadata(raw_html)
        canonical_url = _absolute_http_url(metadata.get("canonical_url"), final_url)
        og_url = _absolute_http_url(metadata.get("og_url"), final_url)
        publisher_url = next(
            (
                candidate
                for candidate in (canonical_url, og_url)
                if candidate and not _is_google_news_url(candidate)
            ),
            None,
        )
        if publisher_url:
            return _ResolvedUrl(
                publisher_url,
                final_url,
                "resolved_to_publisher_metadata",
                latency_ms,
                canonical_url=canonical_url,
                og_url=og_url,
            )
        return _ResolvedUrl(
            None,
            final_url,
            "google_news_unresolved",
            latency_ms,
            "google_wrapper_unresolved",
            canonical_url,
            og_url,
        )
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        latency_ms = int((monotonic() - started) * 1000)
        return _ResolvedUrl(None, final_url, "resolve_error", latency_ms, type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail dry runs.
        latency_ms = int((monotonic() - started) * 1000)
        return _ResolvedUrl(None, final_url, "resolve_error", latency_ms, type(exc).__name__)


def _skipped_record(
    article: Article,
    *,
    tickers: tuple[str, ...],
    reason: str,
    url_classification: str,
    requested_url: str,
    resolution_status: str,
    final_url: str | None = None,
    latency_ms: int = 0,
    queue_item: ExtractionQueueItem | None = None,
    canonical_link: str | None = None,
    og_url: str | None = None,
    extraction_quality_grade: str = "title_only",
    extraction_quality_reasons: tuple[str, ...] = (),
) -> ArticleExtractionRecord:
    failure_reasons = _normalize_reasons((reason, *extraction_quality_reasons))
    return ArticleExtractionRecord(
        article_id=article.article_id,
        canonical_url=article.canonical_url,
        title=article.title,
        extraction_status="skipped",
        extraction_basis=_fallback_basis(article),
        error_class=_primary_failure_reason(failure_reasons),
        failure_reasons=failure_reasons,
        url_classification=url_classification,
        requested_url=requested_url,
        fetch_url=None,
        resolved_url=None,
        resolution_status=resolution_status,
        final_url=final_url or article.canonical_url,
        latency_ms=latency_ms,
        content_type=None,
        content_length=0,
        text_hash=None,
        extracted_preview=None,
        extractor=None,
        extraction_method_used=None,
        extraction_failure_reason=_primary_failure_reason(failure_reasons),
        fetched=False,
        tickers=tickers,
        source_publisher=str(queue_item.article.metadata.get("source_name") or "") if queue_item else "",
        source_provider=str(queue_item.article.metadata.get("provider") or "") if queue_item else "",
        queue_score=queue_item.score if queue_item else 0.0,
        queue_reasons=queue_item.score_reasons if queue_item else (),
        canonical_link=canonical_link,
        og_url=og_url,
        extraction_quality_grade=extraction_quality_grade,
        extraction_quality_reasons=extraction_quality_reasons,
    )


def _fetched_failure_record(
    article: Article,
    *,
    tickers: tuple[str, ...],
    reason: str,
    url_classification: str,
    requested_url: str,
    fetch_url: str,
    resolved_url: str | None,
    resolution_status: str,
    final_url: str,
    latency_ms: int,
    content_type: str | None,
    content_length: int,
    queue_item: ExtractionQueueItem | None = None,
    status_code: int | None = None,
    extraction_quality_grade: str = "title_only",
    extraction_quality_reasons: tuple[str, ...] = (),
) -> ArticleExtractionRecord:
    failure_reasons = _normalize_reasons((reason, *extraction_quality_reasons, _fallback_reason(article)))
    return ArticleExtractionRecord(
        article_id=article.article_id,
        canonical_url=article.canonical_url,
        title=article.title,
        extraction_status="failed",
        extraction_basis=_fallback_basis(article),
        error_class=_primary_failure_reason(failure_reasons),
        failure_reasons=failure_reasons,
        url_classification=url_classification,
        requested_url=requested_url,
        fetch_url=fetch_url,
        resolved_url=resolved_url,
        resolution_status=resolution_status,
        final_url=final_url,
        latency_ms=latency_ms,
        content_type=content_type,
        content_length=content_length,
        text_hash=None,
        extracted_preview=None,
        extractor=None,
        extraction_method_used=None,
        extraction_failure_reason=_primary_failure_reason(failure_reasons),
        fetched=True,
        tickers=tickers,
        source_publisher=str(queue_item.article.metadata.get("source_name") or "") if queue_item else "",
        source_provider=str(queue_item.article.metadata.get("provider") or "") if queue_item else "",
        queue_score=queue_item.score if queue_item else 0.0,
        queue_reasons=queue_item.score_reasons if queue_item else (),
        status_code=status_code,
        extraction_quality_grade=extraction_quality_grade,
        extraction_quality_reasons=extraction_quality_reasons,
    )


def _is_html_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    lowered = content_type.lower()
    return any(html_type in lowered for html_type in HTML_CONTENT_TYPES)


def _increment(counts: object, reason: str) -> None:
    if not isinstance(counts, dict):
        return
    counts[reason] = int(counts.get(reason, 0)) + 1


def _success_rates(
    records: tuple[ArticleExtractionRecord, ...],
    field_name: str,
) -> dict[str, float]:
    totals: dict[str, int] = {}
    successes: dict[str, int] = {}
    for record in records:
        if not (record.fetched or record.cache_hit):
            continue
        key = str(getattr(record, field_name) or "unknown")
        totals[key] = totals.get(key, 0) + 1
        if record.accepted_as_full_text:
            successes[key] = successes.get(key, 0) + 1
    return {
        key: round(successes.get(key, 0) / total, 4)
        for key, total in sorted(totals.items())
        if total
    }


def _failure_reasons(
    extraction_basis: str,
    extraction_error: str | None,
    quality_reasons: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if extraction_basis == "full_text":
        return ()
    reasons: list[str] = ["no_article_body", *quality_reasons]
    if extraction_error:
        reasons.extend(part.strip() for part in extraction_error.split(";") if part.strip())
    if extraction_basis == "snippet":
        reasons.append("snippet_fallback")
    elif extraction_basis == "title":
        reasons.append("title_fallback")
    return _normalize_reasons(tuple(reasons))


def _fallback_reason(article: Article) -> str:
    return "snippet_fallback" if _fallback_basis(article) == "snippet" else "title_fallback"


def _normalize_reasons(reasons: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for reason in reasons:
        clean = reason.strip()
        if not clean or clean in normalized:
            continue
        normalized.append(clean)
    return tuple(normalized)


def _primary_failure_reason(reasons: tuple[str, ...]) -> str | None:
    return reasons[0] if reasons else None
