"""Opt-in capped article page fetching and extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from time import monotonic
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from .dedup import DedupeCluster
from .extract import extract_article
from .models import Article
from .recency import article_recency
from .tickers import match_tickers


DEFAULT_MAX_ARTICLE_FETCHES = 25
DEFAULT_MAX_FETCHES_PER_TICKER = 2
DEFAULT_FETCH_TIMEOUT_SECONDS = 8.0
DEFAULT_ARTICLE_FETCH_USER_AGENT = "StonkNewsPipeline/0.1 (+local dry-run article extraction)"


@dataclass(frozen=True)
class ArticleExtractionRecord:
    article_id: str | None
    canonical_url: str
    title: str
    extraction_status: str
    extraction_basis: str
    error_class: str | None
    final_url: str | None
    latency_ms: int
    content_type: str | None
    content_length: int
    text_hash: str | None
    extracted_preview: str | None
    extractor: str | None
    fetched: bool
    tickers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "article_id": self.article_id,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "extraction_status": self.extraction_status,
            "extraction_basis": self.extraction_basis,
            "error_class": self.error_class,
            "final_url": self.final_url,
            "latency_ms": self.latency_ms,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "text_hash": self.text_hash,
            "extracted_preview": self.extracted_preview,
            "extractor": self.extractor,
            "fetched": self.fetched,
            "tickers": list(self.tickers),
        }


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

    @property
    def basis_counts(self) -> dict[str, int]:
        counts = {"full_text": 0, "snippet": 0, "title": 0}
        for record in self.records:
            if record.extraction_basis in counts:
                counts[record.extraction_basis] += 1
        return counts

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
) -> tuple[dict[str, Article], ArticleFetchSummary]:
    """Fetch at most one article page for selected top event clusters."""
    max_article_fetches = max(0, min(DEFAULT_MAX_ARTICLE_FETCHES, int(max_article_fetches)))
    max_fetches_per_ticker = max(1, int(max_fetches_per_ticker))
    fetch_timeout_seconds = max(0.1, float(fetch_timeout_seconds))
    selected = _select_fetch_candidates(
        tuple(clusters),
        run_date=run_date,
        max_article_fetches=max_article_fetches,
        max_fetches_per_ticker=max_fetches_per_ticker,
    )
    enriched: dict[str, Article] = {}
    records: list[ArticleExtractionRecord] = []

    for article, tickers in selected:
        enriched_article, record = _fetch_and_extract(
            article,
            tickers=tickers,
            timeout_seconds=fetch_timeout_seconds,
            user_agent=user_agent,
        )
        records.append(record)
        if enriched_article is not None:
            enriched[article.canonical_url] = enriched_article

    successful = sum(1 for record in records if record.extraction_basis == "full_text")
    failed = len(records) - successful
    return enriched, ArticleFetchSummary(
        enabled=True,
        attempted_fetches=len(records),
        successful_extractions=successful,
        failed_extractions=failed,
        max_article_fetches=max_article_fetches,
        max_fetches_per_ticker=max_fetches_per_ticker,
        fetch_timeout_seconds=fetch_timeout_seconds,
        records=tuple(records),
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
        max_article_fetches=max(0, min(DEFAULT_MAX_ARTICLE_FETCHES, int(max_article_fetches))),
        max_fetches_per_ticker=max(1, int(max_fetches_per_ticker)),
        fetch_timeout_seconds=max(0.1, float(fetch_timeout_seconds)),
        reason=reason,
    )


def _select_fetch_candidates(
    clusters: tuple[DedupeCluster, ...],
    *,
    run_date: str,
    max_article_fetches: int,
    max_fetches_per_ticker: int,
) -> list[tuple[Article, tuple[str, ...]]]:
    selected: list[tuple[Article, tuple[str, ...]]] = []
    ticker_counts: dict[str, int] = {}
    seen_urls: set[str] = set()
    for cluster in sorted(clusters, key=lambda item: _cluster_rank(item, run_date), reverse=True):
        if len(selected) >= max_article_fetches:
            break
        article = cluster.canonical_article
        if article.canonical_url in seen_urls or not _is_fetchable_url(article.canonical_url):
            continue
        tickers = _cluster_tickers(cluster)
        if not tickers:
            continue
        if any(ticker_counts.get(ticker, 0) >= max_fetches_per_ticker for ticker in tickers):
            continue
        selected.append((article, tickers))
        seen_urls.add(article.canonical_url)
        for ticker in tickers:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    return selected


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


def _fetch_and_extract(
    article: Article,
    *,
    tickers: tuple[str, ...],
    timeout_seconds: float,
    user_agent: str,
) -> tuple[Article | None, ArticleExtractionRecord]:
    started = monotonic()
    final_url = article.canonical_url
    content_type = None
    content_length = 0
    try:
        request = Request(article.canonical_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
            final_url = getattr(response, "url", article.canonical_url)
            content_type = response.headers.get("Content-Type") if getattr(response, "headers", None) else None
        content_length = len(raw_bytes)
        raw_html = raw_bytes.decode("utf-8", errors="replace")
        result = extract_article(
            raw_html=raw_html,
            url=final_url,
            title=article.title,
            snippet=article.snippet,
        )
        latency_ms = int((monotonic() - started) * 1000)
        text = result.main_text.strip()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        record = ArticleExtractionRecord(
            article_id=article.article_id,
            canonical_url=article.canonical_url,
            title=article.title,
            extraction_status=result.extraction_status,
            extraction_basis=result.extraction_basis if result.extraction_basis != "failed" else _fallback_basis(article),
            error_class=result.extraction_error,
            final_url=final_url,
            latency_ms=latency_ms,
            content_type=content_type,
            content_length=content_length,
            text_hash=text_hash,
            extracted_preview=_preview(text),
            extractor=result.extractor,
            fetched=True,
            tickers=tickers,
        )
        if result.extraction_basis == "full_text" and text:
            return _with_full_text(article, text, record), record
        return None, record
    except Exception as exc:  # noqa: BLE001 - fetch failures must not fail dry runs.
        latency_ms = int((monotonic() - started) * 1000)
        return None, ArticleExtractionRecord(
            article_id=article.article_id,
            canonical_url=article.canonical_url,
            title=article.title,
            extraction_status="failed",
            extraction_basis=_fallback_basis(article),
            error_class=type(exc).__name__,
            final_url=final_url,
            latency_ms=latency_ms,
            content_type=content_type,
            content_length=content_length,
            text_hash=None,
            extracted_preview=None,
            extractor=None,
            fetched=True,
            tickers=tickers,
        )


def _with_full_text(article: Article, full_text: str, record: ArticleExtractionRecord) -> Article:
    metadata = {
        **article.metadata,
        "extraction_status": record.extraction_status,
        "extraction_basis": record.extraction_basis,
        "extraction_final_url": record.final_url,
        "extraction_text_hash": record.text_hash,
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


def _fallback_basis(article: Article) -> str:
    if article.snippet and article.snippet.strip():
        return "snippet"
    return "title"


def _preview(text: str) -> str | None:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return None
    return collapsed[:280]
