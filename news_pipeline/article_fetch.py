"""Opt-in capped article page fetching and extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from .dedup import DedupeCluster
from .extract import extract_article, extraction_dependency_status
from .models import Article
from .recency import article_recency
from .tickers import match_tickers


DEFAULT_MAX_ARTICLE_FETCHES = 25
DEFAULT_MAX_FETCHES_PER_TICKER = 2
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
    extractor_diagnostics: dict[str, bool] | None = None

    @property
    def basis_counts(self) -> dict[str, int]:
        counts = {"full_text": 0, "snippet": 0, "title": 0}
        for record in self.records:
            if not record.fetched:
                continue
            if record.extraction_basis in counts:
                counts[record.extraction_basis] += 1
        return counts

    @property
    def publisher_article_fetches(self) -> int:
        return sum(1 for record in self.records if record.fetched and record.fetch_url and not _is_google_news_url(record.fetch_url))

    @property
    def google_news_wrappers_skipped(self) -> int:
        return sum(1 for record in self.records if "google_news_unresolved" in record.failure_reasons)

    @property
    def google_news_wrappers_resolved(self) -> int:
        return sum(1 for record in self.records if record.resolution_status == "resolved_to_publisher")

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
            method = record.extraction_method_used
            if not method:
                continue
            counts[method] = counts.get(method, 0) + 1
        return counts

    @property
    def extraction_failure_reason(self) -> str | None:
        reasons = self.failure_reason_counts
        if not reasons:
            return None
        return sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]

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
            "extraction_failure_reason": self.extraction_failure_reason,
            "extractor_diagnostics": self.extractor_diagnostics or extraction_dependency_status(),
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
    enriched: dict[str, Article] = {}
    records: list[ArticleExtractionRecord] = []
    ticker_counts: dict[str, int] = {}
    seen_urls: set[str] = set()

    for article, tickers, requested_url, fetch_url, url_classification, resolution_status, resolution_record in _iter_fetch_candidates(
        tuple(clusters),
        run_date=run_date,
        max_article_fetches=max_article_fetches,
        max_fetches_per_ticker=max_fetches_per_ticker,
        fetch_timeout_seconds=fetch_timeout_seconds,
        user_agent=user_agent,
        ticker_counts=ticker_counts,
        seen_urls=seen_urls,
    ):
        if resolution_record is not None:
            records.append(resolution_record)
            continue
        enriched_article, record = _fetch_and_extract(
            article,
            tickers=tickers,
            requested_url=requested_url,
            fetch_url=fetch_url,
            url_classification=url_classification,
            resolution_status=resolution_status,
            timeout_seconds=fetch_timeout_seconds,
            user_agent=user_agent,
        )
        records.append(record)
        if enriched_article is not None:
            enriched[article.canonical_url] = enriched_article

    successful = sum(1 for record in records if record.fetched and record.extraction_basis == "full_text")
    fetched = sum(1 for record in records if record.fetched)
    failed = fetched - successful
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
        extractor_diagnostics=extraction_dependency_status(),
    )


def _iter_fetch_candidates(
    clusters: tuple[DedupeCluster, ...],
    *,
    run_date: str,
    max_article_fetches: int,
    max_fetches_per_ticker: int,
    fetch_timeout_seconds: float,
    user_agent: str,
    ticker_counts: dict[str, int],
    seen_urls: set[str],
):
    attempted_fetches = 0
    for cluster in sorted(clusters, key=lambda item: _cluster_rank(item, run_date), reverse=True):
        if attempted_fetches >= max_article_fetches:
            break
        tickers = _cluster_tickers(cluster)
        if not tickers:
            continue
        if any(ticker_counts.get(ticker, 0) >= max_fetches_per_ticker for ticker in tickers):
            continue
        for candidate_article in _cluster_fetch_articles(cluster):
            if attempted_fetches >= max_article_fetches:
                break
            if candidate_article.canonical_url in seen_urls or not _is_fetchable_url(candidate_article.canonical_url):
                continue
            seen_urls.add(candidate_article.canonical_url)
            article = cluster.canonical_article
            requested_url = candidate_article.canonical_url
            url_classification = classify_article_url(requested_url)
            if url_classification == URL_CLASS_UNSUPPORTED:
                yield article, tickers, requested_url, None, url_classification, "unsupported", _skipped_record(
                    article,
                    tickers=tickers,
                    reason="unsupported_or_unknown",
                    url_classification=url_classification,
                    requested_url=requested_url,
                    resolution_status="unsupported",
                )
                continue
            resolution_status = "direct_publisher_url"
            fetch_url = requested_url
            if url_classification == URL_CLASS_GOOGLE_NEWS_WRAPPER:
                resolved = _resolve_google_news_url(
                    requested_url,
                    timeout_seconds=fetch_timeout_seconds,
                    user_agent=user_agent,
                )
                if resolved.resolved_url is None:
                    yield article, tickers, requested_url, None, url_classification, resolved.status, _skipped_record(
                        article,
                        tickers=tickers,
                        reason=resolved.error_class or "google_news_unresolved",
                        url_classification=url_classification,
                        requested_url=requested_url,
                        resolution_status=resolved.status,
                        final_url=resolved.final_url,
                        latency_ms=resolved.latency_ms,
                    )
                    continue
                fetch_url = resolved.resolved_url
                resolution_status = "resolved_to_publisher"
            attempted_fetches += 1
            for ticker in tickers:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            yield article, tickers, requested_url, fetch_url, url_classification, resolution_status, None
            break


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
) -> tuple[Article | None, ArticleExtractionRecord]:
    started = monotonic()
    final_url = fetch_url
    content_type = None
    content_length = 0
    try:
        request = Request(fetch_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
            final_url = getattr(response, "url", fetch_url)
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
            )
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
        failure_reasons = _failure_reasons(result.extraction_basis, result.extraction_error)
        extraction_failure_reason = result.extraction_failure_reason or _primary_failure_reason(failure_reasons)
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
            failure_reasons=_normalize_reasons(("fetch_error", type(exc).__name__, _fallback_reason(article))),
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
            extraction_failure_reason=_primary_failure_reason(_normalize_reasons(("fetch_error", type(exc).__name__, _fallback_reason(article)))),
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
        "extraction_requested_url": record.requested_url,
        "extraction_fetch_url": record.fetch_url,
        "extraction_url_classification": record.url_classification,
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


@dataclass(frozen=True)
class _ResolvedUrl:
    resolved_url: str | None
    final_url: str | None
    status: str
    latency_ms: int
    error_class: str | None = None


def _cluster_fetch_articles(cluster: DedupeCluster) -> tuple[Article, ...]:
    articles = list(cluster.articles)
    articles.sort(key=lambda article: (_article_url_rank(article), article.title, article.canonical_url))
    return tuple(articles)


def _article_url_rank(article: Article) -> tuple[int, str]:
    url_classification = classify_article_url(article.canonical_url)
    source_name = str(article.metadata.get("source_name") or "").lower()
    provider = str(article.metadata.get("provider") or "").lower()
    preferred_source = any(name in source_name or name in provider for name in ("yahoo", "cnbc", "marketwatch"))
    if url_classification == URL_CLASS_DIRECT_PUBLISHER and preferred_source:
        return (0, article.canonical_url)
    if url_classification == URL_CLASS_DIRECT_PUBLISHER:
        return (1, article.canonical_url)
    if url_classification == URL_CLASS_GOOGLE_NEWS_WRAPPER:
        return (2, article.canonical_url)
    return (3, article.canonical_url)


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
        latency_ms = int((monotonic() - started) * 1000)
        if final_url and not _is_google_news_url(final_url):
            return _ResolvedUrl(final_url, final_url, "resolved_to_publisher", latency_ms)
        return _ResolvedUrl(None, final_url, "google_news_unresolved", latency_ms, "google_news_unresolved")
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
) -> ArticleExtractionRecord:
    failure_reasons = _normalize_reasons((reason,))
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
) -> ArticleExtractionRecord:
    failure_reasons = _normalize_reasons((reason, _fallback_reason(article)))
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
    )


def _is_html_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    lowered = content_type.lower()
    return any(html_type in lowered for html_type in HTML_CONTENT_TYPES)


def _failure_reasons(extraction_basis: str, extraction_error: str | None) -> tuple[str, ...]:
    if extraction_basis == "full_text":
        return ()
    reasons: list[str] = []
    if extraction_error:
        reasons.extend(part.strip() for part in extraction_error.split(";") if part.strip())
    if "no_article_body" not in reasons:
        reasons.append("no_article_body")
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
