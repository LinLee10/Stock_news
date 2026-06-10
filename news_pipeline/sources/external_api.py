"""Quota-limited live adapters for explicitly enabled external news APIs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from time import monotonic
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from news_pipeline.article_types import classify_article_type
from news_pipeline.dedup import canonicalize_url
from news_pipeline.models import Article
from news_pipeline.ticker_matching import assess_ticker_matches
from news_pipeline.tickers import TrackedTicker

from .query_planner import TickerQueryPlan, provider_query_plans
from .source_registry import SourceProfile


HttpJsonFetcher = Callable[
    [str, Mapping[str, object], Mapping[str, str], float],
    Mapping[str, object] | Sequence[object],
]


@dataclass(frozen=True)
class ExternalApiAttempt:
    provider: str
    status: str
    request_count: int
    article_count: int
    query_id: str | None = None
    ticker: str | None = None
    latency_ms: int = 0
    status_code: int | None = None
    skipped_reason: str | None = None
    error_class: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "status": self.status,
            "request_count": self.request_count,
            "article_count": self.article_count,
            "query_id": self.query_id,
            "ticker": self.ticker,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "skipped_reason": self.skipped_reason,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ExternalApiCollectionResult:
    articles: tuple[Article, ...]
    attempts: tuple[ExternalApiAttempt, ...]
    diagnostics: Mapping[str, object]


PROVIDER_ORDER = ("marketaux", "nyt", "finnhub_news", "gnews", "newsapi")
PROVIDER_ENV_VARS = {
    "marketaux": "MARKETAUX_API_KEY",
    "nyt": "NYT_API_KEY",
    "finnhub_news": "FINNHUB_KEY",
    "gnews": "GNEWS_KEY",
    "newsapi": "NEWSAPI_KEY",
}
DEFAULT_MAX_EXTERNAL_ARTICLES_PER_PROVIDER = 80
DEFAULT_MAX_EXTERNAL_ARTICLES_PER_TICKER = 40
DEFAULT_MAX_EXTERNAL_PROVIDER_SHARE_POST_DEDUP = 0.60
DEFAULT_PROVIDER_CONCENTRATION_WARNING_THRESHOLD = 0.60
DEFAULT_TICKER_CONCENTRATION_WARNING_THRESHOLD = 0.35


def collect_external_api_articles(
    *,
    profiles: Mapping[str, SourceProfile],
    query_plans: Sequence[TickerQueryPlan],
    tracked_tickers: Sequence[TrackedTicker],
    provider_flags: Mapping[str, bool],
    global_enabled: bool,
    environ: Mapping[str, str],
    total_request_budget: int,
    provider_request_budgets: Mapping[str, int],
    run_date: str,
    timeout_seconds: float,
    fetch_json: HttpJsonFetcher | None = None,
) -> ExternalApiCollectionResult:
    fetcher = fetch_json or _fetch_json
    remaining_total = max(0, int(total_request_budget))
    articles: list[Article] = []
    attempts: list[ExternalApiAttempt] = []
    skipped_reasons: dict[str, str] = {}
    requests_by_provider: Counter[str] = Counter()
    articles_by_provider: Counter[str] = Counter()
    articles_by_ticker: Counter[str] = Counter()
    effective_requests: dict[str, dict[str, object]] = {}

    for provider in PROVIDER_ORDER:
        profile = profiles.get(provider)
        if profile is None:
            skipped_reasons[provider] = "source_profile_missing"
            continue
        reason = _provider_skip_reason(
            provider,
            global_enabled=global_enabled,
            provider_enabled=bool(provider_flags.get(provider)),
            environ=environ,
            remaining_total=remaining_total,
            provider_budget=max(0, int(provider_request_budgets.get(provider, 0))),
        )
        if reason:
            skipped_reasons[provider] = reason
            attempts.append(
                ExternalApiAttempt(
                    provider=provider,
                    status="skipped",
                    request_count=0,
                    article_count=0,
                    skipped_reason=reason,
                )
            )
            continue

        provider_budget = min(
            remaining_total,
            max(0, int(provider_request_budgets.get(provider, 0))),
        )
        plans = _selected_plans(
            provider_query_plans(query_plans, provider),
            provider_budget,
        )
        for plan in plans:
            if remaining_total <= 0 or requests_by_provider[provider] >= provider_budget:
                break
            started = monotonic()
            requests_by_provider[provider] += 1
            remaining_total -= 1
            try:
                endpoint, safe_params = _safe_provider_request(
                    provider,
                    plan,
                    run_date=run_date,
                )
                effective_requests.setdefault(
                    provider,
                    {
                        "endpoint": endpoint,
                        "params": safe_params,
                    },
                )
                payload = _request_provider(
                    provider,
                    plan,
                    api_key=environ[PROVIDER_ENV_VARS[provider]],
                    run_date=run_date,
                    timeout_seconds=timeout_seconds,
                    fetch_json=fetcher,
                )
                normalized = _normalize_provider_payload(
                    provider,
                    payload,
                    plan=plan,
                    profile=profile,
                    api_request_count=requests_by_provider[provider],
                    tracked_tickers=tracked_tickers,
                )
                articles.extend(normalized)
                articles_by_provider[provider] += len(normalized)
                for article in normalized:
                    for symbol in _article_symbols(article):
                        articles_by_ticker[symbol] += 1
                attempts.append(
                    ExternalApiAttempt(
                        provider=provider,
                        status="success",
                        request_count=1,
                        article_count=len(normalized),
                        query_id=plan.query_id,
                        ticker=plan.ticker,
                        latency_ms=int((monotonic() - started) * 1000),
                        status_code=200,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate provider requests.
                attempts.append(
                    ExternalApiAttempt(
                        provider=provider,
                        status="failure",
                        request_count=1,
                        article_count=0,
                        query_id=plan.query_id,
                        ticker=plan.ticker,
                        latency_ms=int((monotonic() - started) * 1000),
                        status_code=exc.code if isinstance(exc, HTTPError) else None,
                        error_class=type(exc).__name__,
                        error_message=_safe_external_error(exc),
                    )
                )

    unique = _unique_articles(articles)
    gnews_attempts = tuple(
        attempt for attempt in attempts if attempt.provider == "gnews"
    )
    nyt_attempts = tuple(attempt for attempt in attempts if attempt.provider == "nyt")
    diagnostics = {
        "external_api_provider_usage": [attempt.as_dict() for attempt in attempts],
        "external_api_provider_skipped_reasons": dict(sorted(skipped_reasons.items())),
        "external_api_requests_used_by_provider": dict(sorted(requests_by_provider.items())),
        "raw_external_articles_by_provider": dict(sorted(articles_by_provider.items())),
        "raw_external_articles_by_ticker": dict(sorted(articles_by_ticker.items())),
        # Compatibility aliases for existing artifact consumers.
        "external_api_articles_by_provider": dict(sorted(articles_by_provider.items())),
        "external_api_articles_by_ticker": dict(sorted(articles_by_ticker.items())),
        "external_api_total_requests_used": sum(requests_by_provider.values()),
        "external_api_total_request_budget": max(0, int(total_request_budget)),
        "quota_budget_remaining_estimate": max(0, remaining_total),
        "gnews_status_code": _provider_status_code(gnews_attempts),
        "gnews_error_reason": _provider_error_reason(gnews_attempts),
        "gnews_requests_attempted": sum(
            attempt.request_count for attempt in gnews_attempts
        ),
        "gnews_articles_returned": sum(
            attempt.article_count for attempt in gnews_attempts
        ),
        "gnews_skipped_reason": skipped_reasons.get("gnews"),
        "gnews_effective_endpoint_without_key": (
            effective_requests.get("gnews", {}).get("endpoint")
        ),
        "gnews_effective_params_without_key": (
            effective_requests.get("gnews", {}).get("params") or {}
        ),
        "nyt_requests_attempted": sum(
            attempt.request_count for attempt in nyt_attempts
        ),
        "nyt_articles_returned": sum(
            attempt.article_count for attempt in nyt_attempts
        ),
        "nyt_zero_result_queries": sum(
            attempt.status == "success" and attempt.article_count == 0
            for attempt in nyt_attempts
        ),
        "nyt_role": "context_news_api",
    }
    return ExternalApiCollectionResult(
        articles=tuple(unique),
        attempts=tuple(attempts),
        diagnostics=diagnostics,
    )


def balance_external_sentiment_articles(
    articles: Sequence[Article],
    *,
    max_articles_per_provider: int = DEFAULT_MAX_EXTERNAL_ARTICLES_PER_PROVIDER,
    max_articles_per_ticker: int = DEFAULT_MAX_EXTERNAL_ARTICLES_PER_TICKER,
    max_provider_share: float = DEFAULT_MAX_EXTERNAL_PROVIDER_SHARE_POST_DEDUP,
) -> tuple[Article, ...]:
    external = sorted(
        (article for article in articles if _is_external_article(article)),
        key=_external_article_priority,
        reverse=True,
    )
    selected: list[Article] = []
    provider_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    provider_cap = max(1, int(max_articles_per_provider))
    ticker_cap = max(1, int(max_articles_per_ticker))

    for article in external:
        provider = _external_provider(article)
        tickers = _external_article_tickers(article)
        if provider_counts[provider] >= provider_cap:
            continue
        if tickers and any(ticker_counts[ticker] >= ticker_cap for ticker in tickers):
            continue
        selected.append(article)
        provider_counts[provider] += 1
        for ticker in tickers:
            ticker_counts[ticker] += 1

    share_limit = max(0.0, min(1.0, float(max_provider_share)))
    if 0.0 < share_limit < 1.0 and len(provider_counts) > 1:
        selected = _trim_provider_concentration(selected, share_limit)

    selected_urls = {article.canonical_url for article in selected}
    return tuple(
        article
        for article in articles
        if not _is_external_article(article)
        or article.canonical_url in selected_urls
    )


def build_external_quality_diagnostics(
    *,
    raw_articles_by_provider: Mapping[str, int],
    raw_articles_by_ticker: Mapping[str, int],
    post_dedup_articles: Sequence[Article],
    sentiment_articles: Sequence[Article],
    provider_warning_threshold: float = DEFAULT_PROVIDER_CONCENTRATION_WARNING_THRESHOLD,
    ticker_warning_threshold: float = DEFAULT_TICKER_CONCENTRATION_WARNING_THRESHOLD,
) -> dict[str, object]:
    post_provider, post_ticker = _external_counts(post_dedup_articles)
    sentiment_provider, sentiment_ticker = _external_counts(sentiment_articles)
    raw_provider_share, raw_provider = _top_share(raw_articles_by_provider)
    post_provider_share, post_provider_name = _top_share(post_provider)
    raw_ticker_share, raw_ticker = _top_share(raw_articles_by_ticker)
    post_ticker_share, post_ticker_name = _top_share(post_ticker)
    provider_threshold = max(0.0, min(1.0, provider_warning_threshold))
    ticker_threshold = max(0.0, min(1.0, ticker_warning_threshold))
    return {
        "raw_external_articles_by_provider": dict(
            sorted((str(key), int(value)) for key, value in raw_articles_by_provider.items())
        ),
        "post_dedup_external_articles_by_provider": dict(sorted(post_provider.items())),
        "articles_used_for_sentiment_by_provider": dict(
            sorted(sentiment_provider.items())
        ),
        "raw_external_articles_by_ticker": dict(
            sorted((str(key), int(value)) for key, value in raw_articles_by_ticker.items())
        ),
        "post_dedup_external_articles_by_ticker": dict(sorted(post_ticker.items())),
        "articles_used_for_sentiment_by_ticker": dict(sorted(sentiment_ticker.items())),
        "top_provider_share_raw": raw_provider_share,
        "top_provider_raw": raw_provider,
        "top_provider_share_post_dedup": post_provider_share,
        "top_provider_post_dedup": post_provider_name,
        "top_ticker_share_raw": raw_ticker_share,
        "top_ticker_raw": raw_ticker,
        "top_ticker_share_post_dedup": post_ticker_share,
        "top_ticker_post_dedup": post_ticker_name,
        "provider_concentration_warning": (
            post_provider_share > provider_threshold
        ),
        "ticker_concentration_warning": post_ticker_share > ticker_threshold,
    }


def _trim_provider_concentration(
    articles: list[Article],
    share_limit: float,
) -> list[Article]:
    selected = list(articles)
    while selected:
        counts = Counter(_external_provider(article) for article in selected)
        if len(counts) <= 1:
            break
        provider, count = counts.most_common(1)[0]
        if count / len(selected) <= share_limit + 0.000001:
            break
        removal_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if _external_provider(selected[index]) == provider
            ),
            None,
        )
        if removal_index is None:
            break
        selected.pop(removal_index)
    return selected


def _external_counts(
    articles: Sequence[Article],
) -> tuple[Counter[str], Counter[str]]:
    providers: Counter[str] = Counter()
    tickers: Counter[str] = Counter()
    for article in articles:
        if not _is_external_article(article):
            continue
        providers[_external_provider(article)] += 1
        for ticker in _external_article_tickers(article):
            tickers[ticker] += 1
    return providers, tickers


def _top_share(counts: Mapping[str, int]) -> tuple[float, str | None]:
    positive = [(str(key), int(value)) for key, value in counts.items() if int(value) > 0]
    total = sum(value for _key, value in positive)
    if not total:
        return 0.0, None
    key, value = sorted(positive, key=lambda item: (-item[1], item[0]))[0]
    return round(value / total, 4), key


def _is_external_article(article: Article) -> bool:
    return bool(article.metadata.get("external_api_used"))


def _external_provider(article: Article) -> str:
    return str(
        article.metadata.get("api_provider")
        or article.metadata.get("source_id")
        or article.metadata.get("provider")
        or "unknown"
    )


def _external_article_tickers(article: Article) -> tuple[str, ...]:
    explicit = _article_symbols(article)
    if explicit:
        return explicit
    return tuple(match.ticker for match in assess_ticker_matches(article))


def _external_article_priority(article: Article) -> tuple[float, str, str]:
    return (
        float(article.metadata.get("acquisition_score") or 0.0),
        str(article.published_at or ""),
        article.canonical_url,
    )


def _provider_skip_reason(
    provider: str,
    *,
    global_enabled: bool,
    provider_enabled: bool,
    environ: Mapping[str, str],
    remaining_total: int,
    provider_budget: int,
) -> str | None:
    if not global_enabled:
        return "global_external_api_flag_disabled"
    if not provider_enabled:
        return "provider_flag_disabled"
    if not environ.get(PROVIDER_ENV_VARS[provider]):
        return "missing_api_key"
    if provider_budget <= 0:
        return "provider_request_budget_exhausted"
    if remaining_total <= 0:
        return "total_request_budget_exhausted"
    return None


def _selected_plans(
    plans: Sequence[TickerQueryPlan],
    budget: int,
) -> tuple[TickerQueryPlan, ...]:
    selected: list[TickerQueryPlan] = []
    seen_tickers: set[str] = set()
    for plan in sorted(plans, key=lambda item: (-item.priority, item.ticker, item.query_id)):
        if len(selected) >= budget:
            break
        if plan.ticker in seen_tickers:
            continue
        selected.append(plan)
        seen_tickers.add(plan.ticker)
    return tuple(selected)


def _request_provider(
    provider: str,
    plan: TickerQueryPlan,
    *,
    api_key: str,
    run_date: str,
    timeout_seconds: float,
    fetch_json: HttpJsonFetcher,
) -> Mapping[str, object] | Sequence[object]:
    if provider == "marketaux":
        return fetch_json(
            "https://api.marketaux.com/v1/news/all",
            {
                "symbols": plan.ticker,
                "filter_entities": "true",
                "must_have_entities": "true",
                "group_similar": "true",
                "language": "en",
                "limit": 10,
                "api_token": api_key,
            },
            {},
            timeout_seconds,
        )
    if provider == "nyt":
        return fetch_json(
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            {
                "q": plan.query_text,
                "sort": "newest",
                "begin_date": (date.fromisoformat(run_date) - timedelta(days=7)).strftime(
                    "%Y%m%d"
                ),
                "api-key": api_key,
            },
            {},
            timeout_seconds,
        )
    if provider == "finnhub_news":
        run_day = date.fromisoformat(run_date)
        return fetch_json(
            "https://finnhub.io/api/v1/company-news",
            {
                "symbol": plan.ticker,
                "from": (run_day - timedelta(days=7)).isoformat(),
                "to": run_day.isoformat(),
                "token": api_key,
            },
            {},
            timeout_seconds,
        )
    if provider == "gnews":
        return fetch_json(
            "https://gnews.io/api/v4/search",
            {
                "q": plan.query_text,
                "lang": "en",
                "country": "us",
                "max": 10,
                "apikey": api_key,
            },
            {},
            timeout_seconds,
        )
    if provider == "newsapi":
        return fetch_json(
            "https://newsapi.org/v2/everything",
            {
                "q": plan.query_text,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
            },
            {"X-Api-Key": api_key},
            timeout_seconds,
        )
    raise KeyError(provider)


def _safe_provider_request(
    provider: str,
    plan: TickerQueryPlan,
    *,
    run_date: str,
) -> tuple[str, dict[str, object]]:
    if provider == "marketaux":
        return (
            "https://api.marketaux.com/v1/news/all",
            {
                "symbols": plan.ticker,
                "filter_entities": "true",
                "must_have_entities": "true",
                "group_similar": "true",
                "language": "en",
                "limit": 10,
            },
        )
    if provider == "nyt":
        return (
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            {
                "q": plan.query_text,
                "sort": "newest",
                "begin_date": (
                    date.fromisoformat(run_date) - timedelta(days=7)
                ).strftime("%Y%m%d"),
            },
        )
    if provider == "finnhub_news":
        run_day = date.fromisoformat(run_date)
        return (
            "https://finnhub.io/api/v1/company-news",
            {
                "symbol": plan.ticker,
                "from": (run_day - timedelta(days=7)).isoformat(),
                "to": run_day.isoformat(),
            },
        )
    if provider == "gnews":
        return (
            "https://gnews.io/api/v4/search",
            {
                "q": plan.query_text,
                "lang": "en",
                "country": "us",
                "max": 10,
            },
        )
    if provider == "newsapi":
        return (
            "https://newsapi.org/v2/everything",
            {
                "q": plan.query_text,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
            },
        )
    raise KeyError(provider)


def _normalize_provider_payload(
    provider: str,
    payload: Mapping[str, object] | Sequence[object],
    *,
    plan: TickerQueryPlan,
    profile: SourceProfile,
    api_request_count: int,
    tracked_tickers: Sequence[TrackedTicker],
) -> list[Article]:
    normalized: list[Article] = []
    for item in _provider_items(provider, payload):
        if not isinstance(item, Mapping):
            continue
        try:
            article = _normalize_item(
                provider,
                item,
                plan=plan,
                profile=profile,
                api_request_count=api_request_count,
            )
        except (TypeError, ValueError):
            continue
        if article is None:
            continue
        matches = assess_ticker_matches(article, tracked_tickers)
        confidence = next(
            (match.confidence for match in matches if match.ticker == plan.ticker),
            float(article.metadata.get("ticker_match_confidence") or 0.0),
        )
        normalized.append(
            Article(
                canonical_url=article.canonical_url,
                title=article.title,
                article_id=article.article_id,
                published_at=article.published_at,
                full_text=article.full_text,
                snippet=article.snippet,
                metadata={
                    **article.metadata,
                    "ticker_match_confidence": confidence,
                    "article_type": classify_article_type(article).primary_type,
                },
                created_at=article.created_at,
            )
        )
    return normalized


def _provider_items(
    provider: str,
    payload: Mapping[str, object] | Sequence[object],
) -> Sequence[object]:
    if provider == "marketaux":
        items = payload.get("data", ()) if isinstance(payload, Mapping) else ()
    elif provider == "nyt":
        response = payload.get("response", {}) if isinstance(payload, Mapping) else {}
        items = response.get("docs", ()) if isinstance(response, Mapping) else ()
    elif provider == "finnhub_news":
        items = payload
    else:
        items = payload.get("articles", ()) if isinstance(payload, Mapping) else ()
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        return items
    return ()


def _normalize_item(
    provider: str,
    item: Mapping[str, object],
    *,
    plan: TickerQueryPlan,
    profile: SourceProfile,
    api_request_count: int,
) -> Article | None:
    if provider == "nyt":
        headline = item.get("headline") or {}
        title = str(
            headline.get("main") if isinstance(headline, Mapping) else ""
        ).strip()
        url = str(item.get("web_url") or "").strip()
        snippet = str(
            item.get("lead_paragraph")
            or item.get("abstract")
            or item.get("snippet")
            or ""
        ).strip()
        published_at = str(item.get("pub_date") or "") or None
        source = str(item.get("source") or "The New York Times")
        provider_id = str(item.get("_id") or "") or None
        company_specific = _company_specific(title, snippet, plan)
        symbols = (plan.ticker,) if company_specific else ()
        context_only = not company_specific
    elif provider == "finnhub_news":
        title = str(item.get("headline") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("summary") or "").strip()
        published_at = _finnhub_published_at(item.get("datetime"))
        source = str(item.get("source") or "Finnhub")
        provider_id = str(item.get("id") or "") or None
        symbols = (plan.ticker,)
        company_specific = True
        context_only = False
    else:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("description") or item.get("snippet") or "").strip()
        published_at = str(item.get("published_at") or item.get("publishedAt") or "") or None
        source_value = item.get("source") or {}
        source = (
            str(source_value.get("name") or source_value.get("id") or "")
            if isinstance(source_value, Mapping)
            else str(source_value or "")
        )
        provider_id = str(item.get("uuid") or item.get("id") or "") or None
        entities = item.get("entities") or ()
        entity_symbols = tuple(
            str(entity.get("symbol") or "").upper()
            for entity in entities
            if isinstance(entity, Mapping) and entity.get("symbol")
        )
        company_specific = _company_specific(title, snippet, plan)
        symbols = entity_symbols or ((plan.ticker,) if company_specific else ())
        context_only = False
    if not title or not url:
        return None
    return Article(
        canonical_url=canonicalize_url(url),
        title=title,
        article_id=provider_id,
        published_at=published_at,
        snippet=snippet or None,
        metadata={
            "provider": provider,
            "source_provider": provider,
            "source_id": profile.source_id,
            "source_name": source or profile.publisher_name,
            "publisher_name": source or profile.publisher_name,
            "source_family": profile.source_family,
            "source_quality_tier": profile.source_quality_tier,
            "source_priority": profile.source_priority,
            "extraction_priority": profile.extraction_priority,
            "fetch_allowed": profile.fetch_allowed,
            "extract_allowed": profile.extract_allowed,
            "paywall_likely": profile.paywall_likely,
            "symbols": list(symbols),
            "ticker_candidates": list(symbols),
            "query_id": plan.query_id,
            "query_text": plan.query_text,
            "api_provider": provider,
            "api_request_count": api_request_count,
            "external_api_used": True,
            "external_context_only": context_only,
            "ticker_match_confidence": 0.95 if company_specific else 0.45,
            "description": snippet or None,
            "direct_source": True,
        },
    )


def _company_specific(title: str, snippet: str, plan: TickerQueryPlan) -> bool:
    text = f"{title} {snippet}".casefold()
    return plan.company.casefold() in text or (
        plan.ticker.casefold() in text and "stock" in text
    )


def _finnhub_published_at(value: object) -> str | None:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return str(value or "") or None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _article_symbols(article: Article) -> tuple[str, ...]:
    symbols = article.metadata.get("symbols") or ()
    return tuple(str(symbol).upper() for symbol in symbols if symbol)


def _unique_articles(articles: Sequence[Article]) -> list[Article]:
    by_url: dict[str, Article] = {}
    for article in articles:
        by_url.setdefault(article.canonical_url, article)
    return list(by_url.values())


def _fetch_json(
    endpoint: str,
    params: Mapping[str, object],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Mapping[str, object] | Sequence[object]:
    query = urlencode(
        [(key, value) for key, value in params.items() if value is not None]
    )
    request = Request(
        f"{endpoint}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "StonkNewsPipeline/0.1",
            **headers,
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _provider_status_code(attempts: Sequence[ExternalApiAttempt]) -> int | None:
    successful = next(
        (attempt.status_code for attempt in attempts if attempt.status == "success"),
        None,
    )
    if successful is not None:
        return successful
    return next(
        (attempt.status_code for attempt in attempts if attempt.status_code is not None),
        None,
    )


def _provider_error_reason(attempts: Sequence[ExternalApiAttempt]) -> str | None:
    reasons = sorted(
        {
            str(attempt.error_message)
            for attempt in attempts
            if attempt.error_message
        }
    )
    return ",".join(reasons) if reasons else None


def _safe_external_error(error: Exception) -> str:
    if isinstance(error, HTTPError):
        return f"http_status_{error.code}"
    if isinstance(error, URLError):
        return f"network_error_{type(error.reason).__name__}"
    if isinstance(error, (TimeoutError, json.JSONDecodeError)):
        return type(error).__name__
    return "external_api_request_failed"
