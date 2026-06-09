"""Priority scheduler for source-aware backend article acquisition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from news_pipeline.article_fetch import URL_CLASS_DIRECT_PUBLISHER, classify_article_url
from news_pipeline.models import Article
from news_pipeline.ticker_matching import assess_ticker_matches
from news_pipeline.tickers import TrackedTicker

from .acquisition_scoring import annotate_acquisition_scores, source_diversity_metrics
from .compliance import collection_decision
from .live_rss import LiveRssAttempt, collect_live_rss_articles
from .rss_config import RssFeedConfig, RssSourceFamilyConfig
from .sec_edgar import SecCollectionAttempt, collect_sec_edgar_articles
from .source_registry import (
    COMPANY_IR,
    DIRECT_NEWS_PUBLISHER,
    GOOGLE_NEWS_BACKSTOP,
    MARKET_DATA_OR_ANALYSIS,
    PAID_NEWS_API,
    PRESS_RELEASE_WIRE,
    REGULATORY_OFFICIAL,
    SOURCE_FAMILY_ORDER,
    CompanyIrProfile,
    SourceProfile,
)


PaidCollector = Callable[[Sequence[TrackedTicker]], Sequence[Article]]
SecCollector = Callable[..., tuple[list[Article], list[SecCollectionAttempt]]]
RssCollector = Callable[..., tuple[list[Article], list[LiveRssAttempt]]]


@dataclass(frozen=True)
class SourceAcquisitionAttempt:
    provider: str
    source_family: str
    source_id: str
    status: str
    article_count: int
    latency_ms: int = 0
    error_class: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "source_family": self.source_family,
            "source_id": self.source_id,
            "status": self.status,
            "article_count": self.article_count,
            "latency_ms": self.latency_ms,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SourceScheduleResult:
    articles: tuple[Article, ...]
    attempts: tuple[SourceAcquisitionAttempt, ...]
    diagnostics: Mapping[str, object]


def schedule_sources(
    *,
    profiles: Sequence[SourceProfile],
    tracked_tickers: Sequence[TrackedTicker],
    company_ir_profiles: Mapping[str, CompanyIrProfile],
    run_date: str,
    user_agent: str,
    timeout_seconds: float,
    retries: int,
    target_backend_articles: int,
    minimum_backend_articles: int,
    max_backend_articles: int,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
    max_google_news_share: float,
    include_press_release_feeds: bool,
    include_sec_feeds: bool,
    paid_api_global_enabled: bool,
    paid_provider_flags: Mapping[str, bool],
    environ: Mapping[str, str],
    quota_guard_allows: bool = True,
    rss_collector: RssCollector = collect_live_rss_articles,
    sec_collector: SecCollector = collect_sec_edgar_articles,
    paid_collectors: Mapping[str, PaidCollector] | None = None,
) -> SourceScheduleResult:
    profile_by_id = {profile.source_id: profile for profile in profiles}
    enabled = [
        profile
        for profile in profiles
        if profile.enabled_by_default
        and (include_press_release_feeds or profile.source_family != PRESS_RELEASE_WIRE)
        and (include_sec_feeds or profile.source_family != REGULATORY_OFFICIAL)
    ]
    articles: list[Article] = []
    attempts: list[SourceAcquisitionAttempt] = []
    failed_profiles: set[str] = set()
    missing_ir_profiles = [
        ticker.symbol
        for ticker in tracked_tickers
        if ticker.symbol not in company_ir_profiles
    ]
    paid_skips: dict[str, str] = {}

    for family in SOURCE_FAMILY_ORDER:
        family_profiles = [
            profile for profile in enabled if profile.source_family == family
        ]
        if family == REGULATORY_OFFICIAL:
            if not family_profiles:
                continue
            sec_articles, sec_attempts = sec_collector(
                tracked_tickers,
                run_date=run_date,
                user_agent=user_agent,
                timeout_seconds=timeout_seconds,
            )
            articles.extend(sec_articles)
            attempts.extend(_sec_attempt_rows(sec_attempts))
            if sec_attempts and not any(row.status == "success" for row in sec_attempts):
                failed_profiles.add("sec_edgar")
        elif family == COMPANY_IR:
            ir_profiles = [
                company_ir_profiles[ticker.symbol]
                for ticker in tracked_tickers
                if ticker.symbol in company_ir_profiles
            ]
            rss_profiles = _company_ir_source_profiles(ir_profiles)
            fetched, rows = _collect_rss_profiles(
                rss_profiles,
                family=COMPANY_IR,
                timeout_seconds=timeout_seconds,
                retries=retries,
                user_agent=user_agent,
                max_articles_per_source=max_articles_per_source,
                max_articles_per_ticker=max_articles_per_ticker,
                max_total_articles=max_backend_articles,
                rss_collector=rss_collector,
            )
            articles.extend(fetched)
            attempts.extend(rows)
        elif family in {
            PRESS_RELEASE_WIRE,
            DIRECT_NEWS_PUBLISHER,
            MARKET_DATA_OR_ANALYSIS,
        }:
            fetched, rows = _collect_rss_profiles(
                family_profiles,
                family=family,
                timeout_seconds=timeout_seconds,
                retries=retries,
                user_agent=user_agent,
                max_articles_per_source=max_articles_per_source,
                max_articles_per_ticker=max_articles_per_ticker,
                max_total_articles=max_backend_articles,
                rss_collector=rss_collector,
            )
            articles.extend(fetched)
            attempts.extend(rows)
            failed_profiles.update(
                row.source_id for row in rows if row.status == "failure"
            )
        elif family == PAID_NEWS_API:
            paid_articles, paid_attempts, paid_skipped = _collect_paid_profiles(
                [profile for profile in profiles if profile.source_family == PAID_NEWS_API],
                tracked_tickers=tracked_tickers,
                global_enabled=paid_api_global_enabled,
                provider_flags=paid_provider_flags,
                environ=environ,
                quota_guard_allows=quota_guard_allows,
                collectors=paid_collectors or {},
            )
            articles.extend(paid_articles)
            attempts.extend(paid_attempts)
            paid_skips.update(paid_skipped)
        elif family == GOOGLE_NEWS_BACKSTOP:
            google_budget = _google_backstop_budget(
                non_google_count=len(_unique_articles(articles)),
                target_backend_articles=target_backend_articles,
                minimum_backend_articles=minimum_backend_articles,
                max_backend_articles=max_backend_articles,
                max_google_news_share=max_google_news_share,
            )
            if google_budget <= 0:
                continue
            fetched, rows = _collect_rss_profiles(
                family_profiles,
                family=GOOGLE_NEWS_BACKSTOP,
                timeout_seconds=timeout_seconds,
                retries=retries,
                user_agent=user_agent,
                max_articles_per_source=max_articles_per_source,
                max_articles_per_ticker=max_articles_per_ticker,
                max_total_articles=google_budget,
                rss_collector=rss_collector,
            )
            articles.extend(fetched)
            attempts.extend(rows)

    annotated = annotate_acquisition_scores(
        _unique_articles(articles),
        profile_by_id,
    )
    selected = _cap_backend_pool(
        annotated,
        max_backend_articles=max_backend_articles,
        max_articles_per_source=max_articles_per_source,
        max_articles_per_ticker=max_articles_per_ticker,
    )
    diagnostics = _source_diagnostics(
        selected,
        profiles=profiles,
        enabled_profiles=enabled,
        failed_profiles=failed_profiles,
        attempts=attempts,
        missing_ir_profiles=missing_ir_profiles,
        paid_skips=paid_skips,
        max_google_news_share=max_google_news_share,
        target_backend_articles=target_backend_articles,
        minimum_backend_articles=minimum_backend_articles,
        max_backend_articles=max_backend_articles,
    )
    return SourceScheduleResult(
        articles=tuple(selected),
        attempts=tuple(attempts),
        diagnostics=diagnostics,
    )


def paid_provider_decision(
    profile: SourceProfile,
    *,
    global_enabled: bool,
    provider_enabled: bool,
    environ: Mapping[str, str],
    quota_guard_allows: bool,
    collector_available: bool,
) -> str | None:
    if not global_enabled:
        return "global_paid_api_flag_disabled"
    if not provider_enabled:
        return "provider_flag_disabled"
    if profile.api_key_env_var and not environ.get(profile.api_key_env_var):
        return "missing_api_key"
    if not quota_guard_allows:
        return "quota_or_budget_guard_blocked"
    if not collector_available:
        return "adapter_not_configured_for_live_calls"
    return None


def _collect_paid_profiles(
    profiles: Sequence[SourceProfile],
    *,
    tracked_tickers: Sequence[TrackedTicker],
    global_enabled: bool,
    provider_flags: Mapping[str, bool],
    environ: Mapping[str, str],
    quota_guard_allows: bool,
    collectors: Mapping[str, PaidCollector],
) -> tuple[list[Article], list[SourceAcquisitionAttempt], dict[str, str]]:
    articles: list[Article] = []
    attempts: list[SourceAcquisitionAttempt] = []
    skipped: dict[str, str] = {}
    for profile in profiles:
        collector = collectors.get(profile.source_id)
        reason = paid_provider_decision(
            profile,
            global_enabled=global_enabled,
            provider_enabled=bool(provider_flags.get(profile.source_id)),
            environ=environ,
            quota_guard_allows=quota_guard_allows,
            collector_available=collector is not None,
        )
        if reason:
            skipped[profile.source_id] = reason
            attempts.append(
                SourceAcquisitionAttempt(
                    provider=profile.source_id,
                    source_family=PAID_NEWS_API,
                    source_id=profile.source_id,
                    status="skipped",
                    article_count=0,
                    metadata={"reason": reason},
                )
            )
            continue
        try:
            fetched = [
                _annotate_profile(article, profile)
                for article in collector(tracked_tickers)
            ]
            articles.extend(fetched)
            attempts.append(
                SourceAcquisitionAttempt(
                    provider=profile.source_id,
                    source_family=PAID_NEWS_API,
                    source_id=profile.source_id,
                    status="success",
                    article_count=len(fetched),
                )
            )
        except Exception as exc:  # noqa: BLE001 - paid provider failures are isolated.
            attempts.append(
                SourceAcquisitionAttempt(
                    provider=profile.source_id,
                    source_family=PAID_NEWS_API,
                    source_id=profile.source_id,
                    status="failure",
                    article_count=0,
                    error_class=type(exc).__name__,
                    error_message=str(exc)[:200] or None,
                )
            )
    return articles, attempts, skipped


def _collect_rss_profiles(
    profiles: Sequence[SourceProfile],
    *,
    family: str,
    timeout_seconds: float,
    retries: int,
    user_agent: str,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
    max_total_articles: int,
    rss_collector: RssCollector,
) -> tuple[list[Article], list[SourceAcquisitionAttempt]]:
    allowed_profiles: list[SourceProfile] = []
    compliance_attempts: list[SourceAcquisitionAttempt] = []
    for profile in profiles:
        candidate_url = (
            profile.feed_urls[0]
            if profile.feed_urls
            else profile.ticker_query_templates[0]
            if profile.ticker_query_templates
            else f"https://{profile.domain}/"
        )
        method = (
            "ticker_rss_search"
            if profile.source_id == "google_news_rss_search"
            else "rss"
        )
        decision = collection_decision(
            profile,
            discovery_method=method,
            url=candidate_url,
            user_agent=user_agent,
        )
        if decision.allowed:
            allowed_profiles.append(profile)
        else:
            compliance_attempts.append(
                SourceAcquisitionAttempt(
                    provider=profile.source_id,
                    source_family=family,
                    source_id=profile.source_id,
                    status="skipped",
                    article_count=0,
                    metadata={"reason": decision.reason},
                )
            )
    source_families = tuple(
        RssSourceFamilyConfig(
            name=profile.source_id,
            display_name=profile.publisher_name,
            category=family,
            feeds=tuple(
                RssFeedConfig(
                    feed_id=f"{profile.source_id}_{index}",
                    url=url,
                    publisher_name=profile.publisher_name,
                )
                for index, url in enumerate(profile.feed_urls, start=1)
            ),
            ticker_search=profile.source_id == "google_news_rss_search",
        )
        for profile in allowed_profiles
        if profile.feed_urls or profile.source_id == "google_news_rss_search"
    )
    if not source_families:
        return [], compliance_attempts
    fetched, rss_attempts = rss_collector(
        source_families=source_families,
        timeout_seconds=timeout_seconds,
        retries=retries,
        user_agent=user_agent,
        max_articles_per_source=max_articles_per_source,
        max_articles_per_ticker=max_articles_per_ticker,
        max_total_articles=max_total_articles,
        prefer_direct_sources=True,
        max_google_news_share=1.0,
        include_press_release_feeds=True,
    )
    profile_by_id = {profile.source_id: profile for profile in allowed_profiles}
    annotated = [
        _annotate_profile(article, profile_by_id[str(article.metadata.get("provider"))])
        if str(article.metadata.get("provider")) in profile_by_id
        else article
        for article in fetched
    ]
    rows = [
        SourceAcquisitionAttempt(
            provider=attempt.provider,
            source_family=attempt.source_family,
            source_id=attempt.provider,
            status=attempt.status,
            article_count=attempt.article_count,
            latency_ms=attempt.latency_ms,
            error_class=attempt.error_class,
            error_message=attempt.error_message,
            metadata={
                "feed_id": attempt.feed_id,
                "feed_url": attempt.feed_url,
                "attempts": attempt.attempts,
                "fetched_article_count": attempt.fetched_article_count,
            },
        )
        for attempt in rss_attempts
    ]
    return annotated, compliance_attempts + rows


def _company_ir_source_profiles(
    profiles: Sequence[CompanyIrProfile],
) -> tuple[SourceProfile, ...]:
    return tuple(
        SourceProfile(
            source_id=f"company_ir_{profile.ticker.lower()}",
            source_family=COMPANY_IR,
            publisher_name=f"{profile.company_name} Investor Relations",
            domain="",
            source_quality_tier=1,
            enabled_by_default=True,
            discovery_methods=("rss",),
            feed_urls=(profile.ir_rss_url,) if profile.ir_rss_url else (),
            source_priority=95,
            extraction_priority=95,
        )
        for profile in profiles
        if profile.ir_rss_url
    )


def _sec_attempt_rows(
    attempts: Sequence[SecCollectionAttempt],
) -> list[SourceAcquisitionAttempt]:
    return [
        SourceAcquisitionAttempt(
            provider="sec_edgar",
            source_family=REGULATORY_OFFICIAL,
            source_id="sec_edgar",
            status=attempt.status,
            article_count=attempt.article_count,
            error_class=attempt.error_class,
            error_message=attempt.error_message,
            metadata={"ticker": attempt.ticker},
        )
        for attempt in attempts
    ]


def _annotate_profile(article: Article, profile: SourceProfile) -> Article:
    issuer_promotional = profile.source_family == PRESS_RELEASE_WIRE
    return Article(
        canonical_url=article.canonical_url,
        title=article.title,
        article_id=article.article_id,
        published_at=article.published_at,
        full_text=article.full_text,
        snippet=article.snippet,
        metadata={
            **article.metadata,
            "provider": profile.source_id,
            "source_id": profile.source_id,
            "source_family": profile.source_family,
            "publisher_name": profile.publisher_name,
            "source_name": article.metadata.get("source_name") or profile.publisher_name,
            "source_quality_tier": profile.source_quality_tier,
            "source_priority": profile.source_priority,
            "extraction_priority": profile.extraction_priority,
            "fetch_allowed": profile.fetch_allowed,
            "extract_allowed": profile.extract_allowed,
            "paywall_likely": profile.paywall_likely,
            "issuer_promotional": issuer_promotional,
            "direct_source": profile.source_family != GOOGLE_NEWS_BACKSTOP,
        },
        created_at=article.created_at,
    )


def _google_backstop_budget(
    *,
    non_google_count: int,
    target_backend_articles: int,
    minimum_backend_articles: int,
    max_backend_articles: int,
    max_google_news_share: float,
) -> int:
    maximum = max(1, max_backend_articles)
    target = min(maximum, max(1, target_backend_articles))
    minimum = min(target, max(1, minimum_backend_articles))
    if non_google_count < minimum:
        return min(maximum - non_google_count, target - non_google_count)
    if non_google_count < target:
        return min(maximum - non_google_count, target - non_google_count)
    share = max(0.0, min(1.0, max_google_news_share))
    if share <= 0:
        return 0
    if share >= 1:
        return maximum - non_google_count
    ratio_limit = int(non_google_count * share / (1.0 - share))
    return max(0, min(ratio_limit, maximum - non_google_count))


def _cap_backend_pool(
    articles: Sequence[Article],
    *,
    max_backend_articles: int,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
) -> list[Article]:
    selected: list[Article] = []
    source_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    ordered = sorted(
        articles,
        key=lambda article: (
            float(article.metadata.get("acquisition_score") or 0.0),
            str(article.published_at or ""),
        ),
        reverse=True,
    )
    for article in ordered:
        if len(selected) >= max(1, max_backend_articles):
            break
        source_id = str(article.metadata.get("source_id") or "unknown")
        if source_counts[source_id] >= max(1, max_articles_per_source):
            continue
        tickers = _article_tickers(article)
        if tickers and any(
            ticker_counts[ticker] >= max(1, max_articles_per_ticker)
            for ticker in tickers
        ):
            continue
        selected.append(article)
        source_counts[source_id] += 1
        for ticker in tickers:
            ticker_counts[ticker] += 1
    return selected


def _unique_articles(articles: Sequence[Article]) -> list[Article]:
    by_url: dict[str, Article] = {}
    for article in articles:
        existing = by_url.get(article.canonical_url)
        if existing is None:
            by_url[article.canonical_url] = article
            continue
        existing_priority = int(existing.metadata.get("source_priority") or 0)
        candidate_priority = int(article.metadata.get("source_priority") or 0)
        if candidate_priority > existing_priority:
            by_url[article.canonical_url] = article
    return list(by_url.values())


def _article_tickers(article: Article) -> tuple[str, ...]:
    explicit = article.metadata.get("symbols")
    if isinstance(explicit, (list, tuple)):
        symbols = tuple(str(symbol).upper() for symbol in explicit if symbol)
        if symbols:
            return symbols
    ticker = str(article.metadata.get("ticker") or "").upper()
    if ticker:
        return (ticker,)
    return tuple(match.ticker for match in assess_ticker_matches(article))


def _source_diagnostics(
    articles: Sequence[Article],
    *,
    profiles: Sequence[SourceProfile],
    enabled_profiles: Sequence[SourceProfile],
    failed_profiles: set[str],
    attempts: Sequence[SourceAcquisitionAttempt],
    missing_ir_profiles: Sequence[str],
    paid_skips: Mapping[str, str],
    max_google_news_share: float,
    target_backend_articles: int,
    minimum_backend_articles: int,
    max_backend_articles: int,
) -> dict[str, object]:
    family_counts = Counter(
        str(article.metadata.get("source_family") or "unknown")
        for article in articles
    )
    source_counts = Counter(
        str(article.metadata.get("source_id") or article.metadata.get("provider") or "unknown")
        for article in articles
    )
    publisher_counts = Counter(
        str(
            article.metadata.get("source_name")
            or article.metadata.get("publisher_name")
            or article.metadata.get("provider")
            or "unknown"
        )
        for article in articles
    )
    google_count = family_counts[GOOGLE_NEWS_BACKSTOP]
    direct_count = (
        family_counts[DIRECT_NEWS_PUBLISHER]
        + family_counts[REGULATORY_OFFICIAL]
        + family_counts[COMPANY_IR]
        + family_counts[PRESS_RELEASE_WIRE]
    )
    total = len(articles)
    direct_url_count = sum(
        classify_article_url(article.canonical_url) == URL_CLASS_DIRECT_PUBLISHER
        for article in articles
    )
    google_share = google_count / total if total else 0.0
    paid_count = family_counts[PAID_NEWS_API]
    diversity = source_diversity_metrics(articles)
    return {
        "source_family_counts": dict(sorted(family_counts.items())),
        "articles_by_source_id": dict(sorted(source_counts.items())),
        "articles_by_source_family": dict(sorted(family_counts.items())),
        "direct_vs_aggregator_ratio": round(direct_count / total, 4) if total else 0.0,
        "direct_publisher_url_ratio": round(direct_url_count / total, 4) if total else 0.0,
        "google_news_backstop_count": google_count,
        "google_news_article_count": google_count,
        "google_news_share": round(google_share, 4),
        "google_news_share_capped": google_share <= max_google_news_share + 0.0001,
        "google_news_share_cap_relaxed_for_minimum": (
            google_share > max_google_news_share + 0.0001
            and (total - google_count) < minimum_backend_articles
        ),
        "source_pool_google_dominated": google_share > 0.5,
        "official_source_count": family_counts[REGULATORY_OFFICIAL],
        "company_ir_count": family_counts[COMPANY_IR],
        "press_release_wire_count": family_counts[PRESS_RELEASE_WIRE],
        "direct_publisher_count": family_counts[DIRECT_NEWS_PUBLISHER],
        "direct_source_article_count": total - google_count - paid_count,
        "paid_api_count": paid_count,
        "paid_api_status": "enabled" if paid_count else "disabled_or_skipped",
        "paid_api_skipped_reasons": dict(sorted(paid_skips.items())),
        "missing_company_ir_profiles": list(missing_ir_profiles),
        "source_profiles_loaded": len(profiles),
        "source_profiles_enabled": len(enabled_profiles),
        "source_profiles_failed": sorted(failed_profiles),
        "source_profiles": [profile.as_dict() for profile in profiles],
        "source_diversity_score": diversity["source_diversity_score"],
        "source_balance_score": diversity["source_balance_score"],
        "top_direct_source_publishers": _top_publishers_excluding_family(
            articles,
            GOOGLE_NEWS_BACKSTOP,
        ),
        "top_google_news_publishers": _top_publishers_for_family(
            articles,
            GOOGLE_NEWS_BACKSTOP,
        ),
        "target_backend_articles": target_backend_articles,
        "minimum_backend_articles": minimum_backend_articles,
        "max_backend_articles": max_backend_articles,
        "source_scheduler_family_order": list(SOURCE_FAMILY_ORDER),
        "attempt_count": len(attempts),
        "success_count": sum(row.status == "success" for row in attempts),
        "failure_count": sum(row.status == "failure" for row in attempts),
        "attempts": [row.as_dict() for row in attempts],
    }


def _top_publishers_for_family(
    articles: Sequence[Article],
    source_family: str,
) -> dict[str, int]:
    counts = Counter(
        str(
            article.metadata.get("source_name")
            or article.metadata.get("publisher_name")
            or article.metadata.get("provider")
            or "unknown"
        )
        for article in articles
        if str(article.metadata.get("source_family") or "") == source_family
    )
    return dict(counts.most_common(10))


def _top_publishers_excluding_family(
    articles: Sequence[Article],
    excluded_family: str,
) -> dict[str, int]:
    counts = Counter(
        str(
            article.metadata.get("source_name")
            or article.metadata.get("publisher_name")
            or article.metadata.get("provider")
            or "unknown"
        )
        for article in articles
        if str(article.metadata.get("source_family") or "") != excluded_family
    )
    return dict(counts.most_common(10))
