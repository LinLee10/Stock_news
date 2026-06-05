"""Source quality tiers and conservative article filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from urllib.parse import urlparse

from .models import Article
from .tickers import match_tickers


TIER_1_HIGH_TRUST = 1
TIER_2_USABLE = 2
TIER_3_LOW_PRIORITY = 3
TIER_4_EXCLUDE_BY_DEFAULT = 4
DEFAULT_MIN_SOURCE_QUALITY_TIER = TIER_3_LOW_PRIORITY
MIN_BETTER_SOURCES_PER_TICKER = 2

TIER_LABELS = {
    TIER_1_HIGH_TRUST: "tier_1_high_trust",
    TIER_2_USABLE: "tier_2_usable",
    TIER_3_LOW_PRIORITY: "tier_3_low_priority",
    TIER_4_EXCLUDE_BY_DEFAULT: "tier_4_exclude_by_default",
}

TIER_1_NAMES = {
    "associated press",
    "ap",
    "barron's",
    "barrons",
    "bloomberg",
    "cnbc",
    "financial times",
    "investor's business daily",
    "investors business daily",
    "marketwatch",
    "reuters",
    "sec",
    "wall street journal",
    "wsj",
}
TIER_1_DOMAINS = {
    "apnews.com",
    "barrons.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "investors.com",
    "marketwatch.com",
    "reuters.com",
    "sec.gov",
    "wsj.com",
}
INVESTOR_RELATIONS_MARKERS = (
    "investor relations",
    "ir.",
    "/ir/",
    "/investors",
    "/investor-relations",
)

TIER_2_NAMES = {
    "barchart",
    "gurufocus",
    "investing.com",
    "marketbeat",
    "morningstar",
    "seeking alpha",
    "simply wall st",
    "stock titan",
    "stockstory",
    "the motley fool",
    "tipranks",
    "tradingview",
    "yahoo finance",
    "zacks",
}
TIER_2_DOMAINS = {
    "barchart.com",
    "finance.yahoo.com",
    "fool.com",
    "gurufocus.com",
    "investing.com",
    "marketbeat.com",
    "morningstar.com",
    "seekingalpha.com",
    "simplywall.st",
    "stocktitan.net",
    "stockstory.org",
    "tipranks.com",
    "tradingview.com",
    "zacks.com",
}

TIER_3_NAMES = {
    "chartmill",
    "gotrade",
    "marketscreener",
    "moomoo",
    "quiver quantitative",
    "stocktwits",
    "trefis",
}
TIER_3_DOMAINS = {
    "chartmill.com",
    "gotrade.com",
    "marketscreener.com",
    "moomoo.com",
    "quiverquant.com",
    "stocktwits.com",
    "trefis.com",
}

TIER_4_NAMES = {"mshale"}
TIER_4_DOMAINS = {"foreignpolicyjournal.com", "mshale.com"}
UNRELATED_TERMS = {
    "baseball",
    "basketball",
    "celebrity",
    "cricket",
    "football",
    "hollywood",
    "nfl",
    "nba",
    "soccer",
    "tennis",
}
SPAM_PATTERNS = (
    re.compile(r"\bwatch\s+now\b", re.I),
    re.compile(r"\bfree\s+crypto\b", re.I),
    re.compile(r"\bshocking\b", re.I),
    re.compile(r"\bmust\s+see\b", re.I),
    re.compile(r"\byou\s+won'?t\s+believe\b", re.I),
    re.compile(r"\bai[- ]generated\s+video\b", re.I),
)
RANDOM_EVENT_TERMS = {
    "award",
    "concert",
    "festival",
    "game",
    "match",
    "movie",
    "parade",
    "wedding",
}


@dataclass(frozen=True)
class SourceQuality:
    tier: int
    label: str
    reason: str
    publisher: str
    domain: str
    excluded_by_default: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "label": self.label,
            "reason": self.reason,
            "publisher": self.publisher,
            "domain": self.domain,
            "excluded_by_default": self.excluded_by_default,
        }


@dataclass(frozen=True)
class SourceQualitySummary:
    total_articles: int = 0
    visible_articles: int = 0
    excluded_articles: int = 0
    low_priority_visible_articles: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    visible_tier_counts: dict[str, int] = field(default_factory=dict)
    excluded_tier_counts: dict[str, int] = field(default_factory=dict)
    excluded_sources: tuple[str, ...] = ()
    min_source_quality_tier: int = DEFAULT_MIN_SOURCE_QUALITY_TIER
    include_low_quality_sources: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "total_articles": self.total_articles,
            "visible_articles": self.visible_articles,
            "excluded_articles": self.excluded_articles,
            "low_priority_visible_articles": self.low_priority_visible_articles,
            "tier_counts": dict(self.tier_counts),
            "visible_tier_counts": dict(self.visible_tier_counts),
            "excluded_tier_counts": dict(self.excluded_tier_counts),
            "excluded_sources": list(self.excluded_sources),
            "min_source_quality_tier": self.min_source_quality_tier,
            "include_low_quality_sources": self.include_low_quality_sources,
        }


@dataclass(frozen=True)
class SourceQualityFilterResult:
    visible_articles: tuple[Article, ...]
    excluded_articles: tuple[Article, ...]
    summary: SourceQualitySummary
    diagnostics: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary.as_dict(),
            "diagnostics": list(self.diagnostics),
        }


def assess_article_source(article: Article) -> SourceQuality:
    publisher = _publisher(article)
    domain = _domain(article.canonical_url)
    title = article.title
    title_reason = _title_exclusion_reason(title)
    if title_reason:
        return _quality(TIER_4_EXCLUDE_BY_DEFAULT, title_reason, publisher, domain)
    if _matches_any(publisher, TIER_4_NAMES) or domain in TIER_4_DOMAINS:
        return _quality(TIER_4_EXCLUDE_BY_DEFAULT, "excluded_source", publisher, domain)
    if _is_investor_relations(article):
        return _quality(TIER_1_HIGH_TRUST, "company_investor_relations", publisher, domain)
    if _matches_any(publisher, TIER_1_NAMES) or domain in TIER_1_DOMAINS:
        return _quality(TIER_1_HIGH_TRUST, "trusted_publisher", publisher, domain)
    if _matches_any(publisher, TIER_2_NAMES) or domain in TIER_2_DOMAINS:
        return _quality(TIER_2_USABLE, "usable_financial_source", publisher, domain)
    if _matches_any(publisher, TIER_3_NAMES) or domain in TIER_3_DOMAINS:
        return _quality(TIER_3_LOW_PRIORITY, "low_priority_source", publisher, domain)
    return _quality(TIER_2_USABLE, "unlisted_source", publisher, domain)


def annotate_article_quality(article: Article) -> Article:
    quality = assess_article_source(article)
    return Article(
        canonical_url=article.canonical_url,
        title=article.title,
        article_id=article.article_id,
        published_at=article.published_at,
        full_text=article.full_text,
        snippet=article.snippet,
        metadata={
            **article.metadata,
            "source_quality_tier": quality.tier,
            "source_quality_label": quality.label,
            "source_quality_reason": quality.reason,
            "source_quality_domain": quality.domain,
            "source_quality_publisher": quality.publisher,
        },
        created_at=article.created_at,
    )


def filter_articles_by_source_quality(
    articles: list[Article] | tuple[Article, ...],
    *,
    include_low_quality_sources: bool = False,
    min_source_quality_tier: int = DEFAULT_MIN_SOURCE_QUALITY_TIER,
) -> SourceQualityFilterResult:
    annotated = tuple(annotate_article_quality(article) for article in articles)
    min_tier = _normalize_min_tier(min_source_quality_tier)
    better_counts_by_ticker = _better_source_counts_by_ticker(annotated, min_tier)
    visible: list[Article] = []
    excluded: list[Article] = []
    diagnostics: list[dict[str, object]] = []

    for article in annotated:
        quality = assess_article_source(article)
        decision, reason = _visibility_decision(
            article,
            quality,
            better_counts_by_ticker,
            include_low_quality_sources=include_low_quality_sources,
            min_source_quality_tier=min_tier,
        )
        if decision:
            visible.append(article)
        else:
            excluded.append(article)
        diagnostics.append(
            {
                "title": article.title,
                "canonical_url": article.canonical_url,
                "source_name": _publisher(article),
                "source_quality": quality.as_dict(),
                "visible": decision,
                "decision_reason": reason,
            }
        )

    summary = _summary(
        annotated,
        visible,
        excluded,
        min_source_quality_tier=min_tier,
        include_low_quality_sources=include_low_quality_sources,
    )
    return SourceQualityFilterResult(tuple(visible), tuple(excluded), summary, tuple(diagnostics))


def source_quality_sort_key(article: Article) -> tuple[int, int, str, str]:
    quality = assess_article_source(article)
    direct_rank = 0 if _is_direct_publisher_url(article.canonical_url) else 1
    return (quality.tier, direct_rank, article.title, article.canonical_url)


def source_quality_link_sort_key(title: str, url: str, source: str | None = None) -> tuple[int, str, str]:
    quality = assess_article_source(
        Article(
            canonical_url=url,
            title=title,
            metadata={"source_name": source or ""},
        )
    )
    return (quality.tier, title, url)


def quality_label_for_article(article: Article) -> str:
    return assess_article_source(article).label


def _visibility_decision(
    article: Article,
    quality: SourceQuality,
    better_counts_by_ticker: dict[str, int],
    *,
    include_low_quality_sources: bool,
    min_source_quality_tier: int,
) -> tuple[bool, str]:
    if include_low_quality_sources:
        return True, "included_by_flag"
    if quality.tier > min_source_quality_tier:
        return False, "below_min_source_quality_tier"
    if quality.tier == TIER_4_EXCLUDE_BY_DEFAULT:
        return False, "tier_4_excluded_by_default"
    if quality.tier == TIER_3_LOW_PRIORITY:
        tickers = _article_tickers(article)
        if not tickers:
            return False, "tier_3_without_ticker"
        if any(better_counts_by_ticker.get(ticker, 0) >= MIN_BETTER_SOURCES_PER_TICKER for ticker in tickers):
            return False, "tier_3_deprioritized_by_better_sources"
    return True, "visible"


def _better_source_counts_by_ticker(articles: tuple[Article, ...], min_tier: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for article in articles:
        quality = assess_article_source(article)
        if quality.tier > min_tier or quality.tier >= TIER_3_LOW_PRIORITY:
            continue
        for ticker in _article_tickers(article):
            counts[ticker] = counts.get(ticker, 0) + 1
    return counts


def _summary(
    all_articles: tuple[Article, ...],
    visible: list[Article],
    excluded: list[Article],
    *,
    min_source_quality_tier: int,
    include_low_quality_sources: bool,
) -> SourceQualitySummary:
    return SourceQualitySummary(
        total_articles=len(all_articles),
        visible_articles=len(visible),
        excluded_articles=len(excluded),
        low_priority_visible_articles=sum(
            1 for article in visible if assess_article_source(article).tier == TIER_3_LOW_PRIORITY
        ),
        tier_counts=_tier_counts(all_articles),
        visible_tier_counts=_tier_counts(visible),
        excluded_tier_counts=_tier_counts(excluded),
        excluded_sources=tuple(sorted({_publisher(article) or _domain(article.canonical_url) for article in excluded})),
        min_source_quality_tier=min_source_quality_tier,
        include_low_quality_sources=include_low_quality_sources,
    )


def _tier_counts(articles) -> dict[str, int]:
    counts = {label: 0 for label in TIER_LABELS.values()}
    for article in articles:
        label = assess_article_source(article).label
        counts[label] = counts.get(label, 0) + 1
    return counts


def _quality(tier: int, reason: str, publisher: str, domain: str) -> SourceQuality:
    return SourceQuality(
        tier=tier,
        label=TIER_LABELS[tier],
        reason=reason,
        publisher=publisher,
        domain=domain,
        excluded_by_default=tier == TIER_4_EXCLUDE_BY_DEFAULT,
    )


def _publisher(article: Article) -> str:
    source_name = str(article.metadata.get("source_name") or "").strip()
    provider = str(article.metadata.get("provider") or "").strip()
    return source_name or provider


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    return host


def _matches_any(value: str, names: set[str]) -> bool:
    normalized = value.casefold()
    return any(name in normalized for name in names)


def _is_investor_relations(article: Article) -> bool:
    text = f"{article.canonical_url} {_publisher(article)} {article.title}".casefold()
    return any(marker in text for marker in INVESTOR_RELATIONS_MARKERS)


def _title_exclusion_reason(title: str) -> str | None:
    lowered = title.casefold()
    if any(term in lowered for term in UNRELATED_TERMS):
        return "unrelated_sports_or_celebrity_terms"
    if any(pattern.search(title) for pattern in SPAM_PATTERNS):
        return "obvious_spam_pattern"
    if _looks_like_random_event_title(lowered):
        return "multiple_unrelated_entities_random_event_words"
    return None


def _looks_like_random_event_title(lowered_title: str) -> bool:
    ticker_like = len(re.findall(r"\b[A-Z]{2,5}\b", lowered_title.upper()))
    event_terms = sum(1 for term in RANDOM_EVENT_TERMS if term in lowered_title)
    return ticker_like >= 3 and event_terms >= 1


def _article_tickers(article: Article) -> tuple[str, ...]:
    text = " ".join(part for part in (article.title, article.snippet or "") if part)
    return tuple(sorted({ticker.symbol for ticker in match_tickers(text)}))


def _is_direct_publisher_url(url: str) -> bool:
    host = _domain(url)
    return bool(host) and host not in {"news.google.com", "google.com"}


def _normalize_min_tier(value: int) -> int:
    return max(TIER_1_HIGH_TRUST, min(TIER_4_EXCLUDE_BY_DEFAULT, int(value)))
