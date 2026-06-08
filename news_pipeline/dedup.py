"""URL canonicalization and deterministic dedupe keys."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import re
from typing import Callable, Iterable
from urllib.parse import parse_qsl, urlencode, unquote, urlparse, urlunparse

from .article_types import (
    ANALYST_RATING_OR_PRICE_TARGET,
    EARNINGS_OR_RESULTS,
    GENERIC_BUY_SELL_HOLD_OPINION,
    MACRO_OR_SECTOR_ROUNDUP,
    STOCK_PRICE_MOVE,
    classify_article_text,
    classify_article_type,
)
from .models import Article
from .ticker_matching import assess_ticker_matches, primary_ticker
from .tickers import match_tickers


TRACKING_PARAMS = {
    "_source",
    "campaign",
    "content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "medium",
    "ref",
    "source",
    "term",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
REDIRECT_HOSTS = {
    "google.com",
    "news.google.com",
    "finance.yahoo.com",
    "l.facebook.com",
    "lnkd.in",
}
REDIRECT_PARAMS = ("url", "u", "target")
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}
EVENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings_results": (
        "earnings",
        "results",
        "quarter",
        "quarterly",
        "revenue",
        "profit",
        "eps",
        "guidance",
        "sales",
    ),
    "analyst_rating_price_target": (
        "analyst",
        "rating",
        "ratings",
        "upgrade",
        "upgrades",
        "downgrade",
        "downgrades",
        "price target",
        "target",
        "initiates",
        "coverage",
        "opinion",
        "opinions",
    ),
    "stock_move": (
        "crash",
        "crashed",
        "drop",
        "dropped",
        "down",
        "fall",
        "fell",
        "plunge",
        "plunges",
        "rally",
        "rallies",
        "rise",
        "rises",
        "rose",
        "sink",
        "sinks",
        "soar",
        "soars",
        "surge",
        "surges",
        "up",
        "jumps",
        "popped",
        "tumbling",
    ),
    "product_ai_announcement": (
        "ai",
        "artificial intelligence",
        "chip",
        "chips",
        "product",
        "platform",
        "launch",
        "launches",
        "unveils",
        "announces",
        "architecture",
    ),
    "partnership_contract": (
        "partnership",
        "partner",
        "contract",
        "customer",
        "deal with",
        "collaboration",
    ),
    "acquisition_deal": (
        "acquisition",
        "acquire",
        "acquires",
        "buyout",
        "merger",
        "takeover",
    ),
    "regulatory_legal": (
        "lawsuit",
        "legal",
        "regulatory",
        "regulator",
        "probe",
        "investigation",
        "antitrust",
        "senate",
        "court",
    ),
    "general_buy_sell_opinion": (
        "buy",
        "sell",
        "hold",
        "should you",
        "best stocks",
        "stock to buy",
        "is it time",
        "before",
        "recommend",
        "undervalued",
        "overweight",
        "valuation",
    ),
}
CONFLICTING_EVENT_TYPES = {
    frozenset(("general_buy_sell_opinion", "stock_move")),
    frozenset(("analyst_rating_price_target", "product_ai_announcement")),
    frozenset(("analyst_rating_price_target", "regulatory_legal")),
    frozenset(("product_ai_announcement", "regulatory_legal")),
    frozenset(("earnings_results", "regulatory_legal")),
    frozenset(("acquisition_deal", "regulatory_legal")),
}


SemanticSimilarity = Callable[[Article, Article], float | None]


@dataclass(frozen=True)
class SourceLink:
    title: str
    url: str
    publisher: str | None
    provider: str | None
    published_at: str | None


@dataclass(frozen=True)
class DedupeCluster:
    canonical_article: Article
    alternate_source_links: tuple[str, ...]
    articles: tuple[Article, ...]
    duplicate_reasons: tuple[str, ...]
    primary_link: str
    supporting_links: tuple[SourceLink, ...]
    publisher_count: int
    source_count: int
    publisher_names: tuple[str, ...]
    source_providers: tuple[str, ...]
    cluster_id: str = ""
    primary_ticker: str | None = None
    matched_tickers: tuple[str, ...] = ()
    related_tickers: tuple[str, ...] = ()
    event_type: str = "unknown"
    primary_article_id: str = ""
    supporting_article_ids: tuple[str, ...] = ()
    supporting_publishers: tuple[str, ...] = ()
    source_diversity: int = 0
    publisher_diversity: int = 0


def canonicalize_url(url: str) -> str:
    """Normalize a URL for local dedupe without fetching it."""
    url = url.strip()
    if not url:
        return ""

    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc.startswith("m."):
        netloc = netloc[2:]
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    redirected = _safe_unwrap_redirect(netloc, parsed.query)
    if redirected:
        return canonicalize_url(redirected)

    params = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    params.sort(key=lambda item: (item[0].lower(), item[1]))

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    path = _cleanup_syndicated_path(path)

    return urlunparse((scheme, netloc, path, "", urlencode(params, doseq=True), ""))


def normalize_title(title: str) -> str:
    """Normalize a title for deterministic key generation."""
    title = re.sub(r"[\W_]+", " ", title.casefold())
    return re.sub(r"\s+", " ", title).strip()


def dedup_key(provider: str, url: str, title: str) -> str:
    """Create a stable local dedupe key from provider grouping, URL, and title."""
    material = "|".join(
        [
            provider.strip().casefold(),
            canonicalize_url(url),
            normalize_title(title),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def title_similarity(left: str, right: str) -> float:
    """Compare titles with normalized sequence and token overlap signals."""
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = _title_tokens(left_norm)
    right_tokens = _title_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return sequence_score
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(sequence_score, token_score)


def cluster_articles(
    articles: Iterable[Article],
    *,
    title_threshold: float = 0.82,
    semantic_similarity: SemanticSimilarity | None = None,
    semantic_threshold: float = 0.9,
) -> list[DedupeCluster]:
    """Cluster canonical articles without network or external dependencies."""
    clusters: list[_MutableCluster] = []

    for article in articles:
        normalized_article = _normalize_article_url(article)
        match: _MutableCluster | None = None
        reason: str | None = None

        for cluster in clusters:
            reason = _duplicate_reason(
                normalized_article,
                cluster.canonical_article,
                title_threshold=title_threshold,
                semantic_similarity=semantic_similarity,
                semantic_threshold=semantic_threshold,
            )
            if reason:
                match = cluster
                break

        if match is None:
            clusters.append(
                _MutableCluster(
                    canonical_article=normalized_article,
                    articles=[normalized_article],
                    duplicate_reasons=[],
                )
            )
        else:
            match.articles.append(normalized_article)
            match.duplicate_reasons.append(reason or "duplicate")

    return [cluster.to_public() for cluster in clusters]


@dataclass
class _MutableCluster:
    canonical_article: Article
    articles: list[Article]
    duplicate_reasons: list[str]

    def to_public(self) -> DedupeCluster:
        supporting_links = tuple(_source_link(article) for article in self.articles)
        links = tuple(dict.fromkeys(link.url for link in supporting_links if link.url != self.canonical_article.canonical_url))
        publisher_names = tuple(sorted({link.publisher for link in supporting_links if link.publisher}))
        source_providers = tuple(sorted({link.provider for link in supporting_links if link.provider}))
        assessments = [
            assessment
            for article in self.articles
            for assessment in assess_ticker_matches(article)
        ]
        matched_tickers = tuple(sorted({assessment.ticker for assessment in assessments}))
        related_tickers = tuple(sorted({assessment.ticker for assessment in assessments if assessment.related}))
        article_ids = tuple(_article_identifier(article) for article in self.articles)
        canonical_id = _article_identifier(self.canonical_article)
        return DedupeCluster(
            canonical_article=self.canonical_article,
            alternate_source_links=links,
            articles=tuple(self.articles),
            duplicate_reasons=tuple(self.duplicate_reasons),
            primary_link=self.canonical_article.canonical_url,
            supporting_links=supporting_links,
            publisher_count=len(publisher_names),
            source_count=len(source_providers),
            publisher_names=publisher_names,
            source_providers=source_providers,
            cluster_id=_cluster_identifier(self.canonical_article),
            primary_ticker=primary_ticker(self.canonical_article),
            matched_tickers=matched_tickers,
            related_tickers=related_tickers,
            event_type=classify_article_type(self.canonical_article).primary_type,
            primary_article_id=canonical_id,
            supporting_article_ids=tuple(article_id for article_id in article_ids if article_id != canonical_id),
            supporting_publishers=publisher_names,
            source_diversity=len(source_providers),
            publisher_diversity=len(publisher_names),
        )


def _duplicate_reason(
    article: Article,
    canonical: Article,
    *,
    title_threshold: float,
    semantic_similarity: SemanticSimilarity | None,
    semantic_threshold: float,
) -> str | None:
    if article.canonical_url == canonical.canonical_url:
        return "exact_url"
    if normalize_title(article.title) == normalize_title(canonical.title):
        return "exact_title"
    similarity = title_similarity(article.title, canonical.title)
    if (
        _ticker_sets_compatible(article, canonical)
        and _event_types_compatible(article, canonical, similarity)
        and similarity >= title_threshold
    ):
        return "similar_title"
    if (
        _ticker_sets_compatible(article, canonical)
        and _event_types_compatible(article, canonical, similarity)
        and _same_ticker_near_publish_date(article, canonical)
        and similarity >= 0.72
    ):
        return "same_ticker_near_publish_date"
    if semantic_similarity is not None:
        score = semantic_similarity(article, canonical)
        if score is not None and score >= semantic_threshold:
            return "semantic_similarity"
    return None


def _normalize_article_url(article: Article) -> Article:
    canonical_url = canonicalize_url(article.canonical_url)
    if canonical_url == article.canonical_url:
        return article
    return Article(
        canonical_url=canonical_url,
        title=article.title,
        article_id=article.article_id,
        published_at=article.published_at,
        full_text=article.full_text,
        snippet=article.snippet,
        metadata=article.metadata,
        created_at=article.created_at,
    )


def _safe_unwrap_redirect(netloc: str, query: str) -> str | None:
    if netloc not in REDIRECT_HOSTS:
        return None
    params = dict(parse_qsl(query, keep_blank_values=True))
    for param in REDIRECT_PARAMS:
        target = params.get(param)
        if not target:
            continue
        target = unquote(target)
        parsed_target = urlparse(target)
        if parsed_target.scheme in {"http", "https"} and parsed_target.netloc:
            return target
    return None


def _cleanup_syndicated_path(path: str) -> str:
    path = re.sub(r"/amp/?$", "", path)
    path = re.sub(r"\.amp$", "", path)
    path = re.sub(r"\.amp\.html$", ".html", path)
    return path or "/"


def _title_tokens(normalized_title: str) -> set[str]:
    return {
        token
        for token in normalized_title.split()
        if token and token not in TITLE_STOPWORDS
    }


def _same_ticker_near_publish_date(article: Article, canonical: Article) -> bool:
    article_tickers = {ticker.symbol for ticker in match_tickers(_article_match_text(article))}
    canonical_tickers = {ticker.symbol for ticker in match_tickers(_article_match_text(canonical))}
    if not article_tickers or not canonical_tickers or not (article_tickers & canonical_tickers):
        return False
    article_date = _parse_date(article.published_at)
    canonical_date = _parse_date(canonical.published_at)
    if article_date is None or canonical_date is None:
        return False
    return abs((article_date.date() - canonical_date.date()).days) <= 1


def _ticker_sets_compatible(article: Article, canonical: Article) -> bool:
    article_primary = primary_ticker(article)
    canonical_primary = primary_ticker(canonical)
    if article_primary and canonical_primary:
        return article_primary == canonical_primary
    article_tickers = {
        match.ticker
        for match in assess_ticker_matches(article)
        if not match.related
    }
    canonical_tickers = {
        match.ticker
        for match in assess_ticker_matches(canonical)
        if not match.related
    }
    return not article_tickers or not canonical_tickers or bool(article_tickers & canonical_tickers)


def _event_types_compatible(article: Article, canonical: Article, similarity: float) -> bool:
    article_types = set(classify_article_type(article).event_types)
    canonical_types = set(classify_article_type(canonical).event_types)
    if not article_types or not canonical_types:
        return True
    if (
        (MACRO_OR_SECTOR_ROUNDUP in article_types) != (MACRO_OR_SECTOR_ROUNDUP in canonical_types)
        and similarity < 0.96
    ):
        return False
    if (
        ANALYST_RATING_OR_PRICE_TARGET in article_types
        and EARNINGS_OR_RESULTS in canonical_types
    ) or (
        EARNINGS_OR_RESULTS in article_types
        and ANALYST_RATING_OR_PRICE_TARGET in canonical_types
    ):
        return similarity >= 0.96
    if (
        GENERIC_BUY_SELL_HOLD_OPINION in article_types
        and STOCK_PRICE_MOVE in canonical_types
    ) or (
        STOCK_PRICE_MOVE in article_types
        and GENERIC_BUY_SELL_HOLD_OPINION in canonical_types
    ):
        return similarity >= 0.96
    if article_types & canonical_types:
        return True
    if similarity >= 0.94:
        return True
    return not any(frozenset((left, right)) in CONFLICTING_EVENT_TYPES for left in article_types for right in canonical_types)


def event_types_for_title(title: str) -> frozenset[str]:
    return frozenset(classify_article_text(title).event_types)


def _contains_keyword(normalized_title: str, keyword: str) -> bool:
    keyword = normalize_title(keyword)
    if not keyword:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", normalized_title) is not None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _article_match_text(article: Article) -> str:
    return " ".join(part for part in (article.title, article.snippet or "") if part)


def _source_link(article: Article) -> SourceLink:
    metadata = article.metadata
    return SourceLink(
        title=article.title,
        url=article.canonical_url,
        publisher=metadata.get("source_name") or metadata.get("publisher"),
        provider=metadata.get("provider"),
        published_at=article.published_at,
    )


def _article_identifier(article: Article) -> str:
    if article.article_id:
        return article.article_id
    return f"art_{hashlib.sha256(article.canonical_url.encode('utf-8')).hexdigest()[:16]}"


def _cluster_identifier(article: Article) -> str:
    material = f"{canonicalize_url(article.canonical_url)}|{normalize_title(article.title)}"
    return f"cluster_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"
