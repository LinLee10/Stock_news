"""URL canonicalization and deterministic dedupe keys."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import re
from typing import Callable, Iterable
from urllib.parse import parse_qsl, urlencode, unquote, urlparse, urlunparse

from .models import Article
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
    if _ticker_sets_compatible(article, canonical) and title_similarity(article.title, canonical.title) >= title_threshold:
        return "similar_title"
    if (
        _ticker_sets_compatible(article, canonical)
        and _same_ticker_near_publish_date(article, canonical)
        and title_similarity(article.title, canonical.title) >= 0.55
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
    article_tickers = {ticker.symbol for ticker in match_tickers(_article_match_text(article))}
    canonical_tickers = {ticker.symbol for ticker in match_tickers(_article_match_text(canonical))}
    return not article_tickers or not canonical_tickers or bool(article_tickers & canonical_tickers)


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
