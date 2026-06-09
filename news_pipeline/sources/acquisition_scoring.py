"""Deterministic source acquisition and diversity scoring."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

from news_pipeline.article_fetch import URL_CLASS_DIRECT_PUBLISHER, classify_article_url
from news_pipeline.article_types import ARTICLE_TYPE_SERIOUSNESS, classify_article_type
from news_pipeline.models import Article
from news_pipeline.ticker_matching import assess_ticker_matches

from .source_registry import (
    COMPANY_IR,
    DIRECT_NEWS_PUBLISHER,
    GOOGLE_NEWS_BACKSTOP,
    MARKET_DATA_OR_ANALYSIS,
    PAID_NEWS_API,
    PRESS_RELEASE_WIRE,
    REGULATORY_OFFICIAL,
    SourceProfile,
)


SOURCE_FAMILY_WEIGHTS = {
    REGULATORY_OFFICIAL: 1.0,
    COMPANY_IR: 0.95,
    DIRECT_NEWS_PUBLISHER: 0.9,
    PRESS_RELEASE_WIRE: 0.72,
    MARKET_DATA_OR_ANALYSIS: 0.68,
    PAID_NEWS_API: 0.7,
    GOOGLE_NEWS_BACKSTOP: 0.35,
}


def annotate_acquisition_scores(
    articles: Sequence[Article],
    profiles: Mapping[str, SourceProfile],
) -> list[Article]:
    publisher_counts = Counter(_publisher(article) for article in articles)
    seen_urls: set[str] = set()
    scored: list[Article] = []
    for article in articles:
        source_id = str(article.metadata.get("source_id") or article.metadata.get("provider") or "")
        profile = profiles.get(source_id)
        family = str(article.metadata.get("source_family") or (profile.source_family if profile else "unknown"))
        quality_tier = int(
            article.metadata.get("source_quality_tier")
            or (profile.source_quality_tier if profile else 2)
        )
        matches = assess_ticker_matches(article)
        ticker_specificity = max((match.confidence for match in matches), default=0.0) * 100
        source_priority = float(profile.source_priority if profile else 50)
        publisher_quality = {1: 100.0, 2: 75.0, 3: 40.0, 4: 0.0}.get(quality_tier, 60.0)
        family_weight = SOURCE_FAMILY_WEIGHTS.get(family, 0.5)
        diversity = 100.0 / max(1, publisher_counts[_publisher(article)])
        novelty = 100.0 if article.canonical_url not in seen_urls else 0.0
        seen_urls.add(article.canonical_url)
        extraction_likelihood = (
            95.0
            if classify_article_url(article.canonical_url) == URL_CLASS_DIRECT_PUBLISHER
            else 20.0
        )
        seriousness = ARTICLE_TYPE_SERIOUSNESS.get(
            classify_article_type(article).primary_type,
            0,
        )
        acquisition_score = (
            source_priority * 0.22
            + publisher_quality * 0.18
            + ticker_specificity * 0.2
            + diversity * 0.08
            + novelty * 0.08
            + extraction_likelihood * 0.14
            + max(0.0, min(100.0, 50.0 + seriousness * 3.0)) * 0.1
        )
        scored.append(
            Article(
                canonical_url=article.canonical_url,
                title=article.title,
                article_id=article.article_id,
                published_at=article.published_at,
                full_text=article.full_text,
                snippet=article.snippet,
                metadata={
                    **article.metadata,
                    "source_id": source_id or "unknown",
                    "source_family": family,
                    "source_priority_score": round(source_priority, 2),
                    "source_diversity_score": round(diversity, 2),
                    "source_family_weight": round(family_weight, 3),
                    "publisher_quality_score": round(publisher_quality, 2),
                    "ticker_specificity_score": round(ticker_specificity, 2),
                    "novelty_score": round(novelty, 2),
                    "extraction_likelihood_score": round(extraction_likelihood, 2),
                    "acquisition_score": round(acquisition_score, 3),
                },
                created_at=article.created_at,
            )
        )
    return scored


def source_diversity_metrics(articles: Iterable[Article]) -> dict[str, float]:
    rows = tuple(articles)
    if not rows:
        return {"source_diversity_score": 0.0, "source_balance_score": 0.0}
    publishers = Counter(_publisher(article) for article in rows)
    families = Counter(str(article.metadata.get("source_family") or "unknown") for article in rows)
    diversity = _normalized_entropy(publishers)
    balance = _normalized_entropy(families)
    return {
        "source_diversity_score": round(diversity * 100, 2),
        "source_balance_score": round(balance * 100, 2),
    }


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * log(count / total)
        for count in counts.values()
        if count
    )
    return entropy / log(len(counts))


def _publisher(article: Article) -> str:
    publisher = str(
        article.metadata.get("source_name")
        or article.metadata.get("publisher_name")
        or article.metadata.get("provider")
        or ""
    ).strip()
    return publisher or urlparse(article.canonical_url).netloc.lower() or "unknown"
