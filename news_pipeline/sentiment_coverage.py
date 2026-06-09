"""Weighted article sentiment inputs and ticker-level coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .article_types import (
    ANALYST_RATING_OR_PRICE_TARGET,
    ARTICLE_TYPE_SERIOUSNESS,
    GENERIC_BUY_SELL_HOLD_OPINION,
    MACRO_OR_SECTOR_ROUNDUP,
    PREDICTION_OR_PRICE_TARGET_CLICKBAIT,
    classify_article_type,
)
from .dedup import DedupeCluster
from .models import Article
from .recency import article_recency, recency_weight
from .sentiment import analyze_sentiment
from .source_quality import assess_article_source
from .ticker_matching import HIGH, LOW, TickerMatchAssessment, assess_ticker_matches
from .tickers import TrackedTicker, load_tracked_tickers


@dataclass(frozen=True)
class WeightedArticleSentiment:
    ticker: str
    canonical_url: str
    title: str
    cluster_id: str
    sentiment_raw: float
    sentiment_basis: str
    sentiment_weight: float
    sentiment_weight_reasons: tuple[str, ...]
    ticker_match_confidence: float
    ticker_match_confidence_label: str
    ticker_match_reason: str
    article_type: str
    primary_cluster_article: bool


@dataclass(frozen=True)
class TickerSentimentCoverage:
    ticker: str
    article_count_scored: int
    full_text_scored_count: int
    snippet_scored_count: int
    title_scored_count: int
    weighted_sentiment: float
    positive_article_count: int
    negative_article_count: int
    neutral_article_count: int
    high_confidence_article_count: int
    low_confidence_article_count: int
    top_positive_cluster: str | None
    top_negative_cluster: str | None
    sentiment_coverage_grade: str


def build_weighted_sentiment_coverage(
    *,
    articles: Sequence[Article],
    clusters: Sequence[DedupeCluster],
    run_date: str,
    tracked_tickers: Sequence[TrackedTicker] | None = None,
) -> tuple[tuple[WeightedArticleSentiment, ...], Mapping[str, TickerSentimentCoverage]]:
    tracked = tuple(tracked_tickers or load_tracked_tickers())
    cluster_by_url = {
        article.canonical_url: cluster
        for cluster in clusters
        for article in cluster.articles
    }
    inputs: list[WeightedArticleSentiment] = []
    for article in articles:
        cluster = cluster_by_url.get(article.canonical_url)
        if cluster is None:
            continue
        primary = article.canonical_url == cluster.canonical_article.canonical_url
        for match in assess_ticker_matches(article, tracked):
            inputs.append(
                _weighted_input(
                    article,
                    cluster,
                    match,
                    run_date=run_date,
                    primary=primary,
                    tracked=tracked,
                )
            )

    grouped = {ticker.symbol: [] for ticker in tracked}
    for item in inputs:
        grouped.setdefault(item.ticker, []).append(item)
    coverage = {
        ticker.symbol: _coverage_row(ticker.symbol, grouped.get(ticker.symbol, ()))
        for ticker in tracked
    }
    return tuple(inputs), coverage


def _weighted_input(
    article: Article,
    cluster: DedupeCluster,
    match: TickerMatchAssessment,
    *,
    run_date: str,
    primary: bool,
    tracked: Sequence[TrackedTicker],
) -> WeightedArticleSentiment:
    basis = "full_text" if article.full_text else "snippet" if article.snippet else "title"
    text = article.full_text or article.snippet or article.title
    sentiment = analyze_sentiment(article.article_id or article.canonical_url, text, basis)
    quality = assess_article_source(article)
    classification = classify_article_type(article)
    recency = article_recency(
        run_date=run_date,
        published_at=article.published_at,
        collected_at=article.created_at,
        archive_context=bool(article.metadata.get("archive_context")),
    )
    tracked_by_symbol = {ticker.symbol: ticker for ticker in tracked}
    ticker = tracked_by_symbol[match.ticker]

    factors = {
        "source_quality": {1: 1.3, 2: 1.0, 3: 0.55, 4: 0.0}.get(quality.tier, 0.8),
        "ticker_match": {HIGH: 1.0, "medium": 0.65, LOW: 0.2}[match.confidence_label],
        "article_type": _article_type_factor(classification.primary_type),
        "recency": recency_weight(recency.recency_bucket),
        "extraction_basis": {"full_text": 1.25, "snippet": 0.75, "title": 0.4}[basis],
        "dedupe_uniqueness": 1.0 if primary else 0.35,
        "tracked_relevance": 1.1 if ticker.group == "portfolio" else 1.0,
        "source_family": _source_family_factor(article),
    }
    weight = 1.0
    reasons = []
    for name, factor in factors.items():
        weight *= factor
        reasons.append(f"{name}={factor:.2f}")
    return WeightedArticleSentiment(
        ticker=match.ticker,
        canonical_url=article.canonical_url,
        title=article.title,
        cluster_id=cluster.cluster_id,
        sentiment_raw=sentiment.score,
        sentiment_basis=basis,
        sentiment_weight=round(weight, 6),
        sentiment_weight_reasons=tuple(reasons),
        ticker_match_confidence=match.confidence,
        ticker_match_confidence_label=match.confidence_label,
        ticker_match_reason=match.reason,
        article_type=classification.primary_type,
        primary_cluster_article=primary,
    )


def _article_type_factor(article_type: str) -> float:
    if article_type == PREDICTION_OR_PRICE_TARGET_CLICKBAIT:
        return 0.2
    if article_type == GENERIC_BUY_SELL_HOLD_OPINION:
        return 0.35
    if article_type == MACRO_OR_SECTOR_ROUNDUP:
        return 0.45
    if article_type == ANALYST_RATING_OR_PRICE_TARGET:
        return 0.65
    seriousness = ARTICLE_TYPE_SERIOUSNESS.get(article_type, 0)
    return min(1.25, 0.85 + max(0, seriousness) / 35.0)


def _source_family_factor(article: Article) -> float:
    family = str(article.metadata.get("source_family") or "")
    if family == "press_release_wire" or article.metadata.get("issuer_promotional"):
        return 0.65
    if family == "google_news_backstop":
        return 0.75
    if family in {"regulatory_official", "company_ir"}:
        return 1.15
    return 1.0


def _coverage_row(
    ticker: str,
    items: Sequence[WeightedArticleSentiment],
) -> TickerSentimentCoverage:
    total_weight = sum(item.sentiment_weight for item in items)
    weighted = (
        sum(item.sentiment_raw * item.sentiment_weight for item in items) / total_weight
        if total_weight
        else 0.0
    )
    positive = [item for item in items if item.sentiment_raw > 0.05]
    negative = [item for item in items if item.sentiment_raw < -0.05]
    high_signal_positive = [
        item for item in positive if item.ticker_match_confidence_label != LOW
    ]
    high_signal_negative = [
        item for item in negative if item.ticker_match_confidence_label != LOW
    ]
    high_count = sum(1 for item in items if item.ticker_match_confidence_label == HIGH)
    low_count = sum(1 for item in items if item.ticker_match_confidence_label == LOW)
    full_text_count = sum(1 for item in items if item.sentiment_basis == "full_text")
    snippet_count = sum(1 for item in items if item.sentiment_basis == "snippet")
    title_count = sum(1 for item in items if item.sentiment_basis == "title")
    return TickerSentimentCoverage(
        ticker=ticker,
        article_count_scored=len(items),
        full_text_scored_count=full_text_count,
        snippet_scored_count=snippet_count,
        title_scored_count=title_count,
        weighted_sentiment=round(weighted, 4),
        positive_article_count=len(positive),
        negative_article_count=len(negative),
        neutral_article_count=len(items) - len(positive) - len(negative),
        high_confidence_article_count=high_count,
        low_confidence_article_count=low_count,
        top_positive_cluster=(
            max(
                high_signal_positive,
                key=lambda item: item.sentiment_raw * item.sentiment_weight,
            ).title
            if high_signal_positive
            else None
        ),
        top_negative_cluster=(
            min(
                high_signal_negative,
                key=lambda item: item.sentiment_raw * item.sentiment_weight,
            ).title
            if high_signal_negative
            else None
        ),
        sentiment_coverage_grade=_coverage_grade(
            article_count=len(items),
            high_count=high_count,
            full_text_count=full_text_count,
            snippet_count=snippet_count,
        ),
    )


def _coverage_grade(
    *,
    article_count: int,
    high_count: int,
    full_text_count: int,
    snippet_count: int,
) -> str:
    if full_text_count >= 3 and high_count >= 3:
        return "strong"
    if article_count >= 4 and high_count >= 2 and (full_text_count >= 1 or snippet_count >= 4):
        return "moderate"
    return "weak"
