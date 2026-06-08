"""Deterministic summaries and reading priorities for report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

from .article_fetch import URL_CLASS_DIRECT_PUBLISHER, classify_article_url
from .dedup import DedupeCluster, event_types_for_title
from .models import Article
from .recency import article_recency
from .sentiment import analyze_sentiment
from .source_quality import TIER_4_EXCLUDE_BY_DEFAULT, assess_article_source
from .tickers import TrackedTicker, load_tracked_tickers, match_tickers


READ_FIRST = "read_first"
READ_NEXT = "read_next"
BACKGROUND_ONLY = "background_only"

SUMMARY_TERMS = (
    "stock",
    "shares",
    "earnings",
    "guidance",
    "analyst",
    "upgrade",
    "downgrade",
    "ai",
    "demand",
    "revenue",
    "chip",
    "data center",
    "regulatory",
    "acquisition",
)

EVENT_SERIOUSNESS = {
    "regulatory_legal": 14,
    "earnings_results": 13,
    "acquisition_deal": 12,
    "partnership_contract": 9,
    "product_ai_announcement": 8,
    "stock_move": 7,
    "analyst_rating_price_target": 3,
    "general_buy_sell_opinion": -2,
}

EVENT_MATTERS = {
    "regulatory_legal": "It may create a material legal or regulatory catalyst.",
    "earnings_results": "It may reset near-term expectations for revenue, earnings, or guidance.",
    "acquisition_deal": "It may change ownership, strategy, or valuation expectations.",
    "partnership_contract": "It may affect commercial demand or future revenue visibility.",
    "product_ai_announcement": "It may affect product demand and competitive positioning.",
    "stock_move": "It helps explain the current move in investor attention or shares.",
    "analyst_rating_price_target": "It reflects an analyst view, not a company-reported operating event.",
    "general_buy_sell_opinion": "It is background opinion rather than a new operating catalyst.",
}


@dataclass(frozen=True)
class ArticleMicroSummary:
    canonical_url: str
    title: str
    tickers: tuple[str, ...]
    article_summary: str
    summary_basis: str
    summary_confidence: float
    summary_warning: str | None
    source_quality_tier: int
    source_quality_label: str
    recency_bucket: str
    direct_publisher_url: bool
    sentiment_score: float
    event_types: tuple[str, ...]
    ranking_score: float


@dataclass(frozen=True)
class RankedArticleRecommendation:
    ticker: str
    title: str
    url: str
    source: str
    article_summary: str
    summary_basis: str
    summary_confidence: float
    summary_warning: str | None
    ranking_score: float
    reading_priority: str
    source_quality_label: str
    recency_bucket: str
    direct_publisher_url: bool
    sentiment_score: float


@dataclass(frozen=True)
class ClusterIntelligence:
    ticker: str
    title: str
    primary_link: str
    cluster_summary: str
    cluster_summary_basis: str
    cluster_reading_priority: str
    ranking_score: float


@dataclass(frozen=True)
class TickerDailySummary:
    ticker: str
    company_name: str
    ticker_daily_summary: str
    top_positive_story: str | None
    top_negative_story: str | None
    read_first_story: str | None
    read_next_story: str | None
    background_story: str | None


@dataclass(frozen=True)
class MarketIntelligence:
    article_summaries: tuple[ArticleMicroSummary, ...]
    ranked_reads_by_ticker: Mapping[str, tuple[RankedArticleRecommendation, ...]]
    cluster_intelligence: Mapping[tuple[str, str], ClusterIntelligence]
    ticker_summaries: Mapping[str, TickerDailySummary]


def summarize_article(article: Article, *, ticker: TrackedTicker | None = None) -> ArticleMicroSummary:
    """Create a short extractive summary with an explicit source basis."""
    text, basis, confidence, warning = _summary_source(article)
    tickers = match_tickers(" ".join(part for part in (article.title, article.snippet or "", text) if part))
    preferred_ticker = ticker or (tickers[0] if tickers else None)
    summary = _best_sentence(text, preferred_ticker)
    quality = assess_article_source(article)
    sentiment = analyze_sentiment(
        article.article_id or article.canonical_url,
        text,
        basis,
    )
    resolved_url = str(article.metadata.get("extraction_final_url") or "")
    direct = (
        classify_article_url(article.canonical_url) == URL_CLASS_DIRECT_PUBLISHER
        or (
            bool(resolved_url)
            and classify_article_url(resolved_url) == URL_CLASS_DIRECT_PUBLISHER
        )
    )
    event_types = tuple(sorted(event_types_for_title(article.title)))
    return ArticleMicroSummary(
        canonical_url=article.canonical_url,
        title=article.title,
        tickers=tuple(sorted(item.symbol for item in tickers)),
        article_summary=summary,
        summary_basis=basis,
        summary_confidence=confidence,
        summary_warning=warning,
        source_quality_tier=quality.tier,
        source_quality_label=quality.label,
        recency_bucket="unknown",
        direct_publisher_url=direct,
        sentiment_score=sentiment.score,
        event_types=event_types,
        ranking_score=0.0,
    )


def build_market_intelligence(
    *,
    articles: Sequence[Article],
    clusters: Sequence[DedupeCluster],
    run_date: str,
    tracked_tickers: Sequence[TrackedTicker] | None = None,
) -> MarketIntelligence:
    """Build deterministic report intelligence without persisting article text."""
    tracked = tuple(tracked_tickers or load_tracked_tickers())
    article_by_url = {article.canonical_url: article for article in articles}
    cluster_context = _cluster_context(clusters)
    summaries: list[ArticleMicroSummary] = []

    for article in articles:
        base = summarize_article(article)
        recency = article_recency(
            run_date=run_date,
            published_at=article.published_at,
            collected_at=article.created_at,
            archive_context=bool(article.metadata.get("archive_context")),
        )
        publisher_count, source_count = cluster_context.get(article.canonical_url, (1, 1))
        score = _ranking_score(
            base,
            publisher_count=publisher_count,
            source_count=source_count,
        )
        summaries.append(
            ArticleMicroSummary(
                **{
                    **base.__dict__,
                    "recency_bucket": recency.recency_bucket,
                    "ranking_score": score,
                }
            )
        )

    ranked_reads = _ranked_reads_by_ticker(summaries, articles, tracked)
    cluster_intelligence = _build_cluster_intelligence(
        clusters,
        article_by_url,
        summaries,
        run_date=run_date,
    )
    ticker_summaries = {
        ticker.symbol: _ticker_summary(
            ticker,
            ranked_reads.get(ticker.symbol, ()),
            cluster_intelligence,
        )
        for ticker in tracked
    }
    return MarketIntelligence(
        article_summaries=tuple(sorted(summaries, key=lambda item: (-item.ranking_score, item.title, item.canonical_url))),
        ranked_reads_by_ticker=ranked_reads,
        cluster_intelligence=cluster_intelligence,
        ticker_summaries=ticker_summaries,
    )


def _summary_source(article: Article) -> tuple[str, str, float, str | None]:
    if article.full_text and article.full_text.strip():
        return article.full_text.strip(), "full_text", 0.9, None
    if article.snippet and article.snippet.strip():
        return (
            article.snippet.strip(),
            "snippet",
            0.65,
            "Full article text was unavailable; summary uses the publisher or feed snippet.",
        )
    return (
        article.title.strip(),
        "title",
        0.4,
        "Full article text and snippet were unavailable; summary uses the title only.",
    )


def _best_sentence(text: str, ticker: TrackedTicker | None) -> str:
    sentences = _clean_sentences(text)
    if not sentences:
        return "No summary text was available."
    preferred_terms = list(SUMMARY_TERMS)
    if ticker is not None:
        preferred_terms.extend(ticker.match_terms)

    def sentence_score(item: tuple[int, str]) -> tuple[int, int, int]:
        index, sentence = item
        lowered = sentence.casefold()
        matches = sum(1 for term in preferred_terms if _contains_term(lowered, term.casefold()))
        return matches, min(len(sentence), 240), -index

    selected = max(enumerate(sentences), key=sentence_score)[1]
    return _shorten_sentence(selected)


def _clean_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    if not collapsed:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|(?:\s+[|•]\s+)", collapsed)
    return [
        sentence
        for piece in pieces
        if (sentence := re.sub(r"\s+", " ", piece).strip(" -\t")) and len(sentence) >= 20
    ] or [collapsed]


def _shorten_sentence(sentence: str, *, limit: int = 260) -> str:
    sentence = sentence.strip()
    if len(sentence) > limit:
        shortened = sentence[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
        sentence = f"{shortened}."
    elif sentence[-1:] not in ".!?":
        sentence = f"{sentence}."
    return sentence


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _cluster_context(clusters: Sequence[DedupeCluster]) -> dict[str, tuple[int, int]]:
    context: dict[str, tuple[int, int]] = {}
    for cluster in clusters:
        diversity = (max(1, cluster.publisher_count), max(1, cluster.source_count))
        for article in cluster.articles:
            current = context.get(article.canonical_url, (1, 1))
            context[article.canonical_url] = (
                max(current[0], diversity[0]),
                max(current[1], diversity[1]),
            )
    return context


def _ranking_score(
    summary: ArticleMicroSummary,
    *,
    publisher_count: int,
    source_count: int,
) -> float:
    quality_points = {1: 30, 2: 22, 3: 8, 4: -100}.get(summary.source_quality_tier, 12)
    recency_points = {
        "today_signal": 20,
        "recent_pulse": 15,
        "weekly_trend": 9,
        "background_context": 3,
    }.get(summary.recency_bucket, 0)
    basis_points = {"full_text": 18, "snippet": 10, "title": 4}[summary.summary_basis]
    ticker_points = 12 if len(summary.tickers) == 1 else 7 if summary.tickers else 0
    direct_points = 10 if summary.direct_publisher_url else 0
    diversity_points = min(8, max(0, publisher_count - 1) * 3 + max(0, source_count - 1) * 2)
    event_points = max((EVENT_SERIOUSNESS.get(event, 0) for event in summary.event_types), default=0)
    sentiment_points = min(8.0, abs(summary.sentiment_score) * 8.0)
    analyst_penalty = (
        -12
        if "analyst_rating_price_target" in summary.event_types
        and publisher_count <= 1
        and source_count <= 1
        else 0
    )
    return round(
        quality_points
        + recency_points
        + basis_points
        + ticker_points
        + direct_points
        + diversity_points
        + event_points
        + sentiment_points
        + analyst_penalty,
        2,
    )


def _ranked_reads_by_ticker(
    summaries: Sequence[ArticleMicroSummary],
    articles: Sequence[Article],
    tracked: Sequence[TrackedTicker],
) -> dict[str, tuple[RankedArticleRecommendation, ...]]:
    article_by_url = {article.canonical_url: article for article in articles}
    grouped: dict[str, list[ArticleMicroSummary]] = {ticker.symbol: [] for ticker in tracked}
    for summary in summaries:
        for ticker in summary.tickers:
            if ticker in grouped:
                grouped[ticker].append(summary)

    result: dict[str, tuple[RankedArticleRecommendation, ...]] = {}
    for ticker in tracked:
        candidates = sorted(
            grouped[ticker.symbol],
            key=lambda item: (-item.ranking_score, item.title, item.canonical_url),
        )
        read_first_url = next(
            (
                item.canonical_url
                for item in candidates
                if item.source_quality_tier <= 2
                and item.source_quality_tier != TIER_4_EXCLUDE_BY_DEFAULT
            ),
            None,
        )
        read_next_url = next(
            (item.canonical_url for item in candidates if item.canonical_url != read_first_url),
            None,
        )
        rows = []
        for item in candidates:
            if item.canonical_url == read_first_url:
                priority = READ_FIRST
            elif item.canonical_url == read_next_url:
                priority = READ_NEXT
            else:
                priority = BACKGROUND_ONLY
            article = article_by_url[item.canonical_url]
            quality = assess_article_source(article)
            resolved_url = str(article.metadata.get("extraction_final_url") or "")
            recommended_url = (
                resolved_url
                if resolved_url
                and classify_article_url(resolved_url) == URL_CLASS_DIRECT_PUBLISHER
                else item.canonical_url
            )
            rows.append(
                RankedArticleRecommendation(
                    ticker=ticker.symbol,
                    title=item.title,
                    url=recommended_url,
                    source=quality.publisher or quality.domain or "unknown source",
                    article_summary=item.article_summary,
                    summary_basis=item.summary_basis,
                    summary_confidence=item.summary_confidence,
                    summary_warning=item.summary_warning,
                    ranking_score=item.ranking_score,
                    reading_priority=priority,
                    source_quality_label=item.source_quality_label,
                    recency_bucket=item.recency_bucket,
                    direct_publisher_url=item.direct_publisher_url,
                    sentiment_score=item.sentiment_score,
                )
            )
        result[ticker.symbol] = tuple(rows)
    return result


def _build_cluster_intelligence(
    clusters: Sequence[DedupeCluster],
    article_by_url: Mapping[str, Article],
    summaries: Sequence[ArticleMicroSummary],
    *,
    run_date: str,
) -> dict[tuple[str, str], ClusterIntelligence]:
    summary_by_url = {item.canonical_url: item for item in summaries}
    grouped: dict[str, list[tuple[DedupeCluster, ArticleMicroSummary, float, str]]] = {}
    for cluster in clusters:
        available = [
            summary_by_url[article.canonical_url]
            for article in cluster.articles
            if article.canonical_url in summary_by_url
        ]
        if not available:
            canonical = article_by_url.get(cluster.canonical_article.canonical_url, cluster.canonical_article)
            fallback = summarize_article(canonical)
            available = [fallback]
        best = max(available, key=lambda item: (item.ranking_score, item.summary_confidence, item.title))
        tickers = {
            ticker.symbol
            for article in cluster.articles
            for ticker in match_tickers(
                " ".join(
                    part
                    for part in (
                        article.title,
                        article.snippet or "",
                        article_by_url.get(article.canonical_url, article).full_text or "",
                    )
                    if part
                )
            )
        }
        event_types = set(event_types_for_title(cluster.canonical_article.title))
        score = round(
            best.ranking_score
            + min(10, max(0, cluster.publisher_count - 1) * 4 + max(0, cluster.source_count - 1) * 2),
            2,
        )
        why = _why_it_matters(event_types)
        for ticker in sorted(tickers):
            summary = _cluster_summary(ticker, best.article_summary, why)
            grouped.setdefault(ticker, []).append((cluster, best, score, summary))

    result: dict[tuple[str, str], ClusterIntelligence] = {}
    for ticker, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (-item[2], item[0].canonical_article.title))
        for index, (cluster, best, score, summary) in enumerate(ordered):
            priority = READ_FIRST if index == 0 and best.source_quality_tier <= 2 else READ_NEXT if index <= 1 else BACKGROUND_ONLY
            result[(ticker, cluster.canonical_article.title)] = ClusterIntelligence(
                ticker=ticker,
                title=cluster.canonical_article.title,
                primary_link=cluster.primary_link,
                cluster_summary=summary,
                cluster_summary_basis=best.summary_basis,
                cluster_reading_priority=priority,
                ranking_score=score,
            )
    return result


def _why_it_matters(event_types: Iterable[str]) -> str:
    serious = sorted(
        event_types,
        key=lambda event: (-EVENT_SERIOUSNESS.get(event, 0), event),
    )
    return EVENT_MATTERS.get(serious[0], "It is relevant to current coverage and investor attention.") if serious else (
        "It is relevant to current coverage and investor attention."
    )


def _cluster_summary(ticker: str, article_summary: str, why: str) -> str:
    first = article_summary.strip()
    if ticker.casefold() not in first.casefold():
        first = f"{ticker}: {first}"
    return f"{first} {why}"


def _ticker_summary(
    ticker: TrackedTicker,
    reads: Sequence[RankedArticleRecommendation],
    cluster_intelligence: Mapping[tuple[str, str], ClusterIntelligence],
) -> TickerDailySummary:
    ticker_clusters = sorted(
        (
            item
            for (symbol, _title), item in cluster_intelligence.items()
            if symbol == ticker.symbol
        ),
        key=lambda item: (-item.ranking_score, item.title),
    )
    read_first = next((item for item in reads if item.reading_priority == READ_FIRST), None)
    read_next = next((item for item in reads if item.reading_priority == READ_NEXT), None)
    background = next((item for item in reads if item.reading_priority == BACKGROUND_ONLY), None)
    positive = max((item for item in reads if item.sentiment_score > 0), key=lambda item: item.sentiment_score, default=None)
    negative = min((item for item in reads if item.sentiment_score < 0), key=lambda item: item.sentiment_score, default=None)
    if ticker_clusters:
        focus = ticker_clusters[0].cluster_summary
        summary = focus if focus.casefold().startswith(f"{ticker.symbol.casefold()}:") else f"{ticker.symbol}: {focus}"
    else:
        summary = f"{ticker.symbol}: No matched story was available in the current report window."
    if read_first is not None:
        summary += f" Read first: {read_first.title} because it has the strongest deterministic source, recency, extraction, and event score."
    return TickerDailySummary(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        ticker_daily_summary=summary,
        top_positive_story=positive.title if positive else None,
        top_negative_story=negative.title if negative else None,
        read_first_story=read_first.title if read_first else None,
        read_next_story=read_next.title if read_next else None,
        background_story=background.title if background else None,
    )
