"""Deterministic summaries and reading priorities for report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

from .article_fetch import URL_CLASS_DIRECT_PUBLISHER, classify_article_url
from .article_types import (
    ANALYST_RATING_OR_PRICE_TARGET,
    ARTICLE_TYPE_SERIOUSNESS,
    CUSTOMER_OR_DEMAND_SIGNAL,
    EARNINGS_OR_RESULTS,
    GENERIC_BUY_SELL_HOLD_OPINION,
    GUIDANCE_OR_FORECAST,
    MACRO_OR_SECTOR_ROUNDUP,
    PARTNERSHIP_OR_CONTRACT,
    PREDICTION_OR_PRICE_TARGET_CLICKBAIT,
    PRODUCT_OR_AI_OR_CHIP_NEWS,
    REGULATORY_OR_LEGAL,
    STOCK_PRICE_MOVE,
    classify_article_type,
)
from .dedup import DedupeCluster
from .models import Article
from .recency import article_recency
from .sentiment import analyze_sentiment
from .source_quality import TIER_4_EXCLUDE_BY_DEFAULT, assess_article_source
from .ticker_matching import assess_ticker_matches
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

EVENT_SERIOUSNESS = ARTICLE_TYPE_SERIOUSNESS

EVENT_MATTERS = {
    REGULATORY_OR_LEGAL: "It may create a material legal or regulatory catalyst.",
    EARNINGS_OR_RESULTS: "It may reset near-term expectations for revenue or earnings.",
    GUIDANCE_OR_FORECAST: "It may reset near-term guidance or forecast expectations.",
    PARTNERSHIP_OR_CONTRACT: "It may affect commercial demand or future revenue visibility.",
    CUSTOMER_OR_DEMAND_SIGNAL: "It may provide a direct signal about customer demand.",
    PRODUCT_OR_AI_OR_CHIP_NEWS: "It may affect product demand and competitive positioning.",
    STOCK_PRICE_MOVE: "It helps explain the current move in investor attention or shares.",
    ANALYST_RATING_OR_PRICE_TARGET: "It reflects an analyst view, not a company-reported operating event.",
    GENERIC_BUY_SELL_HOLD_OPINION: "It is background opinion rather than a new operating catalyst.",
    MACRO_OR_SECTOR_ROUNDUP: "It provides broad market context rather than a ticker-specific catalyst.",
    PREDICTION_OR_PRICE_TARGET_CLICKBAIT: "It is speculative commentary rather than reported operating news.",
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
    article_type: str
    primary_ticker: str | None
    ticker_match_confidence: Mapping[str, float]
    ticker_match_reasons: Mapping[str, str]
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
    ticker_match_basis: str
    ticker_specific: bool
    read_first_reason: str | None


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


@dataclass(frozen=True)
class _ClusterContext:
    publisher_count: int
    source_count: int
    primary_ticker: str | None
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class _TickerMatch:
    basis: str
    strong_match: bool
    ticker_specific: bool
    multi_ticker: bool
    primary_ticker: str | None


def summarize_article(article: Article, *, ticker: TrackedTicker | None = None) -> ArticleMicroSummary:
    """Create a short extractive summary with an explicit source basis."""
    text, basis, confidence, warning = _summary_source(article)
    tickers = match_tickers(" ".join(part for part in (article.title, article.snippet or "", text) if part))
    preferred_ticker = ticker or (tickers[0] if tickers else None)
    summary = _best_sentence(text, preferred_ticker)
    quality = assess_article_source(article)
    classification = classify_article_type(article)
    ticker_matches = assess_ticker_matches(article)
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
        event_types=classification.event_types,
        article_type=classification.primary_type,
        primary_ticker=next((match.ticker for match in ticker_matches if match.primary), None),
        ticker_match_confidence={match.ticker: match.confidence for match in ticker_matches},
        ticker_match_reasons={match.ticker: match.reason for match in ticker_matches},
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
    cluster_context = _cluster_context(clusters, article_by_url, tracked)
    summaries: list[ArticleMicroSummary] = []

    for article in articles:
        base = summarize_article(article)
        recency = article_recency(
            run_date=run_date,
            published_at=article.published_at,
            collected_at=article.created_at,
            archive_context=bool(article.metadata.get("archive_context")),
        )
        context = cluster_context.get(article.canonical_url, _ClusterContext(1, 1, None, ()))
        score = _ranking_score(
            base,
            publisher_count=context.publisher_count,
            source_count=context.source_count,
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

    ranked_reads = _ranked_reads_by_ticker(summaries, articles, tracked, cluster_context)
    cluster_intelligence = _build_cluster_intelligence(
        clusters,
        article_by_url,
        summaries,
        tracked,
        cluster_context,
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


def _cluster_context(
    clusters: Sequence[DedupeCluster],
    article_by_url: Mapping[str, Article],
    tracked: Sequence[TrackedTicker],
) -> dict[str, _ClusterContext]:
    context: dict[str, _ClusterContext] = {}
    for cluster in clusters:
        canonical = article_by_url.get(cluster.canonical_article.canonical_url, cluster.canonical_article)
        primary_ticker = _primary_ticker(canonical, tracked)
        cluster_tickers = tuple(
            sorted(
                {
                    ticker.symbol
                    for article in cluster.articles
                    for ticker in match_tickers(_article_match_text(article_by_url.get(article.canonical_url, article)))
                }
            )
        )
        for article in cluster.articles:
            current = context.get(article.canonical_url, _ClusterContext(1, 1, None, ()))
            context[article.canonical_url] = _ClusterContext(
                publisher_count=max(current.publisher_count, max(1, cluster.publisher_count)),
                source_count=max(current.source_count, max(1, cluster.source_count)),
                primary_ticker=primary_ticker or current.primary_ticker,
                tickers=tuple(sorted(set(current.tickers) | set(cluster_tickers))),
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
        if ANALYST_RATING_OR_PRICE_TARGET in summary.event_types
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
    cluster_context: Mapping[str, _ClusterContext],
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
            (
                (
                    item,
                    _ticker_match(
                        article_by_url[item.canonical_url],
                        ticker,
                        cluster_context.get(item.canonical_url, _ClusterContext(1, 1, None, ())),
                    ),
                )
                for item in grouped[ticker.symbol]
            ),
            key=lambda candidate: (
                -_ticker_ranking_score(candidate[0], candidate[1]),
                candidate[0].title,
                candidate[0].canonical_url,
            ),
        )
        ticker_specific_candidates = [
            candidate
            for candidate in candidates
            if candidate[1].ticker_specific and not _read_first_disallowed(candidate[0])
        ]
        read_first_url = next(
            (
                item.canonical_url
                for item, match in ticker_specific_candidates
                if item.source_quality_tier <= 2
                and item.source_quality_tier != TIER_4_EXCLUDE_BY_DEFAULT
            ),
            None,
        )
        if read_first_url is None and not ticker_specific_candidates:
            read_first_url = next(
                (
                    item.canonical_url
                    for item, match in candidates
                    if match.strong_match
                    and match.basis not in {"snippet_related", "full_text_related"}
                    and item.source_quality_tier <= 2
                    and item.source_quality_tier != TIER_4_EXCLUDE_BY_DEFAULT
                    and not _read_first_disallowed(item)
                ),
                None,
            )
        read_next_url = next(
            (
                item.canonical_url
                for item, match in candidates
                if item.canonical_url != read_first_url
                and match.ticker_specific
                and not _read_first_disallowed(item)
            ),
            None,
        )
        rows = []
        for item, match in candidates:
            if item.canonical_url == read_first_url:
                priority = READ_FIRST
            elif item.canonical_url == read_next_url:
                priority = READ_NEXT
            else:
                priority = BACKGROUND_ONLY
            article = article_by_url[item.canonical_url]
            ticker_summary = summarize_article(article, ticker=ticker)
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
                    article_summary=ticker_summary.article_summary,
                    summary_basis=ticker_summary.summary_basis,
                    summary_confidence=ticker_summary.summary_confidence,
                    summary_warning=ticker_summary.summary_warning,
                    ranking_score=_ticker_ranking_score(item, match),
                    reading_priority=priority,
                    source_quality_label=item.source_quality_label,
                    recency_bucket=item.recency_bucket,
                    direct_publisher_url=item.direct_publisher_url,
                    sentiment_score=item.sentiment_score,
                    ticker_match_basis=match.basis,
                    ticker_specific=match.ticker_specific,
                    read_first_reason=_read_first_reason(item, match) if priority == READ_FIRST else None,
                )
            )
        result[ticker.symbol] = tuple(rows)
    return result


def _build_cluster_intelligence(
    clusters: Sequence[DedupeCluster],
    article_by_url: Mapping[str, Article],
    summaries: Sequence[ArticleMicroSummary],
    tracked: Sequence[TrackedTicker],
    cluster_context: Mapping[str, _ClusterContext],
    *,
    run_date: str,
) -> dict[tuple[str, str], ClusterIntelligence]:
    summary_by_url = {item.canonical_url: item for item in summaries}
    ticker_lookup = {ticker.symbol: ticker for ticker in tracked}
    grouped: dict[str, list[tuple[DedupeCluster, ArticleMicroSummary, _TickerMatch, float, str]]] = {}
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
        event_types = set(classify_article_type(cluster.canonical_article).event_types)
        why = _why_it_matters(event_types)
        for symbol in sorted(tickers):
            ticker = ticker_lookup.get(symbol)
            if ticker is None:
                continue
            matched = [
                (
                    item,
                    _ticker_match(
                        article_by_url.get(item.canonical_url, cluster.canonical_article),
                        ticker,
                        cluster_context.get(item.canonical_url, _ClusterContext(1, 1, None, ())),
                    ),
                )
                for item in available
            ]
            eligible = [
                candidate
                for candidate in matched
                if candidate[1].ticker_specific or candidate[1].basis == "title_multi_ticker"
            ]
            if not eligible:
                continue
            best, match = max(
                eligible,
                key=lambda candidate: (
                    _ticker_ranking_score(candidate[0], candidate[1]),
                    candidate[0].summary_confidence,
                    candidate[0].title,
                ),
            )
            article = article_by_url.get(best.canonical_url, cluster.canonical_article)
            ticker_summary = summarize_article(article, ticker=ticker)
            score = round(
                _ticker_ranking_score(best, match)
                + min(10, max(0, cluster.publisher_count - 1) * 4 + max(0, cluster.source_count - 1) * 2),
                2,
            )
            summary = _cluster_summary(symbol, ticker_summary.article_summary, why)
            grouped.setdefault(symbol, []).append((cluster, best, match, score, summary))

    result: dict[tuple[str, str], ClusterIntelligence] = {}
    for ticker, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (-item[3], item[0].canonical_article.title))
        has_specific = any(match.ticker_specific for _cluster, _best, match, _score, _summary in ordered)
        promoted = 0
        for cluster, best, match, score, summary in ordered:
            promotable = (
                best.source_quality_tier <= 2
                and not _read_first_disallowed(best)
                and (match.ticker_specific or not has_specific)
            )
            if promotable and promoted == 0:
                priority = READ_FIRST
                promoted += 1
            elif promotable and promoted == 1:
                priority = READ_NEXT
                promoted += 1
            else:
                priority = BACKGROUND_ONLY
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


def _article_match_text(article: Article) -> str:
    return " ".join(part for part in (article.title, article.snippet or "", article.full_text or "") if part)


def _primary_ticker(article: Article, tracked: Sequence[TrackedTicker]) -> str | None:
    matches = [ticker for ticker in tracked if _matches_ticker(article.title, ticker)]
    if not matches:
        return None
    return min(matches, key=lambda ticker: _ticker_position(article.title, ticker)).symbol


def _ticker_position(text: str, ticker: TrackedTicker) -> int:
    positions = [
        match.start()
        for term in ticker.match_terms
        for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.IGNORECASE)
    ]
    return min(positions, default=len(text))


def _matches_ticker(text: str, ticker: TrackedTicker) -> bool:
    return any(match.symbol == ticker.symbol for match in match_tickers(text))


def _ticker_match(article: Article, ticker: TrackedTicker, context: _ClusterContext) -> _TickerMatch:
    title_match = _matches_ticker(article.title, ticker)
    snippet_match = _matches_ticker(article.snippet or "", ticker)
    full_text_match = _matches_ticker(article.full_text or "", ticker)
    title_tickers = tuple(match.symbol for match in match_tickers(article.title))
    multi_ticker = len(set(context.tickers) | set(title_tickers)) > 1
    primary_match = context.primary_ticker == ticker.symbol
    other_subject = _headline_centers_other_company(article.title, ticker)
    if title_match and len(title_tickers) > 1:
        basis = "title_multi_ticker"
    elif primary_match:
        basis = "cluster_primary"
    elif title_match:
        basis = "title"
    elif snippet_match and other_subject:
        basis = "snippet_related"
    elif snippet_match:
        basis = "snippet"
    elif full_text_match:
        basis = "full_text_related"
    else:
        basis = "none"
    ticker_specific = basis in {"cluster_primary", "title", "snippet"}
    if multi_ticker and not primary_match and basis != "title":
        ticker_specific = False
    return _TickerMatch(
        basis=basis,
        strong_match=basis != "none",
        ticker_specific=ticker_specific,
        multi_ticker=multi_ticker,
        primary_ticker=context.primary_ticker,
    )


def _headline_centers_other_company(title: str, ticker: TrackedTicker) -> bool:
    if _matches_ticker(title, ticker):
        return False
    company = r"[A-Z][A-Za-z0-9.&'-]*(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,3}"
    event = (
        r"stock|shares|earnings|results|revenue|guidance|raises|reports|reviews|"
        r"launches|unveils|signs|expands|soars|falls|sinks|jumps"
    )
    possessive = rf"^{company}(?:'s|\u2019s)\s+(?:{event})\b"
    direct = rf"^{company}\s+(?:{event})\b"
    return re.search(possessive, title) is not None or re.search(direct, title) is not None


def _ticker_ranking_score(summary: ArticleMicroSummary, match: _TickerMatch) -> float:
    match_points = {
        "cluster_primary": 28,
        "title": 26,
        "snippet": 16,
        "title_multi_ticker": 8,
        "snippet_related": -8,
        "full_text_related": -12,
        "none": -40,
    }.get(match.basis, 0)
    multi_ticker_penalty = -18 if match.multi_ticker and not match.ticker_specific else 0
    headline_penalty = _headline_penalty(summary.title)
    return round(summary.ranking_score + match_points + multi_ticker_penalty + headline_penalty, 2)


def _headline_penalty(title: str) -> int:
    lowered = title.casefold()
    if re.search(r"\bprediction\b.*\b(stock|shares|trade|price|worth)\b", lowered):
        return -45
    if re.search(r"\bis .+ (?:a )?good stock to buy(?: now)?\b", lowered):
        return -45
    if any(phrase in lowered for phrase in ("stocks to watch", "best stocks to buy", "top stocks to buy")):
        return -24
    return 0


def _read_first_disallowed(summary: ArticleMicroSummary) -> bool:
    lowered = summary.title.casefold()
    return (
        _headline_penalty(summary.title) <= -40
        or GENERIC_BUY_SELL_HOLD_OPINION in summary.event_types
        or PREDICTION_OR_PRICE_TARGET_CLICKBAIT in summary.event_types
        or (
            ANALYST_RATING_OR_PRICE_TARGET in summary.event_types
            and len(summary.tickers) <= 1
        )
        or any(phrase in lowered for phrase in ("buy now", "sell now", "price target"))
    )


def _read_first_reason(summary: ArticleMicroSummary, match: _TickerMatch) -> str:
    events = set(summary.event_types)
    lowered = summary.title.casefold()
    regulatory_terms = (
        "antitrust",
        "backdoor",
        "ban",
        "court",
        "export",
        "investigation",
        "lawsuit",
        "legal",
        "probe",
        "regulator",
        "regulatory",
        "restriction",
        "scrutiny",
    )
    earnings_terms = ("earnings", "eps", "guidance", "profit", "quarter", "results", "revenue")
    if REGULATORY_OR_LEGAL in events or any(term in lowered for term in regulatory_terms):
        return "because it covers a material regulatory or legal issue"
    if (
        EARNINGS_OR_RESULTS in events
        or GUIDANCE_OR_FORECAST in events
    ) and any(term in lowered for term in earnings_terms):
        return "because it covers earnings or guidance"
    if STOCK_PRICE_MOVE in events:
        return "because it explains a notable stock move"
    if (
        PRODUCT_OR_AI_OR_CHIP_NEWS in events
        or PARTNERSHIP_OR_CONTRACT in events
        or CUSTOMER_OR_DEMAND_SIGNAL in events
    ):
        return "because it is the clearest ticker-specific catalyst today"
    if summary.source_quality_tier == 1:
        return "because it comes from a higher-trust source"
    if summary.summary_basis == "snippet":
        return "because it is the best available ticker-specific article, but only snippet text was available"
    return "because it is the clearest ticker-specific catalyst available"


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
    positive = max(
        (item for item in reads if item.ticker_specific and item.sentiment_score > 0),
        key=lambda item: item.sentiment_score,
        default=None,
    )
    negative = min(
        (item for item in reads if item.ticker_specific and item.sentiment_score < 0),
        key=lambda item: item.sentiment_score,
        default=None,
    )
    if read_first is not None:
        summary = f"{ticker.symbol}: {read_first.article_summary}"
    elif ticker_clusters:
        focus = ticker_clusters[0].cluster_summary
        summary = focus if focus.casefold().startswith(f"{ticker.symbol.casefold()}:") else f"{ticker.symbol}: {focus}"
    else:
        summary = f"{ticker.symbol}: No matched story was available in the current report window."
    if read_first is not None:
        summary += f" Read first: {read_first.title} {read_first.read_first_reason or ''}.".replace(" .", ".")
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
