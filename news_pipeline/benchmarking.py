"""External sentiment benchmark comparisons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from .models import Article
from .sentiment import analyze_sentiment
from .sentiment_coverage import TickerSentimentCoverage
from .tickers import load_tracked_tickers


@dataclass(frozen=True)
class ArticleSentimentBenchmark:
    ticker: str
    canonical_url: str
    title: str
    internal_sentiment_raw: float
    internal_sentiment_weighted: float
    external_alpha_vantage_sentiment: float
    external_alpha_vantage_label: str
    relevance_score: float
    sentiment_disagreement_flag: bool
    sentiment_disagreement_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_alpha_vantage_benchmarks(
    *,
    articles: Sequence[Article],
    ticker_coverage: Mapping[str, TickerSentimentCoverage],
) -> tuple[
    tuple[ArticleSentimentBenchmark, ...],
    Mapping[str, TickerSentimentCoverage],
]:
    rows: list[ArticleSentimentBenchmark] = []
    grouped: dict[str, list[ArticleSentimentBenchmark]] = defaultdict(list)
    tracked = {ticker.symbol for ticker in load_tracked_tickers()}
    for article in articles:
        if article.metadata.get("api_provider") != "alpha_vantage_news":
            continue
        internal = analyze_sentiment(
            article.article_id or article.canonical_url,
            article.full_text or article.snippet or article.title,
            "full_text" if article.full_text else "snippet" if article.snippet else "title",
        )
        for entry in article.metadata.get("ticker_sentiment") or ():
            if not isinstance(entry, Mapping):
                continue
            ticker = str(entry.get("ticker") or "").upper()
            external = _optional_float(entry.get("ticker_sentiment_score"))
            if ticker not in tracked or external is None:
                continue
            relevance = max(
                0.0,
                min(1.0, _optional_float(entry.get("relevance_score")) or 1.0),
            )
            disagreement, reason = _sentiment_disagreement(
                internal.score,
                external,
            )
            row = ArticleSentimentBenchmark(
                ticker=ticker,
                canonical_url=article.canonical_url,
                title=article.title,
                internal_sentiment_raw=round(internal.score, 4),
                internal_sentiment_weighted=round(
                    internal.score * relevance,
                    4,
                ),
                external_alpha_vantage_sentiment=round(external, 4),
                external_alpha_vantage_label=str(
                    entry.get("ticker_sentiment_label")
                    or article.metadata.get("overall_sentiment_label")
                    or "unknown"
                ),
                relevance_score=round(relevance, 4),
                sentiment_disagreement_flag=disagreement,
                sentiment_disagreement_reason=reason,
            )
            rows.append(row)
            grouped[ticker].append(row)

    updated = dict(ticker_coverage)
    for ticker, coverage in ticker_coverage.items():
        benchmark_rows = grouped.get(ticker, ())
        total_relevance = sum(row.relevance_score for row in benchmark_rows)
        alpha_weighted = (
            sum(
                row.external_alpha_vantage_sentiment * row.relevance_score
                for row in benchmark_rows
            )
            / total_relevance
            if total_relevance
            else 0.0
        )
        disagreement_count = sum(
            row.sentiment_disagreement_flag for row in benchmark_rows
        )
        updated[ticker] = replace(
            coverage,
            internal_weighted_sentiment=coverage.weighted_sentiment,
            alpha_vantage_weighted_sentiment=round(alpha_weighted, 4),
            benchmark_coverage_count=len(benchmark_rows),
            benchmark_disagreement_count=disagreement_count,
            benchmark_alignment_grade=_alignment_grade(
                len(benchmark_rows),
                disagreement_count,
            ),
        )
    return tuple(rows), updated


def benchmark_rows_with_ticker_summary(
    rows: Sequence[ArticleSentimentBenchmark],
    coverage: Mapping[str, TickerSentimentCoverage],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            **row.as_dict(),
            "internal_weighted_sentiment": coverage[
                row.ticker
            ].internal_weighted_sentiment,
            "alpha_vantage_weighted_sentiment": coverage[
                row.ticker
            ].alpha_vantage_weighted_sentiment,
            "benchmark_coverage_count": coverage[
                row.ticker
            ].benchmark_coverage_count,
            "benchmark_disagreement_count": coverage[
                row.ticker
            ].benchmark_disagreement_count,
            "benchmark_alignment_grade": coverage[
                row.ticker
            ].benchmark_alignment_grade,
        }
        for row in rows
    )


def _sentiment_disagreement(
    internal: float,
    external: float,
) -> tuple[bool, str]:
    internal_direction = _direction(internal)
    external_direction = _direction(external)
    if (
        internal_direction != "neutral"
        and external_direction != "neutral"
        and internal_direction != external_direction
    ):
        return True, "direction_mismatch"
    if abs(internal - external) >= 0.5:
        return True, "magnitude_gap"
    return False, "aligned_within_threshold"


def _direction(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def _alignment_grade(coverage: int, disagreements: int) -> str:
    if coverage <= 0:
        return "unavailable"
    ratio = disagreements / coverage
    if ratio <= 0.2:
        return "aligned"
    if ratio <= 0.5:
        return "mixed"
    return "divergent"


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
