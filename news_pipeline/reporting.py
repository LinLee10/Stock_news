"""Report data contract and local artifact writers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from html import escape
from pathlib import Path
from typing import Iterable, Mapping

from .charts import write_placeholder_chart
from .models import RunResult
from .sentiment_coverage import TickerSentimentCoverage, WeightedArticleSentiment
from .source_quality import SourceQualitySummary
from .summaries import ArticleMicroSummary, RankedArticleRecommendation, TickerDailySummary


@dataclass(frozen=True)
class ArticleLink:
    title: str
    url: str
    source: str | None = None
    source_quality_label: str = "tier_2_usable"


@dataclass(frozen=True)
class ExtractionSummary:
    article_pages_fetched: int = 0
    publisher_article_fetches: int = 0
    google_news_wrappers_skipped: int = 0
    google_news_wrappers_resolved: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    snippet_fallbacks: int = 0
    title_fallbacks: int = 0
    sentiment_basis_counts: Mapping[str, int] = field(default_factory=lambda: {"full_text": 0, "snippet": 0, "title": 0})
    top_extraction_failure_reasons: Mapping[str, int] = field(default_factory=dict)
    extraction_method_counts: Mapping[str, int] = field(default_factory=dict)
    extraction_failure_reason: str | None = None
    extractor_diagnostics: Mapping[str, bool] = field(default_factory=lambda: {
        "trafilatura_available": False,
        "newspaper3k_available": False,
        "internal_parser_available": True,
    })
    source_quality_summary: SourceQualitySummary = field(default_factory=SourceQualitySummary)
    show_excluded_source_diagnostics: bool = False
    extraction_queue_size: int = 0
    extraction_selected_count: int = 0
    extraction_skipped_count: int = 0
    extraction_skipped_reasons: Mapping[str, int] = field(default_factory=dict)
    extraction_success_rate: float = 0.0
    extraction_success_rate_by_publisher: Mapping[str, float] = field(default_factory=dict)
    extraction_success_rate_by_source_provider: Mapping[str, float] = field(default_factory=dict)
    extraction_attempted_count: int = 0
    extraction_budget_unused_count: int = 0
    direct_publisher_candidates: int = 0
    google_wrapper_candidates: int = 0
    google_wrappers_unresolved: int = 0
    full_text_accepted_count: int = 0
    usable_full_text_count: int = 0
    weak_text_count: int = 0
    snippet_fallback_count: int = 0
    title_fallback_count: int = 0
    blocked_or_shell_count: int = 0
    extraction_quality_grade_counts: Mapping[str, int] = field(default_factory=dict)
    extractor_method_success_counts: Mapping[str, int] = field(default_factory=dict)
    publisher_success_rates: Mapping[str, float] = field(default_factory=dict)
    publisher_profiles: tuple[Mapping[str, object], ...] = ()
    top_unresolved_wrapper_publishers: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendArticlePoolSummary:
    backend_candidate_articles: int = 0
    backend_visible_articles: int = 0
    backend_scored_articles: int = 0
    backend_extracted_articles: int = 0


@dataclass(frozen=True)
class EmailDisplaySummary:
    email_visible_stories: int = 0
    email_visible_ranked_reads: int = 0
    max_email_stories: int = 60
    max_ranked_reads_per_ticker: int = 3


@dataclass(frozen=True)
class EventClusterRow:
    ticker: str
    title: str
    primary_link: str
    publisher_count: int
    source_count: int
    article_count: int
    first_seen_at: str | None = None
    latest_seen_at: str | None = None
    primary_published_at: str | None = None
    recency_bucket: str = "unknown"
    tickers_mentioned: tuple[str, ...] = ()
    weighted_cluster_sentiment: float | None = None
    extraction_basis: str = "not_fetched"
    source_quality_label: str = "tier_2_usable"
    cluster_summary: str = ""
    cluster_summary_basis: str = "title"
    cluster_reading_priority: str = "background_only"
    ranking_score: float = 0.0
    supporting_links: tuple[ArticleLink, ...] = ()


@dataclass(frozen=True)
class PortfolioSentimentRow:
    ticker: str
    company_name: str
    sentiment_30d: float
    article_count_30d: int
    sentiment_basis: str
    today_signal_sentiment: float = 0.0
    recent_pulse_sentiment: float = 0.0
    weekly_trend_sentiment: float = 0.0
    background_context_sentiment: float = 0.0
    weighted_sentiment_score: float = 0.0
    article_count_24h: int = 0
    article_count_3d: int = 0
    article_count_7d: int = 0
    mention_velocity: str = "limited_history"
    source_diversity: int = 0
    top_event_clusters: tuple[EventClusterRow, ...] = ()


@dataclass(frozen=True)
class WatchlistForecastRow:
    ticker: str
    next_close_direction: str
    confidence: float
    driver: str


@dataclass(frozen=True)
class MentionLeaderRow:
    ticker: str
    mentions_7d: int
    sentiment_avg: float


@dataclass(frozen=True)
class MostMentionedRow:
    ticker: str
    mentions: int
    rank: int


@dataclass(frozen=True)
class EmergingNameRow:
    ticker: str
    company_name: str
    mentions_7d: int
    prior_mentions_30d: int
    reason: str


@dataclass(frozen=True)
class DailyReportInput:
    report_date: str
    portfolio_sentiment: tuple[PortfolioSentimentRow, ...] = ()
    watchlist_sentiment: tuple[PortfolioSentimentRow, ...] = ()
    watchlist_forecasts: tuple[WatchlistForecastRow, ...] = ()
    mention_leaders_7d: tuple[MentionLeaderRow, ...] = ()
    most_mentioned: tuple[MostMentionedRow, ...] = ()
    emerging_names: tuple[EmergingNameRow, ...] = ()
    article_links_by_ticker: Mapping[str, tuple[ArticleLink, ...]] = field(default_factory=dict)
    event_clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]] = field(default_factory=dict)
    portfolio_summaries: tuple[TickerDailySummary, ...] = ()
    watchlist_summaries: tuple[TickerDailySummary, ...] = ()
    ranked_reads_by_ticker: Mapping[str, tuple[RankedArticleRecommendation, ...]] = field(default_factory=dict)
    article_summaries: tuple[ArticleMicroSummary, ...] = ()
    extraction_summary: ExtractionSummary = field(default_factory=ExtractionSummary)
    data_source_label: str = "local RSS fixtures"
    report_warnings: tuple[str, ...] = ()
    backend_article_pool_summary: BackendArticlePoolSummary = field(default_factory=BackendArticlePoolSummary)
    source_coverage_diagnostics: Mapping[str, object] = field(default_factory=dict)
    extraction_coverage_diagnostics: Mapping[str, object] = field(default_factory=dict)
    dedupe_diagnostics: Mapping[str, object] = field(default_factory=dict)
    ticker_match_confidence_summary: Mapping[str, object] = field(default_factory=dict)
    article_type_counts: Mapping[str, int] = field(default_factory=dict)
    ticker_sentiment_coverage: Mapping[str, TickerSentimentCoverage] = field(default_factory=dict)
    email_display_summary: EmailDisplaySummary = field(default_factory=EmailDisplaySummary)
    backend_sentiment_inputs: tuple[WeightedArticleSentiment, ...] = ()
    sentiment_benchmark_rows: tuple[Mapping[str, object], ...] = ()
    sec_event_candidates: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class DailyReportContract:
    report_date: str
    output_dir: str
    portfolio_30d_sentiment_table: tuple[PortfolioSentimentRow, ...]
    watchlist_sentiment_table: tuple[PortfolioSentimentRow, ...]
    watchlist_next_close_table: tuple[WatchlistForecastRow, ...]
    mention_leaders_7d_table: tuple[MentionLeaderRow, ...]
    top_10_most_mentioned_table: tuple[MostMentionedRow, ...]
    emerging_names_table: tuple[EmergingNameRow, ...]
    recency_sections: Mapping[str, tuple[PortfolioSentimentRow, ...]]
    top_event_clusters: tuple[EventClusterRow, ...]
    event_clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]]
    supporting_article_links: Mapping[str, tuple[ArticleLink, ...]]
    portfolio_summaries: tuple[TickerDailySummary, ...]
    watchlist_summaries: tuple[TickerDailySummary, ...]
    ranked_reads_by_ticker: Mapping[str, tuple[RankedArticleRecommendation, ...]]
    article_summaries: tuple[ArticleMicroSummary, ...]
    extraction_summary: ExtractionSummary
    csv_attachments: tuple[str, ...]
    chart_attachments: tuple[str, ...]
    html_preview_report: str
    daily_summary: str
    data_source_label: str
    report_warnings: tuple[str, ...]
    backend_article_pool_summary: BackendArticlePoolSummary
    source_coverage_diagnostics: Mapping[str, object]
    extraction_coverage_diagnostics: Mapping[str, object]
    dedupe_diagnostics: Mapping[str, object]
    ticker_match_confidence_summary: Mapping[str, object]
    article_type_counts: Mapping[str, int]
    ticker_sentiment_coverage: Mapping[str, TickerSentimentCoverage]
    email_display_summary: EmailDisplaySummary
    supplemental_csv_artifacts: tuple[str, ...]


def summarize_run(run: RunResult) -> dict[str, object]:
    """Return a simple serializable run summary."""
    return {
        "run_id": run.run_id,
        "status": run.status,
        "articles_seen": run.articles_seen,
        "articles_stored": run.articles_stored,
        "duplicates": run.duplicates,
        "error_count": len(run.errors),
    }


def build_daily_report(
    report_input: DailyReportInput,
    *,
    artifacts_dir: str | Path = "artifacts",
) -> DailyReportContract:
    """Build report tables and write local CSV/chart attachments."""
    output_dir = _run_output_dir(artifacts_dir, report_input.report_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_10 = tuple(sorted(report_input.most_mentioned, key=lambda row: row.rank)[:10])
    report_warnings = _report_warnings(report_input.report_date, report_input.report_warnings)
    recency_sections = _recency_section_rows(report_input)
    top_event_clusters = _top_event_clusters(report_input.event_clusters_by_ticker)
    csv_attachments = (
        _write_csv(
            output_dir / "portfolio_30d_sentiment.csv",
            _sentiment_csv_fields(),
            report_input.portfolio_sentiment,
        ),
        _write_csv(
            output_dir / "watchlist_sentiment.csv",
            _sentiment_csv_fields(),
            report_input.watchlist_sentiment,
        ),
        _write_csv(
            output_dir / "watchlist_next_close.csv",
            ("ticker", "next_close_direction", "confidence", "driver"),
            report_input.watchlist_forecasts,
        ),
        _write_csv(
            output_dir / "mention_leaders_7d.csv",
            ("ticker", "mentions_7d", "sentiment_avg"),
            report_input.mention_leaders_7d,
        ),
        _write_csv(
            output_dir / "top_10_most_mentioned.csv",
            ("ticker", "mentions", "rank"),
            top_10,
        ),
        _write_csv(
            output_dir / "emerging_names.csv",
            ("ticker", "company_name", "mentions_7d", "prior_mentions_30d", "reason"),
            report_input.emerging_names,
        ),
    )
    supplemental_csv_artifacts = (
        _write_csv(
            output_dir / "backend_articles_scored.csv",
            (
                "ticker",
                "canonical_url",
                "title",
                "cluster_id",
                "sentiment_raw",
                "sentiment_basis",
                "sentiment_weight",
                "sentiment_weight_reasons",
                "ticker_match_confidence",
                "ticker_match_confidence_label",
                "ticker_match_reason",
                "article_type",
                "primary_cluster_article",
            ),
            report_input.backend_sentiment_inputs,
        ),
        _write_csv(
            output_dir / "ticker_sentiment_coverage.csv",
            (
                "ticker",
                "article_count_scored",
                "full_text_scored_count",
                "snippet_scored_count",
                "title_scored_count",
                "weighted_sentiment",
                "positive_article_count",
                "negative_article_count",
                "neutral_article_count",
                "high_confidence_article_count",
                "low_confidence_article_count",
                "top_positive_cluster",
                "top_negative_cluster",
                "sentiment_coverage_grade",
                "internal_weighted_sentiment",
                "alpha_vantage_weighted_sentiment",
                "benchmark_coverage_count",
                "benchmark_disagreement_count",
                "benchmark_alignment_grade",
            ),
            report_input.ticker_sentiment_coverage.values(),
        ),
        _write_dict_rows_csv(
            output_dir / "ticker_sentiment_benchmark.csv",
            report_input.sentiment_benchmark_rows,
        ),
        _write_dict_rows_csv(
            output_dir / "sec_event_candidates.csv",
            report_input.sec_event_candidates,
        ),
        _write_mapping_csv(
            output_dir / "article_type_counts.csv",
            "article_type",
            "count",
            report_input.article_type_counts,
        ),
        _write_mapping_csv(
            output_dir / "extraction_diagnostics.csv",
            "diagnostic",
            "value",
            report_input.extraction_coverage_diagnostics,
        ),
        _write_dict_rows_csv(
            output_dir / "source_profiles.csv",
            tuple(report_input.source_coverage_diagnostics.get("source_profiles", ())),
        ),
        _write_mapping_csv(
            output_dir / "source_family_counts.csv",
            "source_family",
            "article_count",
            report_input.source_coverage_diagnostics.get("source_family_counts", {}),
        ),
        _write_mapping_csv(
            output_dir / "source_acquisition_diagnostics.csv",
            "diagnostic",
            "value",
            report_input.source_coverage_diagnostics,
        ),
        _write_mapping_csv(
            output_dir / "source_diversity_diagnostics.csv",
            "diagnostic",
            "value",
            {
                "source_diversity_score": report_input.source_coverage_diagnostics.get(
                    "source_diversity_score",
                    0.0,
                ),
                "source_balance_score": report_input.source_coverage_diagnostics.get(
                    "source_balance_score",
                    0.0,
                ),
                "direct_vs_aggregator_ratio": report_input.source_coverage_diagnostics.get(
                    "direct_vs_aggregator_ratio",
                    0.0,
                ),
                "google_news_share": report_input.source_coverage_diagnostics.get(
                    "google_news_share",
                    0.0,
                ),
            },
        ),
        _write_mapping_csv(
            output_dir / "external_api_skipped_reasons.csv",
            "provider",
            "reason",
            report_input.source_coverage_diagnostics.get(
                "external_api_provider_skipped_reasons",
                {},
            ),
        ),
        _write_mapping_csv(
            output_dir / "paid_api_skipped_reasons.csv",
            "provider",
            "reason",
            report_input.source_coverage_diagnostics.get("paid_api_skipped_reasons", {}),
        ),
        _write_value_rows_csv(
            output_dir / "missing_company_ir_profiles.csv",
            "ticker",
            report_input.source_coverage_diagnostics.get("missing_company_ir_profiles", ()),
        ),
    )
    chart_attachments = (
        write_placeholder_chart(
            output_dir / "portfolio_sentiment.svg",
            "Portfolio Recency Sentiment",
            [row.ticker for row in report_input.portfolio_sentiment],
        ),
        write_placeholder_chart(
            output_dir / "mention_leaders.svg",
            "Top 7 Day Mention Leaders",
            [row.ticker for row in report_input.mention_leaders_7d],
        ),
        write_placeholder_chart(
            output_dir / "watchlist_sentiment.svg",
            "Watchlist Recency Sentiment",
            [row.ticker for row in report_input.watchlist_sentiment],
        ),
    )
    html_preview_report = _write_html_preview(output_dir / "daily_report.html", report_input, top_10)

    return DailyReportContract(
        report_date=report_input.report_date,
        output_dir=str(output_dir),
        portfolio_30d_sentiment_table=report_input.portfolio_sentiment,
        watchlist_sentiment_table=report_input.watchlist_sentiment,
        watchlist_next_close_table=report_input.watchlist_forecasts,
        mention_leaders_7d_table=report_input.mention_leaders_7d,
        top_10_most_mentioned_table=top_10,
        emerging_names_table=report_input.emerging_names,
        recency_sections=recency_sections,
        top_event_clusters=top_event_clusters,
        event_clusters_by_ticker=report_input.event_clusters_by_ticker,
        supporting_article_links=report_input.article_links_by_ticker,
        portfolio_summaries=report_input.portfolio_summaries,
        watchlist_summaries=report_input.watchlist_summaries,
        ranked_reads_by_ticker=report_input.ranked_reads_by_ticker,
        article_summaries=report_input.article_summaries,
        extraction_summary=report_input.extraction_summary,
        csv_attachments=csv_attachments,
        chart_attachments=chart_attachments,
        html_preview_report=html_preview_report,
        daily_summary=_plain_english_summary(report_input, top_10),
        data_source_label=report_input.data_source_label,
        report_warnings=report_warnings,
        backend_article_pool_summary=report_input.backend_article_pool_summary,
        source_coverage_diagnostics=report_input.source_coverage_diagnostics,
        extraction_coverage_diagnostics=report_input.extraction_coverage_diagnostics,
        dedupe_diagnostics=report_input.dedupe_diagnostics,
        ticker_match_confidence_summary=report_input.ticker_match_confidence_summary,
        article_type_counts=report_input.article_type_counts,
        ticker_sentiment_coverage=report_input.ticker_sentiment_coverage,
        email_display_summary=report_input.email_display_summary,
        supplemental_csv_artifacts=supplemental_csv_artifacts,
    )


def _run_output_dir(artifacts_dir: str | Path, report_date: str) -> Path:
    base = Path(artifacts_dir)
    if base.name != "artifacts":
        base = base / "artifacts"
    return base / "runs" / report_date


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})
    return str(path)


def _write_mapping_csv(
    path: Path,
    key_name: str,
    value_name: str,
    values: Mapping[str, object],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(key_name, value_name))
        writer.writeheader()
        for key, value in sorted(values.items()):
            writer.writerow({key_name: key, value_name: value})
    return str(path)


def _write_dict_rows_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> str:
    rows = tuple(rows)
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ("source_id",))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return str(path)


def _write_value_rows_csv(
    path: Path,
    fieldname: str,
    values: Iterable[object],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(fieldname,))
        writer.writeheader()
        for value in values:
            writer.writerow({fieldname: value})
    return str(path)


def _sentiment_csv_fields() -> tuple[str, ...]:
    return (
        "ticker",
        "company_name",
        "sentiment_30d",
        "article_count_30d",
        "sentiment_basis",
        "today_signal_sentiment",
        "recent_pulse_sentiment",
        "weekly_trend_sentiment",
        "background_context_sentiment",
        "weighted_sentiment_score",
        "article_count_24h",
        "article_count_3d",
        "article_count_7d",
        "mention_velocity",
        "source_diversity",
    )


def _recency_section_rows(report_input: DailyReportInput) -> dict[str, tuple[PortfolioSentimentRow, ...]]:
    rows = list(report_input.portfolio_sentiment) + list(report_input.watchlist_sentiment)
    return {
        "today_signal": tuple(_rank_recency_rows(rows, "today_signal_sentiment", "article_count_24h")),
        "recent_pulse": tuple(_rank_recency_rows(rows, "recent_pulse_sentiment", "article_count_3d")),
        "weekly_trend": tuple(_rank_recency_rows(rows, "weekly_trend_sentiment", "article_count_7d")),
        "background_context": tuple(_rank_recency_rows(rows, "background_context_sentiment", "article_count_30d")),
    }


def _rank_recency_rows(
    rows: list[PortfolioSentimentRow],
    sentiment_field: str,
    count_field: str,
) -> list[PortfolioSentimentRow]:
    return sorted(
        rows,
        key=lambda row: (getattr(row, count_field), abs(getattr(row, sentiment_field)), row.source_diversity),
        reverse=True,
    )[:8]


def _top_event_clusters(
    clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]],
    *,
    limit: int = 25,
) -> tuple[EventClusterRow, ...]:
    clusters = [cluster for clusters in clusters_by_ticker.values() for cluster in clusters]
    bucket_rank = {
        "today_signal": 4,
        "recent_pulse": 3,
        "weekly_trend": 2,
        "background_context": 1,
    }
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                cluster.ranking_score,
                _quality_rank(cluster.source_quality_label),
                bucket_rank.get(cluster.recency_bucket, 0),
                cluster.source_count,
                cluster.publisher_count,
                cluster.article_count,
            ),
            reverse=True,
        )[:limit]
    )


def _plain_english_summary(
    report_input: DailyReportInput,
    top_10: tuple[MostMentionedRow, ...],
) -> str:
    portfolio_count = len(report_input.portfolio_sentiment)
    matched_portfolio_count = sum(1 for row in report_input.portfolio_sentiment if row.article_count_30d)
    watchlist_count = len(report_input.watchlist_forecasts)
    matched_watchlist_count = sum(1 for row in report_input.watchlist_sentiment if row.article_count_30d)
    top_mentions = ", ".join(row.ticker for row in top_10[:3]) or "none"
    emerging = ", ".join(row.ticker for row in report_input.emerging_names[:3]) or "none"
    return (
        f"Daily report for {report_input.report_date}: "
        f"{matched_portfolio_count} of {portfolio_count} portfolio names have configured ticker coverage, "
        f"{matched_watchlist_count} of {len(report_input.watchlist_sentiment)} watchlist names have configured ticker coverage, "
        f"{watchlist_count} watchlist names have placeholder direction rows from current report sentiment, "
        f"top mention leaders are {top_mentions}, and emerging names are {emerging}."
    )


def _write_html_preview(
    path: Path,
    report_input: DailyReportInput,
    top_10: tuple[MostMentionedRow, ...],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"  <title>Portfolio and Watchlist Market Briefing - {escape(report_input.report_date)}</title>",
        "  <style>",
        "    body{font-family:Arial,sans-serif;margin:24px;line-height:1.45;color:#17202a;background:#fff}",
        "    h1,h2{color:#17202a}",
        "    table{border-collapse:collapse;width:100%;margin:12px 0 24px}",
        "    th,td{border:1px solid #d7dde5;padding:8px;text-align:left;font-size:14px}",
        "    th{background:#eef3f8}",
        "    .note{background:#fff8e6;border-left:4px solid #c98900;padding:12px;margin:12px 0 20px}",
        "    .summary{background:#edf7f1;border-left:4px solid #248a4b;padding:12px;margin:12px 0 20px}",
        "    .briefing{background:#f0f6ff;border-left:4px solid #2563eb;padding:12px;margin:12px 0 20px}",
        "    .briefing ul{margin:8px 0 0;padding-left:20px}",
        "    .empty{color:#64748b}",
        "    .muted{color:#64748b;font-size:13px}",
        "    .num{text-align:right}",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Portfolio and Watchlist Market Briefing - {escape(report_input.report_date)}</h1>",
        "  <p class=\"muted\">This briefing is not investment advice. Direction rows are not predictions.</p>",
        _top_briefing_html(report_input, top_10),
        _source_coverage_line_html(report_input.source_coverage_diagnostics),
        _ticker_summaries_html("Portfolio Summary", report_input.portfolio_summaries),
        _ticker_summaries_html("Watchlist Summary", report_input.watchlist_summaries),
        _event_clusters_html(report_input.event_clusters_by_ticker),
        _ranked_reads_html(report_input.ranked_reads_by_ticker),
        _sentiment_coverage_html(report_input.ticker_sentiment_coverage),
        _recency_sections_html(report_input),
        _sentiment_table_html("Portfolio Recency Sentiment", report_input.portfolio_sentiment),
        _sentiment_table_html("Watchlist Recency Sentiment", report_input.watchlist_sentiment),
        _forecast_table_html(report_input.watchlist_forecasts),
        _mention_leaders_html(report_input.mention_leaders_7d),
        _top_10_html(top_10),
        _emerging_names_html(report_input.emerging_names),
        _article_links_html(report_input.article_links_by_ticker),
        "  <h2>Source and Extraction Diagnostics</h2>",
        _report_metadata_html(report_input, top_10),
        _backend_pool_diagnostics_html(report_input),
        _source_acquisition_diagnostics_html(report_input.source_coverage_diagnostics),
        _source_quality_summary_html(report_input.extraction_summary.source_quality_summary),
        _extraction_summary_html(report_input.extraction_summary),
        "</body>",
        "</html>",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")
    return str(path)


def _report_warnings(report_date: str, existing: tuple[str, ...]) -> tuple[str, ...]:
    warnings = list(existing)
    if report_date != date.today().isoformat():
        warnings.append("Run date differs from current date.")
    return tuple(dict.fromkeys(warnings))


def _report_metadata_html(
    report_input: DailyReportInput,
    top_10: tuple[MostMentionedRow, ...],
) -> str:
    warnings = _report_warnings(report_input.report_date, report_input.report_warnings)
    body = [
        "  <table>",
        "    <tr><th>Report Diagnostic</th><th>Value</th></tr>",
        f"    <tr><td>Run date</td><td>{escape(report_input.report_date)}</td></tr>",
        f"    <tr><td>Current local date</td><td>{escape(date.today().isoformat())}</td></tr>",
        f"    <tr><td>Data source</td><td>{escape(report_input.data_source_label)}</td></tr>",
        "    <tr><td>Delivery mode</td><td>local report only; no email sent</td></tr>",
        f"    <tr><td>External free-tier API status</td><td>{escape(str(report_input.source_coverage_diagnostics.get('external_api_status', 'disabled')))}</td></tr>",
        f"    <tr><td>Report summary</td><td>{escape(_plain_english_summary(report_input, top_10))}</td></tr>",
    ]
    if warnings:
        body.append(f"    <tr><td>Warnings</td><td>{escape(' '.join(warnings))}</td></tr>")
    body.append("  </table>")
    body.append(
        "  <p class=\"muted\">Sentiment is deterministic placeholder logic until a stronger model is wired in. "
        "Summaries use extracted full text when available, otherwise snippets or titles.</p>"
    )
    return "\n".join(body)


def _sentiment_coverage_html(
    coverage: Mapping[str, TickerSentimentCoverage],
) -> str:
    rows = [row for row in coverage.values() if row.article_count_scored]
    body = [
        "  <h2>Sentiment Coverage Summary</h2>",
        "  <table>",
        "    <tr><th>Ticker</th><th>Coverage</th><th>Weighted Sentiment</th><th>Scored</th><th>Full Text</th><th>Snippets</th><th>High Confidence</th><th>Low Confidence</th></tr>",
    ]
    if rows:
        body.extend(
            "    <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td>{escape(row.sentiment_coverage_grade)}</td>"
            f"<td class=\"num\">{row.weighted_sentiment:.4f}</td>"
            f"<td class=\"num\">{row.article_count_scored}</td>"
            f"<td class=\"num\">{row.full_text_scored_count}</td>"
            f"<td class=\"num\">{row.snippet_scored_count}</td>"
            f"<td class=\"num\">{row.high_confidence_article_count}</td>"
            f"<td class=\"num\">{row.low_confidence_article_count}</td>"
            "</tr>"
            for row in sorted(rows, key=lambda item: item.ticker)
        )
    else:
        body.append("    <tr><td colspan=\"8\" class=\"empty\">No weighted sentiment coverage was available.</td></tr>")
    body.append("  </table>")
    return "\n".join(body)


def _backend_pool_diagnostics_html(report_input: DailyReportInput) -> str:
    backend = report_input.backend_article_pool_summary
    email = report_input.email_display_summary
    confidence = report_input.ticker_match_confidence_summary
    diagnostics = report_input.source_coverage_diagnostics
    selection_reasons = "; ".join(
        f"{ticker}: {', '.join(str(reason) for reason in reasons)}"
        for ticker, reasons in sorted(
            (diagnostics.get("alpha_vantage_selection_reasons") or {}).items()
        )
    )
    sentiment_changes = "; ".join(
        f"{ticker}: {float(values.get('change', 0.0)):+.4f}"
        for ticker, values in sorted(
            (diagnostics.get("sentiment_change_since_prior_run") or {}).items()
        )
    )
    return "\n".join(
        [
            "  <h2>Backend and Email Pool Summary</h2>",
            "  <table>",
            "    <tr><th>Backend Candidates</th><th>Backend Visible</th><th>Backend Scored</th><th>Backend Full Text</th><th>Email Stories</th><th>Email Ranked Reads</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{backend.backend_candidate_articles}</td>"
            f"<td class=\"num\">{backend.backend_visible_articles}</td>"
            f"<td class=\"num\">{backend.backend_scored_articles}</td>"
            f"<td class=\"num\">{backend.backend_extracted_articles}</td>"
            f"<td class=\"num\">{email.email_visible_stories}</td>"
            f"<td class=\"num\">{email.email_visible_ranked_reads}</td>"
            "</tr>",
            "  </table>",
            "  <h3>Benchmark and Event Memory</h3>",
            "  <table>",
            "    <tr><th>Alpha Vantage Requests</th><th>Alpha Vantage Articles</th><th>Benchmark Disagreements</th><th>SEC Event Candidates</th><th>Event Memory Records</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{int(diagnostics.get('alpha_vantage_news_requests_used', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('alpha_vantage_news_articles_returned', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('alpha_vantage_benchmark_disagreement_count', 0))}</td>"
            f"<td class=\"num\">{sum(int(value) for value in (diagnostics.get('sec_event_candidates_by_form_type') or {}).values())}</td>"
            f"<td class=\"num\">{int(diagnostics.get('event_memory_records_written', 0))}</td>"
            "</tr>",
            "  </table>",
            "  <table>",
            "    <tr><th>Alpha Selected Tickers</th><th>Weak Coverage Tickers</th><th>History Status</th><th>New Events</th><th>Exact URL Repeats</th><th>Fuzzy Repeats</th><th>Sentiment Changes</th></tr>",
            "    <tr>"
            f"<td>{escape(', '.join(diagnostics.get('alpha_vantage_selected_tickers') or ()) or 'none')}</td>"
            f"<td>{escape(', '.join(diagnostics.get('weak_coverage_tickers') or ()) or 'none')}</td>"
            f"<td>{escape(str(diagnostics.get('history_status') or 'history_building'))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('new_events_since_prior_run', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('exact_repeated_events_from_prior_run', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('fuzzy_repeated_events_from_prior_run', 0))}</td>"
            f"<td class=\"num\">{len(diagnostics.get('sentiment_change_since_prior_run') or {})}</td>"
            "</tr>",
            "  </table>",
            "  <p><strong>Event identity methods:</strong> "
            f"{escape(str(diagnostics.get('event_identity_method_counts') or {}))}; "
            f"similarity threshold: {float(diagnostics.get('event_similarity_threshold', 0.0)):.2f}; "
            f"lookback days: {int(diagnostics.get('event_memory_lookback_days', 3))}; "
            f"prior runs: {len(diagnostics.get('prior_runs_considered') or ())}; "
            f"prior records: {int(diagnostics.get('prior_event_records_considered', 0))}</p>",
            f"  <p><strong>Alpha selection reasons:</strong> {escape(selection_reasons or 'none')}</p>",
            f"  <p><strong>Sentiment changes since prior run:</strong> {escape(sentiment_changes or 'none')}</p>",
            "  <table>",
            "    <tr><th>Ticker Match Confidence</th><th>Count</th></tr>",
            f"    <tr><td>High</td><td class=\"num\">{int(confidence.get('high_confidence_matches', 0))}</td></tr>",
            f"    <tr><td>Medium</td><td class=\"num\">{int(confidence.get('medium_confidence_matches', 0))}</td></tr>",
            f"    <tr><td>Low</td><td class=\"num\">{int(confidence.get('low_confidence_matches', 0))}</td></tr>",
            "  </table>",
        ]
    )


def _recency_sections_html(report_input: DailyReportInput) -> str:
    recency_sections = _recency_section_rows(report_input)
    return "\n".join(
        [
            _recency_bucket_html(
                "Today's Signal",
                "0-24 hours before run date. Weight: 1.0.",
                list(recency_sections["today_signal"]),
                "today_signal_sentiment",
                "article_count_24h",
            ),
            _recency_bucket_html(
                "Recent Pulse",
                "1-3 days before run date. Weight: 0.7.",
                list(recency_sections["recent_pulse"]),
                "recent_pulse_sentiment",
                "article_count_3d",
            ),
            _recency_bucket_html(
                "Weekly Trend",
                "4-7 days before run date. Weight: 0.4.",
                list(recency_sections["weekly_trend"]),
                "weekly_trend_sentiment",
                "article_count_7d",
            ),
            _recency_bucket_html(
                "Background Context",
                "8-30 days before run date. Weight: 0.15. Older archive context is excluded from daily sentiment.",
                list(recency_sections["background_context"]),
                "background_context_sentiment",
                "article_count_30d",
            ),
        ]
    )


def _top_briefing_html(
    report_input: DailyReportInput,
    top_10: tuple[MostMentionedRow, ...],
) -> str:
    bullets = [
        _top_mention_bullet(top_10),
        _event_bullet(report_input, direction="negative"),
        _event_bullet(report_input, direction="positive"),
        _quality_caveat_bullet(report_input),
        "Watchlist direction rows are placeholder direction from current report sentiment, not predictions.",
    ]
    return "\n".join(
        [
            "  <section class=\"briefing\">",
            "    <h2>Daily Briefing</h2>",
            "    <ul>",
            *[f"      <li>{escape(bullet)}</li>" for bullet in bullets if bullet][:5],
            "    </ul>",
            "  </section>",
        ]
    )


def _source_coverage_line_html(diagnostics: Mapping[str, object]) -> str:
    external = diagnostics.get("external_api_count", 0)
    external_label = (
        str(external)
        if external
        else str(diagnostics.get("external_api_status") or "disabled")
    )
    dominated = diagnostics.get("google_dominated_tickers") or ()
    return (
        "  <p class=\"muted\"><strong>Source Coverage:</strong> "
        f"Official or SEC: {int(diagnostics.get('official_source_count', 0))} &middot; "
        f"Direct publishers: {int(diagnostics.get('direct_publisher_count', 0))} &middot; "
        f"Press wires: {int(diagnostics.get('press_release_wire_count', 0))} &middot; "
        f"External free tier APIs: {escape(external_label)} &middot; "
        f"Google backstop: {int(diagnostics.get('google_news_backstop_count', 0))} &middot; "
        f"Google dominated tickers: {len(dominated)}</p>"
    )


def _source_acquisition_diagnostics_html(
    diagnostics: Mapping[str, object],
) -> str:
    raw_external = sum(
        int(value)
        for value in (
            diagnostics.get("raw_external_articles_by_provider") or {}
        ).values()
    )
    post_dedup_external = sum(
        int(value)
        for value in (
            diagnostics.get("post_dedup_external_articles_by_provider") or {}
        ).values()
    )
    sentiment_external = sum(
        int(value)
        for value in (
            diagnostics.get("articles_used_for_sentiment_by_provider") or {}
        ).values()
    )
    return "\n".join(
        [
            "  <h2>Source Acquisition Summary</h2>",
            "  <table>",
            "    <tr><th>Official</th><th>Company IR</th><th>Press Wires</th><th>Direct Publishers</th><th>External APIs</th><th>Google Backstop</th><th>Google Share</th><th>Google-Dominated Tickers</th><th>Diversity</th><th>Balance</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{int(diagnostics.get('official_source_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('company_ir_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('press_release_wire_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('direct_publisher_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('external_api_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('google_news_backstop_count', 0))}</td>"
            f"<td class=\"num\">{float(diagnostics.get('google_news_share', 0.0)):.1%}</td>"
            f"<td class=\"num\">{len(diagnostics.get('google_dominated_tickers') or ())}</td>"
            f"<td class=\"num\">{float(diagnostics.get('source_diversity_score', 0.0)):.1f}</td>"
            f"<td class=\"num\">{float(diagnostics.get('source_balance_score', 0.0)):.1f}</td>"
            "</tr>",
            "  </table>",
            "  <h3>External API Quality Diagnostics</h3>",
            "  <table>",
            "    <tr><th>Raw External</th><th>Post-Dedup External</th><th>Used for Sentiment</th><th>Top Provider Raw</th><th>Top Provider Post-Dedup</th><th>Provider Warning</th><th>Top Ticker Raw</th><th>Top Ticker Post-Dedup</th><th>Ticker Warning</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{raw_external}</td>"
            f"<td class=\"num\">{post_dedup_external}</td>"
            f"<td class=\"num\">{sentiment_external}</td>"
            f"<td class=\"num\">{float(diagnostics.get('top_provider_share_raw', 0.0)):.1%}</td>"
            f"<td class=\"num\">{float(diagnostics.get('top_provider_share_post_dedup', 0.0)):.1%}</td>"
            f"<td>{escape(str(bool(diagnostics.get('provider_concentration_warning'))).lower())}</td>"
            f"<td class=\"num\">{float(diagnostics.get('top_ticker_share_raw', 0.0)):.1%}</td>"
            f"<td class=\"num\">{float(diagnostics.get('top_ticker_share_post_dedup', 0.0)):.1%}</td>"
            f"<td>{escape(str(bool(diagnostics.get('ticker_concentration_warning'))).lower())}</td>"
            "</tr>",
            "  </table>",
            "  <table>",
            "    <tr><th>Provider</th><th>Requests</th><th>Articles</th><th>Status</th><th>Detail</th></tr>",
            "    <tr>"
            "<td>GNews</td>"
            f"<td class=\"num\">{int(diagnostics.get('gnews_requests_attempted', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('gnews_articles_returned', 0))}</td>"
            f"<td>{escape(str(diagnostics.get('gnews_status_code') or diagnostics.get('gnews_skipped_reason') or 'no_result'))}</td>"
            f"<td>{escape(str(diagnostics.get('gnews_error_reason') or 'none'))}; "
            f"429 count: {int(diagnostics.get('gnews_rate_limited_count', 0))}; "
            f"retry after: {escape(str(diagnostics.get('gnews_retry_after_seconds') or 'not provided'))}</td>"
            "</tr>",
            "    <tr>"
            "<td>NYT context</td>"
            f"<td class=\"num\">{int(diagnostics.get('nyt_requests_attempted', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('nyt_articles_returned', 0))}</td>"
            f"<td>{escape(str(diagnostics.get('nyt_status_code') or diagnostics.get('nyt_role') or 'context_news_api'))}</td>"
            f"<td>{escape(str(diagnostics.get('nyt_error_reason') or 'none'))}; "
            f"zero-result queries: {int(diagnostics.get('nyt_zero_result_queries', 0))}</td>"
            "</tr>",
            "  </table>",
        ]
    )


def _top_mention_bullet(top_10: tuple[MostMentionedRow, ...]) -> str:
    if top_10:
        leader = top_10[0]
        return f"Top mention leader: {leader.ticker} with {leader.mentions} current report mention(s)."
    return "Top mention leader: no configured ticker had measurable current report volume."


def _event_bullet(report_input: DailyReportInput, *, direction: str) -> str:
    clusters = _top_event_clusters(report_input.event_clusters_by_ticker)
    scored = [cluster for cluster in clusters if cluster.weighted_cluster_sentiment is not None]
    if not scored:
        if direction == "negative":
            return "Biggest negative event: no clearly negative scored event cluster was available."
        return "High-attention event: no scored event clusters were available."
    if direction == "negative":
        cluster = min(scored, key=lambda item: (item.weighted_cluster_sentiment or 0.0, -item.article_count))
        if (cluster.weighted_cluster_sentiment or 0.0) >= 0:
            return "Biggest negative event: no clearly negative scored event cluster was available."
        return f"Biggest negative event: {cluster.ticker} - {cluster.title}."
    cluster = max(scored, key=lambda item: ((item.weighted_cluster_sentiment or 0.0), item.article_count))
    if (cluster.weighted_cluster_sentiment or 0.0) > 0:
        return f"Biggest positive event: {cluster.ticker} - {cluster.title}."
    high_attention = max(clusters, key=lambda item: (item.article_count, item.source_count), default=cluster)
    return f"High-attention event: {high_attention.ticker} - {high_attention.title}."


def _quality_caveat_bullet(report_input: DailyReportInput) -> str:
    source_summary = report_input.extraction_summary.source_quality_summary
    extraction = report_input.extraction_summary
    return (
        f"Source filters show {source_summary.visible_articles} visible article(s) and "
        f"{source_summary.excluded_articles} excluded article(s); full text extraction succeeded for "
        f"{extraction.successful_extractions} article(s), with snippet/title fallbacks still used."
    )


def _extraction_summary_html(summary: ExtractionSummary) -> str:
    basis_counts = {basis: int(summary.sentiment_basis_counts.get(basis, 0)) for basis in ("full_text", "snippet", "title")}
    return "\n".join(
        [
            "  <h2>Article Extraction Summary</h2>",
            "  <table>",
            "    <tr><th>Article Fetch Attempts</th><th>Publisher Article Fetches</th><th>Google Wrappers Skipped</th><th>Google Wrappers Resolved</th><th>Full Text Successes</th><th>Snippet Fallbacks</th><th>Title Fallbacks</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{summary.article_pages_fetched}</td>"
            f"<td class=\"num\">{summary.publisher_article_fetches}</td>"
            f"<td class=\"num\">{summary.google_news_wrappers_skipped}</td>"
            f"<td class=\"num\">{summary.google_news_wrappers_resolved}</td>"
            f"<td class=\"num\">{summary.successful_extractions}</td>"
            f"<td class=\"num\">{summary.snippet_fallbacks}</td>"
            f"<td class=\"num\">{summary.title_fallbacks}</td>"
            "</tr>",
            "  </table>",
            "  <table>",
            "    <tr><th>Full Text Basis</th><th>Snippet Basis</th><th>Title Basis</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{basis_counts['full_text']}</td>"
            f"<td class=\"num\">{basis_counts['snippet']}</td>"
            f"<td class=\"num\">{basis_counts['title']}</td>"
            "</tr>",
            "  </table>",
            "  <table>",
            "    <tr><th>Queue Size</th><th>Selected</th><th>Skipped</th><th>Success Rate</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{summary.extraction_queue_size}</td>"
            f"<td class=\"num\">{summary.extraction_selected_count}</td>"
            f"<td class=\"num\">{summary.extraction_skipped_count}</td>"
            f"<td class=\"num\">{summary.extraction_success_rate:.1%}</td>"
            "</tr>",
            "  </table>",
            "  <table>",
            "    <tr><th>Strong Full Text</th><th>Usable Full Text</th><th>Weak Text</th><th>Blocked or Shell</th><th>Unresolved Wrappers</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{int(summary.extraction_quality_grade_counts.get('strong_full_text', 0))}</td>"
            f"<td class=\"num\">{summary.usable_full_text_count}</td>"
            f"<td class=\"num\">{summary.weak_text_count}</td>"
            f"<td class=\"num\">{summary.blocked_or_shell_count}</td>"
            f"<td class=\"num\">{summary.google_wrappers_unresolved}</td>"
            "</tr>",
            "  </table>",
            _failure_reasons_html(summary.top_extraction_failure_reasons),
            _extraction_diagnostics_html(summary),
        ]
    )


def _source_quality_summary_html(summary: SourceQualitySummary) -> str:
    return "\n".join(
        [
            "  <h2>Source Quality Summary</h2>",
            "  <table>",
            "    <tr><th>Total Articles</th><th>Visible Articles</th><th>Excluded Articles</th><th>Tier 1</th><th>Tier 2</th><th>Tier 3 Visible</th><th>Tier 3 Hidden</th><th>Tier 4 Excluded</th><th>Unknown</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{summary.total_articles}</td>"
            f"<td class=\"num\">{summary.visible_articles}</td>"
            f"<td class=\"num\">{summary.excluded_articles}</td>"
            f"<td class=\"num\">{summary.tier_1_articles}</td>"
            f"<td class=\"num\">{summary.tier_2_articles}</td>"
            f"<td class=\"num\">{summary.tier_3_visible_articles}</td>"
            f"<td class=\"num\">{summary.tier_3_hidden_articles}</td>"
            f"<td class=\"num\">{summary.tier_4_excluded_articles}</td>"
            f"<td class=\"num\">{summary.unknown_articles}</td>"
            "</tr>",
            "  </table>",
            _excluded_sources_html(summary),
        ]
    )


def _excluded_sources_html(summary: SourceQualitySummary) -> str:
    parts = []
    if summary.excluded_sources:
        parts.append(f"Excluded articles by filter: {_source_list(summary.excluded_sources)}.")
    if summary.hidden_sources:
        parts.append(f"Hidden lower priority publishers: {_source_list(summary.hidden_sources)}.")
    visible_high_quality = tuple(
        source
        for source in getattr(summary, "visible_sources", ())
        if source not in set(summary.unclassified_sources)
    )
    if visible_high_quality:
        parts.append(f"Visible high quality publishers: {_source_list(visible_high_quality)}.")
    if summary.unclassified_sources:
        parts.append(f"Unclassified publishers shown: {_source_list(summary.unclassified_sources)}.")
    if not parts:
        parts.append("No source quality exclusions or lower-priority hides were applied.")
    return f"  <p class=\"muted\">{escape(' '.join(parts))}</p>"


def _source_list(sources: tuple[str, ...]) -> str:
    shown = ", ".join(sources[:8])
    if len(sources) > 8:
        return f"{shown} and {len(sources) - 8} more"
    return shown


def _extraction_diagnostics_html(summary: ExtractionSummary) -> str:
    diagnostics = summary.extractor_diagnostics
    return "\n".join(
        [
            "  <table>",
            "    <tr><th>Extraction Diagnostic</th><th>Value</th></tr>",
            f"    <tr><td>trafilatura_available</td><td>{escape(_availability(diagnostics.get('trafilatura_available')))}</td></tr>",
            f"    <tr><td>newspaper3k_available</td><td>{escape(_availability(diagnostics.get('newspaper3k_available')))}</td></tr>",
            f"    <tr><td>internal_parser_available</td><td>{escape(_availability(diagnostics.get('internal_parser_available')))}</td></tr>",
            f"    <tr><td>extraction_method_used</td><td>{escape(_method_counts(summary.extraction_method_counts))}</td></tr>",
            f"    <tr><td>extraction_failure_reason</td><td>{escape(summary.extraction_failure_reason or 'none')}</td></tr>",
            "  </table>",
        ]
    )


def _failure_reasons_html(reasons: Mapping[str, int]) -> str:
    if not reasons:
        return "  <p class=\"empty\">No extraction failure reasons recorded.</p>"
    rows = sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[:8]
    body = [
        "  <h3>Top Extraction Failure Reasons</h3>",
        "  <table>",
        "    <tr><th>Reason</th><th>Count</th></tr>",
    ]
    body.extend(
        f"    <tr><td>{escape(reason)}</td><td class=\"num\">{int(count)}</td></tr>"
        for reason, count in rows
    )
    body.append("  </table>")
    return "\n".join(body)


def _availability(value: bool | None) -> str:
    return "available" if value else "missing"


def _method_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{method}={int(count)}"
        for method, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    )


def _recency_bucket_html(
    title: str,
    description: str,
    rows: list[PortfolioSentimentRow],
    sentiment_field: str,
    count_field: str,
) -> str:
    ranked = sorted(
        rows,
        key=lambda row: (getattr(row, count_field), abs(getattr(row, sentiment_field)), row.source_diversity),
        reverse=True,
    )[:8]
    body = [
        f"  <h2>{escape(title)}</h2>",
        f"  <p>{escape(description)}</p>",
        "  <table>",
        "    <tr><th>Ticker</th><th>Sentiment</th><th>Articles</th><th>Source Diversity</th><th>Velocity</th></tr>",
    ]
    if ranked:
        body.extend(
            "    <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td class=\"num\">{getattr(row, sentiment_field):.4f}</td>"
            f"<td class=\"num\">{getattr(row, count_field)}</td>"
            f"<td class=\"num\">{row.source_diversity}</td>"
            f"<td>{escape(_display_velocity(row.mention_velocity))}</td>"
            "</tr>"
            for row in ranked
        )
    else:
        body.append("    <tr><td colspan=\"5\" class=\"empty\">No configured ticker has recency data in this bucket.</td></tr>")
    body.append("  </table>")
    return "\n".join(body)


def _sentiment_table_html(title: str, rows: tuple[PortfolioSentimentRow, ...]) -> str:
    body = [
        f"  <h2>{escape(title)}</h2>",
        "  <table>",
        "    <tr><th>Ticker</th><th>Company</th><th>Weighted</th><th>Today</th><th>1-3D</th><th>4-7D</th><th>8-30D</th><th>24H</th><th>3D</th><th>7D</th><th>30D</th><th>Velocity</th><th>Sources</th><th>Basis</th></tr>",
    ]
    body.extend(
        "    <tr>"
        f"<td>{escape(row.ticker)}</td>"
        f"<td>{escape(row.company_name)}</td>"
        f"<td class=\"num\">{row.weighted_sentiment_score:.4f}</td>"
        f"<td class=\"num\">{row.today_signal_sentiment:.4f}</td>"
        f"<td class=\"num\">{row.recent_pulse_sentiment:.4f}</td>"
        f"<td class=\"num\">{row.weekly_trend_sentiment:.4f}</td>"
        f"<td class=\"num\">{row.background_context_sentiment:.4f}</td>"
        f"<td class=\"num\">{row.article_count_24h}</td>"
        f"<td class=\"num\">{row.article_count_3d}</td>"
        f"<td class=\"num\">{row.article_count_7d}</td>"
        f"<td class=\"num\">{row.article_count_30d}</td>"
        f"<td>{escape(_display_velocity(row.mention_velocity))}</td>"
        f"<td class=\"num\">{row.source_diversity}</td>"
        f"<td>{escape(row.sentiment_basis)}</td>"
        "</tr>"
        for row in rows
    )
    body.append("  </table>")
    return "\n".join(body)


def _forecast_table_html(rows: tuple[WatchlistForecastRow, ...]) -> str:
    body = [
        "  <h2>Watchlist Next Close Direction</h2>",
        "  <p class=\"note\">Placeholder direction logic: direction is derived from current report sentiment. These rows are not real predictions.</p>",
        "  <table>",
        "    <tr><th>Ticker</th><th>Direction</th><th>Confidence</th><th>Driver</th></tr>",
    ]
    body.extend(
        "    <tr>"
        f"<td>{escape(row.ticker)}</td>"
        f"<td>{escape(row.next_close_direction)}</td>"
        f"<td class=\"num\">{row.confidence:.4f}</td>"
        f"<td>{escape(row.driver)}</td>"
        "</tr>"
        for row in rows
    )
    body.append("  </table>")
    return "\n".join(body)


def _mention_leaders_html(rows: tuple[MentionLeaderRow, ...]) -> str:
    body = [
        "  <h2>Top 7 Day Mention Leaders</h2>",
        "  <table>",
        "    <tr><th>Ticker</th><th>Mentions</th><th>Average Sentiment</th></tr>",
    ]
    if rows:
        body.extend(
            "    <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td class=\"num\">{row.mentions_7d}</td>"
            f"<td class=\"num\">{row.sentiment_avg:.4f}</td>"
            "</tr>"
            for row in rows
        )
    else:
        body.append("    <tr><td colspan=\"3\" class=\"empty\">No configured tickers were mentioned in current report data.</td></tr>")
    body.append("  </table>")
    return "\n".join(body)


def _top_10_html(rows: tuple[MostMentionedRow, ...]) -> str:
    body = [
        "  <h2>Top 10 Most Mentioned Tickers</h2>",
        "  <table>",
        "    <tr><th>Rank</th><th>Ticker</th><th>Mentions</th></tr>",
    ]
    if rows:
        body.extend(
            "    <tr>"
            f"<td class=\"num\">{row.rank}</td>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td class=\"num\">{row.mentions}</td>"
            "</tr>"
            for row in rows
        )
    else:
        body.append("    <tr><td colspan=\"3\" class=\"empty\">No configured tickers were mentioned in current report data.</td></tr>")
    body.append("  </table>")
    return "\n".join(body)


def _emerging_names_html(rows: tuple[EmergingNameRow, ...]) -> str:
    body = [
        "  <h2>Emerging Names Based On Mention Velocity</h2>",
        "  <table>",
        "    <tr><th>Ticker</th><th>Company</th><th>Mentions</th><th>Prior Mentions</th><th>Reason</th></tr>",
    ]
    if rows:
        body.extend(
            "    <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td>{escape(row.company_name)}</td>"
            f"<td class=\"num\">{row.mentions_7d}</td>"
            f"<td class=\"num\">{row.prior_mentions_30d}</td>"
            f"<td>{escape(row.reason)}</td>"
            "</tr>"
            for row in rows
        )
    else:
        body.append("    <tr><td colspan=\"5\" class=\"empty\">No emerging watchlist names were found in current report data.</td></tr>")
    body.append("  </table>")
    return "\n".join(body)


def _article_links_html(links_by_ticker: Mapping[str, tuple[ArticleLink, ...]]) -> str:
    body = ["  <h2>Read More By Ticker</h2>"]
    linked = False
    for ticker, links in sorted(links_by_ticker.items()):
        if not links:
            continue
        linked = True
        body.append(f"  <h3>{escape(ticker)}</h3>")
        body.append("  <ul>")
        visible_links = links[:10]
        for link in visible_links:
            source = f" ({escape(link.source)})" if link.source else ""
            body.append(f"    <li><a href=\"{escape(link.url, quote=True)}\">{escape(link.title)}</a>{source}</li>")
        if len(links) > len(visible_links):
            body.append(f"    <li class=\"empty\">+{len(links) - len(visible_links)} more links in JSON artifacts</li>")
        body.append("  </ul>")
    if not linked:
        body.append("  <p class=\"empty\">No article links matched configured tickers.</p>")
    return "\n".join(body)


def _event_clusters_html(clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]]) -> str:
    body = ["  <h2>Stories to Watch</h2>"]
    shown = False
    for ticker, clusters in sorted(clusters_by_ticker.items()):
        visible_clusters = clusters[:5]
        if not visible_clusters:
            continue
        shown = True
        body.append(f"  <h3>{escape(ticker)}</h3>")
        body.append("  <table>")
        body.append("    <tr><th>Story</th><th>Summary</th><th>Priority</th><th>Basis</th><th>Bucket</th><th>Publishers</th><th>Sources</th></tr>")
        for cluster in visible_clusters:
            body.append(
                "    <tr>"
                f"<td>{escape(cluster.title)}<br><a href=\"{escape(cluster.primary_link, quote=True)}\">Open story</a></td>"
                f"<td>{escape(cluster.cluster_summary or cluster.title)}</td>"
                f"<td>{escape(cluster.cluster_reading_priority)}</td>"
                f"<td>{escape(cluster.cluster_summary_basis)}</td>"
                f"<td>{escape(cluster.recency_bucket)}</td>"
                f"<td class=\"num\">{cluster.publisher_count}</td>"
                f"<td class=\"num\">{cluster.source_count}</td>"
                "</tr>"
            )
        body.append("  </table>")
    if not shown:
        body.append("  <p class=\"empty\">No event clusters matched configured tickers.</p>")
    return "\n".join(body)


def _ticker_summaries_html(title: str, rows: tuple[TickerDailySummary, ...]) -> str:
    body = [f"  <h2>{escape(title)}</h2>"]
    covered = [row for row in rows if row.read_first_story or row.read_next_story or row.background_story]
    if not covered:
        body.append("  <p class=\"empty\">No matched stories were available for configured names.</p>")
        return "\n".join(body)
    body.append("  <ul>")
    body.extend(f"    <li>{escape(row.ticker_daily_summary)}</li>" for row in covered)
    body.append("  </ul>")
    return "\n".join(body)


def _ranked_reads_html(
    reads_by_ticker: Mapping[str, tuple[RankedArticleRecommendation, ...]],
) -> str:
    body = ["  <h2>Ranked Reads By Ticker</h2>"]
    shown = False
    for ticker, reads in sorted(reads_by_ticker.items()):
        visible = [read for read in reads if read.reading_priority != "background_only"][:2]
        if not visible:
            continue
        shown = True
        body.append(f"  <h3>{escape(ticker)}</h3>")
        body.append("  <ul>")
        for read in visible:
            warning = f" {read.summary_warning}" if read.summary_warning else ""
            body.append(
                "    <li>"
                f"<strong>{escape(read.reading_priority)}:</strong> "
                f"<a href=\"{escape(read.url, quote=True)}\">{escape(read.title)}</a> "
                f"({escape(read.source)}, {escape(read.summary_basis)}) - "
                f"{escape(read.article_summary)}"
                f"<span class=\"muted\">{escape(warning)}</span>"
                "</li>"
            )
        body.append("  </ul>")
    if not shown:
        body.append("  <p class=\"empty\">No ranked reads were available for configured tickers.</p>")
    return "\n".join(body)


def _sentiment_movers_sentence(report_input: DailyReportInput) -> str:
    rows = list(report_input.portfolio_sentiment) + list(report_input.watchlist_sentiment)
    covered = [row for row in rows if row.article_count_30d]
    if not covered:
        return "No configured ticker had current report articles, so sentiment did not move in this dry run."
    strongest = max(covered, key=lambda row: abs(row.sentiment_30d))
    return (
        f"{strongest.ticker} had the largest current report sentiment signal at {strongest.sentiment_30d:.2f} "
        f"from {strongest.article_count_30d} article(s), using {strongest.sentiment_basis} sentiment basis."
    )


def _format_optional_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _news_volume_sentence(top_10: tuple[MostMentionedRow, ...]) -> str:
    if not top_10:
        return "No configured ticker had measurable current report news volume."
    leaders = ", ".join(f"{row.ticker} ({row.mentions})" for row in top_10[:3])
    return f"The highest current report news volume came from {leaders}."


def _attention_sentence(report_input: DailyReportInput) -> str:
    attention = [
        row.ticker
        for row in report_input.watchlist_forecasts
        if row.next_close_direction != "uncertain" or row.confidence >= 0.5
    ]
    if not attention and report_input.mention_leaders_7d:
        attention = [report_input.mention_leaders_7d[0].ticker]
    if not attention:
        return "No ticker deserves elevated attention from this limited report set."
    return f"Tickers deserving attention from current coverage: {', '.join(dict.fromkeys(attention[:5]))}."


def _quality_rank(label: str) -> int:
    ranks = {
        "tier_1_high_trust": 4,
        "tier_2_usable": 3,
        "tier_3_low_priority": 2,
        "tier_4_exclude_by_default": 1,
    }
    return ranks.get(label, 3)


def _display_velocity(value: str) -> str:
    return "history building" if value == "limited_history" else value
