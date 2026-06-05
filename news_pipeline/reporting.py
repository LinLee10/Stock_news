"""Report data contract and local artifact writers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Iterable, Mapping

from .charts import write_placeholder_chart
from .models import RunResult
from .source_quality import SourceQualitySummary


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
    extraction_summary: ExtractionSummary = field(default_factory=ExtractionSummary)
    data_source_label: str = "local RSS fixtures"


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
    extraction_summary: ExtractionSummary
    csv_attachments: tuple[str, ...]
    chart_attachments: tuple[str, ...]
    html_preview_report: str
    daily_summary: str
    data_source_label: str


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
        extraction_summary=report_input.extraction_summary,
        csv_attachments=csv_attachments,
        chart_attachments=chart_attachments,
        html_preview_report=html_preview_report,
        daily_summary=_plain_english_summary(report_input, top_10),
        data_source_label=report_input.data_source_label,
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
        f"  <title>News Pipeline Dry Run - {escape(report_input.report_date)}</title>",
        "  <style>",
        "    body{font-family:Arial,sans-serif;margin:24px;line-height:1.45;color:#17202a;background:#fff}",
        "    h1,h2{color:#17202a}",
        "    table{border-collapse:collapse;width:100%;margin:12px 0 24px}",
        "    th,td{border:1px solid #d7dde5;padding:8px;text-align:left;font-size:14px}",
        "    th{background:#eef3f8}",
        "    .note{background:#fff8e6;border-left:4px solid #c98900;padding:12px;margin:12px 0 20px}",
        "    .summary{background:#edf7f1;border-left:4px solid #248a4b;padding:12px;margin:12px 0 20px}",
        "    .empty{color:#64748b}",
        "    .muted{color:#64748b;font-size:13px}",
        "    .num{text-align:right}",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>News Pipeline Dry Run - {escape(report_input.report_date)}</h1>",
        f"  <div class=\"note\"><strong>Data source:</strong> {escape(report_input.data_source_label)}. No paid APIs were called and no email was sent.</div>",
        "  <div class=\"note\"><strong>Model status:</strong> Sentiment is deterministic placeholder logic, and watchlist direction rows are not real predictions. This report is not investment advice.</div>",
        f"  <div class=\"summary\">{escape(_plain_english_summary(report_input, top_10))}</div>",
        "  <h2>Plain English Readout</h2>",
        "  <ul>",
        f"    <li>{escape(_sentiment_movers_sentence(report_input))}</li>",
        f"    <li>{escape(_news_volume_sentence(top_10))}</li>",
        f"    <li>{escape(_attention_sentence(report_input))}</li>",
        "    <li>All sentiment basis labels are shown as full_text, snippet, title, or no_articles.</li>",
        "  </ul>",
        _source_quality_summary_html(report_input.extraction_summary.source_quality_summary),
        _extraction_summary_html(report_input.extraction_summary),
        _recency_sections_html(report_input),
        _sentiment_table_html("Portfolio Recency Sentiment", report_input.portfolio_sentiment),
        _sentiment_table_html("Watchlist Recency Sentiment", report_input.watchlist_sentiment),
        _forecast_table_html(report_input.watchlist_forecasts),
        _mention_leaders_html(report_input.mention_leaders_7d),
        _top_10_html(top_10),
        _emerging_names_html(report_input.emerging_names),
        _event_clusters_html(report_input.event_clusters_by_ticker),
        _article_links_html(report_input.article_links_by_ticker),
        "</body>",
        "</html>",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")
    return str(path)


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
            _failure_reasons_html(summary.top_extraction_failure_reasons),
            _extraction_diagnostics_html(summary),
        ]
    )


def _source_quality_summary_html(summary: SourceQualitySummary) -> str:
    tier_counts = summary.visible_tier_counts or summary.tier_counts
    return "\n".join(
        [
            "  <h2>Source Quality Summary</h2>",
            "  <table>",
            "    <tr><th>Total Articles</th><th>Visible Articles</th><th>Excluded Articles</th><th>Tier 1</th><th>Tier 2</th><th>Tier 3 Visible</th><th>Tier 4 Excluded</th></tr>",
            "    <tr>"
            f"<td class=\"num\">{summary.total_articles}</td>"
            f"<td class=\"num\">{summary.visible_articles}</td>"
            f"<td class=\"num\">{summary.excluded_articles}</td>"
            f"<td class=\"num\">{int(tier_counts.get('tier_1_high_trust', 0))}</td>"
            f"<td class=\"num\">{int(tier_counts.get('tier_2_usable', 0))}</td>"
            f"<td class=\"num\">{summary.low_priority_visible_articles}</td>"
            f"<td class=\"num\">{int(summary.excluded_tier_counts.get('tier_4_exclude_by_default', 0))}</td>"
            "</tr>",
            "  </table>",
            _excluded_sources_html(summary),
        ]
    )


def _excluded_sources_html(summary: SourceQualitySummary) -> str:
    if not summary.excluded_sources:
        return "  <p class=\"muted\">No low-quality sources were excluded.</p>"
    sources = ", ".join(summary.excluded_sources[:8])
    suffix = f" and {len(summary.excluded_sources) - 8} more" if len(summary.excluded_sources) > 8 else ""
    return f"  <p class=\"muted\">Excluded or hidden low-quality sources: {escape(sources + suffix)}.</p>"


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
            f"<td>{escape(row.mention_velocity)}</td>"
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
        f"<td>{escape(row.mention_velocity)}</td>"
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
    body = ["  <h2>Article Links Grouped By Ticker And Event Cluster</h2>"]
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
    body = ["  <h2>Top Event Clusters By Recency And Source Diversity</h2>"]
    shown = False
    for ticker, clusters in sorted(clusters_by_ticker.items()):
        visible_clusters = clusters[:5]
        if not visible_clusters:
            continue
        shown = True
        body.append(f"  <h3>{escape(ticker)}</h3>")
        body.append("  <table>")
        body.append("    <tr><th>Event</th><th>Bucket</th><th>Sentiment</th><th>Extraction Basis</th><th>First Seen</th><th>Latest Seen</th><th>Articles</th><th>Publishers</th><th>Sources</th><th>Visible Links</th></tr>")
        for cluster in visible_clusters:
            links = sorted(cluster.supporting_links, key=lambda link: (link.source_quality_label, link.title, link.url))[:3]
            links_html = "<br>".join(
                f"<a href=\"{escape(link.url, quote=True)}\">{escape(link.source or link.title)}</a>"
                for link in links
            )
            if len(cluster.supporting_links) > len(links):
                links_html += f"<br><span class=\"empty\">+{len(cluster.supporting_links) - len(links)} more in JSON artifacts</span>"
            body.append(
                "    <tr>"
                f"<td><a href=\"{escape(cluster.primary_link, quote=True)}\">{escape(cluster.title)}</a></td>"
                f"<td>{escape(cluster.recency_bucket)}</td>"
                f"<td class=\"num\">{_format_optional_score(cluster.weighted_cluster_sentiment)}</td>"
                f"<td>{escape(cluster.extraction_basis)}</td>"
                f"<td>{escape(cluster.first_seen_at or '')}</td>"
                f"<td>{escape(cluster.latest_seen_at or '')}</td>"
                f"<td class=\"num\">{cluster.article_count}</td>"
                f"<td class=\"num\">{cluster.publisher_count}</td>"
                f"<td class=\"num\">{cluster.source_count}</td>"
                f"<td>{links_html}</td>"
                "</tr>"
            )
        body.append("  </table>")
    if not shown:
        body.append("  <p class=\"empty\">No event clusters matched configured tickers.</p>")
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
