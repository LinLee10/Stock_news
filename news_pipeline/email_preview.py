"""Local-only email preview artifacts for dry-run reports."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping

from .reporting import DailyReportContract
from .summaries import RankedArticleRecommendation, TickerDailySummary


@dataclass(frozen=True)
class EmailPreviewManifest:
    subject: str
    html_preview_path: str
    intended_attachments: tuple[str, ...]
    delivery_mode: str = "local_preview_only"


class PreviewEmailSender:
    """Write the report as an email-shaped local preview without network IO."""

    def write_preview(self, report: DailyReportContract) -> EmailPreviewManifest:
        output_dir = Path(report.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subject = f"Portfolio and Watchlist Market Briefing - {report.report_date}"
        attachments = tuple(report.csv_attachments)
        preview_path = output_dir / "email_preview.html"
        preview_path.write_text(
            _render_email_preview(report, subject, attachments),
            encoding="utf-8",
        )
        return EmailPreviewManifest(
            subject=subject,
            html_preview_path=str(preview_path),
            intended_attachments=attachments,
        )


def _render_email_preview(
    report: DailyReportContract,
    subject: str,
    attachments: tuple[str, ...],
) -> str:
    sections = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"  <title>{escape(subject)}</title>",
        "  <style>",
        "    body{font-family:Arial,sans-serif;margin:0;background:#f5f7fa;color:#17202a;line-height:1.45}",
        "    .wrap{max-width:960px;margin:0 auto;background:#fff;padding:24px}",
        "    h1,h2{color:#17202a}",
        "    table{border-collapse:collapse;width:100%;margin:12px 0 24px}",
        "    th,td{border:1px solid #d7dde5;padding:8px;text-align:left;font-size:14px}",
        "    th{background:#eef3f8}",
        "    .note{background:#fff8e6;border-left:4px solid #c98900;padding:12px;margin:12px 0 20px}",
        "    .summary{background:#edf7f1;border-left:4px solid #248a4b;padding:12px;margin:12px 0 20px}",
        "    .briefing{background:#f0f6ff;border-left:4px solid #2563eb;padding:12px;margin:12px 0 20px}",
        "    .briefing ul{margin:8px 0 0;padding-left:20px}",
        "    .num{text-align:right}",
        "    .empty{color:#64748b}",
        "    .muted{color:#64748b;font-size:13px}",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class=\"wrap\">",
        f"    <h1>{escape(subject)}</h1>",
        "    <p class=\"muted\">This briefing is not investment advice. Direction rows are not predictions.</p>",
        _top_briefing(report),
        _source_coverage_line(report),
        _ticker_summaries("Portfolio Summary", report.portfolio_summaries),
        _ticker_summaries("Watchlist Summary", report.watchlist_summaries),
        _event_clusters(report),
        _ranked_reads(report.ranked_reads_by_ticker),
        _sentiment_coverage(report),
        _recency_sections(report),
        _sentiment_table("Portfolio Recency Sentiment", report.portfolio_30d_sentiment_table),
        _sentiment_table("Watchlist Recency Sentiment", report.watchlist_sentiment_table),
        _forecast_table(report),
        _mention_leaders_table(report),
        _top_mentions_table(report),
        _emerging_names_table(report),
        _article_links(report),
        "    <h2>Source and Extraction Diagnostics</h2>",
        _report_metadata(report),
        _backend_pool_summary(report),
        _source_acquisition_summary(report),
        _source_quality_summary(report),
        _extraction_summary(report),
        _attachment_manifest(attachments),
        "  </div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(sections)


def _report_metadata(report: DailyReportContract) -> str:
    body = [
        "    <table>",
        "      <tr><th>Report Diagnostic</th><th>Value</th></tr>",
        f"      <tr><td>Run date</td><td>{escape(report.report_date)}</td></tr>",
        f"      <tr><td>Data source</td><td>{escape(report.data_source_label)}</td></tr>",
        "      <tr><td>Delivery mode</td><td><strong>Preview only:</strong> no live email provider was contacted.</td></tr>",
        "      <tr><td>SMTP send status</td><td>not sent</td></tr>",
        f"      <tr><td>Paid API status</td><td>{escape(str(report.source_coverage_diagnostics.get('paid_api_status', 'disabled')))}</td></tr>",
        f"      <tr><td>Report summary</td><td>{escape(report.daily_summary)}</td></tr>",
    ]
    if report.report_warnings:
        body.append(
            f"      <tr><td>Warnings</td><td>{escape(' '.join(report.report_warnings))}</td></tr>"
        )
    body.extend(
        [
            "    </table>",
            "    <p class=\"muted\">Sentiment is deterministic placeholder logic until a stronger model is wired in. "
            "Summaries use extracted full text when available, otherwise snippets or titles.</p>",
        ]
    )
    return "\n".join(body)


def _top_briefing(report: DailyReportContract) -> str:
    bullets = [
        _top_mention_bullet(report),
        _event_bullet(report, direction="negative"),
        _event_bullet(report, direction="positive"),
        _quality_caveat_bullet(report),
        "Watchlist direction rows are placeholder direction from current report sentiment, not predictions.",
    ]
    return "\n".join(
        [
            "    <section class=\"briefing\">",
            "      <h2>Daily Briefing</h2>",
            "      <ul>",
            *[f"        <li>{escape(bullet)}</li>" for bullet in bullets if bullet][:5],
            "      </ul>",
            "    </section>",
        ]
    )


def _source_coverage_line(report: DailyReportContract) -> str:
    diagnostics = report.source_coverage_diagnostics
    paid = diagnostics.get("paid_api_count", 0)
    paid_label = str(paid) if paid else "disabled"
    return (
        "    <p class=\"muted\"><strong>Source Coverage:</strong> "
        f"Official filings: {int(diagnostics.get('official_source_count', 0))} &middot; "
        f"Press releases: {int(diagnostics.get('press_release_wire_count', 0))} &middot; "
        f"Direct publishers: {int(diagnostics.get('direct_publisher_count', 0))} &middot; "
        f"Google backstop: {int(diagnostics.get('google_news_backstop_count', 0))} &middot; "
        f"Paid APIs: {escape(paid_label)}</p>"
    )


def _top_mention_bullet(report: DailyReportContract) -> str:
    if report.top_10_most_mentioned_table:
        leader = report.top_10_most_mentioned_table[0]
        return f"Top mention leader: {leader.ticker} with {leader.mentions} current report mention(s)."
    return "Top mention leader: no configured ticker had measurable current report volume."


def _event_bullet(report: DailyReportContract, *, direction: str) -> str:
    scored = [cluster for cluster in report.top_event_clusters if cluster.weighted_cluster_sentiment is not None]
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
    high_attention = max(report.top_event_clusters, key=lambda item: (item.article_count, item.source_count), default=cluster)
    return f"High-attention event: {high_attention.ticker} - {high_attention.title}."


def _quality_caveat_bullet(report: DailyReportContract) -> str:
    source_summary = report.extraction_summary.source_quality_summary
    extraction = report.extraction_summary
    return (
        f"Source filters show {source_summary.visible_articles} visible article(s) and "
        f"{source_summary.excluded_articles} excluded article(s); full text extraction succeeded for "
        f"{extraction.successful_extractions} article(s), with snippet/title fallbacks still used."
    )


def _sentiment_coverage(report: DailyReportContract) -> str:
    rows = [row for row in report.ticker_sentiment_coverage.values() if row.article_count_scored]
    body = [
        "    <h2>Sentiment Coverage Summary</h2>",
        "    <table>",
        "      <tr><th>Ticker</th><th>Coverage</th><th>Weighted Sentiment</th><th>Scored</th><th>Full Text</th><th>Snippets</th></tr>",
    ]
    if rows:
        body.extend(
            "      <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td>{escape(row.sentiment_coverage_grade)}</td>"
            f"<td class=\"num\">{row.weighted_sentiment:.4f}</td>"
            f"<td class=\"num\">{row.article_count_scored}</td>"
            f"<td class=\"num\">{row.full_text_scored_count}</td>"
            f"<td class=\"num\">{row.snippet_scored_count}</td>"
            "</tr>"
            for row in sorted(rows, key=lambda item: item.ticker)
        )
    else:
        body.append("      <tr><td colspan=\"6\" class=\"empty\">No weighted sentiment coverage was available.</td></tr>")
    body.append("    </table>")
    return "\n".join(body)


def _backend_pool_summary(report: DailyReportContract) -> str:
    backend = report.backend_article_pool_summary
    email = report.email_display_summary
    return "\n".join(
        [
            "    <h2>Backend and Email Pool Summary</h2>",
            "    <table>",
            "      <tr><th>Backend Candidates</th><th>Backend Visible</th><th>Backend Scored</th><th>Backend Full Text</th><th>Email Stories</th><th>Email Ranked Reads</th></tr>",
            "      <tr>"
            f"<td class=\"num\">{backend.backend_candidate_articles}</td>"
            f"<td class=\"num\">{backend.backend_visible_articles}</td>"
            f"<td class=\"num\">{backend.backend_scored_articles}</td>"
            f"<td class=\"num\">{backend.backend_extracted_articles}</td>"
            f"<td class=\"num\">{email.email_visible_stories}</td>"
            f"<td class=\"num\">{email.email_visible_ranked_reads}</td>"
            "</tr>",
            "    </table>",
        ]
    )


def _source_acquisition_summary(report: DailyReportContract) -> str:
    diagnostics = report.source_coverage_diagnostics
    return "\n".join(
        [
            "    <h2>Source Acquisition Summary</h2>",
            "    <table>",
            "      <tr><th>Official</th><th>Company IR</th><th>Press Wires</th><th>Direct Publishers</th><th>Google Backstop</th><th>Google Share</th></tr>",
            "      <tr>"
            f"<td class=\"num\">{int(diagnostics.get('official_source_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('company_ir_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('press_release_wire_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('direct_publisher_count', 0))}</td>"
            f"<td class=\"num\">{int(diagnostics.get('google_news_backstop_count', 0))}</td>"
            f"<td class=\"num\">{float(diagnostics.get('google_news_share', 0.0)):.1%}</td>"
            "</tr>",
            "    </table>",
        ]
    )


def _sentiment_table(title: str, rows: tuple[object, ...]) -> str:
    body = [
        f"    <h2>{escape(title)}</h2>",
        "    <table>",
        "      <tr><th>Ticker</th><th>Company</th><th>Weighted</th><th>Today</th><th>1-3D</th><th>4-7D</th><th>8-30D</th><th>24H</th><th>3D</th><th>7D</th><th>30D</th><th>Velocity</th><th>Sources</th><th>Basis</th></tr>",
    ]
    body.extend(
        "      <tr>"
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
    body.append("    </table>")
    return "\n".join(body)


def _extraction_summary(report: DailyReportContract) -> str:
    summary = report.extraction_summary
    basis_counts = {basis: int(summary.sentiment_basis_counts.get(basis, 0)) for basis in ("full_text", "snippet", "title")}
    total_basis = sum(basis_counts.values())
    full_text_coverage = basis_counts["full_text"] / total_basis if total_basis else 0.0
    parts = [
        "    <h2>Article Extraction Summary</h2>",
        "    <table>",
        "      <tr><th>Selected</th><th>Full Text Accepted</th><th>Usable Full Text</th><th>Coverage Rate</th></tr>",
        "      <tr>"
        f"<td class=\"num\">{summary.extraction_selected_count}</td>"
        f"<td class=\"num\">{summary.full_text_accepted_count}</td>"
        f"<td class=\"num\">{summary.usable_full_text_count}</td>"
        f"<td class=\"num\">{full_text_coverage:.1%}</td>"
        "</tr>",
        "    </table>",
        "    <table>",
        "      <tr><th>Full Text Basis</th><th>Snippet Basis</th><th>Title Basis</th></tr>",
        "      <tr>"
        f"<td class=\"num\">{basis_counts['full_text']}</td>"
        f"<td class=\"num\">{basis_counts['snippet']}</td>"
        f"<td class=\"num\">{basis_counts['title']}</td>"
        "</tr>",
        "    </table>",
    ]
    if total_basis and full_text_coverage < 0.4:
        parts.append(
            "    <p class=\"muted\">Full text coverage is low; sentiment still relies heavily on snippets or titles.</p>"
        )
    return "\n".join(parts)


def _source_quality_summary(report: DailyReportContract) -> str:
    summary = report.extraction_summary.source_quality_summary
    return "\n".join(
        [
            "    <h2>Source Quality Summary</h2>",
            "    <table>",
            "      <tr><th>Total Articles</th><th>Visible Articles</th><th>Excluded Articles</th><th>Tier 1</th><th>Tier 2</th><th>Tier 3 Visible</th><th>Tier 3 Hidden</th><th>Tier 4 Excluded</th><th>Unknown</th></tr>",
            "      <tr>"
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
            "    </table>",
            _excluded_sources(summary),
        ]
    )


def _excluded_sources(summary) -> str:
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
    return f"    <p class=\"muted\">{escape(' '.join(parts))}</p>"


def _source_list(sources: tuple[str, ...]) -> str:
    shown = ", ".join(sources[:8])
    if len(sources) > 8:
        return f"{shown} and {len(sources) - 8} more"
    return shown


def _display_velocity(value: str) -> str:
    return "history building" if value == "limited_history" else value


def _extraction_diagnostics(summary) -> str:
    diagnostics = summary.extractor_diagnostics
    return "\n".join(
        [
            "    <table>",
            "      <tr><th>Extraction Diagnostic</th><th>Value</th></tr>",
            f"      <tr><td>trafilatura_available</td><td>{escape(_availability(diagnostics.get('trafilatura_available')))}</td></tr>",
            f"      <tr><td>newspaper3k_available</td><td>{escape(_availability(diagnostics.get('newspaper3k_available')))}</td></tr>",
            f"      <tr><td>internal_parser_available</td><td>{escape(_availability(diagnostics.get('internal_parser_available')))}</td></tr>",
            f"      <tr><td>extraction_method_used</td><td>{escape(_method_counts(summary.extraction_method_counts))}</td></tr>",
            f"      <tr><td>extraction_failure_reason</td><td>{escape(summary.extraction_failure_reason or 'none')}</td></tr>",
            "    </table>",
        ]
    )


def _failure_reasons(reasons) -> str:
    if not reasons:
        return "    <p class=\"empty\">No extraction failure reasons recorded.</p>"
    rows = sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[:8]
    body = [
        "    <h3>Top Extraction Failure Reasons</h3>",
        "    <table>",
        "      <tr><th>Reason</th><th>Count</th></tr>",
    ]
    body.extend(
        f"      <tr><td>{escape(reason)}</td><td class=\"num\">{int(count)}</td></tr>"
        for reason, count in rows
    )
    body.append("    </table>")
    return "\n".join(body)


def _availability(value) -> str:
    return "available" if value else "missing"


def _method_counts(counts) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{method}={int(count)}"
        for method, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    )


def _recency_sections(report: DailyReportContract) -> str:
    rows = list(report.portfolio_30d_sentiment_table) + list(report.watchlist_sentiment_table)
    return "\n".join(
        [
            _recency_bucket("Today's Signal", "0-24 hours before run date. Weight: 1.0.", rows, "today_signal_sentiment", "article_count_24h"),
            _recency_bucket("Recent Pulse", "1-3 days before run date. Weight: 0.7.", rows, "recent_pulse_sentiment", "article_count_3d"),
            _recency_bucket("Weekly Trend", "4-7 days before run date. Weight: 0.4.", rows, "weekly_trend_sentiment", "article_count_7d"),
            _recency_bucket("Background Context", "8-30 days before run date. Weight: 0.15.", rows, "background_context_sentiment", "article_count_30d"),
        ]
    )


def _recency_bucket(title, description, rows, sentiment_field, count_field):
    ranked = sorted(
        rows,
        key=lambda row: (getattr(row, count_field), abs(getattr(row, sentiment_field)), row.source_diversity),
        reverse=True,
    )[:8]
    body = [
        f"    <h2>{escape(title)}</h2>",
        f"    <p>{escape(description)}</p>",
        "    <table>",
        "      <tr><th>Ticker</th><th>Sentiment</th><th>Articles</th><th>Source Diversity</th><th>Velocity</th></tr>",
    ]
    body.extend(
        "      <tr>"
        f"<td>{escape(row.ticker)}</td>"
        f"<td class=\"num\">{getattr(row, sentiment_field):.4f}</td>"
        f"<td class=\"num\">{getattr(row, count_field)}</td>"
        f"<td class=\"num\">{row.source_diversity}</td>"
        f"<td>{escape(_display_velocity(row.mention_velocity))}</td>"
        "</tr>"
        for row in ranked
    )
    body.append("    </table>")
    return "\n".join(body)


def _forecast_table(report: DailyReportContract) -> str:
    body = [
        "    <h2>Watchlist Next Close Direction</h2>",
        "    <p class=\"note\">Placeholder direction logic: direction is derived from current report sentiment. These rows are not real predictions.</p>",
        "    <table>",
        "      <tr><th>Ticker</th><th>Direction</th><th>Confidence</th><th>Driver</th></tr>",
    ]
    body.extend(
        "      <tr>"
        f"<td>{escape(row.ticker)}</td>"
        f"<td>{escape(row.next_close_direction)}</td>"
        f"<td class=\"num\">{row.confidence:.4f}</td>"
        f"<td>{escape(row.driver)}</td>"
        "</tr>"
        for row in report.watchlist_next_close_table
    )
    body.append("    </table>")
    return "\n".join(body)


def _mention_leaders_table(report: DailyReportContract) -> str:
    body = [
        "    <h2>Top 7 Day Mention Leaders</h2>",
        "    <table>",
        "      <tr><th>Ticker</th><th>Mentions</th><th>Average Sentiment</th></tr>",
    ]
    if report.mention_leaders_7d_table:
        body.extend(
            "      <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td class=\"num\">{row.mentions_7d}</td>"
            f"<td class=\"num\">{row.sentiment_avg:.4f}</td>"
            "</tr>"
            for row in report.mention_leaders_7d_table
        )
    else:
        body.append("      <tr><td colspan=\"3\" class=\"empty\">No configured tickers were mentioned in current report data.</td></tr>")
    body.append("    </table>")
    return "\n".join(body)


def _top_mentions_table(report: DailyReportContract) -> str:
    body = [
        "    <h2>Top 10 Most Mentioned Tickers</h2>",
        "    <table>",
        "      <tr><th>Rank</th><th>Ticker</th><th>Mentions</th></tr>",
    ]
    if report.top_10_most_mentioned_table:
        body.extend(
            "      <tr>"
            f"<td class=\"num\">{row.rank}</td>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td class=\"num\">{row.mentions}</td>"
            "</tr>"
            for row in report.top_10_most_mentioned_table
        )
    else:
        body.append("      <tr><td colspan=\"3\" class=\"empty\">No configured tickers were mentioned in current report data.</td></tr>")
    body.append("    </table>")
    return "\n".join(body)


def _emerging_names_table(report: DailyReportContract) -> str:
    body = [
        "    <h2>Emerging Names Based On Mention Velocity</h2>",
        "    <table>",
        "      <tr><th>Ticker</th><th>Company</th><th>Mentions</th><th>Prior Mentions</th><th>Reason</th></tr>",
    ]
    if report.emerging_names_table:
        body.extend(
            "      <tr>"
            f"<td>{escape(row.ticker)}</td>"
            f"<td>{escape(row.company_name)}</td>"
            f"<td class=\"num\">{row.mentions_7d}</td>"
            f"<td class=\"num\">{row.prior_mentions_30d}</td>"
            f"<td>{escape(row.reason)}</td>"
            "</tr>"
            for row in report.emerging_names_table
        )
    else:
        body.append("      <tr><td colspan=\"5\" class=\"empty\">No emerging watchlist names were found in current report data.</td></tr>")
    body.append("    </table>")
    return "\n".join(body)


def _article_links(report: DailyReportContract) -> str:
    body = ["    <h2>Read More By Ticker</h2>"]
    linked = False
    for ticker, links in sorted(report.supporting_article_links.items()):
        if not links:
            continue
        linked = True
        body.append(f"    <h3>{escape(ticker)}</h3>")
        body.append("    <ul>")
        visible_links = links[:10]
        for link in visible_links:
            source = f" ({escape(link.source)})" if link.source else ""
            body.append(f"      <li><a href=\"{escape(link.url, quote=True)}\">{escape(link.title)}</a>{source}</li>")
        if len(links) > len(visible_links):
            body.append(f"      <li class=\"empty\">+{len(links) - len(visible_links)} more links in JSON artifacts</li>")
        body.append("    </ul>")
    if not linked:
        body.append("    <p class=\"empty\">No article links matched configured tickers.</p>")
    return "\n".join(body)


def _event_clusters(report: DailyReportContract) -> str:
    body = ["    <h2>Stories to Watch</h2>"]
    shown = False
    for ticker, clusters in sorted(report.event_clusters_by_ticker.items()):
        visible_clusters = sorted(clusters, key=lambda cluster: (-cluster.ranking_score, cluster.title))[:5]
        if not visible_clusters:
            continue
        shown = True
        body.append(f"    <h3>{escape(ticker)}</h3>")
        body.append("    <table>")
        body.append("      <tr><th>Story</th><th>Summary</th><th>Priority</th><th>Basis</th><th>Bucket</th><th>Publishers</th><th>Sources</th></tr>")
        for cluster in visible_clusters:
            body.append(
                "      <tr>"
                f"<td>{escape(cluster.title)}<br><a href=\"{escape(cluster.primary_link, quote=True)}\">Open story</a></td>"
                f"<td>{escape(cluster.cluster_summary or cluster.title)}</td>"
                f"<td>{escape(cluster.cluster_reading_priority)}</td>"
                f"<td>{escape(cluster.cluster_summary_basis)}</td>"
                f"<td>{escape(cluster.recency_bucket)}</td>"
                f"<td class=\"num\">{cluster.publisher_count}</td>"
                f"<td class=\"num\">{cluster.source_count}</td>"
                "</tr>"
            )
        body.append("    </table>")
    if not shown:
        body.append("    <p class=\"empty\">No event clusters matched configured tickers.</p>")
    return "\n".join(body)


def _ticker_summaries(title: str, rows: tuple[TickerDailySummary, ...]) -> str:
    body = [f"    <h2>{escape(title)}</h2>"]
    covered = [row for row in rows if row.read_first_story or row.read_next_story or row.background_story]
    if not covered:
        body.append("    <p class=\"empty\">No matched stories were available for configured names.</p>")
        return "\n".join(body)
    body.append("    <ul>")
    body.extend(f"      <li>{escape(row.ticker_daily_summary)}</li>" for row in covered)
    body.append("    </ul>")
    return "\n".join(body)


def _ranked_reads(
    reads_by_ticker: Mapping[str, tuple[RankedArticleRecommendation, ...]],
) -> str:
    body = ["    <h2>Ranked Reads By Ticker</h2>"]
    shown = False
    for ticker, reads in sorted(reads_by_ticker.items()):
        visible = [read for read in reads if read.reading_priority != "background_only"][:2]
        if not visible:
            continue
        shown = True
        body.append(f"    <h3>{escape(ticker)}</h3>")
        body.append("    <ul>")
        for read in visible:
            warning = f" {read.summary_warning}" if read.summary_warning else ""
            body.append(
                "      <li>"
                f"<strong>{escape(read.reading_priority)}:</strong> "
                f"<a href=\"{escape(read.url, quote=True)}\">{escape(read.title)}</a> "
                f"({escape(read.source)}, {escape(read.summary_basis)}) - "
                f"{escape(read.article_summary)}"
                f"<span class=\"muted\">{escape(warning)}</span>"
                "</li>"
            )
        body.append("    </ul>")
    if not shown:
        body.append("    <p class=\"empty\">No ranked reads were available for configured tickers.</p>")
    return "\n".join(body)


def _format_optional_score(value):
    return "" if value is None else f"{value:.4f}"


def _attachment_manifest(attachments: tuple[str, ...]) -> str:
    body = [
        "    <h2>Intended Attachments</h2>",
        "    <p>These files would be attached.</p>",
        "    <ul>",
    ]
    if attachments:
        body.extend(f"      <li>{escape(Path(path).name)}</li>" for path in attachments)
    else:
        body.append("      <li class=\"empty\">No attachments were generated.</li>")
    body.append("    </ul>")
    return "\n".join(body)
