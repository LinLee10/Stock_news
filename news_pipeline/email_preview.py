"""Local-only email preview artifacts for dry-run reports."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from .reporting import DailyReportContract


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
        subject = f"Daily Stock News Sentiment Report - {report.report_date}"
        attachments = tuple(report.csv_attachments + report.chart_attachments)
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
        "    .num{text-align:right}",
        "    .empty{color:#64748b}",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class=\"wrap\">",
        f"    <h1>{escape(subject)}</h1>",
        "    <div class=\"note\"><strong>Preview only:</strong> This file is a local email preview. No SMTP, Gmail, Resend, or other live email provider was contacted.</div>",
        "    <div class=\"note\"><strong>Data source:</strong> Local RSS fixture files by default, plus free live RSS only when explicitly enabled. Watchlist next-close directions use placeholder dry-run logic.</div>",
        "    <div class=\"note\"><strong>Model status:</strong> Sentiment is deterministic placeholder logic until a stronger model is wired in. This preview is not investment advice.</div>",
        f"    <div class=\"summary\">{escape(report.daily_summary)}</div>",
        _extraction_summary(report),
        _recency_sections(report),
        _sentiment_table("Portfolio Recency Sentiment", report.portfolio_30d_sentiment_table),
        _sentiment_table("Watchlist Recency Sentiment", report.watchlist_sentiment_table),
        _forecast_table(report),
        _mention_leaders_table(report),
        _top_mentions_table(report),
        _emerging_names_table(report),
        _event_clusters(report),
        _article_links(report),
        _attachment_manifest(attachments),
        "  </div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(sections)


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
        f"<td>{escape(row.mention_velocity)}</td>"
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
    return "\n".join(
        [
            "    <h2>Article Extraction Summary</h2>",
            "    <table>",
            "      <tr><th>Article Fetch Attempts</th><th>Publisher Article Fetches</th><th>Google Wrappers Skipped</th><th>Google Wrappers Resolved</th><th>Full Text Successes</th><th>Snippet Fallbacks</th><th>Title Fallbacks</th></tr>",
            "      <tr>"
            f"<td class=\"num\">{summary.article_pages_fetched}</td>"
            f"<td class=\"num\">{summary.publisher_article_fetches}</td>"
            f"<td class=\"num\">{summary.google_news_wrappers_skipped}</td>"
            f"<td class=\"num\">{summary.google_news_wrappers_resolved}</td>"
            f"<td class=\"num\">{summary.successful_extractions}</td>"
            f"<td class=\"num\">{summary.snippet_fallbacks}</td>"
            f"<td class=\"num\">{summary.title_fallbacks}</td>"
            "</tr>",
            "    </table>",
            "    <table>",
            "      <tr><th>Full Text Basis</th><th>Snippet Basis</th><th>Title Basis</th><th>Trafilatura</th><th>Newspaper3k</th><th>Internal Parser</th></tr>",
            "      <tr>"
            f"<td class=\"num\">{basis_counts['full_text']}</td>"
            f"<td class=\"num\">{basis_counts['snippet']}</td>"
            f"<td class=\"num\">{basis_counts['title']}</td>"
            f"<td>{escape(_availability(summary.extractor_diagnostics.get('trafilatura_available')))}</td>"
            f"<td>{escape(_availability(summary.extractor_diagnostics.get('newspaper3k_available')))}</td>"
            f"<td>{escape(_availability(summary.extractor_diagnostics.get('internal_parser_available')))}</td>"
            "</tr>",
            "    </table>",
            _failure_reasons(summary.top_extraction_failure_reasons),
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
        f"<td>{escape(row.mention_velocity)}</td>"
        "</tr>"
        for row in ranked
    )
    body.append("    </table>")
    return "\n".join(body)


def _forecast_table(report: DailyReportContract) -> str:
    body = [
        "    <h2>Watchlist Next Close Direction</h2>",
        "    <p class=\"note\">Placeholder forecast logic: direction is derived from fixture sentiment only. This is not a live model prediction.</p>",
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
        body.append("      <tr><td colspan=\"3\" class=\"empty\">No configured tickers were mentioned in fixture data.</td></tr>")
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
        body.append("      <tr><td colspan=\"3\" class=\"empty\">No configured tickers were mentioned in fixture data.</td></tr>")
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
        body.append("      <tr><td colspan=\"5\" class=\"empty\">No emerging watchlist names were found in fixture data.</td></tr>")
    body.append("    </table>")
    return "\n".join(body)


def _article_links(report: DailyReportContract) -> str:
    body = ["    <h2>Article Links Grouped By Ticker And Event Cluster</h2>"]
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
        body.append("    <p class=\"empty\">No fixture article links matched configured tickers.</p>")
    return "\n".join(body)


def _event_clusters(report: DailyReportContract) -> str:
    body = ["    <h2>Top Event Clusters By Recency And Source Diversity</h2>"]
    shown = False
    for ticker, clusters in sorted(report.event_clusters_by_ticker.items()):
        visible_clusters = clusters[:5]
        if not visible_clusters:
            continue
        shown = True
        body.append(f"    <h3>{escape(ticker)}</h3>")
        body.append("    <table>")
        body.append("      <tr><th>Event</th><th>Bucket</th><th>Sentiment</th><th>Extraction Basis</th><th>Articles</th><th>Publishers</th><th>Sources</th></tr>")
        for cluster in visible_clusters:
            body.append(
                "      <tr>"
                f"<td><a href=\"{escape(cluster.primary_link, quote=True)}\">{escape(cluster.title)}</a></td>"
                f"<td>{escape(cluster.recency_bucket)}</td>"
                f"<td class=\"num\">{_format_optional_score(cluster.weighted_cluster_sentiment)}</td>"
                f"<td>{escape(cluster.extraction_basis)}</td>"
                f"<td class=\"num\">{cluster.article_count}</td>"
                f"<td class=\"num\">{cluster.publisher_count}</td>"
                f"<td class=\"num\">{cluster.source_count}</td>"
                "</tr>"
            )
        body.append("    </table>")
    if not shown:
        body.append("    <p class=\"empty\">No event clusters matched configured tickers.</p>")
    return "\n".join(body)


def _format_optional_score(value):
    return "" if value is None else f"{value:.4f}"


def _attachment_manifest(attachments: tuple[str, ...]) -> str:
    body = [
        "    <h2>Intended Attachments</h2>",
        "    <p>These local files would be attached by a future live email sender.</p>",
        "    <ul>",
    ]
    if attachments:
        body.extend(f"      <li>{escape(Path(path).name)}</li>" for path in attachments)
    else:
        body.append("      <li class=\"empty\">No attachments were generated.</li>")
    body.append("    </ul>")
    return "\n".join(body)
