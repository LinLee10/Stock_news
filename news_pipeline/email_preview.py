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
        "    <div class=\"note\"><strong>Preview only:</strong> no live email provider was contacted.</div>",
        f"    <div class=\"note\"><strong>Data source:</strong> {escape(report.data_source_label)}. No paid APIs were called.</div>",
        "    <div class=\"note\"><strong>Model status:</strong> Sentiment is deterministic placeholder logic, and watchlist direction rows are not real predictions. This preview is not investment advice.</div>",
        f"    <div class=\"summary\">{escape(report.daily_summary)}</div>",
        _top_briefing(report),
        _event_clusters(report),
        _article_links(report),
        _source_quality_summary(report),
        _extraction_summary(report),
        _recency_sections(report),
        _sentiment_table("Portfolio Recency Sentiment", report.portfolio_30d_sentiment_table),
        _sentiment_table("Watchlist Recency Sentiment", report.watchlist_sentiment_table),
        _forecast_table(report),
        _mention_leaders_table(report),
        _top_mentions_table(report),
        _emerging_names_table(report),
        _attachment_manifest(attachments),
        "  </div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(sections)


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
            "      <tr><th>Full Text Basis</th><th>Snippet Basis</th><th>Title Basis</th></tr>",
            "      <tr>"
            f"<td class=\"num\">{basis_counts['full_text']}</td>"
            f"<td class=\"num\">{basis_counts['snippet']}</td>"
            f"<td class=\"num\">{basis_counts['title']}</td>"
            "</tr>",
            "    </table>",
            _failure_reasons(summary.top_extraction_failure_reasons),
            _extraction_diagnostics(summary),
        ]
    )


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
        body.append("    <p class=\"empty\">No article links matched configured tickers.</p>")
    return "\n".join(body)


def _event_clusters(report: DailyReportContract) -> str:
    body = ["    <h2>Top Event Clusters By Recency And Source Diversity</h2>"]
    shown = False
    for ticker, clusters in sorted(report.event_clusters_by_ticker.items()):
        visible_clusters = sorted(clusters, key=lambda cluster: (cluster.source_quality_label, cluster.title))[:5]
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
        "    <p>These files would be attached.</p>",
        "    <ul>",
    ]
    if attachments:
        body.extend(f"      <li>{escape(Path(path).name)}</li>" for path in attachments)
    else:
        body.append("      <li class=\"empty\">No attachments were generated.</li>")
    body.append("    </ul>")
    return "\n".join(body)
