"""Report data contract and local artifact writers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .charts import write_placeholder_chart
from .models import RunResult


@dataclass(frozen=True)
class ArticleLink:
    title: str
    url: str
    source: str | None = None


@dataclass(frozen=True)
class PortfolioSentimentRow:
    ticker: str
    company_name: str
    sentiment_30d: float
    article_count_30d: int
    sentiment_basis: str


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
    watchlist_forecasts: tuple[WatchlistForecastRow, ...] = ()
    mention_leaders_7d: tuple[MentionLeaderRow, ...] = ()
    most_mentioned: tuple[MostMentionedRow, ...] = ()
    emerging_names: tuple[EmergingNameRow, ...] = ()
    article_links_by_ticker: Mapping[str, tuple[ArticleLink, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class DailyReportContract:
    report_date: str
    output_dir: str
    portfolio_30d_sentiment_table: tuple[PortfolioSentimentRow, ...]
    watchlist_next_close_table: tuple[WatchlistForecastRow, ...]
    mention_leaders_7d_table: tuple[MentionLeaderRow, ...]
    top_10_most_mentioned_table: tuple[MostMentionedRow, ...]
    emerging_names_table: tuple[EmergingNameRow, ...]
    supporting_article_links: Mapping[str, tuple[ArticleLink, ...]]
    csv_attachments: tuple[str, ...]
    chart_attachments: tuple[str, ...]
    daily_summary: str


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
    csv_attachments = (
        _write_csv(
            output_dir / "portfolio_30d_sentiment.csv",
            ("ticker", "company_name", "sentiment_30d", "article_count_30d", "sentiment_basis"),
            report_input.portfolio_sentiment,
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
            "Portfolio 30 Day Sentiment",
            [row.ticker for row in report_input.portfolio_sentiment],
        ),
        write_placeholder_chart(
            output_dir / "mention_leaders.svg",
            "7 Day Mention Leaders",
            [row.ticker for row in report_input.mention_leaders_7d],
        ),
    )

    return DailyReportContract(
        report_date=report_input.report_date,
        output_dir=str(output_dir),
        portfolio_30d_sentiment_table=report_input.portfolio_sentiment,
        watchlist_next_close_table=report_input.watchlist_forecasts,
        mention_leaders_7d_table=report_input.mention_leaders_7d,
        top_10_most_mentioned_table=top_10,
        emerging_names_table=report_input.emerging_names,
        supporting_article_links=report_input.article_links_by_ticker,
        csv_attachments=csv_attachments,
        chart_attachments=chart_attachments,
        daily_summary=_plain_english_summary(report_input, top_10),
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


def _plain_english_summary(
    report_input: DailyReportInput,
    top_10: tuple[MostMentionedRow, ...],
) -> str:
    portfolio_count = len(report_input.portfolio_sentiment)
    watchlist_count = len(report_input.watchlist_forecasts)
    top_mentions = ", ".join(row.ticker for row in top_10[:3]) or "none"
    emerging = ", ".join(row.ticker for row in report_input.emerging_names[:3]) or "none"
    return (
        f"Daily report for {report_input.report_date}: "
        f"{portfolio_count} portfolio names have 30 day sentiment coverage, "
        f"{watchlist_count} watchlist names have next close direction forecasts, "
        f"top mention leaders are {top_mentions}, and emerging names are {emerging}."
    )
