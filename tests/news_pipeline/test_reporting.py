import csv
import tempfile
from pathlib import Path
import unittest

from news_pipeline.reporting import (
    ArticleLink,
    DailyReportInput,
    EmergingNameRow,
    EventClusterRow,
    MentionLeaderRow,
    MostMentionedRow,
    PortfolioSentimentRow,
    WatchlistForecastRow,
    build_daily_report,
)


class ReportingTests(unittest.TestCase):
    def test_daily_report_contract_writes_artifacts_under_run_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_fake_report_input(), artifacts_dir=Path(temp_dir))

            output_dir = Path(report.output_dir)
            self.assertEqual(output_dir, Path(temp_dir) / "artifacts" / "runs" / "2026-06-03")
            self.assertTrue(output_dir.exists())
            self.assertEqual(len(report.csv_attachments), 6)
            self.assertEqual(len(report.chart_attachments), 3)
            for attachment in (
                report.csv_attachments
                + report.chart_attachments
                + report.supplemental_csv_artifacts
            ):
                path = Path(attachment)
                self.assertTrue(path.exists())
                self.assertTrue(path.is_relative_to(output_dir))
            self.assertTrue(Path(report.html_preview_report).exists())
            self.assertTrue(Path(report.html_preview_report).is_relative_to(output_dir))
            self.assertIn("Daily report for 2026-06-03", report.daily_summary)
            supplemental_names = {Path(path).name for path in report.supplemental_csv_artifacts}
            self.assertTrue(
                {
                    "source_profiles.csv",
                    "source_family_counts.csv",
                    "source_acquisition_diagnostics.csv",
                    "source_diversity_diagnostics.csv",
                    "paid_api_skipped_reasons.csv",
                    "missing_company_ir_profiles.csv",
                }.issubset(supplemental_names)
            )

    def test_report_tables_and_links_are_in_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_fake_report_input(), artifacts_dir=Path(temp_dir))

            self.assertEqual(report.portfolio_30d_sentiment_table[0].ticker, "AAPL")
            self.assertEqual(report.watchlist_sentiment_table[0].ticker, "NVDA")
            self.assertEqual(report.watchlist_next_close_table[0].next_close_direction, "up")
            self.assertEqual(report.mention_leaders_7d_table[0].ticker, "NVDA")
            self.assertEqual(len(report.top_10_most_mentioned_table), 10)
            self.assertEqual(report.emerging_names_table[0].ticker, "ARM")
            self.assertEqual(report.recency_sections["today_signal"][0].ticker, "NVDA")
            self.assertEqual(report.top_event_clusters[0].ticker, "AAPL")
            self.assertEqual(report.supporting_article_links["AAPL"][0].url, "https://example.com/aapl")

    def test_csv_attachment_contains_expected_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_fake_report_input(), artifacts_dir=Path(temp_dir))
            portfolio_csv = Path(report.csv_attachments[0])

            with portfolio_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["ticker"], "AAPL")
            self.assertEqual(rows[0]["sentiment_basis"], "full_text")

    def test_report_output_caps_visible_links_and_event_clusters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_large_link_report_input(), artifacts_dir=Path(temp_dir))
            html = Path(report.html_preview_report).read_text(encoding="utf-8")

            self.assertIn("Today&#x27;s Signal", html)
            self.assertIn("Recent Pulse", html)
            self.assertIn("Weekly Trend", html)
            self.assertIn("Background Context", html)
            self.assertIn("Portfolio Recency Sentiment", html)
            self.assertIn("Watchlist Recency Sentiment", html)
            self.assertIn("Top 7 Day Mention Leaders", html)
            self.assertIn("Emerging Names Based On Mention Velocity", html)
            self.assertIn("Portfolio and Watchlist Market Briefing", html)
            self.assertIn("Stories to Watch", html)
            self.assertIn("Read More By Ticker", html)
            self.assertIn("This briefing is not investment advice", html)
            self.assertIn("deterministic placeholder logic", html)
            self.assertIn("Source Coverage:", html)
            self.assertIn("+5 more links in JSON artifacts", html)
            self.assertIn("Event 4", html)
            self.assertNotIn("Event 5", html)
            self.assertNotIn("Article 14", html)
            self.assertLess(html.index("Stories to Watch"), html.index("Read More By Ticker"))
            self.assertLess(html.index("Read More By Ticker"), html.index("Source Quality Summary"))

    def test_report_warns_when_run_date_differs_from_current_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_fake_report_input(), artifacts_dir=Path(temp_dir))
            html = Path(report.html_preview_report).read_text(encoding="utf-8")

            self.assertIn("Run date differs from current date.", report.report_warnings)
            self.assertIn("Run date differs from current date.", html)
            self.assertGreater(html.index("Run date differs from current date."), html.index("Read More By Ticker"))

    def test_default_output_dir_is_repo_artifacts_runs_not_root(self):
        report = build_daily_report(_fake_report_input(), artifacts_dir="artifacts")
        output_dir = Path(report.output_dir)
        try:
            self.assertEqual(output_dir.parts[:2], ("artifacts", "runs"))
            self.assertEqual(output_dir.name, "2026-06-03")
        finally:
            for attachment in (
                report.csv_attachments
                + report.chart_attachments
                + report.supplemental_csv_artifacts
            ):
                Path(attachment).unlink(missing_ok=True)
            Path(report.html_preview_report).unlink(missing_ok=True)
            output_dir.rmdir()
            try:
                (Path("artifacts") / "runs").rmdir()
            except OSError:
                pass


def _fake_report_input() -> DailyReportInput:
    return DailyReportInput(
        report_date="2026-06-03",
        portfolio_sentiment=(
            PortfolioSentimentRow("AAPL", "Apple Inc.", 0.42, 12, "full_text", 0.5, 0.3, 0.1, 0.0, 0.35, 2, 5, 9, "limited_history", 2),
            PortfolioSentimentRow("MSFT", "Microsoft Corp.", 0.12, 8, "snippet", 0.1, 0.2, 0.0, 0.0, 0.13, 1, 4, 6, "limited_history", 1),
        ),
        watchlist_sentiment=(
            PortfolioSentimentRow("NVDA", "NVIDIA", 0.31, 19, "snippet", 0.4, 0.2, 0.1, 0.0, 0.29, 3, 8, 14, "limited_history", 3),
            PortfolioSentimentRow("TSLA", "Tesla", -0.14, 5, "title", -0.2, -0.1, 0.0, 0.0, -0.16, 1, 2, 4, "limited_history", 1),
        ),
        watchlist_forecasts=(
            WatchlistForecastRow("NVDA", "up", 0.73, "positive sentiment and mentions"),
            WatchlistForecastRow("TSLA", "uncertain", 0.51, "mixed headlines"),
        ),
        mention_leaders_7d=(
            MentionLeaderRow("NVDA", 19, 0.31),
            MentionLeaderRow("AAPL", 13, 0.22),
        ),
        most_mentioned=tuple(
            MostMentionedRow(f"T{i}", 100 - i, i)
            for i in range(1, 13)
        ),
        emerging_names=(
            EmergingNameRow("ARM", "Arm Holdings", 6, 1, "mentions accelerated"),
        ),
        article_links_by_ticker={
            "AAPL": (
                ArticleLink("Apple earnings coverage", "https://example.com/aapl", "Example"),
            )
        },
        event_clusters_by_ticker={
            "AAPL": (
                EventClusterRow("AAPL", "Apple earnings coverage", "https://example.com/aapl", 1, 1, 1),
            )
        },
        source_coverage_diagnostics={
            "official_source_count": 1,
            "company_ir_count": 0,
            "press_release_wire_count": 2,
            "direct_publisher_count": 4,
            "google_news_backstop_count": 3,
            "google_news_share": 0.3,
            "paid_api_count": 0,
            "paid_api_status": "disabled_or_skipped",
            "source_diversity_score": 80.0,
            "source_balance_score": 70.0,
            "source_family_counts": {
                "regulatory_official": 1,
                "press_release_wire": 2,
                "direct_news_publisher": 4,
                "google_news_backstop": 3,
            },
            "source_profiles": (
                {
                    "source_id": "example",
                    "source_family": "direct_news_publisher",
                    "publisher_name": "Example",
                },
            ),
            "paid_api_skipped_reasons": {"marketaux": "global_paid_api_flag_disabled"},
            "missing_company_ir_profiles": ("AAPL",),
        },
    )


def _large_link_report_input() -> DailyReportInput:
    return DailyReportInput(
        report_date="2026-06-03",
        portfolio_sentiment=(
            PortfolioSentimentRow("AAPL", "Apple Inc.", 0.1, 15, "snippet", 0.2, 0.1, 0.0, 0.0, 0.15, 2, 6, 10, "limited_history", 3),
        ),
        watchlist_sentiment=(
            PortfolioSentimentRow("NVDA", "NVIDIA", 0.2, 8, "snippet", 0.3, 0.2, 0.0, 0.0, 0.24, 1, 4, 7, "limited_history", 2),
        ),
        article_links_by_ticker={
            "AAPL": tuple(
                ArticleLink(f"Article {index}", f"https://example.com/aapl/{index}", "Example")
                for index in range(15)
            )
        },
        event_clusters_by_ticker={
            "AAPL": tuple(
                EventClusterRow(
                    "AAPL",
                    f"Event {index}",
                    f"https://example.com/event/{index}",
                    2,
                    2,
                    5,
                    supporting_links=tuple(
                        ArticleLink(f"Supporting {index}-{link}", f"https://example.com/event/{index}/{link}", "Example")
                        for link in range(5)
                    ),
                )
                for index in range(7)
            )
        },
    )


if __name__ == "__main__":
    unittest.main()
