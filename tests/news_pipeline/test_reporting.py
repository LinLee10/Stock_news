import csv
import tempfile
from pathlib import Path
import unittest

from news_pipeline.reporting import (
    ArticleLink,
    DailyReportInput,
    EmergingNameRow,
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
            self.assertEqual(len(report.csv_attachments), 5)
            self.assertEqual(len(report.chart_attachments), 2)
            for attachment in report.csv_attachments + report.chart_attachments:
                path = Path(attachment)
                self.assertTrue(path.exists())
                self.assertTrue(path.is_relative_to(output_dir))
            self.assertIn("Daily report for 2026-06-03", report.daily_summary)

    def test_report_tables_and_links_are_in_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_fake_report_input(), artifacts_dir=Path(temp_dir))

            self.assertEqual(report.portfolio_30d_sentiment_table[0].ticker, "AAPL")
            self.assertEqual(report.watchlist_next_close_table[0].next_close_direction, "up")
            self.assertEqual(report.mention_leaders_7d_table[0].ticker, "NVDA")
            self.assertEqual(len(report.top_10_most_mentioned_table), 10)
            self.assertEqual(report.emerging_names_table[0].ticker, "ARM")
            self.assertEqual(report.supporting_article_links["AAPL"][0].url, "https://example.com/aapl")

    def test_csv_attachment_contains_expected_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_daily_report(_fake_report_input(), artifacts_dir=Path(temp_dir))
            portfolio_csv = Path(report.csv_attachments[0])

            with portfolio_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["ticker"], "AAPL")
            self.assertEqual(rows[0]["sentiment_basis"], "full_text")

    def test_default_output_dir_is_repo_artifacts_runs_not_root(self):
        report = build_daily_report(_fake_report_input(), artifacts_dir="artifacts")
        output_dir = Path(report.output_dir)
        try:
            self.assertEqual(output_dir.parts[:2], ("artifacts", "runs"))
            self.assertEqual(output_dir.name, "2026-06-03")
        finally:
            for attachment in report.csv_attachments + report.chart_attachments:
                Path(attachment).unlink(missing_ok=True)
            output_dir.rmdir()
            (Path("artifacts") / "runs").rmdir()


def _fake_report_input() -> DailyReportInput:
    return DailyReportInput(
        report_date="2026-06-03",
        portfolio_sentiment=(
            PortfolioSentimentRow("AAPL", "Apple Inc.", 0.42, 12, "full_text"),
            PortfolioSentimentRow("MSFT", "Microsoft Corp.", 0.12, 8, "snippet"),
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
    )


if __name__ == "__main__":
    unittest.main()
