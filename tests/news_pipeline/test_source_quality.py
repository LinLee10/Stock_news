import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from news_pipeline.cli import main
from news_pipeline.models import Article
from news_pipeline.source_quality import (
    TIER_1_HIGH_TRUST,
    TIER_2_USABLE,
    TIER_3_LOW_PRIORITY,
    TIER_4_EXCLUDE_BY_DEFAULT,
    assess_article_source,
    filter_articles_by_source_quality,
)


class FakeProvider:
    def __init__(self, articles):
        self._articles = articles

    def articles(self):
        return list(self._articles)


class SourceQualityTests(unittest.TestCase):
    def test_source_quality_tiers_classify_known_sources(self):
        self.assertEqual(assess_article_source(_article("https://www.reuters.com/markets/nvda", "Reuters")).tier, TIER_1_HIGH_TRUST)
        self.assertEqual(assess_article_source(_article("https://finance.yahoo.com/news/nvda", "Yahoo Finance")).tier, TIER_2_USABLE)
        self.assertEqual(assess_article_source(_article("https://stocktwits.com/nvda", "Stocktwits")).tier, TIER_3_LOW_PRIORITY)
        self.assertEqual(assess_article_source(_article("https://mshale.com/nvda", "Mshale")).tier, TIER_4_EXCLUDE_BY_DEFAULT)

    def test_tier_4_sources_are_hidden_from_visible_email_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                temp_dir,
                articles=[
                    _article("https://www.cnbc.com/nvda-quality", "CNBC", title="NVDA quality earnings coverage"),
                    _article("https://mshale.com/nvda-noise", "Mshale", title="NVDA Mshale noisy page"),
                ],
            )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            email_html = (output_dir / "email_preview.html").read_text(encoding="utf-8")
            diagnostics = json.loads((output_dir / "source_quality.json").read_text(encoding="utf-8"))

        self.assertIn("NVDA quality earnings coverage", email_html)
        self.assertNotIn("NVDA Mshale noisy page", email_html)
        self.assertEqual(payload["raw_article_count"], 2)
        self.assertEqual(payload["visible_article_count"], 1)
        self.assertEqual(payload["excluded_article_count"], 1)
        self.assertEqual(diagnostics["summary"]["excluded_tier_counts"]["tier_4_exclude_by_default"], 1)

    def test_include_low_quality_flag_restores_excluded_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                temp_dir,
                extra_args=["--include-low-quality-sources"],
                articles=[
                    _article("https://www.cnbc.com/nvda-quality", "CNBC", title="NVDA quality earnings coverage"),
                    _article("https://mshale.com/nvda-noise", "Mshale", title="NVDA Mshale noisy page"),
                ],
            )
            payload = json.loads(stdout)
            email_html = (Path(payload["output_dir"]) / "email_preview.html").read_text(encoding="utf-8")

        self.assertIn("NVDA Mshale noisy page", email_html)
        self.assertEqual(payload["visible_article_count"], 2)
        self.assertEqual(payload["excluded_article_count"], 0)

    def test_tier_3_sources_are_lower_priority_than_better_sources(self):
        result = filter_articles_by_source_quality(
            [
                _article("https://www.cnbc.com/nvda-1", "CNBC", title="NVDA results from CNBC"),
                _article("https://finance.yahoo.com/news/nvda-2", "Yahoo Finance", title="NVDA results from Yahoo Finance"),
                _article("https://stocktwits.com/nvda-3", "Stocktwits", title="NVDA chatter from Stocktwits"),
            ]
        )
        self.assertEqual([article.metadata["source_name"] for article in result.visible_articles], ["CNBC", "Yahoo Finance"])
        self.assertEqual(result.excluded_articles[0].metadata["source_name"], "Stocktwits")
        self.assertEqual(result.summary.tier_3_visible_articles, 0)
        self.assertEqual(result.summary.tier_3_hidden_articles, 1)

    def test_unknown_unclassified_sources_are_counted(self):
        result = filter_articles_by_source_quality(
            [
                Article(
                    canonical_url="https://unknown.example.com/nvda",
                    title="NVDA from an unlisted source",
                    snippet="NVDA stock news.",
                    published_at="2026-06-03T10:00:00+00:00",
                    metadata={},
                ),
            ]
        )

        self.assertEqual(result.summary.visible_articles, 1)
        self.assertEqual(result.summary.unknown_articles, 1)
        self.assertEqual(result.summary.unclassified_sources, ("unknown.example.com",))

    def test_cnbc_is_not_labeled_low_quality_in_source_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                temp_dir,
                articles=[
                    _article("https://www.cnbc.com/nvda-1", "CNBC", title="NVDA earnings from CNBC"),
                    _article("https://finance.yahoo.com/news/nvda-2", "Yahoo Finance", title="NVDA earnings from Yahoo Finance"),
                    _article("https://stocktwits.com/nvda-3", "Stocktwits", title="NVDA chatter from Stocktwits"),
                ],
            )
            payload = json.loads(stdout)
            email_html = (Path(payload["output_dir"]) / "email_preview.html").read_text(encoding="utf-8")

        self.assertIn("Hidden lower-priority sources: Stocktwits.", email_html)
        self.assertNotIn("low-quality sources: CNBC", email_html)
        self.assertNotIn("Excluded or hidden low-quality sources", email_html)
        self.assertNotIn("CNBC.", email_html.split("Source Quality Summary", 1)[1].split("Article Extraction Summary", 1)[0])

    def test_source_quality_summary_appears_and_fixture_wording_is_removed_for_live_report(self):
        live_feed = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Live</title>
<item><title>NVDA live RSS mention from CNBC</title><link>https://www.cnbc.com/nvda-live</link><source>CNBC</source><pubDate>Wed, 3 Jun 2026 10:00:00 GMT</pubDate><description>NVDA live RSS mention.</description></item>
</channel></rss>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(live_feed)):
                stdout = _run_cli(
                    temp_dir,
                    extra_args=[
                        "--enable-live-rss",
                        "--live-rss-url",
                        "https://example.com/rss.xml",
                        "--live-rss-retries",
                        "0",
                    ],
                    articles=[],
                )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            email_html = (output_dir / "email_preview.html").read_text(encoding="utf-8")
            markdown = (output_dir / "daily_report.md").read_text(encoding="utf-8")

        self.assertIn("Source Quality Summary", email_html)
        self.assertIn("free live RSS feeds", email_html)
        self.assertNotIn("fixture sentiment coverage", email_html)
        self.assertNotIn("fixture sentiment coverage", markdown)
        self.assertNotIn("placeholder forecast from stored fixture sentiment", email_html)

    def test_no_email_or_paid_provider_calls_are_needed_for_source_quality_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("smtplib.SMTP", side_effect=AssertionError("SMTP must not be called")):
                stdout = _run_cli(
                    temp_dir,
                    articles=[
                        _article("https://www.cnbc.com/nvda-quality", "CNBC", title="NVDA quality earnings coverage"),
                    ],
                )
            payload = json.loads(stdout)

        self.assertEqual(payload["email_sending"], "preview_only")
        self.assertFalse(payload["paid_apis_enabled"])


class FakeHttpResponse:
    def __init__(self, body):
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


def _run_cli(temp_dir, *, articles, extra_args=None):
    argv = [
        "dry-run-daily",
        "--run-date",
        "2026-06-03",
        "--artifacts-dir",
        str(Path(temp_dir) / "artifacts"),
    ]
    if extra_args:
        argv.extend(extra_args)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(
            argv,
            fake_providers={"google_news_rss": FakeProvider(articles)},
            environ={},
        )
    assert exit_code == 0
    return stdout.getvalue().strip()


def _article(url, source, *, title="NVDA stock story"):
    return Article(
        canonical_url=url,
        title=title,
        snippet="NVDA stock news.",
        published_at="2026-06-03T10:00:00+00:00",
        metadata={"provider": "google_news_rss", "source_name": source},
    )


if __name__ == "__main__":
    unittest.main()
