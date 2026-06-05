import contextlib
import csv
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from news_pipeline.cli import main
from news_pipeline.models import Article
from news_pipeline.storage import SQLiteStore
from news_pipeline.tickers import symbols


LIVE_RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Live Mock Finance</title>
    <item>
      <title>NVIDIA live RSS update</title>
      <link>https://live.example.com/news/nvidia-live</link>
      <pubDate>Wed, 3 Jun 2026 10:00:00 GMT</pubDate>
      <source>Live Mock Wire</source>
      <description>NVDA stock news from a mocked live RSS source.</description>
    </item>
  </channel>
</rss>
"""

MULTI_LIVE_RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Live Mock Finance</title>
    <item>
      <title>NVIDIA beats earnings expectations</title>
      <link>https://live.example.com/news/nvidia-live-1</link>
      <pubDate>Wed, 3 Jun 2026 10:00:00 GMT</pubDate>
      <source>Live Mock Wire</source>
      <description>NVDA stock earnings news from a mocked live RSS source.</description>
    </item>
    <item>
      <title>NVIDIA stock upgraded by analyst</title>
      <link>https://live.example.com/news/nvidia-live-2</link>
      <pubDate>Wed, 3 Jun 2026 11:00:00 GMT</pubDate>
      <source>Live Mock Wire</source>
      <description>NVDA stock analyst rating news from a mocked live RSS source.</description>
    </item>
    <item>
      <title>NVIDIA faces antitrust lawsuit</title>
      <link>https://live.example.com/news/nvidia-live-3</link>
      <pubDate>Wed, 3 Jun 2026 12:00:00 GMT</pubDate>
      <source>Live Mock Wire</source>
      <description>NVDA stock legal news from a mocked live RSS source.</description>
    </item>
  </channel>
</rss>
"""

ARTICLE_HTML = """<!doctype html>
<html>
  <head><title>NVIDIA article page</title></head>
  <body>
    <article><p>NVIDIA reports weak demand and a loss in this full article.</p></article>
  </body>
</html>
"""


class FakeProvider:
    def __init__(self, article):
        self.article = article

    def articles(self):
        return [self.article]


class FakeHttpResponse:
    def __init__(self, body):
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class FakeArticleResponse(FakeHttpResponse):
    url = "https://publisher.example.com/final-article"

    def __init__(self, body):
        super().__init__(body)
        self.headers = {"Content-Type": "text/html; charset=utf-8"}


class CliTests(unittest.TestCase):
    def test_init_db_writes_under_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli("init-db", temp_dir)
            payload = json.loads(stdout)
            output_dir = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03"

            self.assertEqual(Path(payload["output_dir"]), output_dir)
            self.assertTrue((output_dir / "news_pipeline.sqlite3").exists())
            self.assertTrue((output_dir / "init_db.json").exists())
            self.assertIn("articles", payload["tables"])

    def test_validate_providers_does_not_print_secret_values(self):
        secret = "resend-secret-value"
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                "validate-providers",
                temp_dir,
                environ={"RESEND_API_KEY": secret},
            )
            output = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03" / "provider_validation.json"
            payload = output.read_text(encoding="utf-8")

            self.assertNotIn(secret, stdout)
            self.assertNotIn(secret, payload)
            self.assertIn('"key_state": "present"', payload)

    def test_collect_skips_paid_fake_providers_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                "collect",
                temp_dir,
                fake_providers={
                    "google_news_rss": FakeProvider(
                        Article(canonical_url="https://example.com/rss", title="RSS story")
                    ),
                    "marketaux": FakeProvider(
                        Article(canonical_url="https://example.com/paid", title="Paid story")
                    ),
                },
            )
            payload = json.loads(stdout)
            output = Path(payload["output"])
            stored = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["article_count"], 1)
            self.assertEqual(stored["articles"][0]["title"], "RSS story")
            self.assertNotIn("Paid story", output.read_text(encoding="utf-8"))

    def test_collect_includes_paid_fake_providers_when_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                "collect",
                temp_dir,
                extra_args=["--enable-paid-apis"],
                fake_providers={
                    "marketaux": FakeProvider(
                        Article(canonical_url="https://example.com/paid", title="Paid story")
                    ),
                },
            )

            self.assertEqual(json.loads(stdout)["article_count"], 1)

    def test_pipeline_commands_write_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_providers = {
                "google_news_rss": FakeProvider(
                    Article(
                        canonical_url="https://example.com/story?utm_source=rss",
                        title="Nvidia beats estimates",
                        snippet="Nvidia reported strong growth.",
                    )
                )
            }

            for command, filename in [
                ("extract", "extractions.json"),
                ("dedup", "dedupe_clusters.json"),
                ("score", "sentiment_scores.json"),
                ("report", "report_contract.json"),
                ("dry-run-daily", "dry_run_daily.json"),
            ]:
                _run_cli(command, temp_dir, fake_providers=fake_providers)
                output = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03" / filename
                self.assertTrue(output.exists(), command)

    def test_dry_run_daily_never_enables_email_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli("dry-run-daily", temp_dir)
            payload = json.loads(stdout)

            self.assertEqual(payload["status"], "dry_run_complete")
            self.assertEqual(payload["email_sending"], "preview_only")
            self.assertEqual(payload["email_delivery_mode"], "local_preview_only")
            self.assertFalse(payload["email_send_enabled"])
            self.assertFalse(payload["paid_apis_enabled"])
            self.assertTrue(Path(payload["email_preview_html"]).exists())

    def test_dry_run_daily_uses_configured_tickers_and_fixture_articles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli("dry-run-daily", temp_dir)
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])

            collected = json.loads((output_dir / "collected_articles.json").read_text(encoding="utf-8"))
            contract = json.loads((output_dir / "report_contract.json").read_text(encoding="utf-8"))["report"]
            markdown = (output_dir / "daily_report.md").read_text(encoding="utf-8")
            html = (output_dir / "daily_report.html").read_text(encoding="utf-8")
            self.assertGreater(collected["article_count"], 0)
            self.assertTrue(any("NVIDIA" in article["title"] for article in collected["articles"]))
            self.assertEqual(
                [row["ticker"] for row in contract["portfolio_30d_sentiment_table"]],
                list(symbols("portfolio")),
            )
            self.assertEqual(
                [row["ticker"] for row in contract["watchlist_next_close_table"]],
                list(symbols("watchlist")),
            )
            self.assertNotIn("AAPL", [row["ticker"] for row in contract["portfolio_30d_sentiment_table"]])
            self.assertNotIn("AAPL", [row["ticker"] for row in contract["watchlist_next_close_table"]])
            self.assertIn("NVDA", contract["supporting_article_links"])
            self.assertTrue(contract["supporting_article_links"]["NVDA"])
            self.assertIn("https://example.com/news/nvidia-new-chip", markdown)
            self.assertIn("local RSS fixture files by default", markdown)
            self.assertIn("Placeholder forecast logic", markdown)
            self.assertIn("Watchlist Recency Sentiment", html)
            self.assertIn("Article Links Grouped By Ticker And Event Cluster", html)
            self.assertIn("full_text, snippet, title, or no_articles", html)
            self.assertTrue(Path(contract["html_preview_report"]).exists())

            with (output_dir / "portfolio_30d_sentiment.csv").open(newline="", encoding="utf-8") as handle:
                portfolio_rows = list(csv.DictReader(handle))
            with (output_dir / "watchlist_sentiment.csv").open(newline="", encoding="utf-8") as handle:
                watchlist_sentiment_rows = list(csv.DictReader(handle))
            with (output_dir / "watchlist_next_close.csv").open(newline="", encoding="utf-8") as handle:
                watchlist_rows = list(csv.DictReader(handle))

            self.assertEqual([row["ticker"] for row in portfolio_rows], list(symbols("portfolio")))
            self.assertEqual([row["ticker"] for row in watchlist_sentiment_rows], list(symbols("watchlist")))
            self.assertEqual([row["ticker"] for row in watchlist_rows], list(symbols("watchlist")))

    def test_dry_run_daily_persists_run_data_to_sqlite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli("dry-run-daily", temp_dir)
            payload = json.loads(stdout)
            run_id = payload["run_id"]
            store = SQLiteStore(payload["database_path"])
            try:
                run = store.get_run(run_id)
                articles = store.list_run_articles(run_id)
                all_articles = store.list_articles()
                sources = store.list_article_sources(run_id)
                provider_validation = store.list_provider_validation(run_id)
                clusters = store.list_dedupe_clusters(run_id)
                sentiments = store.list_sentiment_results(run_id)
                mentions = store.list_ticker_mentions(run_id)
            finally:
                store.close()

            self.assertEqual(run["run_date"], "2026-06-03")
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["articles_seen"], payload["article_count"])
            self.assertEqual(len(articles), payload["article_count"])
            self.assertEqual(len(sources), payload["article_count"])
            self.assertEqual(len(provider_validation), 8)
            self.assertEqual(len(clusters), payload["cluster_count"])
            self.assertEqual(len(sentiments), payload["score_count"])
            self.assertTrue(any(row["ticker"] == "NVDA" for row in mentions))
            self.assertTrue(all(row["full_text"] is None for row in all_articles))

    def test_dry_run_daily_same_date_is_idempotent_in_sqlite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = json.loads(_run_cli("dry-run-daily", temp_dir))
            first_counts = _database_counts(first["database_path"])

            second = json.loads(_run_cli("dry-run-daily", temp_dir))
            second_counts = _database_counts(second["database_path"])

            self.assertEqual(first["run_id"], "dry-run-2026-06-03")
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first_counts, second_counts)
            self.assertEqual(second_counts["runs"], 1)
            self.assertEqual(second_counts["provider_validation"], 8)
            self.assertEqual(second_counts["provider_usage"], 0)
            self.assertEqual(second_counts["run_articles"], second["article_count"])
            self.assertEqual(second_counts["dedupe_clusters"], second["cluster_count"])
            self.assertEqual(second_counts["sentiment_results"], second["score_count"])

    def test_dry_run_daily_live_rss_same_date_is_idempotent_in_sqlite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            extra_args = [
                "--enable-live-rss",
                "--live-rss-url",
                "https://example.com/rss/nvda.xml",
                "--live-rss-retries",
                "0",
            ]
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(LIVE_RSS_FIXTURE)):
                first = json.loads(_run_cli("dry-run-daily", temp_dir, extra_args=extra_args))
                first_counts = _database_counts(first["database_path"])
                second = json.loads(_run_cli("dry-run-daily", temp_dir, extra_args=extra_args))
                second_counts = _database_counts(second["database_path"])

            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first_counts, second_counts)
            self.assertEqual(second_counts["runs"], 1)
            self.assertEqual(second_counts["provider_usage"], 1)
            self.assertEqual(second_counts["provider_validation"], 8)
            self.assertEqual(second_counts["run_articles"], second["article_count"])

    def test_dry_run_daily_writes_email_preview_without_email_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("smtplib.SMTP", side_effect=AssertionError("SMTP must not be called")):
                with patch("smtplib.SMTP_SSL", side_effect=AssertionError("SMTP_SSL must not be called")):
                    stdout = _run_cli(
                        "dry-run-daily",
                        temp_dir,
                        extra_args=["--enable-email-send"],
                    )
            payload = json.loads(stdout)
            preview_path = Path(payload["email_preview_html"])
            preview = preview_path.read_text(encoding="utf-8")

            self.assertTrue(preview_path.exists())
            self.assertEqual(preview_path.name, "email_preview.html")
            self.assertTrue(preview_path.is_relative_to(Path(payload["output_dir"])))
            self.assertEqual(payload["email_sending"], "preview_only")
            self.assertEqual(payload["email_delivery_mode"], "local_preview_only")
            self.assertIn("No SMTP, Gmail, Resend, or other live email provider was contacted.", preview)
            self.assertIn("Daily report for 2026-06-03", preview)
            self.assertIn("Portfolio Recency Sentiment", preview)
            self.assertIn("Watchlist Recency Sentiment", preview)
            self.assertIn("Article Links Grouped By Ticker And Event Cluster", preview)
            self.assertIn("https://example.com/news/nvidia-new-chip", preview)
            self.assertIn("Intended Attachments", preview)
            self.assertIn("portfolio_30d_sentiment.csv", preview)
            self.assertIn("watchlist_sentiment.svg", preview)
            self.assertTrue(payload["intended_email_attachments"])

    def test_dry_run_daily_default_does_not_attempt_live_rss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", side_effect=AssertionError("live RSS must be opt-in")):
                stdout = _run_cli("dry-run-daily", temp_dir)
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            provider_validation = json.loads((output_dir / "provider_validation.json").read_text(encoding="utf-8"))

            self.assertFalse(payload["live_rss_enabled"])
            self.assertFalse(provider_validation["live_rss"]["enabled"])
            self.assertEqual(provider_validation["live_rss"]["attempt_count"], 0)

    def test_dry_run_daily_live_rss_opt_in_uses_mocked_http_and_records_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(LIVE_RSS_FIXTURE)) as urlopen:
                stdout = _run_cli(
                    "dry-run-daily",
                    temp_dir,
                    extra_args=[
                        "--enable-live-rss",
                        "--live-rss-url",
                        "https://example.com/rss/nvda.xml",
                        "--live-rss-timeout-seconds",
                        "1",
                        "--live-rss-retries",
                        "0",
                        "--live-rss-user-agent",
                        "UnitTestAgent/1.0",
                    ],
                )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            collected = json.loads((output_dir / "collected_articles.json").read_text(encoding="utf-8"))
            provider_validation = json.loads((output_dir / "provider_validation.json").read_text(encoding="utf-8"))
            store = SQLiteStore(payload["database_path"])
            try:
                usage_rows = store.list_provider_usage()
            finally:
                store.close()

            self.assertTrue(payload["live_rss_enabled"])
            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.headers["User-agent"], "UnitTestAgent/1.0")
            self.assertTrue(any(article["title"] == "NVIDIA live RSS update" for article in collected["articles"]))
            self.assertEqual(collected["article_count"], 1)
            self.assertFalse(any(article["canonical_url"].startswith("https://example.com/news/") for article in collected["articles"]))
            self.assertEqual(provider_validation["live_rss"]["attempt_count"], 1)
            self.assertEqual(provider_validation["live_rss"]["success_count"], 1)
            self.assertEqual(provider_validation["live_rss"]["article_count"], 1)
            self.assertEqual(provider_validation["live_rss"]["source_counts"]["google_news_rss_search"], 1)
            self.assertEqual(len(usage_rows), 1)
            self.assertEqual(usage_rows[0]["provider"], "google_news_rss_search")
            self.assertEqual(usage_rows[0]["operation"], "discover")
            self.assertEqual(usage_rows[0]["status"], "success")
            self.assertEqual(usage_rows[0]["quota_cost"], 0)
            self.assertEqual(usage_rows[0]["article_count"], 1)

    def test_dry_run_daily_live_rss_can_explicitly_include_fixtures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(LIVE_RSS_FIXTURE)):
                stdout = _run_cli(
                    "dry-run-daily",
                    temp_dir,
                    extra_args=[
                        "--enable-live-rss",
                        "--include-fixtures",
                        "--live-rss-url",
                        "https://example.com/rss/nvda.xml",
                        "--live-rss-retries",
                        "0",
                    ],
                )

            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            collected = json.loads((output_dir / "collected_articles.json").read_text(encoding="utf-8"))

            self.assertGreater(collected["article_count"], 1)
            self.assertTrue(collected["fixtures_included"])
            self.assertTrue(any(article["title"] == "NVIDIA live RSS update" for article in collected["articles"]))
            self.assertTrue(any(article["canonical_url"].startswith("https://example.com/news/") for article in collected["articles"]))

    def test_dry_run_daily_live_rss_failure_is_recorded_without_failing_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", side_effect=TimeoutError("mock timeout")):
                stdout = _run_cli(
                    "dry-run-daily",
                    temp_dir,
                    extra_args=[
                        "--enable-live-rss",
                        "--live-rss-url",
                        "https://example.com/rss/failing.xml",
                        "--live-rss-retries",
                        "0",
                    ],
                )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            provider_validation = json.loads((output_dir / "provider_validation.json").read_text(encoding="utf-8"))
            store = SQLiteStore(payload["database_path"])
            try:
                usage_rows = store.list_provider_usage()
            finally:
                store.close()

            self.assertEqual(payload["status"], "dry_run_complete")
            self.assertEqual(provider_validation["live_rss"]["attempt_count"], 1)
            self.assertEqual(provider_validation["live_rss"]["failure_count"], 1)
            self.assertEqual(provider_validation["live_rss"]["attempts"][0]["status"], "failure")
            self.assertEqual(provider_validation["live_rss"]["attempts"][0]["error_class"], "TimeoutError")
            self.assertEqual(len(usage_rows), 1)
            self.assertEqual(usage_rows[0]["status"], "failure")
            self.assertEqual(usage_rows[0]["error_class"], "TimeoutError")

    def test_dry_run_daily_default_does_not_fetch_article_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.article_fetch.urlopen", side_effect=AssertionError("article pages must be opt-in")):
                stdout = _run_cli("dry-run-daily", temp_dir)
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            extractions = json.loads((output_dir / "article_extractions.json").read_text(encoding="utf-8"))

            self.assertFalse(payload["live_article_fetch_enabled"])
            self.assertEqual(payload["article_pages_fetched"], 0)
            self.assertGreater(payload["sentiment_basis_counts"]["snippet"], 0)
            self.assertFalse(extractions["article_fetch"]["enabled"])
            self.assertEqual(extractions["article_fetch"]["attempted_fetches"], 0)

    def test_enable_live_article_fetch_fetches_top_live_cluster(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(LIVE_RSS_FIXTURE)):
                with patch("news_pipeline.article_fetch.urlopen", return_value=FakeArticleResponse(ARTICLE_HTML)) as article_urlopen:
                    stdout = _run_cli(
                        "dry-run-daily",
                        temp_dir,
                        extra_args=[
                            "--enable-live-rss",
                            "--enable-live-article-fetch",
                            "--live-rss-url",
                            "https://example.com/rss/nvda.xml",
                            "--live-rss-retries",
                            "0",
                        ],
                    )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            collected = json.loads((output_dir / "collected_articles.json").read_text(encoding="utf-8"))
            extractions = json.loads((output_dir / "article_extractions.json").read_text(encoding="utf-8"))["article_fetch"]
            scores = json.loads((output_dir / "sentiment_scores.json").read_text(encoding="utf-8"))["scores"]
            report_html = (output_dir / "daily_report.html").read_text(encoding="utf-8")

            self.assertTrue(payload["live_article_fetch_enabled"])
            self.assertEqual(article_urlopen.call_count, 1)
            self.assertEqual(payload["article_pages_fetched"], 1)
            self.assertEqual(payload["successful_extractions"], 1)
            self.assertEqual(extractions["attempted_fetches"], 1)
            self.assertEqual(extractions["successful_extractions"], 1)
            self.assertEqual(extractions["records"][0]["extraction_basis"], "full_text")
            self.assertTrue(collected["articles"][0]["has_full_text"])
            self.assertEqual(scores[0]["basis"], "full_text")
            self.assertEqual(scores[0]["label"], "negative")
            self.assertIn("Article Extraction Summary", report_html)
            self.assertIn("<td class=\"num\">1</td>", report_html)
            self.assertIn("full_text", report_html)

    def test_live_article_fetch_caps_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(MULTI_LIVE_RSS_FIXTURE)):
                with patch("news_pipeline.article_fetch.urlopen", return_value=FakeArticleResponse(ARTICLE_HTML)) as article_urlopen:
                    stdout = _run_cli(
                        "dry-run-daily",
                        temp_dir,
                        extra_args=[
                            "--enable-live-rss",
                            "--enable-live-article-fetch",
                            "--live-rss-url",
                            "https://example.com/rss/nvda.xml",
                            "--live-rss-retries",
                            "0",
                            "--max-article-fetches",
                            "3",
                            "--max-fetches-per-ticker",
                            "2",
                        ],
                    )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            extractions = json.loads((output_dir / "article_extractions.json").read_text(encoding="utf-8"))["article_fetch"]

            self.assertEqual(payload["article_count"], 3)
            self.assertEqual(article_urlopen.call_count, 2)
            self.assertEqual(extractions["attempted_fetches"], 2)
            self.assertEqual(extractions["max_fetches_per_ticker"], 2)

    def test_live_article_fetch_failures_do_not_fail_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(LIVE_RSS_FIXTURE)):
                with patch("news_pipeline.article_fetch.urlopen", side_effect=TimeoutError("article timeout")):
                    stdout = _run_cli(
                        "dry-run-daily",
                        temp_dir,
                        extra_args=[
                            "--enable-live-rss",
                            "--enable-live-article-fetch",
                            "--live-rss-url",
                            "https://example.com/rss/nvda.xml",
                            "--live-rss-retries",
                            "0",
                        ],
                    )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            extractions = json.loads((output_dir / "article_extractions.json").read_text(encoding="utf-8"))["article_fetch"]
            scores = json.loads((output_dir / "sentiment_scores.json").read_text(encoding="utf-8"))["scores"]

            self.assertEqual(payload["status"], "dry_run_complete")
            self.assertEqual(extractions["attempted_fetches"], 1)
            self.assertEqual(extractions["failed_extractions"], 1)
            self.assertEqual(extractions["records"][0]["error_class"], "TimeoutError")
            self.assertEqual(scores[0]["basis"], "snippet")

    def test_report_includes_extraction_summary_and_sentiment_basis_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(LIVE_RSS_FIXTURE)):
                with patch("news_pipeline.article_fetch.urlopen", return_value=FakeArticleResponse(ARTICLE_HTML)):
                    stdout = _run_cli(
                        "dry-run-daily",
                        temp_dir,
                        extra_args=[
                            "--enable-live-rss",
                            "--enable-live-article-fetch",
                            "--live-rss-url",
                            "https://example.com/rss/nvda.xml",
                            "--live-rss-retries",
                            "0",
                        ],
                    )
            payload = json.loads(stdout)
            output_dir = Path(payload["output_dir"])
            contract = json.loads((output_dir / "report_contract.json").read_text(encoding="utf-8"))["report"]
            markdown = (output_dir / "daily_report.md").read_text(encoding="utf-8")
            html = (output_dir / "daily_report.html").read_text(encoding="utf-8")
            email_html = (output_dir / "email_preview.html").read_text(encoding="utf-8")

            self.assertEqual(contract["extraction_summary"]["article_pages_fetched"], 1)
            self.assertEqual(contract["extraction_summary"]["successful_extractions"], 1)
            self.assertEqual(contract["extraction_summary"]["sentiment_basis_counts"]["full_text"], 1)
            self.assertIn("Article Extraction Summary", markdown)
            self.assertIn("Article Extraction Summary", html)
            self.assertIn("Article Extraction Summary", email_html)
            self.assertIn("Extraction Basis", html)
            self.assertIn("Extraction Basis", email_html)


def _run_cli(
    command,
    temp_dir,
    *,
    extra_args=None,
    fake_providers=None,
    environ=None,
):
    argv = [
        command,
        "--run-date",
        "2026-06-03",
        "--artifacts-dir",
        str(Path(temp_dir) / "artifacts"),
    ]
    if extra_args:
        argv.extend(extra_args)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(argv, fake_providers=fake_providers, environ=environ or {})
    assert exit_code == 0
    return stdout.getvalue().strip()


def _database_counts(database_path):
    store = SQLiteStore(database_path)
    try:
        return {
            table: store.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
            for table in (
                "runs",
                "articles",
                "run_articles",
                "article_sources",
                "provider_usage",
                "provider_validation",
                "dedupe_clusters",
                "article_extractions",
                "sentiment_results",
                "ticker_mentions",
            )
        }
    finally:
        store.close()


if __name__ == "__main__":
    unittest.main()
