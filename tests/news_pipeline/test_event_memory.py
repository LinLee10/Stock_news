import json
from pathlib import Path
import tempfile
import unittest

from news_pipeline.article_fetch import ArticleFetchSummary
from news_pipeline.dedup import cluster_articles
from news_pipeline.event_memory import (
    EventMemoryRecord,
    PriorEventMemorySnapshot,
    alpha_vantage_selection_history,
    build_event_memory_records,
    compare_event_memory,
    write_event_memory_artifacts,
)
from news_pipeline.models import Article
from news_pipeline.storage import SQLiteStore


def _memory_record(
    ticker: str,
    canonical_url: str,
    sentiment: float,
) -> EventMemoryRecord:
    return EventMemoryRecord(
        article_id=f"{ticker}-{canonical_url.rsplit('/', 1)[-1]}",
        canonical_url=canonical_url,
        published_at="2026-06-09T12:00:00Z",
        ticker=ticker,
        company=ticker,
        source_provider="example",
        source_family="direct_news_publisher",
        article_type="company_news",
        cluster_id=f"cluster-{ticker}",
        ticker_match_confidence=1.0,
        extraction_basis="snippet",
        extraction_quality_grade="snippet",
        internal_sentiment=sentiment,
        external_sentiment_provider=None,
        external_sentiment=None,
        event_type="company_news",
        event_summary="Example event",
        run_id="current-run",
        run_date="2026-06-09",
    )


class EventMemoryTests(unittest.TestCase):
    def test_no_prior_run_reports_history_building(self):
        comparison = compare_event_memory(
            (_memory_record("NVDA", "https://example.com/current", 0.2),),
            PriorEventMemorySnapshot(available=False),
        )

        self.assertFalse(comparison.prior_run_available)
        self.assertEqual(comparison.history_status, "history_building")
        self.assertEqual(comparison.new_events_since_prior_run, 0)
        self.assertEqual(comparison.repeated_events_from_prior_run, 0)
        self.assertEqual(comparison.sentiment_change_since_prior_run, {})

    def test_prior_run_comparison_counts_new_repeated_and_sentiment_changes(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            run_id="prior-run",
            run_date="2026-06-08",
            records=(
                _memory_record(
                    "NVDA",
                    "https://example.com/repeated",
                    0.1,
                ).as_dict(),
                _memory_record(
                    "AMD",
                    "https://example.com/prior-only",
                    -0.2,
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/repeated",
                    0.4,
                ),
                _memory_record(
                    "META",
                    "https://example.com/new",
                    0.3,
                ),
            ),
            prior,
        )

        self.assertTrue(comparison.prior_run_available)
        self.assertEqual(comparison.history_status, "compared_to_prior_run")
        self.assertEqual(comparison.new_events_since_prior_run, 1)
        self.assertEqual(comparison.repeated_events_from_prior_run, 1)
        self.assertEqual(
            comparison.sentiment_change_since_prior_run["NVDA"]["change"],
            0.3,
        )

    def test_selection_history_identifies_google_dominance_and_benchmark_gaps(self):
        snapshot = PriorEventMemorySnapshot(
            available=True,
            records=(
                {
                    **_memory_record(
                        "META",
                        "https://example.com/google-1",
                        0.1,
                    ).as_dict(),
                    "source_family": "google_news_backstop",
                },
                {
                    **_memory_record(
                        "META",
                        "https://example.com/google-2",
                        0.1,
                    ).as_dict(),
                    "source_family": "google_news_backstop",
                },
                {
                    **_memory_record(
                        "META",
                        "https://example.com/direct",
                        0.1,
                    ).as_dict(),
                    "source_family": "direct_news_publisher",
                },
                {
                    **_memory_record(
                        "MU",
                        "https://example.com/benchmark",
                        0.2,
                    ).as_dict(),
                    "external_sentiment_provider": "alpha_vantage_news",
                },
            ),
        )

        history = alpha_vantage_selection_history(
            snapshot,
            minimum_articles_per_ticker=4,
        )

        self.assertEqual(history["benchmark_coverage_counts"], {"MU": 1})
        self.assertEqual(history["google_dominated_tickers"], ["META"])
        self.assertEqual(history["weak_coverage_tickers"], ["META", "MU"])

    def test_event_memory_is_written_to_sqlite_and_artifacts(self):
        article = Article(
            article_id="sec-nvda-8k",
            canonical_url="https://www.sec.gov/example/nvda-8k",
            title="NVIDIA files 8-K with the SEC",
            snippet="NVIDIA filed a material current report.",
            published_at="2026-06-09T00:00:00Z",
            metadata={
                "provider": "sec_edgar",
                "source_provider": "sec_edgar",
                "source_family": "regulatory_official",
                "symbols": ["NVDA"],
                "ticker": "NVDA",
                "filing_form_type": "8-K",
                "filing_event_type": "material_event",
                "sec_event_summary": "NVIDIA filed a material current report.",
                "ticker_match_confidence": 1.0,
            },
        )
        clusters = tuple(cluster_articles((article,)))
        records = build_event_memory_records(
            articles=(article,),
            clusters=clusters,
            article_fetch_summary=ArticleFetchSummary(enabled=False),
            article_ids_by_url={article.canonical_url: article.article_id},
            run_id="run-2026-06-09",
            run_date="2026-06-09",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].ticker, "NVDA")
        self.assertEqual(records[0].event_type, "material_event")
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path = write_event_memory_artifacts(
                records,
                output_dir=temp_dir,
            )
            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["article_id"], "sec-nvda-8k")
            self.assertTrue(Path(csv_path).exists())

            store = SQLiteStore(Path(temp_dir) / "pipeline.sqlite3")
            try:
                store.initialize_schema()
                store.record_event_memory(records[0].as_dict())
                stored = store.list_event_memory("run-2026-06-09")
            finally:
                store.close()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["event_type"], "material_event")

    def test_alpha_vantage_event_memory_uses_ticker_specific_sentiment(self):
        article = Article(
            article_id="alpha-multi-ticker",
            canonical_url="https://example.com/alpha-multi-ticker",
            title="NVIDIA and AMD expand AI chip programs",
            snippet="NVIDIA and AMD announced separate chip updates.",
            published_at="2026-06-09T12:00:00Z",
            metadata={
                "provider": "alpha_vantage_news",
                "source_provider": "alpha_vantage_news",
                "source_family": "external_market_news_api",
                "symbols": ["NVDA", "AMD"],
                "external_sentiment_provider": "alpha_vantage_news",
                "external_sentiment": 0.7,
                "ticker_sentiment": [
                    {
                        "ticker": "NVDA",
                        "ticker_sentiment_score": "0.7",
                    },
                    {
                        "ticker": "AMD",
                        "ticker_sentiment_score": "-0.2",
                    },
                ],
            },
        )
        records = build_event_memory_records(
            articles=(article,),
            clusters=tuple(cluster_articles((article,))),
            article_fetch_summary=ArticleFetchSummary(enabled=False),
            article_ids_by_url={article.canonical_url: article.article_id},
            run_id="run-2026-06-09",
            run_date="2026-06-09",
        )

        scores = {record.ticker: record.external_sentiment for record in records}
        self.assertEqual(scores["NVDA"], 0.7)
        self.assertEqual(scores["AMD"], -0.2)


if __name__ == "__main__":
    unittest.main()
