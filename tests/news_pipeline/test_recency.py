import unittest

from news_pipeline.cli import _cluster_to_dict, _sentiment_row_from_stored
from news_pipeline.dedup import cluster_articles
from news_pipeline.models import Article, SentimentResult
from news_pipeline.recency import article_recency, recency_bucket_for_age, recency_weight
from news_pipeline.reporting import EventClusterRow
from news_pipeline.tickers import ticker_lookup


class RecencyTests(unittest.TestCase):
    def test_bucket_assignment_for_fixed_dates(self):
        run_date = "2026-06-04"

        self.assertEqual(
            article_recency(run_date=run_date, published_at="2026-06-03T01:00:00+00:00", collected_at=None).recency_bucket,
            "today_signal",
        )
        self.assertEqual(
            article_recency(run_date=run_date, published_at="2026-06-02T00:00:00+00:00", collected_at=None).recency_bucket,
            "recent_pulse",
        )
        self.assertEqual(
            article_recency(run_date=run_date, published_at="2026-05-30T00:00:00+00:00", collected_at=None).recency_bucket,
            "weekly_trend",
        )
        self.assertEqual(
            article_recency(run_date=run_date, published_at="2026-05-20T00:00:00+00:00", collected_at=None).recency_bucket,
            "background_context",
        )
        self.assertEqual(recency_bucket_for_age(721), "archive_context")

    def test_time_decay_weighting(self):
        self.assertEqual(recency_weight("today_signal"), 1.0)
        self.assertEqual(recency_weight("recent_pulse"), 0.7)
        self.assertEqual(recency_weight("weekly_trend"), 0.4)
        self.assertEqual(recency_weight("background_context"), 0.15)

    def test_missing_or_malformed_dates_fall_back_to_collected_at(self):
        recency = article_recency(
            run_date="2026-06-04",
            published_at="not-a-date",
            collected_at="2026-06-03T12:00:00+00:00",
        )

        self.assertEqual(recency.source, "collected_at")
        self.assertEqual(recency.recency_bucket, "today_signal")
        self.assertEqual(recency.article_age_hours, 12.0)

    def test_ticker_level_recency_metrics(self):
        ticker = ticker_lookup()["NVDA"]
        row = _sentiment_row_from_stored(
            ticker,
            (
                _item("today_signal", 0.6, "Yahoo Finance"),
                _item("recent_pulse", 0.2, "CNBC"),
                _item("weekly_trend", -0.1, "MarketWatch"),
                _item("background_context", 0.1, "Yahoo Finance"),
            ),
            (
                EventClusterRow("NVDA", "NVIDIA event", "https://example.com/event", 2, 2, 3),
            ),
        )

        self.assertEqual(row.article_count_24h, 1)
        self.assertEqual(row.article_count_3d, 2)
        self.assertEqual(row.article_count_7d, 3)
        self.assertEqual(row.article_count_30d, 4)
        self.assertEqual(row.source_diversity, 3)
        self.assertEqual(row.today_signal_sentiment, 0.6)
        self.assertEqual(row.recent_pulse_sentiment, 0.2)
        self.assertEqual(row.weekly_trend_sentiment, -0.1)
        self.assertEqual(row.background_context_sentiment, 0.1)
        self.assertEqual(row.mention_velocity, "limited_history")
        self.assertEqual(row.top_event_clusters[0].title, "NVIDIA event")

    def test_event_cluster_recency_fields(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://example.com/nvda-a",
                    title="NVIDIA announces AI chip",
                    snippet="NVDA chip update",
                    published_at="2026-06-03T10:00:00+00:00",
                    metadata={"provider": "yahoo_finance_rss", "source_name": "Yahoo Finance"},
                    created_at="2026-06-03T11:00:00+00:00",
                ),
                Article(
                    canonical_url="https://example.com/nvda-b",
                    title="NVIDIA announces AI chips",
                    snippet="NVDA chip update",
                    published_at="2026-06-02T10:00:00+00:00",
                    metadata={"provider": "cnbc_rss", "source_name": "CNBC"},
                    created_at="2026-06-02T11:00:00+00:00",
                ),
            ]
        )

        payload = _cluster_to_dict(clusters[0], "2026-06-04")

        self.assertEqual(payload["first_seen_at"], "2026-06-02T10:00:00+00:00")
        self.assertEqual(payload["latest_seen_at"], "2026-06-03T10:00:00+00:00")
        self.assertEqual(payload["primary_published_at"], "2026-06-03T10:00:00+00:00")
        self.assertEqual(payload["recency_bucket"], "today_signal")
        self.assertIn("NVDA", payload["tickers_mentioned"])
        self.assertIsNotNone(payload["weighted_cluster_sentiment"])

    def test_background_context_does_not_dominate_today_signal(self):
        ticker = ticker_lookup()["NVDA"]
        row = _sentiment_row_from_stored(
            ticker,
            (
                _item("today_signal", 1.0, "Source A"),
                _item("background_context", -1.0, "Source B"),
            ),
            (),
        )

        self.assertGreater(row.weighted_sentiment_score, 0.0)
        self.assertEqual(row.today_signal_sentiment, 1.0)
        self.assertEqual(row.background_context_sentiment, -1.0)


def _item(bucket, score, source_name):
    return {
        "score": SentimentResult("article-id", score, "positive" if score >= 0 else "negative", 0.5, "snippet"),
        "recency": article_recency(
            run_date="2026-06-04",
            published_at=_published_at_for_bucket(bucket),
            collected_at=None,
        ),
        "source": {"source_name": source_name},
    }


def _published_at_for_bucket(bucket):
    return {
        "today_signal": "2026-06-03T12:00:00+00:00",
        "recent_pulse": "2026-06-02T00:00:00+00:00",
        "weekly_trend": "2026-05-30T00:00:00+00:00",
        "background_context": "2026-05-20T00:00:00+00:00",
    }[bucket]


if __name__ == "__main__":
    unittest.main()
