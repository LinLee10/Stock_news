import unittest

from news_pipeline.provider_usage import ProviderUsageRecorder
from news_pipeline.storage import SQLiteStore


class ProviderUsageTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore(":memory:")
        self.store.initialize_schema()

    def tearDown(self):
        self.store.close()

    def test_schema_contains_expected_tables(self):
        self.assertTrue(
            {
                "articles",
                "article_sources",
                "ticker_mentions",
                "sentiment_results",
                "provider_usage",
                "runs",
            }.issubset(self.store.table_names())
        )

    def test_provider_usage_recording_is_local(self):
        recorder = ProviderUsageRecorder(self.store)

        row_id = recorder.record(
            "rss",
            "discover",
            "success",
            quota_cost=0,
            article_count=3,
            latency_ms=12,
            metadata={"feed": "local-fixture"},
        )
        rows = self.store.list_provider_usage()

        self.assertEqual(row_id, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "rss")
        self.assertEqual(rows[0]["operation"], "discover")
        self.assertEqual(rows[0]["quota_cost"], 0)
        self.assertEqual(rows[0]["article_count"], 3)


if __name__ == "__main__":
    unittest.main()
