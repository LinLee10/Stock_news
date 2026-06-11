import unittest

from news_pipeline.benchmarking import build_alpha_vantage_benchmarks
from news_pipeline.models import Article
from news_pipeline.sentiment_coverage import TickerSentimentCoverage


class BenchmarkingTests(unittest.TestCase):
    def test_internal_and_alpha_vantage_sentiment_are_compared(self):
        article = Article(
            canonical_url="https://example.com/nvda-benchmark",
            title="NVIDIA outlook benchmark",
            snippet="NVIDIA faces a severe loss and declining demand.",
            published_at="2026-06-09T12:00:00Z",
            metadata={
                "api_provider": "alpha_vantage_news",
                "ticker_sentiment": [
                    {
                        "ticker": "NVDA",
                        "relevance_score": "0.8",
                        "ticker_sentiment_score": "0.75",
                        "ticker_sentiment_label": "Bullish",
                    }
                ],
            },
        )
        coverage = {
            "NVDA": TickerSentimentCoverage(
                ticker="NVDA",
                article_count_scored=2,
                full_text_scored_count=1,
                snippet_scored_count=1,
                title_scored_count=0,
                weighted_sentiment=-0.2,
                positive_article_count=0,
                negative_article_count=2,
                neutral_article_count=0,
                high_confidence_article_count=2,
                low_confidence_article_count=0,
                top_positive_cluster=None,
                top_negative_cluster="cluster-1",
                sentiment_coverage_grade="good",
            )
        }

        rows, updated = build_alpha_vantage_benchmarks(
            articles=(article,),
            ticker_coverage=coverage,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].external_alpha_vantage_sentiment, 0.75)
        self.assertEqual(rows[0].external_alpha_vantage_label, "Bullish")
        self.assertTrue(rows[0].sentiment_disagreement_flag)
        self.assertEqual(
            rows[0].sentiment_disagreement_reason,
            "direction_mismatch",
        )
        self.assertEqual(updated["NVDA"].internal_weighted_sentiment, -0.2)
        self.assertEqual(
            updated["NVDA"].alpha_vantage_weighted_sentiment,
            0.75,
        )
        self.assertEqual(updated["NVDA"].benchmark_coverage_count, 1)
        self.assertEqual(updated["NVDA"].benchmark_disagreement_count, 1)
        self.assertEqual(updated["NVDA"].benchmark_alignment_grade, "divergent")


if __name__ == "__main__":
    unittest.main()
