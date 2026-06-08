import unittest

from news_pipeline.models import Article
from news_pipeline.ticker_matching import HIGH, LOW, assess_ticker_matches


class TickerMatchConfidenceTests(unittest.TestCase):
    def test_company_specific_title_is_high_confidence(self):
        article = Article(
            canonical_url="https://example.com/nvidia",
            title="NVIDIA raises guidance on AI chip demand",
        )

        match = next(item for item in assess_ticker_matches(article) if item.ticker == "NVDA")

        self.assertEqual(match.confidence_label, HIGH)
        self.assertTrue(match.primary)

    def test_related_ticker_mention_is_low_confidence(self):
        article = Article(
            canonical_url="https://example.com/marvell",
            title="Marvell reports quarterly earnings",
            snippet="NVIDIA was mentioned as another AI chip supplier.",
        )

        match = next(item for item in assess_ticker_matches(article) if item.ticker == "NVDA")

        self.assertEqual(match.confidence_label, LOW)
        self.assertTrue(match.related)


if __name__ == "__main__":
    unittest.main()
