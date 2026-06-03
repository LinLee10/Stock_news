import unittest

from news_pipeline.models import Article, ProviderUsage, SentimentResult, TickerMention


class ModelTests(unittest.TestCase):
    def test_article_requires_title_and_url(self):
        article = Article(canonical_url="https://example.com/a", title="Example")

        self.assertEqual(article.canonical_url, "https://example.com/a")
        self.assertEqual(article.title, "Example")

    def test_ticker_mentions_are_uppercased(self):
        mention = TickerMention(article_id="art_1", ticker="aapl", confidence=0.8)

        self.assertEqual(mention.ticker, "AAPL")

    def test_sentiment_basis_is_validated(self):
        with self.assertRaises(ValueError):
            SentimentResult(
                article_id="art_1",
                score=0.1,
                label="positive",
                confidence=0.7,
                basis="unknown",
            )

    def test_provider_usage_rejects_negative_quota_cost(self):
        with self.assertRaises(ValueError):
            ProviderUsage(
                provider="rss",
                operation="discover",
                status="failed",
                quota_cost=-1,
            )


if __name__ == "__main__":
    unittest.main()
