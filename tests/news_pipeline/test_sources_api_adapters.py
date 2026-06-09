import unittest

from news_pipeline.sources.alpha_vantage import AlphaVantageSource
from news_pipeline.sources.gnews import GNewsSource
from news_pipeline.sources.finnhub import FinnhubNewsSource
from news_pipeline.sources.marketaux import MarketauxSource


class ApiAdapterTests(unittest.TestCase):
    def test_alpha_vantage_news_sentiment_and_calendar_metadata(self):
        source = AlphaVantageSource(
            {
                "feed": [
                    {
                        "title": "Nvidia reports record growth",
                        "url": "https://example.com/nvda?utm_source=av",
                        "summary": "Chip demand remains strong.",
                        "source": "Example",
                        "time_published": "20240115T100000",
                        "overall_sentiment_score": 0.42,
                        "overall_sentiment_label": "Bullish",
                        "ticker_sentiment": [{"ticker": "NVDA", "ticker_sentiment_score": "0.5"}],
                        "calendar": {"event": "earnings"},
                    }
                ]
            }
        )

        article = source.articles()[0]

        self.assertEqual(article.canonical_url, "https://example.com/nvda")
        self.assertEqual(article.metadata["symbols"], ["NVDA"])
        self.assertEqual(article.metadata["raw_metadata"]["overall_sentiment_label"], "Bullish")
        self.assertEqual(article.metadata["raw_metadata"]["calendar"], {"event": "earnings"})

    def test_marketaux_entity_sentiment_and_similar_articles(self):
        source = MarketauxSource(
            {
                "data": [
                    {
                        "uuid": "m1",
                        "title": "Tesla shares surge",
                        "url": "https://example.com/tsla",
                        "description": "Shares moved higher.",
                        "published_at": "2024-01-15T10:00:00Z",
                        "source": {"name": "Marketaux Wire"},
                        "entities": [
                            {"symbol": "TSLA", "sentiment_score": 0.71, "sentiment": "positive"}
                        ],
                        "similar": [{"uuid": "m2", "url": "https://example.com/tsla-copy"}],
                    }
                ]
            }
        )

        article = source.articles()[0]

        self.assertEqual(article.metadata["provider_article_id"], "m1")
        self.assertEqual(article.metadata["source_name"], "Marketaux Wire")
        self.assertEqual(article.metadata["symbols"], ["TSLA"])
        self.assertEqual(
            article.metadata["raw_metadata"]["entity_sentiment"],
            [{"symbol": "TSLA", "sentiment_score": 0.71, "sentiment": "positive"}],
        )
        self.assertEqual(article.metadata["raw_metadata"]["similar_articles"][0]["uuid"], "m2")

    def test_gnews_full_content_is_preserved(self):
        source = GNewsSource(
            {
                "articles": [
                    {
                        "title": "Microsoft announces product update",
                        "url": "https://example.com/msft",
                        "description": "Short summary.",
                        "content": "Full article content from mocked GNews response.",
                        "publishedAt": "2024-01-15T10:00:00Z",
                        "source": {"name": "GNews Source"},
                    }
                ]
            }
        )

        article = source.articles()[0]

        self.assertEqual(article.full_text, "Full article content from mocked GNews response.")
        self.assertEqual(article.snippet, "Short summary.")
        self.assertEqual(article.metadata["source_name"], "GNews Source")

    def test_finnhub_company_news_is_normalized_from_mocked_payload(self):
        source = FinnhubNewsSource(
            {
                "NVDA": [
                    {
                        "id": 7,
                        "headline": "NVIDIA launches a new AI platform",
                        "url": "https://example.com/nvda-platform",
                        "summary": "The company announced a product update.",
                        "source": "Example",
                        "datetime": 1781000000,
                        "category": "company",
                        "related": "NVDA",
                    }
                ]
            }
        )

        article = source.articles(("NVDA",))[0]

        self.assertEqual(article.metadata["provider"], "finnhub_news")
        self.assertEqual(article.metadata["symbols"], ["NVDA"])
        self.assertEqual(article.metadata["provider_article_id"], "7")


if __name__ == "__main__":
    unittest.main()
