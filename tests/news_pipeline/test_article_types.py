import unittest

from news_pipeline.article_types import (
    ANALYST_RATING_OR_PRICE_TARGET,
    EARNINGS_OR_RESULTS,
    GENERIC_BUY_SELL_HOLD_OPINION,
    MACRO_OR_SECTOR_ROUNDUP,
    PRODUCT_OR_AI_OR_CHIP_NEWS,
    REGULATORY_OR_LEGAL,
    STOCK_PRICE_MOVE,
    classify_article_type,
)
from news_pipeline.models import Article


class ArticleTypeTests(unittest.TestCase):
    def test_classifies_required_article_types(self):
        cases = (
            ("NVIDIA reports quarterly earnings and revenue growth", EARNINGS_OR_RESULTS),
            ("Analyst upgrades NVIDIA and raises price target", ANALYST_RATING_OR_PRICE_TARGET),
            ("NVIDIA stock falls after customer warning", STOCK_PRICE_MOVE),
            ("NVIDIA unveils new AI chip platform", PRODUCT_OR_AI_OR_CHIP_NEWS),
            ("NVIDIA faces antitrust lawsuit in court", REGULATORY_OR_LEGAL),
            ("Is NVIDIA a good stock to buy now?", GENERIC_BUY_SELL_HOLD_OPINION),
            ("NVIDIA, AMD and Marvell stocks to watch this week", MACRO_OR_SECTOR_ROUNDUP),
        )
        for title, expected in cases:
            with self.subTest(title=title):
                article = Article(canonical_url="https://example.com/story", title=title)
                self.assertEqual(classify_article_type(article).primary_type, expected)


if __name__ == "__main__":
    unittest.main()
