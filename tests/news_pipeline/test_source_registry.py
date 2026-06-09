import unittest

from news_pipeline.sources.source_registry import (
    GOOGLE_NEWS_BACKSTOP,
    PAID_NEWS_API,
    REGULATORY_OFFICIAL,
    SOURCE_FAMILY_ORDER,
    load_source_profiles,
    source_profile_by_id,
)


class SourceRegistryTests(unittest.TestCase):
    def test_registry_loads_requested_source_profiles(self):
        profiles = load_source_profiles()
        source_ids = {profile.source_id for profile in profiles}

        self.assertTrue(
            {
                "sec_edgar",
                "pr_newswire",
                "business_wire",
                "globenewswire",
                "reuters",
                "cnbc",
                "marketwatch",
                "yahoo_finance",
                "investors_business_daily",
                "barrons",
                "marketbeat",
                "investing_com",
                "barchart",
                "seeking_alpha",
                "motley_fool",
                "zacks",
                "stockstory",
                "marketaux",
                "finnhub_news",
                "alpha_vantage",
                "gnews",
                "google_news_rss_search",
            }.issubset(source_ids)
        )
        self.assertEqual(source_profile_by_id("sec_edgar").source_family, REGULATORY_OFFICIAL)
        self.assertEqual(
            source_profile_by_id("google_news_rss_search").source_family,
            GOOGLE_NEWS_BACKSTOP,
        )
        self.assertEqual(source_profile_by_id("marketaux").source_family, PAID_NEWS_API)

    def test_scheduler_family_order_places_google_last(self):
        self.assertEqual(SOURCE_FAMILY_ORDER[0], REGULATORY_OFFICIAL)
        self.assertEqual(SOURCE_FAMILY_ORDER[-1], GOOGLE_NEWS_BACKSTOP)


if __name__ == "__main__":
    unittest.main()
