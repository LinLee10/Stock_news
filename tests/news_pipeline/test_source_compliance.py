import unittest

from news_pipeline.sources.compliance import collection_decision, extraction_decision
from news_pipeline.sources.source_registry import DIRECT_NEWS_PUBLISHER, SourceProfile


class SourceComplianceTests(unittest.TestCase):
    def test_profile_fetch_and_extract_flags_are_enforced(self):
        profile = SourceProfile(
            source_id="disabled",
            source_family=DIRECT_NEWS_PUBLISHER,
            publisher_name="Disabled",
            domain="disabled.example.com",
            source_quality_tier=2,
            enabled_by_default=True,
            discovery_methods=("rss",),
            feed_urls=("https://disabled.example.com/rss",),
            fetch_allowed=False,
            extract_allowed=False,
        )

        collection = collection_decision(
            profile,
            discovery_method="rss",
            url=profile.feed_urls[0],
            user_agent="UnitTest/1.0",
        )
        extraction = extraction_decision(profile)

        self.assertFalse(collection.allowed)
        self.assertEqual(collection.reason, "fetch_disabled_by_profile")
        self.assertFalse(extraction.allowed)
        self.assertEqual(extraction.reason, "extraction_disabled_by_profile")


if __name__ == "__main__":
    unittest.main()
