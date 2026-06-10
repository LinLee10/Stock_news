import unittest

from news_pipeline.provider_registry import get_provider_config, iter_provider_configs
from news_pipeline.provider_validation import (
    ProviderCheckResult,
    redact_key,
    validate_provider,
)


class FakeChecker:
    def check(self, config):
        return ProviderCheckResult(
            status="fake_ok",
            remaining_quota=123,
            quota_truth_source="fake_response",
        )


class ProviderValidationTests(unittest.TestCase):
    def test_registry_contains_expected_providers(self):
        names = {config.name for config in iter_provider_configs()}

        self.assertEqual(
            names,
            {
                "google_news_rss_search",
                "yahoo_finance_rss",
                "cnbc_rss",
                "marketwatch_rss",
                "sec_edgar",
                "marketaux",
                "nyt",
                "gnews",
                "finnhub_news",
                "newsapi",
                "resend",
            },
        )

    def test_rss_provider_requires_no_key(self):
        result = validate_provider("google_news_rss_search", environ={})

        self.assertEqual(result.key_state, "not_required")
        self.assertEqual(result.last_status, "dry_run_ok")

    def test_keyed_provider_records_presence_only(self):
        secret = "secret-nyt-value"
        result = validate_provider("nyt", environ={"NYT_API_KEY": secret})
        safe = result.as_safe_dict()

        self.assertEqual(result.key_state, "present")
        self.assertNotIn(secret, repr(result))
        self.assertNotIn(secret, repr(safe))
        self.assertNotIn("ALPHA_VANTAGE_KEY", safe)

    def test_missing_key_is_reported_without_value(self):
        result = validate_provider("marketaux", environ={})

        self.assertEqual(result.key_state, "missing")
        self.assertEqual(result.last_status, "missing_key")

    def test_redact_key_never_returns_original(self):
        self.assertEqual(redact_key("secret"), "<redacted>")
        self.assertEqual(redact_key(None), "<missing>")

    def test_fake_checker_supports_quota_status(self):
        result = validate_provider(
            get_provider_config("gnews"),
            environ={"GNEWS_KEY": "secret-gnews-value"},
            checker=FakeChecker(),
        )

        self.assertEqual(result.last_status, "fake_ok")
        self.assertEqual(result.remaining_quota, 123)
        self.assertEqual(result.quota_truth_source, "fake_response")


if __name__ == "__main__":
    unittest.main()
