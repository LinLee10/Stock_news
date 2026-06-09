import unittest

from news_pipeline.models import Article
from news_pipeline.sources.acquisition_scoring import source_diversity_metrics
from news_pipeline.sources.live_rss import LiveRssAttempt
from news_pipeline.sources.sec_edgar import SecCollectionAttempt
from news_pipeline.sources.source_registry import (
    COMPANY_IR,
    DIRECT_NEWS_PUBLISHER,
    GOOGLE_NEWS_BACKSTOP,
    MARKET_DATA_OR_ANALYSIS,
    PAID_NEWS_API,
    PRESS_RELEASE_WIRE,
    REGULATORY_OFFICIAL,
    CompanyIrProfile,
    SourceProfile,
)
from news_pipeline.sources.source_scheduler import schedule_sources
from news_pipeline.tickers import TrackedTicker


NVDA = TrackedTicker("NVDA", "NVIDIA", ("Nvidia",), "portfolio")


def _profile(source_id, family, *, paid=False, key=None):
    return SourceProfile(
        source_id=source_id,
        source_family=family,
        publisher_name=source_id,
        domain=f"{source_id}.example.com",
        source_quality_tier=1 if family != PAID_NEWS_API else 2,
        enabled_by_default=not paid,
        paid_required=paid,
        api_key_env_var=key,
        discovery_methods=("rss",) if not paid else ("api",),
        feed_urls=(f"https://{source_id}.example.com/rss",) if not paid else (),
        source_priority=80,
        extraction_priority=80,
    )


def _article(source_id, family, index=0):
    return Article(
        canonical_url=f"https://{source_id}.example.com/nvda-{index}",
        title=f"NVIDIA earnings update {index}",
        snippet="NVDA stock revenue and guidance update.",
        metadata={
            "provider": source_id,
            "source_id": source_id,
            "source_name": source_id,
            "source_family": family,
        },
    )


class SourceSchedulerTests(unittest.TestCase):
    def test_scheduler_runs_source_families_in_priority_order(self):
        profiles = (
            _profile("sec_edgar", REGULATORY_OFFICIAL),
            _profile("company_ir_configured", COMPANY_IR),
            _profile("wire", PRESS_RELEASE_WIRE),
            _profile("direct", DIRECT_NEWS_PUBLISHER),
            _profile("analysis", MARKET_DATA_OR_ANALYSIS),
            _profile("google_news_rss_search", GOOGLE_NEWS_BACKSTOP),
        )

        def fake_sec(*args, **kwargs):
            return [_article("sec_edgar", REGULATORY_OFFICIAL)], [
                SecCollectionAttempt("NVDA", "success", 1)
            ]

        result = schedule_sources(
            profiles=profiles,
            tracked_tickers=(NVDA,),
            company_ir_profiles={
                "NVDA": CompanyIrProfile(
                    "NVDA",
                    "NVIDIA",
                    ir_rss_url="https://ir.nvidia.example.com/rss",
                )
            },
            run_date="2026-06-09",
            user_agent="test",
            timeout_seconds=1,
            retries=0,
            target_backend_articles=20,
            minimum_backend_articles=10,
            max_backend_articles=30,
            max_articles_per_source=30,
            max_articles_per_ticker=30,
            max_google_news_share=0.5,
            include_press_release_feeds=True,
            include_sec_feeds=True,
            paid_api_global_enabled=False,
            paid_provider_flags={},
            environ={},
            rss_collector=_fake_rss_collector,
            sec_collector=fake_sec,
        )

        ordered_families = [attempt.source_family for attempt in result.attempts]
        self.assertEqual(
            ordered_families,
            [
                REGULATORY_OFFICIAL,
                COMPANY_IR,
                PRESS_RELEASE_WIRE,
                DIRECT_NEWS_PUBLISHER,
                MARKET_DATA_OR_ANALYSIS,
                GOOGLE_NEWS_BACKSTOP,
            ],
        )
        wire = next(
            article
            for article in result.articles
            if article.metadata["source_family"] == PRESS_RELEASE_WIRE
        )
        self.assertTrue(wire.metadata["issuer_promotional"])

    def test_google_fills_recall_gap_but_is_capped_after_target(self):
        profiles = (
            _profile("direct", DIRECT_NEWS_PUBLISHER),
            _profile("google_news_rss_search", GOOGLE_NEWS_BACKSTOP),
        )

        def many_articles(**kwargs):
            family = kwargs["source_families"][0]
            count = 8 if family.category == DIRECT_NEWS_PUBLISHER else 20
            articles = [
                _article(family.name, family.category, index)
                for index in range(count)
            ]
            attempt = LiveRssAttempt(
                provider=family.name,
                source_family=family.category,
                feed_id=f"{family.name}_1",
                feed_url="https://example.com/rss",
                status="success",
                article_count=count,
                fetched_article_count=count,
                latency_ms=1,
                attempts=1,
            )
            return articles[: kwargs["max_total_articles"]], [attempt]

        result = schedule_sources(
            profiles=profiles,
            tracked_tickers=(NVDA,),
            company_ir_profiles={},
            run_date="2026-06-09",
            user_agent="test",
            timeout_seconds=1,
            retries=0,
            target_backend_articles=5,
            minimum_backend_articles=3,
            max_backend_articles=10,
            max_articles_per_source=30,
            max_articles_per_ticker=30,
            max_google_news_share=0.2,
            include_press_release_feeds=True,
            include_sec_feeds=False,
            paid_api_global_enabled=False,
            paid_provider_flags={},
            environ={},
            rss_collector=many_articles,
        )

        self.assertEqual(len(result.articles), 10)
        self.assertEqual(result.diagnostics["google_news_backstop_count"], 2)
        self.assertEqual(result.diagnostics["google_news_share"], 0.2)
        self.assertTrue(result.diagnostics["google_news_share_capped"])
        self.assertFalse(
            result.diagnostics["google_news_share_cap_relaxed_for_minimum"]
        )

    def test_google_cap_is_reported_as_relaxed_below_minimum_target(self):
        profiles = (
            _profile("direct", DIRECT_NEWS_PUBLISHER),
            _profile("google_news_rss_search", GOOGLE_NEWS_BACKSTOP),
        )

        def sparse_direct(**kwargs):
            family = kwargs["source_families"][0]
            count = 1 if family.category == DIRECT_NEWS_PUBLISHER else 10
            articles = [
                _article(family.name, family.category, index)
                for index in range(count)
            ]
            return articles[: kwargs["max_total_articles"]], [
                LiveRssAttempt(
                    provider=family.name,
                    source_family=family.category,
                    feed_id=f"{family.name}_1",
                    feed_url="https://example.com/rss",
                    status="success",
                    article_count=count,
                    fetched_article_count=count,
                    latency_ms=1,
                    attempts=1,
                )
            ]

        result = schedule_sources(
            profiles=profiles,
            tracked_tickers=(NVDA,),
            company_ir_profiles={},
            run_date="2026-06-09",
            user_agent="test",
            timeout_seconds=1,
            retries=0,
            target_backend_articles=8,
            minimum_backend_articles=8,
            max_backend_articles=10,
            max_articles_per_source=20,
            max_articles_per_ticker=20,
            max_google_news_share=0.5,
            include_press_release_feeds=True,
            include_sec_feeds=False,
            paid_api_global_enabled=False,
            paid_provider_flags={},
            environ={},
            rss_collector=sparse_direct,
        )

        self.assertEqual(len(result.articles), 8)
        self.assertFalse(result.diagnostics["google_news_share_capped"])
        self.assertTrue(
            result.diagnostics["google_news_share_cap_relaxed_for_minimum"]
        )

    def test_missing_company_ir_profiles_are_reported_without_failure(self):
        result = schedule_sources(
            profiles=(),
            tracked_tickers=(NVDA,),
            company_ir_profiles={},
            run_date="2026-06-09",
            user_agent="test",
            timeout_seconds=1,
            retries=0,
            target_backend_articles=5,
            minimum_backend_articles=3,
            max_backend_articles=10,
            max_articles_per_source=10,
            max_articles_per_ticker=10,
            max_google_news_share=0.5,
            include_press_release_feeds=True,
            include_sec_feeds=False,
            paid_api_global_enabled=False,
            paid_provider_flags={},
            environ={},
        )

        self.assertEqual(result.articles, ())
        self.assertEqual(result.diagnostics["missing_company_ir_profiles"], ["NVDA"])

    def test_fetch_disabled_profile_is_not_collected(self):
        profile = SourceProfile(
            **{
                **_profile("disabled", DIRECT_NEWS_PUBLISHER).__dict__,
                "fetch_allowed": False,
            }
        )
        calls = []

        def collector(**kwargs):
            calls.append(kwargs)
            return [], []

        result = schedule_sources(
            profiles=(profile,),
            tracked_tickers=(NVDA,),
            company_ir_profiles={},
            run_date="2026-06-09",
            user_agent="test",
            timeout_seconds=1,
            retries=0,
            target_backend_articles=5,
            minimum_backend_articles=3,
            max_backend_articles=10,
            max_articles_per_source=10,
            max_articles_per_ticker=10,
            max_google_news_share=0.5,
            include_press_release_feeds=True,
            include_sec_feeds=False,
            paid_api_global_enabled=False,
            paid_provider_flags={},
            environ={},
            rss_collector=collector,
        )

        self.assertEqual(calls, [])
        self.assertEqual(result.attempts[0].status, "skipped")
        self.assertEqual(
            result.attempts[0].metadata["reason"],
            "fetch_disabled_by_profile",
        )

    def test_paid_provider_requires_all_guards_before_collector_runs(self):
        paid = _profile(
            "marketaux",
            PAID_NEWS_API,
            paid=True,
            key="MARKETAUX_API_KEY",
        )
        calls = []

        def collector(tickers):
            calls.append(tuple(ticker.symbol for ticker in tickers))
            return [_article("marketaux", PAID_NEWS_API)]

        disabled = _schedule_paid(
            paid,
            global_enabled=False,
            provider_enabled=True,
            environ={"MARKETAUX_API_KEY": "test-value"},
            collector=collector,
        )
        self.assertEqual(calls, [])
        self.assertEqual(
            disabled.diagnostics["paid_api_skipped_reasons"]["marketaux"],
            "global_paid_api_flag_disabled",
        )

        missing_key = _schedule_paid(
            paid,
            global_enabled=True,
            provider_enabled=True,
            environ={},
            collector=collector,
        )
        self.assertEqual(calls, [])
        self.assertEqual(
            missing_key.diagnostics["paid_api_skipped_reasons"]["marketaux"],
            "missing_api_key",
        )

        enabled = _schedule_paid(
            paid,
            global_enabled=True,
            provider_enabled=True,
            environ={"MARKETAUX_API_KEY": "test-value"},
            collector=collector,
        )
        self.assertEqual(calls, [("NVDA",)])
        self.assertEqual(enabled.diagnostics["paid_api_count"], 1)

    def test_source_diversity_score_increases_with_more_publishers(self):
        one_source = [
            _article("google_news_rss_search", GOOGLE_NEWS_BACKSTOP, index)
            for index in range(3)
        ]
        mixed = [
            _article("direct_one", DIRECT_NEWS_PUBLISHER, 1),
            _article("direct_two", DIRECT_NEWS_PUBLISHER, 2),
            _article("wire", PRESS_RELEASE_WIRE, 3),
        ]

        self.assertGreater(
            source_diversity_metrics(mixed)["source_diversity_score"],
            source_diversity_metrics(one_source)["source_diversity_score"],
        )


def _fake_rss_collector(**kwargs):
    family = kwargs["source_families"][0]
    article = _article(family.name, family.category)
    attempt = LiveRssAttempt(
        provider=family.name,
        source_family=family.category,
        feed_id=f"{family.name}_1",
        feed_url="https://example.com/rss",
        status="success",
        article_count=1,
        fetched_article_count=1,
        latency_ms=1,
        attempts=1,
    )
    return [article], [attempt]


def _schedule_paid(profile, *, global_enabled, provider_enabled, environ, collector):
    return schedule_sources(
        profiles=(profile,),
        tracked_tickers=(NVDA,),
        company_ir_profiles={},
        run_date="2026-06-09",
        user_agent="test",
        timeout_seconds=1,
        retries=0,
        target_backend_articles=5,
        minimum_backend_articles=3,
        max_backend_articles=10,
        max_articles_per_source=10,
        max_articles_per_ticker=10,
        max_google_news_share=0.5,
        include_press_release_feeds=True,
        include_sec_feeds=False,
        paid_api_global_enabled=global_enabled,
        paid_provider_flags={"marketaux": provider_enabled},
        environ=environ,
        paid_collectors={"marketaux": collector},
    )


if __name__ == "__main__":
    unittest.main()
