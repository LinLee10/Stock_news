import unittest
from urllib.error import HTTPError

from news_pipeline.models import Article
from news_pipeline.sources.external_api import (
    balance_external_sentiment_articles,
    build_external_quality_diagnostics,
    collect_external_api_articles,
)
from news_pipeline.sources.query_planner import plan_ticker_queries
from news_pipeline.sources.source_registry import load_source_profiles
from news_pipeline.tickers import TrackedTicker


NVDA = TrackedTicker("NVDA", "NVIDIA", ("Nvidia",), "portfolio")


def _profiles():
    return {
        profile.source_id: profile
        for profile in load_source_profiles()
        if profile.source_id
        in {
            "alpha_vantage_news",
            "marketaux",
            "nyt",
            "finnhub_news",
            "gnews",
            "newsapi",
        }
    }


def _collect(*, flags, environ, total=5, budgets=None, fetch_json=None):
    return collect_external_api_articles(
        profiles=_profiles(),
        query_plans=plan_ticker_queries((NVDA,)),
        tracked_tickers=(NVDA,),
        provider_flags=flags,
        global_enabled=True,
        environ=environ,
        total_request_budget=total,
        provider_request_budgets=budgets
        or {
            "alpha_vantage_news": 1,
            "marketaux": 1,
            "nyt": 1,
            "finnhub_news": 1,
            "gnews": 1,
            "newsapi": 1,
        },
        run_date="2026-06-09",
        timeout_seconds=1,
        fetch_json=fetch_json,
    )


class ExternalApiTests(unittest.TestCase):
    def test_alpha_vantage_news_is_disabled_without_provider_flag(self):
        calls = []
        result = _collect(
            flags={},
            environ={"ALPHA_VANTAGE_KEY": "test-key"},
            fetch_json=lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result.diagnostics["external_api_provider_skipped_reasons"][
                "alpha_vantage_news"
            ],
            "provider_flag_disabled",
        )

    def test_alpha_vantage_news_requires_global_external_api_flag(self):
        calls = []
        result = collect_external_api_articles(
            profiles=_profiles(),
            query_plans=plan_ticker_queries((NVDA,)),
            tracked_tickers=(NVDA,),
            provider_flags={"alpha_vantage_news": True},
            global_enabled=False,
            environ={"ALPHA_VANTAGE_KEY": "test-key"},
            total_request_budget=5,
            provider_request_budgets={"alpha_vantage_news": 1},
            run_date="2026-06-09",
            timeout_seconds=1,
            fetch_json=lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result.diagnostics["external_api_provider_skipped_reasons"][
                "alpha_vantage_news"
            ],
            "global_external_api_flag_disabled",
        )

    def test_alpha_vantage_missing_key_skips_safely(self):
        calls = []
        result = _collect(
            flags={"alpha_vantage_news": True},
            environ={},
            fetch_json=lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result.diagnostics["alpha_vantage_news_skipped_reason"],
            "missing_api_key",
        )

    def test_alpha_vantage_response_normalizes_benchmark_metadata(self):
        calls = []

        def fetch_json(endpoint, params, headers, timeout):
            calls.append((endpoint, params, headers))
            return {
                "feed": [
                    {
                        "title": "NVIDIA demand rises with AI infrastructure spending",
                        "url": "https://publisher.example.com/nvidia-alpha-vantage",
                        "summary": "NVIDIA revenue outlook improved.",
                        "source": "Example Wire",
                        "time_published": "20260609T120000",
                        "overall_sentiment_score": "0.31",
                        "overall_sentiment_label": "Somewhat-Bullish",
                        "topics": [
                            {
                                "topic": "Technology",
                                "relevance_score": "0.9",
                            }
                        ],
                        "ticker_sentiment": [
                            {
                                "ticker": "NVDA",
                                "relevance_score": "0.82",
                                "ticker_sentiment_score": "0.44",
                                "ticker_sentiment_label": "Bullish",
                            }
                        ],
                    }
                ]
            }

        result = _collect(
            flags={"alpha_vantage_news": True},
            environ={"ALPHA_VANTAGE_KEY": "test-key"},
            fetch_json=fetch_json,
        )

        endpoint, params, headers = calls[0]
        self.assertEqual(endpoint, "https://www.alphavantage.co/query")
        self.assertEqual(params["function"], "NEWS_SENTIMENT")
        self.assertEqual(params["tickers"], "NVDA")
        self.assertEqual(params["apikey"], "test-key")
        self.assertEqual(headers, {})
        self.assertEqual(len(result.articles), 1)
        article = result.articles[0]
        self.assertEqual(article.metadata["overall_sentiment_score"], 0.31)
        self.assertEqual(article.metadata["relevance_score"], 0.82)
        self.assertEqual(
            article.metadata["ticker_sentiment"][0]["ticker_sentiment_score"],
            "0.44",
        )
        self.assertEqual(article.metadata["topics"][0]["topic"], "Technology")
        self.assertEqual(
            article.metadata["alpha_vantage_time_published"],
            "20260609T120000",
        )
        self.assertEqual(
            article.metadata["alpha_vantage_source"],
            "Example Wire",
        )
        self.assertTrue(article.metadata["benchmark_only"])
        self.assertEqual(
            result.diagnostics["alpha_vantage_news_requests_used"],
            1,
        )
        self.assertEqual(
            result.diagnostics["alpha_vantage_news_articles_returned"],
            1,
        )
        self.assertNotIn(
            "apikey",
            result.diagnostics[
                "alpha_vantage_news_effective_params_without_key"
            ],
        )

    def test_external_apis_are_disabled_without_global_flag(self):
        calls = []
        result = collect_external_api_articles(
            profiles=_profiles(),
            query_plans=plan_ticker_queries((NVDA,)),
            tracked_tickers=(NVDA,),
            provider_flags={"marketaux": True},
            global_enabled=False,
            environ={"MARKETAUX_API_KEY": "test-key"},
            total_request_budget=5,
            provider_request_budgets={"marketaux": 1},
            run_date="2026-06-09",
            timeout_seconds=1,
            fetch_json=lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result.diagnostics["external_api_provider_skipped_reasons"]["marketaux"],
            "global_external_api_flag_disabled",
        )

    def test_missing_key_skips_without_calling_provider(self):
        calls = []
        result = _collect(
            flags={"marketaux": True},
            environ={},
            fetch_json=lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result.diagnostics["external_api_provider_skipped_reasons"]["marketaux"],
            "missing_api_key",
        )

    def test_total_budget_runs_marketaux_before_later_providers(self):
        calls = []

        def fetch_json(endpoint, params, headers, timeout):
            calls.append(endpoint)
            return {
                "data": [
                    {
                        "uuid": "marketaux-1",
                        "title": "NVIDIA shares rise on AI data center demand",
                        "description": "NVIDIA stock gained as data center revenue grew.",
                        "url": "https://publisher.example.com/nvidia-ai",
                        "published_at": "2026-06-09T12:00:00Z",
                        "source": "Example Publisher",
                        "entities": [{"symbol": "NVDA"}],
                    }
                ]
            }

        result = _collect(
            flags={"marketaux": True, "nyt": True},
            environ={
                "MARKETAUX_API_KEY": "marketaux-test-key",
                "NYT_API_KEY": "nyt-test-key",
            },
            total=1,
            fetch_json=fetch_json,
        )

        self.assertEqual(calls, ["https://api.marketaux.com/v1/news/all"])
        self.assertEqual(
            result.diagnostics["external_api_requests_used_by_provider"],
            {"marketaux": 1},
        )
        self.assertEqual(result.articles[0].metadata["source_family"], "external_market_news_api")
        self.assertEqual(result.articles[0].metadata["symbols"], ["NVDA"])

    def test_nyt_context_result_is_not_ticker_coverage_without_company_match(self):
        calls = []

        def fetch_json(endpoint, params, headers, timeout):
            calls.append((endpoint, params, headers))
            return {
                "response": {
                    "docs": [
                        {
                            "_id": "nyt://article/1",
                            "headline": {"main": "Technology shares move broadly higher"},
                            "abstract": "Chip companies gained with the broader market.",
                            "web_url": "https://www.nytimes.com/example",
                            "pub_date": "2026-06-09T10:00:00Z",
                            "source": "The New York Times",
                        }
                    ]
                }
            }

        result = _collect(
            flags={"nyt": True},
            environ={"NYT_API_KEY": "nyt-test-key"},
            fetch_json=fetch_json,
        )

        endpoint, params, headers = calls[0]
        self.assertEqual(
            endpoint,
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
        )
        self.assertEqual(params["api-key"], "nyt-test-key")
        self.assertEqual(headers, {})
        self.assertEqual(len(result.articles), 1)
        article = result.articles[0]
        self.assertTrue(article.metadata["external_context_only"])
        self.assertEqual(article.metadata["symbols"], [])
        self.assertEqual(article.metadata["source_family"], "context_news_api")
        self.assertEqual(result.diagnostics["nyt_status_code"], 200)
        self.assertIsNone(result.diagnostics["nyt_error_reason"])
        self.assertEqual(
            result.diagnostics["nyt_effective_endpoint_without_key"],
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
        )
        safe_params = result.diagnostics["nyt_effective_params_without_key"]
        self.assertNotIn("api-key", safe_params)
        self.assertIn("NVIDIA", safe_params["q"])
        self.assertEqual(safe_params["begin_date"], "20260510")

    def test_nyt_malformed_documents_do_not_fail_the_request(self):
        def fetch_json(endpoint, params, headers, timeout):
            self.assertIn("articlesearch", endpoint)
            return {
                "response": {
                    "docs": [
                        {
                            "_id": "nyt://article/malformed",
                            "headline": {"main": "Malformed URL"},
                            "web_url": "https://[invalid",
                        },
                        {
                            "_id": "nyt://article/valid",
                            "headline": {"main": "NVIDIA shares rise on AI demand"},
                            "abstract": "NVIDIA stock gained after a company update.",
                            "web_url": "https://www.nytimes.com/valid-example",
                            "pub_date": "2026-06-09T10:00:00Z",
                            "source": "The New York Times",
                        },
                    ]
                }
            }

        result = _collect(
            flags={"nyt": True},
            environ={"NYT_API_KEY": "nyt-test-key"},
            fetch_json=fetch_json,
        )

        self.assertEqual(len(result.articles), 1)
        self.assertEqual(result.articles[0].article_id, "nyt://article/valid")
        self.assertEqual(
            result.diagnostics["external_api_requests_used_by_provider"],
            {"nyt": 1},
        )

    def test_nyt_null_documents_are_treated_as_an_empty_response(self):
        result = _collect(
            flags={"nyt": True},
            environ={"NYT_API_KEY": "nyt-test-key"},
            fetch_json=lambda *args: {"response": {"docs": None}},
        )

        self.assertEqual(result.articles, ())
        nyt_attempt = next(
            attempt for attempt in result.attempts if attempt.provider == "nyt"
        )
        self.assertEqual(nyt_attempt.status, "success")
        self.assertEqual(nyt_attempt.article_count, 0)
        self.assertEqual(result.diagnostics["nyt_requests_attempted"], 1)
        self.assertEqual(result.diagnostics["nyt_articles_returned"], 0)
        self.assertEqual(result.diagnostics["nyt_zero_result_queries"], 1)
        self.assertEqual(result.diagnostics["nyt_role"], "context_news_api")
        self.assertEqual(result.diagnostics["nyt_status_code"], 200)
        self.assertIsNone(result.diagnostics["nyt_error_reason"])

    def test_gnews_uses_expected_endpoint_and_safe_parameters(self):
        calls = []

        def fetch_json(endpoint, params, headers, timeout):
            calls.append((endpoint, params, headers))
            return {
                "articles": [
                    {
                        "title": "NVIDIA expands AI data center systems",
                        "description": "NVIDIA stock coverage from a direct publisher.",
                        "url": "https://publisher.example.com/nvidia-gnews",
                        "publishedAt": "2026-06-09T10:00:00Z",
                        "source": {"name": "Example News"},
                    }
                ]
            }

        result = _collect(
            flags={"gnews": True},
            environ={"GNEWS_KEY": "gnews-test-key"},
            fetch_json=fetch_json,
        )

        endpoint, params, headers = calls[0]
        self.assertEqual(endpoint, "https://gnews.io/api/v4/search")
        self.assertEqual(params["apikey"], "gnews-test-key")
        self.assertEqual(params["lang"], "en")
        self.assertEqual(params["country"], "us")
        self.assertEqual(params["max"], 10)
        self.assertIn("NVIDIA", params["q"])
        self.assertEqual(headers, {})
        self.assertEqual(result.diagnostics["gnews_status_code"], 200)
        self.assertIsNone(result.diagnostics["gnews_error_reason"])
        self.assertEqual(result.diagnostics["gnews_requests_attempted"], 1)
        self.assertEqual(result.diagnostics["gnews_articles_returned"], 1)
        self.assertEqual(
            result.diagnostics["gnews_effective_endpoint_without_key"],
            "https://gnews.io/api/v4/search",
        )
        safe_params = result.diagnostics["gnews_effective_params_without_key"]
        self.assertNotIn("apikey", safe_params)
        self.assertEqual(safe_params["country"], "us")

    def test_gnews_http_error_is_recorded_without_crashing(self):
        def fetch_json(endpoint, params, headers, timeout):
            raise HTTPError(
                endpoint,
                429,
                "quota",
                {"Retry-After": "12"},
                None,
            )

        result = _collect(
            flags={"gnews": True},
            environ={"GNEWS_KEY": "gnews-test-key"},
            fetch_json=fetch_json,
        )

        self.assertEqual(result.articles, ())
        self.assertEqual(result.diagnostics["gnews_status_code"], 429)
        self.assertEqual(result.diagnostics["gnews_error_reason"], "rate_limited")
        self.assertEqual(result.diagnostics["gnews_requests_attempted"], 1)
        self.assertEqual(result.diagnostics["gnews_rate_limited_count"], 1)
        self.assertEqual(result.diagnostics["gnews_retry_after_seconds"], 12.0)
        gnews_attempt = next(
            attempt for attempt in result.attempts if attempt.provider == "gnews"
        )
        self.assertEqual(gnews_attempt.status, "rate_limited")
        self.assertEqual(gnews_attempt.retry_after_seconds, 12.0)

    def test_provider_budget_limits_requests(self):
        calls = []

        result = _collect(
            flags={"marketaux": True},
            environ={"MARKETAUX_API_KEY": "test-key"},
            budgets={"marketaux": 0},
            fetch_json=lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result.diagnostics["external_api_provider_skipped_reasons"]["marketaux"],
            "provider_request_budget_exhausted",
        )

    def test_finnhub_gnews_and_newsapi_responses_normalize(self):
        def fetch_json(endpoint, params, headers, timeout):
            if "finnhub" in endpoint:
                return [
                    {
                        "id": 1,
                        "headline": "NVIDIA reports new AI infrastructure demand",
                        "summary": "NVDA shares rose after the company update.",
                        "url": "https://finnhub-publisher.example.com/nvda",
                        "datetime": 1781020800,
                        "source": "Example Wire",
                    }
                ]
            if "gnews" in endpoint:
                return {
                    "articles": [
                        {
                            "title": "NVIDIA expands data center partnership",
                            "description": "NVIDIA stock coverage focused on a new contract.",
                            "url": "https://gnews-publisher.example.com/nvda",
                            "publishedAt": "2026-06-09T10:00:00Z",
                            "source": {"name": "Example News"},
                        }
                    ]
                }
            return {
                "articles": [
                    {
                        "title": "NVIDIA guidance draws investor attention",
                        "description": "NVIDIA shares moved after its outlook update.",
                        "url": "https://newsapi-publisher.example.com/nvda",
                        "publishedAt": "2026-06-09T09:00:00Z",
                        "source": {"name": "Example Daily"},
                    }
                ]
            }

        result = _collect(
            flags={
                "finnhub_news": True,
                "gnews": True,
                "newsapi": True,
            },
            environ={
                "FINNHUB_KEY": "finnhub-test-key",
                "GNEWS_KEY": "gnews-test-key",
                "NEWSAPI_KEY": "newsapi-test-key",
            },
            total=3,
            fetch_json=fetch_json,
        )

        self.assertEqual(
            result.diagnostics["external_api_articles_by_provider"],
            {"finnhub_news": 1, "gnews": 1, "newsapi": 1},
        )
        self.assertEqual(len(result.articles), 3)
        self.assertTrue(all(article.metadata["external_api_used"] for article in result.articles))

    def test_post_dedup_counts_and_concentration_warnings_are_reported(self):
        post_dedup = (
            *(
                _external_article(
                    f"https://example.com/finnhub-{index}",
                    provider="finnhub_news",
                    ticker="AMD",
                )
                for index in range(4)
            ),
            _external_article(
                "https://example.com/gnews-1",
                provider="gnews",
                ticker="NVDA",
            ),
        )

        diagnostics = build_external_quality_diagnostics(
            raw_articles_by_provider={"finnhub_news": 20, "gnews": 2},
            raw_articles_by_ticker={"AMD": 18, "NVDA": 4},
            post_dedup_articles=post_dedup,
            sentiment_articles=post_dedup[:3],
        )

        self.assertEqual(
            diagnostics["raw_external_articles_by_provider"],
            {"finnhub_news": 20, "gnews": 2},
        )
        self.assertEqual(
            diagnostics["post_dedup_external_articles_by_provider"],
            {"finnhub_news": 4, "gnews": 1},
        )
        self.assertEqual(
            diagnostics["articles_used_for_sentiment_by_provider"],
            {"finnhub_news": 3},
        )
        self.assertTrue(diagnostics["provider_concentration_warning"])
        self.assertTrue(diagnostics["ticker_concentration_warning"])

    def test_balancing_limits_scoring_without_changing_raw_diagnostics(self):
        articles = tuple(
            [
                _external_article(
                    f"https://example.com/finnhub-{index}",
                    provider="finnhub_news",
                    ticker=f"T{index}",
                )
                for index in range(6)
            ]
            + [
                _external_article(
                    f"https://example.com/gnews-{index}",
                    provider="gnews",
                    ticker=f"G{index}",
                )
                for index in range(2)
            ]
        )

        balanced = balance_external_sentiment_articles(
            articles,
            max_articles_per_provider=3,
            max_articles_per_ticker=2,
            max_provider_share=0.60,
        )
        diagnostics = build_external_quality_diagnostics(
            raw_articles_by_provider={"finnhub_news": 100, "gnews": 10},
            raw_articles_by_ticker={"AMD": 90, "NVDA": 20},
            post_dedup_articles=articles,
            sentiment_articles=balanced,
        )

        self.assertEqual(
            diagnostics["raw_external_articles_by_provider"]["finnhub_news"],
            100,
        )
        self.assertEqual(
            diagnostics["post_dedup_external_articles_by_provider"]["finnhub_news"],
            6,
        )
        self.assertEqual(
            diagnostics["articles_used_for_sentiment_by_provider"],
            {"finnhub_news": 3, "gnews": 2},
        )
        self.assertLessEqual(
            max(diagnostics["articles_used_for_sentiment_by_provider"].values())
            / sum(diagnostics["articles_used_for_sentiment_by_provider"].values()),
            0.60,
        )
        dominant = next(
            article
            for article in balanced
            if article.metadata["api_provider"] == "finnhub_news"
        )
        alternative = next(
            article
            for article in balanced
            if article.metadata["api_provider"] == "gnews"
        )
        self.assertLess(dominant.metadata["dominant_provider_downweight"], 1.0)
        self.assertGreater(alternative.metadata["source_diversity_bonus"], 1.0)
        self.assertIn("direct_api_bonus", dominant.metadata)

    def test_alpha_vantage_benchmark_articles_do_not_enter_internal_scoring(self):
        benchmark = _external_article(
            "https://example.com/alpha-vantage",
            provider="alpha_vantage_news",
            ticker="NVDA",
        )
        benchmark = Article(
            canonical_url=benchmark.canonical_url,
            title=benchmark.title,
            snippet=benchmark.snippet,
            published_at=benchmark.published_at,
            metadata={**benchmark.metadata, "benchmark_only": True},
        )

        self.assertEqual(
            balance_external_sentiment_articles((benchmark,)),
            (),
        )


def _external_article(url, *, provider, ticker):
    return Article(
        canonical_url=url,
        title=f"{ticker} company news",
        snippet=f"{ticker} stock update.",
        published_at="2026-06-09T10:00:00Z",
        metadata={
            "provider": provider,
            "source_id": provider,
            "api_provider": provider,
            "external_api_used": True,
            "symbols": [ticker],
            "acquisition_score": 80.0,
        },
    )


if __name__ == "__main__":
    unittest.main()
