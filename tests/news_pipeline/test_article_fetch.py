import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from news_pipeline.article_fetch import (
    URL_CLASS_DIRECT_PUBLISHER,
    URL_CLASS_GOOGLE_NEWS_WRAPPER,
    URL_CLASS_UNSUPPORTED,
    build_extraction_queue,
    classify_article_url,
    fetch_top_cluster_articles,
)
from news_pipeline.dedup import DedupeCluster, SourceLink, cluster_articles
from news_pipeline.models import Article


ARTICLE_PARAGRAPHS = (
    "NVIDIA reported quarterly revenue growth as demand for artificial intelligence infrastructure remained strong.",
    "The company said data center customers continued ordering accelerated computing systems during the period.",
    "Executives told investors that supply improved while demand for newer chips remained above available capacity.",
    "NVIDIA shares moved after the report because revenue and guidance exceeded the estimates cited by analysts.",
    "Management announced additional product shipments and described investment in software and networking capacity.",
    "The company expects customer demand to remain a central driver while it expands production with its partners.",
)
ARTICLE_HTML = f"""<!doctype html>
<html>
  <head><title>NVIDIA article page</title></head>
  <body>
    <article>{''.join(f'<p>{paragraph}</p>' for paragraph in ARTICLE_PARAGRAPHS)}</article>
  </body>
</html>
"""

EMPTY_HTML = "<!doctype html><html><head><title>Empty</title></head><body></body></html>"


class FakeHttpResponse:
    def __init__(self, body, *, url, content_type="text/html; charset=utf-8"):
        self.body = body.encode("utf-8")
        self.url = url
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class ArticleFetchTests(unittest.TestCase):
    def test_google_news_wrapper_urls_are_classified(self):
        self.assertEqual(
            classify_article_url("https://news.google.com/rss/articles/example?oc=5"),
            URL_CLASS_GOOGLE_NEWS_WRAPPER,
        )
        self.assertEqual(
            classify_article_url("https://news.google.com/articles/example"),
            URL_CLASS_GOOGLE_NEWS_WRAPPER,
        )
        self.assertEqual(
            classify_article_url("https://news.google.com/read/example"),
            URL_CLASS_GOOGLE_NEWS_WRAPPER,
        )
        self.assertEqual(
            classify_article_url("https://finance.yahoo.com/news/nvidia-story"),
            URL_CLASS_DIRECT_PUBLISHER,
        )
        self.assertEqual(classify_article_url("mailto:news@example.com"), URL_CLASS_UNSUPPORTED)

    def test_direct_publisher_url_is_prioritized_over_google_wrapper(self):
        google = _article("https://news.google.com/rss/articles/google", "NVIDIA wrapper", provider="google_news_rss_search")
        direct = _article("https://finance.yahoo.com/news/nvidia", "NVIDIA publisher", provider="yahoo_finance_rss", source="Yahoo Finance")
        cluster = _cluster(google, (google, direct))

        with patch("news_pipeline.article_fetch.urlopen", return_value=_html_response("https://finance.yahoo.com/news/nvidia")) as urlopen:
            enriched, summary = fetch_top_cluster_articles((cluster,), run_date="2026-06-04")

        self.assertEqual(summary.attempted_fetches, 1)
        self.assertEqual(summary.publisher_article_fetches, 1)
        self.assertEqual(summary.google_news_wrappers_skipped, 0)
        self.assertEqual(urlopen.call_args.args[0].full_url, "https://finance.yahoo.com/news/nvidia")
        self.assertIn(google.canonical_url, enriched)

    def test_higher_quality_direct_publisher_url_is_prioritized_for_fetch(self):
        low_priority = _article("https://stocktwits.com/nvda-noise", "NVIDIA Stocktwits", provider="stocktwits", source="Stocktwits")
        high_quality = _article("https://www.marketwatch.com/story/nvda-quality", "NVIDIA MarketWatch", provider="marketwatch_rss", source="MarketWatch")
        cluster = _cluster(low_priority, (low_priority, high_quality))

        with patch("news_pipeline.article_fetch.urlopen", return_value=_html_response(high_quality.canonical_url)) as urlopen:
            _enriched, summary = fetch_top_cluster_articles((cluster,), run_date="2026-06-04")

        self.assertEqual(summary.attempted_fetches, 1)
        self.assertEqual(urlopen.call_args.args[0].full_url, high_quality.canonical_url)

    def test_direct_url_is_selected_before_google_wrapper_when_budget_is_limited(self):
        google = _article("https://news.google.com/rss/articles/google", "NVIDIA wrapper")
        direct = _article("https://www.cnbc.com/nvidia", "NVIDIA publisher", provider="cnbc_rss", source="CNBC")

        def fake_urlopen(request, timeout):
            if "news.google.com" in request.full_url:
                return FakeHttpResponse(EMPTY_HTML, url=request.full_url)
            return _html_response(request.full_url)

        with patch("news_pipeline.article_fetch.urlopen", side_effect=fake_urlopen):
            _enriched, summary = fetch_top_cluster_articles(
                (_cluster(google, (google,)), _cluster(direct, (direct,))),
                run_date="2026-06-04",
                max_article_fetches=1,
            )

        self.assertEqual(summary.attempted_fetches, 1)
        self.assertEqual(summary.publisher_article_fetches, 1)
        self.assertEqual(summary.google_news_wrappers_skipped, 0)
        self.assertEqual(summary.extraction_selected_count, 1)
        self.assertEqual(summary.extraction_skipped_reasons["max_article_fetches"], 1)
        self.assertEqual(summary.extraction_skipped_count, 1)

    def test_resolved_google_news_url_can_be_fetched_as_publisher_url(self):
        google = _article("https://news.google.com/rss/articles/google", "NVIDIA wrapper")
        publisher_url = "https://publisher.example.com/nvidia"

        def fake_urlopen(request, timeout):
            if "news.google.com" in request.full_url:
                return FakeHttpResponse("", url=publisher_url)
            return _html_response(publisher_url)

        with patch("news_pipeline.article_fetch.urlopen", side_effect=fake_urlopen):
            enriched, summary = fetch_top_cluster_articles((_cluster(google, (google,)),), run_date="2026-06-04")

        self.assertEqual(summary.google_news_wrappers_resolved, 1)
        self.assertEqual(summary.publisher_article_fetches, 1)
        self.assertEqual(summary.successful_extractions, 1)
        self.assertEqual(summary.records[0].requested_url, google.canonical_url)
        self.assertEqual(summary.records[0].fetch_url, publisher_url)
        self.assertEqual(summary.records[0].resolution_status, "resolved_to_publisher")
        self.assertIn(google.canonical_url, enriched)

    def test_unresolved_google_wrapper_does_not_consume_extraction_budget(self):
        google = _article("https://news.google.com/rss/articles/unresolved", "NVIDIA wrapper")
        with patch(
            "news_pipeline.article_fetch.urlopen",
            return_value=FakeHttpResponse(EMPTY_HTML, url=google.canonical_url),
        ):
            _enriched, summary = fetch_top_cluster_articles(
                (_cluster(google, (google,)),),
                run_date="2026-06-04",
                max_article_fetches=1,
            )

        self.assertEqual(summary.attempted_fetches, 0)
        self.assertEqual(summary.extraction_selected_count, 0)
        self.assertEqual(summary.as_dict()["extraction_budget_unused_count"], 1)
        self.assertEqual(summary.google_wrappers_unresolved, 1)
        self.assertIn("google_wrapper_unresolved", summary.records[0].failure_reasons)

    def test_failure_reasons_are_explicit_for_non_article_html_and_fallbacks(self):
        direct = _article("https://publisher.example.com/empty", "NVIDIA empty", snippet="NVIDIA snippet text.")
        with patch("news_pipeline.article_fetch.urlopen", return_value=FakeHttpResponse(EMPTY_HTML, url=direct.canonical_url)):
            _enriched, summary = fetch_top_cluster_articles((_cluster(direct, (direct,)),), run_date="2026-06-04")

        record = summary.records[0]
        self.assertEqual(record.error_class, "no_article_body")
        self.assertIn("no_article_body", record.failure_reasons)
        self.assertIn("snippet_fallback", record.failure_reasons)
        self.assertEqual(record.extraction_basis, "snippet")
        self.assertEqual(record.extraction_method_used, "fallback")
        self.assertIn("no_article_body", record.extraction_failure_reason)

    def test_unsupported_content_type_is_recorded_explicitly(self):
        direct = _article("https://publisher.example.com/file.pdf", "NVIDIA pdf", snippet="NVIDIA snippet text.")
        response = FakeHttpResponse("%PDF mock", url=direct.canonical_url, content_type="application/pdf")

        with patch("news_pipeline.article_fetch.urlopen", return_value=response):
            _enriched, summary = fetch_top_cluster_articles((_cluster(direct, (direct,)),), run_date="2026-06-04")

        record = summary.records[0]
        self.assertEqual(record.error_class, "unsupported_content_type")
        self.assertIn("unsupported_content_type", record.failure_reasons)
        self.assertIn("snippet_fallback", record.failure_reasons)

    def test_title_fallback_is_recorded_explicitly(self):
        direct = _article("https://publisher.example.com/empty-title", "NVIDIA empty", snippet=None)
        with patch("news_pipeline.article_fetch.urlopen", return_value=FakeHttpResponse(EMPTY_HTML, url=direct.canonical_url)):
            _enriched, summary = fetch_top_cluster_articles((_cluster(direct, (direct,)),), run_date="2026-06-04")

        record = summary.records[0]
        self.assertEqual(record.extraction_basis, "title")
        self.assertIn("title_fallback", record.failure_reasons)

    def test_extraction_record_includes_explicit_method_and_failure_diagnostics(self):
        direct = _article("https://publisher.example.com/nvidia", "NVIDIA publisher")
        with patch("news_pipeline.article_fetch.urlopen", return_value=_html_response(direct.canonical_url)):
            _enriched, summary = fetch_top_cluster_articles((_cluster(direct, (direct,)),), run_date="2026-06-04")

        record = summary.records[0]
        payload = summary.as_dict()
        self.assertEqual(record.extraction_basis, "full_text")
        self.assertIn(
            record.extraction_method_used,
            {
                "trafilatura_standard",
                "trafilatura_favor_recall",
                "trafilatura_baseline",
                "internal_article_parser",
            },
        )
        self.assertIsNone(record.extraction_failure_reason)
        self.assertIn("extraction_method_used", payload["records"][0])
        self.assertIn("extraction_failure_reason", payload["records"][0])
        self.assertIn("extraction_method_counts", payload)
        self.assertIn("extraction_quality_grade_counts", payload)
        self.assertTrue(record.accepted_as_full_text)

    def test_cache_prevents_duplicate_fetch_for_same_url(self):
        direct = _article("https://publisher.example.com/nvidia-cache", "NVIDIA publisher")
        cluster = _cluster(direct, (direct,))
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "extraction_cache.json"
            with patch("news_pipeline.article_fetch.urlopen", return_value=_html_response(direct.canonical_url)) as first_fetch:
                _enriched, first_summary = fetch_top_cluster_articles(
                    (cluster,),
                    run_date="2026-06-04",
                    cache_path=cache_path,
                )
            with patch(
                "news_pipeline.article_fetch.urlopen",
                side_effect=AssertionError("cached URL must not be fetched again"),
            ):
                enriched, second_summary = fetch_top_cluster_articles(
                    (cluster,),
                    run_date="2026-06-04",
                    cache_path=cache_path,
                )

        self.assertEqual(first_fetch.call_count, 1)
        self.assertEqual(first_summary.successful_extractions, 1)
        self.assertEqual(second_summary.attempted_fetches, 0)
        self.assertEqual(second_summary.successful_extractions, 1)
        self.assertTrue(second_summary.records[0].cache_hit)
        self.assertIn(direct.canonical_url, enriched)

    def test_extraction_queue_prioritizes_high_quality_ticker_specific_article(self):
        high = _article(
            "https://reuters.com/nvidia-results",
            "NVIDIA earnings beat estimates",
            provider="reuters_rss",
            source="Reuters",
        )
        weak = _article(
            "https://stocktwits.com/nvidia-opinion",
            "Is NVIDIA a good stock to buy now?",
            provider="stocktwits",
            source="Stocktwits",
        )

        queue = build_extraction_queue(
            cluster_articles((weak, high)),
            run_date="2026-06-08",
            include_low_quality_sources=False,
            min_source_quality_tier=3,
        )

        self.assertEqual(queue[0].article.canonical_url, high.canonical_url)
        self.assertGreater(queue[0].score, queue[1].score)

    def test_extraction_queue_excludes_low_quality_source_by_default(self):
        excluded = _article(
            "https://stocktwits.com/nvidia",
            "NVIDIA stock news",
            provider="stocktwits",
            source="Stocktwits",
        )

        queue = build_extraction_queue(
            cluster_articles((excluded,)),
            run_date="2026-06-08",
            include_low_quality_sources=False,
            min_source_quality_tier=3,
        )

        self.assertFalse(queue[0].eligible)
        self.assertEqual(queue[0].skip_reason, "source_quality_excluded")


def _article(url, title, *, snippet="NVIDIA stock news.", provider="google_news_rss_search", source="Google News"):
    return Article(
        canonical_url=url,
        title=title,
        snippet=snippet,
        published_at="2026-06-04T10:00:00+00:00",
        metadata={"provider": provider, "source_name": source},
    )


def _cluster(canonical, articles):
    return DedupeCluster(
        canonical_article=canonical,
        alternate_source_links=tuple(article.canonical_url for article in articles if article != canonical),
        articles=tuple(articles),
        duplicate_reasons=(),
        primary_link=canonical.canonical_url,
        supporting_links=tuple(
            SourceLink(article.title, article.canonical_url, article.metadata.get("source_name"), article.metadata.get("provider"), article.published_at)
            for article in articles
        ),
        publisher_count=1,
        source_count=1,
        publisher_names=("Test Publisher",),
        source_providers=("test",),
    )


def _html_response(url):
    return FakeHttpResponse(ARTICLE_HTML, url=url)


if __name__ == "__main__":
    unittest.main()
