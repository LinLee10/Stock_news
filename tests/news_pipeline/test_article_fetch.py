import unittest
from unittest.mock import patch

from news_pipeline.article_fetch import (
    URL_CLASS_DIRECT_PUBLISHER,
    URL_CLASS_GOOGLE_NEWS_WRAPPER,
    URL_CLASS_UNSUPPORTED,
    classify_article_url,
    fetch_top_cluster_articles,
)
from news_pipeline.dedup import DedupeCluster, SourceLink
from news_pipeline.models import Article


ARTICLE_HTML = """<!doctype html>
<html>
  <head><title>NVIDIA article page</title></head>
  <body>
    <article><p>NVIDIA reports weak demand and a loss in this full article.</p></article>
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

    def test_unresolved_google_wrapper_does_not_consume_fetch_budget_when_direct_url_is_available(self):
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
        self.assertEqual(summary.google_news_wrappers_skipped, 1)
        self.assertEqual(summary.records[0].error_class, "google_news_unresolved")
        self.assertFalse(summary.records[0].fetched)

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

    def test_failure_reasons_are_explicit_for_non_article_html_and_fallbacks(self):
        direct = _article("https://publisher.example.com/empty", "NVIDIA empty", snippet="NVIDIA snippet text.")
        with patch("news_pipeline.article_fetch.urlopen", return_value=FakeHttpResponse(EMPTY_HTML, url=direct.canonical_url)):
            _enriched, summary = fetch_top_cluster_articles((_cluster(direct, (direct,)),), run_date="2026-06-04")

        record = summary.records[0]
        self.assertEqual(record.error_class, "optional_extractors_unavailable")
        self.assertIn("no_article_body", record.failure_reasons)
        self.assertIn("snippet_fallback", record.failure_reasons)
        self.assertEqual(record.extraction_basis, "snippet")

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
