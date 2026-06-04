import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from news_pipeline.sources.live_rss import collect_live_rss_articles
from news_pipeline.sources.rss_config import RSS_SOURCE_FAMILIES


def _rss(title, link, source, description="NVDA stock news"):
    return f"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>{source}</title>
    <item>
      <title>{title}</title>
      <link>{link}</link>
      <pubDate>Wed, 3 Jun 2026 10:00:00 GMT</pubDate>
      <source>{source}</source>
      <description>{description}</description>
    </item>
  </channel>
</rss>
"""


def _rss_items(symbol, source, count=3):
    items = []
    for index in range(count):
        items.append(
            f"""
    <item>
      <title>{symbol} stock story {index}</title>
      <link>https://example.com/{symbol.lower()}/{index}</link>
      <pubDate>Wed, 3 Jun 2026 10:0{index}:00 GMT</pubDate>
      <source>{source}</source>
      <description>{symbol} stock news from mocked Google RSS.</description>
    </item>
"""
        )
    return f"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>{source}</title>
    {''.join(items)}
  </channel>
</rss>
"""


def _rss_two_items():
    return """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Yahoo Finance</title>
    <item>
      <title>Honeywell spins off quantum unit</title>
      <link>https://finance.yahoo.com/news/honeywell-quantum</link>
      <pubDate>Wed, 3 Jun 2026 10:00:00 GMT</pubDate>
      <source>Yahoo Finance</source>
      <description>Industrial company update.</description>
    </item>
    <item>
      <title>NVIDIA stock rises after analyst upgrade</title>
      <link>https://finance.yahoo.com/news/nvidia-upgrade</link>
      <pubDate>Wed, 3 Jun 2026 10:10:00 GMT</pubDate>
      <source>Yahoo Finance</source>
      <description>NVDA stock news from Yahoo Finance.</description>
    </item>
  </channel>
</rss>
"""


class FakeHttpResponse:
    def __init__(self, body):
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class LiveRssTests(unittest.TestCase):
    def test_direct_yahoo_cnbc_marketwatch_families_are_mock_fetchable(self):
        direct_families = tuple(
            family
            for family in RSS_SOURCE_FAMILIES
            if family.name in {"yahoo_finance_rss", "cnbc_rss", "marketwatch_rss"}
        )

        def fake_urlopen(request, timeout):
            url = request.full_url
            if "finance.yahoo.com" in url:
                return FakeHttpResponse(_rss("NVIDIA earnings update", "https://finance.yahoo.com/news/nvda", "Yahoo Finance"))
            if "cnbc.com" in url:
                return FakeHttpResponse(_rss("NVIDIA earnings update", "https://www.cnbc.com/nvda", "CNBC"))
            if "dowjones.io" in url:
                return FakeHttpResponse(_rss("NVIDIA earnings update", "https://www.marketwatch.com/nvda", "MarketWatch"))
            raise AssertionError(url)

        with patch("news_pipeline.sources.live_rss.urlopen", side_effect=fake_urlopen):
            articles, attempts = collect_live_rss_articles(
                source_families=direct_families,
                retries=0,
            )

        self.assertEqual({attempt.provider for attempt in attempts}, {"yahoo_finance_rss", "cnbc_rss", "marketwatch_rss"})
        self.assertEqual({attempt.status for attempt in attempts}, {"success"})
        self.assertEqual(len(articles), 3)
        self.assertEqual({article.metadata["provider"] for article in articles}, {"yahoo_finance_rss", "cnbc_rss", "marketwatch_rss"})

    def test_one_direct_feed_failure_does_not_drop_successful_feed(self):
        direct_families = tuple(
            family
            for family in RSS_SOURCE_FAMILIES
            if family.name in {"yahoo_finance_rss", "cnbc_rss"}
        )

        def fake_urlopen(request, timeout):
            if "finance.yahoo.com" in request.full_url:
                raise TimeoutError("mock timeout")
            return FakeHttpResponse(_rss("NVIDIA chip demand rises", "https://www.cnbc.com/nvda-demand", "CNBC"))

        with patch("news_pipeline.sources.live_rss.urlopen", side_effect=fake_urlopen):
            articles, attempts = collect_live_rss_articles(
                source_families=direct_families,
                retries=0,
            )

        self.assertEqual(len(articles), 1)
        self.assertEqual([attempt.status for attempt in attempts], ["failure", "success"])
        self.assertEqual(attempts[0].error_class, "TimeoutError")

    def test_caps_keep_live_rss_collection_readable(self):
        google_family = tuple(
            family for family in RSS_SOURCE_FAMILIES if family.name == "google_news_rss_search"
        )

        def fake_urlopen(request, timeout):
            return FakeHttpResponse(_rss("NVIDIA stock update", "https://example.com/live/nvda-stock-update", "Google News"))

        with patch("news_pipeline.sources.live_rss.urlopen", side_effect=fake_urlopen):
            articles, attempts = collect_live_rss_articles(
                source_families=google_family,
                retries=0,
                max_articles_per_source=5,
                max_articles_per_ticker=3,
                max_total_articles=4,
            )

        self.assertEqual(len(attempts), 21)
        self.assertLessEqual(len(articles), 4)
        self.assertLessEqual(sum(attempt.article_count for attempt in attempts), 4)

    def test_google_news_source_cap_is_fair_across_ticker_feeds(self):
        google_family = tuple(
            family for family in RSS_SOURCE_FAMILIES if family.name == "google_news_rss_search"
        )

        def fake_urlopen(request, timeout):
            query = parse_qs(urlparse(request.full_url).query).get("q", ["NVDA stock news"])[0]
            symbol = query.split()[0]
            return FakeHttpResponse(_rss_items(symbol, "Google News", count=3))

        with patch("news_pipeline.sources.live_rss.urlopen", side_effect=fake_urlopen):
            articles, attempts = collect_live_rss_articles(
                source_families=google_family,
                retries=0,
                max_articles_per_source=6,
                max_articles_per_ticker=3,
                max_total_articles=6,
            )

        contributing_feeds = [attempt.feed_id for attempt in attempts if attempt.article_count]
        self.assertEqual(len(articles), 6)
        self.assertEqual(len(contributing_feeds), 6)
        self.assertIn("google_news_sndk", contributing_feeds)
        self.assertIn("google_news_nvda", contributing_feeds)

    def test_direct_broad_feeds_only_keep_configured_ticker_matches(self):
        direct_families = tuple(
            family for family in RSS_SOURCE_FAMILIES if family.name == "yahoo_finance_rss"
        )

        with patch("news_pipeline.sources.live_rss.urlopen", return_value=FakeHttpResponse(_rss_two_items())):
            articles, attempts = collect_live_rss_articles(
                source_families=direct_families,
                retries=0,
            )

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "NVIDIA stock rises after analyst upgrade")
        self.assertEqual(attempts[0].fetched_article_count, 2)
        self.assertEqual(attempts[0].article_count, 1)


if __name__ == "__main__":
    unittest.main()
