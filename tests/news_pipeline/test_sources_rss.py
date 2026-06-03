import unittest

from news_pipeline.sources.rss import RssSource


RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Example Finance</title>
    <item>
      <title>Apple beats estimates</title>
      <link>https://www.example.com/news/apple?utm_source=rss&amp;id=1</link>
      <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
      <source>Example Wire</source>
      <description>Apple reported strong quarterly growth.</description>
    </item>
  </channel>
</rss>
"""


class RssSourceTests(unittest.TestCase):
    def test_rss_normalizes_items_to_articles(self):
        source = RssSource(RSS_FIXTURE, provider_name="google_news_rss")

        articles = source.articles()

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Apple beats estimates")
        self.assertEqual(articles[0].canonical_url, "https://example.com/news/apple?id=1")
        self.assertEqual(articles[0].snippet, "Apple reported strong quarterly growth.")
        self.assertEqual(articles[0].metadata["provider"], "google_news_rss")
        self.assertEqual(articles[0].metadata["source_name"], "Example Wire")
        self.assertIn("2024-01-15T10:00:00", articles[0].published_at)


if __name__ == "__main__":
    unittest.main()
