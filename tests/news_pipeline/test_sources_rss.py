import unittest

from news_pipeline.sources.rss import RssSource, clean_rss_snippet


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

    def test_clean_rss_snippet_removes_google_news_html(self):
        snippet = (
            '<a href="https://news.google.com/rss/articles/abc">NVIDIA stock jumps</a>'
            "&nbsp;&nbsp;<font color=\"#6f6f6f\">Example Publisher</font>"
        )

        self.assertEqual(clean_rss_snippet(snippet), "NVIDIA stock jumps Example Publisher")

    def test_configured_source_name_is_used_before_generic_channel_title(self):
        source = RssSource(
            RSS_FIXTURE.replace("<source>Example Wire</source>", ""),
            provider_name="cnbc_rss",
            default_source_name="CNBC",
        )

        self.assertEqual(source.articles()[0].metadata["source_name"], "CNBC")


if __name__ == "__main__":
    unittest.main()
