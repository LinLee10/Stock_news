import unittest

from news_pipeline.extract import choose_extraction_basis, extract_article


HTML_FIXTURE = """
<html>
  <head>
    <title>Apple reports strong growth</title>
    <meta name="author" content="Jane Analyst">
    <meta property="article:published_time" content="2026-06-03T10:00:00Z">
  </head>
  <body>
    <article>
      <p>Apple reported strong growth in services revenue.</p>
      <p>The company said demand remained resilient.</p>
    </article>
  </body>
</html>
"""


class ExtractTests(unittest.TestCase):
    def test_extract_article_from_local_html_fixture(self):
        result = extract_article(raw_html=HTML_FIXTURE, url="https://example.com/aapl")

        self.assertEqual(result.title, "Apple reports strong growth")
        self.assertIn("Apple reported strong growth", result.main_text)
        self.assertIn("demand remained resilient", result.main_text)
        self.assertEqual(result.publish_date, "2026-06-03T10:00:00Z")
        self.assertEqual(result.author, "Jane Analyst")
        self.assertEqual(result.extraction_status, "success")
        self.assertEqual(result.extraction_basis, "full_text")
        self.assertIsNone(result.extraction_error)

    def test_extract_article_falls_back_to_snippet_without_full_text(self):
        result = extract_article(
            raw_html="<html><head><title>Fallback title</title></head><body></body></html>",
            url="https://example.com/empty",
            snippet="Short provider snippet.",
        )

        self.assertEqual(result.title, "Fallback title")
        self.assertEqual(result.main_text, "Short provider snippet.")
        self.assertEqual(result.extraction_status, "fallback")
        self.assertEqual(result.extraction_basis, "snippet")

    def test_extract_article_falls_back_to_title(self):
        result = extract_article(
            raw_html="<html><body></body></html>",
            url="https://example.com/title",
            title="Only title available",
        )

        self.assertEqual(result.main_text, "Only title available")
        self.assertEqual(result.extraction_basis, "title")

    def test_extract_article_failed_when_no_text_available(self):
        result = extract_article(raw_html="<html><body></body></html>", url="https://example.com/failed")

        self.assertEqual(result.extraction_status, "failed")
        self.assertEqual(result.extraction_basis, "failed")
        self.assertEqual(result.main_text, "")
        self.assertIsNotNone(result.extraction_error)

    def test_choose_extraction_basis_uses_expected_order(self):
        self.assertEqual(
            choose_extraction_basis(full_text="Full", snippet="Snippet", title="Title").extraction_basis,
            "full_text",
        )
        self.assertEqual(
            choose_extraction_basis(snippet="Snippet", title="Title").extraction_basis,
            "snippet",
        )
        self.assertEqual(choose_extraction_basis(title="Title").extraction_basis, "title")
        self.assertEqual(choose_extraction_basis().extraction_basis, "failed")


if __name__ == "__main__":
    unittest.main()
