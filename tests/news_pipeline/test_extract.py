import unittest
from unittest.mock import patch

from news_pipeline import extract as extract_module
from news_pipeline.extract import (
    ExtractionAttempt,
    choose_extraction_basis,
    extract_article,
    extraction_dependency_status,
    score_extraction_quality,
)


PARAGRAPHS = (
    "Apple reported strong growth in services revenue as customers increased spending across its installed base.",
    "The company said demand remained resilient during the quarter and management maintained its operating priorities.",
    "Executives told investors that product availability improved while supply constraints eased across several markets.",
    "Apple shares moved after the results because revenue and profit exceeded the expectations cited by analysts.",
    "The company also described continued investment in artificial intelligence features and supporting infrastructure.",
    "Management said the business expects to balance near-term spending with long-term product development plans.",
)
LONG_TEXT = "\n\n".join(PARAGRAPHS)
HTML_FIXTURE = f"""
<html>
  <head>
    <title>Apple reports strong growth</title>
    <meta name="author" content="Jane Analyst">
    <meta property="article:published_time" content="2026-06-03T10:00:00Z">
    <link rel="canonical" href="https://example.com/canonical-aapl">
    <meta property="og:url" content="https://example.com/og-aapl">
  </head>
  <body><article>{''.join(f'<p>{paragraph}</p>' for paragraph in PARAGRAPHS)}</article></body>
</html>
"""


class ExtractTests(unittest.TestCase):
    def test_extract_article_from_local_html_fixture(self):
        result = extract_article(
            raw_html=HTML_FIXTURE,
            url="https://example.com/aapl",
            ticker_terms=("AAPL", "Apple"),
        )

        self.assertEqual(result.title, "Apple reports strong growth")
        self.assertIn("Apple reported strong growth", result.main_text)
        self.assertEqual(result.publish_date, "2026-06-03T10:00:00Z")
        self.assertEqual(result.author, "Jane Analyst")
        self.assertEqual(result.extraction_basis, "full_text")
        self.assertTrue(result.accepted_as_full_text)
        self.assertIn(result.extraction_quality_grade, {"strong_full_text", "usable_full_text"})
        self.assertEqual(result.canonical_url, "https://example.com/canonical-aapl")
        self.assertEqual(result.og_url, "https://example.com/og-aapl")

    def test_trafilatura_standard_is_tried_first(self):
        accepted = _attempt("trafilatura_standard", LONG_TEXT)
        with patch("news_pipeline.extract._extract_with_trafilatura_standard", return_value=accepted), patch(
            "news_pipeline.extract._extract_with_trafilatura_favor_recall",
            side_effect=AssertionError("favor recall should not run after accepted standard extraction"),
        ):
            result = extract_article(
                raw_html=HTML_FIXTURE,
                url="https://example.com/aapl",
                title="Apple reports strong growth",
                ticker_terms=("Apple",),
            )

        self.assertEqual(result.extraction_method_used, "trafilatura_standard")
        self.assertEqual([attempt.method_name for attempt in result.attempts], ["trafilatura_standard"])

    def test_favor_recall_runs_when_standard_fails(self):
        with patch(
            "news_pipeline.extract._extract_with_trafilatura_standard",
            return_value=ExtractionAttempt("trafilatura_standard", failure_reason="no_extractable_text"),
        ), patch(
            "news_pipeline.extract._extract_with_trafilatura_favor_recall",
            return_value=_attempt("trafilatura_favor_recall", LONG_TEXT),
        ):
            result = extract_article(
                raw_html=HTML_FIXTURE,
                url="https://example.com/aapl",
                title="Apple reports strong growth",
                ticker_terms=("Apple",),
            )

        self.assertEqual(result.extraction_method_used, "trafilatura_favor_recall")
        self.assertEqual(
            [attempt.method_name for attempt in result.attempts],
            ["trafilatura_standard", "trafilatura_favor_recall"],
        )

    def test_baseline_runs_before_internal_parser_when_needed(self):
        failed_standard = ExtractionAttempt("trafilatura_standard", failure_reason="no_extractable_text")
        failed_recall = ExtractionAttempt("trafilatura_favor_recall", failure_reason="no_extractable_text")
        with patch("news_pipeline.extract._extract_with_trafilatura_standard", return_value=failed_standard), patch(
            "news_pipeline.extract._extract_with_trafilatura_favor_recall", return_value=failed_recall
        ), patch(
            "news_pipeline.extract._extract_with_trafilatura_baseline",
            return_value=_attempt("trafilatura_baseline", LONG_TEXT),
        ), patch(
            "news_pipeline.extract._extract_with_internal_parser",
            side_effect=AssertionError("internal parser should not run after accepted baseline extraction"),
        ):
            result = extract_article(
                raw_html=HTML_FIXTURE,
                url="https://example.com/aapl",
                title="Apple reports strong growth",
                ticker_terms=("Apple",),
            )

        self.assertEqual(result.extraction_method_used, "trafilatura_baseline")

    def test_internal_parser_is_used_when_trafilatura_is_missing(self):
        original_find_spec = extract_module.importlib.util.find_spec

        def fake_find_spec(name):
            if name == "trafilatura":
                return None
            return original_find_spec(name)

        with patch("news_pipeline.extract.importlib.util.find_spec", side_effect=fake_find_spec):
            result = extract_article(
                raw_html=HTML_FIXTURE,
                url="https://example.com/aapl",
                ticker_terms=("Apple",),
            )

        self.assertEqual(result.extraction_basis, "full_text")
        self.assertEqual(result.extraction_method_used, "internal_article_parser")

    def test_google_news_wrapper_shell_is_rejected(self):
        result = extract_article(
            raw_html=HTML_FIXTURE,
            url="https://news.google.com/rss/articles/example",
            snippet="Apple provider snippet.",
            ticker_terms=("Apple",),
        )

        self.assertEqual(result.extraction_basis, "snippet")
        self.assertEqual(result.extraction_quality_grade, "blocked_or_shell")
        self.assertIn("google_wrapper_unresolved", result.extraction_quality_reasons)

    def test_cookie_navigation_shell_is_rejected(self):
        html = "<html><body><nav>Home Markets Login</nav><p>Accept all cookies to continue.</p></body></html>"
        result = extract_article(
            raw_html=html,
            url="https://example.com/shell",
            snippet="Apple provider snippet.",
            ticker_terms=("Apple",),
        )

        self.assertEqual(result.extraction_quality_grade, "blocked_or_shell")
        self.assertIn("blocked_or_shell", result.extraction_quality_reasons)

    def test_paywall_or_login_reason_is_recorded(self):
        quality = score_extraction_quality(
            "Subscribe to continue. Sign in to continue.",
            final_url="https://example.com/paywall",
        )

        self.assertEqual(quality.grade, "blocked_or_shell")
        self.assertIn("paywall_or_login", quality.reasons)

    def test_boilerplate_heavy_reason_is_recorded(self):
        quality = score_extraction_quality(
            "Privacy policy. Terms of use. Cookie policy. All rights reserved.",
            final_url="https://example.com/navigation",
        )

        self.assertIn("boilerplate_heavy", quality.reasons)

    def test_short_text_is_not_strong_full_text(self):
        quality = score_extraction_quality(
            "Apple reported quarterly results.",
            source_title="Apple quarterly results",
            ticker_terms=("Apple",),
        )

        self.assertNotEqual(quality.grade, "strong_full_text")
        self.assertIn("too_short", quality.reasons)

    def test_missing_ticker_context_is_recorded(self):
        quality = score_extraction_quality(
            LONG_TEXT.replace("Apple", "Another company"),
            source_title="Another company reports strong growth",
            ticker_terms=("NVIDIA", "NVDA"),
            paragraph_count=6,
        )

        self.assertIn("missing_ticker_context", quality.reasons)

    def test_ticker_specific_article_text_is_accepted(self):
        quality = score_extraction_quality(
            LONG_TEXT,
            source_title="Apple reports strong growth",
            ticker_terms=("Apple",),
            paragraph_count=6,
        )

        self.assertTrue(quality.accepted_as_full_text)
        self.assertIn(quality.grade, {"strong_full_text", "usable_full_text"})

    def test_extract_article_falls_back_to_snippet_and_title(self):
        snippet_result = extract_article(
            raw_html="<html><body></body></html>",
            url="https://example.com/empty",
            snippet="Short provider snippet.",
            title="Fallback title",
        )
        title_result = extract_article(
            raw_html="<html><body></body></html>",
            url="https://example.com/title",
            title="Only title available",
        )

        self.assertEqual(snippet_result.extraction_basis, "snippet")
        self.assertEqual(title_result.extraction_basis, "title")

    def test_optional_dependency_status_remains_safe(self):
        status = extraction_dependency_status()

        self.assertIn("trafilatura_available", status)
        self.assertIn("readability_available", status)
        self.assertTrue(status["internal_parser_available"])

    def test_choose_extraction_basis_uses_expected_order(self):
        self.assertEqual(
            choose_extraction_basis(full_text="Full", snippet="Snippet", title="Title").extraction_basis,
            "full_text",
        )
        self.assertEqual(choose_extraction_basis(snippet="Snippet", title="Title").extraction_basis, "snippet")
        self.assertEqual(choose_extraction_basis(title="Title").extraction_basis, "title")
        self.assertEqual(choose_extraction_basis().extraction_basis, "failed")


def _attempt(method_name, text):
    return ExtractionAttempt(
        method_name=method_name,
        text=text,
        text_length=len(text),
        sentence_count=6,
        paragraph_count=6,
        metadata={"title": "Apple reports strong growth"},
        success=True,
    )


if __name__ == "__main__":
    unittest.main()
