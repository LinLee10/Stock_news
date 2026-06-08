import unittest

from news_pipeline.dedup import cluster_articles
from news_pipeline.models import Article
from news_pipeline.summaries import (
    READ_FIRST,
    build_market_intelligence,
    summarize_article,
)


class SummaryTests(unittest.TestCase):
    def test_article_summary_uses_full_text_when_available(self):
        article = Article(
            canonical_url="https://reuters.com/technology/nvidia-demand",
            title="NVIDIA demand update",
            full_text=(
                "Executives spoke at an industry event. "
                "NVIDIA said data center revenue and AI chip demand remained strong."
            ),
            snippet="A shorter feed snippet.",
        )

        summary = summarize_article(article)

        self.assertEqual(summary.summary_basis, "full_text")
        self.assertIn("data center revenue", summary.article_summary)
        self.assertIsNone(summary.summary_warning)

    def test_article_summary_falls_back_to_snippet(self):
        article = Article(
            canonical_url="https://finance.yahoo.com/news/nvidia-guidance",
            title="NVIDIA update",
            snippet="NVIDIA shares rose after the company raised revenue guidance.",
        )

        summary = summarize_article(article)

        self.assertEqual(summary.summary_basis, "snippet")
        self.assertIn("raised revenue guidance", summary.article_summary)
        self.assertIn("snippet", summary.summary_warning)

    def test_article_summary_falls_back_to_title(self):
        article = Article(
            canonical_url="https://example.com/nvidia-title-only",
            title="NVIDIA shares fall after analyst downgrade",
        )

        summary = summarize_article(article)

        self.assertEqual(summary.summary_basis, "title")
        self.assertEqual(summary.article_summary, "NVIDIA shares fall after analyst downgrade.")
        self.assertIn("title only", summary.summary_warning)

    def test_event_cluster_and_ticker_summaries_are_created(self):
        article = Article(
            canonical_url="https://reuters.com/technology/nvidia-results",
            title="NVIDIA earnings beat expectations",
            published_at="2026-06-05T00:00:00+00:00",
            full_text="NVIDIA reported stronger data center revenue and raised guidance for AI chip demand.",
            metadata={"source_name": "Reuters"},
        )
        clusters = cluster_articles([article])

        intelligence = build_market_intelligence(
            articles=[article],
            clusters=clusters,
            run_date="2026-06-05",
        )

        cluster = intelligence.cluster_intelligence[("NVDA", article.title)]
        ticker = intelligence.ticker_summaries["NVDA"]
        self.assertIn("NVIDIA", cluster.cluster_summary)
        self.assertEqual(cluster.cluster_summary_basis, "full_text")
        self.assertIn("NVDA:", ticker.ticker_daily_summary)
        self.assertEqual(ticker.read_first_story, article.title)

    def test_ranking_prefers_high_quality_full_text_direct_article(self):
        strong = Article(
            canonical_url="https://reuters.com/technology/nvidia-regulatory-review",
            title="NVIDIA faces regulatory review of AI chips",
            published_at="2026-06-05T00:00:00+00:00",
            full_text="Regulators opened a review affecting NVIDIA AI chip sales and data center demand.",
            metadata={"source_name": "Reuters"},
        )
        weak = Article(
            canonical_url="https://news.google.com/rss/articles/nvidia-note",
            title="Analyst changes NVIDIA price target",
            published_at="2026-06-05T00:00:00+00:00",
            snippet="An analyst changed a price target for NVIDIA shares.",
            metadata={"source_name": "Yahoo Finance"},
        )

        intelligence = build_market_intelligence(
            articles=[weak, strong],
            clusters=cluster_articles([weak, strong]),
            run_date="2026-06-05",
        )

        reads = intelligence.ranked_reads_by_ticker["NVDA"]
        read_first = next(read for read in reads if read.reading_priority == READ_FIRST)
        self.assertEqual(read_first.url, strong.canonical_url)
        self.assertEqual(read_first.summary_basis, "full_text")
        self.assertTrue(read_first.direct_publisher_url)

    def test_excluded_source_cannot_be_read_first(self):
        excluded = Article(
            canonical_url="https://mshale.com/nvidia-stock",
            title="NVIDIA stock rises on AI chip demand",
            published_at="2026-06-05T00:00:00+00:00",
            full_text="NVIDIA stock rose as AI chip demand increased.",
            metadata={"source_name": "Mshale"},
        )

        intelligence = build_market_intelligence(
            articles=[excluded],
            clusters=cluster_articles([excluded]),
            run_date="2026-06-05",
        )

        self.assertNotIn(
            READ_FIRST,
            [read.reading_priority for read in intelligence.ranked_reads_by_ticker["NVDA"]],
        )


if __name__ == "__main__":
    unittest.main()
