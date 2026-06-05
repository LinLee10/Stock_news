import unittest

from news_pipeline.dedup import canonicalize_url, cluster_articles, dedup_key, normalize_title
from news_pipeline.models import Article


class DedupTests(unittest.TestCase):
    def test_canonicalize_url_removes_tracking_and_sorts_params(self):
        url = (
            "HTTPS://www.Example.com:443/News/Article/"
            "?utm_source=feed&token=MiXeD&id=42&utm_medium=email#section"
        )

        self.assertEqual(
            canonicalize_url(url),
            "https://example.com/News/Article?id=42&token=MiXeD",
        )

    def test_normalize_title_is_case_and_space_insensitive(self):
        self.assertEqual(normalize_title("  Apple   Beats Estimates "), "apple beats estimates")

    def test_dedup_key_is_deterministic_for_equivalent_inputs(self):
        first = dedup_key(
            "RSS",
            "https://example.com/article?id=1&utm_source=x",
            "Company Beats Estimates",
        )
        second = dedup_key(
            "rss",
            "HTTPS://EXAMPLE.com/article?utm_medium=y&id=1",
            " company   beats estimates ",
        )

        self.assertEqual(first, second)

    def test_canonicalize_url_unwraps_safe_publisher_redirects(self):
        redirected = (
            "https://news.google.com/read?url=https%3A%2F%2Fwww.publisher.com%2Fstory"
            "%3Futm_campaign%3Dfeed%26id%3D9"
        )

        self.assertEqual(canonicalize_url(redirected), "https://publisher.com/story?id=9")

    def test_exact_url_dedup_with_tracking_params(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://example.com/story?id=42&utm_source=rss",
                    title="Apple beats quarterly estimates",
                ),
                Article(
                    canonical_url="HTTPS://www.example.com:443/story?utm_medium=email&id=42",
                    title="Different syndicated headline",
                ),
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].duplicate_reasons, ("exact_url",))
        self.assertEqual(clusters[0].alternate_source_links, ())
        self.assertEqual(clusters[0].canonical_article.canonical_url, "https://example.com/story?id=42")

    def test_similar_headline_across_publishers_merges(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/apple-earnings",
                    title="Apple beats estimates after strong iPhone demand",
                    metadata={"provider": "google_news_rss_search", "source_name": "Publisher A"},
                ),
                Article(
                    canonical_url="https://publisher-b.com/markets/aapl-results",
                    title="Apple beats estimates after strong iPhone sales",
                    metadata={"provider": "yahoo_finance_rss", "source_name": "Publisher B"},
                ),
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].duplicate_reasons, ("similar_title",))
        self.assertEqual(
            clusters[0].alternate_source_links,
            ("https://publisher-b.com/markets/aapl-results",),
        )
        self.assertEqual(clusters[0].publisher_count, 2)
        self.assertEqual(clusters[0].source_count, 2)
        self.assertEqual([link.publisher for link in clusters[0].supporting_links], ["Publisher A", "Publisher B"])

    def test_duplicate_story_across_google_yahoo_cnbc_marketwatch_merges_one_event(self):
        articles = [
            Article(
                canonical_url="https://news.google.com/read?url=https%3A%2F%2Fpublisher.com%2Fnvidia-ai-chip",
                title="NVIDIA unveils new AI chip for data centers",
                published_at="2026-06-03T10:00:00+00:00",
                metadata={"provider": "google_news_rss_search", "source_name": "Google News"},
            ),
            Article(
                canonical_url="https://finance.yahoo.com/news/nvidia-ai-chip",
                title="NVIDIA unveils new AI chip for data centres",
                published_at="2026-06-03T10:05:00+00:00",
                metadata={"provider": "yahoo_finance_rss", "source_name": "Yahoo Finance"},
            ),
            Article(
                canonical_url="https://www.cnbc.com/nvidia-ai-chip",
                title="NVIDIA unveils new AI chip for data centers",
                published_at="2026-06-03T10:10:00+00:00",
                metadata={"provider": "cnbc_rss", "source_name": "CNBC"},
            ),
            Article(
                canonical_url="https://www.marketwatch.com/nvidia-ai-chip",
                title="Nvidia unveils new AI chip for data centers",
                published_at="2026-06-03T10:20:00+00:00",
                metadata={"provider": "marketwatch_rss", "source_name": "MarketWatch"},
            ),
        ]

        clusters = cluster_articles(articles)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].publisher_count, 4)
        self.assertEqual(clusters[0].source_count, 4)
        self.assertEqual(len(clusters[0].supporting_links), 4)
        self.assertIn("https://cnbc.com/nvidia-ai-chip", clusters[0].alternate_source_links)

    def test_different_articles_do_not_merge(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/apple-earnings",
                    title="Apple beats estimates after strong iPhone demand",
                ),
                Article(
                    canonical_url="https://publisher-b.com/tesla-factory",
                    title="Tesla pauses factory expansion after regulatory review",
                ),
            ]
        )

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].duplicate_reasons, ())
        self.assertEqual(clusters[1].duplicate_reasons, ())

    def test_different_stories_about_same_ticker_same_day_do_not_merge(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/nvda-chip",
                    title="NVIDIA unveils new AI chip for data centers",
                    snippet="NVDA introduces accelerator hardware.",
                    published_at="2026-06-03T10:00:00+00:00",
                ),
                Article(
                    canonical_url="https://publisher-b.com/nvda-lawsuit",
                    title="NVIDIA faces antitrust lawsuit in Europe",
                    snippet="NVDA legal challenge expands.",
                    published_at="2026-06-03T12:00:00+00:00",
                ),
            ]
        )

        self.assertEqual(len(clusters), 2)

    def test_generic_stock_move_titles_for_different_tickers_do_not_merge(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/asml-popped",
                    title="Why ASML Stock Popped Today",
                    snippet="ASML shares rose after chip demand news.",
                    published_at="2026-06-04T10:00:00+00:00",
                ),
                Article(
                    canonical_url="https://publisher-b.com/micron-popped",
                    title="Why Micron Stock Popped Today",
                    snippet="Micron shares rose after memory chip demand news.",
                    published_at="2026-06-04T10:05:00+00:00",
                ),
            ]
        )

        self.assertEqual(len(clusters), 2)

    def test_generic_drop_titles_for_sandisk_and_micron_do_not_merge(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/sandisk-drop",
                    title="Why Did Sandisk Stock Drop Today?",
                    snippet="SanDisk shares fell after analyst commentary.",
                    published_at="2026-06-04T10:00:00+00:00",
                ),
                Article(
                    canonical_url="https://publisher-b.com/micron-drop",
                    title="Why Did Micron Stock Drop Today?",
                    snippet="Micron shares fell after semiconductor market news.",
                    published_at="2026-06-04T10:05:00+00:00",
                ),
            ]
        )

        self.assertEqual(len(clusters), 2)

    def test_micron_buy_opinion_does_not_merge_with_micron_crash_event(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/micron-buy-before-june",
                    title="Should You Buy Micron Stock Before June 24? History Has a Clear Answer.",
                    snippet="Micron stock opinion before upcoming events.",
                    published_at="2026-06-04T10:00:00+00:00",
                ),
                Article(
                    canonical_url="https://publisher-b.com/micron-crash-today",
                    title="Why Micron Stock Crashed Today",
                    snippet="Micron shares dropped after semiconductor market news.",
                    published_at="2026-06-04T10:15:00+00:00",
                ),
            ]
        )

        self.assertEqual(len(clusters), 2)

    def test_same_ticker_earnings_duplicate_across_publishers_merges(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/micron-earnings",
                    title="Micron beats earnings estimates after strong memory demand",
                    snippet="Micron quarterly results topped expectations.",
                    published_at="2026-06-04T10:00:00+00:00",
                    metadata={"provider": "google_news_rss_search", "source_name": "Publisher A"},
                ),
                Article(
                    canonical_url="https://publisher-b.com/micron-results",
                    title="Micron beats earnings estimates on strong memory demand",
                    snippet="Micron quarterly results topped expectations.",
                    published_at="2026-06-04T10:20:00+00:00",
                    metadata={"provider": "yahoo_finance_rss", "source_name": "Publisher B"},
                ),
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].duplicate_reasons, ("similar_title",))

    def test_similar_ai_chip_announcement_across_publishers_merges(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/nvidia-ai-chip",
                    title="NVIDIA unveils new AI chip for data centers",
                    published_at="2026-06-04T10:00:00+00:00",
                    metadata={"provider": "google_news_rss_search", "source_name": "Publisher A"},
                ),
                Article(
                    canonical_url="https://publisher-b.com/nvidia-ai-chip",
                    title="Nvidia unveils new AI chip for data centres",
                    published_at="2026-06-04T10:05:00+00:00",
                    metadata={"provider": "cnbc_rss", "source_name": "Publisher B"},
                ),
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].duplicate_reasons, ("similar_title",))

    def test_analyst_upgrade_does_not_merge_with_product_announcement(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/nvidia-analyst-upgrade",
                    title="NVIDIA stock upgraded as analyst raises price target",
                    snippet="Analyst rating update for NVDA shares.",
                    published_at="2026-06-04T10:00:00+00:00",
                ),
                Article(
                    canonical_url="https://publisher-b.com/nvidia-ai-chip",
                    title="NVIDIA unveils new AI chip platform for data centers",
                    snippet="Product announcement for NVDA hardware.",
                    published_at="2026-06-04T10:10:00+00:00",
                ),
            ]
        )

        self.assertEqual(len(clusters), 2)

    def test_supporting_links_preserve_publisher_and_provider_names(self):
        clusters = cluster_articles(
            [
                Article(
                    canonical_url="https://publisher-a.com/nvidia-chip",
                    title="NVIDIA unveils new AI chip for data centers",
                    metadata={"provider": "google_news_rss_search", "source_name": "Publisher A"},
                ),
                Article(
                    canonical_url="https://publisher-b.com/nvidia-chip",
                    title="NVIDIA unveils new AI chip for data centers",
                    metadata={"provider": "cnbc_rss", "source_name": "Publisher B"},
                ),
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            [(link.publisher, link.provider) for link in clusters[0].supporting_links],
            [("Publisher A", "google_news_rss_search"), ("Publisher B", "cnbc_rss")],
        )

    def test_optional_semantic_similarity_hook_can_merge(self):
        def fake_semantic(left, right):
            return 0.95

        clusters = cluster_articles(
            [
                Article(canonical_url="https://a.com/one", title="Chip demand lifts Nvidia"),
                Article(canonical_url="https://b.com/two", title="AI processors support NVDA outlook"),
            ],
            semantic_similarity=fake_semantic,
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].duplicate_reasons, ("semantic_similarity",))


if __name__ == "__main__":
    unittest.main()
