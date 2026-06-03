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
                ),
                Article(
                    canonical_url="https://publisher-b.com/markets/aapl-results",
                    title="Apple beats estimates after strong iPhone sales",
                ),
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].duplicate_reasons, ("similar_title",))
        self.assertEqual(
            clusters[0].alternate_source_links,
            ("https://publisher-b.com/markets/aapl-results",),
        )

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
