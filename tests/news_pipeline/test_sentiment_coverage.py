import unittest

from news_pipeline.dedup import cluster_articles
from news_pipeline.models import Article
from news_pipeline.sentiment_coverage import build_weighted_sentiment_coverage


class SentimentCoverageTests(unittest.TestCase):
    def test_weighted_sentiment_is_not_equal_average(self):
        strong = _article(
            "https://reuters.com/nvidia-growth",
            "NVIDIA earnings beat estimates",
            full_text="NVIDIA reported strong growth, profit and record revenue.",
            source="Reuters",
        )
        weak = _article(
            "https://stocktwits.com/nvidia-risk",
            "NVIDIA stock risk",
            snippet="NVIDIA shares face weak demand and losses.",
            source="Stocktwits",
        )

        inputs, coverage = build_weighted_sentiment_coverage(
            articles=(strong, weak),
            clusters=cluster_articles((strong, weak)),
            run_date="2026-06-08",
        )
        nvda = [item for item in inputs if item.ticker == "NVDA"]
        equal_average = sum(item.sentiment_raw for item in nvda) / len(nvda)

        self.assertNotEqual(coverage["NVDA"].weighted_sentiment, round(equal_average, 4))

    def test_related_mentions_do_not_dominate_ticker_sentiment(self):
        nvda = _article(
            "https://reuters.com/nvidia",
            "NVIDIA reports strong AI demand",
            full_text="NVIDIA reported strong growth and record demand.",
            source="Reuters",
        )
        marvell = _article(
            "https://reuters.com/marvell",
            "Marvell reports weak quarterly results",
            snippet="Marvell reported losses. NVIDIA was mentioned as a competitor.",
            source="Reuters",
        )

        inputs, coverage = build_weighted_sentiment_coverage(
            articles=(nvda, marvell),
            clusters=cluster_articles((nvda, marvell)),
            run_date="2026-06-08",
        )
        related = next(item for item in inputs if item.ticker == "NVDA" and "marvell" in item.canonical_url)
        primary = next(item for item in inputs if item.ticker == "NVDA" and "nvidia" in item.canonical_url)

        self.assertLess(related.sentiment_weight, primary.sentiment_weight)
        self.assertGreater(coverage["NVDA"].weighted_sentiment, 0)
        self.assertEqual(coverage["NVDA"].top_positive_cluster, nvda.title)

    def test_coverage_is_weak_for_mostly_snippets(self):
        articles = tuple(
            _article(
                f"https://example.com/nvidia-{index}",
                f"NVIDIA stock update {index}",
                snippet="NVIDIA stock news from a publisher snippet.",
            )
            for index in range(3)
        )

        _inputs, coverage = build_weighted_sentiment_coverage(
            articles=articles,
            clusters=cluster_articles(articles),
            run_date="2026-06-08",
        )

        self.assertEqual(coverage["NVDA"].sentiment_coverage_grade, "weak")

    def test_coverage_improves_with_high_confidence_full_text(self):
        articles = tuple(
            _article(
                f"https://reuters.com/nvidia-{index}",
                f"NVIDIA earnings update {index}",
                full_text="NVIDIA reported earnings, revenue and strong demand.",
                source="Reuters",
            )
            for index in range(3)
        )

        _inputs, coverage = build_weighted_sentiment_coverage(
            articles=articles,
            clusters=cluster_articles(articles),
            run_date="2026-06-08",
        )

        self.assertEqual(coverage["NVDA"].sentiment_coverage_grade, "strong")


def _article(url, title, *, full_text=None, snippet=None, source="Yahoo Finance"):
    return Article(
        canonical_url=url,
        title=title,
        published_at="2026-06-08T08:00:00+00:00",
        full_text=full_text,
        snippet=snippet,
        metadata={"source_name": source},
    )


if __name__ == "__main__":
    unittest.main()
