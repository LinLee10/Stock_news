import unittest

from news_pipeline.models import Article, TickerMention
from news_pipeline.sentiment import (
    FinBertSentimentScorer,
    RuleBasedSentimentScorer,
    SentimentScorer,
    analyze_sentiment,
)


class FakeFinBertPipeline:
    def __call__(self, text):
        return [{"label": "positive", "score": 0.91}]


class SentimentTests(unittest.TestCase):
    def test_rule_based_scores_article_with_full_text_basis(self):
        scorer = RuleBasedSentimentScorer()
        article = Article(
            canonical_url="https://example.com/aapl",
            title="Apple update",
            full_text="Apple beats estimates with strong profit growth.",
        )

        result = scorer.score_article(article)

        self.assertEqual(result.label, "positive")
        self.assertGreater(result.score, 0)
        self.assertEqual(result.model_name, "rule_based_fake")
        self.assertEqual(result.sentiment_basis, "full_text")
        self.assertIn("Apple beats estimates", result.evidence_window)

    def test_rule_based_uses_snippet_then_title_basis(self):
        scorer = RuleBasedSentimentScorer()
        snippet_article = Article(
            canonical_url="https://example.com/snippet",
            title="Neutral title",
            snippet="Company faces weak demand and a loss.",
        )
        title_article = Article(
            canonical_url="https://example.com/title",
            title="Company beats expectations",
        )

        self.assertEqual(scorer.score_article(snippet_article).sentiment_basis, "snippet")
        self.assertEqual(scorer.score_article(title_article).sentiment_basis, "title")

    def test_rule_based_scores_ticker_mention_window(self):
        scorer = RuleBasedSentimentScorer()
        article = Article(
            canonical_url="https://example.com/nvda",
            title="Markets update",
            full_text="Macro news was mixed. NVDA beats expectations with record growth today.",
        )
        mention = TickerMention(article_id="art_1", ticker="NVDA", confidence=0.9)

        result = scorer.score_ticker_mentions(article, [mention])[0]

        self.assertEqual(result.ticker, "NVDA")
        self.assertEqual(result.label, "positive")
        self.assertIn("NVDA beats expectations", result.evidence_window)

    def test_finbert_does_not_load_model_on_import_or_default_init(self):
        scorer = FinBertSentimentScorer()
        article = Article(canonical_url="https://example.com/a", title="Company beats estimates")

        with self.assertRaises(RuntimeError):
            scorer.score_article(article)

    def test_finbert_supports_injected_fake_pipeline(self):
        scorer = FinBertSentimentScorer(pipeline=FakeFinBertPipeline(), model_name="fake-finbert")
        article = Article(canonical_url="https://example.com/a", title="Company beats estimates")

        result = scorer.score_article(article)

        self.assertEqual(result.label, "positive")
        self.assertEqual(result.score, 0.91)
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.model_name, "fake-finbert")

    def test_analyze_sentiment_compatibility_helper(self):
        result = analyze_sentiment("art_1", "Company misses estimates after weak demand.", "title")

        self.assertEqual(result.label, "negative")
        self.assertEqual(result.basis, "title")
        self.assertEqual(result.model, "rule_based_fake")

    def test_protocol_is_runtime_assignable_by_shape(self):
        scorer: SentimentScorer = RuleBasedSentimentScorer()
        self.assertEqual(scorer.model_name, "rule_based_fake")


if __name__ == "__main__":
    unittest.main()
