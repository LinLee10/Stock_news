import unittest

from news_pipeline.dedup import cluster_articles
from news_pipeline.models import Article
from news_pipeline.summaries import (
    BACKGROUND_ONLY,
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

    def test_nvda_does_not_select_marvell_story_when_nvda_story_exists(self):
        nvda = _article(
            "https://reuters.com/technology/nvidia-ai-demand",
            "NVIDIA expands AI data center platform",
            "NVIDIA said customer demand for its AI data center platform remains strong.",
        )
        marvell = _article(
            "https://reuters.com/technology/marvell-results",
            "Marvell earnings beat expectations and joins S&P 500",
            "Marvell raised guidance. NVIDIA was mentioned as another large AI chip supplier.",
        )

        intelligence = _intelligence(nvda, marvell)
        reads = intelligence.ranked_reads_by_ticker["NVDA"]

        self.assertEqual(_read_first(reads).title, nvda.title)
        self.assertEqual(_read(reads, marvell.title).reading_priority, BACKGROUND_ONLY)
        self.assertFalse(_read(reads, marvell.title).ticker_specific)

    def test_nvda_does_not_promote_intel_story_from_related_snippet_mention(self):
        intel = Article(
            canonical_url="https://marketwatch.com/story/intel-customers",
            title="Intel's stock soars as its customer roster grows",
            published_at="2026-06-08T08:00:00+00:00",
            snippet="Intel could serve as a backup chip manufacturer for NVIDIA and Google.",
            metadata={"source_name": "MarketWatch.com - Top Stories"},
        )

        reads = _intelligence(intel).ranked_reads_by_ticker["NVDA"]

        self.assertNotIn(READ_FIRST, [read.reading_priority for read in reads])
        self.assertEqual(reads[0].ticker_match_basis, "snippet_related")
        self.assertFalse(reads[0].ticker_specific)

    def test_meta_does_not_select_broadcom_story(self):
        meta = _article(
            "https://reuters.com/technology/meta-ai-infrastructure",
            "Meta expands AI infrastructure spending",
            "Meta Platforms outlined new AI data center investment.",
        )
        broadcom = _article(
            "https://reuters.com/technology/broadcom-results",
            "Broadcom raises AI revenue guidance",
            "Broadcom raised guidance and listed Meta among several customers.",
        )

        reads = _intelligence(meta, broadcom).ranked_reads_by_ticker["META"]

        self.assertEqual(_read_first(reads).title, meta.title)
        self.assertEqual(_read(reads, broadcom.title).reading_priority, BACKGROUND_ONLY)
        self.assertFalse(_read(reads, broadcom.title).ticker_specific)

    def test_coreweave_does_not_select_core_scientific_story(self):
        coreweave = _article(
            "https://reuters.com/technology/coreweave-contract",
            "CoreWeave signs new AI infrastructure contract",
            "CoreWeave announced a new data center customer contract.",
        )
        core_scientific = _article(
            "https://reuters.com/technology/core-scientific-deal",
            "Core Scientific reviews acquisition proposal",
            "Core Scientific discussed a deal after an earlier CoreWeave proposal.",
        )

        reads = _intelligence(coreweave, core_scientific).ranked_reads_by_ticker["CRWV"]

        self.assertEqual(_read_first(reads).title, coreweave.title)
        self.assertEqual(_read(reads, core_scientific.title).reading_priority, BACKGROUND_ONLY)
        self.assertFalse(_read(reads, core_scientific.title).ticker_specific)

    def test_multi_ticker_roundup_is_background_when_specific_story_exists(self):
        specific = _article(
            "https://reuters.com/technology/nvidia-guidance",
            "NVIDIA raises data center guidance",
            "NVIDIA raised revenue guidance on AI chip demand.",
        )
        roundup = _article(
            "https://finance.yahoo.com/news/ai-stocks-watch",
            "NVIDIA, AMD and Marvell stocks to watch this week",
            "The roundup discusses NVIDIA, AMD and Marvell.",
            source="Yahoo Finance",
        )

        reads = _intelligence(roundup, specific).ranked_reads_by_ticker["NVDA"]

        self.assertEqual(_read_first(reads).title, specific.title)
        self.assertEqual(_read(reads, roundup.title).reading_priority, BACKGROUND_ONLY)

    def test_multi_ticker_roundup_can_be_read_first_only_when_no_specific_story_exists(self):
        roundup = _article(
            "https://finance.yahoo.com/news/ai-stocks-watch",
            "NVIDIA, AMD and Marvell stocks to watch this week",
            "The roundup discusses NVIDIA, AMD and Marvell.",
            source="Yahoo Finance",
        )

        reads = _intelligence(roundup).ranked_reads_by_ticker["AMD"]

        self.assertEqual(_read_first(reads).title, roundup.title)

    def test_prediction_headline_is_not_read_first_by_default(self):
        prediction = _article(
            "https://finance.yahoo.com/news/nvidia-prediction",
            "Prediction: NVIDIA stock will trade at $250",
            "A columnist predicts a future NVIDIA stock price.",
            source="Yahoo Finance",
        )

        reads = _intelligence(prediction).ranked_reads_by_ticker["NVDA"]

        self.assertNotIn(READ_FIRST, [read.reading_priority for read in reads])

    def test_good_stock_to_buy_headline_is_not_read_first_by_default(self):
        opinion = _article(
            "https://finance.yahoo.com/news/nvidia-buy-now",
            "Is NVIDIA a good stock to buy now?",
            "The article gives a general opinion about NVIDIA stock.",
            source="Yahoo Finance",
        )

        reads = _intelligence(opinion).ranked_reads_by_ticker["NVDA"]

        self.assertNotIn(READ_FIRST, [read.reading_priority for read in reads])

    def test_read_first_reason_is_specific_and_not_old_scoring_phrase(self):
        article = _article(
            "https://reuters.com/technology/nvidia-results",
            "NVIDIA earnings beat expectations and guidance rises",
            "NVIDIA reported earnings and raised revenue guidance.",
        )

        intelligence = _intelligence(article)
        read_first = _read_first(intelligence.ranked_reads_by_ticker["NVDA"])
        summary = intelligence.ticker_summaries["NVDA"].ticker_daily_summary

        self.assertEqual(read_first.read_first_reason, "because it covers earnings or guidance")
        self.assertNotIn("strongest deterministic source", summary)

    def test_china_chip_sales_story_does_not_get_earnings_reason(self):
        article = _article(
            "https://investors.com/nvidia-china-sales",
            "NVIDIA stock falls on concerns of backdoor AI chip sales to China",
            "NVIDIA shares fell as investors assessed possible chip export restrictions.",
            source="Investor's Business Daily",
        )

        read_first = _read_first(_intelligence(article).ranked_reads_by_ticker["NVDA"])

        self.assertEqual(
            read_first.read_first_reason,
            "because it covers a material regulatory or legal issue",
        )


def _article(url, title, full_text, *, source="Reuters"):
    return Article(
        canonical_url=url,
        title=title,
        published_at="2026-06-08T08:00:00+00:00",
        full_text=full_text,
        metadata={"source_name": source},
    )


def _intelligence(*articles):
    return build_market_intelligence(
        articles=list(articles),
        clusters=cluster_articles(articles),
        run_date="2026-06-08",
    )


def _read_first(reads):
    return next(read for read in reads if read.reading_priority == READ_FIRST)


def _read(reads, title):
    return next(read for read in reads if read.title == title)


if __name__ == "__main__":
    unittest.main()
