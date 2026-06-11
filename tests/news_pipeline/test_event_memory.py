import csv
import json
from pathlib import Path
import tempfile
import unittest

from news_pipeline.article_fetch import ArticleFetchSummary
from news_pipeline.dedup import cluster_articles
from news_pipeline.event_memory import (
    DEFAULT_EVENT_MEMORY_LOOKBACK_DAYS,
    EVENT_SIMILARITY_THRESHOLD,
    EventMemoryRecord,
    PriorEventMemorySnapshot,
    alpha_vantage_selection_history,
    build_event_pair_review,
    build_event_memory_records,
    compare_event_memory,
    event_date_bucket,
    event_identity_fingerprint,
    load_latest_prior_event_memory,
    normalize_event_title,
    write_event_pair_review_artifacts,
    write_event_memory_artifacts,
)
from news_pipeline.models import Article, RunResult, TickerMention
from news_pipeline.storage import SQLiteStore


def _memory_record(
    ticker: str,
    canonical_url: str,
    sentiment: float,
    *,
    title: str = "Example event",
    published_at: str = "2026-06-09T12:00:00Z",
    company: str | None = None,
    event_type: str = "company_news",
    article_type: str = "company_news",
    source_family: str = "direct_news_publisher",
) -> EventMemoryRecord:
    company = company or ticker
    normalized_title = normalize_event_title(
        title,
        ticker=ticker,
        company=company,
    )
    date_bucket = event_date_bucket(published_at)
    return EventMemoryRecord(
        article_id=f"{ticker}-{canonical_url.rsplit('/', 1)[-1]}",
        canonical_url=canonical_url,
        published_at=published_at,
        ticker=ticker,
        company=company,
        source_provider="example",
        source_family=source_family,
        article_type=article_type,
        cluster_id=f"cluster-{ticker}",
        ticker_match_confidence=1.0,
        extraction_basis="snippet",
        extraction_quality_grade="snippet",
        internal_sentiment=sentiment,
        external_sentiment_provider=None,
        external_sentiment=None,
        event_type=event_type,
        event_title=title,
        normalized_event_title=normalized_title,
        published_date_bucket=date_bucket,
        event_identity_fingerprint=event_identity_fingerprint(
            ticker=ticker,
            normalized_title=normalized_title,
            event_type=event_type,
            article_type=article_type,
            company=company,
            source_family=source_family,
            published_date_bucket=date_bucket,
        ),
        event_summary="Example event",
        run_id="current-run",
        run_date="2026-06-09",
    )


def _write_prior_run(
    runs_dir: Path,
    *,
    run_date: str,
    record: EventMemoryRecord,
) -> None:
    output_dir = runs_dir / run_date
    output_dir.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(output_dir / "news_pipeline.sqlite3")
    try:
        store.initialize_schema()
        run_id = f"run-{run_date}"
        store.record_run(
            RunResult(run_id=run_id, status="completed"),
            run_date=run_date,
        )
        store.record_event_memory(
            {
                **record.as_dict(),
                "run_id": run_id,
                "run_date": run_date,
            }
        )
    finally:
        store.close()


class EventMemoryTests(unittest.TestCase):
    def test_event_title_normalization_removes_filler_and_ticker_noise(self):
        normalized = normalize_event_title(
            "Breaking News: NVDA / NVIDIA Corporation Stock Update - "
            "NVIDIA unveils Blackwell AI chips",
            ticker="NVDA",
            company="NVIDIA Corporation",
        )

        self.assertEqual(normalized, "launch blackwell ai chips")

    def test_no_prior_run_reports_history_building(self):
        comparison = compare_event_memory(
            (_memory_record("NVDA", "https://example.com/current", 0.2),),
            PriorEventMemorySnapshot(available=False),
        )

        self.assertFalse(comparison.prior_run_available)
        self.assertEqual(comparison.history_status, "history_building")
        self.assertEqual(comparison.new_events_since_prior_run, 0)
        self.assertEqual(comparison.repeated_events_from_prior_run, 0)
        self.assertEqual(
            comparison.event_identity_method_counts,
            {
                "exact_url_repeat": 0,
                "fuzzy_event_repeat": 0,
                "likely_new_event": 0,
            },
        )
        self.assertEqual(
            comparison.event_similarity_threshold,
            EVENT_SIMILARITY_THRESHOLD,
        )
        self.assertEqual(comparison.sentiment_change_since_prior_run, {})

    def test_same_ticker_and_canonical_url_matches_exactly(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            run_id="prior-run",
            run_date="2026-06-08",
            records=(
                _memory_record(
                    "NVDA",
                    "https://example.com/repeated",
                    0.1,
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/repeated",
                    0.4,
                ),
            ),
            prior,
        )

        self.assertTrue(comparison.prior_run_available)
        self.assertEqual(comparison.history_status, "compared_to_prior_run")
        self.assertEqual(comparison.new_events_since_prior_run, 0)
        self.assertEqual(comparison.repeated_events_from_prior_run, 1)
        self.assertEqual(comparison.exact_repeated_events_from_prior_run, 1)
        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 0)
        self.assertEqual(
            comparison.event_identity_matches[0]["category"],
            "exact_url_repeat",
        )
        self.assertEqual(
            comparison.sentiment_change_since_prior_run["NVDA"]["change"],
            0.3,
        )

    def test_same_ticker_different_url_similar_title_matches_fuzzily(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            records=(
                _memory_record(
                    "NVDA",
                    "https://publisher-a.example.com/blackwell",
                    0.1,
                    title="NVIDIA launches new Blackwell AI chips for data centers",
                    company="NVIDIA",
                    published_at="2026-06-08T23:30:00Z",
                    event_type="product_news",
                    article_type="product_or_ai_or_chip_news",
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://publisher-b.example.com/nvidia-chip",
                    0.2,
                    title="Nvidia unveils Blackwell chips for AI data centers",
                    company="NVIDIA",
                    published_at="2026-06-09T08:00:00Z",
                    event_type="product_news",
                    article_type="product_or_ai_or_chip_news",
                    source_family="external_market_news_api",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.exact_repeated_events_from_prior_run, 0)
        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 1)
        self.assertEqual(comparison.new_events_since_prior_run, 0)
        self.assertEqual(
            comparison.event_identity_matches[0]["category"],
            "fuzzy_event_repeat",
        )
        self.assertGreaterEqual(
            comparison.event_identity_matches[0]["title_similarity"],
            EVENT_SIMILARITY_THRESHOLD,
        )
        self.assertEqual(
            comparison.event_memory_lookback_days,
            DEFAULT_EVENT_MEMORY_LOOKBACK_DAYS,
        )

    def test_different_ticker_with_similar_title_does_not_match(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            records=(
                _memory_record(
                    "AMD",
                    "https://example.com/amd-chip",
                    0.1,
                    title="AMD launches new AI chip for data centers",
                    company="Advanced Micro Devices",
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/nvidia-chip",
                    0.2,
                    title="NVIDIA launches new AI chip for data centers",
                    company="NVIDIA",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 0)
        self.assertEqual(comparison.new_events_since_prior_run, 1)

    def test_same_ticker_with_unrelated_title_does_not_match(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            records=(
                _memory_record(
                    "NVDA",
                    "https://example.com/earnings",
                    0.1,
                    title="NVIDIA reports quarterly earnings and raises guidance",
                    company="NVIDIA",
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/product",
                    0.2,
                    title="NVIDIA launches robotics software platform",
                    company="NVIDIA",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 0)
        self.assertEqual(comparison.new_events_since_prior_run, 1)

    def test_three_day_lookback_matches_similar_event_across_different_urls(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            lookback_days=3,
            records=(
                _memory_record(
                    "NVDA",
                    "https://example.com/prior-product",
                    0.1,
                    title="NVIDIA launches Blackwell AI chips for data centers",
                    company="NVIDIA",
                    published_at="2026-06-06T12:00:00Z",
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/current-product",
                    0.2,
                    title="Nvidia unveils Blackwell chips for AI data centers",
                    company="NVIDIA",
                    published_at="2026-06-09T12:00:00Z",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 1)
        self.assertEqual(comparison.new_events_since_prior_run, 0)

    def test_event_outside_lookback_is_not_matched(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            lookback_days=3,
            records=(
                _memory_record(
                    "NVDA",
                    "https://example.com/prior-product",
                    0.1,
                    title="NVIDIA launches Blackwell AI chips for data centers",
                    company="NVIDIA",
                    published_at="2026-06-01T12:00:00Z",
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/current-product",
                    0.2,
                    title="Nvidia unveils Blackwell chips for AI data centers",
                    company="NVIDIA",
                    published_at="2026-06-09T12:00:00Z",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 0)
        self.assertEqual(comparison.new_events_since_prior_run, 1)

    def test_exact_url_match_wins_before_fuzzy_matching(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            records=(
                _memory_record(
                    "NVDA",
                    "https://example.com/exact",
                    0.1,
                    title="NVIDIA launches Blackwell AI chips for data centers",
                    company="NVIDIA",
                ).as_dict(),
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/fuzzy",
                    0.2,
                    title="Nvidia unveils Blackwell chips for AI data centers",
                    company="NVIDIA",
                ),
                _memory_record(
                    "NVDA",
                    "https://example.com/exact",
                    0.3,
                    title="NVIDIA executive comments on software demand",
                    company="NVIDIA",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.exact_repeated_events_from_prior_run, 1)
        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 0)
        self.assertEqual(comparison.new_events_since_prior_run, 1)
        self.assertEqual(
            comparison.event_identity_matches[1]["category"],
            "exact_url_repeat",
        )

    def test_loader_considers_multiple_prior_runs_inside_lookback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            _write_prior_run(
                runs_dir,
                run_date="2026-06-08",
                record=_memory_record(
                    "NVDA",
                    "https://example.com/june-8",
                    0.1,
                ),
            )
            _write_prior_run(
                runs_dir,
                run_date="2026-06-09",
                record=_memory_record(
                    "META",
                    "https://example.com/june-9",
                    0.2,
                ),
            )
            _write_prior_run(
                runs_dir,
                run_date="2026-06-07",
                record=_memory_record(
                    "AMD",
                    "https://example.com/outside-window",
                    0.3,
                ),
            )

            snapshot = load_latest_prior_event_memory(
                runs_dir=runs_dir,
                run_date="2026-06-11",
                lookback_days=3,
            )

        self.assertTrue(snapshot.available)
        self.assertEqual(
            [run["run_date"] for run in snapshot.prior_runs],
            ["2026-06-09", "2026-06-08"],
        )
        self.assertEqual(len(snapshot.records), 2)
        self.assertEqual(
            {record["_prior_run_date"] for record in snapshot.records},
            {"2026-06-08", "2026-06-09"},
        )

    def test_loader_reconstructs_records_from_legacy_run_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            output_dir = runs_dir / "2026-06-09"
            output_dir.mkdir(parents=True)
            store = SQLiteStore(output_dir / "news_pipeline.sqlite3")
            try:
                store.initialize_schema()
                run_id = "run-2026-06-09"
                store.record_run(
                    RunResult(run_id=run_id, status="completed"),
                    run_date="2026-06-09",
                )
                article = Article(
                    article_id="legacy-nvda-story",
                    canonical_url="https://example.com/legacy-nvda-story",
                    title="NVIDIA launches Blackwell AI chips for data centers",
                    snippet="NVIDIA announced a new data center chip.",
                    published_at="2026-06-09T12:00:00Z",
                )
                article_id = store.add_run_article(run_id, article)
                store.add_ticker_mention(
                    TickerMention(
                        article_id=article_id,
                        ticker="NVDA",
                        company_name="NVIDIA",
                        confidence=0.95,
                        basis="title",
                    ),
                    run_id=run_id,
                )
                store.connection.execute("DROP TABLE event_memory")
                store.connection.commit()
            finally:
                store.close()

            snapshot = load_latest_prior_event_memory(
                runs_dir=runs_dir,
                run_date="2026-06-11",
                lookback_days=3,
            )

        self.assertTrue(snapshot.available)
        self.assertEqual(len(snapshot.records), 1)
        self.assertEqual(snapshot.records[0]["ticker"], "NVDA")
        self.assertEqual(
            snapshot.records[0]["comparison_title"],
            "NVIDIA launches Blackwell AI chips for data centers",
        )
        self.assertEqual(
            snapshot.prior_runs[0]["event_memory_basis"],
            "legacy_run_tables",
        )

    def test_loader_reports_history_building_when_prior_runs_are_outside_lookback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            _write_prior_run(
                runs_dir,
                run_date="2026-06-07",
                record=_memory_record(
                    "NVDA",
                    "https://example.com/outside-window",
                    0.1,
                ),
            )

            snapshot = load_latest_prior_event_memory(
                runs_dir=runs_dir,
                run_date="2026-06-11",
                lookback_days=3,
            )
            comparison = compare_event_memory(
                (
                    _memory_record(
                        "NVDA",
                        "https://example.com/current",
                        0.2,
                        published_at="2026-06-11T12:00:00Z",
                    ),
                ),
                snapshot,
            )

        self.assertFalse(snapshot.available)
        self.assertEqual(comparison.history_status, "history_building")
        self.assertEqual(comparison.prior_runs_considered, ())

    def test_older_prior_run_can_supply_match_when_latest_cannot(self):
        prior = PriorEventMemorySnapshot(
            available=True,
            run_id="run-2026-06-09",
            run_date="2026-06-09",
            lookback_days=3,
            prior_runs=(
                {"run_id": "run-2026-06-09", "run_date": "2026-06-09"},
                {"run_id": "run-2026-06-08", "run_date": "2026-06-08"},
            ),
            records=(
                {
                    **_memory_record(
                        "NVDA",
                        "https://example.com/latest-unrelated",
                        0.1,
                        title="NVIDIA executive comments on software demand",
                        company="NVIDIA",
                        published_at="2026-06-09T12:00:00Z",
                    ).as_dict(),
                    "_prior_run_id": "run-2026-06-09",
                    "_prior_run_date": "2026-06-09",
                },
                {
                    **_memory_record(
                        "NVDA",
                        "https://example.com/older-related",
                        0.2,
                        title="NVIDIA launches Blackwell AI chips for data centers",
                        company="NVIDIA",
                        published_at="2026-06-08T12:00:00Z",
                    ).as_dict(),
                    "_prior_run_id": "run-2026-06-08",
                    "_prior_run_date": "2026-06-08",
                },
            ),
        )
        comparison = compare_event_memory(
            (
                _memory_record(
                    "NVDA",
                    "https://example.com/current-related",
                    0.3,
                    title="Nvidia unveils Blackwell chips for AI data centers",
                    company="NVIDIA",
                    published_at="2026-06-11T12:00:00Z",
                ),
            ),
            prior,
        )

        self.assertEqual(comparison.fuzzy_repeated_events_from_prior_run, 1)
        self.assertEqual(
            comparison.event_identity_matches[0]["prior_run_date"],
            "2026-06-08",
        )
        self.assertEqual(len(comparison.prior_runs_considered), 2)
        self.assertEqual(comparison.prior_event_records_considered, 2)

    def test_event_pair_review_writes_all_review_categories(self):
        current_records, prior = _review_fixture()
        comparison = compare_event_memory(current_records, prior)
        review = build_event_pair_review(
            current_records=current_records,
            prior_snapshot=prior,
            comparison=comparison,
        )

        methods = {row["match_method"] for row in review.rows}
        self.assertEqual(
            methods,
            {
                "exact_url_repeat",
                "fuzzy_event_repeat",
                "near_miss",
                "likely_new_event",
            },
        )
        self.assertTrue(
            all(
                row["threshold_used"] == EVENT_SIMILARITY_THRESHOLD
                for row in review.rows
            )
        )
        self.assertTrue(
            all(row["recommended_label"] == "" for row in review.rows)
        )
        near_miss = next(
            row for row in review.rows
            if row["match_method"] == "near_miss"
        )
        self.assertLess(
            near_miss["similarity_score"],
            near_miss["threshold_used"],
        )
        self.assertGreaterEqual(
            near_miss["similarity_score"],
            near_miss["threshold_used"] - 0.08,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path = write_event_pair_review_artifacts(
                review,
                output_dir=temp_dir,
            )
            json_rows = json.loads(
                Path(json_path).read_text(encoding="utf-8")
            )
            with Path(csv_path).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(json_rows), 4)
        self.assertEqual(len(csv_rows), 4)
        self.assertEqual(
            {row["match_method"] for row in json_rows},
            methods,
        )
        self.assertTrue(
            all(row["recommended_label"] == "" for row in csv_rows)
        )

    def test_event_pair_review_cap_is_respected(self):
        current_records, prior = _review_fixture()
        comparison = compare_event_memory(current_records, prior)
        review = build_event_pair_review(
            current_records=current_records,
            prior_snapshot=prior,
            comparison=comparison,
            max_pairs=2,
        )

        self.assertEqual(len(review.rows), 2)
        self.assertEqual(review.diagnostics()["event_pair_review_count"], 2)

    def test_event_pair_review_near_miss_margin_is_respected(self):
        current_records, prior = _review_fixture()
        comparison = compare_event_memory(current_records, prior)
        narrow_review = build_event_pair_review(
            current_records=current_records,
            prior_snapshot=prior,
            comparison=comparison,
            near_miss_margin=0.01,
        )
        broad_review = build_event_pair_review(
            current_records=current_records,
            prior_snapshot=prior,
            comparison=comparison,
            near_miss_margin=0.08,
        )

        self.assertEqual(
            narrow_review.diagnostics()["near_miss_review_pairs"],
            0,
        )
        self.assertEqual(
            broad_review.diagnostics()["near_miss_review_pairs"],
            1,
        )

    def test_selection_history_identifies_google_dominance_and_benchmark_gaps(self):
        snapshot = PriorEventMemorySnapshot(
            available=True,
            records=(
                {
                    **_memory_record(
                        "META",
                        "https://example.com/google-1",
                        0.1,
                    ).as_dict(),
                    "source_family": "google_news_backstop",
                },
                {
                    **_memory_record(
                        "META",
                        "https://example.com/google-2",
                        0.1,
                    ).as_dict(),
                    "source_family": "google_news_backstop",
                },
                {
                    **_memory_record(
                        "META",
                        "https://example.com/direct",
                        0.1,
                    ).as_dict(),
                    "source_family": "direct_news_publisher",
                },
                {
                    **_memory_record(
                        "MU",
                        "https://example.com/benchmark",
                        0.2,
                    ).as_dict(),
                    "external_sentiment_provider": "alpha_vantage_news",
                },
            ),
        )

        history = alpha_vantage_selection_history(
            snapshot,
            minimum_articles_per_ticker=4,
        )

        self.assertEqual(history["benchmark_coverage_counts"], {"MU": 1})
        self.assertEqual(history["google_dominated_tickers"], ["META"])
        self.assertEqual(history["weak_coverage_tickers"], ["META", "MU"])

    def test_event_memory_is_written_to_sqlite_and_artifacts(self):
        article = Article(
            article_id="sec-nvda-8k",
            canonical_url="https://www.sec.gov/example/nvda-8k",
            title="NVIDIA files 8-K with the SEC",
            snippet="NVIDIA filed a material current report.",
            published_at="2026-06-09T00:00:00Z",
            metadata={
                "provider": "sec_edgar",
                "source_provider": "sec_edgar",
                "source_family": "regulatory_official",
                "symbols": ["NVDA"],
                "ticker": "NVDA",
                "filing_form_type": "8-K",
                "filing_event_type": "material_event",
                "sec_event_summary": "NVIDIA filed a material current report.",
                "ticker_match_confidence": 1.0,
            },
        )
        clusters = tuple(cluster_articles((article,)))
        records = build_event_memory_records(
            articles=(article,),
            clusters=clusters,
            article_fetch_summary=ArticleFetchSummary(enabled=False),
            article_ids_by_url={article.canonical_url: article.article_id},
            run_id="run-2026-06-09",
            run_date="2026-06-09",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].ticker, "NVDA")
        self.assertEqual(records[0].event_type, "material_event")
        self.assertEqual(records[0].event_title, "NVIDIA files 8-K with the SEC")
        self.assertEqual(records[0].published_date_bucket, "2026-06-09")
        self.assertTrue(records[0].normalized_event_title)
        self.assertEqual(len(records[0].event_identity_fingerprint), 64)
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path = write_event_memory_artifacts(
                records,
                output_dir=temp_dir,
            )
            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["article_id"], "sec-nvda-8k")
            self.assertTrue(Path(csv_path).exists())

            store = SQLiteStore(Path(temp_dir) / "pipeline.sqlite3")
            try:
                store.initialize_schema()
                store.record_event_memory(records[0].as_dict())
                stored = store.list_event_memory("run-2026-06-09")
            finally:
                store.close()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["event_type"], "material_event")
            self.assertEqual(
                stored[0]["normalized_event_title"],
                records[0].normalized_event_title,
            )
            self.assertEqual(
                stored[0]["event_identity_fingerprint"],
                records[0].event_identity_fingerprint,
            )

    def test_alpha_vantage_event_memory_uses_ticker_specific_sentiment(self):
        article = Article(
            article_id="alpha-multi-ticker",
            canonical_url="https://example.com/alpha-multi-ticker",
            title="NVIDIA and AMD expand AI chip programs",
            snippet="NVIDIA and AMD announced separate chip updates.",
            published_at="2026-06-09T12:00:00Z",
            metadata={
                "provider": "alpha_vantage_news",
                "source_provider": "alpha_vantage_news",
                "source_family": "external_market_news_api",
                "symbols": ["NVDA", "AMD"],
                "external_sentiment_provider": "alpha_vantage_news",
                "external_sentiment": 0.7,
                "ticker_sentiment": [
                    {
                        "ticker": "NVDA",
                        "ticker_sentiment_score": "0.7",
                    },
                    {
                        "ticker": "AMD",
                        "ticker_sentiment_score": "-0.2",
                    },
                ],
            },
        )
        records = build_event_memory_records(
            articles=(article,),
            clusters=tuple(cluster_articles((article,))),
            article_fetch_summary=ArticleFetchSummary(enabled=False),
            article_ids_by_url={article.canonical_url: article.article_id},
            run_id="run-2026-06-09",
            run_date="2026-06-09",
        )

        scores = {record.ticker: record.external_sentiment for record in records}
        self.assertEqual(scores["NVDA"], 0.7)
        self.assertEqual(scores["AMD"], -0.2)


def _review_fixture():
    prior_records = (
        _memory_record(
            "NVDA",
            "https://example.com/nvda-exact",
            0.1,
            title="NVIDIA reports quarterly earnings",
            company="NVIDIA",
        ).as_dict(),
        _memory_record(
            "AMD",
            "https://example.com/amd-prior",
            0.1,
            title="AMD launches Blackwell AI chips for data centers",
            company="AMD",
        ).as_dict(),
        _memory_record(
            "META",
            "https://example.com/meta-prior",
            0.1,
            title="Meta launches Blackwell AI chips for data centers",
            company="Meta",
        ).as_dict(),
        _memory_record(
            "MU",
            "https://example.com/mu-prior",
            0.1,
            title="Micron reports quarterly earnings",
            company="Micron Technology",
        ).as_dict(),
    )
    current_records = (
        _memory_record(
            "NVDA",
            "https://example.com/nvda-exact",
            0.2,
            title="NVIDIA reports quarterly earnings",
            company="NVIDIA",
        ),
        _memory_record(
            "AMD",
            "https://example.com/amd-current",
            0.2,
            title="AMD unveils Blackwell AI chips for data centers",
            company="AMD",
        ),
        _memory_record(
            "META",
            "https://example.com/meta-current",
            0.2,
            title="Meta announces Blackwell AI chips for cloud customers",
            company="Meta",
        ),
        _memory_record(
            "MU",
            "https://example.com/mu-current",
            0.2,
            title="Micron opens a new memory fabrication facility",
            company="Micron Technology",
        ),
    )
    return current_records, PriorEventMemorySnapshot(
        available=True,
        run_id="prior-run",
        run_date="2026-06-08",
        records=prior_records,
    )


if __name__ == "__main__":
    unittest.main()
