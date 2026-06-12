import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from news_pipeline.cli import build_parser, main
from news_pipeline.dedup import cluster_articles
from news_pipeline.event_memory import (
    EVENT_SIMILARITY_THRESHOLD,
    EventMemoryRecord,
    PriorEventMemorySnapshot,
    compare_event_memory,
    event_date_bucket,
    event_identity_fingerprint,
    normalize_event_title,
)
from news_pipeline.llm_briefing import (
    LLM_BRIEFING_TIERS,
    LlmBriefingTier,
    briefing_response_schema,
    build_llm_briefing_input,
    classify_llm_source_quality,
    estimate_llm_briefing_cost,
    llm_event_priority_score,
)
from news_pipeline.models import Article
from news_pipeline.source_quality import UNKNOWN_SOURCE_LABEL


class CountingLlmClient:
    def __init__(self):
        self.call_count = 0

    def create_briefing(self, **kwargs):
        self.call_count += 1
        return {
            "portfolio_pulse": "Mocked response",
            "no_buy_sell_advice": "No investment advice.",
        }


class LlmBriefingTests(unittest.TestCase):
    def test_tier_config_values(self):
        daily = LLM_BRIEFING_TIERS["daily"]
        strong = LLM_BRIEFING_TIERS["strong"]
        premium = LLM_BRIEFING_TIERS["premium"]

        self.assertEqual(
            (daily.model, daily.max_input_tokens, daily.max_output_tokens),
            ("gpt-5.4-mini", 60_000, 5_000),
        )
        self.assertEqual(daily.cost_cap_usd, 0.25)
        self.assertEqual(
            (strong.model, strong.max_input_tokens, strong.max_output_tokens),
            ("gpt-5.4", 80_000, 7_000),
        )
        self.assertEqual(strong.cost_cap_usd, 1.00)
        self.assertEqual(
            (
                premium.model,
                premium.max_input_tokens,
                premium.max_output_tokens,
            ),
            ("gpt-5.5", 100_000, 10_000),
        )
        self.assertEqual(premium.cost_cap_usd, 3.00)

    def test_cost_cap_blocks_oversized_request(self):
        restrictive = LlmBriefingTier(
            name="test",
            model="test-model",
            purpose="test",
            max_input_tokens=100_000,
            max_output_tokens=1_000,
            cost_cap_usd=0.01,
            input_cost_per_million_tokens=100.0,
            output_cost_per_million_tokens=100.0,
        )

        estimate = estimate_llm_briefing_cost(
            {"event_packets": [{"title": "x" * 30_000}]},
            restrictive,
        )

        self.assertTrue(estimate.blocked)
        self.assertEqual(
            estimate.blocked_reason,
            "estimated_cost_exceeds_tier_cap",
        )
        self.assertGreater(estimate.estimated_cost_usd, 0.01)

    def test_structured_packet_has_required_evidence_without_raw_article_text(self):
        article = Article(
            canonical_url="https://example.com/nvidia-launch",
            title="NVIDIA launches a new AI data center platform",
            full_text="private full article body that must not be included",
            snippet="NVIDIA introduced a new data center platform.",
            published_at="2026-06-11T12:00:00Z",
            metadata={
                "provider": "example",
                "source_name": "Reuters",
                "source_family": "direct_news_publisher",
            },
        )
        cluster = cluster_articles((article,))[0]
        record = replace(
            _record(),
            cluster_id=cluster.cluster_id,
            canonical_url=article.canonical_url,
            event_title=article.title,
        )
        comparison = compare_event_memory(
            (record,),
            PriorEventMemorySnapshot(available=False),
        )

        payload = build_llm_briefing_input(
            records=(record,),
            clusters=(cluster,),
            comparison=comparison,
            run_date="2026-06-12",
            tier=LLM_BRIEFING_TIERS["daily"],
        )
        packet = payload["event_packets"][0]
        serialized = json.dumps(payload)

        self.assertEqual(packet["ticker"], "NVDA")
        self.assertEqual(packet["company"], "NVIDIA")
        self.assertEqual(packet["portfolio_watchlist_status"], "portfolio")
        self.assertIn("event_type", packet)
        self.assertIn("new_repeated_status", packet)
        self.assertIn("similarity_score", packet)
        self.assertIn("days_active", packet)
        self.assertIn("sentiment_score", packet)
        self.assertIn("sentiment_change", packet)
        self.assertIn("source_quality", packet)
        self.assertIn("source_count", packet)
        self.assertIn("top_titles", packet)
        self.assertIn("urls", packet)
        self.assertIn("uncertainty", packet)
        self.assertNotIn(article.full_text, serialized)
        self.assertNotIn('"full_text"', serialized)
        self.assertIn("Do not provide buy", serialized)

    def test_response_schema_requires_no_buy_sell_advice_section(self):
        schema = briefing_response_schema()

        self.assertIn("no_buy_sell_advice", schema["required"])
        self.assertIn("portfolio_pulse", schema["properties"])
        self.assertIn(
            "holdings_with_meaningful_change",
            schema["properties"],
        )
        self.assertIn("watchlist_activations", schema["properties"])
        self.assertIn("top_monitor_names", schema["properties"])
        self.assertIn("repeated_vs_new_events", schema["properties"])
        self.assertIn("uncertainty_notes", schema["properties"])
        self.assertIn("evidence", schema["properties"])

    def test_priority_score_favors_portfolio_over_watchlist_noise(self):
        portfolio_score, _ = llm_event_priority_score(
            _priority_packet(
                group="portfolio",
                article_type="product_or_ai_or_chip_news",
            )
        )
        watchlist_score, _ = llm_event_priority_score(
            _priority_packet(
                group="watchlist",
                article_type="generic_buy_sell_hold_opinion",
            )
        )

        self.assertGreater(portfolio_score, watchlist_score)

    def test_repeated_developing_event_beats_generic_isolated_article(self):
        repeated_score, _ = llm_event_priority_score(
            _priority_packet(
                group="watchlist",
                event_status="fuzzy_event_repeat",
                article_type="product_or_ai_or_chip_news",
            )
        )
        isolated_score, _ = llm_event_priority_score(
            _priority_packet(
                group="watchlist",
                event_status="likely_new_event",
                article_type="generic_buy_sell_hold_opinion",
            )
        )

        self.assertGreater(repeated_score, isolated_score)

    def test_official_signal_boosts_priority(self):
        normal_score, _ = llm_event_priority_score(_priority_packet())
        official_score, reasons = llm_event_priority_score(
            _priority_packet(official=True)
        )

        self.assertGreater(official_score, normal_score)
        self.assertIn("official_or_sec_signal", reasons)

    def test_uncertainty_reduces_priority(self):
        clear_score, _ = llm_event_priority_score(_priority_packet())
        uncertain_score, reasons = llm_event_priority_score(
            _priority_packet(
                uncertainty=(
                    "snippet_based_sentiment",
                    "event_type_unknown",
                )
            )
        )

        self.assertLess(uncertain_score, clear_score)
        self.assertIn("uncertainty_penalty", reasons)

    def test_packet_and_per_ticker_caps_are_respected(self):
        records = tuple(
            replace(
                _record(),
                article_id=f"nvda-{index}",
                canonical_url=f"https://example.com/nvda-{index}",
                cluster_id=f"cluster-{index}",
                event_title=f"NVIDIA material event {index}",
            )
            for index in range(5)
        ) + tuple(
            replace(
                _record(),
                article_id=f"meta-{index}",
                canonical_url=f"https://example.com/meta-{index}",
                ticker="META",
                company="Meta",
                cluster_id=f"meta-cluster-{index}",
                event_title=f"Meta material event {index}",
            )
            for index in range(4)
        )
        comparison = compare_event_memory(
            records,
            PriorEventMemorySnapshot(available=False),
        )

        payload = build_llm_briefing_input(
            records=records,
            clusters=(),
            comparison=comparison,
            run_date="2026-06-12",
            tier=LLM_BRIEFING_TIERS["daily"],
            max_event_packets=4,
            max_events_per_ticker=2,
            include_low_priority_events=True,
        )
        ticker_counts = {
            packet["ticker"]: packet["event_count"]
            for packet in payload["ticker_packets"]
        }

        self.assertEqual(payload["event_packet_count"], 4)
        self.assertLessEqual(max(ticker_counts.values()), 2)

    def test_low_priority_events_are_excluded_by_default(self):
        record = replace(
            _record(),
            ticker="META",
            company="Meta",
            article_type="generic_buy_sell_hold_opinion",
            event_type="generic_buy_sell_hold_opinion",
            extraction_basis="snippet",
        )
        comparison = compare_event_memory(
            (record,),
            PriorEventMemorySnapshot(available=True, records=()),
        )

        payload = build_llm_briefing_input(
            records=(record,),
            clusters=(),
            comparison=comparison,
            run_date="2026-06-12",
            tier=LLM_BRIEFING_TIERS["daily"],
        )

        self.assertEqual(payload["event_packets_before_filter"], 1)
        self.assertEqual(payload["event_packet_count"], 0)
        self.assertEqual(payload["packets_dropped_low_priority"], 1)

    def test_known_source_family_is_not_unclassified(self):
        quality = classify_llm_source_quality(
            Article(
                canonical_url="https://news.google.com/example",
                title="NVIDIA market update",
                metadata={
                    "provider": "google_news_rss_search",
                    "source_family": "google_news_backstop",
                },
            )
        )

        self.assertNotEqual(quality.label, UNKNOWN_SOURCE_LABEL)
        self.assertEqual(quality.label, "tier_3_low_priority")

    def test_llm_is_disabled_by_default(self):
        client = CountingLlmClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, output_dir, stdout = _run_dry_run(
                temp_dir,
                client=client,
            )

            self.assertFalse(payload["llm_briefing_enabled"])
            self.assertEqual(payload["llm_briefing_status"], "disabled")
            self.assertEqual(client.call_count, 0)
            self.assertFalse(
                (output_dir / "llm_briefing_input.json").exists()
            )
            self.assertNotIn("OPENAI_API_KEY", stdout)

    def test_unconfirmed_llm_does_not_require_key_or_call_client(self):
        client = CountingLlmClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, output_dir, _stdout = _run_dry_run(
                temp_dir,
                extra_args=["--enable-llm-briefing"],
                client=client,
            )

            self.assertEqual(
                payload["llm_briefing_status"],
                "confirmation_required",
            )
            self.assertEqual(
                payload["llm_briefing_blocked_reason"],
                "llm_confirm_call_required",
            )
            self.assertEqual(client.call_count, 0)
            self.assertTrue(
                (output_dir / "llm_briefing_input.json").exists()
            )

    def test_estimate_only_writes_safe_artifacts_without_calling_client(self):
        secret = "test-openai-secret-that-must-not-appear"
        client = CountingLlmClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, output_dir, stdout = _run_dry_run(
                temp_dir,
                extra_args=[
                    "--enable-llm-briefing",
                    "--llm-briefing-tier",
                    "daily",
                    "--llm-estimate-cost-only",
                ],
                environ={"OPENAI_API_KEY": secret},
                client=client,
            )
            input_text = (
                output_dir / "llm_briefing_input.json"
            ).read_text(encoding="utf-8")
            preview = json.loads(
                (
                    output_dir / "llm_briefing_preview.json"
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(payload["llm_briefing_status"], "estimate_only")
            self.assertFalse(payload["llm_briefing_call_attempted"])
            self.assertFalse(payload["llm_briefing_call_completed"])
            self.assertEqual(client.call_count, 0)
            self.assertGreater(
                payload["llm_briefing_estimated_input_tokens"],
                0,
            )
            self.assertLessEqual(
                payload["llm_briefing_estimated_cost_usd"],
                payload["llm_briefing_cost_cap_usd"],
            )
            self.assertEqual(preview["status"], "estimate_only")
            self.assertNotIn(secret, stdout)
            self.assertNotIn(secret, input_text)
            self.assertNotIn(secret, json.dumps(preview))

    def test_confirmed_llm_without_key_skips_safely(self):
        client = CountingLlmClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, _output_dir, _stdout = _run_dry_run(
                temp_dir,
                extra_args=[
                    "--enable-llm-briefing",
                    "--llm-confirm-call",
                ],
                client=client,
            )

            self.assertEqual(payload["llm_briefing_status"], "missing_api_key")
            self.assertEqual(client.call_count, 0)

    def test_cost_cap_blocks_confirmed_client_call(self):
        restrictive = LlmBriefingTier(
            name="daily",
            model="gpt-5.4-mini",
            purpose="test",
            max_input_tokens=60_000,
            max_output_tokens=5_000,
            cost_cap_usd=0.000001,
            input_cost_per_million_tokens=0.75,
            output_cost_per_million_tokens=4.50,
        )
        client = CountingLlmClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "news_pipeline.cli.get_llm_briefing_tier",
                return_value=restrictive,
            ):
                payload, _output_dir, _stdout = _run_dry_run(
                    temp_dir,
                    extra_args=[
                        "--enable-llm-briefing",
                        "--llm-confirm-call",
                    ],
                    environ={"OPENAI_API_KEY": "mock-only-key"},
                    client=client,
                )

            self.assertEqual(payload["llm_briefing_status"], "blocked")
            self.assertEqual(
                payload["llm_briefing_blocked_reason"],
                "estimated_cost_exceeds_tier_cap",
            )
            self.assertFalse(payload["llm_briefing_call_attempted"])
            self.assertEqual(client.call_count, 0)

    def test_premium_tier_requires_explicit_selection(self):
        parser = build_parser()
        default_args = parser.parse_args(["dry-run-daily"])
        premium_args = parser.parse_args(
            [
                "dry-run-daily",
                "--enable-llm-briefing",
                "--llm-briefing-tier",
                "premium",
            ]
        )

        self.assertEqual(default_args.llm_briefing_tier, "daily")
        self.assertEqual(premium_args.llm_briefing_tier, "premium")
        self.assertEqual(EVENT_SIMILARITY_THRESHOLD, 0.75)

    def test_loads_llm_briefing_from_prior_populated_run(self):
        client = CountingLlmClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            source_run = _write_populated_run(
                Path(temp_dir) / "artifacts" / "runs" / "2026-06-10"
            )
            payload, output_dir, stdout, exit_code = _invoke_dry_run(
                temp_dir,
                extra_args=[
                    "--enable-llm-briefing",
                    "--llm-briefing-from-run",
                    str(source_run),
                ],
                environ={"OPENAI_API_KEY": "unused-test-key"},
                client=client,
            )
            input_text = (
                output_dir / "llm_briefing_input.json"
            ).read_text(encoding="utf-8")

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                payload["status"],
                "llm_briefing_preview_complete",
            )
            self.assertEqual(payload["llm_briefing_status"], "estimate_only")
            self.assertEqual(payload["llm_event_packet_count"], 1)
            self.assertEqual(payload["llm_ticker_packet_count"], 1)
            self.assertEqual(
                payload["llm_input_source_run"],
                str(source_run),
            )
            self.assertGreater(payload["llm_estimated_input_tokens"], 0)
            self.assertEqual(payload["llm_estimated_output_tokens"], 5_000)
            self.assertLessEqual(
                payload["llm_estimated_cost_usd"],
                payload["llm_cost_cap_usd"],
            )
            self.assertEqual(payload["email_sending"], "not_invoked")
            self.assertEqual(client.call_count, 0)
            self.assertTrue(
                (output_dir / "llm_briefing_preview.json").exists()
            )
            self.assertNotIn("raw body must stay outside packet", input_text)
            self.assertNotIn("unused-test-key", input_text)
            self.assertNotIn("unused-test-key", stdout)

    def test_latest_populated_run_skips_newer_empty_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "artifacts" / "runs"
            source_run = _write_populated_run(runs_dir / "2026-06-10")
            empty_run = runs_dir / "2026-06-11"
            empty_run.mkdir(parents=True)
            (empty_run / "event_memory_daily.json").write_text(
                "[]",
                encoding="utf-8",
            )

            payload, _output_dir, _stdout, exit_code = _invoke_dry_run(
                temp_dir,
                extra_args=[
                    "--enable-llm-briefing",
                    "--llm-briefing-from-latest-populated-run",
                ],
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                payload["llm_input_source_run"],
                str(source_run),
            )
            self.assertEqual(payload["llm_event_packet_count"], 1)

    def test_latest_populated_run_fails_clearly_when_none_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, _output_dir, _stdout, exit_code = _invoke_dry_run(
                temp_dir,
                extra_args=[
                    "--enable-llm-briefing",
                    "--llm-briefing-from-latest-populated-run",
                ],
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn(
                "No populated LLM briefing run found",
                payload["reason"],
            )


def _record() -> EventMemoryRecord:
    title = "NVIDIA launches a new AI data center platform"
    normalized = normalize_event_title(
        title,
        ticker="NVDA",
        company="NVIDIA",
    )
    published_at = "2026-06-11T12:00:00Z"
    date_bucket = event_date_bucket(published_at)
    return EventMemoryRecord(
        article_id="nvda-event",
        canonical_url="https://example.com/nvidia-launch",
        published_at=published_at,
        ticker="NVDA",
        company="NVIDIA",
        source_provider="example",
        source_family="direct_news_publisher",
        article_type="product_or_ai_or_chip_news",
        cluster_id="",
        ticker_match_confidence=0.95,
        extraction_basis="full_text",
        extraction_quality_grade="strong_full_text",
        internal_sentiment=0.4,
        external_sentiment_provider=None,
        external_sentiment=None,
        event_type="product_news",
        event_title=title,
        normalized_event_title=normalized,
        published_date_bucket=date_bucket,
        event_identity_fingerprint=event_identity_fingerprint(
            ticker="NVDA",
            normalized_title=normalized,
            event_type="product_news",
            article_type="product_or_ai_or_chip_news",
            company="NVIDIA",
            source_family="direct_news_publisher",
            published_date_bucket=date_bucket,
        ),
        event_summary="NVIDIA introduced a new platform.",
        run_id="run-2026-06-12",
        run_date="2026-06-12",
    )


def _priority_packet(
    *,
    group="watchlist",
    event_status="likely_new_event",
    article_type="product_or_ai_or_chip_news",
    official=False,
    uncertainty=(),
):
    return {
        "ticker": "META",
        "portfolio_watchlist_status": group,
        "new_repeated_status": event_status,
        "source_quality": {"tier": 2, "label": "tier_2_usable"},
        "source_count": 1,
        "sentiment_change": None,
        "official_signal": official,
        "benchmark_disagreement": False,
        "article_type": article_type,
        "source_family": "direct_news_publisher",
        "uncertainty": uncertainty,
        "top_titles": ["Example event"],
    }


def _run_dry_run(
    temp_dir,
    *,
    extra_args=None,
    environ=None,
    client=None,
):
    payload, output_dir, stdout, exit_code = _invoke_dry_run(
        temp_dir,
        extra_args=extra_args,
        environ=environ,
        client=client,
    )
    if exit_code != 0:
        raise AssertionError(stdout)
    return payload, output_dir, stdout


def _invoke_dry_run(
    temp_dir,
    *,
    extra_args=None,
    environ=None,
    client=None,
):
    argv = [
        "dry-run-daily",
        "--run-date",
        "2026-06-12",
        "--artifacts-dir",
        str(Path(temp_dir) / "artifacts"),
    ]
    if extra_args:
        argv.extend(extra_args)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(
            argv,
            environ=environ or {},
            llm_client=client,
        )
    payload = json.loads(stdout.getvalue())
    return (
        payload,
        Path(payload["output_dir"]),
        stdout.getvalue(),
        exit_code,
    )


def _write_populated_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = replace(
        _record(),
        cluster_id="cluster-nvda",
        run_id=f"run-{run_dir.name}",
        run_date=run_dir.name,
    )
    (run_dir / "event_memory_daily.json").write_text(
        json.dumps([record.as_dict()], indent=2),
        encoding="utf-8",
    )
    (run_dir / "event_memory_comparison.json").write_text(
        json.dumps(
            {
                "prior_run_available": True,
                "history_status": "compared_to_prior_run",
                "prior_run_id": "prior-run",
                "prior_run_date": "2026-06-09",
                "event_memory_lookback_days": 3,
                "prior_runs_considered": [
                    {
                        "run_id": "prior-run",
                        "run_date": "2026-06-09",
                    }
                ],
                "prior_event_records_considered": 1,
                "new_events_since_prior_run": 1,
                "repeated_events_from_prior_run": 0,
                "exact_repeated_events_from_prior_run": 0,
                "fuzzy_repeated_events_from_prior_run": 0,
                "event_identity_method_counts": {
                    "exact_url_repeat": 0,
                    "fuzzy_event_repeat": 0,
                    "likely_new_event": 1,
                },
                "event_similarity_threshold": 0.75,
                "event_identity_matches": [
                    {
                        "category": "likely_new_event",
                        "current_record_index": 0,
                        "ticker": "NVDA",
                        "title_similarity": 0.0,
                    }
                ],
                "sentiment_change_since_prior_run": {
                    "NVDA": {
                        "prior": 0.1,
                        "current": 0.4,
                        "change": 0.3,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "dedupe_clusters.json").write_text(
        json.dumps(
            {
                "cluster_count": 1,
                "clusters": [
                    {
                        "canonical_url": record.canonical_url,
                        "title": record.event_title,
                        "primary_link": record.canonical_url,
                        "cluster_id": record.cluster_id,
                        "publisher_count": 1,
                        "source_count": 1,
                        "publisher_names": ["Reuters"],
                        "source_providers": ["example"],
                        "supporting_links": [],
                        "alternate_source_links": [],
                        "duplicate_reasons": [],
                        "primary_ticker": "NVDA",
                        "matched_tickers": ["NVDA"],
                        "related_tickers": [],
                        "event_type": record.event_type,
                        "primary_article_id": record.article_id,
                        "supporting_article_ids": [],
                        "supporting_publishers": [],
                        "source_diversity": 1,
                        "publisher_diversity": 1,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "collected_articles.json").write_text(
        json.dumps(
            {
                "articles": [
                    {
                        "canonical_url": record.canonical_url,
                        "title": record.event_title,
                        "full_text": (
                            "raw body must stay outside packet"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return run_dir


if __name__ == "__main__":
    unittest.main()
