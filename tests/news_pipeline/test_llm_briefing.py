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
    estimate_llm_briefing_cost,
)
from news_pipeline.models import Article


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


def _run_dry_run(
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
    if exit_code != 0:
        raise AssertionError(stdout.getvalue())
    payload = json.loads(stdout.getvalue())
    return payload, Path(payload["output_dir"]), stdout.getvalue()


if __name__ == "__main__":
    unittest.main()
