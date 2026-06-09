"""Safe local orchestration CLI for the canonical news pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .article_fetch import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_ARTICLE_FETCHES,
    DEFAULT_MAX_FETCHES_PER_TICKER,
    ArticleFetchSummary,
    URL_CLASS_DIRECT_PUBLISHER,
    classify_article_url,
    disabled_article_fetch_summary,
    fetch_top_cluster_articles,
)
from .article_types import article_type_counts, classify_article_type
from .dedup import cluster_articles
from .email_preview import PreviewEmailSender
from .email_sender import (
    DEFAULT_EMAIL_ATTACHMENT_NAMES,
    DEFAULT_MAX_ATTACHMENT_BYTES,
    DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES,
    EmailSendError,
    EmailSender,
    LocalPreviewEmailSender,
    PREVIEW_MODE,
    REAL_SEND_MODE,
    SmtpEmailSender,
    build_report_email_payload,
)
from .models import Article, ArticleSource, RunResult, SentimentResult, TickerMention
from .provider_registry import iter_provider_configs
from .provider_usage import ProviderUsageRecorder
from .provider_validation import ProviderChecker, validate_provider
from .recency import article_recency, recency_weight
from .reporting import (
    ArticleLink,
    BackendArticlePoolSummary,
    DailyReportInput,
    DailyReportContract,
    EmailDisplaySummary,
    EmergingNameRow,
    EventClusterRow,
    ExtractionSummary,
    MentionLeaderRow,
    MostMentionedRow,
    PortfolioSentimentRow,
    WatchlistForecastRow,
    build_daily_report,
)
from .sentiment_coverage import build_weighted_sentiment_coverage
from .sentiment import analyze_sentiment
from .source_quality import (
    DEFAULT_MIN_SOURCE_QUALITY_TIER,
    SourceQualitySummary,
    assess_article_source,
    filter_articles_by_source_quality,
    source_quality_link_sort_key,
)
from .sources.live_rss import (
    DEFAULT_LIVE_RSS_RETRIES,
    DEFAULT_LIVE_RSS_TIMEOUT_SECONDS,
    DEFAULT_LIVE_RSS_USER_AGENT,
    collect_live_rss_articles,
    default_live_rss_urls,
)
from .sources.rss_config import (
    DEFAULT_MAX_GOOGLE_NEWS_SHARE,
    DEFAULT_MAX_ARTICLES_PER_SOURCE,
    DEFAULT_MAX_ARTICLES_PER_TICKER,
    DEFAULT_MAX_TOTAL_LIVE_ARTICLES,
    GOOGLE_NEWS_DISCOVERY,
)
from .sources.rss import RssSource
from .sources.source_registry import (
    COMPANY_IR_PROFILES,
    GOOGLE_NEWS_BACKSTOP,
    load_source_profiles,
)
from .sources.source_scheduler import schedule_sources
from .storage import SQLiteStore, initialize_database
from .summaries import MarketIntelligence, build_market_intelligence
from .ticker_matching import assess_ticker_matches, confidence_summary
from .tickers import TrackedTicker, load_portfolio, load_tracked_tickers, load_watchlist, match_tickers


class FakeProvider(Protocol):
    def articles(self) -> list[Article]:
        """Return fake local articles without network access."""


DEFAULT_RSS_FIXTURES_DIR = Path("tests/fixtures/rss")
DEFAULT_MAX_EMAIL_STORIES = 60
DEFAULT_MAX_RANKED_READS_PER_TICKER = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="news-pipeline")
    subcommands = parser.add_subparsers(dest="command", required=True)

    for command in (
        "init-db",
        "validate-providers",
        "collect",
        "extract",
        "dedup",
        "score",
        "report",
        "dry-run-daily",
    ):
        subparser = subcommands.add_parser(command)
        _add_safe_common_args(subparser)
        if command == "init-db":
            subparser.add_argument("--database-path")
        if command == "dry-run-daily":
            _add_live_rss_args(subparser)
            _add_live_article_fetch_args(subparser)
            _add_source_quality_args(subparser)
            subparser.add_argument("--run-id")

    send_parser = subcommands.add_parser("send-daily-report")
    _add_send_daily_report_args(send_parser)

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fake_providers: Mapping[str, FakeProvider] | None = None,
    provider_checker: ProviderChecker | None = None,
    environ: Mapping[str, str] | None = None,
    email_sender: EmailSender | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = _run_output_dir(args.artifacts_dir, args.run_date)
    if args.command == "send-daily-report":
        return _send_daily_report(args, output_dir, environ=environ, email_sender=email_sender)

    output_dir.mkdir(parents=True, exist_ok=True)
    context = _safe_context(args)

    if args.command == "init-db":
        database_path = Path(args.database_path) if args.database_path else output_dir / "news_pipeline.sqlite3"
        store = initialize_database(database_path)
        try:
            payload = {
                **context,
                "database_path": str(database_path),
                "tables": sorted(store.table_names()),
            }
        finally:
            store.close()
        _write_json(output_dir / "init_db.json", payload)
        _print_json(payload)
        return 0

    if args.command == "validate-providers":
        results = [
            validate_provider(
                config,
                dry_run=args.dry_run,
                environ=environ,
                checker=provider_checker,
            ).as_safe_dict()
            for config in iter_provider_configs()
        ]
        payload = {**context, "providers": results}
        _write_json(output_dir / "provider_validation.json", payload)
        _print_json(payload)
        return 0

    if args.command == "collect":
        articles = _collect_articles(args, fake_providers)
        payload = {**context, "article_count": len(articles), "articles": [_article_to_dict(article, args.run_date) for article in articles]}
        _write_json(output_dir / "collected_articles.json", payload)
        _print_json({**context, "article_count": len(articles), "output": str(output_dir / "collected_articles.json")})
        return 0

    if args.command == "extract":
        articles = _collect_articles(args, fake_providers)
        payload = {
            **context,
            "extractions": [
                {
                    "canonical_url": article.canonical_url,
                    "title": article.title,
                    "extraction_basis": "full_text" if article.full_text else "snippet" if article.snippet else "title",
                    "extraction_status": "ready",
                }
                for article in articles
            ],
        }
        _write_json(output_dir / "extractions.json", payload)
        _print_json({**context, "extraction_count": len(payload["extractions"]), "output": str(output_dir / "extractions.json")})
        return 0

    if args.command == "dedup":
        articles = _collect_articles(args, fake_providers)
        clusters = cluster_articles(articles)
        payload = {
            **context,
            "cluster_count": len(clusters),
            "clusters": [
                {
                    "canonical_url": cluster.canonical_article.canonical_url,
                    "title": cluster.canonical_article.title,
                    "alternate_source_links": list(cluster.alternate_source_links),
                    "duplicate_reasons": list(cluster.duplicate_reasons),
                }
                for cluster in clusters
            ],
        }
        _write_json(output_dir / "dedupe_clusters.json", payload)
        _print_json({**context, "cluster_count": len(clusters), "output": str(output_dir / "dedupe_clusters.json")})
        return 0

    if args.command == "score":
        articles = _canonical_articles(_collect_articles(args, fake_providers))
        scores = [
            analyze_sentiment(
                article.article_id or article.canonical_url,
                article.full_text or article.snippet or article.title,
                "full_text" if article.full_text else "snippet" if article.snippet else "title",
            )
            for article in articles
        ]
        payload = {**context, "scores": [_dataclass_to_dict(score) for score in scores]}
        _write_json(output_dir / "sentiment_scores.json", payload)
        _print_json({**context, "score_count": len(scores), "output": str(output_dir / "sentiment_scores.json")})
        return 0

    if args.command == "report":
        articles = _canonical_articles(_collect_articles(args, fake_providers))
        report = build_daily_report(_daily_report_input(args.run_date, articles), artifacts_dir=args.artifacts_dir)
        markdown_report = _write_markdown_report(report)
        payload = {**context, "report": _dataclass_to_dict(report)}
        _write_json(output_dir / "report_contract.json", payload)
        _print_json(
            {
                **context,
                "output_dir": report.output_dir,
                "csv_attachments": list(report.csv_attachments),
                "html_preview_report": report.html_preview_report,
                "local_report": markdown_report,
            }
        )
        return 0

    if args.command == "dry-run-daily":
        run_id = args.run_id or _run_id(args.run_date)
        database_path = output_dir / "news_pipeline.sqlite3"
        store = initialize_database(database_path)
        try:
            store.reset_run(run_id)
            provider_results = _provider_validation_results(args, provider_checker, environ)
            for result in provider_results:
                store.record_provider_validation(run_id, result)

            store.record_run(
                RunResult(
                    run_id=run_id,
                    status="started",
                ),
                run_date=args.run_date,
            )

            collected_articles, live_rss_summary = _collect_daily_articles(
                args,
                fake_providers,
                store,
                run_id,
                environ=environ,
            )
            source_quality_filter = filter_articles_by_source_quality(
                collected_articles,
                include_low_quality_sources=args.include_low_quality_sources,
                min_source_quality_tier=args.min_source_quality_tier,
            )
            articles = list(source_quality_filter.visible_articles)
            _write_json(
                output_dir / "provider_validation.json",
                {**context, "providers": provider_results, "live_rss": live_rss_summary},
            )
            _write_json(
                output_dir / "source_quality.json",
                {
                    **context,
                    **source_quality_filter.as_dict(),
                    "excluded_articles": [
                        _article_to_dict(article, args.run_date)
                        for article in source_quality_filter.excluded_articles
                    ]
                    if args.show_excluded_source_diagnostics
                    else [],
                },
            )
            discovery_clusters = cluster_articles(articles)
            enriched_articles_by_url, article_fetch_summary = _run_article_fetch_stage(args, discovery_clusters)
            articles_for_storage = [_enriched_article(article, enriched_articles_by_url) for article in articles]
            clusters = cluster_articles(articles_for_storage)
            market_intelligence = build_market_intelligence(
                articles=articles_for_storage,
                clusters=clusters,
                run_date=args.run_date,
            )
            backend_sentiment_inputs, ticker_sentiment_coverage = build_weighted_sentiment_coverage(
                articles=articles_for_storage,
                clusters=clusters,
                run_date=args.run_date,
            )
            persisted_article_ids = _persist_run_articles(store, run_id, articles_for_storage)
            scores = _score_articles(articles_for_storage)
            sentiment_basis_counts = _score_dict_basis_counts(scores)
            _persist_article_extractions(store, run_id, article_fetch_summary, persisted_article_ids)
            _persist_dedupe_clusters(store, run_id, clusters, args.run_date)
            _persist_sentiment_and_mentions(store, run_id, articles_for_storage)

            store.record_run(
                RunResult(
                    run_id=run_id,
                    status="completed",
                    articles_seen=len(articles),
                    articles_stored=len(persisted_article_ids),
                    duplicates=max(0, len(articles) - len(clusters)),
                ),
                run_date=args.run_date,
            )
            report = build_daily_report(
                _daily_report_input_from_store(
                    store,
                    run_id,
                    args.run_date,
                    article_fetch_summary,
                    source_quality_summary=source_quality_filter.summary,
                    data_source_label=_data_source_label(args),
                    show_excluded_source_diagnostics=args.show_excluded_source_diagnostics,
                    market_intelligence=market_intelligence,
                    backend_article_pool_summary=BackendArticlePoolSummary(
                        backend_candidate_articles=len(collected_articles),
                        backend_visible_articles=len(articles_for_storage),
                        backend_scored_articles=len({item.canonical_url for item in backend_sentiment_inputs}),
                        backend_extracted_articles=sum(1 for article in articles_for_storage if article.full_text),
                    ),
                    source_coverage_diagnostics={
                        "live_rss": live_rss_summary,
                        "source_quality": source_quality_filter.summary.as_dict(),
                        **live_rss_summary,
                    },
                    dedupe_diagnostics=_dedupe_diagnostics(articles_for_storage, clusters),
                    ticker_match_confidence_summary=confidence_summary(articles_for_storage),
                    article_type_count_summary=article_type_counts(articles_for_storage),
                    ticker_sentiment_coverage=ticker_sentiment_coverage,
                    backend_sentiment_inputs=backend_sentiment_inputs,
                    max_email_stories=max(1, int(args.max_email_stories)),
                    max_ranked_reads_per_ticker=max(1, int(args.max_ranked_reads_per_ticker)),
                ),
                artifacts_dir=args.artifacts_dir,
            )
        finally:
            store.close()
        markdown_report = _write_markdown_report(report)
        email_preview = PreviewEmailSender().write_preview(report)
        _write_json(
            output_dir / "collected_articles.json",
            {**context, "article_count": len(articles), "articles": [_article_to_dict(article, args.run_date) for article in articles_for_storage]},
        )
        _write_json(output_dir / "article_extractions.json", {**context, "article_fetch": article_fetch_summary.as_dict()})
        _write_json(
            output_dir / "publisher_extraction_profiles.json",
            {
                **context,
                "publisher_profiles": list(article_fetch_summary.publisher_profiles),
            },
        )
        _write_json(
            output_dir / "dedupe_clusters.json",
            {
                **context,
                "cluster_count": len(clusters),
                "clusters": [_cluster_to_dict(cluster, args.run_date) for cluster in clusters],
            },
        )
        _write_json(output_dir / "sentiment_scores.json", {**context, "score_count": len(scores), "scores": scores})
        _write_json(output_dir / "report_contract.json", {**context, "report": _dataclass_to_dict(report)})
        payload = {
            **context,
            "status": "dry_run_complete",
            "email_sending": "preview_only",
            "paid_apis": "enabled" if args.enable_paid_apis else "disabled",
            "paid_news_apis": (
                "enabled" if args.enable_paid_news_apis else "disabled"
            ),
            "run_id": run_id,
            "database_path": str(database_path),
            "article_count": len(articles),
            "raw_article_count": len(collected_articles),
            "visible_article_count": source_quality_filter.summary.visible_articles,
            "excluded_article_count": source_quality_filter.summary.excluded_articles,
            "source_quality_summary": source_quality_filter.summary.as_dict(),
            "cluster_count": len(clusters),
            "score_count": len(scores),
            "article_pages_fetched": article_fetch_summary.attempted_fetches,
            "publisher_article_fetches": article_fetch_summary.publisher_article_fetches,
            "google_news_wrappers_skipped": article_fetch_summary.google_news_wrappers_skipped,
            "google_news_wrappers_resolved": article_fetch_summary.google_news_wrappers_resolved,
            "successful_extractions": article_fetch_summary.successful_extractions,
            "failed_extractions": article_fetch_summary.failed_extractions,
            "snippet_fallbacks": article_fetch_summary.snippet_fallbacks,
            "title_fallbacks": article_fetch_summary.title_fallbacks,
            "top_extraction_failure_reasons": article_fetch_summary.failure_reason_counts,
            "extraction_method_counts": article_fetch_summary.extraction_method_counts,
            "extraction_failure_reason": article_fetch_summary.extraction_failure_reason,
            "extractor_diagnostics": article_fetch_summary.extractor_diagnostics or {},
            "sentiment_basis_counts": sentiment_basis_counts,
            "backend_candidate_articles": len(collected_articles),
            "backend_visible_articles": len(articles_for_storage),
            "backend_scored_articles": len({item.canonical_url for item in backend_sentiment_inputs}),
            "backend_extracted_articles": sum(1 for article in articles_for_storage if article.full_text),
            "email_visible_stories": report.email_display_summary.email_visible_stories,
            "email_visible_ranked_reads": report.email_display_summary.email_visible_ranked_reads,
            "article_type_counts": dict(report.article_type_counts),
            "ticker_match_confidence_summary": dict(report.ticker_match_confidence_summary),
            "ticker_sentiment_coverage": {
                ticker: _dataclass_to_dict(row)
                for ticker, row in report.ticker_sentiment_coverage.items()
            },
            "extraction_queue_size": article_fetch_summary.extraction_queue_size,
            "extraction_selected_count": article_fetch_summary.extraction_selected_count,
            "extraction_skipped_count": article_fetch_summary.extraction_skipped_count,
            "extraction_success_rate": article_fetch_summary.extraction_success_rate,
            "extraction_budget_unused_count": max(
                0,
                article_fetch_summary.max_article_fetches - article_fetch_summary.extraction_selected_count,
            ),
            "direct_publisher_candidates": article_fetch_summary.direct_publisher_candidates,
            "google_wrapper_candidates": article_fetch_summary.google_wrapper_candidates,
            "google_wrappers_unresolved": article_fetch_summary.google_wrappers_unresolved,
            "full_text_accepted_count": article_fetch_summary.successful_extractions,
            "weak_text_count": article_fetch_summary.extraction_quality_grade_counts.get("weak_text", 0),
            "extraction_quality_grade_counts": article_fetch_summary.extraction_quality_grade_counts,
            "extractor_method_success_counts": article_fetch_summary.extraction_method_success_counts,
            "publisher_success_rates": article_fetch_summary.extraction_success_rate_by_publisher,
            "extraction_candidate_direct_ratio": article_fetch_summary.extraction_candidate_direct_ratio,
            "full_text_success_by_source_family": article_fetch_summary.full_text_success_by_source_family,
            "direct_source_article_count": live_rss_summary.get("direct_source_article_count", 0),
            "google_news_article_count": live_rss_summary.get("google_news_article_count", 0),
            "direct_publisher_url_ratio": live_rss_summary.get("direct_publisher_url_ratio", 0.0),
            "articles_by_source_family": live_rss_summary.get("articles_by_source_family", {}),
            "source_family_counts": live_rss_summary.get("source_family_counts", {}),
            "official_source_count": live_rss_summary.get("official_source_count", 0),
            "company_ir_count": live_rss_summary.get("company_ir_count", 0),
            "press_release_wire_count": live_rss_summary.get("press_release_wire_count", 0),
            "direct_publisher_count": live_rss_summary.get("direct_publisher_count", 0),
            "google_news_backstop_count": live_rss_summary.get("google_news_backstop_count", 0),
            "google_news_share": live_rss_summary.get("google_news_share", 0.0),
            "paid_api_count": live_rss_summary.get("paid_api_count", 0),
            "paid_api_skipped_reasons": live_rss_summary.get("paid_api_skipped_reasons", {}),
            "source_diversity_score": live_rss_summary.get("source_diversity_score", 0.0),
            "source_balance_score": live_rss_summary.get("source_balance_score", 0.0),
            "output_dir": report.output_dir,
            "html_preview_report": report.html_preview_report,
            "email_preview_html": email_preview.html_preview_path,
            "intended_email_attachments": list(email_preview.intended_attachments),
            "email_delivery_mode": email_preview.delivery_mode,
            "local_report": markdown_report,
        }
        _write_json(output_dir / "dry_run_daily.json", payload)
        _print_json(payload)
        return 0

    parser.error("unknown command")
    return 2


def _add_safe_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-date", default=_today_local_iso())
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--rss-fixtures-dir", default=str(DEFAULT_RSS_FIXTURES_DIR))
    parser.add_argument("--include-fixtures", action="store_true")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-paid-apis", action="store_true", default=False)
    parser.add_argument("--enable-paid-news-apis", action="store_true", default=False)
    parser.add_argument("--enable-email-send", action="store_true", default=False)


def _add_live_rss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-live-rss", action="store_true", default=False)
    parser.add_argument("--live-rss-url", action="append", default=None)
    parser.add_argument("--live-rss-timeout-seconds", type=float, default=DEFAULT_LIVE_RSS_TIMEOUT_SECONDS)
    parser.add_argument("--live-rss-retries", type=int, default=DEFAULT_LIVE_RSS_RETRIES)
    parser.add_argument("--live-rss-user-agent", default=DEFAULT_LIVE_RSS_USER_AGENT)
    parser.add_argument(
        "--max-articles-per-source",
        "--live-rss-max-articles-per-source",
        dest="live_rss_max_articles_per_source",
        type=int,
        default=DEFAULT_MAX_ARTICLES_PER_SOURCE,
    )
    parser.add_argument(
        "--max-articles-per-ticker",
        "--live-rss-max-articles-per-ticker",
        dest="live_rss_max_articles_per_ticker",
        type=int,
        default=DEFAULT_MAX_ARTICLES_PER_TICKER,
    )
    parser.add_argument(
        "--max-total-live-articles",
        "--live-rss-max-total-articles",
        "--max-backend-articles",
        dest="live_rss_max_total_articles",
        type=int,
        default=DEFAULT_MAX_TOTAL_LIVE_ARTICLES,
    )
    parser.add_argument("--target-backend-articles", type=int, default=250)
    parser.add_argument("--minimum-backend-articles", type=int, default=150)
    parser.add_argument("--max-email-stories", type=int, default=DEFAULT_MAX_EMAIL_STORIES)
    parser.add_argument(
        "--max-ranked-reads-per-ticker",
        type=int,
        default=DEFAULT_MAX_RANKED_READS_PER_TICKER,
    )
    parser.add_argument(
        "--prefer-direct-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-google-news-share",
        type=float,
        default=DEFAULT_MAX_GOOGLE_NEWS_SHARE,
    )
    parser.add_argument(
        "--include-press-release-feeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-sec-feeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enable-marketaux", action="store_true", default=False)
    parser.add_argument("--enable-finnhub-news", action="store_true", default=False)
    parser.add_argument("--enable-alpha-vantage-news", action="store_true", default=False)
    parser.add_argument("--enable-gnews", action="store_true", default=False)


def _add_live_article_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-live-article-fetch", action="store_true", default=False)
    parser.add_argument("--max-article-fetches", type=int, default=DEFAULT_MAX_ARTICLE_FETCHES)
    parser.add_argument("--max-fetches-per-ticker", type=int, default=DEFAULT_MAX_FETCHES_PER_TICKER)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=DEFAULT_FETCH_TIMEOUT_SECONDS)


def _add_source_quality_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-low-quality-sources", action="store_true", default=False)
    parser.add_argument("--min-source-quality-tier", type=int, choices=(1, 2, 3, 4), default=DEFAULT_MIN_SOURCE_QUALITY_TIER)
    parser.add_argument("--show-excluded-source-diagnostics", action="store_true", default=False)


def _add_send_daily_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-date", default=_today_local_iso())
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--to")
    parser.add_argument("--confirm-send", action="store_true", default=False)
    parser.add_argument("--attachment", action="append", default=None)
    parser.add_argument("--max-attachment-bytes", type=int, default=DEFAULT_MAX_ATTACHMENT_BYTES)
    parser.add_argument("--max-total-attachment-bytes", type=int, default=DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES)
    parser.add_argument("--backend", choices=("smtp",), default="smtp")


def _safe_context(args: argparse.Namespace) -> dict[str, object]:
    return {
        "command": args.command,
        "run_date": args.run_date,
        "dry_run": getattr(args, "dry_run", not bool(getattr(args, "confirm_send", False))),
        "paid_apis_enabled": getattr(args, "enable_paid_apis", False),
        "paid_news_apis_enabled": getattr(args, "enable_paid_news_apis", False),
        "email_send_enabled": getattr(args, "enable_email_send", False),
        "live_rss_enabled": bool(getattr(args, "enable_live_rss", False)),
        "live_article_fetch_enabled": bool(getattr(args, "enable_live_article_fetch", False)),
        "include_low_quality_sources": bool(getattr(args, "include_low_quality_sources", False)),
        "min_source_quality_tier": int(getattr(args, "min_source_quality_tier", DEFAULT_MIN_SOURCE_QUALITY_TIER)),
        "show_excluded_source_diagnostics": bool(getattr(args, "show_excluded_source_diagnostics", False)),
        "fixtures_included": bool(getattr(args, "include_fixtures", False)) or not bool(getattr(args, "enable_live_rss", False)),
        "output_dir": str(_run_output_dir(args.artifacts_dir, args.run_date)),
        "rss_fixtures_dir": getattr(args, "rss_fixtures_dir", ""),
        "max_total_live_articles": int(getattr(args, "live_rss_max_total_articles", DEFAULT_MAX_TOTAL_LIVE_ARTICLES)),
        "target_backend_articles": int(getattr(args, "target_backend_articles", 250)),
        "minimum_backend_articles": int(getattr(args, "minimum_backend_articles", 150)),
        "max_articles_per_ticker": int(getattr(args, "live_rss_max_articles_per_ticker", DEFAULT_MAX_ARTICLES_PER_TICKER)),
        "max_articles_per_source": int(getattr(args, "live_rss_max_articles_per_source", DEFAULT_MAX_ARTICLES_PER_SOURCE)),
        "max_email_stories": int(getattr(args, "max_email_stories", DEFAULT_MAX_EMAIL_STORIES)),
        "max_ranked_reads_per_ticker": int(
            getattr(args, "max_ranked_reads_per_ticker", DEFAULT_MAX_RANKED_READS_PER_TICKER)
        ),
        "prefer_direct_sources": bool(getattr(args, "prefer_direct_sources", True)),
        "max_google_news_share": float(
            getattr(args, "max_google_news_share", DEFAULT_MAX_GOOGLE_NEWS_SHARE)
        ),
        "include_press_release_feeds": bool(
            getattr(args, "include_press_release_feeds", True)
        ),
        "include_sec_feeds": bool(getattr(args, "include_sec_feeds", True)),
        "paid_news_provider_flags": {
            "marketaux": bool(getattr(args, "enable_marketaux", False)),
            "finnhub_news": bool(getattr(args, "enable_finnhub_news", False)),
            "alpha_vantage": bool(getattr(args, "enable_alpha_vantage_news", False)),
            "gnews": bool(getattr(args, "enable_gnews", False)),
        },
    }


def _today_local_iso() -> str:
    return date.today().isoformat()


def _send_daily_report(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
    email_sender: EmailSender | None = None,
) -> int:
    context = _safe_context(args)
    recipient = str(args.to or "").strip()
    if not recipient:
        _print_json(
            {
                **context,
                "status": "refused",
                "reason": "missing_recipient",
                "sent": False,
                "confirm_send": bool(args.confirm_send),
            }
        )
        return 2

    attachment_names = tuple(DEFAULT_EMAIL_ATTACHMENT_NAMES) + tuple(args.attachment or ())
    try:
        payload, manifest = build_report_email_payload(
            run_date=args.run_date,
            output_dir=output_dir,
            recipient=recipient,
            attachment_names=attachment_names,
            max_attachment_bytes=max(1, int(args.max_attachment_bytes)),
            max_total_attachment_bytes=max(1, int(args.max_total_attachment_bytes)),
            delivery_mode=REAL_SEND_MODE if args.confirm_send else PREVIEW_MODE,
        )
    except EmailSendError as error:
        _print_json(
            {
                **context,
                "status": "refused",
                "reason": str(error),
                "sent": False,
                "confirm_send": bool(args.confirm_send),
            }
        )
        return 2

    sender: EmailSender
    if args.confirm_send:
        sender = email_sender or SmtpEmailSender(environ)
    else:
        sender = LocalPreviewEmailSender()

    try:
        result = sender.send(payload)
    except EmailSendError as error:
        _print_json(
            {
                **context,
                "status": "refused",
                "reason": str(error),
                "sent": False,
                "confirm_send": bool(args.confirm_send),
                "backend": sender.backend_name,
            }
        )
        return 2
    except Exception as error:
        _print_json(
            {
                **context,
                "status": "failed",
                "reason": f"send_failed:{type(error).__name__}",
                "sent": False,
                "confirm_send": bool(args.confirm_send),
                "backend": sender.backend_name,
            }
        )
        return 1

    manifest_output = output_dir / "email_send_manifest.json"
    safe_payload = {
        **context,
        "status": result.status,
        "send_mode": "confirmed_send" if args.confirm_send else "dry_run_no_send",
        "confirm_send": bool(args.confirm_send),
        "would_send": not bool(args.confirm_send),
        "sent": result.sent,
        "backend": result.backend,
        "to": payload.to,
        "subject": payload.subject,
        "preview_path": payload.preview_path,
        "report_artifacts": list(payload.report_artifacts),
        "attachment_manifest": manifest.as_safe_dict(),
        "manifest_output": str(manifest_output),
    }
    _write_json(manifest_output, safe_payload)
    _print_json(safe_payload)
    return 0


def _run_output_dir(artifacts_dir: str | Path, run_date: str) -> Path:
    base = Path(artifacts_dir)
    if base.name != "artifacts":
        base = base / "artifacts"
    return base / "runs" / run_date


def _run_id(run_date: str) -> str:
    return f"dry-run-{run_date}"


def _collect_fake_articles(
    fake_providers: Mapping[str, FakeProvider] | None,
    *,
    enable_paid: bool,
) -> list[Article]:
    if not fake_providers:
        return []
    articles: list[Article] = []
    for provider_name, provider in fake_providers.items():
        if _is_paid_provider(provider_name) and not enable_paid:
            continue
        articles.extend(provider.articles())
    return articles


def _collect_articles(
    args: argparse.Namespace,
    fake_providers: Mapping[str, FakeProvider] | None,
) -> list[Article]:
    if fake_providers is not None:
        return _collect_fake_articles(fake_providers, enable_paid=args.enable_paid_apis)
    return _collect_fixture_rss_articles(Path(args.rss_fixtures_dir))


def _collect_daily_articles(
    args: argparse.Namespace,
    fake_providers: Mapping[str, FakeProvider] | None,
    store: SQLiteStore,
    run_id: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[Article], dict[str, object]]:
    articles = (
        _collect_articles(args, fake_providers)
        if not getattr(args, "enable_live_rss", False) or getattr(args, "include_fixtures", False)
        else []
    )
    if not getattr(args, "enable_live_rss", False):
        return articles, {
            "enabled": False,
            "attempt_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "article_count": 0,
            "source_counts": {},
            "direct_source_article_count": 0,
            "google_news_article_count": 0,
            "direct_publisher_url_ratio": 0.0,
            "google_news_share": 0.0,
            "google_news_share_capped": True,
            "articles_by_source_family": {},
            "top_direct_source_publishers": {},
            "top_google_news_publishers": {},
            "source_family_counts": {},
            "articles_by_source_id": {},
            "google_news_backstop_count": 0,
            "official_source_count": 0,
            "company_ir_count": 0,
            "press_release_wire_count": 0,
            "direct_publisher_count": 0,
            "paid_api_count": 0,
            "paid_api_skipped_reasons": {},
            "missing_company_ir_profiles": [],
            "source_profiles_loaded": 0,
            "source_profiles_enabled": 0,
            "source_profiles_failed": [],
            "source_diversity_score": 0.0,
            "source_balance_score": 0.0,
            "attempts": [],
        }

    feed_urls = tuple(args.live_rss_url) if args.live_rss_url else None
    if feed_urls:
        live_articles, rss_attempts = collect_live_rss_articles(
            feed_urls=feed_urls,
            timeout_seconds=max(0.1, float(args.live_rss_timeout_seconds)),
            retries=max(0, int(args.live_rss_retries)),
            user_agent=str(args.live_rss_user_agent),
            max_articles_per_source=max(1, int(args.live_rss_max_articles_per_source)),
            max_articles_per_ticker=max(1, int(args.live_rss_max_articles_per_ticker)),
            max_total_articles=max(1, int(args.live_rss_max_total_articles)),
            prefer_direct_sources=bool(args.prefer_direct_sources),
            max_google_news_share=max(0.0, min(1.0, float(args.max_google_news_share))),
            include_press_release_feeds=bool(args.include_press_release_feeds),
        )
        attempts = rss_attempts
        scheduler_diagnostics = _direct_source_diagnostics(
            live_articles,
            max_google_news_share=max(0.0, min(1.0, float(args.max_google_news_share))),
        )
    else:
        schedule = schedule_sources(
            profiles=load_source_profiles(),
            tracked_tickers=load_tracked_tickers(),
            company_ir_profiles=COMPANY_IR_PROFILES,
            run_date=args.run_date,
            user_agent=str(args.live_rss_user_agent),
            timeout_seconds=max(0.1, float(args.live_rss_timeout_seconds)),
            retries=max(0, int(args.live_rss_retries)),
            target_backend_articles=max(1, int(args.target_backend_articles)),
            minimum_backend_articles=max(1, int(args.minimum_backend_articles)),
            max_backend_articles=max(1, int(args.live_rss_max_total_articles)),
            max_articles_per_source=max(1, int(args.live_rss_max_articles_per_source)),
            max_articles_per_ticker=max(1, int(args.live_rss_max_articles_per_ticker)),
            max_google_news_share=max(0.0, min(1.0, float(args.max_google_news_share))),
            include_press_release_feeds=bool(args.include_press_release_feeds),
            include_sec_feeds=bool(args.include_sec_feeds),
            paid_api_global_enabled=bool(args.enable_paid_news_apis),
            paid_provider_flags={
                "marketaux": bool(args.enable_marketaux),
                "finnhub_news": bool(args.enable_finnhub_news),
                "alpha_vantage": bool(args.enable_alpha_vantage_news),
                "gnews": bool(args.enable_gnews),
            },
            environ=environ or {},
        )
        live_articles = list(schedule.articles)
        attempts = list(schedule.attempts)
        scheduler_diagnostics = dict(schedule.diagnostics)
    recorder = ProviderUsageRecorder(store)
    for attempt in attempts:
        attempt_metadata = (
            {
                "feed_id": attempt.feed_id,
                "feed_url": attempt.feed_url,
                "source_family": attempt.source_family,
                "attempts": attempt.attempts,
                "fetched_article_count": attempt.fetched_article_count,
            }
            if hasattr(attempt, "feed_id")
            else {
                **dict(getattr(attempt, "metadata", None) or {}),
                "source_family": attempt.source_family,
                "source_id": getattr(attempt, "source_id", attempt.provider),
            }
        )
        recorder.record(
            attempt.provider,
            "discover",
            attempt.status,
            quota_cost=0,
            article_count=attempt.article_count,
            latency_ms=attempt.latency_ms,
            error_class=attempt.error_class,
            metadata={"run_id": run_id, **attempt_metadata},
            run_id=run_id,
        )

    source_counts: dict[str, int] = {}
    for attempt in attempts:
        source_counts[attempt.provider] = source_counts.get(attempt.provider, 0) + attempt.article_count
    return articles + live_articles, {
        "enabled": True,
        "attempt_count": len(attempts),
        "success_count": sum(1 for attempt in attempts if attempt.status == "success"),
        "failure_count": sum(1 for attempt in attempts if attempt.status != "success"),
        "article_count": len(live_articles),
        "source_counts": source_counts,
        **scheduler_diagnostics,
        "caps": {
            "max_articles_per_source": max(1, int(args.live_rss_max_articles_per_source)),
            "max_articles_per_ticker": max(1, int(args.live_rss_max_articles_per_ticker)),
            "max_total_articles": max(1, int(args.live_rss_max_total_articles)),
            "max_google_news_share": max(0.0, min(1.0, float(args.max_google_news_share))),
            "target_backend_articles": max(1, int(args.target_backend_articles)),
            "minimum_backend_articles": max(1, int(args.minimum_backend_articles)),
        },
        "attempts": [attempt.as_dict() for attempt in attempts],
    }


def _direct_source_diagnostics(
    articles: Sequence[Article],
    *,
    max_google_news_share: float,
) -> dict[str, object]:
    family_counts: dict[str, int] = {}
    direct_publishers: dict[str, int] = {}
    google_publishers: dict[str, int] = {}
    direct_url_count = 0
    google_count = 0
    for article in articles:
        family = str(article.metadata.get("source_family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        publisher = str(
            article.metadata.get("source_name")
            or article.metadata.get("provider")
            or "unknown"
        )
        if family == GOOGLE_NEWS_DISCOVERY:
            google_count += 1
            google_publishers[publisher] = google_publishers.get(publisher, 0) + 1
        else:
            direct_publishers[publisher] = direct_publishers.get(publisher, 0) + 1
        if classify_article_url(article.canonical_url) == URL_CLASS_DIRECT_PUBLISHER:
            direct_url_count += 1
    total = len(articles)
    google_share = google_count / total if total else 0.0
    return {
        "direct_source_article_count": total - google_count,
        "google_news_article_count": google_count,
        "direct_publisher_url_ratio": round(direct_url_count / total, 4) if total else 0.0,
        "google_news_share": round(google_share, 4),
        "google_news_share_capped": google_share <= max_google_news_share + 0.0001,
        "articles_by_source_family": dict(sorted(family_counts.items())),
        "source_family_counts": dict(sorted(family_counts.items())),
        "articles_by_source_id": {},
        "google_news_backstop_count": google_count,
        "official_source_count": 0,
        "company_ir_count": 0,
        "press_release_wire_count": int(family_counts.get("press_release_wire", 0)),
        "direct_publisher_count": int(family_counts.get("direct_news_publisher", 0)),
        "paid_api_count": 0,
        "paid_api_skipped_reasons": {},
        "missing_company_ir_profiles": [],
        "source_profiles_loaded": 0,
        "source_profiles_enabled": 0,
        "source_profiles_failed": [],
        "source_diversity_score": 0.0,
        "source_balance_score": 0.0,
        "top_direct_source_publishers": _top_counts(direct_publishers),
        "top_google_news_publishers": _top_counts(google_publishers),
    }


def _top_counts(counts: Mapping[str, int], *, limit: int = 10) -> dict[str, int]:
    return dict(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    )


def _collect_fixture_rss_articles(fixtures_dir: Path) -> list[Article]:
    if not fixtures_dir.exists() or not fixtures_dir.is_dir():
        return []

    articles: list[Article] = []
    for fixture_path in sorted(fixtures_dir.glob("*.xml")):
        feed_xml = fixture_path.read_text(encoding="utf-8")
        source = RssSource(feed_xml, provider_name=f"fixture_rss_{fixture_path.stem}")
        articles.extend(source.articles())
    return articles


def _run_article_fetch_stage(args: argparse.Namespace, clusters: Sequence[object]) -> tuple[dict[str, Article], ArticleFetchSummary]:
    if not getattr(args, "enable_live_article_fetch", False):
        return {}, disabled_article_fetch_summary(
            reason="disabled",
            max_article_fetches=args.max_article_fetches,
            max_fetches_per_ticker=args.max_fetches_per_ticker,
            fetch_timeout_seconds=args.fetch_timeout_seconds,
        )
    if not getattr(args, "enable_live_rss", False):
        return {}, disabled_article_fetch_summary(
            reason="live_rss_required",
            max_article_fetches=args.max_article_fetches,
            max_fetches_per_ticker=args.max_fetches_per_ticker,
            fetch_timeout_seconds=args.fetch_timeout_seconds,
        )
    if not clusters:
        return {}, disabled_article_fetch_summary(
            reason="no_live_rss_clusters",
            max_article_fetches=args.max_article_fetches,
            max_fetches_per_ticker=args.max_fetches_per_ticker,
            fetch_timeout_seconds=args.fetch_timeout_seconds,
        )
    return fetch_top_cluster_articles(
        clusters,
        run_date=args.run_date,
        max_article_fetches=args.max_article_fetches,
        max_fetches_per_ticker=args.max_fetches_per_ticker,
        fetch_timeout_seconds=args.fetch_timeout_seconds,
        include_low_quality_sources=args.include_low_quality_sources,
        min_source_quality_tier=args.min_source_quality_tier,
        cache_path=_run_output_dir(args.artifacts_dir, args.run_date) / "extraction_cache.json",
    )


def _enriched_article(article: Article, enriched_articles_by_url: Mapping[str, Article]) -> Article:
    return enriched_articles_by_url.get(article.canonical_url, article)


def _canonical_articles(articles: list[Article]) -> list[Article]:
    return [cluster.canonical_article for cluster in cluster_articles(articles)]


def _is_paid_provider(provider_name: str) -> bool:
    return provider_name in {"alpha_vantage", "marketaux", "gnews", "finnhub_news"}


def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _print_json(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _article_to_dict(article: Article, run_date: str | None = None) -> dict[str, object]:
    recency = (
        article_recency(
            run_date=run_date,
            published_at=article.published_at,
            collected_at=article.created_at,
            archive_context=bool(article.metadata.get("archive_context")),
        )
        if run_date
        else None
    )
    classification = classify_article_type(article)
    ticker_matches = assess_ticker_matches(article)
    payload = {
        "canonical_url": article.canonical_url,
        "title": article.title,
        "article_id": article.article_id,
        "published_at": article.published_at,
        "collected_at": article.created_at,
        "snippet": article.snippet,
        "has_full_text": bool(article.full_text),
        "metadata": article.metadata,
        "article_type": classification.primary_type,
        "article_types": list(classification.event_types),
        "ticker_matches": [
            {
                "ticker": match.ticker,
                "ticker_match_confidence": match.confidence,
                "ticker_match_confidence_label": match.confidence_label,
                "ticker_match_reason": match.reason,
                "ticker_match_basis": match.basis,
                "primary": match.primary,
                "related": match.related,
            }
            for match in ticker_matches
        ],
    }
    if recency is not None:
        payload.update(
            {
                "article_age_hours": recency.article_age_hours,
                "recency_bucket": recency.recency_bucket,
                "recency_timestamp_source": recency.source,
            }
        )
    return payload


def _dataclass_to_dict(value: object) -> dict[str, object]:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected dataclass value, got {type(value)!r}")


def _daily_report_input(run_date: str, articles: Sequence[Article]) -> DailyReportInput:
    scored_articles = _scored_articles_by_ticker(articles)
    clusters = cluster_articles(articles)
    market_intelligence = build_market_intelligence(
        articles=articles,
        clusters=clusters,
        run_date=run_date,
    )
    article_links = {
        ticker.symbol: tuple(
            ArticleLink(
                title=item["article"].title,
                url=item["article"].canonical_url,
                source=item["article"].metadata.get("source_name") or item["article"].metadata.get("provider"),
            )
            for item in scored_articles.get(ticker.symbol, ())
        )
        for ticker in load_tracked_tickers()
    }

    portfolio_rows = tuple(
        PortfolioSentimentRow(
            ticker.symbol,
            ticker.company_name,
            _average_score(scored_articles.get(ticker.symbol, ())),
            len(scored_articles.get(ticker.symbol, ())),
            _sentiment_basis(scored_articles.get(ticker.symbol, ())),
        )
        for ticker in load_portfolio()
    )
    watchlist_sentiment_rows = tuple(
        PortfolioSentimentRow(
            ticker.symbol,
            ticker.company_name,
            _average_score(scored_articles.get(ticker.symbol, ())),
            len(scored_articles.get(ticker.symbol, ())),
            _sentiment_basis(scored_articles.get(ticker.symbol, ())),
        )
        for ticker in load_watchlist()
    )
    watchlist_rows = tuple(
        _watchlist_forecast_row(ticker, scored_articles.get(ticker.symbol, ()))
        for ticker in load_watchlist()
    )
    mentioned = [
        (
            ticker.symbol,
            len(scored_articles.get(ticker.symbol, ())),
            _average_score(scored_articles.get(ticker.symbol, ())),
        )
        for ticker in load_tracked_tickers()
        if scored_articles.get(ticker.symbol)
    ]
    mentioned.sort(key=lambda item: (-item[1], item[0]))

    return DailyReportInput(
        report_date=run_date,
        portfolio_sentiment=portfolio_rows,
        watchlist_sentiment=watchlist_sentiment_rows,
        watchlist_forecasts=watchlist_rows,
        mention_leaders_7d=tuple(MentionLeaderRow(symbol, count, score) for symbol, count, score in mentioned),
        most_mentioned=tuple(
            MostMentionedRow(symbol, count, rank)
            for rank, (symbol, count, _score) in enumerate(mentioned[:10], start=1)
        ),
        emerging_names=tuple(
            EmergingNameRow(ticker.symbol, ticker.company_name, len(scored_articles.get(ticker.symbol, ())), 0, "configured ticker mention")
            for ticker in load_watchlist()
            if scored_articles.get(ticker.symbol)
        ),
        article_links_by_ticker=article_links,
        portfolio_summaries=tuple(
            market_intelligence.ticker_summaries[ticker.symbol]
            for ticker in load_portfolio()
        ),
        watchlist_summaries=tuple(
            market_intelligence.ticker_summaries[ticker.symbol]
            for ticker in load_watchlist()
        ),
        ranked_reads_by_ticker=market_intelligence.ranked_reads_by_ticker,
        article_summaries=market_intelligence.article_summaries,
        data_source_label="local RSS fixtures",
    )


def _daily_report_input_from_store(
    store: SQLiteStore,
    run_id: str,
    run_date: str,
    article_fetch_summary: ArticleFetchSummary | None = None,
    source_quality_summary: SourceQualitySummary | None = None,
    data_source_label: str = "local RSS fixtures",
    show_excluded_source_diagnostics: bool = False,
    market_intelligence: MarketIntelligence | None = None,
    backend_article_pool_summary: BackendArticlePoolSummary | None = None,
    source_coverage_diagnostics: Mapping[str, object] | None = None,
    dedupe_diagnostics: Mapping[str, object] | None = None,
    ticker_match_confidence_summary: Mapping[str, object] | None = None,
    article_type_count_summary: Mapping[str, int] | None = None,
    ticker_sentiment_coverage: Mapping[str, object] | None = None,
    backend_sentiment_inputs: Sequence[object] = (),
    max_email_stories: int = DEFAULT_MAX_EMAIL_STORIES,
    max_ranked_reads_per_ticker: int = DEFAULT_MAX_RANKED_READS_PER_TICKER,
) -> DailyReportInput:
    articles_by_id = {str(row["article_id"]): row for row in store.list_run_articles(run_id)}
    sources_by_article: dict[str, dict[str, object]] = {}
    for source in store.list_article_sources(run_id):
        article_id = str(source["article_id"])
        sources_by_article.setdefault(article_id, source)
    sentiments_by_article = {
        str(row["article_id"]): SentimentResult(
            article_id=str(row["article_id"]),
            score=float(row["score"]),
            label=str(row["label"]),
            confidence=float(row["confidence"]),
            basis=str(row["basis"]),
            model=str(row["model"]),
            created_at=str(row["created_at"]),
        )
        for row in store.list_sentiment_results(run_id)
    }
    sentiment_basis_counts = _sentiment_basis_counts(sentiments_by_article.values())
    extractions_by_article = {str(row["article_id"]): row for row in store.list_article_extractions(run_id)}
    mentions_by_ticker: dict[str, list[dict[str, object]]] = {ticker.symbol: [] for ticker in load_tracked_tickers()}
    for mention in store.list_ticker_mentions(run_id):
        ticker = str(mention["ticker"])
        article_id = str(mention["article_id"])
        if article_id not in articles_by_id or article_id not in sentiments_by_article:
            continue
        mentions_by_ticker.setdefault(ticker, []).append(
            {
                "article": articles_by_id[article_id],
                "source": sources_by_article.get(article_id, {}),
                "score": sentiments_by_article[article_id],
                "basis": mention["basis"],
                "match_confidence": float(mention.get("confidence") or 0.0),
                "recency": _row_recency(run_date, articles_by_id[article_id]),
            }
        )

    backend_event_clusters_by_ticker = _event_clusters_by_ticker_from_store(
        store,
        run_id,
        extractions_by_article,
        market_intelligence=market_intelligence,
    )
    event_clusters_by_ticker = _cap_event_clusters(
        backend_event_clusters_by_ticker,
        max_email_stories=max_email_stories,
    )
    ranked_reads_by_ticker = _cap_ranked_reads(
        market_intelligence.ranked_reads_by_ticker if market_intelligence else {},
        max_ranked_reads_per_ticker=max_ranked_reads_per_ticker,
    )
    article_links = _article_links_from_event_clusters(event_clusters_by_ticker)
    portfolio_rows = tuple(
        _sentiment_row_from_stored(
            ticker,
            mentions_by_ticker.get(ticker.symbol, ()),
            event_clusters_by_ticker.get(ticker.symbol, ()),
            coverage=(ticker_sentiment_coverage or {}).get(ticker.symbol),
        )
        for ticker in load_portfolio()
    )
    watchlist_sentiment_rows = tuple(
        _sentiment_row_from_stored(
            ticker,
            mentions_by_ticker.get(ticker.symbol, ()),
            event_clusters_by_ticker.get(ticker.symbol, ()),
            coverage=(ticker_sentiment_coverage or {}).get(ticker.symbol),
        )
        for ticker in load_watchlist()
    )
    watchlist_rows = tuple(
        _watchlist_forecast_row_from_stored(ticker, mentions_by_ticker.get(ticker.symbol, ()))
        for ticker in load_watchlist()
    )
    mentioned = [
        (
            ticker.symbol,
            len(mentions_by_ticker.get(ticker.symbol, ())),
            _stored_average_score(mentions_by_ticker.get(ticker.symbol, ())),
        )
        for ticker in load_tracked_tickers()
        if mentions_by_ticker.get(ticker.symbol)
    ]
    mentioned.sort(key=lambda item: (-item[1], item[0]))
    return DailyReportInput(
        report_date=run_date,
        portfolio_sentiment=portfolio_rows,
        watchlist_sentiment=watchlist_sentiment_rows,
        watchlist_forecasts=watchlist_rows,
        mention_leaders_7d=tuple(MentionLeaderRow(symbol, count, score) for symbol, count, score in mentioned),
        most_mentioned=tuple(
            MostMentionedRow(symbol, count, rank)
            for rank, (symbol, count, _score) in enumerate(mentioned[:10], start=1)
        ),
        emerging_names=tuple(
            EmergingNameRow(ticker.symbol, ticker.company_name, len(mentions_by_ticker.get(ticker.symbol, ())), 0, "live RSS mention")
            for ticker in load_watchlist()
            if mentions_by_ticker.get(ticker.symbol)
        ),
        article_links_by_ticker=article_links,
        event_clusters_by_ticker=event_clusters_by_ticker,
        portfolio_summaries=tuple(
            market_intelligence.ticker_summaries[ticker.symbol]
            for ticker in load_portfolio()
        )
        if market_intelligence
        else (),
        watchlist_summaries=tuple(
            market_intelligence.ticker_summaries[ticker.symbol]
            for ticker in load_watchlist()
        )
        if market_intelligence
        else (),
        ranked_reads_by_ticker=ranked_reads_by_ticker,
        article_summaries=market_intelligence.article_summaries if market_intelligence else (),
        extraction_summary=_extraction_summary_from_fetch_summary(
            article_fetch_summary,
            extractions_by_article,
            sentiment_basis_counts,
            source_quality_summary=source_quality_summary,
            show_excluded_source_diagnostics=show_excluded_source_diagnostics,
        ),
        data_source_label=data_source_label,
        backend_article_pool_summary=backend_article_pool_summary or BackendArticlePoolSummary(),
        source_coverage_diagnostics=source_coverage_diagnostics or {},
        extraction_coverage_diagnostics=(
            {
                **article_fetch_summary.as_dict(),
                "full_text_basis_count": int(sentiment_basis_counts.get("full_text", 0)),
                "snippet_basis_count": int(sentiment_basis_counts.get("snippet", 0)),
                "title_basis_count": int(sentiment_basis_counts.get("title", 0)),
                "snippet_fallback_count": int(sentiment_basis_counts.get("snippet", 0)),
                "title_fallback_count": int(sentiment_basis_counts.get("title", 0)),
            }
            if article_fetch_summary
            else {}
        ),
        dedupe_diagnostics=dedupe_diagnostics or {},
        ticker_match_confidence_summary=ticker_match_confidence_summary or {},
        article_type_counts=article_type_count_summary or {},
        ticker_sentiment_coverage=ticker_sentiment_coverage or {},
        email_display_summary=EmailDisplaySummary(
            email_visible_stories=sum(len(rows) for rows in event_clusters_by_ticker.values()),
            email_visible_ranked_reads=sum(len(rows) for rows in ranked_reads_by_ticker.values()),
            max_email_stories=max_email_stories,
            max_ranked_reads_per_ticker=max_ranked_reads_per_ticker,
        ),
        backend_sentiment_inputs=tuple(backend_sentiment_inputs),
    )


def _event_clusters_by_ticker_from_store(
    store: SQLiteStore,
    run_id: str,
    extractions_by_article: Mapping[str, Mapping[str, object]] | None = None,
    *,
    market_intelligence: MarketIntelligence | None = None,
) -> dict[str, tuple[EventClusterRow, ...]]:
    grouped: dict[str, list[EventClusterRow]] = {ticker.symbol: [] for ticker in load_tracked_tickers()}
    extractions_by_article = extractions_by_article or {}
    for row in store.list_dedupe_clusters(run_id):
        title = str(row["title"])
        canonical_article_id = str(row.get("canonical_article_id") or "")
        extraction = extractions_by_article.get(canonical_article_id, {})
        supporting_links = tuple(
            ArticleLink(
                title=str(link.get("title") or title),
                url=str(link.get("url") or row["canonical_url"]),
                source=str(link.get("publisher") or link.get("provider") or ""),
                source_quality_label=assess_article_source(
                    Article(
                        canonical_url=str(link.get("url") or row["canonical_url"]),
                        title=str(link.get("title") or title),
                        metadata={"source_name": str(link.get("publisher") or link.get("provider") or "")},
                    )
                ).label,
            )
            for link in json.loads(str(row.get("supporting_links_json") or "[]"))
        )
        sorted_supporting_links = tuple(
            sorted(
                supporting_links,
                key=lambda link: source_quality_link_sort_key(link.title, link.url, link.source),
            )
        )
        primary_link = sorted_supporting_links[0].url if sorted_supporting_links else str(row.get("primary_link") or row["canonical_url"])
        primary_quality_label = sorted_supporting_links[0].source_quality_label if sorted_supporting_links else assess_article_source(
            Article(canonical_url=str(row.get("primary_link") or row["canonical_url"]), title=title)
        ).label
        event = EventClusterRow(
            ticker="",
            title=title,
            primary_link=primary_link,
            publisher_count=int(row.get("publisher_count") or 0),
            source_count=int(row.get("source_count") or 0),
            article_count=len(json.loads(str(row.get("article_ids_json") or "[]"))) or max(1, len(supporting_links)),
            first_seen_at=row.get("first_seen_at"),
            latest_seen_at=row.get("latest_seen_at"),
            primary_published_at=row.get("primary_published_at"),
            recency_bucket=str(row.get("recency_bucket") or "unknown"),
            tickers_mentioned=tuple(json.loads(str(row.get("tickers_mentioned_json") or "[]"))),
            weighted_cluster_sentiment=(
                float(row["weighted_cluster_sentiment"])
                if row.get("weighted_cluster_sentiment") is not None
                else None
            ),
            extraction_basis=str(extraction.get("extraction_basis") or "not_fetched"),
            source_quality_label=primary_quality_label,
            supporting_links=sorted_supporting_links,
        )
        tickers = match_tickers(title)
        if not tickers:
            tickers = match_tickers(" ".join(link.title for link in supporting_links[:3]))
        for ticker in tickers:
            intelligence = (
                market_intelligence.cluster_intelligence.get((ticker.symbol, event.title))
                if market_intelligence
                else None
            )
            grouped.setdefault(ticker.symbol, []).append(
                EventClusterRow(
                    ticker=ticker.symbol,
                    title=event.title,
                    primary_link=event.primary_link,
                    publisher_count=event.publisher_count,
                    source_count=event.source_count,
                    article_count=event.article_count,
                    first_seen_at=event.first_seen_at,
                    latest_seen_at=event.latest_seen_at,
                    primary_published_at=event.primary_published_at,
                    recency_bucket=event.recency_bucket,
                    tickers_mentioned=event.tickers_mentioned,
                    weighted_cluster_sentiment=event.weighted_cluster_sentiment,
                    extraction_basis=event.extraction_basis,
                    source_quality_label=event.source_quality_label,
                    cluster_summary=intelligence.cluster_summary if intelligence else event.title,
                    cluster_summary_basis=intelligence.cluster_summary_basis if intelligence else event.extraction_basis,
                    cluster_reading_priority=(
                        intelligence.cluster_reading_priority if intelligence else "background_only"
                    ),
                    ranking_score=intelligence.ranking_score if intelligence else 0.0,
                    supporting_links=event.supporting_links,
                )
            )
    return {
        symbol: tuple(
            sorted(
                rows,
                key=lambda event: (
                    -event.ranking_score,
                    event.source_quality_label,
                    -event.publisher_count,
                    -event.article_count,
                    event.title,
                ),
            )[:10]
        )
        for symbol, rows in grouped.items()
    }


def _article_links_from_event_clusters(
    event_clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]],
) -> dict[str, tuple[ArticleLink, ...]]:
    links_by_ticker: dict[str, tuple[ArticleLink, ...]] = {}
    for ticker in load_tracked_tickers():
        links: list[ArticleLink] = []
        for event in event_clusters_by_ticker.get(ticker.symbol, ()):
            links.append(ArticleLink(event.title, event.primary_link, f"{event.publisher_count} publisher(s)", event.source_quality_label))
        links_by_ticker[ticker.symbol] = tuple(links)
    return links_by_ticker


def _cap_event_clusters(
    clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]],
    *,
    max_email_stories: int,
) -> dict[str, tuple[EventClusterRow, ...]]:
    ordered = sorted(
        (
            cluster
            for clusters in clusters_by_ticker.values()
            for cluster in clusters
        ),
        key=lambda cluster: (
            -cluster.ranking_score,
            cluster.source_quality_label,
            -cluster.publisher_count,
            cluster.title,
        ),
    )
    grouped: dict[str, list[EventClusterRow]] = {}
    selected_count = 0
    for cluster in ordered:
        if selected_count >= max(1, max_email_stories):
            break
        if len(grouped.get(cluster.ticker, ())) >= 3:
            continue
        grouped.setdefault(cluster.ticker, []).append(cluster)
        selected_count += 1
    return {
        ticker: tuple(rows)
        for ticker, rows in grouped.items()
    }


def _cap_ranked_reads(
    reads_by_ticker: Mapping[str, tuple[object, ...]],
    *,
    max_ranked_reads_per_ticker: int,
) -> dict[str, tuple[object, ...]]:
    return {
        ticker: tuple(
            read
            for read in reads
            if getattr(read, "reading_priority", "background_only") != "background_only"
        )[: max(1, max_ranked_reads_per_ticker)]
        for ticker, reads in reads_by_ticker.items()
    }


def _extraction_summary_from_fetch_summary(
    article_fetch_summary: ArticleFetchSummary | None,
    extractions_by_article: Mapping[str, Mapping[str, object]],
    sentiment_basis_counts: Mapping[str, int],
    *,
    source_quality_summary: SourceQualitySummary | None = None,
    show_excluded_source_diagnostics: bool = False,
) -> ExtractionSummary:
    quality_summary = source_quality_summary or SourceQualitySummary()
    if article_fetch_summary is not None:
        return ExtractionSummary(
            article_pages_fetched=article_fetch_summary.attempted_fetches,
            publisher_article_fetches=article_fetch_summary.publisher_article_fetches,
            google_news_wrappers_skipped=article_fetch_summary.google_news_wrappers_skipped,
            google_news_wrappers_resolved=article_fetch_summary.google_news_wrappers_resolved,
            successful_extractions=article_fetch_summary.successful_extractions,
            failed_extractions=article_fetch_summary.failed_extractions,
            snippet_fallbacks=article_fetch_summary.snippet_fallbacks,
            title_fallbacks=article_fetch_summary.title_fallbacks,
            sentiment_basis_counts=sentiment_basis_counts,
            top_extraction_failure_reasons=article_fetch_summary.failure_reason_counts,
            extraction_method_counts=article_fetch_summary.extraction_method_counts,
            extraction_failure_reason=article_fetch_summary.extraction_failure_reason,
            extractor_diagnostics=article_fetch_summary.extractor_diagnostics or {},
            source_quality_summary=quality_summary,
            show_excluded_source_diagnostics=show_excluded_source_diagnostics,
            extraction_queue_size=article_fetch_summary.extraction_queue_size,
            extraction_selected_count=article_fetch_summary.extraction_selected_count,
            extraction_skipped_count=article_fetch_summary.extraction_skipped_count,
            extraction_skipped_reasons=article_fetch_summary.extraction_skipped_reasons,
            extraction_success_rate=article_fetch_summary.extraction_success_rate,
            extraction_success_rate_by_publisher=article_fetch_summary.extraction_success_rate_by_publisher,
            extraction_success_rate_by_source_provider=article_fetch_summary.extraction_success_rate_by_source_provider,
            extraction_attempted_count=article_fetch_summary.attempted_fetches,
            extraction_budget_unused_count=max(
                0,
                article_fetch_summary.max_article_fetches - article_fetch_summary.extraction_selected_count,
            ),
            direct_publisher_candidates=article_fetch_summary.direct_publisher_candidates,
            google_wrapper_candidates=article_fetch_summary.google_wrapper_candidates,
            google_wrappers_unresolved=article_fetch_summary.google_wrappers_unresolved,
            full_text_accepted_count=article_fetch_summary.successful_extractions,
            usable_full_text_count=article_fetch_summary.extraction_quality_grade_counts.get("usable_full_text", 0),
            weak_text_count=article_fetch_summary.extraction_quality_grade_counts.get("weak_text", 0),
            snippet_fallback_count=int(sentiment_basis_counts.get("snippet", 0)),
            title_fallback_count=int(sentiment_basis_counts.get("title", 0)),
            blocked_or_shell_count=article_fetch_summary.extraction_quality_grade_counts.get("blocked_or_shell", 0),
            extraction_quality_grade_counts=article_fetch_summary.extraction_quality_grade_counts,
            extractor_method_success_counts=article_fetch_summary.extraction_method_success_counts,
            publisher_success_rates=article_fetch_summary.extraction_success_rate_by_publisher,
            publisher_profiles=article_fetch_summary.publisher_profiles,
            top_unresolved_wrapper_publishers=article_fetch_summary.top_unresolved_wrapper_publishers,
        )
    fetched_count = sum(1 for row in extractions_by_article.values() if int(row.get("fetched") or 0))
    success_count = sum(1 for row in extractions_by_article.values() if str(row.get("extraction_basis")) == "full_text")
    return ExtractionSummary(
        article_pages_fetched=fetched_count,
        publisher_article_fetches=fetched_count,
        successful_extractions=success_count,
        failed_extractions=fetched_count - success_count,
        snippet_fallbacks=sum(1 for row in extractions_by_article.values() if str(row.get("extraction_basis")) == "snippet"),
        title_fallbacks=sum(1 for row in extractions_by_article.values() if str(row.get("extraction_basis")) == "title"),
        sentiment_basis_counts=sentiment_basis_counts,
        extraction_method_counts=_stored_extraction_method_counts(extractions_by_article),
        extraction_failure_reason=_stored_extraction_failure_reason(extractions_by_article),
        source_quality_summary=quality_summary,
        show_excluded_source_diagnostics=show_excluded_source_diagnostics,
    )


def _sentiment_row_from_stored(
    ticker: TrackedTicker,
    items: Sequence[dict[str, object]],
    event_clusters: Sequence[EventClusterRow],
    *,
    coverage: object | None = None,
) -> PortfolioSentimentRow:
    usable_items = [
        item
        for item in items
        if float(item.get("match_confidence", 1.0)) >= 0.4
        if item["recency"].recency_bucket in {"today_signal", "recent_pulse", "weekly_trend", "background_context"}
    ]
    bucket_scores = {
        bucket: _stored_average_score([item for item in usable_items if item["recency"].recency_bucket == bucket])
        for bucket in ("today_signal", "recent_pulse", "weekly_trend", "background_context")
    }
    counts = {
        bucket: sum(1 for item in usable_items if item["recency"].recency_bucket == bucket)
        for bucket in ("today_signal", "recent_pulse", "weekly_trend", "background_context")
    }
    article_count_24h = counts["today_signal"]
    article_count_3d = counts["today_signal"] + counts["recent_pulse"]
    article_count_7d = article_count_3d + counts["weekly_trend"]
    article_count_30d = article_count_7d + counts["background_context"]
    source_diversity = len(
        {
            str(item["source"].get("source_name") or item["source"].get("provider") or "")
            for item in usable_items
            if item.get("source")
        }
    )
    return PortfolioSentimentRow(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        sentiment_30d=_stored_average_score(usable_items),
        article_count_30d=article_count_30d,
        sentiment_basis=_stored_sentiment_basis(usable_items),
        today_signal_sentiment=bucket_scores["today_signal"],
        recent_pulse_sentiment=bucket_scores["recent_pulse"],
        weekly_trend_sentiment=bucket_scores["weekly_trend"],
        background_context_sentiment=bucket_scores["background_context"],
        weighted_sentiment_score=(
            float(getattr(coverage, "weighted_sentiment"))
            if coverage is not None
            else _weighted_sentiment_score(usable_items)
        ),
        article_count_24h=article_count_24h,
        article_count_3d=article_count_3d,
        article_count_7d=article_count_7d,
        mention_velocity=_mention_velocity(article_count_3d, article_count_30d, counts["background_context"]),
        source_diversity=source_diversity,
        top_event_clusters=tuple(event_clusters[:5]),
    )


def _weighted_sentiment_score(items: Sequence[dict[str, object]]) -> float:
    weighted_scores = [
        (
            float(item["score"].score),
            recency_weight(str(item["recency"].recency_bucket)),
        )
        for item in items
        if recency_weight(str(item["recency"].recency_bucket)) > 0
    ]
    total_weight = sum(weight for _score, weight in weighted_scores)
    if not total_weight:
        return 0.0
    return round(sum(score * weight for score, weight in weighted_scores) / total_weight, 4)


def _mention_velocity(article_count_3d: int, article_count_30d: int, background_count: int) -> str:
    if article_count_30d < 5 or background_count == 0:
        return "limited_history"
    recent_daily_rate = article_count_3d / 3.0
    context_daily_rate = max((article_count_30d - article_count_3d) / 27.0, 0.01)
    return f"{recent_daily_rate / context_daily_rate:.2f}x"


def _row_recency(run_date: str, article_row: Mapping[str, object]):
    return article_recency(
        run_date=run_date,
        published_at=article_row.get("published_at"),
        collected_at=article_row.get("created_at"),
        archive_context=False,
    )


def _provider_validation_results(
    args: argparse.Namespace,
    provider_checker: ProviderChecker | None,
    environ: Mapping[str, str] | None,
) -> list[dict[str, object]]:
    return [
        validate_provider(config, dry_run=args.dry_run, environ=environ, checker=provider_checker).as_safe_dict()
        for config in iter_provider_configs()
    ]


def _data_source_label(args: argparse.Namespace) -> str:
    if getattr(args, "enable_live_rss", False) and getattr(args, "include_fixtures", False):
        return "source-aware live acquisition plus local RSS fixtures"
    if getattr(args, "enable_live_rss", False):
        return "source-aware official, issuer, wire, publisher, and RSS acquisition"
    return "local RSS fixtures"


def _persist_run_articles(store: SQLiteStore, run_id: str, articles: Sequence[Article]) -> dict[str, str]:
    article_ids: dict[str, str] = {}
    for article in articles:
        article_id = store.add_run_article(run_id, article, store_full_text=False)
        article_ids[article.canonical_url] = article_id
        metadata = article.metadata
        store.add_article_source(
            ArticleSource(
                article_id=article_id,
                provider=str(metadata.get("provider") or "fixture_rss"),
                url=article.canonical_url,
                provider_article_id=metadata.get("provider_article_id"),
                title=article.title,
                snippet=article.snippet,
                published_at=article.published_at,
                source_name=metadata.get("source_name"),
                raw_metadata=metadata.get("raw_metadata") if isinstance(metadata.get("raw_metadata"), dict) else {},
            ),
            run_id=run_id,
        )
    return article_ids


def _persist_dedupe_clusters(store: SQLiteStore, run_id: str, clusters: Sequence[object], run_date: str) -> None:
    for index, cluster in enumerate(clusters, start=1):
        canonical = cluster.canonical_article
        canonical_article_id = store.article_id_for_url(canonical.canonical_url)
        article_ids = [
            store.article_id_for_url(article.canonical_url)
            for article in cluster.articles
        ]
        cluster_recency = _cluster_recency(cluster, run_date)
        store.record_dedupe_cluster(
            run_id=run_id,
            cluster_index=index,
            canonical_article_id=canonical_article_id,
            canonical_url=canonical.canonical_url,
            title=canonical.title,
            alternate_source_links=cluster.alternate_source_links,
            duplicate_reasons=cluster.duplicate_reasons,
            article_ids=article_ids,
            primary_link=cluster.primary_link,
            publisher_count=cluster.publisher_count,
            source_count=cluster.source_count,
            publisher_names=cluster.publisher_names,
            source_providers=cluster.source_providers,
            supporting_links=(_dataclass_to_dict(link) for link in cluster.supporting_links),
            first_seen_at=cluster_recency["first_seen_at"],
            latest_seen_at=cluster_recency["latest_seen_at"],
            primary_published_at=cluster_recency["primary_published_at"],
            recency_bucket=str(cluster_recency["recency_bucket"]),
            tickers_mentioned=cluster_recency["tickers_mentioned"],
            weighted_cluster_sentiment=cluster_recency["weighted_cluster_sentiment"],
        )


def _persist_article_extractions(
    store: SQLiteStore,
    run_id: str,
    article_fetch_summary: ArticleFetchSummary,
    article_ids_by_url: Mapping[str, str],
) -> None:
    for record in article_fetch_summary.records:
        article_id = article_ids_by_url.get(record.canonical_url) or record.article_id
        if not article_id:
            continue
        store.record_article_extraction(
            run_id=run_id,
            article_id=article_id,
            canonical_url=record.canonical_url,
            extraction_status=record.extraction_status,
            extraction_basis=record.extraction_basis,
            error_class=record.error_class,
            final_url=record.final_url,
            latency_ms=record.latency_ms,
            content_type=record.content_type,
            content_length=record.content_length,
            text_hash=record.text_hash,
            extracted_preview=record.extracted_preview,
            extractor=record.extractor,
            extraction_method_used=record.extraction_method_used,
            extraction_failure_reason=record.extraction_failure_reason,
            fetched=record.fetched,
            tickers=record.tickers,
        )


def _persist_sentiment_and_mentions(store: SQLiteStore, run_id: str, articles: Sequence[Article]) -> None:
    ticker_lookup = {ticker.symbol: ticker for ticker in load_tracked_tickers()}
    for article in articles:
        article_id = store.add_run_article(run_id, article, store_full_text=False)
        sentiment = analyze_sentiment(
            article_id,
            _article_text(article),
            _article_basis(article),
        )
        store.add_sentiment_result(sentiment, run_id=run_id)
        for match in assess_ticker_matches(article):
            tracked = ticker_lookup[match.ticker]
            store.add_ticker_mention(
                TickerMention(
                    article_id=article_id,
                    ticker=match.ticker,
                    confidence=match.confidence,
                    company_name=tracked.company_name,
                    basis=sentiment.basis,
                ),
                run_id=run_id,
            )


def _scored_articles_by_ticker(articles: Sequence[Article]) -> dict[str, tuple[dict[str, object], ...]]:
    grouped: dict[str, list[dict[str, object]]] = {ticker.symbol: [] for ticker in load_tracked_tickers()}
    for article in articles:
        text = _article_text(article)
        score = analyze_sentiment(
            article.article_id or article.canonical_url,
            text,
            _article_basis(article),
        )
        for ticker in match_tickers(text):
            grouped[ticker.symbol].append({"article": article, "score": score})
    return {symbol: tuple(items) for symbol, items in grouped.items()}


def _score_articles(articles: Sequence[Article]) -> list[dict[str, object]]:
    return [
        _dataclass_to_dict(
            analyze_sentiment(
                article.article_id or article.canonical_url,
                _article_text(article),
                _article_basis(article),
            )
        )
        for article in articles
    ]


def _score_dict_basis_counts(scores: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"full_text": 0, "snippet": 0, "title": 0}
    for score in scores:
        basis = str(score.get("basis") or "")
        if basis in counts:
            counts[basis] += 1
    return counts


def _stored_extraction_method_counts(extractions_by_article: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in extractions_by_article.values():
        method = str(row.get("extraction_method_used") or row.get("extractor") or "")
        if not method:
            continue
        counts[method] = counts.get(method, 0) + 1
    return counts


def _stored_extraction_failure_reason(extractions_by_article: Mapping[str, Mapping[str, object]]) -> str | None:
    counts: dict[str, int] = {}
    for row in extractions_by_article.values():
        reason = str(row.get("extraction_failure_reason") or row.get("error_class") or "")
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]


def _article_text(article: Article) -> str:
    return article.full_text or article.snippet or article.title


def _article_basis(article: Article) -> str:
    if article.full_text:
        return "full_text"
    if article.snippet:
        return "snippet"
    return "title"


def _average_score(items: Sequence[dict[str, object]]) -> float:
    if not items:
        return 0.0
    return round(sum(float(item["score"].score) for item in items) / len(items), 4)


def _sentiment_basis(items: Sequence[dict[str, object]]) -> str:
    if not items:
        return "no_articles"
    bases = [str(item["score"].basis) for item in items]
    for basis in ("full_text", "snippet", "title"):
        if basis in bases:
            return basis
    return bases[0]


def _watchlist_forecast_row(ticker: TrackedTicker, items: Sequence[dict[str, object]]) -> WatchlistForecastRow:
    average = _average_score(items)
    if not items:
        return WatchlistForecastRow(ticker.symbol, "uncertain", 0.0, "no current report articles")
    if average > 0.1:
        direction = "up"
    elif average < -0.1:
        direction = "down"
    elif abs(average) <= 0.1:
        direction = "flat"
    else:
        direction = "uncertain"
    confidence = round(min(1.0, 0.5 + abs(average) / 2.0), 4)
    return WatchlistForecastRow(
        ticker.symbol,
        direction,
        confidence,
        f"current report sentiment from {len(items)} article(s)",
    )


def _stored_average_score(items: Sequence[dict[str, object]]) -> float:
    if not items:
        return 0.0
    return round(sum(float(item["score"].score) for item in items) / len(items), 4)


def _stored_sentiment_basis(items: Sequence[dict[str, object]]) -> str:
    if not items:
        return "no_articles"
    bases = [str(item["score"].basis) for item in items]
    for basis in ("full_text", "snippet", "title"):
        if basis in bases:
            return basis
    return bases[0]


def _sentiment_basis_counts(scores: Sequence[SentimentResult]) -> dict[str, int]:
    counts = {"full_text": 0, "snippet": 0, "title": 0}
    for score in scores:
        if score.basis in counts:
            counts[score.basis] += 1
    return counts


def _watchlist_forecast_row_from_stored(ticker: TrackedTicker, items: Sequence[dict[str, object]]) -> WatchlistForecastRow:
    average = _stored_average_score(items)
    if not items:
        return WatchlistForecastRow(ticker.symbol, "uncertain", 0.0, "no current report articles")
    if average > 0.1:
        direction = "up"
    elif average < -0.1:
        direction = "down"
    else:
        direction = "flat"
    confidence = round(min(1.0, 0.5 + abs(average) / 2.0), 4)
    return WatchlistForecastRow(
        ticker.symbol,
        direction,
        confidence,
        f"placeholder direction from current report sentiment for {len(items)} article(s)",
    )


def _cluster_recency(cluster, run_date: str) -> dict[str, object]:
    article_infos = [
        (
            article,
            article_recency(
                run_date=run_date,
                published_at=article.published_at,
                collected_at=article.created_at,
                archive_context=bool(article.metadata.get("archive_context")),
            ),
        )
        for article in cluster.articles
    ]
    timestamps = sorted(info.timestamp_utc for _article, info in article_infos if info.timestamp_utc)
    latest_info = max(
        (info for _article, info in article_infos if info.timestamp_utc),
        key=lambda info: info.timestamp_utc,
        default=None,
    )
    primary_info = article_recency(
        run_date=run_date,
        published_at=cluster.canonical_article.published_at,
        collected_at=cluster.canonical_article.created_at,
        archive_context=bool(cluster.canonical_article.metadata.get("archive_context")),
    )
    tickers = tuple(
        sorted(
            {
                ticker.symbol
                for article in cluster.articles
                for ticker in match_tickers(_article_text(article))
            }
        )
    )
    weighted_scores = []
    for article, info in article_infos:
        weight = recency_weight(info.recency_bucket)
        if weight <= 0:
            continue
        score = analyze_sentiment(
            article.article_id or article.canonical_url,
            _article_text(article),
            _article_basis(article),
        )
        weighted_scores.append((score.score, weight))
    total_weight = sum(weight for _score, weight in weighted_scores)
    weighted_cluster_sentiment = (
        round(sum(score * weight for score, weight in weighted_scores) / total_weight, 4)
        if total_weight
        else None
    )
    return {
        "first_seen_at": timestamps[0] if timestamps else None,
        "latest_seen_at": timestamps[-1] if timestamps else None,
        "primary_published_at": primary_info.timestamp_utc,
        "recency_bucket": latest_info.recency_bucket if latest_info else "unknown",
        "tickers_mentioned": tickers,
        "weighted_cluster_sentiment": weighted_cluster_sentiment,
    }


def _cluster_to_dict(cluster, run_date: str) -> dict[str, object]:
    cluster_recency = _cluster_recency(cluster, run_date)
    return {
        "canonical_url": cluster.canonical_article.canonical_url,
        "title": cluster.canonical_article.title,
        "primary_link": cluster.primary_link,
        "first_seen_at": cluster_recency["first_seen_at"],
        "latest_seen_at": cluster_recency["latest_seen_at"],
        "primary_published_at": cluster_recency["primary_published_at"],
        "recency_bucket": cluster_recency["recency_bucket"],
        "tickers_mentioned": list(cluster_recency["tickers_mentioned"]),
        "weighted_cluster_sentiment": cluster_recency["weighted_cluster_sentiment"],
        "publisher_count": cluster.publisher_count,
        "source_count": cluster.source_count,
        "publisher_names": list(cluster.publisher_names),
        "source_providers": list(cluster.source_providers),
        "supporting_links": [_dataclass_to_dict(link) for link in cluster.supporting_links],
        "alternate_source_links": list(cluster.alternate_source_links),
        "duplicate_reasons": list(cluster.duplicate_reasons),
        "cluster_id": cluster.cluster_id,
        "primary_ticker": cluster.primary_ticker,
        "matched_tickers": list(cluster.matched_tickers),
        "related_tickers": list(cluster.related_tickers),
        "event_type": cluster.event_type,
        "primary_article_id": cluster.primary_article_id,
        "supporting_article_ids": list(cluster.supporting_article_ids),
        "supporting_publishers": list(cluster.supporting_publishers),
        "source_diversity": cluster.source_diversity,
        "publisher_diversity": cluster.publisher_diversity,
    }


def _dedupe_diagnostics(
    articles: Sequence[Article],
    clusters: Sequence[object],
) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    for cluster in clusters:
        for reason in cluster.duplicate_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "input_article_count": len(articles),
        "cluster_count": len(clusters),
        "duplicate_article_count": max(0, len(articles) - len(clusters)),
        "duplicate_reason_counts": reason_counts,
        "multi_publisher_cluster_count": sum(1 for cluster in clusters if cluster.publisher_count > 1),
        "multi_source_cluster_count": sum(1 for cluster in clusters if cluster.source_count > 1),
    }


def _write_markdown_report(report: DailyReportContract) -> str:
    output_dir = Path(report.output_dir)
    report_path = output_dir / "daily_report.md"
    lines = [
        f"# Portfolio and Watchlist Market Briefing - {report.report_date}",
        "",
        "This briefing is not investment advice. Direction rows are not predictions.",
        "",
        *_markdown_daily_briefing(report),
        *_markdown_source_coverage_line(report),
        "",
        *_markdown_ticker_summaries("Portfolio Summary", report.portfolio_summaries),
        "",
        *_markdown_ticker_summaries("Watchlist Summary", report.watchlist_summaries),
        "",
        *_markdown_stories_to_watch(report),
        "",
        *_markdown_ranked_reads(report),
        "",
        *_markdown_sentiment_coverage(report),
        "",
        *_markdown_recency_sections(report),
        "",
        "## Portfolio Recency Sentiment",
        "",
        "| Ticker | Company | Weighted | Today | 1-3D | 4-7D | 8-30D | 24H | 3D | 7D | 30D | Velocity | Sources | Basis |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    lines.extend(
        _markdown_sentiment_row(row)
        for row in report.portfolio_30d_sentiment_table
    )
    lines.extend(
        [
            "",
            "## Watchlist Recency Sentiment",
            "",
            "| Ticker | Company | Weighted | Today | 1-3D | 4-7D | 8-30D | 24H | 3D | 7D | 30D | Velocity | Sources | Basis |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    lines.extend(
        _markdown_sentiment_row(row)
        for row in report.watchlist_sentiment_table
    )
    lines.extend(
        [
            "",
            "## Watchlist Next Close",
            "",
            "Placeholder direction logic: direction is derived from current report sentiment. These rows are not real predictions.",
            "",
            "| Ticker | Direction | Confidence | Driver |",
            "| --- | --- | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {row.ticker} | {row.next_close_direction} | {row.confidence:.4f} | {_escape_markdown(row.driver)} |"
        for row in report.watchlist_next_close_table
    )
    lines.extend(
        [
            "",
            "## Top 7 Day Mention Leaders",
            "",
            "| Ticker | Mentions | Average Sentiment |",
            "| --- | ---: | ---: |",
        ]
    )
    if report.mention_leaders_7d_table:
        lines.extend(
            f"| {row.ticker} | {row.mentions_7d} | {row.sentiment_avg:.4f} |"
            for row in report.mention_leaders_7d_table
        )
    else:
        lines.append("| none | 0 | 0.0000 |")
    lines.extend(
        [
            "",
            "## Top 10 Most Mentioned Tickers",
            "",
            "| Rank | Ticker | Mentions |",
            "| ---: | --- | ---: |",
        ]
    )
    if report.top_10_most_mentioned_table:
        lines.extend(
            f"| {row.rank} | {row.ticker} | {row.mentions} |"
            for row in report.top_10_most_mentioned_table
        )
    else:
        lines.append("| 0 | none | 0 |")
    lines.extend(
        [
            "",
            "## Emerging Names Based On Mention Velocity",
            "",
            "| Ticker | Company | Mentions | Prior Mentions | Reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    if report.emerging_names_table:
        lines.extend(
            f"| {row.ticker} | {_escape_markdown(row.company_name)} | {row.mentions_7d} | {row.prior_mentions_30d} | {_escape_markdown(row.reason)} |"
            for row in report.emerging_names_table
        )
    else:
        lines.append("| none | none | 0 | 0 | no emerging watchlist names in current report data |")
    lines.extend(["", *_markdown_read_more(report), "", *_markdown_diagnostics(report)])
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(report_path)


def _markdown_daily_briefing(report: DailyReportContract) -> list[str]:
    bullets = []
    if report.top_10_most_mentioned_table:
        leader = report.top_10_most_mentioned_table[0]
        bullets.append(f"Top mention leader: {leader.ticker} with {leader.mentions} current report mention(s).")
    for cluster in report.top_event_clusters[:2]:
        bullets.append(cluster.cluster_summary or f"{cluster.ticker}: {cluster.title}")
    bullets.append(
        f"Full text extraction succeeded for {report.extraction_summary.successful_extractions} article(s); "
        "remaining summaries use snippets or titles."
    )
    bullets.append("Direction rows are not predictions.")
    return ["## Daily Briefing", "", *[f"- {bullet}" for bullet in bullets[:5]]]


def _markdown_ticker_summaries(
    title: str,
    rows: tuple[object, ...],
) -> list[str]:
    lines = [f"## {title}", ""]
    covered = [
        row
        for row in rows
        if row.read_first_story or row.read_next_story or row.background_story
    ]
    if not covered:
        return [*lines, "No matched stories were available for configured names."]
    lines.extend(f"- {_escape_markdown(row.ticker_daily_summary)}" for row in covered)
    return lines


def _markdown_source_coverage_line(report: DailyReportContract) -> list[str]:
    diagnostics = report.source_coverage_diagnostics
    paid = diagnostics.get("paid_api_count", 0)
    paid_label = str(paid) if paid else "disabled"
    return [
        "**Source Coverage:** "
        f"Official filings: {int(diagnostics.get('official_source_count', 0))} | "
        f"Press releases: {int(diagnostics.get('press_release_wire_count', 0))} | "
        f"Direct publishers: {int(diagnostics.get('direct_publisher_count', 0))} | "
        f"Google backstop: {int(diagnostics.get('google_news_backstop_count', 0))} | "
        f"Paid APIs: {paid_label}",
        "",
    ]


def _markdown_stories_to_watch(report: DailyReportContract) -> list[str]:
    lines = ["## Stories to Watch", ""]
    written = False
    for ticker, clusters in sorted(report.event_clusters_by_ticker.items()):
        visible = clusters[:5]
        if not visible:
            continue
        written = True
        lines.extend(
            [
                f"### {ticker}",
                "",
                "| Story | Summary | Priority | Basis | Bucket | Publishers | Sources |",
                "| --- | --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| [{_escape_markdown(cluster.title)}]({cluster.primary_link}) | "
            f"{_escape_markdown(cluster.cluster_summary or cluster.title)} | "
            f"{cluster.cluster_reading_priority} | {cluster.cluster_summary_basis} | "
            f"{cluster.recency_bucket} | {cluster.publisher_count} | {cluster.source_count} |"
            for cluster in visible
        )
        lines.append("")
    if not written:
        lines.append("No event clusters matched configured tickers.")
    return lines


def _markdown_ranked_reads(report: DailyReportContract) -> list[str]:
    lines = ["## Ranked Reads By Ticker", ""]
    written = False
    for ticker, reads in sorted(report.ranked_reads_by_ticker.items()):
        visible = [read for read in reads if read.reading_priority != "background_only"][:2]
        if not visible:
            continue
        written = True
        lines.append(f"### {ticker}")
        for read in visible:
            lines.append(
                f"- **{read.reading_priority}:** [{_escape_markdown(read.title)}]({read.url}) "
                f"({_escape_markdown(read.source)}, {read.summary_basis}) - "
                f"{_escape_markdown(read.article_summary)}"
            )
        lines.append("")
    if not written:
        lines.append("No ranked reads were available for configured tickers.")
    return lines


def _markdown_sentiment_coverage(report: DailyReportContract) -> list[str]:
    lines = [
        "## Sentiment Coverage Summary",
        "",
        "| Ticker | Coverage | Weighted Sentiment | Scored | Full Text | Snippets | High Confidence | Low Confidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [row for row in report.ticker_sentiment_coverage.values() if row.article_count_scored]
    if not rows:
        return [*lines, "| none | weak | 0.0000 | 0 | 0 | 0 | 0 | 0 |"]
    lines.extend(
        f"| {row.ticker} | {row.sentiment_coverage_grade} | {row.weighted_sentiment:.4f} | "
        f"{row.article_count_scored} | {row.full_text_scored_count} | {row.snippet_scored_count} | "
        f"{row.high_confidence_article_count} | {row.low_confidence_article_count} |"
        for row in sorted(rows, key=lambda item: item.ticker)
    )
    return lines


def _markdown_read_more(report: DailyReportContract) -> list[str]:
    lines = ["## Read More By Ticker", ""]
    linked = False
    for ticker, links in sorted(report.supporting_article_links.items()):
        if not links:
            continue
        linked = True
        lines.append(f"### {ticker}")
        visible_links = links[:10]
        for link in visible_links:
            source = f" ({_escape_markdown(link.source)})" if link.source else ""
            lines.append(f"- [{_escape_markdown(link.title)}]({link.url}){source}")
        if len(links) > len(visible_links):
            lines.append(f"- +{len(links) - len(visible_links)} more links in JSON artifacts")
        lines.append("")
    if not linked:
        lines.append("No article links matched configured tickers.")
    return lines


def _markdown_diagnostics(report: DailyReportContract) -> list[str]:
    warnings = " ".join(report.report_warnings) or "none"
    backend = report.backend_article_pool_summary
    email = report.email_display_summary
    return [
        "## Source and Extraction Diagnostics",
        "",
        "| Report Diagnostic | Value |",
        "| --- | --- |",
        f"| Run date | {report.report_date} |",
        f"| Data source | {_escape_markdown(report.data_source_label)} |",
        "| Delivery mode | local report only; no email sent |",
        f"| Paid API status | {_escape_markdown(str(report.source_coverage_diagnostics.get('paid_api_status', 'disabled')))} |",
        f"| Warnings | {_escape_markdown(warnings)} |",
        f"| Report summary | {_escape_markdown(report.daily_summary)} |",
        "",
        "Sentiment is deterministic placeholder logic until a stronger model is wired in. "
        "Summaries use extracted full text when available, otherwise snippets or titles.",
        "",
        "### Backend and Email Pool Summary",
        "",
        "| Backend Candidates | Backend Visible | Backend Scored | Backend Full Text | Email Stories | Email Ranked Reads |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {backend.backend_candidate_articles} | {backend.backend_visible_articles} | "
        f"{backend.backend_scored_articles} | {backend.backend_extracted_articles} | "
        f"{email.email_visible_stories} | {email.email_visible_ranked_reads} |",
        "",
        "### Source Acquisition Summary",
        "",
        "| Official | Company IR | Press Wires | Direct Publishers | Google Backstop | Google Share | Diversity | Balance |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {int(report.source_coverage_diagnostics.get('official_source_count', 0))} | "
        f"{int(report.source_coverage_diagnostics.get('company_ir_count', 0))} | "
        f"{int(report.source_coverage_diagnostics.get('press_release_wire_count', 0))} | "
        f"{int(report.source_coverage_diagnostics.get('direct_publisher_count', 0))} | "
        f"{int(report.source_coverage_diagnostics.get('google_news_backstop_count', 0))} | "
        f"{float(report.source_coverage_diagnostics.get('google_news_share', 0.0)):.1%} | "
        f"{float(report.source_coverage_diagnostics.get('source_diversity_score', 0.0)):.1f} | "
        f"{float(report.source_coverage_diagnostics.get('source_balance_score', 0.0)):.1f} |",
        "",
        "### Source Quality Summary",
        "",
        "| Total Articles | Visible Articles | Excluded Articles | Tier 1 | Tier 2 | Tier 3 Visible | Tier 3 Hidden | Tier 4 Excluded | Unknown |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _markdown_source_quality_summary_row(report.extraction_summary.source_quality_summary),
        "",
        "### Article Extraction Summary",
        "",
        "| Article Fetch Attempts | Publisher Article Fetches | Google Wrappers Skipped | Google Wrappers Resolved | Full Text Successes | Snippet Fallbacks | Title Fallbacks |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _markdown_extraction_summary_row(report.extraction_summary),
        "",
        "| Full Text Basis | Snippet Basis | Title Basis | Trafilatura | Newspaper3k | Internal Parser |",
        "| ---: | ---: | ---: | --- | --- | --- |",
        _markdown_extractor_status_row(report.extraction_summary),
        "",
        "| Extraction Diagnostic | Value |",
        "| --- | --- |",
        *_markdown_extraction_diagnostic_rows(report.extraction_summary),
        f"| extraction_queue_size | {report.extraction_summary.extraction_queue_size} |",
        f"| extraction_selected_count | {report.extraction_summary.extraction_selected_count} |",
        f"| extraction_skipped_count | {report.extraction_summary.extraction_skipped_count} |",
        f"| extraction_success_rate | {report.extraction_summary.extraction_success_rate:.4f} |",
        "",
        *_markdown_failure_reason_rows(report.extraction_summary),
    ]


def _escape_markdown(value: str | None) -> str:
    return (value or "").replace("|", "\\|")


def _markdown_source_quality_summary_row(summary: SourceQualitySummary) -> str:
    return (
        f"| {summary.total_articles} | {summary.visible_articles} | {summary.excluded_articles} | "
        f"{summary.tier_1_articles} | "
        f"{summary.tier_2_articles} | "
        f"{summary.tier_3_visible_articles} | "
        f"{summary.tier_3_hidden_articles} | "
        f"{summary.tier_4_excluded_articles} | "
        f"{summary.unknown_articles} |"
    )


def _markdown_extraction_summary_row(summary: ExtractionSummary) -> str:
    return (
        f"| {summary.article_pages_fetched} | {summary.publisher_article_fetches} | "
        f"{summary.google_news_wrappers_skipped} | {summary.google_news_wrappers_resolved} | "
        f"{summary.successful_extractions} | {summary.snippet_fallbacks} | {summary.title_fallbacks} |"
    )


def _markdown_extractor_status_row(summary: ExtractionSummary) -> str:
    basis_counts = {basis: int(summary.sentiment_basis_counts.get(basis, 0)) for basis in ("full_text", "snippet", "title")}
    diagnostics = summary.extractor_diagnostics
    return (
        f"| {basis_counts['full_text']} | {basis_counts['snippet']} | {basis_counts['title']} | "
        f"{_availability(diagnostics.get('trafilatura_available'))} | "
        f"{_availability(diagnostics.get('newspaper3k_available'))} | "
        f"{_availability(diagnostics.get('internal_parser_available'))} |"
    )


def _markdown_extraction_diagnostic_rows(summary: ExtractionSummary) -> list[str]:
    diagnostics = summary.extractor_diagnostics
    return [
        f"| trafilatura_available | {_availability(diagnostics.get('trafilatura_available'))} |",
        f"| newspaper3k_available | {_availability(diagnostics.get('newspaper3k_available'))} |",
        f"| internal_parser_available | {_availability(diagnostics.get('internal_parser_available'))} |",
        f"| extraction_method_used | {_escape_markdown(_method_counts(summary.extraction_method_counts))} |",
        f"| extraction_failure_reason | {_escape_markdown(summary.extraction_failure_reason or 'none')} |",
    ]


def _markdown_failure_reason_rows(summary: ExtractionSummary) -> list[str]:
    lines = [
        "### Top Extraction Failure Reasons",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    if not summary.top_extraction_failure_reasons:
        lines.append("| none | 0 |")
        return lines
    lines.extend(
        f"| {_escape_markdown(reason)} | {int(count)} |"
        for reason, count in sorted(
            summary.top_extraction_failure_reasons.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:8]
    )
    return lines


def _availability(value: bool | None) -> str:
    return "available" if value else "missing"


def _method_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{method}={int(count)}"
        for method, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    )


def _markdown_recency_sections(report: DailyReportContract) -> list[str]:
    rows = list(report.portfolio_30d_sentiment_table) + list(report.watchlist_sentiment_table)
    lines: list[str] = []
    for title, description, sentiment_field, count_field in (
        ("Today's Signal", "0-24 hours before run date. Weight: 1.0.", "today_signal_sentiment", "article_count_24h"),
        ("Recent Pulse", "1-3 days before run date. Weight: 0.7.", "recent_pulse_sentiment", "article_count_3d"),
        ("Weekly Trend", "4-7 days before run date. Weight: 0.4.", "weekly_trend_sentiment", "article_count_7d"),
        ("Background Context", "8-30 days before run date. Weight: 0.15. Older archive context is excluded from daily sentiment.", "background_context_sentiment", "article_count_30d"),
    ):
        ranked = sorted(
            rows,
            key=lambda row: (getattr(row, count_field), abs(getattr(row, sentiment_field)), row.source_diversity),
            reverse=True,
        )[:8]
        lines.extend(
            [
                f"## {title}",
                "",
                description,
                "",
                "| Ticker | Sentiment | Articles | Source Diversity | Velocity |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        if ranked:
            lines.extend(
                f"| {row.ticker} | {getattr(row, sentiment_field):.4f} | {getattr(row, count_field)} | {row.source_diversity} | {row.mention_velocity} |"
                for row in ranked
            )
        else:
            lines.append("| none | 0.0000 | 0 | 0 | limited_history |")
        lines.append("")
    return lines


def _markdown_sentiment_row(row) -> str:
    return (
        f"| {row.ticker} | {_escape_markdown(row.company_name)} | {row.weighted_sentiment_score:.4f} | "
        f"{row.today_signal_sentiment:.4f} | {row.recent_pulse_sentiment:.4f} | "
        f"{row.weekly_trend_sentiment:.4f} | {row.background_context_sentiment:.4f} | "
        f"{row.article_count_24h} | {row.article_count_3d} | {row.article_count_7d} | "
        f"{row.article_count_30d} | {row.mention_velocity} | {row.source_diversity} | {row.sentiment_basis} |"
    )


def _optional_markdown_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _run_command(
    command: str,
    args: argparse.Namespace,
    output_dir: Path,
    fake_providers: Mapping[str, FakeProvider] | None,
    provider_checker: ProviderChecker | None,
    environ: Mapping[str, str] | None,
) -> None:
    if command == "validate-providers":
        results = [
            validate_provider(config, dry_run=args.dry_run, environ=environ, checker=provider_checker).as_safe_dict()
            for config in iter_provider_configs()
        ]
        _write_json(output_dir / "provider_validation.json", {**_safe_context(args), "providers": results})


if __name__ == "__main__":
    raise SystemExit(main())
