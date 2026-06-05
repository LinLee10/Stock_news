"""Safe local orchestration CLI for the canonical news pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .article_fetch import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_ARTICLE_FETCHES,
    DEFAULT_MAX_FETCHES_PER_TICKER,
    ArticleFetchSummary,
    disabled_article_fetch_summary,
    fetch_top_cluster_articles,
)
from .dedup import cluster_articles
from .email_preview import PreviewEmailSender
from .models import Article, ArticleSource, RunResult, SentimentResult, TickerMention
from .provider_registry import iter_provider_configs
from .provider_usage import ProviderUsageRecorder
from .provider_validation import ProviderChecker, validate_provider
from .recency import article_recency, recency_weight
from .reporting import (
    ArticleLink,
    DailyReportInput,
    DailyReportContract,
    EmergingNameRow,
    EventClusterRow,
    ExtractionSummary,
    MentionLeaderRow,
    MostMentionedRow,
    PortfolioSentimentRow,
    WatchlistForecastRow,
    build_daily_report,
)
from .sentiment import analyze_sentiment
from .sources.live_rss import (
    DEFAULT_LIVE_RSS_RETRIES,
    DEFAULT_LIVE_RSS_TIMEOUT_SECONDS,
    DEFAULT_LIVE_RSS_USER_AGENT,
    collect_live_rss_articles,
    default_live_rss_urls,
)
from .sources.rss_config import (
    DEFAULT_MAX_ARTICLES_PER_SOURCE,
    DEFAULT_MAX_ARTICLES_PER_TICKER,
    DEFAULT_MAX_TOTAL_LIVE_ARTICLES,
)
from .sources.rss import RssSource
from .storage import SQLiteStore, initialize_database
from .tickers import TrackedTicker, load_portfolio, load_tracked_tickers, load_watchlist, match_tickers


class FakeProvider(Protocol):
    def articles(self) -> list[Article]:
        """Return fake local articles without network access."""


DEFAULT_RSS_FIXTURES_DIR = Path("tests/fixtures/rss")


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
            subparser.add_argument("--run-id")

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fake_providers: Mapping[str, FakeProvider] | None = None,
    provider_checker: ProviderChecker | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = _run_output_dir(args.artifacts_dir, args.run_date)
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

            articles, live_rss_summary = _collect_daily_articles(args, fake_providers, store, run_id)
            _write_json(
                output_dir / "provider_validation.json",
                {**context, "providers": provider_results, "live_rss": live_rss_summary},
            )
            clusters = cluster_articles(articles)
            enriched_articles_by_url, article_fetch_summary = _run_article_fetch_stage(args, clusters)
            articles_for_storage = [_enriched_article(article, enriched_articles_by_url) for article in articles]
            canonical_articles = [
                _enriched_article(cluster.canonical_article, enriched_articles_by_url)
                for cluster in clusters
            ]
            persisted_article_ids = _persist_run_articles(store, run_id, articles_for_storage)
            scores = _score_articles(canonical_articles)
            sentiment_basis_counts = _score_dict_basis_counts(scores)
            _persist_article_extractions(store, run_id, article_fetch_summary, persisted_article_ids)
            _persist_dedupe_clusters(store, run_id, clusters, args.run_date)
            _persist_sentiment_and_mentions(store, run_id, canonical_articles)

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
                _daily_report_input_from_store(store, run_id, args.run_date, article_fetch_summary),
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
            "run_id": run_id,
            "database_path": str(database_path),
            "article_count": len(articles),
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
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--rss-fixtures-dir", default=str(DEFAULT_RSS_FIXTURES_DIR))
    parser.add_argument("--include-fixtures", action="store_true")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-paid-apis", action="store_true", default=False)
    parser.add_argument("--enable-email-send", action="store_true", default=False)


def _add_live_rss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-live-rss", action="store_true", default=False)
    parser.add_argument("--live-rss-url", action="append", default=None)
    parser.add_argument("--live-rss-timeout-seconds", type=float, default=DEFAULT_LIVE_RSS_TIMEOUT_SECONDS)
    parser.add_argument("--live-rss-retries", type=int, default=DEFAULT_LIVE_RSS_RETRIES)
    parser.add_argument("--live-rss-user-agent", default=DEFAULT_LIVE_RSS_USER_AGENT)
    parser.add_argument("--live-rss-max-articles-per-source", type=int, default=DEFAULT_MAX_ARTICLES_PER_SOURCE)
    parser.add_argument("--live-rss-max-articles-per-ticker", type=int, default=DEFAULT_MAX_ARTICLES_PER_TICKER)
    parser.add_argument("--live-rss-max-total-articles", type=int, default=DEFAULT_MAX_TOTAL_LIVE_ARTICLES)


def _add_live_article_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-live-article-fetch", action="store_true", default=False)
    parser.add_argument("--max-article-fetches", type=int, default=DEFAULT_MAX_ARTICLE_FETCHES)
    parser.add_argument("--max-fetches-per-ticker", type=int, default=DEFAULT_MAX_FETCHES_PER_TICKER)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=DEFAULT_FETCH_TIMEOUT_SECONDS)


def _safe_context(args: argparse.Namespace) -> dict[str, object]:
    return {
        "command": args.command,
        "run_date": args.run_date,
        "dry_run": args.dry_run,
        "paid_apis_enabled": args.enable_paid_apis,
        "email_send_enabled": args.enable_email_send,
        "live_rss_enabled": bool(getattr(args, "enable_live_rss", False)),
        "live_article_fetch_enabled": bool(getattr(args, "enable_live_article_fetch", False)),
        "fixtures_included": bool(getattr(args, "include_fixtures", False)) or not bool(getattr(args, "enable_live_rss", False)),
        "output_dir": str(_run_output_dir(args.artifacts_dir, args.run_date)),
        "rss_fixtures_dir": args.rss_fixtures_dir,
    }


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
            "attempts": [],
        }

    feed_urls = tuple(args.live_rss_url) if args.live_rss_url else None
    live_articles, attempts = collect_live_rss_articles(
        feed_urls=feed_urls,
        timeout_seconds=max(0.1, float(args.live_rss_timeout_seconds)),
        retries=max(0, int(args.live_rss_retries)),
        user_agent=str(args.live_rss_user_agent),
        max_articles_per_source=max(1, int(args.live_rss_max_articles_per_source)),
        max_articles_per_ticker=max(1, int(args.live_rss_max_articles_per_ticker)),
        max_total_articles=max(1, int(args.live_rss_max_total_articles)),
    )
    recorder = ProviderUsageRecorder(store)
    for attempt in attempts:
        recorder.record(
            attempt.provider,
            "discover",
            attempt.status,
            quota_cost=0,
            article_count=attempt.article_count,
            latency_ms=attempt.latency_ms,
            error_class=attempt.error_class,
            metadata={
                "run_id": run_id,
                "feed_id": attempt.feed_id,
                "feed_url": attempt.feed_url,
                "attempts": attempt.attempts,
                "fetched_article_count": attempt.fetched_article_count,
            },
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
        "caps": {
            "max_articles_per_source": max(1, int(args.live_rss_max_articles_per_source)),
            "max_articles_per_ticker": max(1, int(args.live_rss_max_articles_per_ticker)),
            "max_total_articles": max(1, int(args.live_rss_max_total_articles)),
        },
        "attempts": [attempt.as_dict() for attempt in attempts],
    }


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
    )


def _enriched_article(article: Article, enriched_articles_by_url: Mapping[str, Article]) -> Article:
    return enriched_articles_by_url.get(article.canonical_url, article)


def _canonical_articles(articles: list[Article]) -> list[Article]:
    return [cluster.canonical_article for cluster in cluster_articles(articles)]


def _is_paid_provider(provider_name: str) -> bool:
    return provider_name in {"alpha_vantage", "marketaux", "gnews"}


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
    payload = {
        "canonical_url": article.canonical_url,
        "title": article.title,
        "article_id": article.article_id,
        "published_at": article.published_at,
        "collected_at": article.created_at,
        "snippet": article.snippet,
        "has_full_text": bool(article.full_text),
        "metadata": article.metadata,
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
            EmergingNameRow(ticker.symbol, ticker.company_name, len(scored_articles.get(ticker.symbol, ())), 0, "fixture mention")
            for ticker in load_watchlist()
            if scored_articles.get(ticker.symbol)
        ),
        article_links_by_ticker=article_links,
    )


def _daily_report_input_from_store(
    store: SQLiteStore,
    run_id: str,
    run_date: str,
    article_fetch_summary: ArticleFetchSummary | None = None,
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
                "recency": _row_recency(run_date, articles_by_id[article_id]),
            }
        )

    event_clusters_by_ticker = _event_clusters_by_ticker_from_store(store, run_id, extractions_by_article)
    article_links = _article_links_from_event_clusters(event_clusters_by_ticker)
    portfolio_rows = tuple(
        _sentiment_row_from_stored(ticker, mentions_by_ticker.get(ticker.symbol, ()), event_clusters_by_ticker.get(ticker.symbol, ()))
        for ticker in load_portfolio()
    )
    watchlist_sentiment_rows = tuple(
        _sentiment_row_from_stored(ticker, mentions_by_ticker.get(ticker.symbol, ()), event_clusters_by_ticker.get(ticker.symbol, ()))
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
            EmergingNameRow(ticker.symbol, ticker.company_name, len(mentions_by_ticker.get(ticker.symbol, ())), 0, "fixture mention")
            for ticker in load_watchlist()
            if mentions_by_ticker.get(ticker.symbol)
        ),
        article_links_by_ticker=article_links,
        event_clusters_by_ticker=event_clusters_by_ticker,
        extraction_summary=_extraction_summary_from_fetch_summary(
            article_fetch_summary,
            extractions_by_article,
            sentiment_basis_counts,
        ),
    )


def _event_clusters_by_ticker_from_store(
    store: SQLiteStore,
    run_id: str,
    extractions_by_article: Mapping[str, Mapping[str, object]] | None = None,
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
            )
            for link in json.loads(str(row.get("supporting_links_json") or "[]"))
        )
        event = EventClusterRow(
            ticker="",
            title=title,
            primary_link=str(row.get("primary_link") or row["canonical_url"]),
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
            supporting_links=supporting_links,
        )
        tickers = match_tickers(title)
        if not tickers:
            tickers = match_tickers(" ".join(link.title for link in supporting_links[:3]))
        for ticker in tickers:
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
                    supporting_links=event.supporting_links,
                )
            )
    return {
        symbol: tuple(sorted(rows, key=lambda event: (-event.publisher_count, -event.article_count, event.title))[:10])
        for symbol, rows in grouped.items()
    }


def _article_links_from_event_clusters(
    event_clusters_by_ticker: Mapping[str, tuple[EventClusterRow, ...]],
) -> dict[str, tuple[ArticleLink, ...]]:
    links_by_ticker: dict[str, tuple[ArticleLink, ...]] = {}
    for ticker in load_tracked_tickers():
        links: list[ArticleLink] = []
        for event in event_clusters_by_ticker.get(ticker.symbol, ()):
            links.append(ArticleLink(event.title, event.primary_link, f"{event.publisher_count} publisher(s)"))
        links_by_ticker[ticker.symbol] = tuple(links)
    return links_by_ticker


def _extraction_summary_from_fetch_summary(
    article_fetch_summary: ArticleFetchSummary | None,
    extractions_by_article: Mapping[str, Mapping[str, object]],
    sentiment_basis_counts: Mapping[str, int],
) -> ExtractionSummary:
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
    )


def _sentiment_row_from_stored(
    ticker: TrackedTicker,
    items: Sequence[dict[str, object]],
    event_clusters: Sequence[EventClusterRow],
) -> PortfolioSentimentRow:
    usable_items = [
        item
        for item in items
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
        weighted_sentiment_score=_weighted_sentiment_score(usable_items),
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
        for ticker in match_tickers(_article_text(article)):
            tracked = ticker_lookup[ticker.symbol]
            store.add_ticker_mention(
                TickerMention(
                    article_id=article_id,
                    ticker=ticker.symbol,
                    confidence=1.0,
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
        return WatchlistForecastRow(ticker.symbol, "uncertain", 0.0, "no fixture articles")
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
        f"fixture sentiment from {len(items)} article(s)",
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
        return WatchlistForecastRow(ticker.symbol, "uncertain", 0.0, "no fixture articles")
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
        f"placeholder forecast from stored fixture sentiment for {len(items)} article(s)",
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
    }


def _write_markdown_report(report: DailyReportContract) -> str:
    output_dir = Path(report.output_dir)
    report_path = output_dir / "daily_report.md"
    lines = [
        f"# Daily News Pipeline Dry Run - {report.report_date}",
        "",
        "Data source: local RSS fixture files by default, plus free live RSS only when explicitly enabled. Watchlist next-close direction uses placeholder fixture sentiment logic.",
        "",
        "Sentiment is deterministic placeholder logic until a stronger model is wired in. This report is not investment advice.",
        "",
        report.daily_summary,
        "",
        "## Article Extraction Summary",
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
        "",
        *_markdown_failure_reason_rows(report.extraction_summary),
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
            "Placeholder forecast logic: direction is derived from fixture sentiment only. This is not a live model prediction.",
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
        lines.append("| none | none | 0 | 0 | no emerging watchlist names in fixture data |")
    lines.extend(["", "## Top Event Clusters By Recency And Source Diversity", ""])
    event_written = False
    for ticker, clusters in sorted(report.event_clusters_by_ticker.items()):
        visible_clusters = clusters[:5]
        if not visible_clusters:
            continue
        event_written = True
        lines.append(f"### {ticker}")
        lines.extend(
            [
                "| Event | Bucket | Weighted Sentiment | Extraction Basis | First Seen | Latest Seen | Articles | Publishers | Sources |",
                "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for cluster in visible_clusters:
            lines.append(
                f"| [{_escape_markdown(cluster.title)}]({cluster.primary_link}) | {cluster.recency_bucket} | {_optional_markdown_score(cluster.weighted_cluster_sentiment)} | {cluster.extraction_basis} | {cluster.first_seen_at or ''} | {cluster.latest_seen_at or ''} | {cluster.article_count} | {cluster.publisher_count} | {cluster.source_count} |"
            )
        lines.append("")
    if not event_written:
        lines.append("No event clusters matched configured tickers.")
    lines.extend(["", "## Article Links Grouped By Ticker And Event Cluster", ""])
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
        lines.append("No fixture article links matched configured tickers.")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(report_path)


def _escape_markdown(value: str | None) -> str:
    return (value or "").replace("|", "\\|")


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
