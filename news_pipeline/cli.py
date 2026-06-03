"""Safe local orchestration CLI for the canonical news pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from .dedup import cluster_articles
from .models import Article
from .provider_registry import iter_provider_configs
from .provider_validation import ProviderChecker, validate_provider
from .reporting import (
    DailyReportInput,
    MentionLeaderRow,
    MostMentionedRow,
    PortfolioSentimentRow,
    WatchlistForecastRow,
    build_daily_report,
)
from .sentiment import analyze_sentiment
from .storage import initialize_database


class FakeProvider(Protocol):
    def articles(self) -> list[Article]:
        """Return fake local articles without network access."""


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
        articles = _collect_fake_articles(fake_providers, enable_paid=args.enable_paid_apis)
        payload = {**context, "article_count": len(articles), "articles": [_article_to_dict(article) for article in articles]}
        _write_json(output_dir / "collected_articles.json", payload)
        _print_json({**context, "article_count": len(articles), "output": str(output_dir / "collected_articles.json")})
        return 0

    if args.command == "extract":
        articles = _collect_fake_articles(fake_providers, enable_paid=args.enable_paid_apis)
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
        articles = _collect_fake_articles(fake_providers, enable_paid=args.enable_paid_apis)
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
        articles = _collect_fake_articles(fake_providers, enable_paid=args.enable_paid_apis)
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
        report = build_daily_report(_fake_report_input(args.run_date), artifacts_dir=args.artifacts_dir)
        payload = {**context, "report": _dataclass_to_dict(report)}
        _write_json(output_dir / "report_contract.json", payload)
        _print_json({**context, "output_dir": report.output_dir, "csv_attachments": list(report.csv_attachments)})
        return 0

    if args.command == "dry-run-daily":
        _run_command("validate-providers", args, output_dir, fake_providers, provider_checker, environ)
        articles = _collect_fake_articles(fake_providers, enable_paid=args.enable_paid_apis)
        _write_json(output_dir / "collected_articles.json", {**context, "articles": [_article_to_dict(article) for article in articles]})
        _write_json(output_dir / "dedupe_clusters.json", {**context, "cluster_count": len(cluster_articles(articles))})
        _write_json(output_dir / "sentiment_scores.json", {**context, "score_count": len(articles)})
        report = build_daily_report(_fake_report_input(args.run_date), artifacts_dir=args.artifacts_dir)
        payload = {
            **context,
            "status": "dry_run_complete",
            "email_sending": "disabled",
            "paid_apis": "enabled" if args.enable_paid_apis else "disabled",
            "output_dir": report.output_dir,
        }
        _write_json(output_dir / "dry_run_daily.json", payload)
        _print_json(payload)
        return 0

    parser.error("unknown command")
    return 2


def _add_safe_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-paid-apis", action="store_true", default=False)
    parser.add_argument("--enable-email-send", action="store_true", default=False)


def _safe_context(args: argparse.Namespace) -> dict[str, object]:
    return {
        "command": args.command,
        "run_date": args.run_date,
        "dry_run": args.dry_run,
        "paid_apis_enabled": args.enable_paid_apis,
        "email_send_enabled": args.enable_email_send,
        "output_dir": str(_run_output_dir(args.artifacts_dir, args.run_date)),
    }


def _run_output_dir(artifacts_dir: str | Path, run_date: str) -> Path:
    base = Path(artifacts_dir)
    if base.name != "artifacts":
        base = base / "artifacts"
    return base / "runs" / run_date


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


def _is_paid_provider(provider_name: str) -> bool:
    return provider_name in {"alpha_vantage", "marketaux", "gnews"}


def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _print_json(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _article_to_dict(article: Article) -> dict[str, object]:
    return {
        "canonical_url": article.canonical_url,
        "title": article.title,
        "article_id": article.article_id,
        "published_at": article.published_at,
        "snippet": article.snippet,
        "has_full_text": bool(article.full_text),
        "metadata": article.metadata,
    }


def _dataclass_to_dict(value: object) -> dict[str, object]:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected dataclass value, got {type(value)!r}")


def _fake_report_input(run_date: str) -> DailyReportInput:
    return DailyReportInput(
        report_date=run_date,
        portfolio_sentiment=(
            PortfolioSentimentRow("AAPL", "Apple Inc.", 0.2, 3, "snippet"),
        ),
        watchlist_forecasts=(
            WatchlistForecastRow("NVDA", "up", 0.7, "fake dry-run data"),
        ),
        mention_leaders_7d=(
            MentionLeaderRow("NVDA", 5, 0.3),
        ),
        most_mentioned=(
            MostMentionedRow("NVDA", 5, 1),
            MostMentionedRow("AAPL", 3, 2),
        ),
    )


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
