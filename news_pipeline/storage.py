"""SQLite storage for the canonical news pipeline."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

from .models import Article, ArticleSource, ProviderUsage, RunResult, SentimentResult, TickerMention


class SQLiteStore:
    """Small SQLite wrapper with explicit schema creation."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_date TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                articles_seen INTEGER NOT NULL DEFAULT 0,
                articles_stored INTEGER NOT NULL DEFAULT 0,
                duplicates INTEGER NOT NULL DEFAULT 0,
                errors_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS articles (
                article_id TEXT PRIMARY KEY,
                canonical_url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                published_at TEXT,
                full_text TEXT,
                snippet TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_articles (
                run_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                title TEXT NOT NULL,
                published_at TEXT,
                snippet TEXT,
                PRIMARY KEY (run_id, article_id)
            );

            CREATE TABLE IF NOT EXISTS article_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                article_id TEXT,
                provider TEXT NOT NULL,
                url TEXT NOT NULL,
                provider_article_id TEXT,
                title TEXT,
                snippet TEXT,
                published_at TEXT,
                source_name TEXT,
                raw_metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(provider, url)
            );

            CREATE TABLE IF NOT EXISTS ticker_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                article_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                company_name TEXT,
                confidence REAL NOT NULL,
                basis TEXT NOT NULL,
                UNIQUE(run_id, article_id, ticker, basis)
            );

            CREATE TABLE IF NOT EXISTS sentiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                article_id TEXT NOT NULL,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                basis TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS provider_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                provider TEXT NOT NULL,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                quota_cost INTEGER NOT NULL,
                article_count INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                error_class TEXT,
                recorded_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS provider_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                limit_type TEXT NOT NULL,
                reset_window TEXT NOT NULL,
                last_checked_at TEXT NOT NULL,
                last_status TEXT NOT NULL,
                remaining_quota INTEGER,
                quota_truth_source TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                key_state TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dedupe_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                cluster_index INTEGER NOT NULL,
                canonical_article_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                title TEXT NOT NULL,
                primary_link TEXT,
                publisher_count INTEGER NOT NULL DEFAULT 0,
                source_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT,
                latest_seen_at TEXT,
                primary_published_at TEXT,
                recency_bucket TEXT NOT NULL DEFAULT 'unknown',
                tickers_mentioned_json TEXT NOT NULL DEFAULT '[]',
                weighted_cluster_sentiment REAL,
                publisher_names_json TEXT NOT NULL DEFAULT '[]',
                source_providers_json TEXT NOT NULL DEFAULT '[]',
                supporting_links_json TEXT NOT NULL DEFAULT '[]',
                alternate_source_links_json TEXT NOT NULL DEFAULT '[]',
                duplicate_reasons_json TEXT NOT NULL DEFAULT '[]',
                article_ids_json TEXT NOT NULL DEFAULT '[]',
                UNIQUE(run_id, cluster_index)
            );

            CREATE TABLE IF NOT EXISTS article_extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                extraction_status TEXT NOT NULL,
                extraction_basis TEXT NOT NULL,
                error_class TEXT,
                final_url TEXT,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                content_type TEXT,
                content_length INTEGER NOT NULL DEFAULT 0,
                text_hash TEXT,
                extracted_preview TEXT,
                extractor TEXT,
                extraction_method_used TEXT,
                extraction_failure_reason TEXT,
                fetched INTEGER NOT NULL DEFAULT 0,
                tickers_json TEXT NOT NULL DEFAULT '[]',
                UNIQUE(run_id, article_id)
            );

            CREATE TABLE IF NOT EXISTS event_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                published_at TEXT,
                ticker TEXT NOT NULL,
                company TEXT NOT NULL,
                source_provider TEXT NOT NULL,
                source_family TEXT NOT NULL,
                article_type TEXT NOT NULL,
                cluster_id TEXT,
                ticker_match_confidence REAL NOT NULL,
                extraction_basis TEXT NOT NULL,
                extraction_quality_grade TEXT NOT NULL,
                internal_sentiment REAL NOT NULL,
                external_sentiment_provider TEXT,
                external_sentiment REAL,
                event_type TEXT NOT NULL,
                event_summary TEXT NOT NULL,
                run_id TEXT NOT NULL,
                run_date TEXT NOT NULL,
                UNIQUE(run_id, article_id, ticker)
            );
            """
        )
        self._ensure_column("runs", "run_date", "TEXT")
        self._ensure_column("article_sources", "run_id", "TEXT")
        self._ensure_column("ticker_mentions", "run_id", "TEXT")
        self._ensure_column("sentiment_results", "run_id", "TEXT")
        self._ensure_column("provider_usage", "run_id", "TEXT")
        self._ensure_column("dedupe_clusters", "primary_link", "TEXT")
        self._ensure_column("dedupe_clusters", "publisher_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("dedupe_clusters", "source_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("dedupe_clusters", "first_seen_at", "TEXT")
        self._ensure_column("dedupe_clusters", "latest_seen_at", "TEXT")
        self._ensure_column("dedupe_clusters", "primary_published_at", "TEXT")
        self._ensure_column("dedupe_clusters", "recency_bucket", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("dedupe_clusters", "tickers_mentioned_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("dedupe_clusters", "weighted_cluster_sentiment", "REAL")
        self._ensure_column("dedupe_clusters", "publisher_names_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("dedupe_clusters", "source_providers_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("dedupe_clusters", "supporting_links_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("article_extractions", "extraction_method_used", "TEXT")
        self._ensure_column("article_extractions", "extraction_failure_reason", "TEXT")
        self._ensure_column("article_extractions", "tickers_json", "TEXT NOT NULL DEFAULT '[]'")
        self.connection.commit()

    def reset_run(self, run_id: str) -> None:
        """Remove run-scoped rows so reruns replace, not append."""
        legacy_metadata_pattern = f'%"run_id": "{run_id}"%'
        self.connection.execute("DELETE FROM provider_validation WHERE run_id = ?", (run_id,))
        self.connection.execute(
            "DELETE FROM provider_usage WHERE run_id = ? OR metadata_json LIKE ?",
            (run_id, legacy_metadata_pattern),
        )
        self.connection.execute("DELETE FROM article_extractions WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM event_memory WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM dedupe_clusters WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM sentiment_results WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM ticker_mentions WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM article_sources WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM run_articles WHERE run_id = ?", (run_id,))
        self.connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._delete_orphan_articles()
        self.connection.commit()

    def record_run(self, run: RunResult, *, run_date: str | None = None) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, run_date, status, started_at, finished_at, articles_seen,
                articles_stored, duplicates, errors_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run_date,
                run.status,
                run.started_at,
                run.finished_at,
                run.articles_seen,
                run.articles_stored,
                run.duplicates,
                json.dumps(list(run.errors), sort_keys=True),
            ),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_article(self, article: Article) -> str:
        article_id = article.article_id or self.article_id_for_url(article.canonical_url)
        self.connection.execute(
            """
            INSERT INTO articles (
                article_id, canonical_url, title, published_at, full_text, snippet, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_url) DO UPDATE SET
                title=excluded.title,
                published_at=excluded.published_at,
                full_text=excluded.full_text,
                snippet=excluded.snippet
            """,
            (
                article_id,
                article.canonical_url,
                article.title,
                article.published_at,
                article.full_text,
                article.snippet,
                article.created_at,
            ),
        )
        self.connection.commit()
        return article_id

    def add_run_article(self, run_id: str, article: Article, *, store_full_text: bool = False) -> str:
        article_to_store = article
        if not store_full_text and article.full_text:
            article_to_store = Article(
                canonical_url=article.canonical_url,
                title=article.title,
                article_id=article.article_id,
                published_at=article.published_at,
                snippet=article.snippet,
                metadata=article.metadata,
                created_at=article.created_at,
            )
        article_id = self.upsert_article(article_to_store)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO run_articles (
                run_id, article_id, canonical_url, title, published_at, snippet
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                article_id,
                article_to_store.canonical_url,
                article_to_store.title,
                article_to_store.published_at,
                article_to_store.snippet,
            ),
        )
        self.connection.commit()
        return article_id

    def add_article_source(self, source: ArticleSource, *, run_id: str | None = None) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO article_sources (
                run_id, article_id, provider, url, provider_article_id, title, snippet,
                published_at, source_name, raw_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                source.article_id,
                source.provider,
                source.url,
                source.provider_article_id,
                source.title,
                source.snippet,
                source.published_at,
                source.source_name,
                json.dumps(source.raw_metadata, sort_keys=True),
            ),
        )
        self.connection.commit()

    def add_ticker_mention(self, mention: TickerMention, *, run_id: str | None = None) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO ticker_mentions (
                run_id, article_id, ticker, company_name, confidence, basis
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                mention.article_id,
                mention.ticker,
                mention.company_name,
                mention.confidence,
                mention.basis,
            ),
        )
        self.connection.commit()

    def add_sentiment_result(self, sentiment: SentimentResult, *, run_id: str | None = None) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO sentiment_results (
                run_id, article_id, score, label, confidence, basis, model, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sentiment.article_id,
                sentiment.score,
                sentiment.label,
                sentiment.confidence,
                sentiment.basis,
                sentiment.model,
                sentiment.created_at,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_provider_validation(self, run_id: str, result: Mapping[str, object]) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO provider_validation (
                run_id, provider_name, limit_type, reset_window, last_checked_at,
                last_status, remaining_quota, quota_truth_source, dry_run, key_state
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result["provider_name"],
                result["limit_type"],
                result["reset_window"],
                result["last_checked_at"],
                result["last_status"],
                result.get("remaining_quota"),
                result["quota_truth_source"],
                1 if result.get("dry_run") else 0,
                result["key_state"],
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_dedupe_cluster(
        self,
        *,
        run_id: str,
        cluster_index: int,
        canonical_article_id: str,
        canonical_url: str,
        title: str,
        alternate_source_links: Iterable[str] = (),
        duplicate_reasons: Iterable[str] = (),
        article_ids: Iterable[str] = (),
        primary_link: str | None = None,
        publisher_count: int = 0,
        source_count: int = 0,
        publisher_names: Iterable[str] = (),
        source_providers: Iterable[str] = (),
        supporting_links: Iterable[Mapping[str, object]] = (),
        first_seen_at: str | None = None,
        latest_seen_at: str | None = None,
        primary_published_at: str | None = None,
        recency_bucket: str = "unknown",
        tickers_mentioned: Iterable[str] = (),
        weighted_cluster_sentiment: float | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT OR REPLACE INTO dedupe_clusters (
                run_id, cluster_index, canonical_article_id, canonical_url, title,
                primary_link, publisher_count, source_count, first_seen_at, latest_seen_at,
                primary_published_at, recency_bucket, tickers_mentioned_json,
                weighted_cluster_sentiment, publisher_names_json, source_providers_json,
                supporting_links_json, alternate_source_links_json,
                duplicate_reasons_json, article_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                cluster_index,
                canonical_article_id,
                canonical_url,
                title,
                primary_link or canonical_url,
                publisher_count,
                source_count,
                first_seen_at,
                latest_seen_at,
                primary_published_at,
                recency_bucket,
                json.dumps(list(tickers_mentioned), sort_keys=True),
                weighted_cluster_sentiment,
                json.dumps(list(publisher_names), sort_keys=True),
                json.dumps(list(source_providers), sort_keys=True),
                json.dumps(list(supporting_links), sort_keys=True),
                json.dumps(list(alternate_source_links), sort_keys=True),
                json.dumps(list(duplicate_reasons), sort_keys=True),
                json.dumps(list(article_ids), sort_keys=True),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_provider_usage(self, usage: ProviderUsage, *, run_id: str | None = None) -> int:
        usage_run_id = run_id or _metadata_run_id(usage.metadata)
        cursor = self.connection.execute(
            """
            INSERT INTO provider_usage (
                run_id, provider, operation, status, quota_cost, article_count,
                latency_ms, error_class, recorded_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                usage_run_id,
                usage.provider,
                usage.operation,
                usage.status,
                usage.quota_cost,
                usage.article_count,
                usage.latency_ms,
                usage.error_class,
                usage.recorded_at,
                json.dumps(usage.metadata, sort_keys=True),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_article_extraction(
        self,
        *,
        run_id: str,
        article_id: str,
        canonical_url: str,
        extraction_status: str,
        extraction_basis: str,
        error_class: str | None = None,
        final_url: str | None = None,
        latency_ms: int = 0,
        content_type: str | None = None,
        content_length: int = 0,
        text_hash: str | None = None,
        extracted_preview: str | None = None,
        extractor: str | None = None,
        extraction_method_used: str | None = None,
        extraction_failure_reason: str | None = None,
        fetched: bool = False,
        tickers: Iterable[str] = (),
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT OR REPLACE INTO article_extractions (
                run_id, article_id, canonical_url, extraction_status, extraction_basis,
                error_class, final_url, latency_ms, content_type, content_length,
                text_hash, extracted_preview, extractor, extraction_method_used,
                extraction_failure_reason, fetched, tickers_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                article_id,
                canonical_url,
                extraction_status,
                extraction_basis,
                error_class,
                final_url,
                latency_ms,
                content_type,
                content_length,
                text_hash,
                extracted_preview,
                extractor,
                extraction_method_used,
                extraction_failure_reason,
                1 if fetched else 0,
                json.dumps(list(tickers), sort_keys=True),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_event_memory(self, record: Mapping[str, object]) -> int:
        fields = (
            "article_id",
            "canonical_url",
            "published_at",
            "ticker",
            "company",
            "source_provider",
            "source_family",
            "article_type",
            "cluster_id",
            "ticker_match_confidence",
            "extraction_basis",
            "extraction_quality_grade",
            "internal_sentiment",
            "external_sentiment_provider",
            "external_sentiment",
            "event_type",
            "event_summary",
            "run_id",
            "run_date",
        )
        cursor = self.connection.execute(
            f"""
            INSERT OR REPLACE INTO event_memory ({", ".join(fields)})
            VALUES ({", ".join("?" for _field in fields)})
            """,
            tuple(record.get(field) for field in fields),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_provider_usage(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM provider_usage ORDER BY id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_provider_validation(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM provider_validation WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_run_articles(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT ra.*, a.created_at
            FROM run_articles ra
            LEFT JOIN articles a ON a.article_id = ra.article_id
            WHERE ra.run_id = ?
            ORDER BY ra.article_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_articles(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM articles ORDER BY article_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_article_sources(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT * FROM article_sources
            WHERE run_id = ? OR run_id IS NULL
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_ticker_mentions(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM ticker_mentions WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_sentiment_results(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM sentiment_results WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_article_extractions(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM article_extractions WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_dedupe_clusters(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM dedupe_clusters WHERE run_id = ? ORDER BY cluster_index ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_event_memory(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM event_memory WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def table_names(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        if column_name in {str(row["name"]) for row in rows}:
            return
        self.connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _delete_orphan_articles(self) -> None:
        self.connection.execute(
            """
            DELETE FROM articles
            WHERE article_id NOT IN (
                SELECT article_id FROM run_articles
                UNION
                SELECT article_id FROM article_sources WHERE article_id IS NOT NULL
                UNION
                SELECT article_id FROM ticker_mentions
                UNION
                SELECT article_id FROM sentiment_results
                UNION
                SELECT article_id FROM article_extractions
            )
            """
        )

    @staticmethod
    def article_id_for_url(canonical_url: str) -> str:
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        return f"art_{digest}"


def initialize_database(database_path: str | Path) -> SQLiteStore:
    store = SQLiteStore(database_path)
    store.initialize_schema()
    return store


def _metadata_run_id(metadata: Mapping[str, object]) -> str | None:
    value = metadata.get("run_id")
    return str(value) if value else None
