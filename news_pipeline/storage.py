"""SQLite storage for the canonical news pipeline."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Article, ArticleSource, ProviderUsage, RunResult


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

            CREATE TABLE IF NOT EXISTS article_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                article_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                company_name TEXT,
                confidence REAL NOT NULL,
                basis TEXT NOT NULL,
                UNIQUE(article_id, ticker, basis)
            );

            CREATE TABLE IF NOT EXISTS sentiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """
        )
        self.connection.commit()

    def record_run(self, run: RunResult) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, status, started_at, finished_at, articles_seen,
                articles_stored, duplicates, errors_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
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

    def add_article_source(self, source: ArticleSource) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO article_sources (
                article_id, provider, url, provider_article_id, title, snippet,
                published_at, source_name, raw_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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

    def record_provider_usage(self, usage: ProviderUsage) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO provider_usage (
                provider, operation, status, quota_cost, article_count,
                latency_ms, error_class, recorded_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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

    def list_provider_usage(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM provider_usage ORDER BY id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def table_names(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {str(row["name"]) for row in rows}

    @staticmethod
    def article_id_for_url(canonical_url: str) -> str:
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        return f"art_{digest}"


def initialize_database(database_path: str | Path) -> SQLiteStore:
    store = SQLiteStore(database_path)
    store.initialize_schema()
    return store
