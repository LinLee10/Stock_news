"""Core value objects for the canonical news pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SENTIMENT_BASES = {"full_text", "snippet", "title"}
RUN_STATUSES = {"started", "completed", "failed"}


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _require_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _require_probability(value: float, field_name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return value


@dataclass(frozen=True)
class Article:
    canonical_url: str
    title: str
    article_id: str | None = None
    published_at: str | None = None
    full_text: str | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_url", _require_text(self.canonical_url, "canonical_url"))
        object.__setattr__(self, "title", _require_text(self.title, "title"))


@dataclass(frozen=True)
class ArticleSource:
    provider: str
    url: str
    article_id: str | None = None
    provider_article_id: str | None = None
    title: str | None = None
    snippet: str | None = None
    published_at: str | None = None
    source_name: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text(self.provider, "provider"))
        object.__setattr__(self, "url", _require_text(self.url, "url"))


@dataclass(frozen=True)
class TickerMention:
    article_id: str
    ticker: str
    confidence: float
    company_name: str | None = None
    basis: str = "title"

    def __post_init__(self) -> None:
        object.__setattr__(self, "article_id", _require_text(self.article_id, "article_id"))
        object.__setattr__(self, "ticker", _require_text(self.ticker.upper(), "ticker"))
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "confidence"))
        if self.basis not in SENTIMENT_BASES:
            raise ValueError(f"basis must be one of {sorted(SENTIMENT_BASES)}")


@dataclass(frozen=True)
class SentimentResult:
    article_id: str
    score: float
    label: str
    confidence: float
    basis: str
    model: str = "deterministic_stub"
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "article_id", _require_text(self.article_id, "article_id"))
        object.__setattr__(self, "label", _require_text(self.label, "label"))
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("score must be between -1.0 and 1.0")
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "confidence"))
        if self.basis not in SENTIMENT_BASES:
            raise ValueError(f"basis must be one of {sorted(SENTIMENT_BASES)}")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str = "started"
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    articles_seen: int = 0
    articles_stored: int = 0
    duplicates: int = 0
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        if self.status not in RUN_STATUSES:
            raise ValueError(f"status must be one of {sorted(RUN_STATUSES)}")
        for field_name in ("articles_seen", "articles_stored", "duplicates"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True)
class ProviderUsage:
    provider: str
    operation: str
    status: str
    quota_cost: int = 1
    article_count: int = 0
    latency_ms: int = 0
    error_class: str | None = None
    recorded_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text(self.provider, "provider"))
        object.__setattr__(self, "operation", _require_text(self.operation, "operation"))
        object.__setattr__(self, "status", _require_text(self.status, "status"))
        for field_name in ("quota_cost", "article_count", "latency_ms"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
