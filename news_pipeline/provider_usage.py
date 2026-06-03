"""Provider usage accounting without external calls."""

from __future__ import annotations

from .models import ProviderUsage
from .storage import SQLiteStore


class ProviderUsageRecorder:
    """Persist provider usage rows as a first-class pipeline concern."""

    def __init__(self, store: SQLiteStore):
        self.store = store

    def record(
        self,
        provider: str,
        operation: str,
        status: str,
        *,
        quota_cost: int = 1,
        article_count: int = 0,
        latency_ms: int = 0,
        error_class: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        usage = ProviderUsage(
            provider=provider,
            operation=operation,
            status=status,
            quota_cost=quota_cost,
            article_count=article_count,
            latency_ms=latency_ms,
            error_class=error_class,
            metadata=metadata or {},
        )
        return self.store.record_provider_usage(usage)
