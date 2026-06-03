"""Base interfaces for no-network source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from news_pipeline.dedup import canonicalize_url
from news_pipeline.models import Article


@dataclass(frozen=True)
class SourceCandidate:
    provider: str
    url: str
    title: str
    snippet: str | None = None
    published_at: str | None = None
    provider_article_id: str | None = None
    source_name: str | None = None
    symbols: tuple[str, ...] = ()
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class NewsSource(ABC):
    """Interface only; implementations decide how discovery happens."""

    provider_name: str

    @abstractmethod
    def discover(self, symbols: Iterable[str]) -> list[SourceCandidate]:
        """Return candidates without side effects beyond the implementation boundary."""
        raise NotImplementedError


def candidate_to_article(candidate: SourceCandidate) -> Article:
    """Convert a normalized source candidate into a canonical Article."""
    return Article(
        canonical_url=canonicalize_url(candidate.url),
        title=candidate.title,
        published_at=candidate.published_at,
        snippet=candidate.snippet,
        metadata={
            "provider": candidate.provider,
            "provider_article_id": candidate.provider_article_id,
            "source_name": candidate.source_name,
            "symbols": list(candidate.symbols),
            "raw_metadata": candidate.raw_metadata,
        },
    )
