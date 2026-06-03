"""Marketaux adapter using mocked JSON responses only."""

from __future__ import annotations

from typing import Any, Mapping

from .base import NewsSource, SourceCandidate, candidate_to_article
from news_pipeline.models import Article


class MarketauxSource(NewsSource):
    provider_name = "marketaux"

    def __init__(self, payload: Mapping[str, Any] | None = None):
        self.payload = dict(payload or {})

    def discover(self, symbols=None) -> list[SourceCandidate]:
        items = self.payload.get("data") or self.payload.get("articles") or []
        candidates = []
        for item in items:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if not title or not url:
                continue
            entities = item.get("entities") or []
            candidates.append(
                SourceCandidate(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    snippet=item.get("description") or item.get("snippet"),
                    published_at=item.get("published_at"),
                    provider_article_id=item.get("uuid") or item.get("id"),
                    source_name=(item.get("source") or {}).get("name") if isinstance(item.get("source"), dict) else item.get("source"),
                    symbols=tuple(entity.get("symbol", "").upper() for entity in entities if entity.get("symbol")),
                    raw_metadata={
                        "entity_sentiment": [
                            {
                                "symbol": entity.get("symbol"),
                                "sentiment_score": entity.get("sentiment_score"),
                                "sentiment": entity.get("sentiment"),
                            }
                            for entity in entities
                        ],
                        "similar_articles": item.get("similar") or item.get("similar_articles") or [],
                    },
                )
            )
        return candidates

    def articles(self) -> list[Article]:
        return [candidate_to_article(candidate) for candidate in self.discover(())]
