"""GNews adapter using mocked JSON responses only."""

from __future__ import annotations

from typing import Any, Mapping

from .base import NewsSource, SourceCandidate, candidate_to_article
from news_pipeline.models import Article


class GNewsSource(NewsSource):
    provider_name = "gnews"

    def __init__(self, payload: Mapping[str, Any] | None = None):
        self.payload = dict(payload or {})

    def discover(self, symbols=None) -> list[SourceCandidate]:
        items = self.payload.get("articles") or []
        candidates = []
        for item in items:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if not title or not url:
                continue
            source = item.get("source") or {}
            candidates.append(
                SourceCandidate(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    snippet=item.get("description"),
                    published_at=item.get("publishedAt"),
                    source_name=source.get("name") if isinstance(source, dict) else None,
                    raw_metadata={"content": item.get("content")},
                )
            )
        return candidates

    def articles(self) -> list[Article]:
        articles = []
        for candidate in self.discover(()):
            article = candidate_to_article(candidate)
            content = candidate.raw_metadata.get("content")
            if content:
                article = Article(
                    canonical_url=article.canonical_url,
                    title=article.title,
                    published_at=article.published_at,
                    full_text=content,
                    snippet=article.snippet,
                    metadata=article.metadata,
                )
            articles.append(article)
        return articles
