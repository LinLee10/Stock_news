"""Finnhub company-news adapter for mocked responses only."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from news_pipeline.models import Article

from .base import NewsSource, SourceCandidate, candidate_to_article


class FinnhubNewsSource(NewsSource):
    provider_name = "finnhub_news"

    def __init__(self, payloads: Mapping[str, Sequence[Mapping[str, Any]]] | None = None):
        self.payloads = dict(payloads or {})

    def discover(self, symbols: Iterable[str]) -> list[SourceCandidate]:
        candidates: list[SourceCandidate] = []
        for symbol in symbols:
            for item in self.payloads.get(symbol, ()):
                title = str(item.get("headline") or item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                if not title or not url:
                    continue
                candidates.append(
                    SourceCandidate(
                        provider=self.provider_name,
                        url=url,
                        title=title,
                        snippet=item.get("summary"),
                        published_at=str(item.get("datetime") or "") or None,
                        provider_article_id=str(item.get("id") or "") or None,
                        source_name=item.get("source"),
                        symbols=(symbol.upper(),),
                        raw_metadata={
                            "category": item.get("category"),
                            "related": item.get("related"),
                        },
                    )
                )
        return candidates

    def articles(self, symbols: Iterable[str]) -> list[Article]:
        return [candidate_to_article(candidate) for candidate in self.discover(symbols)]
