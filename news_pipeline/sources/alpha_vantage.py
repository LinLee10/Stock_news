"""Alpha Vantage adapter using mocked JSON responses only."""

from __future__ import annotations

from typing import Any, Mapping

from .base import NewsSource, SourceCandidate, candidate_to_article
from news_pipeline.models import Article


class AlphaVantageSource(NewsSource):
    provider_name = "alpha_vantage"

    def __init__(self, payload: Mapping[str, Any] | None = None):
        self.payload = dict(payload or {})

    def discover(self, symbols=None) -> list[SourceCandidate]:
        feed = self.payload.get("feed") or self.payload.get("items") or []
        candidates = []
        for item in feed:
            title = (item.get("title") or item.get("headline") or "").strip()
            url = (item.get("url") or "").strip()
            if not title or not url:
                continue
            ticker_sentiment = item.get("ticker_sentiment") or []
            symbols_tuple = tuple(
                entry.get("ticker", "").upper()
                for entry in ticker_sentiment
                if entry.get("ticker")
            )
            candidates.append(
                SourceCandidate(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    snippet=item.get("summary"),
                    published_at=item.get("time_published") or item.get("published_at"),
                    source_name=item.get("source"),
                    symbols=symbols_tuple,
                    raw_metadata={
                        "overall_sentiment_score": item.get("overall_sentiment_score"),
                        "overall_sentiment_label": item.get("overall_sentiment_label"),
                        "ticker_sentiment": ticker_sentiment,
                        "calendar": item.get("calendar") or item.get("earnings_calendar"),
                    },
                )
            )
        return candidates

    def articles(self) -> list[Article]:
        return [candidate_to_article(candidate) for candidate in self.discover(())]
