"""RSS source adapter using injected local responses only."""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

from .base import NewsSource, SourceCandidate, candidate_to_article
from news_pipeline.models import Article


class RssSource(NewsSource):
    provider_name = "rss"

    def __init__(self, feed_xml: str | None = None, provider_name: str = "rss"):
        self.feed_xml = feed_xml
        self.provider_name = provider_name

    def discover(self, symbols=None) -> list[SourceCandidate]:
        if not self.feed_xml:
            return []

        root = ET.fromstring(self.feed_xml)
        candidates = []
        channel_source = root.findtext("./channel/title")
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            candidates.append(
                SourceCandidate(
                    provider=self.provider_name,
                    url=link,
                    title=title,
                    snippet=(item.findtext("description") or item.findtext("summary") or "").strip() or None,
                    published_at=_parse_rss_date(item.findtext("pubDate") or item.findtext("published")),
                    source_name=(item.findtext("source") or channel_source or "").strip() or None,
                )
            )
        return candidates

    def articles(self) -> list[Article]:
        return [candidate_to_article(candidate) for candidate in self.discover(())]


def _parse_rss_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return value.strip() or None
