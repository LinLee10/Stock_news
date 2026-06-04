"""Centralized free RSS source-family configuration."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

from news_pipeline.tickers import load_tracked_tickers


@dataclass(frozen=True)
class RssFeedConfig:
    feed_id: str
    url: str
    enabled: bool = True


@dataclass(frozen=True)
class RssSourceFamilyConfig:
    name: str
    display_name: str
    feeds: tuple[RssFeedConfig, ...] = ()
    ticker_search: bool = False
    enabled: bool = True


DEFAULT_MAX_ARTICLES_PER_SOURCE = 80
DEFAULT_MAX_ARTICLES_PER_TICKER = 25
DEFAULT_MAX_TOTAL_LIVE_ARTICLES = 300


RSS_SOURCE_FAMILIES: tuple[RssSourceFamilyConfig, ...] = (
    RssSourceFamilyConfig(
        name="google_news_rss_search",
        display_name="Google News RSS Search",
        ticker_search=True,
    ),
    RssSourceFamilyConfig(
        name="yahoo_finance_rss",
        display_name="Yahoo Finance RSS",
        feeds=(
            RssFeedConfig(
                feed_id="yahoo_finance_news",
                url="https://finance.yahoo.com/news/rssindex",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="cnbc_rss",
        display_name="CNBC RSS",
        feeds=(
            RssFeedConfig(
                feed_id="cnbc_top_news",
                url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="marketwatch_rss",
        display_name="MarketWatch RSS",
        feeds=(
            RssFeedConfig(
                feed_id="marketwatch_top_stories",
                url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
            ),
        ),
    ),
)


def enabled_source_families() -> tuple[RssSourceFamilyConfig, ...]:
    return tuple(family for family in RSS_SOURCE_FAMILIES if family.enabled)


def configured_feed_count() -> int:
    return sum(len(_feeds_for_family(family)) for family in enabled_source_families())


def feed_urls_for_family(family: RssSourceFamilyConfig) -> tuple[RssFeedConfig, ...]:
    return _feeds_for_family(family)


def _feeds_for_family(family: RssSourceFamilyConfig) -> tuple[RssFeedConfig, ...]:
    if family.ticker_search:
        return tuple(
            RssFeedConfig(
                feed_id=f"google_news_{ticker.symbol.lower()}",
                url=(
                    "https://news.google.com/rss/search?"
                    f"q={quote_plus(ticker.symbol + ' stock news')}&hl=en-US&gl=US&ceid=US:en"
                ),
            )
            for ticker in load_tracked_tickers()
        )
    return tuple(feed for feed in family.feeds if feed.enabled)
