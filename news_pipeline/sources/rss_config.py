"""Centralized free RSS source-family configuration."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

from news_pipeline.tickers import load_tracked_tickers

from .source_registry import (
    COMPANY_IR,
    DIRECT_NEWS_PUBLISHER,
    GOOGLE_NEWS_BACKSTOP,
    MARKET_DATA_OR_ANALYSIS,
    PRESS_RELEASE_WIRE,
    REGULATORY_OFFICIAL,
)

# Backward-compatible names used by the RSS collector.
GOOGLE_NEWS_DISCOVERY = GOOGLE_NEWS_BACKSTOP
SEC_OR_FILING = REGULATORY_OFFICIAL


@dataclass(frozen=True)
class RssFeedConfig:
    feed_id: str
    url: str
    publisher_name: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class RssSourceFamilyConfig:
    name: str
    display_name: str
    category: str
    feeds: tuple[RssFeedConfig, ...] = ()
    ticker_search: bool = False
    enabled: bool = True


DEFAULT_MAX_ARTICLES_PER_SOURCE = 120
DEFAULT_MAX_ARTICLES_PER_TICKER = 40
DEFAULT_MAX_TOTAL_LIVE_ARTICLES = 400
DEFAULT_MAX_GOOGLE_NEWS_SHARE = 0.75


RSS_SOURCE_FAMILIES: tuple[RssSourceFamilyConfig, ...] = (
    RssSourceFamilyConfig(
        name="google_news_rss_search",
        display_name="Google News RSS Search",
        category=GOOGLE_NEWS_DISCOVERY,
        ticker_search=True,
    ),
    RssSourceFamilyConfig(
        name="yahoo_finance_rss",
        display_name="Yahoo Finance RSS",
        category=MARKET_DATA_OR_ANALYSIS,
        feeds=(
            RssFeedConfig(
                feed_id="yahoo_finance_news",
                url="https://finance.yahoo.com/news/rssindex",
                publisher_name="Yahoo Finance",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="cnbc_rss",
        display_name="CNBC RSS",
        category=DIRECT_NEWS_PUBLISHER,
        feeds=(
            RssFeedConfig(
                feed_id="cnbc_top_news",
                url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
                publisher_name="CNBC",
            ),
            RssFeedConfig(
                feed_id="cnbc_technology",
                url="https://www.cnbc.com/id/19854910/device/rss/rss.html",
                publisher_name="CNBC",
            ),
            RssFeedConfig(
                feed_id="cnbc_business",
                url="https://www.cnbc.com/id/10001147/device/rss/rss.html",
                publisher_name="CNBC",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="marketwatch_rss",
        display_name="MarketWatch RSS",
        category=DIRECT_NEWS_PUBLISHER,
        feeds=(
            RssFeedConfig(
                feed_id="marketwatch_top_stories",
                url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
                publisher_name="MarketWatch",
            ),
            RssFeedConfig(
                feed_id="marketwatch_marketpulse",
                url="https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
                publisher_name="MarketWatch",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="pr_newswire_rss",
        display_name="PR Newswire RSS",
        category=PRESS_RELEASE_WIRE,
        feeds=(
            RssFeedConfig(
                feed_id="pr_newswire_technology",
                url="https://www.prnewswire.com/rss/technology-latest-news/technology-latest-news-list.rss",
                publisher_name="PR Newswire",
            ),
            RssFeedConfig(
                feed_id="pr_newswire_financial_services",
                url=(
                    "https://www.prnewswire.com/rss/financial-services-latest-news/"
                    "financial-services-latest-news-list.rss"
                ),
                publisher_name="PR Newswire",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="business_wire_rss",
        display_name="Business Wire Infrastructure RSS",
        category=PRESS_RELEASE_WIRE,
        feeds=(
            RssFeedConfig(
                feed_id="business_wire_natural_resources",
                url="https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpQWA==",
                publisher_name="Business Wire",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="globenewswire_rss",
        display_name="GlobeNewswire Technology RSS",
        category=PRESS_RELEASE_WIRE,
        feeds=(
            RssFeedConfig(
                feed_id="globenewswire_technology",
                url=(
                    "https://www.globenewswire.com/RssFeed/subjectcode/27-Technology/"
                    "feedTitle/GlobeNewswire%20-%20Technology"
                ),
                publisher_name="GlobeNewswire",
            ),
        ),
    ),
    RssSourceFamilyConfig(
        name="sec_company_filings",
        display_name="SEC Company Filings",
        category=SEC_OR_FILING,
        enabled=False,
    ),
    RssSourceFamilyConfig(
        name="company_ir_rss",
        display_name="Explicit Company Investor Relations RSS",
        category=COMPANY_IR,
        enabled=False,
    ),
)


def enabled_source_families(
    *,
    include_press_release_feeds: bool = True,
) -> tuple[RssSourceFamilyConfig, ...]:
    return tuple(
        family
        for family in RSS_SOURCE_FAMILIES
        if family.enabled
        and (include_press_release_feeds or family.category != PRESS_RELEASE_WIRE)
    )


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
