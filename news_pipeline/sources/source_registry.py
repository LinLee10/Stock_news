"""Declarative registry for source-aware news acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


REGULATORY_OFFICIAL = "regulatory_official"
COMPANY_IR = "company_ir"
PRESS_RELEASE_WIRE = "press_release_wire"
DIRECT_NEWS_PUBLISHER = "direct_news_publisher"
MARKET_DATA_OR_ANALYSIS = "market_data_or_analysis"
EXTERNAL_MARKET_NEWS_API = "external_market_news_api"
CONTEXT_NEWS_API = "context_news_api"
EXTERNAL_GENERAL_NEWS_API = "external_general_news_api"
# Backward-compatible import alias. New code and diagnostics use external API terms.
PAID_NEWS_API = EXTERNAL_MARKET_NEWS_API
GOOGLE_NEWS_BACKSTOP = "google_news_backstop"

SOURCE_FAMILY_ORDER = (
    REGULATORY_OFFICIAL,
    COMPANY_IR,
    PRESS_RELEASE_WIRE,
    DIRECT_NEWS_PUBLISHER,
    MARKET_DATA_OR_ANALYSIS,
    EXTERNAL_MARKET_NEWS_API,
    CONTEXT_NEWS_API,
    EXTERNAL_GENERAL_NEWS_API,
    GOOGLE_NEWS_BACKSTOP,
)


@dataclass(frozen=True)
class SourceProfile:
    source_id: str
    source_family: str
    publisher_name: str
    domain: str
    source_quality_tier: int
    enabled_by_default: bool
    paid_required: bool = False
    api_key_env_var: str | None = None
    discovery_methods: tuple[str, ...] = ()
    feed_urls: tuple[str, ...] = ()
    sitemap_urls: tuple[str, ...] = ()
    search_url_templates: tuple[str, ...] = ()
    company_ir_url_templates: tuple[str, ...] = ()
    robots_check_required: bool = False
    default_rate_limit_seconds: float = 0.0
    fetch_allowed: bool = True
    extract_allowed: bool = True
    paywall_likely: bool = False
    javascript_required: bool = False
    canonical_url_strategy: str = "response_or_feed_url"
    ticker_query_templates: tuple[str, ...] = ()
    source_priority: int = 50
    extraction_priority: int = 50
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "publisher_name": self.publisher_name,
            "domain": self.domain,
            "source_quality_tier": self.source_quality_tier,
            "enabled_by_default": self.enabled_by_default,
            "paid_required": self.paid_required,
            "api_key_env_var": self.api_key_env_var,
            "discovery_methods": list(self.discovery_methods),
            "feed_urls": list(self.feed_urls),
            "sitemap_urls": list(self.sitemap_urls),
            "search_url_templates": list(self.search_url_templates),
            "company_ir_url_templates": list(self.company_ir_url_templates),
            "robots_check_required": self.robots_check_required,
            "default_rate_limit_seconds": self.default_rate_limit_seconds,
            "fetch_allowed": self.fetch_allowed,
            "extract_allowed": self.extract_allowed,
            "paywall_likely": self.paywall_likely,
            "javascript_required": self.javascript_required,
            "canonical_url_strategy": self.canonical_url_strategy,
            "ticker_query_templates": list(self.ticker_query_templates),
            "source_priority": self.source_priority,
            "extraction_priority": self.extraction_priority,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CompanyIrProfile:
    ticker: str
    company_name: str
    ir_news_url: str | None = None
    ir_rss_url: str | None = None
    ir_sitemap_url: str | None = None
    press_release_url: str | None = None


SOURCE_PROFILES: tuple[SourceProfile, ...] = (
    SourceProfile(
        source_id="sec_edgar",
        source_family=REGULATORY_OFFICIAL,
        publisher_name="SEC EDGAR",
        domain="sec.gov",
        source_quality_tier=1,
        enabled_by_default=True,
        discovery_methods=("api",),
        default_rate_limit_seconds=0.12,
        extract_allowed=False,
        canonical_url_strategy="sec_accession_url",
        source_priority=100,
        extraction_priority=40,
        notes="Recent submission metadata only; deep filing parsing is out of scope.",
    ),
    SourceProfile(
        source_id="company_ir_configured",
        source_family=COMPANY_IR,
        publisher_name="Configured company investor relations",
        domain="",
        source_quality_tier=1,
        enabled_by_default=True,
        discovery_methods=("rss", "atom", "sitemap"),
        robots_check_required=True,
        source_priority=95,
        extraction_priority=95,
        notes="Only explicit per-ticker profiles are collected.",
    ),
    SourceProfile(
        source_id="pr_newswire",
        source_family=PRESS_RELEASE_WIRE,
        publisher_name="PR Newswire",
        domain="prnewswire.com",
        source_quality_tier=2,
        enabled_by_default=True,
        discovery_methods=("rss",),
        feed_urls=(
            "https://www.prnewswire.com/rss/technology-latest-news/technology-latest-news-list.rss",
            "https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss",
        ),
        source_priority=75,
        extraction_priority=70,
        notes="Issuer-originated material; sentiment weight is capped unless corroborated.",
    ),
    SourceProfile(
        source_id="business_wire",
        source_family=PRESS_RELEASE_WIRE,
        publisher_name="Business Wire",
        domain="businesswire.com",
        source_quality_tier=2,
        enabled_by_default=True,
        discovery_methods=("rss",),
        feed_urls=("https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpQWA==",),
        source_priority=75,
        extraction_priority=70,
        notes="Natural resources and infrastructure feed relevant to energy names.",
    ),
    SourceProfile(
        source_id="globenewswire",
        source_family=PRESS_RELEASE_WIRE,
        publisher_name="GlobeNewswire",
        domain="globenewswire.com",
        source_quality_tier=2,
        enabled_by_default=True,
        discovery_methods=("rss",),
        feed_urls=(
            "https://www.globenewswire.com/RssFeed/subjectcode/27-Technology/"
            "feedTitle/GlobeNewswire%20-%20Technology",
        ),
        source_priority=75,
        extraction_priority=70,
        notes="Issuer-originated material; sentiment weight is capped unless corroborated.",
    ),
    SourceProfile(
        source_id="cnbc",
        source_family=DIRECT_NEWS_PUBLISHER,
        publisher_name="CNBC",
        domain="cnbc.com",
        source_quality_tier=1,
        enabled_by_default=True,
        discovery_methods=("rss",),
        feed_urls=(
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        ),
        source_priority=90,
        extraction_priority=90,
    ),
    SourceProfile(
        source_id="marketwatch",
        source_family=DIRECT_NEWS_PUBLISHER,
        publisher_name="MarketWatch",
        domain="marketwatch.com",
        source_quality_tier=1,
        enabled_by_default=True,
        discovery_methods=("rss",),
        feed_urls=(
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
        ),
        paywall_likely=True,
        source_priority=90,
        extraction_priority=75,
    ),
    SourceProfile(
        source_id="yahoo_finance",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="Yahoo Finance",
        domain="finance.yahoo.com",
        source_quality_tier=2,
        enabled_by_default=True,
        discovery_methods=("rss",),
        feed_urls=("https://finance.yahoo.com/news/rssindex",),
        source_priority=78,
        extraction_priority=78,
    ),
    SourceProfile(
        source_id="reuters",
        source_family=DIRECT_NEWS_PUBLISHER,
        publisher_name="Reuters",
        domain="reuters.com",
        source_quality_tier=1,
        enabled_by_default=False,
        discovery_methods=("profile_only",),
        paywall_likely=False,
        source_priority=98,
        extraction_priority=95,
        notes="No stable public feed is configured; profile supports future licensed or verified surfaces.",
    ),
    SourceProfile(
        source_id="investors_business_daily",
        source_family=DIRECT_NEWS_PUBLISHER,
        publisher_name="Investor's Business Daily",
        domain="investors.com",
        source_quality_tier=1,
        enabled_by_default=False,
        discovery_methods=("news_sitemap", "public_search"),
        sitemap_urls=("https://www.investors.com/news-sitemap.xml",),
        paywall_likely=True,
        robots_check_required=True,
        source_priority=86,
        extraction_priority=65,
    ),
    SourceProfile(
        source_id="barrons",
        source_family=DIRECT_NEWS_PUBLISHER,
        publisher_name="Barron's",
        domain="barrons.com",
        source_quality_tier=1,
        enabled_by_default=False,
        discovery_methods=("news_sitemap",),
        paywall_likely=True,
        robots_check_required=True,
        source_priority=88,
        extraction_priority=55,
    ),
    SourceProfile(
        source_id="marketbeat",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="MarketBeat",
        domain="marketbeat.com",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("public_search",),
        search_url_templates=("https://www.marketbeat.com/stocks/{exchange}/{ticker}/news/",),
        robots_check_required=True,
        source_priority=58,
        extraction_priority=55,
    ),
    SourceProfile(
        source_id="investing_com",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="Investing.com",
        domain="investing.com",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("public_search",),
        robots_check_required=True,
        paywall_likely=True,
        source_priority=58,
        extraction_priority=50,
    ),
    SourceProfile(
        source_id="barchart",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="Barchart",
        domain="barchart.com",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("public_search",),
        robots_check_required=True,
        source_priority=55,
        extraction_priority=50,
    ),
    SourceProfile(
        source_id="seeking_alpha",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="Seeking Alpha",
        domain="seekingalpha.com",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("public_search",),
        robots_check_required=True,
        paywall_likely=True,
        source_priority=52,
        extraction_priority=40,
    ),
    SourceProfile(
        source_id="motley_fool",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="The Motley Fool",
        domain="fool.com",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("news_sitemap",),
        robots_check_required=True,
        source_priority=48,
        extraction_priority=45,
    ),
    SourceProfile(
        source_id="zacks",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="Zacks",
        domain="zacks.com",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("public_search",),
        robots_check_required=True,
        paywall_likely=True,
        source_priority=50,
        extraction_priority=42,
    ),
    SourceProfile(
        source_id="stockstory",
        source_family=MARKET_DATA_OR_ANALYSIS,
        publisher_name="StockStory",
        domain="stockstory.org",
        source_quality_tier=2,
        enabled_by_default=False,
        discovery_methods=("public_search",),
        robots_check_required=True,
        source_priority=52,
        extraction_priority=50,
    ),
    SourceProfile(
        source_id="alpha_vantage_news",
        source_family=EXTERNAL_MARKET_NEWS_API,
        publisher_name="Alpha Vantage News Sentiment",
        domain="alphavantage.co",
        source_quality_tier=2,
        enabled_by_default=False,
        paid_required=False,
        api_key_env_var="ALPHA_VANTAGE_KEY",
        discovery_methods=("api",),
        default_rate_limit_seconds=1.1,
        source_priority=68,
        extraction_priority=60,
        notes="External benchmark sentiment source; never treated as ground truth.",
    ),
    SourceProfile(
        source_id="marketaux",
        source_family=EXTERNAL_MARKET_NEWS_API,
        publisher_name="Marketaux",
        domain="marketaux.com",
        source_quality_tier=2,
        enabled_by_default=False,
        paid_required=False,
        api_key_env_var="MARKETAUX_API_KEY",
        discovery_methods=("api",),
        source_priority=70,
        extraction_priority=70,
        notes="Free-tier market news API; disabled by default and quota limited.",
    ),
    SourceProfile(
        source_id="finnhub_news",
        source_family=EXTERNAL_MARKET_NEWS_API,
        publisher_name="Finnhub",
        domain="finnhub.io",
        source_quality_tier=2,
        enabled_by_default=False,
        paid_required=False,
        api_key_env_var="FINNHUB_KEY",
        discovery_methods=("api",),
        source_priority=70,
        extraction_priority=70,
        notes="Free-tier company news endpoint only; trading endpoints are not used.",
    ),
    SourceProfile(
        source_id="nyt",
        source_family=CONTEXT_NEWS_API,
        publisher_name="The New York Times",
        domain="nytimes.com",
        source_quality_tier=1,
        enabled_by_default=False,
        paid_required=False,
        api_key_env_var="NYT_API_KEY",
        discovery_methods=("api",),
        source_priority=72,
        extraction_priority=65,
        notes="Free-tier Article Search context; not assumed to contain full text.",
    ),
    SourceProfile(
        source_id="gnews",
        source_family=EXTERNAL_GENERAL_NEWS_API,
        publisher_name="GNews",
        domain="gnews.io",
        source_quality_tier=2,
        enabled_by_default=False,
        paid_required=False,
        api_key_env_var="GNEWS_KEY",
        discovery_methods=("api",),
        default_rate_limit_seconds=1.1,
        source_priority=62,
        extraction_priority=60,
        notes="Free-tier general news backup; disabled by default and quota limited.",
    ),
    SourceProfile(
        source_id="newsapi",
        source_family=EXTERNAL_GENERAL_NEWS_API,
        publisher_name="NewsAPI",
        domain="newsapi.org",
        source_quality_tier=2,
        enabled_by_default=False,
        paid_required=False,
        api_key_env_var="NEWSAPI_KEY",
        discovery_methods=("api",),
        source_priority=58,
        extraction_priority=55,
        notes="Free-tier broad news backup; licensing and quota diagnostics apply.",
    ),
    SourceProfile(
        source_id="google_news_rss_search",
        source_family=GOOGLE_NEWS_BACKSTOP,
        publisher_name="Google News",
        domain="news.google.com",
        source_quality_tier=2,
        enabled_by_default=True,
        discovery_methods=("ticker_rss_search",),
        ticker_query_templates=(
            "https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en",
        ),
        fetch_allowed=True,
        extract_allowed=False,
        canonical_url_strategy="resolve_wrapper_then_publisher",
        source_priority=20,
        extraction_priority=10,
        notes="Recall backstop; wrapper URLs must not dominate extraction or sentiment.",
    ),
)


# IR collection is explicit by design. Add only verified issuer-owned profiles here.
COMPANY_IR_PROFILES: Mapping[str, CompanyIrProfile] = {}


def load_source_profiles() -> tuple[SourceProfile, ...]:
    return SOURCE_PROFILES


def source_profile_by_id(source_id: str) -> SourceProfile | None:
    return next((profile for profile in SOURCE_PROFILES if profile.source_id == source_id), None)


def enabled_source_profiles(
    *,
    include_press_release_feeds: bool = True,
    include_sec_feeds: bool = True,
) -> tuple[SourceProfile, ...]:
    return tuple(
        profile
        for profile in SOURCE_PROFILES
        if profile.enabled_by_default
        and (include_press_release_feeds or profile.source_family != PRESS_RELEASE_WIRE)
        and (include_sec_feeds or profile.source_family != REGULATORY_OFFICIAL)
    )
