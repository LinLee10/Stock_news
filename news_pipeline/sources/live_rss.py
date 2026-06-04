"""Opt-in live RSS discovery with retry and timeout controls."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from urllib.request import Request, urlopen

from news_pipeline.models import Article
from news_pipeline.tickers import match_tickers

from .rss_config import (
    DEFAULT_MAX_ARTICLES_PER_SOURCE,
    DEFAULT_MAX_ARTICLES_PER_TICKER,
    DEFAULT_MAX_TOTAL_LIVE_ARTICLES,
    RssFeedConfig,
    RssSourceFamilyConfig,
    enabled_source_families,
    feed_urls_for_family,
)
from .rss import RssSource


DEFAULT_LIVE_RSS_TIMEOUT_SECONDS = 8.0
DEFAULT_LIVE_RSS_RETRIES = 1
DEFAULT_LIVE_RSS_USER_AGENT = "StonkNewsPipeline/0.1 (+local dry-run live RSS test)"


@dataclass(frozen=True)
class LiveRssAttempt:
    provider: str
    feed_id: str
    feed_url: str
    status: str
    article_count: int
    fetched_article_count: int
    latency_ms: int
    attempts: int
    error_class: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "feed_id": self.feed_id,
            "feed_url": self.feed_url,
            "status": self.status,
            "article_count": self.article_count,
            "fetched_article_count": self.fetched_article_count,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class _FetchedFeed:
    provider: str
    feed: RssFeedConfig
    articles: list[Article]
    attempt: LiveRssAttempt


def default_live_rss_urls() -> tuple[str, ...]:
    """Return unpaid RSS search URLs for configured tickers."""
    google_family = next(
        family for family in enabled_source_families()
        if family.name == "google_news_rss_search"
    )
    return tuple(feed.url for feed in feed_urls_for_family(google_family))


def collect_live_rss_articles(
    feed_urls: tuple[str, ...] | None = None,
    *,
    timeout_seconds: float = DEFAULT_LIVE_RSS_TIMEOUT_SECONDS,
    retries: int = DEFAULT_LIVE_RSS_RETRIES,
    user_agent: str = DEFAULT_LIVE_RSS_USER_AGENT,
    source_families: tuple[RssSourceFamilyConfig, ...] | None = None,
    max_articles_per_source: int = DEFAULT_MAX_ARTICLES_PER_SOURCE,
    max_articles_per_ticker: int = DEFAULT_MAX_ARTICLES_PER_TICKER,
    max_total_articles: int = DEFAULT_MAX_TOTAL_LIVE_ARTICLES,
) -> tuple[list[Article], list[LiveRssAttempt]]:
    """Fetch RSS XML only and convert feed items to articles.

    Article HTML extraction is intentionally not performed here.
    """
    fetched_feeds: list[_FetchedFeed] = []
    feed_specs = _feed_specs(feed_urls, source_families)
    for provider, feed in feed_specs:
        fetched_articles, fetch_attempt = _fetch_one_feed(
            provider,
            feed,
            timeout_seconds=timeout_seconds,
            retries=retries,
            user_agent=user_agent,
        )
        fetched_feeds.append(_FetchedFeed(provider, feed, fetched_articles, fetch_attempt))

    articles, accepted_counts = _cap_fetched_feeds(
        fetched_feeds,
        max_articles_per_source=max_articles_per_source,
        max_articles_per_ticker=max_articles_per_ticker,
        max_total_articles=max_total_articles,
    )
    attempts = [
        LiveRssAttempt(
            provider=fetched.attempt.provider,
            feed_id=fetched.attempt.feed_id,
            feed_url=fetched.attempt.feed_url,
            status=fetched.attempt.status,
            article_count=accepted_counts.get(_feed_key(fetched.provider, fetched.feed), 0),
            fetched_article_count=fetched.attempt.fetched_article_count,
            latency_ms=fetched.attempt.latency_ms,
            attempts=fetched.attempt.attempts,
            error_class=fetched.attempt.error_class,
            error_message=fetched.attempt.error_message,
        )
        for fetched in fetched_feeds
    ]
    return articles, attempts


def _cap_fetched_feeds(
    fetched_feeds: list[_FetchedFeed],
    *,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
    max_total_articles: int,
) -> tuple[list[Article], dict[tuple[str, str], int]]:
    accepted: list[Article] = []
    accepted_counts: dict[tuple[str, str], int] = {}
    ticker_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}

    providers = tuple(dict.fromkeys(fetched.provider for fetched in fetched_feeds))
    for provider in providers:
        provider_feeds = [fetched for fetched in fetched_feeds if fetched.provider == provider]
        if provider == "google_news_rss_search":
            _accept_round_robin(
                provider_feeds,
                accepted,
                accepted_counts,
                provider_counts,
                ticker_counts,
                max_articles_per_source=max_articles_per_source,
                max_articles_per_ticker=max_articles_per_ticker,
                max_total_articles=max_total_articles,
            )
        else:
            _accept_sequential(
                provider_feeds,
                accepted,
                accepted_counts,
                provider_counts,
                ticker_counts,
                max_articles_per_source=max_articles_per_source,
                max_articles_per_ticker=max_articles_per_ticker,
                max_total_articles=max_total_articles,
            )
    return accepted, accepted_counts


def _accept_round_robin(
    fetched_feeds: list[_FetchedFeed],
    accepted: list[Article],
    accepted_counts: dict[tuple[str, str], int],
    provider_counts: dict[str, int],
    ticker_counts: dict[str, int],
    *,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
    max_total_articles: int,
) -> None:
    positions = {index: 0 for index in range(len(fetched_feeds))}
    made_progress = True
    while made_progress and len(accepted) < max_total_articles:
        made_progress = False
        for index, fetched in enumerate(fetched_feeds):
            if len(accepted) >= max_total_articles:
                return
            if provider_counts.get(fetched.provider, 0) >= max_articles_per_source:
                return
            while positions[index] < len(fetched.articles):
                article = fetched.articles[positions[index]]
                positions[index] += 1
                if _accept_article(
                    article,
                    fetched,
                    accepted,
                    accepted_counts,
                    provider_counts,
                    ticker_counts,
                    max_articles_per_source=max_articles_per_source,
                    max_articles_per_ticker=max_articles_per_ticker,
                    max_total_articles=max_total_articles,
                ):
                    made_progress = True
                    break


def _accept_sequential(
    fetched_feeds: list[_FetchedFeed],
    accepted: list[Article],
    accepted_counts: dict[tuple[str, str], int],
    provider_counts: dict[str, int],
    ticker_counts: dict[str, int],
    *,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
    max_total_articles: int,
) -> None:
    for fetched in fetched_feeds:
        for article in fetched.articles:
            if len(accepted) >= max_total_articles:
                return
            if provider_counts.get(fetched.provider, 0) >= max_articles_per_source:
                break
            _accept_article(
                article,
                fetched,
                accepted,
                accepted_counts,
                provider_counts,
                ticker_counts,
                max_articles_per_source=max_articles_per_source,
                max_articles_per_ticker=max_articles_per_ticker,
                max_total_articles=max_total_articles,
            )


def _accept_article(
    article: Article,
    fetched: _FetchedFeed,
    accepted: list[Article],
    accepted_counts: dict[tuple[str, str], int],
    provider_counts: dict[str, int],
    ticker_counts: dict[str, int],
    *,
    max_articles_per_source: int,
    max_articles_per_ticker: int,
    max_total_articles: int,
) -> bool:
    if len(accepted) >= max_total_articles:
        return False
    if provider_counts.get(fetched.provider, 0) >= max_articles_per_source:
        return False
    tickers = _matched_tickers(article)
    if not tickers:
        return False
    if any(ticker_counts.get(ticker, 0) >= max_articles_per_ticker for ticker in tickers):
        return False
    accepted.append(article)
    key = _feed_key(fetched.provider, fetched.feed)
    accepted_counts[key] = accepted_counts.get(key, 0) + 1
    provider_counts[fetched.provider] = provider_counts.get(fetched.provider, 0) + 1
    for ticker in tickers:
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    return True


def _feed_key(provider: str, feed: RssFeedConfig) -> tuple[str, str]:
    return (provider, feed.feed_id)


def _matched_tickers(article: Article) -> tuple[str, ...]:
    return tuple(ticker.symbol for ticker in match_tickers(_article_text(article)))


def _fetch_one_feed(
    provider: str,
    feed: RssFeedConfig,
    *,
    timeout_seconds: float,
    retries: int,
    user_agent: str,
) -> tuple[list[Article], LiveRssAttempt]:
    started = monotonic()
    max_attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt_number in range(1, max_attempts + 1):
        try:
            request = Request(feed.url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=timeout_seconds) as response:
                feed_xml = response.read().decode("utf-8", errors="replace")
            articles = RssSource(feed_xml, provider_name=provider).articles()
            latency_ms = int((monotonic() - started) * 1000)
            return articles, LiveRssAttempt(
                provider=provider,
                feed_id=feed.feed_id,
                feed_url=feed.url,
                status="success",
                article_count=len(articles),
                fetched_article_count=len(articles),
                latency_ms=latency_ms,
                attempts=attempt_number,
            )
        except Exception as exc:  # noqa: BLE001 - discovery must isolate source failures.
            last_error = exc

    latency_ms = int((monotonic() - started) * 1000)
    return [], LiveRssAttempt(
        provider=provider,
        feed_id=feed.feed_id,
        feed_url=feed.url,
        status="failure",
        article_count=0,
        fetched_article_count=0,
        latency_ms=latency_ms,
        attempts=max_attempts,
        error_class=type(last_error).__name__ if last_error else "UnknownError",
        error_message=_safe_error_message(last_error),
    )


def _safe_error_message(error: Exception | None) -> str | None:
    if error is None:
        return None
    message = str(error).strip()
    if not message:
        return None
    return message[:200]


def _feed_specs(
    feed_urls: tuple[str, ...] | None,
    source_families: tuple[RssSourceFamilyConfig, ...] | None,
) -> tuple[tuple[str, RssFeedConfig], ...]:
    if feed_urls:
        return tuple(
            (
                "google_news_rss_search",
                RssFeedConfig(feed_id=f"manual_live_rss_{index}", url=url),
            )
            for index, url in enumerate(feed_urls, start=1)
        )
    families = source_families or enabled_source_families()
    return tuple(
        (family.name, feed)
        for family in families
        for feed in feed_urls_for_family(family)
    )


def _article_text(article: Article) -> str:
    return " ".join(part for part in (article.title, article.snippet or "") if part)
