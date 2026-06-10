"""Deterministic ticker-aware query planning for news sources."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

from news_pipeline.tickers import AMBIGUOUS_TICKER_TERMS, TrackedTicker


HIGH_PRECISION = "high"
MEDIUM_PRECISION = "medium"
LOW_PRECISION = "low"

PREFERRED_COMPANY_QUERY_NAMES = {
    "ASML": "ASML Holding",
    "CORZ": "Core Scientific",
    "CRWV": "CoreWeave",
    "META": "Meta Platforms",
    "MRVL": "Marvell Technology",
    "MU": "Micron Technology",
    "PANW": "Palo Alto Networks",
    "PLTR": "Palantir Technologies",
    "VRT": "Vertiv Holdings",
}
WEAK_COVERAGE_TICKERS = frozenset(PREFERRED_COMPANY_QUERY_NAMES)


@dataclass(frozen=True)
class TickerQueryPlan:
    query_id: str
    ticker: str
    company: str
    query_text: str
    query_type: str
    source_family: str
    provider_target: str
    expected_precision: str
    expected_recall: str
    priority: int
    daily_budget_cost_estimate: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "ticker": self.ticker,
            "company": self.company,
            "query_text": self.query_text,
            "query_type": self.query_type,
            "source_family": self.source_family,
            "provider_target": self.provider_target,
            "expected_precision": self.expected_precision,
            "expected_recall": self.expected_recall,
            "priority": self.priority,
            "daily_budget_cost_estimate": self.daily_budget_cost_estimate,
        }


PROVIDER_FAMILIES = {
    "marketaux": "external_market_news_api",
    "nyt": "context_news_api",
    "finnhub_news": "external_market_news_api",
    "gnews": "external_general_news_api",
    "newsapi": "external_general_news_api",
    "google_news_rss_search": "google_news_backstop",
    "press_release_search": "press_release_wire",
}


def plan_ticker_queries(
    tracked_tickers: Sequence[TrackedTicker],
    *,
    provider_targets: Sequence[str] = (
        "marketaux",
        "nyt",
        "finnhub_news",
        "gnews",
        "newsapi",
        "google_news_rss_search",
        "press_release_search",
    ),
) -> tuple[TickerQueryPlan, ...]:
    plans: list[TickerQueryPlan] = []
    for ticker in tracked_tickers:
        for provider in provider_targets:
            plans.extend(_provider_queries(ticker, provider))
    return tuple(
        sorted(
            plans,
            key=lambda plan: (
                -plan.priority,
                plan.ticker,
                plan.provider_target,
                plan.query_type,
            ),
        )
    )


def provider_query_plans(
    plans: Sequence[TickerQueryPlan],
    provider: str,
) -> tuple[TickerQueryPlan, ...]:
    return tuple(plan for plan in plans if plan.provider_target == provider)


def _provider_queries(
    ticker: TrackedTicker,
    provider: str,
) -> tuple[TickerQueryPlan, ...]:
    family = PROVIDER_FAMILIES[provider]
    company = _company_phrase(ticker)
    aliases = _useful_aliases(ticker)
    company_context = " OR ".join(f'"{name}"' for name in (company, *aliases[:2]))
    base = [
        ("company_stock", f"{company_context} stock OR shares", HIGH_PRECISION, "high", 100),
        ("earnings", f"{company_context} earnings OR results OR revenue", HIGH_PRECISION, "medium", 96),
        ("guidance", f"{company_context} guidance OR outlook OR forecast", HIGH_PRECISION, "medium", 94),
        (
            "product_ai",
            f"{company_context} AI OR chips OR \"data center\"",
            HIGH_PRECISION,
            "medium",
            92,
        ),
        (
            "company_catalyst",
            f"{company_context} regulation OR partnership OR contract",
            HIGH_PRECISION,
            "medium",
            90,
        ),
        (
            "analyst_action",
            f"{company_context} analyst OR upgrade OR downgrade",
            MEDIUM_PRECISION,
            "medium",
            82,
        ),
        (
            "official_event",
            f"{company_context} \"SEC filing\" OR \"press release\"",
            HIGH_PRECISION,
            "low",
            88,
        ),
        (
            "ticker_company",
            f"{ticker.symbol} \"{company}\"",
            HIGH_PRECISION,
            "medium",
            98,
        ),
    ]
    if not _ambiguous_ticker(ticker):
        base.append(
            ("ticker_only", f"{ticker.symbol} stock", LOW_PRECISION, "high", 45)
        )

    if provider == "finnhub_news":
        base = [("company_news", ticker.symbol, HIGH_PRECISION, "high", 100)]
    elif provider == "marketaux":
        base = [
            ("entity_symbol", ticker.symbol, HIGH_PRECISION, "high", 100),
            *base[:4],
        ]
    elif provider == "nyt":
        base = [
            (
                "company_context",
                f'"{company}"',
                HIGH_PRECISION,
                "high",
                _coverage_priority(ticker, 105),
            ),
            (
                "company_technology_context",
                f'"{company}" AI OR chips OR semiconductors OR "data centers" OR cloud OR cybersecurity',
                HIGH_PRECISION,
                "medium",
                _coverage_priority(ticker, 100),
            ),
            (
                "company_policy_context",
                f'"{company}" antitrust OR regulation OR "export controls" OR "energy infrastructure"',
                HIGH_PRECISION,
                "medium",
                _coverage_priority(ticker, 98),
            ),
        ]
    elif provider in {"gnews", "newsapi"} and ticker.symbol in WEAK_COVERAGE_TICKERS:
        base = [
            (
                "company_coverage_gap",
                f'"{company}" AI OR earnings OR chips OR regulation OR "data centers"',
                HIGH_PRECISION,
                "high",
                110,
            ),
            *base,
        ]
    elif provider == "press_release_search":
        base = [
            (
                "press_release",
                f"{company_context} earnings OR guidance OR partnership OR contract OR product",
                HIGH_PRECISION,
                "medium",
                95,
            ),
        ]
    elif provider == "google_news_rss_search":
        base = [
            (
                "google_gap_fill",
                f"{company_context} stock OR earnings OR guidance OR AI OR regulation",
                HIGH_PRECISION,
                "high",
                80,
            ),
        ]

    return tuple(
        _plan(
            ticker,
            provider=provider,
            family=family,
            query_type=query_type,
            query_text=query_text,
            precision=precision,
            recall=recall,
            priority=priority,
        )
        for query_type, query_text, precision, recall, priority in base
    )


def _plan(
    ticker: TrackedTicker,
    *,
    provider: str,
    family: str,
    query_type: str,
    query_text: str,
    precision: str,
    recall: str,
    priority: int,
) -> TickerQueryPlan:
    identity = f"{ticker.symbol}|{provider}|{query_type}|{query_text}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return TickerQueryPlan(
        query_id=f"query-{digest}",
        ticker=ticker.symbol,
        company=ticker.company_name,
        query_text=query_text,
        query_type=query_type,
        source_family=family,
        provider_target=provider,
        expected_precision=precision,
        expected_recall=recall,
        priority=priority,
    )


def _company_phrase(ticker: TrackedTicker) -> str:
    return PREFERRED_COMPANY_QUERY_NAMES.get(
        ticker.symbol,
        ticker.company_name.strip(),
    )


def _coverage_priority(ticker: TrackedTicker, priority: int) -> int:
    return priority + (10 if ticker.symbol in WEAK_COVERAGE_TICKERS else 0)


def _useful_aliases(ticker: TrackedTicker) -> tuple[str, ...]:
    company = ticker.company_name.casefold()
    return tuple(
        alias
        for alias in ticker.aliases
        if alias.casefold() != company and len(alias.strip()) >= 3
    )


def _ambiguous_ticker(ticker: TrackedTicker) -> bool:
    return (
        ticker.symbol in AMBIGUOUS_TICKER_TERMS
        or ticker.company_name in AMBIGUOUS_TICKER_TERMS
    )
