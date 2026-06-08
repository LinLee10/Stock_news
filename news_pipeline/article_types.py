"""Deterministic article type classification for ranking and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .models import Article


EARNINGS_OR_RESULTS = "earnings_or_results"
GUIDANCE_OR_FORECAST = "guidance_or_forecast"
ANALYST_RATING_OR_PRICE_TARGET = "analyst_rating_or_price_target"
STOCK_PRICE_MOVE = "stock_price_move"
PRODUCT_OR_AI_OR_CHIP_NEWS = "product_or_ai_or_chip_news"
PARTNERSHIP_OR_CONTRACT = "partnership_or_contract"
CUSTOMER_OR_DEMAND_SIGNAL = "customer_or_demand_signal"
REGULATORY_OR_LEGAL = "regulatory_or_legal"
INSIDER_OR_INSTITUTIONAL_TRADING = "insider_or_institutional_trading"
FILING_OR_SEC_EVENT = "filing_or_sec_event"
MACRO_OR_SECTOR_ROUNDUP = "macro_or_sector_roundup"
GENERIC_BUY_SELL_HOLD_OPINION = "generic_buy_sell_hold_opinion"
PREDICTION_OR_PRICE_TARGET_CLICKBAIT = "prediction_or_price_target_clickbait"
BACKGROUND_PROFILE = "background_profile"
UNKNOWN = "unknown"


ARTICLE_TYPES = (
    EARNINGS_OR_RESULTS,
    GUIDANCE_OR_FORECAST,
    ANALYST_RATING_OR_PRICE_TARGET,
    STOCK_PRICE_MOVE,
    PRODUCT_OR_AI_OR_CHIP_NEWS,
    PARTNERSHIP_OR_CONTRACT,
    CUSTOMER_OR_DEMAND_SIGNAL,
    REGULATORY_OR_LEGAL,
    INSIDER_OR_INSTITUTIONAL_TRADING,
    FILING_OR_SEC_EVENT,
    MACRO_OR_SECTOR_ROUNDUP,
    GENERIC_BUY_SELL_HOLD_OPINION,
    PREDICTION_OR_PRICE_TARGET_CLICKBAIT,
    BACKGROUND_PROFILE,
    UNKNOWN,
)


ARTICLE_TYPE_SERIOUSNESS = {
    REGULATORY_OR_LEGAL: 14,
    EARNINGS_OR_RESULTS: 13,
    GUIDANCE_OR_FORECAST: 13,
    FILING_OR_SEC_EVENT: 11,
    PARTNERSHIP_OR_CONTRACT: 10,
    CUSTOMER_OR_DEMAND_SIGNAL: 9,
    PRODUCT_OR_AI_OR_CHIP_NEWS: 8,
    STOCK_PRICE_MOVE: 7,
    INSIDER_OR_INSTITUTIONAL_TRADING: 6,
    ANALYST_RATING_OR_PRICE_TARGET: 3,
    MACRO_OR_SECTOR_ROUNDUP: 1,
    BACKGROUND_PROFILE: 0,
    GENERIC_BUY_SELL_HOLD_OPINION: -3,
    PREDICTION_OR_PRICE_TARGET_CLICKBAIT: -6,
    UNKNOWN: 0,
}


_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        PREDICTION_OR_PRICE_TARGET_CLICKBAIT,
        (
            r"\bprediction\b",
            r"\bwill (?:trade|be worth|reach)\b",
            r"\bcould (?:soar|rise|fall) \d+%",
            r"\bwhere .* stock will be\b",
        ),
    ),
    (
        EARNINGS_OR_RESULTS,
        (
            r"\bearnings\b",
            r"\bquarterly results\b",
            r"\b(?:q[1-4]|first|second|third|fourth) quarter\b",
            r"\beps\b",
            r"\bprofit\b",
            r"\brevenue (?:beats|misses|rises|falls)\b",
        ),
    ),
    (
        GUIDANCE_OR_FORECAST,
        (
            r"\bguidance\b",
            r"\boutlook\b",
            r"\bforecast\b",
            r"\braises? (?:its )?(?:revenue|profit|sales) view\b",
            r"\bcuts? (?:its )?(?:revenue|profit|sales) view\b",
        ),
    ),
    (
        ANALYST_RATING_OR_PRICE_TARGET,
        (
            r"\banalyst\b",
            r"\bupgrade[sd]?\b",
            r"\bdowngrade[sd]?\b",
            r"\bprice target\b",
            r"\binitiates? coverage\b",
            r"\brating\b",
        ),
    ),
    (
        REGULATORY_OR_LEGAL,
        (
            r"\bantitrust\b",
            r"\bbackdoor\b",
            r"\bcourt\b",
            r"\bexport (?:ban|control|restriction)\b",
            r"\binvestigation\b",
            r"\blawsuit\b",
            r"\blegal\b",
            r"\bprobe\b",
            r"\bregulat(?:or|ory)\b",
            r"\bscrutiny\b",
        ),
    ),
    (
        FILING_OR_SEC_EVENT,
        (
            r"\bsec filing\b",
            r"\bform (?:8-k|10-k|10-q|13d|13g|4)\b",
            r"\bfiles? with the sec\b",
        ),
    ),
    (
        INSIDER_OR_INSTITUTIONAL_TRADING,
        (
            r"\binsider (?:buying|selling|sale|purchase)\b",
            r"\binstitutional (?:stake|holding|ownership)\b",
            r"\bhedge fund\b",
            r"\b13f\b",
        ),
    ),
    (
        PARTNERSHIP_OR_CONTRACT,
        (
            r"\bpartnership\b",
            r"\bpartners? with\b",
            r"\bcontract\b",
            r"\bcollaboration\b",
            r"\bdeal with\b",
        ),
    ),
    (
        CUSTOMER_OR_DEMAND_SIGNAL,
        (
            r"\bcustomer\b",
            r"\bdemand\b",
            r"\borders?\b",
            r"\bbookings?\b",
            r"\bbacklog\b",
        ),
    ),
    (
        PRODUCT_OR_AI_OR_CHIP_NEWS,
        (
            r"\bartificial intelligence\b",
            r"\bai\b",
            r"\bchips?\b",
            r"\bdata cent(?:er|re)\b",
            r"\blaunch(?:es|ed)?\b",
            r"\bplatform\b",
            r"\bproduct\b",
            r"\bunveil(?:s|ed)?\b",
        ),
    ),
    (
        STOCK_PRICE_MOVE,
        (
            r"\bstock (?:falls|fell|jumps|jumped|rallies|rises|rose|sinks|soars|surges|tumbles)\b",
            r"\bshares? (?:fall|fell|jump|rally|rise|rose|sink|soar|surge|tumble)\b",
            r"\bwhy .* stock (?:popped|dropped|crashed|rose|fell)\b",
            r"\bbiggest moves?\b",
        ),
    ),
    (
        MACRO_OR_SECTOR_ROUNDUP,
        (
            r"\bstocks? to watch\b",
            r"\bmarket roundup\b",
            r"\bsector roundup\b",
            r"\bbiggest moves? (?:premarket|midday)\b",
            r"\btop stocks?\b",
        ),
    ),
    (
        GENERIC_BUY_SELL_HOLD_OPINION,
        (
            r"\bshould you buy\b",
            r"\bis .* (?:a )?good stock to buy\b",
            r"\bbuy, sell, or hold\b",
            r"\bstock to buy now\b",
            r"\bbetter buy\b",
            r"\bundervalued\b",
            r"\bovervalued\b",
        ),
    ),
    (
        BACKGROUND_PROFILE,
        (
            r"\bwhat does .* do\b",
            r"\bcompany profile\b",
            r"\beverything you need to know\b",
            r"\ba look at\b",
        ),
    ),
)


_PRIMARY_PRECEDENCE = (
    PREDICTION_OR_PRICE_TARGET_CLICKBAIT,
    GENERIC_BUY_SELL_HOLD_OPINION,
    MACRO_OR_SECTOR_ROUNDUP,
    ANALYST_RATING_OR_PRICE_TARGET,
    EARNINGS_OR_RESULTS,
    GUIDANCE_OR_FORECAST,
    REGULATORY_OR_LEGAL,
    FILING_OR_SEC_EVENT,
    STOCK_PRICE_MOVE,
    PARTNERSHIP_OR_CONTRACT,
    CUSTOMER_OR_DEMAND_SIGNAL,
    PRODUCT_OR_AI_OR_CHIP_NEWS,
    INSIDER_OR_INSTITUTIONAL_TRADING,
    BACKGROUND_PROFILE,
)


@dataclass(frozen=True)
class ArticleTypeClassification:
    primary_type: str
    event_types: tuple[str, ...]
    reasons: tuple[str, ...]


def classify_article_type(article: Article) -> ArticleTypeClassification:
    text = " ".join(
        part
        for part in (
            article.title,
            article.snippet or "",
            article.full_text or "",
            str(article.metadata.get("source_name") or ""),
        )
        if part
    )
    headline = classify_article_text(article.title)
    combined = classify_article_text(text)
    primary = headline.primary_type if headline.primary_type != UNKNOWN else combined.primary_type
    event_types = tuple(
        article_type
        for article_type in ARTICLE_TYPES
        if article_type in set(headline.event_types) | set(combined.event_types)
    )
    reasons = tuple(dict.fromkeys((*headline.reasons, *combined.reasons)))
    return ArticleTypeClassification(primary, event_types, reasons)


def classify_article_text(text: str) -> ArticleTypeClassification:
    normalized = " ".join(text.casefold().split())
    matched: list[str] = []
    reasons: list[str] = []
    for article_type, patterns in _TYPE_PATTERNS:
        pattern = next((pattern for pattern in patterns if re.search(pattern, normalized)), None)
        if pattern is None:
            continue
        matched.append(article_type)
        reasons.append(f"{article_type}:{pattern}")

    event_types = tuple(article_type for article_type in ARTICLE_TYPES if article_type in matched)
    primary = next((article_type for article_type in _PRIMARY_PRECEDENCE if article_type in matched), UNKNOWN)
    if not event_types:
        event_types = (UNKNOWN,)
        reasons = ["unknown:no_classification_rule_matched"]
    return ArticleTypeClassification(primary, event_types, tuple(reasons))


def article_type_counts(articles: Iterable[Article]) -> dict[str, int]:
    counts = {article_type: 0 for article_type in ARTICLE_TYPES}
    for article in articles:
        counts[classify_article_type(article).primary_type] += 1
    return counts
