"""Ticker-specific match confidence and primary-subject detection."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from .article_types import MACRO_OR_SECTOR_ROUNDUP, classify_article_type
from .models import Article
from .tickers import TrackedTicker, load_tracked_tickers, match_tickers


HIGH = "high"
MEDIUM = "medium"
LOW = "low"


@dataclass(frozen=True)
class TickerMatchAssessment:
    ticker: str
    company_name: str
    confidence: float
    confidence_label: str
    reason: str
    basis: str
    primary: bool
    related: bool


def assess_ticker_matches(
    article: Article,
    tracked_tickers: Sequence[TrackedTicker] | None = None,
) -> tuple[TickerMatchAssessment, ...]:
    tracked = tuple(tracked_tickers or load_tracked_tickers())
    title_matches = _matched(article.title, tracked)
    snippet_matches = _matched(article.snippet or "", tracked)
    body_matches = _matched(article.full_text or "", tracked)
    all_symbols = {
        ticker.symbol
        for ticker in (*title_matches, *snippet_matches, *body_matches)
    }
    if not all_symbols:
        return ()

    article_type = classify_article_type(article).primary_type
    title_primary = _first_title_match(article.title, title_matches)
    headline_other_company = _headline_centers_other_company(article.title, title_matches)
    assessments: list[TickerMatchAssessment] = []
    for ticker in tracked:
        if ticker.symbol not in all_symbols:
            continue
        in_title = ticker in title_matches
        in_snippet = ticker in snippet_matches
        in_body = ticker in body_matches
        multi_title = len(title_matches) > 1
        primary = bool(in_title and ticker == title_primary and not multi_title)

        if primary:
            assessment = _assessment(ticker, 0.95, HIGH, "company_or_ticker_in_title", "title", True, False)
        elif in_title and multi_title:
            assessment = _assessment(
                ticker,
                0.45 if article_type == MACRO_OR_SECTOR_ROUNDUP else 0.6,
                MEDIUM,
                "multi_ticker_headline",
                "title",
                False,
                True,
            )
        elif in_title:
            assessment = _assessment(ticker, 0.85, HIGH, "company_or_ticker_in_title", "title", True, False)
        elif in_snippet and headline_other_company:
            assessment = _assessment(ticker, 0.2, LOW, "related_company_mention_in_snippet", "snippet", False, True)
        elif in_snippet:
            assessment = _assessment(ticker, 0.6, MEDIUM, "company_or_ticker_in_snippet", "snippet", False, False)
        elif in_body and headline_other_company:
            assessment = _assessment(ticker, 0.15, LOW, "related_company_mention_in_body", "full_text", False, True)
        else:
            assessment = _assessment(ticker, 0.5, MEDIUM, "company_or_ticker_in_body", "full_text", False, False)
        assessments.append(assessment)
    return tuple(assessments)


def primary_ticker(article: Article) -> str | None:
    return next((match.ticker for match in assess_ticker_matches(article) if match.primary), None)


def confidence_summary(
    articles: Iterable[Article],
) -> dict[str, object]:
    counts = {HIGH: 0, MEDIUM: 0, LOW: 0}
    reasons: dict[str, int] = {}
    for article in articles:
        for match in assess_ticker_matches(article):
            counts[match.confidence_label] += 1
            reasons[match.reason] = reasons.get(match.reason, 0) + 1
    return {
        "high_confidence_matches": counts[HIGH],
        "medium_confidence_matches": counts[MEDIUM],
        "low_confidence_matches": counts[LOW],
        "reason_counts": reasons,
    }


def _assessment(
    ticker: TrackedTicker,
    confidence: float,
    label: str,
    reason: str,
    basis: str,
    primary: bool,
    related: bool,
) -> TickerMatchAssessment:
    return TickerMatchAssessment(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        confidence=confidence,
        confidence_label=label,
        reason=reason,
        basis=basis,
        primary=primary,
        related=related,
    )


def _matched(text: str, tracked: Sequence[TrackedTicker]) -> tuple[TrackedTicker, ...]:
    symbols = {ticker.symbol for ticker in match_tickers(text)}
    return tuple(ticker for ticker in tracked if ticker.symbol in symbols)


def _first_title_match(
    title: str,
    matches: Sequence[TrackedTicker],
) -> TrackedTicker | None:
    if not matches:
        return None
    return min(matches, key=lambda ticker: _ticker_position(title, ticker))


def _ticker_position(text: str, ticker: TrackedTicker) -> int:
    positions = [
        match.start()
        for term in ticker.match_terms
        for match in re.finditer(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
    ]
    return min(positions, default=len(text))


def _headline_centers_other_company(
    title: str,
    title_matches: Sequence[TrackedTicker],
) -> bool:
    if title_matches:
        return True
    company = r"[A-Z][A-Za-z0-9.&'-]*(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,3}"
    event = (
        r"stock|shares|earnings|results|revenue|guidance|raises|reports|reviews|"
        r"launches|unveils|signs|expands|soars|falls|sinks|jumps"
    )
    return bool(
        re.search(rf"^{company}(?:'s|\u2019s)\s+(?:{event})\b", title)
        or re.search(rf"^{company}\s+(?:{event})\b", title)
    )

