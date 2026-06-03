"""First-class portfolio and watchlist ticker configuration."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal


TickerGroup = Literal["portfolio", "watchlist"]


@dataclass(frozen=True)
class TrackedTicker:
    symbol: str
    company_name: str
    aliases: tuple[str, ...]
    group: TickerGroup

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())

    @property
    def match_terms(self) -> tuple[str, ...]:
        return (self.symbol, self.company_name, *self.aliases)


PORTFOLIO: tuple[TrackedTicker, ...] = (
    TrackedTicker("SNDK", "SanDisk", ("Sandisk", "SanDisk Corporation"), "portfolio"),
    TrackedTicker("ASML", "ASML", ("ASML Holding", "ASML Holding N.V."), "portfolio"),
    TrackedTicker("MU", "Micron", ("Micron Technology", "Micron Technology Inc."), "portfolio"),
    TrackedTicker("AVGO", "Broadcom", ("Broadcom Inc.",), "portfolio"),
    TrackedTicker("NBIS", "Nebius", ("Nebius Group", "Nebius Group N.V."), "portfolio"),
    TrackedTicker("NVDA", "NVIDIA", ("Nvidia", "NVIDIA Corporation"), "portfolio"),
    TrackedTicker("PLTR", "Palantir", ("Palantir Technologies", "Palantir Technologies Inc."), "portfolio"),
)

WATCHLIST: tuple[TrackedTicker, ...] = (
    TrackedTicker("ADI", "Analog Devices", ("Analog Devices Inc.",), "watchlist"),
    TrackedTicker("VRT", "Vertiv", ("Vertiv Holdings", "Vertiv Holdings Co."), "watchlist"),
    TrackedTicker("MRVL", "Marvell", ("Marvell Technology", "Marvell Technology Inc."), "watchlist"),
    TrackedTicker("PANW", "Palo Alto Networks", ("Palo Alto",), "watchlist"),
    TrackedTicker("CRWV", "CoreWeave", ("CoreWeave Inc.",), "watchlist"),
    TrackedTicker("APLD", "Applied Digital", ("Applied Digital Corporation",), "watchlist"),
    TrackedTicker("CORZ", "Core Scientific", ("Core Scientific Inc.",), "watchlist"),
    TrackedTicker("GEV", "GE Vernova", ("GEVernova",), "watchlist"),
    TrackedTicker("META", "Meta", ("Meta Platforms", "Meta Platforms Inc.", "Facebook"), "watchlist"),
    TrackedTicker("AMD", "Advanced Micro Devices", ("AMD", "Advanced Micro Devices Inc."), "watchlist"),
    TrackedTicker("TSM", "Taiwan Semiconductor", ("TSMC", "Taiwan Semiconductor Manufacturing"), "watchlist"),
    TrackedTicker("ARM", "Arm Holdings", ("Arm", "Arm Holdings plc"), "watchlist"),
    TrackedTicker("RDDT", "Reddit", ("Reddit Inc.",), "watchlist"),
    TrackedTicker("ACHR", "Archer Aviation", ("Archer", "Archer Aviation Inc."), "watchlist"),
)


def load_portfolio() -> tuple[TrackedTicker, ...]:
    return PORTFOLIO


def load_watchlist() -> tuple[TrackedTicker, ...]:
    return WATCHLIST


def load_tracked_tickers() -> tuple[TrackedTicker, ...]:
    return PORTFOLIO + WATCHLIST


def symbols(group: TickerGroup | None = None) -> tuple[str, ...]:
    tickers = _select_group(group)
    return tuple(ticker.symbol for ticker in tickers)


def ticker_lookup(group: TickerGroup | None = None) -> dict[str, TrackedTicker]:
    return {ticker.symbol: ticker for ticker in _select_group(group)}


def match_tickers(text: str, group: TickerGroup | None = None) -> tuple[TrackedTicker, ...]:
    """Match configured tickers against headline or article text."""
    matches: list[TrackedTicker] = []
    for ticker in _select_group(group):
        if any(_contains_term(text, term) for term in ticker.match_terms):
            matches.append(ticker)
    return tuple(matches)


def _select_group(group: TickerGroup | None) -> tuple[TrackedTicker, ...]:
    if group == "portfolio":
        return PORTFOLIO
    if group == "watchlist":
        return WATCHLIST
    return load_tracked_tickers()


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None
