"""Lightweight collection and extraction compliance decisions."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from .source_registry import SourceProfile


@dataclass(frozen=True)
class ComplianceDecision:
    allowed: bool
    reason: str


def collection_decision(
    profile: SourceProfile,
    *,
    discovery_method: str,
    url: str,
    user_agent: str,
    robots_parser: RobotFileParser | None = None,
) -> ComplianceDecision:
    if not profile.fetch_allowed:
        return ComplianceDecision(False, "fetch_disabled_by_profile")
    if discovery_method in {"rss", "atom", "api", "ticker_rss_search"}:
        return ComplianceDecision(True, "feed_or_api_allowed")
    if profile.javascript_required:
        return ComplianceDecision(False, "javascript_required")
    if not profile.robots_check_required:
        return ComplianceDecision(True, "robots_check_not_required")
    parser = robots_parser
    if parser is None:
        parsed = urlparse(url)
        parser = RobotFileParser(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            parser.read()
        except Exception:  # noqa: BLE001 - inability to verify means do not crawl.
            return ComplianceDecision(False, "robots_check_failed")
    return ComplianceDecision(
        parser.can_fetch(user_agent, url),
        "robots_allowed" if parser.can_fetch(user_agent, url) else "robots_disallowed",
    )


def extraction_decision(profile: SourceProfile) -> ComplianceDecision:
    if not profile.extract_allowed:
        return ComplianceDecision(False, "extraction_disabled_by_profile")
    if profile.javascript_required:
        return ComplianceDecision(False, "javascript_required")
    if profile.paywall_likely:
        return ComplianceDecision(True, "paywall_likely_do_not_bypass")
    return ComplianceDecision(True, "extraction_allowed")
