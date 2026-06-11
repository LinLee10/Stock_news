"""Provider configuration registry without secret values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider_type: str
    limit_type: str
    reset_window: str
    quota_truth_source: str
    key_env_var: str | None = None
    dry_run_supported: bool = True

    @property
    def requires_key(self) -> bool:
        return self.key_env_var is not None


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "google_news_rss_search": ProviderConfig(
        name="google_news_rss_search",
        provider_type="rss",
        limit_type="best_effort",
        reset_window="none",
        quota_truth_source="no_provider_quota",
    ),
    "yahoo_finance_rss": ProviderConfig(
        name="yahoo_finance_rss",
        provider_type="rss",
        limit_type="best_effort",
        reset_window="none",
        quota_truth_source="no_provider_quota",
    ),
    "cnbc_rss": ProviderConfig(
        name="cnbc_rss",
        provider_type="rss",
        limit_type="best_effort",
        reset_window="none",
        quota_truth_source="no_provider_quota",
    ),
    "marketwatch_rss": ProviderConfig(
        name="marketwatch_rss",
        provider_type="rss",
        limit_type="best_effort",
        reset_window="none",
        quota_truth_source="no_provider_quota",
    ),
    "sec_edgar": ProviderConfig(
        name="sec_edgar",
        provider_type="official_api",
        limit_type="published_rate_limit",
        reset_window="none",
        quota_truth_source="sec_fair_access_policy",
    ),
    "marketaux": ProviderConfig(
        name="marketaux",
        provider_type="external_free_tier_api",
        limit_type="monthly_quota",
        reset_window="monthly",
        quota_truth_source="provider_response_or_plan",
        key_env_var="MARKETAUX_API_KEY",
    ),
    "alpha_vantage_news": ProviderConfig(
        name="alpha_vantage_news",
        provider_type="external_free_tier_api",
        limit_type="daily_quota",
        reset_window="daily",
        quota_truth_source="provider_response_or_plan",
        key_env_var="ALPHA_VANTAGE_KEY",
    ),
    "nyt": ProviderConfig(
        name="nyt",
        provider_type="external_free_tier_api",
        limit_type="daily_quota",
        reset_window="daily",
        quota_truth_source="provider_response_or_plan",
        key_env_var="NYT_API_KEY",
    ),
    "gnews": ProviderConfig(
        name="gnews",
        provider_type="external_free_tier_api",
        limit_type="daily_quota",
        reset_window="daily",
        quota_truth_source="provider_response_or_plan",
        key_env_var="GNEWS_KEY",
    ),
    "finnhub_news": ProviderConfig(
        name="finnhub_news",
        provider_type="external_free_tier_api",
        limit_type="plan_quota",
        reset_window="provider_plan",
        quota_truth_source="provider_response_or_plan",
        key_env_var="FINNHUB_KEY",
    ),
    "newsapi": ProviderConfig(
        name="newsapi",
        provider_type="external_free_tier_api",
        limit_type="daily_quota",
        reset_window="daily",
        quota_truth_source="provider_response_or_plan",
        key_env_var="NEWSAPI_KEY",
    ),
    "resend": ProviderConfig(
        name="resend",
        provider_type="email",
        limit_type="plan_quota",
        reset_window="provider_plan",
        quota_truth_source="provider_dashboard_or_response",
        key_env_var="RESEND_API_KEY",
    ),
}


def get_provider_config(name: str) -> ProviderConfig:
    return PROVIDER_REGISTRY[name]


def iter_provider_configs() -> tuple[ProviderConfig, ...]:
    return tuple(PROVIDER_REGISTRY.values())


def key_state(config: ProviderConfig, environ: Mapping[str, str]) -> str:
    """Return key presence only; never return or store the key value."""
    if config.key_env_var is None:
        return "not_required"
    return "present" if bool(environ.get(config.key_env_var)) else "missing"
