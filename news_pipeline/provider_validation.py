"""Dry-run provider validation without live API calls or secret storage."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Protocol

from .models import utc_now_iso
from .provider_registry import ProviderConfig, get_provider_config, key_state


@dataclass(frozen=True)
class ProviderValidationResult:
    provider_name: str
    limit_type: str
    reset_window: str
    last_checked_at: str
    last_status: str
    remaining_quota: int | None
    quota_truth_source: str
    dry_run: bool
    key_state: str

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "limit_type": self.limit_type,
            "reset_window": self.reset_window,
            "last_checked_at": self.last_checked_at,
            "last_status": self.last_status,
            "remaining_quota": self.remaining_quota,
            "quota_truth_source": self.quota_truth_source,
            "dry_run": self.dry_run,
            "key_state": self.key_state,
        }


@dataclass(frozen=True)
class ProviderCheckResult:
    status: str
    remaining_quota: int | None = None
    quota_truth_source: str | None = None


class ProviderChecker(Protocol):
    def check(self, config: ProviderConfig) -> ProviderCheckResult:
        """Return fake or local validation status without receiving secrets."""


def redact_key(value: str | None) -> str:
    if not value:
        return "<missing>"
    return "<redacted>"


def validate_provider(
    provider: str | ProviderConfig,
    *,
    dry_run: bool = True,
    environ: Mapping[str, str] | None = None,
    checker: ProviderChecker | None = None,
) -> ProviderValidationResult:
    """Validate provider readiness without live calls in dry-run mode."""
    config = get_provider_config(provider) if isinstance(provider, str) else provider
    env = os.environ if environ is None else environ
    state = key_state(config, env)

    status = "dry_run_ok"
    remaining_quota: int | None = None
    quota_truth_source = config.quota_truth_source

    if config.requires_key and state == "missing":
        status = "missing_key"

    if checker is not None:
        check = checker.check(config)
        status = check.status
        remaining_quota = check.remaining_quota
        quota_truth_source = check.quota_truth_source or quota_truth_source
    elif not dry_run:
        status = "not_checked_live_calls_disabled"

    return ProviderValidationResult(
        provider_name=config.name,
        limit_type=config.limit_type,
        reset_window=config.reset_window,
        last_checked_at=utc_now_iso(),
        last_status=status,
        remaining_quota=remaining_quota,
        quota_truth_source=quota_truth_source,
        dry_run=dry_run,
        key_state=state,
    )
