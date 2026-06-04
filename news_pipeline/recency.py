"""UTC date normalization and recency weighting for daily reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone


RECENCY_WEIGHTS = {
    "today_signal": 1.0,
    "recent_pulse": 0.7,
    "weekly_trend": 0.4,
    "background_context": 0.15,
    "archive_context": 0.0,
    "unknown": 0.0,
}


@dataclass(frozen=True)
class RecencyInfo:
    timestamp_utc: str | None
    article_age_hours: float | None
    recency_bucket: str
    source: str


def article_recency(
    *,
    run_date: str,
    published_at: str | None,
    collected_at: str | None,
    archive_context: bool = False,
) -> RecencyInfo:
    """Resolve an article timestamp and assign a daily-report recency bucket."""
    run_start = _run_date_start_utc(run_date)
    timestamp = parse_datetime_utc(published_at)
    source = "published_at"
    if timestamp is None:
        timestamp = parse_datetime_utc(collected_at)
        source = "collected_at" if timestamp is not None else "missing"
    if timestamp is None:
        return RecencyInfo(None, None, "unknown", source)

    age_hours = (run_start - timestamp).total_seconds() / 3600.0
    bucket = recency_bucket_for_age(age_hours, archive_context=archive_context)
    return RecencyInfo(timestamp.isoformat(), round(age_hours, 4), bucket, source)


def recency_bucket_for_age(age_hours: float, *, archive_context: bool = False) -> str:
    if archive_context:
        return "archive_context"
    if age_hours < 0:
        return "today_signal"
    if age_hours <= 24:
        return "today_signal"
    if age_hours <= 72:
        return "recent_pulse"
    if age_hours <= 168:
        return "weekly_trend"
    if age_hours <= 720:
        return "background_context"
    return "archive_context"


def recency_weight(bucket: str) -> float:
    return RECENCY_WEIGHTS.get(bucket, 0.0)


def parse_datetime_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _run_date_start_utc(value: str) -> datetime:
    parsed_date = date.fromisoformat(value)
    return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
