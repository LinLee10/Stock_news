"""Optional, cost-capped OpenAI portfolio briefing support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from urllib.request import Request, urlopen

from .dedup import DedupeCluster, SourceLink
from .event_memory import EventMemoryComparison, EventMemoryRecord
from .models import Article
from .source_quality import assess_article_source
from .tickers import ticker_lookup


DEFAULT_LLM_BRIEFING_TIER = "daily"
DEFAULT_MAX_LLM_EVENT_PACKETS = 200
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class LlmBriefingTier:
    name: str
    model: str
    purpose: str
    max_input_tokens: int
    max_output_tokens: int
    cost_cap_usd: float
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


LLM_BRIEFING_TIERS: Mapping[str, LlmBriefingTier] = {
    "daily": LlmBriefingTier(
        name="daily",
        model="gpt-5.4-mini",
        purpose="default daily portfolio/watchlist briefing",
        max_input_tokens=60_000,
        max_output_tokens=5_000,
        cost_cap_usd=0.25,
        input_cost_per_million_tokens=0.75,
        output_cost_per_million_tokens=4.50,
    ),
    "strong": LlmBriefingTier(
        name="strong",
        model="gpt-5.4",
        purpose="optional deeper synthesis",
        max_input_tokens=80_000,
        max_output_tokens=7_000,
        cost_cap_usd=1.00,
        input_cost_per_million_tokens=2.50,
        output_cost_per_million_tokens=15.00,
    ),
    "premium": LlmBriefingTier(
        name="premium",
        model="gpt-5.5",
        purpose="manual one-off deep analysis only",
        max_input_tokens=100_000,
        max_output_tokens=10_000,
        cost_cap_usd=3.00,
        input_cost_per_million_tokens=5.00,
        output_cost_per_million_tokens=30.00,
    ),
}


@dataclass(frozen=True)
class LlmBriefingEstimate:
    estimated_input_tokens: int
    reserved_output_tokens: int
    estimated_cost_usd: float
    cost_cap_usd: float
    blocked: bool
    blocked_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlmBriefingRunData:
    source_run_dir: str
    run_date: str
    records: tuple[EventMemoryRecord, ...]
    clusters: tuple[DedupeCluster, ...]
    comparison: EventMemoryComparison


class LlmBriefingArtifactError(ValueError):
    """Raised when prior-run artifacts cannot produce briefing packets."""


class LlmBriefingClient(Protocol):
    def create_briefing(
        self,
        *,
        model: str,
        input_payload: Mapping[str, object],
        response_schema: Mapping[str, object],
        max_output_tokens: int,
    ) -> Mapping[str, object]:
        """Return a structured portfolio briefing response."""


class OpenAIResponsesClient:
    """Minimal Responses API wrapper, instantiated only after explicit consent."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = OPENAI_RESPONSES_ENDPOINT,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def create_briefing(
        self,
        *,
        model: str,
        input_payload: Mapping[str, object],
        response_schema: Mapping[str, object],
        max_output_tokens: int,
    ) -> Mapping[str, object]:
        request_body = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Explain material portfolio and watchlist changes "
                                "from the supplied evidence. Do not provide buy, "
                                "sell, allocation, or trading advice."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                input_payload,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }
                    ],
                },
            ],
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "portfolio_intelligence_briefing",
                    "strict": True,
                    "schema": response_schema,
                }
            },
        }
        request = Request(
            self._endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("OpenAI response was not a JSON object")
        return payload


def get_llm_briefing_tier(name: str) -> LlmBriefingTier:
    try:
        return LLM_BRIEFING_TIERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown LLM briefing tier: {name}") from exc


def build_llm_briefing_input(
    *,
    records: Sequence[EventMemoryRecord],
    clusters: Sequence[DedupeCluster],
    comparison: EventMemoryComparison,
    run_date: str,
    tier: LlmBriefingTier,
    max_event_packets: int = DEFAULT_MAX_LLM_EVENT_PACKETS,
) -> dict[str, object]:
    tracked = ticker_lookup()
    cluster_by_id = {
        cluster.cluster_id: cluster
        for cluster in clusters
        if cluster.cluster_id
    }
    matches_by_index = {
        int(match["current_record_index"]): match
        for match in comparison.event_identity_matches
        if isinstance(match.get("current_record_index"), int)
    }
    packets = []
    for index, record in enumerate(records):
        cluster = cluster_by_id.get(record.cluster_id)
        article = cluster.canonical_article if cluster else None
        quality = (
            assess_article_source(article)
            if article is not None
            else None
        )
        match = matches_by_index.get(index, {})
        ticker = tracked.get(record.ticker)
        event_status = str(match.get("category") or "history_building")
        uncertainty = _packet_uncertainty(
            record,
            quality_tier=quality.tier if quality else None,
            event_status=event_status,
            prior_available=comparison.prior_run_available,
        )
        packets.append(
            {
                "ticker": record.ticker,
                "company": record.company,
                "portfolio_watchlist_status": (
                    ticker.group if ticker is not None else "untracked"
                ),
                "event_type": record.event_type,
                "article_type": record.article_type,
                "new_repeated_status": event_status,
                "similarity_score": round(
                    float(match.get("title_similarity") or 0.0),
                    4,
                ),
                "days_active": _days_active(
                    record.published_at,
                    run_date=run_date,
                ),
                "sentiment_score": record.internal_sentiment,
                "sentiment_change": _sentiment_change(
                    comparison,
                    record.ticker,
                ),
                "source_quality": {
                    "tier": quality.tier if quality else None,
                    "label": (
                        quality.label
                        if quality
                        else "not_classified"
                    ),
                },
                "source_count": cluster.source_count if cluster else 1,
                "source_provider": record.source_provider,
                "source_family": record.source_family,
                "top_titles": _cluster_titles(record, cluster),
                "urls": _cluster_urls(record, cluster),
                "uncertainty": uncertainty,
            }
        )
    packets.sort(key=_packet_sort_key)
    bounded_packets = packets[: max(1, int(max_event_packets))]
    ticker_packets = _build_ticker_packets(bounded_packets)
    return {
        "briefing_type": "portfolio_watchlist_market_intelligence",
        "run_date": run_date,
        "tier": tier.name,
        "model": tier.model,
        "purpose": tier.purpose,
        "instructions": [
            "Explain what materially changed and why it matters.",
            "Separate new events from repeated coverage.",
            "Use only supplied evidence and surface uncertainty.",
            "Do not provide buy, sell, allocation, price-target, or trading advice.",
        ],
        "no_buy_sell_advice_instruction": (
            "This is market intelligence, not investment advice. Do not "
            "recommend buying, selling, holding, sizing, or trading securities."
        ),
        "history": {
            "status": comparison.history_status,
            "lookback_days": comparison.event_memory_lookback_days,
            "prior_runs_considered": len(
                comparison.prior_runs_considered
            ),
            "exact_repeats": (
                comparison.exact_repeated_events_from_prior_run
            ),
            "fuzzy_repeats": (
                comparison.fuzzy_repeated_events_from_prior_run
            ),
            "likely_new_events": comparison.new_events_since_prior_run,
        },
        "event_packet_count": len(bounded_packets),
        "event_packets_truncated": len(packets) > len(bounded_packets),
        "event_packets": bounded_packets,
        "ticker_packet_count": len(ticker_packets),
        "ticker_packets": ticker_packets,
    }


def load_llm_briefing_run(run_dir: str | Path) -> LlmBriefingRunData:
    directory = Path(run_dir)
    if not directory.is_dir():
        raise LlmBriefingArtifactError(
            f"LLM briefing source run directory not found: {directory}"
        )
    event_path = directory / "event_memory_daily.json"
    rows = _read_json_list(event_path, label="event memory")
    if not rows:
        raise LlmBriefingArtifactError(
            f"LLM briefing source run has no event records: {directory}"
        )
    try:
        records = tuple(
            EventMemoryRecord(
                **{
                    field_name: row.get(field_name)
                    for field_name in EventMemoryRecord.__dataclass_fields__
                }
            )
            for row in rows
        )
    except (TypeError, ValueError) as exc:
        raise LlmBriefingArtifactError(
            f"LLM briefing event memory is invalid: {event_path}"
        ) from exc

    clusters = _load_artifact_clusters(
        directory / "dedupe_clusters.json",
        records,
    )
    comparison = _load_artifact_comparison(
        directory / "event_memory_comparison.json",
        records,
    )
    run_date = str(records[0].run_date or directory.name)
    return LlmBriefingRunData(
        source_run_dir=str(directory),
        run_date=run_date,
        records=records,
        clusters=clusters,
        comparison=comparison,
    )


def find_latest_populated_llm_run(
    artifacts_dir: str | Path,
) -> Path:
    base = Path(artifacts_dir)
    runs_dir = base if base.name == "runs" else base / "runs"
    if not runs_dir.is_dir():
        raise LlmBriefingArtifactError(
            f"No populated LLM briefing run found under: {runs_dir}"
        )
    candidates = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        event_path = candidate / "event_memory_daily.json"
        try:
            rows = _read_json_list(event_path, label="event memory")
        except LlmBriefingArtifactError:
            continue
        if rows:
            return candidate
    raise LlmBriefingArtifactError(
        f"No populated LLM briefing run found under: {runs_dir}"
    )


def estimate_llm_briefing_cost(
    input_payload: Mapping[str, object],
    tier: LlmBriefingTier,
) -> LlmBriefingEstimate:
    serialized = json.dumps(
        input_payload,
        separators=(",", ":"),
        sort_keys=True,
    )
    estimated_input_tokens = estimate_text_tokens(serialized)
    estimated_cost = (
        estimated_input_tokens
        * tier.input_cost_per_million_tokens
        / 1_000_000
        + tier.max_output_tokens
        * tier.output_cost_per_million_tokens
        / 1_000_000
    )
    blocked_reason = None
    if estimated_input_tokens > tier.max_input_tokens:
        blocked_reason = "estimated_input_exceeds_tier_token_limit"
    if estimated_cost > tier.cost_cap_usd:
        blocked_reason = "estimated_cost_exceeds_tier_cap"
    return LlmBriefingEstimate(
        estimated_input_tokens=estimated_input_tokens,
        reserved_output_tokens=tier.max_output_tokens,
        estimated_cost_usd=round(estimated_cost, 6),
        cost_cap_usd=tier.cost_cap_usd,
        blocked=blocked_reason is not None,
        blocked_reason=blocked_reason,
    )


def estimate_text_tokens(text: str) -> int:
    """Use a conservative local estimate without calling a tokenizer API."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def briefing_response_schema() -> dict[str, object]:
    named_item = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "summary": {"type": "string"},
            "evidence_indexes": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["ticker", "summary", "evidence_indexes"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "portfolio_pulse": {"type": "string"},
            "holdings_with_meaningful_change": {
                "type": "array",
                "items": named_item,
            },
            "watchlist_activations": {
                "type": "array",
                "items": named_item,
            },
            "top_monitor_names": {
                "type": "array",
                "items": named_item,
            },
            "repeated_vs_new_events": {"type": "string"},
            "uncertainty_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["ticker", "title", "url", "reason"],
                    "additionalProperties": False,
                },
            },
            "no_buy_sell_advice": {"type": "string"},
        },
        "required": [
            "portfolio_pulse",
            "holdings_with_meaningful_change",
            "watchlist_activations",
            "top_monitor_names",
            "repeated_vs_new_events",
            "uncertainty_notes",
            "evidence",
            "no_buy_sell_advice",
        ],
        "additionalProperties": False,
    }


def write_llm_briefing_artifacts(
    *,
    output_dir: str | Path,
    input_payload: Mapping[str, object],
    preview_payload: Mapping[str, object],
) -> tuple[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "llm_briefing_input.json"
    preview_path = directory / "llm_briefing_preview.json"
    input_path.write_text(
        json.dumps(input_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    preview_path.write_text(
        json.dumps(preview_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(input_path), str(preview_path)


def _packet_uncertainty(
    record: EventMemoryRecord,
    *,
    quality_tier: int | None,
    event_status: str,
    prior_available: bool,
) -> list[str]:
    reasons = []
    if record.ticker_match_confidence < 0.7:
        reasons.append("ticker_match_below_high_confidence")
    if record.extraction_basis != "full_text":
        reasons.append(f"{record.extraction_basis}_based_sentiment")
    if quality_tier is None or quality_tier >= 3:
        reasons.append("source_quality_not_high")
    if event_status == "fuzzy_event_repeat":
        reasons.append("fuzzy_cross_run_identity")
    if not prior_available:
        reasons.append("history_building")
    if record.event_type in {"", "unknown"}:
        reasons.append("event_type_unknown")
    return reasons


def _cluster_titles(
    record: EventMemoryRecord,
    cluster: DedupeCluster | None,
) -> list[str]:
    candidates = [record.event_title]
    if cluster is not None:
        candidates.extend(link.title for link in cluster.supporting_links)
    return _unique_nonempty(candidates, limit=3)


def _cluster_urls(
    record: EventMemoryRecord,
    cluster: DedupeCluster | None,
) -> list[str]:
    candidates = [record.canonical_url]
    if cluster is not None:
        candidates.extend(link.url for link in cluster.supporting_links)
    return _unique_nonempty(candidates, limit=5)


def _unique_nonempty(
    values: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _sentiment_change(
    comparison: EventMemoryComparison,
    ticker: str,
) -> float | None:
    values = comparison.sentiment_change_since_prior_run.get(ticker)
    if not values:
        return None
    return round(float(values.get("change") or 0.0), 4)


def _days_active(published_at: str | None, *, run_date: str) -> int | None:
    published_date = _parse_date(published_at)
    current_date = _parse_date(run_date)
    if published_date is None or current_date is None:
        return None
    return max(0, (current_date - published_date).days)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    normalized = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        try:
            return date.fromisoformat(normalized[:10])
        except ValueError:
            return None


def _packet_sort_key(packet: Mapping[str, object]) -> tuple[object, ...]:
    status_rank = {
        "likely_new_event": 0,
        "fuzzy_event_repeat": 1,
        "exact_url_repeat": 2,
        "history_building": 3,
    }
    group_rank = {
        "portfolio": 0,
        "watchlist": 1,
        "untracked": 2,
    }
    return (
        group_rank.get(
            str(packet.get("portfolio_watchlist_status") or ""),
            3,
        ),
        status_rank.get(str(packet.get("new_repeated_status") or ""), 4),
        -abs(float(packet.get("sentiment_change") or 0.0)),
        -abs(float(packet.get("sentiment_score") or 0.0)),
        str(packet.get("ticker") or ""),
        str((packet.get("top_titles") or [""])[0]),
    )


def _build_ticker_packets(
    event_packets: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for index, packet in enumerate(event_packets):
        ticker = str(packet.get("ticker") or "").upper()
        if ticker:
            grouped.setdefault(ticker, []).append((index, packet))
    ticker_packets = []
    for ticker, indexed_packets in sorted(grouped.items()):
        first = indexed_packets[0][1]
        sentiments = [
            float(packet.get("sentiment_score") or 0.0)
            for _, packet in indexed_packets
        ]
        statuses = [
            str(packet.get("new_repeated_status") or "")
            for _, packet in indexed_packets
        ]
        ticker_packets.append(
            {
                "ticker": ticker,
                "company": str(first.get("company") or ticker),
                "portfolio_watchlist_status": str(
                    first.get("portfolio_watchlist_status") or "untracked"
                ),
                "event_packet_indexes": [
                    index for index, _ in indexed_packets
                ],
                "event_count": len(indexed_packets),
                "new_event_count": statuses.count("likely_new_event"),
                "repeated_event_count": sum(
                    status
                    in {"exact_url_repeat", "fuzzy_event_repeat"}
                    for status in statuses
                ),
                "average_sentiment": round(
                    sum(sentiments) / len(sentiments),
                    4,
                ),
                "sentiment_change": first.get("sentiment_change"),
            }
        )
    return ticker_packets


def _read_json_list(path: Path, *, label: str) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise LlmBriefingArtifactError(
            f"Required {label} artifact is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LlmBriefingArtifactError(
            f"Required {label} artifact is unreadable: {path}"
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(row, Mapping) for row in payload
    ):
        raise LlmBriefingArtifactError(
            f"Required {label} artifact must be a JSON array: {path}"
        )
    return payload


def _read_json_object(
    path: Path,
    *,
    label: str,
) -> Mapping[str, object]:
    if not path.is_file():
        raise LlmBriefingArtifactError(
            f"Required {label} artifact is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LlmBriefingArtifactError(
            f"Required {label} artifact is unreadable: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LlmBriefingArtifactError(
            f"Required {label} artifact must be a JSON object: {path}"
        )
    return payload


def _load_artifact_clusters(
    path: Path,
    records: Sequence[EventMemoryRecord],
) -> tuple[DedupeCluster, ...]:
    if not path.is_file():
        return ()
    payload = _read_json_object(path, label="dedupe cluster")
    rows = payload.get("clusters")
    if not isinstance(rows, list):
        return ()
    record_by_cluster = {
        record.cluster_id: record
        for record in records
        if record.cluster_id
    }
    clusters = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "")
        record = record_by_cluster.get(cluster_id)
        title = str(
            row.get("title")
            or (record.event_title if record else "")
        ).strip()
        canonical_url = str(
            row.get("canonical_url")
            or (record.canonical_url if record else "")
        ).strip()
        if not title or not canonical_url:
            continue
        publisher_names = tuple(
            str(value)
            for value in (row.get("publisher_names") or ())
            if value
        )
        source_providers = tuple(
            str(value)
            for value in (row.get("source_providers") or ())
            if value
        )
        metadata = {
            "source_name": (
                publisher_names[0]
                if publisher_names
                else (record.source_provider if record else "")
            ),
            "provider": (
                record.source_provider
                if record
                else (source_providers[0] if source_providers else "")
            ),
            "source_family": (
                record.source_family if record else "unknown"
            ),
        }
        article = Article(
            canonical_url=canonical_url,
            title=title,
            published_at=(
                record.published_at
                if record
                else _optional_text(row.get("primary_published_at"))
            ),
            metadata=metadata,
        )
        supporting_links = tuple(
            SourceLink(
                title=str(link.get("title") or ""),
                url=str(link.get("url") or ""),
                publisher=_optional_text(link.get("publisher")),
                provider=_optional_text(link.get("provider")),
                published_at=_optional_text(link.get("published_at")),
            )
            for link in (row.get("supporting_links") or ())
            if isinstance(link, Mapping)
            and link.get("title")
            and link.get("url")
        )
        clusters.append(
            DedupeCluster(
                canonical_article=article,
                alternate_source_links=tuple(
                    str(value)
                    for value in (row.get("alternate_source_links") or ())
                    if value
                ),
                articles=(article,),
                duplicate_reasons=tuple(
                    str(value)
                    for value in (row.get("duplicate_reasons") or ())
                    if value
                ),
                primary_link=str(row.get("primary_link") or canonical_url),
                supporting_links=supporting_links,
                publisher_count=int(row.get("publisher_count") or 1),
                source_count=int(row.get("source_count") or 1),
                publisher_names=publisher_names,
                source_providers=source_providers,
                cluster_id=cluster_id,
                primary_ticker=_optional_text(row.get("primary_ticker")),
                matched_tickers=tuple(
                    str(value)
                    for value in (row.get("matched_tickers") or ())
                    if value
                ),
                related_tickers=tuple(
                    str(value)
                    for value in (row.get("related_tickers") or ())
                    if value
                ),
                event_type=str(row.get("event_type") or "unknown"),
                primary_article_id=str(
                    row.get("primary_article_id") or ""
                ),
                supporting_article_ids=tuple(
                    str(value)
                    for value in (
                        row.get("supporting_article_ids") or ()
                    )
                    if value
                ),
                supporting_publishers=tuple(
                    str(value)
                    for value in (
                        row.get("supporting_publishers") or ()
                    )
                    if value
                ),
                source_diversity=int(row.get("source_diversity") or 0),
                publisher_diversity=int(
                    row.get("publisher_diversity") or 0
                ),
            )
        )
    return tuple(clusters)


def _load_artifact_comparison(
    path: Path,
    records: Sequence[EventMemoryRecord],
) -> EventMemoryComparison:
    if not path.is_file():
        return _history_building_comparison(records)
    payload = _read_json_object(path, label="event memory comparison")
    matches = payload.get("event_identity_matches")
    if not isinstance(matches, list):
        matches = []
    return EventMemoryComparison(
        prior_run_available=bool(payload.get("prior_run_available")),
        history_status=str(
            payload.get("history_status") or "history_building"
        ),
        prior_run_id=_optional_text(payload.get("prior_run_id")),
        prior_run_date=_optional_text(payload.get("prior_run_date")),
        event_memory_lookback_days=int(
            payload.get("event_memory_lookback_days") or 3
        ),
        prior_runs_considered=tuple(
            row
            for row in (payload.get("prior_runs_considered") or ())
            if isinstance(row, Mapping)
        ),
        prior_event_records_considered=int(
            payload.get("prior_event_records_considered") or 0
        ),
        new_events_since_prior_run=int(
            payload.get("new_events_since_prior_run") or 0
        ),
        repeated_events_from_prior_run=int(
            payload.get("repeated_events_from_prior_run") or 0
        ),
        exact_repeated_events_from_prior_run=int(
            payload.get("exact_repeated_events_from_prior_run") or 0
        ),
        fuzzy_repeated_events_from_prior_run=int(
            payload.get("fuzzy_repeated_events_from_prior_run") or 0
        ),
        event_identity_method_counts=dict(
            payload.get("event_identity_method_counts") or {}
        ),
        event_similarity_threshold=float(
            payload.get("event_similarity_threshold") or 0.75
        ),
        event_identity_matches=tuple(
            row for row in matches if isinstance(row, Mapping)
        ),
        sentiment_change_since_prior_run=dict(
            payload.get("sentiment_change_since_prior_run") or {}
        ),
    )


def _history_building_comparison(
    records: Sequence[EventMemoryRecord],
) -> EventMemoryComparison:
    matches = tuple(
        {
            "category": "history_building",
            "current_record_index": index,
            "ticker": record.ticker,
            "title_similarity": 0.0,
        }
        for index, record in enumerate(records)
    )
    return EventMemoryComparison(
        prior_run_available=False,
        history_status="history_building",
        prior_run_id=None,
        prior_run_date=None,
        event_memory_lookback_days=3,
        prior_runs_considered=(),
        prior_event_records_considered=0,
        new_events_since_prior_run=0,
        repeated_events_from_prior_run=0,
        exact_repeated_events_from_prior_run=0,
        fuzzy_repeated_events_from_prior_run=0,
        event_identity_method_counts={
            "exact_url_repeat": 0,
            "fuzzy_event_repeat": 0,
            "likely_new_event": 0,
        },
        event_similarity_threshold=0.75,
        event_identity_matches=matches,
        sentiment_change_since_prior_run={},
    )


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
