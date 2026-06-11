"""Offline calibration metrics for manually labeled event-pair reviews."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Sequence


DEFAULT_CALIBRATION_THRESHOLD_MIN = 0.60
DEFAULT_CALIBRATION_THRESHOLD_MAX = 0.95
DEFAULT_CALIBRATION_THRESHOLD_STEP = 0.05
CALIBRATION_REQUIRED_COLUMNS = {
    "similarity_score",
    "recommended_label",
}
CALIBRATION_ACCEPTED_LABELS = {
    "",
    "same_event",
    "different_event",
    "uncertain",
}
CALIBRATION_METRIC_FIELDS = (
    "threshold",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "labeled_pair_count",
    "ignored_pair_count",
)


class EventCalibrationError(ValueError):
    """Raised when labeled calibration input is invalid."""


@dataclass(frozen=True)
class LabeledEventPair:
    similarity_score: float
    label: str


@dataclass(frozen=True)
class CalibrationMetric:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    labeled_pair_count: int
    ignored_pair_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventMatchCalibration:
    metrics: tuple[CalibrationMetric, ...]
    recommended_threshold: float
    recommended_metric: CalibrationMetric
    labeled_pair_count: int
    ignored_pair_count: int
    source_csv: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_csv": self.source_csv,
            "labeled_pair_count": self.labeled_pair_count,
            "ignored_pair_count": self.ignored_pair_count,
            "recommended_threshold": self.recommended_threshold,
            "recommended_metric": self.recommended_metric.as_dict(),
            "metrics": [metric.as_dict() for metric in self.metrics],
            "production_threshold_changed": False,
        }


def calibrate_labeled_event_pairs(
    csv_path: str | Path,
    *,
    threshold_min: float = DEFAULT_CALIBRATION_THRESHOLD_MIN,
    threshold_max: float = DEFAULT_CALIBRATION_THRESHOLD_MAX,
    threshold_step: float = DEFAULT_CALIBRATION_THRESHOLD_STEP,
) -> EventMatchCalibration:
    source_path = Path(csv_path)
    if not source_path.is_file():
        raise EventCalibrationError(
            f"labeled event-pair CSV not found: {source_path}"
        )

    pairs = _read_labeled_pairs(source_path)
    labeled_pairs = tuple(
        pair
        for pair in pairs
        if pair.label in {"same_event", "different_event"}
    )
    ignored_pair_count = len(pairs) - len(labeled_pairs)
    if not labeled_pairs:
        raise EventCalibrationError(
            "labeled event-pair CSV has no same_event or different_event rows"
        )

    thresholds = calibration_thresholds(
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )
    metrics = tuple(
        _metric_for_threshold(
            labeled_pairs,
            threshold=threshold,
            ignored_pair_count=ignored_pair_count,
        )
        for threshold in thresholds
    )
    recommended = recommend_threshold(metrics)
    return EventMatchCalibration(
        metrics=metrics,
        recommended_threshold=recommended.threshold,
        recommended_metric=recommended,
        labeled_pair_count=len(labeled_pairs),
        ignored_pair_count=ignored_pair_count,
        source_csv=str(source_path),
    )


def calibration_thresholds(
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, ...]:
    try:
        minimum = Decimal(str(threshold_min))
        maximum = Decimal(str(threshold_max))
        step = Decimal(str(threshold_step))
    except InvalidOperation as exc:
        raise EventCalibrationError(
            "calibration thresholds must be numeric"
        ) from exc
    if minimum < 0 or maximum > 1 or minimum > maximum:
        raise EventCalibrationError(
            "calibration threshold range must satisfy 0 <= min <= max <= 1"
        )
    if step <= 0:
        raise EventCalibrationError(
            "calibration threshold step must be greater than zero"
        )

    thresholds: list[float] = []
    current = minimum
    while current <= maximum:
        thresholds.append(round(float(current), 4))
        current += step
    return tuple(thresholds)


def recommend_threshold(
    metrics: Sequence[CalibrationMetric],
) -> CalibrationMetric:
    if not metrics:
        raise EventCalibrationError("no calibration metrics were generated")
    return max(
        metrics,
        key=lambda metric: (
            metric.f1,
            metric.precision,
            metric.threshold,
        ),
    )


def write_event_match_calibration_artifacts(
    calibration: EventMatchCalibration,
    *,
    output_dir: str | Path,
) -> tuple[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "event_match_calibration.json"
    csv_path = directory / "event_match_calibration.csv"
    json_path.write_text(
        json.dumps(calibration.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CALIBRATION_METRIC_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            metric.as_dict()
            for metric in calibration.metrics
        )
    return str(json_path), str(csv_path)


def _read_labeled_pairs(csv_path: Path) -> tuple[LabeledEventPair, ...]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(CALIBRATION_REQUIRED_COLUMNS - columns)
        if missing:
            raise EventCalibrationError(
                "labeled event-pair CSV missing required columns: "
                + ", ".join(missing)
            )
        pairs: list[LabeledEventPair] = []
        for row_number, row in enumerate(reader, start=2):
            label = str(row.get("recommended_label") or "").strip().lower()
            if label not in CALIBRATION_ACCEPTED_LABELS:
                raise EventCalibrationError(
                    f"invalid recommended_label at row {row_number}: {label}"
                )
            try:
                similarity = float(row.get("similarity_score") or "")
            except (TypeError, ValueError) as exc:
                raise EventCalibrationError(
                    f"invalid similarity_score at row {row_number}"
                ) from exc
            if not 0.0 <= similarity <= 1.0:
                raise EventCalibrationError(
                    f"similarity_score outside 0..1 at row {row_number}"
                )
            pairs.append(
                LabeledEventPair(
                    similarity_score=similarity,
                    label=label,
                )
            )
    return tuple(pairs)


def _metric_for_threshold(
    labeled_pairs: Sequence[LabeledEventPair],
    *,
    threshold: float,
    ignored_pair_count: int,
) -> CalibrationMetric:
    true_positive = false_positive = true_negative = false_negative = 0
    for pair in labeled_pairs:
        predicted_same = pair.similarity_score >= threshold
        actual_same = pair.label == "same_event"
        if predicted_same and actual_same:
            true_positive += 1
        elif predicted_same:
            false_positive += 1
        elif actual_same:
            false_negative += 1
        else:
            true_negative += 1

    precision = _safe_ratio(
        true_positive,
        true_positive + false_positive,
    )
    recall = _safe_ratio(
        true_positive,
        true_positive + false_negative,
    )
    f1 = _safe_ratio(
        2 * precision * recall,
        precision + recall,
    )
    return CalibrationMetric(
        threshold=threshold,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        labeled_pair_count=len(labeled_pairs),
        ignored_pair_count=ignored_pair_count,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
