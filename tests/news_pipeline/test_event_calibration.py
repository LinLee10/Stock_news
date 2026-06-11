import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from news_pipeline.cli import main
from news_pipeline.event_calibration import (
    CalibrationMetric,
    EventCalibrationError,
    calibrate_labeled_event_pairs,
    recommend_threshold,
    write_event_match_calibration_artifacts,
)
from news_pipeline.event_memory import EVENT_SIMILARITY_THRESHOLD


class EventCalibrationTests(unittest.TestCase):
    def test_valid_labeled_csv_calibration_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "event_pair_review.csv"
            _write_labeled_csv(input_path)
            calibration = calibrate_labeled_event_pairs(input_path)
            json_path, csv_path = write_event_match_calibration_artifacts(
                calibration,
                output_dir=temp_dir,
            )

            payload = json.loads(
                Path(json_path).read_text(encoding="utf-8")
            )
            with Path(csv_path).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(calibration.metrics), 8)
        self.assertEqual(calibration.recommended_threshold, 0.8)
        self.assertEqual(calibration.recommended_metric.precision, 1.0)
        self.assertEqual(calibration.recommended_metric.recall, 1.0)
        self.assertEqual(calibration.recommended_metric.f1, 1.0)
        self.assertEqual(payload["recommended_threshold"], 0.8)
        self.assertFalse(payload["production_threshold_changed"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            set(rows[0]),
            {
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
            },
        )

    def test_blank_and_uncertain_labels_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "event_pair_review.csv"
            _write_labeled_csv(input_path)
            calibration = calibrate_labeled_event_pairs(input_path)

        self.assertEqual(calibration.labeled_pair_count, 4)
        self.assertEqual(calibration.ignored_pair_count, 2)
        self.assertTrue(
            all(
                metric.labeled_pair_count == 4
                and metric.ignored_pair_count == 2
                for metric in calibration.metrics
            )
        )

    def test_missing_required_columns_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "event_pair_review.csv"
            input_path.write_text(
                "similarity_score\n0.9\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                EventCalibrationError,
                "missing required columns: recommended_label",
            ):
                calibrate_labeled_event_pairs(input_path)

    def test_recommended_threshold_tie_breaks_toward_precision(self):
        lower_precision = _metric(
            threshold=0.7,
            precision=0.7,
            recall=0.9,
            f1=0.8,
        )
        higher_precision = _metric(
            threshold=0.8,
            precision=0.9,
            recall=0.7,
            f1=0.8,
        )

        recommended = recommend_threshold(
            (lower_precision, higher_precision)
        )

        self.assertEqual(recommended.threshold, 0.8)
        self.assertEqual(recommended.precision, 0.9)

    def test_calibration_does_not_change_production_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "event_pair_review.csv"
            _write_labeled_csv(input_path)
            calibrate_labeled_event_pairs(input_path)

        self.assertEqual(EVENT_SIMILARITY_THRESHOLD, 0.78)

    def test_cli_calibration_writes_requested_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "labeled_pairs.csv"
            _write_labeled_csv(input_path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "calibrate-event-matching",
                        "--run-date",
                        "2026-06-11",
                        "--artifacts-dir",
                        str(Path(temp_dir) / "artifacts"),
                        "--labeled-event-pairs",
                        str(input_path),
                    ],
                    environ={},
                )
            payload = json.loads(stdout.getvalue())
            calibration_json_exists = Path(
                payload["event_match_calibration_json"]
            ).exists()
            calibration_csv_exists = Path(
                payload["event_match_calibration_csv"]
            ).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "calibration_complete")
        self.assertEqual(payload["recommended_threshold"], 0.8)
        self.assertTrue(calibration_json_exists)
        self.assertTrue(calibration_csv_exists)
        self.assertFalse(payload["production_threshold_changed"])

    def test_cli_missing_columns_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "labeled_pairs.csv"
            input_path.write_text(
                "recommended_label\nsame_event\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "calibrate-event-matching",
                        "--run-date",
                        "2026-06-11",
                        "--artifacts-dir",
                        str(Path(temp_dir) / "artifacts"),
                        "--labeled-event-pairs",
                        str(input_path),
                    ],
                    environ={},
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "error")
        self.assertIn(
            "missing required columns: similarity_score",
            payload["reason"],
        )

    def test_context_docs_describe_calibration_and_major_phase_summary(self):
        repo_root = Path(__file__).resolve().parents[2]
        context = (
            repo_root / "docs" / "NEWS_PIPELINE_CONTEXT.md"
        ).read_text(encoding="utf-8")
        workflow = (
            repo_root / "docs" / "CODEX_WORKFLOW.md"
        ).read_text(encoding="utf-8")

        self.assertIn("calibrate-event-matching", context)
        self.assertIn("event_match_calibration.json", context)
        self.assertIn("Tests and validation", workflow)
        self.assertIn("Safety check", workflow)


def _write_labeled_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("similarity_score", "recommended_label"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"similarity_score": "0.90", "recommended_label": "same_event"},
                {"similarity_score": "0.82", "recommended_label": "same_event"},
                {
                    "similarity_score": "0.75",
                    "recommended_label": "different_event",
                },
                {
                    "similarity_score": "0.65",
                    "recommended_label": "different_event",
                },
                {"similarity_score": "0.80", "recommended_label": ""},
                {"similarity_score": "0.70", "recommended_label": "uncertain"},
            )
        )


def _metric(
    *,
    threshold: float,
    precision: float,
    recall: float,
    f1: float,
) -> CalibrationMetric:
    return CalibrationMetric(
        threshold=threshold,
        true_positive=1,
        false_positive=0,
        true_negative=1,
        false_negative=0,
        precision=precision,
        recall=recall,
        f1=f1,
        labeled_pair_count=2,
        ignored_pair_count=0,
    )


if __name__ == "__main__":
    unittest.main()
