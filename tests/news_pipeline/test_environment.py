import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from news_pipeline.cli import main


class CliEnvironmentTests(unittest.TestCase):
    def test_existing_environment_variables_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / ".env.local").write_text(
                "NYT_API_KEY=dotenv-test-value\n",
                encoding="utf-8",
            )

            with patch("news_pipeline.environment.PROJECT_ROOT", project_root):
                with patch.dict(
                    os.environ,
                    {"NYT_API_KEY": "shell-test-value"},
                    clear=True,
                ):
                    payload = _run_init_db(project_root)
                    self.assertEqual(os.environ["NYT_API_KEY"], "shell-test-value")

            self.assertEqual(payload["environment"]["local_dotenv_status"], "loaded")
            self.assertEqual(payload["environment"]["variables"]["NYT_API_KEY"], "set")

    def test_project_local_dotenv_is_loaded_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / ".env.local").write_text(
                "GNEWS_KEY=dotenv-test-value\n",
                encoding="utf-8",
            )

            with patch("news_pipeline.environment.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {}, clear=True):
                    payload = _run_init_db(project_root)
                    self.assertEqual(os.environ["GNEWS_KEY"], "dotenv-test-value")

            self.assertEqual(payload["environment"]["local_dotenv_status"], "loaded")
            self.assertEqual(payload["environment"]["variables"]["GNEWS_KEY"], "set")

    def test_missing_project_local_dotenv_does_not_crash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with patch("news_pipeline.environment.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {}, clear=True):
                    payload = _run_init_db(project_root)

            self.assertEqual(payload["environment"]["local_dotenv_status"], "missing")
            self.assertTrue(
                all(
                    state == "missing"
                    for state in payload["environment"]["variables"].values()
                )
            )

    def test_secret_values_are_never_printed_or_written_to_diagnostics(self):
        secret = "local-dotenv-secret-marker"
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / ".env.local").write_text(
                f"SMTP_PASS={secret}\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("news_pipeline.environment.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {}, clear=True):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exit_code = main(
                            [
                                "init-db",
                                "--run-date",
                                "2026-06-03",
                                "--artifacts-dir",
                                str(project_root / "artifacts"),
                            ]
                        )

            output_dir = project_root / "artifacts" / "runs" / "2026-06-03"
            diagnostic_artifact = (output_dir / "init_db.json").read_text(
                encoding="utf-8"
            )
            self.assertEqual(exit_code, 0)
            self.assertNotIn(secret, stdout.getvalue())
            self.assertNotIn(secret, stderr.getvalue())
            self.assertNotIn(secret, diagnostic_artifact)
            self.assertEqual(
                json.loads(stdout.getvalue())["environment"]["variables"]["SMTP_PASS"],
                "set",
            )


def _run_init_db(project_root: Path) -> dict[str, object]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(
            [
                "init-db",
                "--run-date",
                "2026-06-03",
                "--artifacts-dir",
                str(project_root / "artifacts"),
            ]
        )
    assert exit_code == 0
    return json.loads(stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
