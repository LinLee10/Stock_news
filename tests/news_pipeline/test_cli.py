import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from news_pipeline.cli import main
from news_pipeline.models import Article


class FakeProvider:
    def __init__(self, article):
        self.article = article

    def articles(self):
        return [self.article]


class CliTests(unittest.TestCase):
    def test_init_db_writes_under_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli("init-db", temp_dir)
            payload = json.loads(stdout)
            output_dir = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03"

            self.assertEqual(Path(payload["output_dir"]), output_dir)
            self.assertTrue((output_dir / "news_pipeline.sqlite3").exists())
            self.assertTrue((output_dir / "init_db.json").exists())
            self.assertIn("articles", payload["tables"])

    def test_validate_providers_does_not_print_secret_values(self):
        secret = "resend-secret-value"
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                "validate-providers",
                temp_dir,
                environ={"RESEND_API_KEY": secret},
            )
            output = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03" / "provider_validation.json"
            payload = output.read_text(encoding="utf-8")

            self.assertNotIn(secret, stdout)
            self.assertNotIn(secret, payload)
            self.assertIn('"key_state": "present"', payload)

    def test_collect_skips_paid_fake_providers_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                "collect",
                temp_dir,
                fake_providers={
                    "google_news_rss": FakeProvider(
                        Article(canonical_url="https://example.com/rss", title="RSS story")
                    ),
                    "marketaux": FakeProvider(
                        Article(canonical_url="https://example.com/paid", title="Paid story")
                    ),
                },
            )
            payload = json.loads(stdout)
            output = Path(payload["output"])
            stored = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["article_count"], 1)
            self.assertEqual(stored["articles"][0]["title"], "RSS story")
            self.assertNotIn("Paid story", output.read_text(encoding="utf-8"))

    def test_collect_includes_paid_fake_providers_when_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli(
                "collect",
                temp_dir,
                extra_args=["--enable-paid-apis"],
                fake_providers={
                    "marketaux": FakeProvider(
                        Article(canonical_url="https://example.com/paid", title="Paid story")
                    ),
                },
            )

            self.assertEqual(json.loads(stdout)["article_count"], 1)

    def test_pipeline_commands_write_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_providers = {
                "google_news_rss": FakeProvider(
                    Article(
                        canonical_url="https://example.com/story?utm_source=rss",
                        title="Nvidia beats estimates",
                        snippet="Nvidia reported strong growth.",
                    )
                )
            }

            for command, filename in [
                ("extract", "extractions.json"),
                ("dedup", "dedupe_clusters.json"),
                ("score", "sentiment_scores.json"),
                ("report", "report_contract.json"),
                ("dry-run-daily", "dry_run_daily.json"),
            ]:
                _run_cli(command, temp_dir, fake_providers=fake_providers)
                output = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03" / filename
                self.assertTrue(output.exists(), command)

    def test_dry_run_daily_never_enables_email_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = _run_cli("dry-run-daily", temp_dir)
            payload = json.loads(stdout)

            self.assertEqual(payload["status"], "dry_run_complete")
            self.assertEqual(payload["email_sending"], "disabled")
            self.assertFalse(payload["email_send_enabled"])
            self.assertFalse(payload["paid_apis_enabled"])


def _run_cli(
    command,
    temp_dir,
    *,
    extra_args=None,
    fake_providers=None,
    environ=None,
):
    argv = [
        command,
        "--run-date",
        "2026-06-03",
        "--artifacts-dir",
        str(Path(temp_dir) / "artifacts"),
    ]
    if extra_args:
        argv.extend(extra_args)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(argv, fake_providers=fake_providers, environ=environ or {})
    assert exit_code == 0
    return stdout.getvalue().strip()


if __name__ == "__main__":
    unittest.main()
