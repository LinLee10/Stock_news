import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from news_pipeline.cli import main
from news_pipeline.email_sender import DEFAULT_EMAIL_ATTACHMENT_NAMES, FakeEmailSender


class EmailSenderCliTests(unittest.TestCase):
    def test_dry_run_daily_never_uses_injected_email_sender(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout, exit_code = _run_cli(
                "dry-run-daily",
                temp_dir,
                extra_args=["--enable-email-send"],
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["email_sending"], "preview_only")
        self.assertFalse(fake_sender.sent_payloads)

    def test_send_daily_report_without_confirm_does_not_send(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                extra_args=["--to", "reader@example.com"],
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 0)
        self.assertFalse(payload["sent"])
        self.assertEqual(payload["send_mode"], "dry_run_no_send")
        self.assertEqual(payload["backend"], "local_preview")
        self.assertFalse(fake_sender.sent_payloads)

    def test_send_daily_report_without_recipient_refuses_to_send(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "refused")
        self.assertEqual(payload["reason"], "missing_recipient")
        self.assertFalse(payload["sent"])
        self.assertFalse(fake_sender.sent_payloads)

    def test_send_daily_report_refuses_missing_email_preview(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = _write_report_artifacts(temp_dir)
            (output_dir / "email_preview.html").unlink()
            stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                extra_args=["--to", "reader@example.com", "--confirm-send"],
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "refused")
        self.assertEqual(payload["reason"], "missing_email_preview_html")
        self.assertFalse(payload["sent"])
        self.assertFalse(fake_sender.sent_payloads)

    def test_send_daily_report_refuses_missing_report_artifacts(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = _write_report_artifacts(temp_dir)
            (output_dir / "report_contract.json").unlink()
            stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                extra_args=["--to", "reader@example.com", "--confirm-send"],
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "refused")
        self.assertIn("missing_report_artifacts", payload["reason"])
        self.assertFalse(payload["sent"])
        self.assertFalse(fake_sender.sent_payloads)

    def test_send_daily_report_enforces_attachment_size_limit(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                extra_args=[
                    "--to",
                    "reader@example.com",
                    "--confirm-send",
                    "--max-attachment-bytes",
                    "1",
                ],
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "refused")
        self.assertIn("attachment_too_large", payload["reason"])
        self.assertFalse(payload["sent"])
        self.assertFalse(fake_sender.sent_payloads)

    def test_send_daily_report_secret_values_are_never_printed(self):
        secret = "smtp-secret-value"
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            with patch("smtplib.SMTP", side_effect=RuntimeError(secret)):
                stdout, exit_code = _run_cli(
                    "send-daily-report",
                    temp_dir,
                    extra_args=["--to", "reader@example.com", "--confirm-send"],
                    environ={
                        "SMTP_HOST": "smtp.example.com",
                        "SMTP_USER": "sender@example.com",
                        "SMTP_PASS": secret,
                    },
                )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn(secret, stdout)

    def test_fake_sender_captures_intended_send_payload_without_network(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            with patch("smtplib.SMTP", side_effect=AssertionError("SMTP must not be called")):
                stdout, exit_code = _run_cli(
                    "send-daily-report",
                    temp_dir,
                    extra_args=["--to", "reader@example.com", "--confirm-send"],
                    email_sender=fake_sender,
                )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["sent"])
        self.assertEqual(payload["backend"], "fake")
        self.assertEqual(len(fake_sender.sent_payloads), 1)
        captured = fake_sender.sent_payloads[0]
        self.assertEqual(captured.to, "reader@example.com")
        self.assertIn("2026-06-03", captured.subject)
        self.assertEqual(
            [attachment.filename for attachment in captured.attachments],
            list(DEFAULT_EMAIL_ATTACHMENT_NAMES),
        )

    def test_real_send_render_does_not_use_preview_wording(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                extra_args=["--to", "reader@example.com", "--confirm-send"],
                email_sender=fake_sender,
            )
            payload = json.loads(stdout)

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["sent"])
        html = fake_sender.sent_payloads[0].html_body
        self.assertIn("Sent email report:", html)
        self.assertIn("delivered through the configured SMTP sender.", html)
        self.assertIn("<h2>Attachments</h2>", html)
        self.assertIn("Attached CSV files.", html)
        self.assertIn("portfolio_30d_sentiment.csv", html)
        self.assertNotIn("watchlist_sentiment.svg", html)
        self.assertNotIn("Preview only:", html)
        self.assertNotIn("future live email sender", html)

    def test_plain_text_fallback_includes_useful_summary(self):
        fake_sender = FakeEmailSender()
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_report_artifacts(temp_dir)
            _stdout, exit_code = _run_cli(
                "send-daily-report",
                temp_dir,
                extra_args=["--to", "reader@example.com", "--confirm-send"],
                email_sender=fake_sender,
            )

        self.assertEqual(exit_code, 0)
        text = fake_sender.sent_payloads[0].plain_text_body
        self.assertIn("Portfolio and Watchlist Market Briefing - 2026-06-03", text)
        self.assertIn("Portfolio coverage: 1 of 2 configured names.", text)
        self.assertIn("Watchlist coverage: 1 of 1 configured names.", text)
        self.assertIn("Top mention leaders: NVDA (3).", text)
        self.assertIn("portfolio_30d_sentiment.csv", text)
        self.assertIn("The HTML part of this email contains the full report.", text)


def _run_cli(
    command,
    temp_dir,
    *,
    extra_args=None,
    environ=None,
    email_sender=None,
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
        exit_code = main(argv, environ=environ or {}, email_sender=email_sender)
    return stdout.getvalue().strip(), exit_code


def _write_report_artifacts(temp_dir):
    output_dir = Path(temp_dir) / "artifacts" / "runs" / "2026-06-03"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "email_preview.html").write_text(
        "<html><body>"
        "<div><strong>Preview only:</strong> This file is a local email preview. No SMTP, Gmail, Resend, or other live email provider was contacted.</div>"
        "<h2>Intended Attachments</h2><p>These files would be attached.</p>"
        "<ul><li>portfolio_30d_sentiment.csv</li><li>watchlist_sentiment.svg</li></ul>"
        "</body></html>",
        encoding="utf-8",
    )
    (output_dir / "daily_report.html").write_text("<html><body>Report</body></html>", encoding="utf-8")
    (output_dir / "daily_report.md").write_text("# Report\n", encoding="utf-8")
    (output_dir / "report_contract.json").write_text(
        json.dumps(
            {
                "report": {
                    "daily_summary": "Daily report for 2026-06-03: NVDA leads mention volume.",
                    "portfolio_30d_sentiment_table": [
                        {"ticker": "NVDA", "article_count_30d": 2},
                        {"ticker": "ASML", "article_count_30d": 0},
                    ],
                    "watchlist_sentiment_table": [
                        {"ticker": "AMD", "article_count_30d": 1},
                    ],
                    "top_10_most_mentioned_table": [
                        {"ticker": "NVDA", "mentions": 3},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    for filename in DEFAULT_EMAIL_ATTACHMENT_NAMES:
        (output_dir / filename).write_text("ticker,value\nNVDA,1\n", encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    unittest.main()
