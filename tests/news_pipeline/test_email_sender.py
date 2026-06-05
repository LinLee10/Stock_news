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
    (output_dir / "email_preview.html").write_text("<html><body>Daily report</body></html>", encoding="utf-8")
    (output_dir / "daily_report.html").write_text("<html><body>Report</body></html>", encoding="utf-8")
    (output_dir / "daily_report.md").write_text("# Report\n", encoding="utf-8")
    (output_dir / "report_contract.json").write_text("{}", encoding="utf-8")
    for filename in DEFAULT_EMAIL_ATTACHMENT_NAMES:
        (output_dir / filename).write_text("ticker,value\nNVDA,1\n", encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    unittest.main()
