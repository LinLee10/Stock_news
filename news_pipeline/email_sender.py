"""Safe email sender abstractions for generated news pipeline reports."""

from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from html import escape
import json
import mimetypes
import os
from pathlib import Path
import re
import smtplib
from typing import Mapping, Protocol, Sequence


DEFAULT_EMAIL_ATTACHMENT_NAMES: tuple[str, ...] = (
    "portfolio_30d_sentiment.csv",
    "watchlist_sentiment.csv",
    "watchlist_next_close.csv",
    "mention_leaders_7d.csv",
    "top_10_most_mentioned.csv",
    "emerging_names.csv",
)
DEFAULT_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES = 20 * 1024 * 1024
PREVIEW_MODE = "preview_mode"
REAL_SEND_MODE = "real_send_mode"


@dataclass(frozen=True)
class AttachmentManifestItem:
    path: str
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class AttachmentManifest:
    attachments: tuple[AttachmentManifestItem, ...]
    total_size_bytes: int
    max_attachment_bytes: int
    max_total_attachment_bytes: int

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "attachments": [
                {
                    "path": item.path,
                    "filename": item.filename,
                    "size_bytes": item.size_bytes,
                }
                for item in self.attachments
            ],
            "total_size_bytes": self.total_size_bytes,
            "max_attachment_bytes": self.max_attachment_bytes,
            "max_total_attachment_bytes": self.max_total_attachment_bytes,
        }


@dataclass(frozen=True)
class EmailSendPayload:
    subject: str
    to: str
    html_body: str
    plain_text_body: str
    attachments: tuple[AttachmentManifestItem, ...]
    preview_path: str
    report_artifacts: tuple[str, ...]
    delivery_mode: str = PREVIEW_MODE

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "to": self.to,
            "preview_path": self.preview_path,
            "report_artifacts": list(self.report_artifacts),
            "delivery_mode": self.delivery_mode,
            "attachments": [
                {
                    "path": item.path,
                    "filename": item.filename,
                    "size_bytes": item.size_bytes,
                }
                for item in self.attachments
            ],
        }


@dataclass(frozen=True)
class EmailSendResult:
    status: str
    backend: str
    sent: bool
    recipient_count: int = 0
    message: str | None = None

    def as_safe_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "backend": self.backend,
            "sent": self.sent,
            "recipient_count": self.recipient_count,
        }
        if self.message:
            payload["message"] = self.message
        return payload


class EmailSender(Protocol):
    backend_name: str

    def send(self, payload: EmailSendPayload) -> EmailSendResult:
        """Send or capture an email payload."""


class LocalPreviewEmailSender:
    backend_name = "local_preview"

    def send(self, payload: EmailSendPayload) -> EmailSendResult:
        return EmailSendResult(
            status="dry_run_complete",
            backend=self.backend_name,
            sent=False,
            recipient_count=1,
            message="No email was sent.",
        )


class FakeEmailSender:
    backend_name = "fake"

    def __init__(self) -> None:
        self.sent_payloads: list[EmailSendPayload] = []

    def send(self, payload: EmailSendPayload) -> EmailSendResult:
        self.sent_payloads.append(payload)
        return EmailSendResult(
            status="sent",
            backend=self.backend_name,
            sent=True,
            recipient_count=1,
        )


class SmtpEmailSender:
    backend_name = "smtp"

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    def send(self, payload: EmailSendPayload) -> EmailSendResult:
        config = _smtp_config_from_environ(self._environ)
        msg = EmailMessage()
        msg["Subject"] = payload.subject
        msg["From"] = config.from_address
        msg["To"] = payload.to
        msg.set_content(payload.plain_text_body)
        msg.add_alternative(payload.html_body, subtype="html")

        for attachment in payload.attachments:
            data = Path(attachment.path).read_bytes()
            maintype, subtype = _mime_type(attachment.filename)
            msg.add_attachment(
                data,
                maintype=maintype,
                subtype=subtype,
                filename=attachment.filename,
            )

        smtp_cls = smtplib.SMTP_SSL if config.use_ssl else smtplib.SMTP
        with smtp_cls(config.host, config.port, timeout=config.timeout_seconds) as smtp:
            if config.starttls:
                smtp.starttls()
            smtp.login(config.username, config.password)
            smtp.send_message(msg)

        return EmailSendResult(
            status="sent",
            backend=self.backend_name,
            sent=True,
            recipient_count=1,
        )


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    from_address: str
    starttls: bool = True
    use_ssl: bool = False
    timeout_seconds: float = 30.0


class EmailSendError(ValueError):
    """Raised when a send request is not safe or complete."""


def build_report_email_payload(
    *,
    run_date: str,
    output_dir: Path,
    recipient: str,
    attachment_names: Sequence[str] = DEFAULT_EMAIL_ATTACHMENT_NAMES,
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
    max_total_attachment_bytes: int = DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES,
    delivery_mode: str = PREVIEW_MODE,
) -> tuple[EmailSendPayload, AttachmentManifest]:
    if not recipient:
        raise EmailSendError("missing_recipient")

    preview_path = output_dir / "email_preview.html"
    if not preview_path.exists():
        raise EmailSendError("missing_email_preview_html")

    report_artifacts = _required_report_artifacts(output_dir)
    missing_reports = [str(path) for path in report_artifacts if not path.exists()]
    if missing_reports:
        raise EmailSendError("missing_report_artifacts:" + ",".join(missing_reports))

    manifest = build_attachment_manifest(
        output_dir,
        attachment_names,
        max_attachment_bytes=max_attachment_bytes,
        max_total_attachment_bytes=max_total_attachment_bytes,
    )
    return (
        EmailSendPayload(
            subject=f"Daily Stock News Sentiment Report - {run_date}",
            to=recipient,
            html_body=_html_for_delivery_mode(
                preview_path.read_text(encoding="utf-8"),
                delivery_mode=delivery_mode,
                attachments=manifest.attachments,
            ),
            plain_text_body=_plain_text_body(
                run_date=run_date,
                output_dir=output_dir,
                attachments=manifest.attachments,
            ),
            attachments=manifest.attachments,
            preview_path=str(preview_path),
            report_artifacts=tuple(str(path) for path in report_artifacts),
            delivery_mode=delivery_mode,
        ),
        manifest,
    )


def build_attachment_manifest(
    output_dir: Path,
    attachment_names: Sequence[str],
    *,
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
    max_total_attachment_bytes: int = DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES,
) -> AttachmentManifest:
    items: list[AttachmentManifestItem] = []
    seen: set[Path] = set()
    for name in attachment_names:
        path = _attachment_path(output_dir, name)
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            raise EmailSendError(f"missing_attachment:{path}")
        if not path.is_file():
            raise EmailSendError(f"attachment_not_file:{path}")
        size = path.stat().st_size
        if size > max_attachment_bytes:
            raise EmailSendError(f"attachment_too_large:{path.name}")
        items.append(
            AttachmentManifestItem(
                path=str(path),
                filename=path.name,
                size_bytes=size,
            )
        )

    total_size = sum(item.size_bytes for item in items)
    if total_size > max_total_attachment_bytes:
        raise EmailSendError("attachments_total_too_large")

    return AttachmentManifest(
        attachments=tuple(items),
        total_size_bytes=total_size,
        max_attachment_bytes=max_attachment_bytes,
        max_total_attachment_bytes=max_total_attachment_bytes,
    )


def _required_report_artifacts(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "daily_report.html",
        output_dir / "daily_report.md",
        output_dir / "report_contract.json",
    )


def _html_for_delivery_mode(
    html: str,
    *,
    delivery_mode: str,
    attachments: tuple[AttachmentManifestItem, ...] = (),
) -> str:
    if delivery_mode != REAL_SEND_MODE:
        return html
    rendered = (
        html.replace(
            "<strong>Preview only:</strong> This file is a local email preview. No SMTP, Gmail, Resend, or other live email provider was contacted.",
            "<strong>Sent email report:</strong> delivered through the configured SMTP sender.",
        )
        .replace(
            "<strong>Preview only:</strong> no live email provider was contacted.",
            "<strong>Sent email report:</strong> delivered through the configured SMTP sender.",
        )
    )
    attachment_html = _render_real_attachment_section(attachments)
    return re.sub(
        r"\s*<h2>(?:Intended Attachments|Attached CSV Files|Attachments)</h2>\s*"
        r"<p>.*?</p>\s*<ul>.*?</ul>",
        "\n" + attachment_html,
        rendered,
        count=1,
        flags=re.DOTALL,
    )


def _render_real_attachment_section(attachments: tuple[AttachmentManifestItem, ...]) -> str:
    rows = ["    <h2>Attachments</h2>", "    <p>Attached CSV files.</p>", "    <ul>"]
    if attachments:
        rows.extend(f"      <li>{escape(item.filename)}</li>" for item in attachments)
    else:
        rows.append("      <li>No attachments were included.</li>")
    rows.append("    </ul>")
    return "\n".join(rows)


def _plain_text_body(
    *,
    run_date: str,
    output_dir: Path,
    attachments: tuple[AttachmentManifestItem, ...],
) -> str:
    report = _safe_report_contract(output_dir)
    summary = str(report.get("daily_summary") or f"Daily report for {run_date}.")
    portfolio_rows = report.get("portfolio_30d_sentiment_table") if isinstance(report.get("portfolio_30d_sentiment_table"), list) else []
    watchlist_rows = report.get("watchlist_sentiment_table") if isinstance(report.get("watchlist_sentiment_table"), list) else []
    top_mentions = report.get("top_10_most_mentioned_table") if isinstance(report.get("top_10_most_mentioned_table"), list) else []
    portfolio_covered = sum(1 for row in portfolio_rows if int(row.get("article_count_30d") or 0) > 0)
    watchlist_covered = sum(1 for row in watchlist_rows if int(row.get("article_count_30d") or 0) > 0)
    leaders = ", ".join(
        f"{row.get('ticker')} ({row.get('mentions')})"
        for row in top_mentions[:5]
        if row.get("ticker")
    ) or "none"
    attachment_names = ", ".join(item.filename for item in attachments) or "none"
    return "\n".join(
        [
            f"Daily Stock News Sentiment Report - {run_date}",
            "",
            summary,
            "",
            f"Portfolio coverage: {portfolio_covered} of {len(portfolio_rows)} configured names.",
            f"Watchlist coverage: {watchlist_covered} of {len(watchlist_rows)} configured names.",
            f"Top mention leaders: {leaders}.",
            f"Attachments: {attachment_names}.",
            "",
            "The HTML part of this email contains the full report.",
        ]
    )


def _safe_report_contract(output_dir: Path) -> dict[str, object]:
    path = output_dir / "report_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    report = payload.get("report") if isinstance(payload, dict) else {}
    return report if isinstance(report, dict) else {}


def _attachment_path(output_dir: Path, name: str) -> Path:
    base = output_dir.resolve()
    path = Path(name)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise EmailSendError(f"attachment_outside_output_dir:{name}") from exc
    return resolved


def _smtp_config_from_environ(environ: Mapping[str, str]) -> SmtpConfig:
    host = environ.get("SMTP_HOST", "")
    username = environ.get("SMTP_USER", "")
    password = environ.get("SMTP_PASS", "")
    if not host or not username or not password:
        raise EmailSendError("missing_smtp_credentials")

    port = int(environ.get("SMTP_PORT", "465" if _env_bool(environ.get("SMTP_USE_SSL")) else "587"))
    return SmtpConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_address=environ.get("SMTP_FROM") or username,
        starttls=_env_bool(environ.get("SMTP_STARTTLS"), default=not _env_bool(environ.get("SMTP_USE_SSL"))),
        use_ssl=_env_bool(environ.get("SMTP_USE_SSL")),
        timeout_seconds=float(environ.get("SMTP_TIMEOUT_SECONDS", "30")),
    )


def _env_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mime_type(filename: str) -> tuple[str, str]:
    guessed, _ = mimetypes.guess_type(filename)
    if not guessed:
        return "application", "octet-stream"
    maintype, subtype = guessed.split("/", 1)
    return maintype, subtype
