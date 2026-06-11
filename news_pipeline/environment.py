"""Safe environment loading for news pipeline CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


EXPECTED_ENV_VARIABLES = (
    "ALPHA_VANTAGE_KEY",
    "NYT_API_KEY",
    "MARKETAUX_API_KEY",
    "GNEWS_KEY",
    "FINNHUB_KEY",
    "NEWSAPI_KEY",
    "SMTP_HOST",
    "SMTP_USER",
    "SMTP_PASS",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_cli_environment(
    environ: Mapping[str, str] | None = None,
    *,
    project_root: Path | None = None,
) -> tuple[Mapping[str, str], str]:
    """Load only the project-local dotenv file for normal CLI invocations."""
    if environ is not None:
        return environ, "not_loaded_injected_environment"

    dotenv_path = (project_root or PROJECT_ROOT) / ".env.local"
    if not dotenv_path.is_file():
        return os.environ, "missing"

    load_dotenv(dotenv_path=dotenv_path, override=False, verbose=False)
    return os.environ, "loaded"


def environment_status(environ: Mapping[str, str]) -> dict[str, str]:
    """Report presence without exposing environment variable values."""
    return {
        name: "set" if environ.get(name) else "missing"
        for name in EXPECTED_ENV_VARIABLES
    }
