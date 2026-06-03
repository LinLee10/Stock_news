"""Configuration helpers for local-only news pipeline commands."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class NewsPipelineConfig:
    database_path: str = "news_pipeline.sqlite3"
    dry_run: bool = True
    email_enabled: bool = False

    @classmethod
    def from_env(cls) -> "NewsPipelineConfig":
        """Load non-secret local configuration without printing values."""
        return cls(
            database_path=os.getenv("NEWS_PIPELINE_DB", cls.database_path),
            dry_run=os.getenv("NEWS_PIPELINE_DRY_RUN", "1") != "0",
            email_enabled=os.getenv("NEWS_PIPELINE_EMAIL_ENABLED", "0") == "1",
        )
