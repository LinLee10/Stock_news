"""Minimal chart attachment writers for report contracts."""

from __future__ import annotations

from pathlib import Path


def write_placeholder_chart(path: str | Path, title: str, labels: list[str]) -> str:
    """Write a tiny deterministic SVG placeholder under the requested path."""
    chart_path = Path(path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    escaped_title = _escape_xml(title)
    escaped_labels = ", ".join(_escape_xml(label) for label in labels) or "No data"
    chart_path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="180">',
                f'  <title>{escaped_title}</title>',
                '  <rect width="640" height="180" fill="#ffffff" stroke="#222222"/>',
                f'  <text x="24" y="48" font-size="20">{escaped_title}</text>',
                f'  <text x="24" y="92" font-size="14">{escaped_labels}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    return str(chart_path)


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
