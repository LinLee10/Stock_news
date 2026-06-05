"""Article extraction with lazy optional extractors and local HTML fallback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from html.parser import HTMLParser
import importlib.util
import json
from typing import Any


EXTRACTION_BASES = {"full_text", "snippet", "title", "failed"}


@dataclass(frozen=True)
class ExtractionResult:
    title: str | None
    main_text: str
    publish_date: str | None
    author: str | None
    extraction_status: str
    extraction_basis: str
    extraction_error: str | None = None
    extractor: str = "none"
    url: str | None = None

    @property
    def text(self) -> str:
        return self.main_text

    @property
    def basis(self) -> str:
        return self.extraction_basis

    @property
    def error(self) -> str | None:
        return self.extraction_error

    @property
    def extraction_method_used(self) -> str:
        return self.extractor

    @property
    def extraction_failure_reason(self) -> str | None:
        return self.extraction_error


def extract_article(
    *,
    raw_html: str,
    url: str,
    title: str | None = None,
    snippet: str | None = None,
) -> ExtractionResult:
    """Extract article fields from provided HTML without fetching a URL."""
    errors: list[str] = []
    try:
        result = _extract_with_trafilatura(raw_html=raw_html, url=url)
    except Exception as exc:
        errors.append(f"trafilatura: {exc.__class__.__name__}")
    else:
        if result and result.main_text.strip():
            return _with_metadata_fallback(result=result, raw_html=raw_html, url=url, title=title)
        if result and result.extraction_error:
            errors.append(f"trafilatura: {result.extraction_error}")

    parsed = _extract_with_html_parser(raw_html=raw_html, url=url)
    if parsed.main_text.strip():
        return parsed

    errors.append("no_article_body")
    fallback = choose_extraction_basis(full_text=None, snippet=snippet, title=title or parsed.title)
    if fallback.extraction_basis == "failed":
        return ExtractionResult(
            title=title or parsed.title,
            main_text="",
            publish_date=parsed.publish_date,
            author=parsed.author,
            extraction_status="failed",
            extraction_basis="failed",
            extraction_error="; ".join(_unique(errors)) or "no_extractable_text",
            extractor="fallback",
            url=url,
        )

    return ExtractionResult(
        title=title or parsed.title,
        main_text=fallback.main_text,
        publish_date=parsed.publish_date,
        author=parsed.author,
        extraction_status="fallback",
        extraction_basis=fallback.extraction_basis,
        extraction_error="; ".join(_unique(errors)) or None,
        extractor="fallback",
        url=url,
    )


def extraction_dependency_status() -> dict[str, bool]:
    """Return optional extractor availability without importing secrets or fetching."""
    return {
        "trafilatura_available": importlib.util.find_spec("trafilatura") is not None,
        "newspaper3k_available": _newspaper3k_available(),
        "internal_parser_available": True,
    }


def choose_extraction_basis(
    *,
    full_text: str | None = None,
    snippet: str | None = None,
    title: str | None = None,
) -> ExtractionResult:
    """Choose the best available text basis without fetching content."""
    if full_text and full_text.strip():
        return ExtractionResult(
            title=title,
            main_text=full_text.strip(),
            publish_date=None,
            author=None,
            extraction_status="success",
            extraction_basis="full_text",
            extractor="provided",
        )
    if snippet and snippet.strip():
        return ExtractionResult(
            title=title,
            main_text=snippet.strip(),
            publish_date=None,
            author=None,
            extraction_status="fallback",
            extraction_basis="snippet",
            extractor="provided",
        )
    if title and title.strip():
        return ExtractionResult(
            title=title.strip(),
            main_text=title.strip(),
            publish_date=None,
            author=None,
            extraction_status="fallback",
            extraction_basis="title",
            extractor="provided",
        )
    return ExtractionResult(
        title=None,
        main_text="",
        publish_date=None,
        author=None,
        extraction_status="failed",
        extraction_basis="failed",
        extraction_error="no_text_available",
        extractor="provided",
    )


def _extract_with_trafilatura(*, raw_html: str, url: str) -> ExtractionResult | None:
    if importlib.util.find_spec("trafilatura") is None:
        return None
    try:
        import trafilatura  # type: ignore
    except ImportError:
        return None

    extracted = trafilatura.extract(
        raw_html,
        url=url,
        output_format="json",
        include_comments=False,
        include_tables=False,
    )
    if not extracted:
        return None

    data: dict[str, Any]
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError:
        data = {"text": extracted}

    text = (data.get("text") or "").strip()
    if not text:
        return None
    return ExtractionResult(
        title=data.get("title"),
        main_text=text,
        publish_date=data.get("date"),
        author=data.get("author"),
        extraction_status="success",
        extraction_basis="full_text",
        extractor="trafilatura",
        url=url,
    )


def _with_metadata_fallback(
    *,
    result: ExtractionResult,
    raw_html: str,
    url: str,
    title: str | None,
) -> ExtractionResult:
    if result.title and result.publish_date and result.author:
        return result

    parsed = _extract_with_html_parser(raw_html=raw_html, url=url)
    return replace(
        result,
        title=result.title or _first_present(title, parsed.title),
        publish_date=result.publish_date or parsed.publish_date,
        author=result.author or parsed.author,
    )


def _extract_with_html_parser(*, raw_html: str, url: str) -> ExtractionResult:
    parser = _ArticleHTMLParser()
    parser.feed(raw_html or "")
    text = "\n\n".join(paragraph for paragraph in parser.paragraphs if paragraph.strip()).strip()
    return ExtractionResult(
        title=parser.title,
        main_text=text,
        publish_date=parser.publish_date,
        author=parser.author,
        extraction_status="success" if text else "failed",
        extraction_basis="full_text" if text else "failed",
        extraction_error=None if text else "no_extractable_text",
        extractor="html_parser",
        url=url,
    )


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: str | None = None
        self.publish_date: str | None = None
        self.author: str | None = None
        self.paragraphs: list[str] = []
        self._in_title = False
        self._in_paragraph = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        if tag == "title":
            self._in_title = True
            self._buffer = []
        elif tag == "p":
            self._in_paragraph = True
            self._buffer = []
        elif tag == "meta":
            name = (attrs_dict.get("name") or attrs_dict.get("property") or "").lower()
            content = attrs_dict.get("content")
            if not content:
                return
            if name in {"article:published_time", "date", "dc.date", "pubdate"} and not self.publish_date:
                self.publish_date = content.strip()
            elif name in {"author", "article:author", "dc.creator"} and not self.author:
                self.author = content.strip()
            elif name in {"og:title", "twitter:title"} and not self.title:
                self.title = content.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._in_title:
            title = _collapse_ws(" ".join(self._buffer))
            if title:
                self.title = self.title or title
            self._in_title = False
            self._buffer = []
        elif tag == "p" and self._in_paragraph:
            paragraph = _collapse_ws(" ".join(self._buffer))
            if paragraph:
                self.paragraphs.append(paragraph)
            self._in_paragraph = False
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_title or self._in_paragraph:
            self._buffer.append(data)


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _first_present(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def _newspaper3k_available() -> bool:
    if importlib.util.find_spec("newspaper") is None:
        return False
    try:
        from newspaper import Article as _NewspaperArticle  # noqa: F401
    except Exception:
        return False
    return True


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return unique_values
