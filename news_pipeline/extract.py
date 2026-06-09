"""Article extraction ensemble and deterministic quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import importlib.util
import json
import re
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


EXTRACTION_BASES = {"full_text", "snippet", "title", "failed"}
FULL_TEXT_GRADES = {"strong_full_text", "usable_full_text"}
SHELL_PHRASES = (
    "accept all cookies",
    "cookie preferences",
    "enable javascript",
    "sign in to continue",
    "subscribe to continue",
    "subscription required",
    "verify you are human",
    "access denied",
    "captcha",
)
BOILERPLATE_PHRASES = (
    "all rights reserved",
    "cookie policy",
    "privacy policy",
    "terms of use",
    "skip to content",
    "newsletter",
    "advertisement",
    "related articles",
    "follow us",
)
ARTICLE_VERBS = (
    "said",
    "reported",
    "announced",
    "expects",
    "forecast",
    "rose",
    "fell",
    "increased",
    "declined",
    "launched",
    "filed",
    "agreed",
)
IGNORED_TAGS = {"script", "style", "nav", "footer", "header", "form", "svg", "noscript", "aside"}


@dataclass(frozen=True)
class ExtractionAttempt:
    method_name: str
    text: str = ""
    text_length: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    success: bool = False
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "text_length": self.text_length,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "metadata": dict(self.metadata),
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ExtractionQuality:
    score: float
    grade: str
    reasons: tuple[str, ...]
    accepted_as_full_text: bool


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
    attempts: tuple[ExtractionAttempt, ...] = ()
    extraction_quality_score: float = 0.0
    extraction_quality_grade: str = "title_only"
    extraction_quality_reasons: tuple[str, ...] = ()
    accepted_as_full_text: bool = False
    paragraph_count: int = 0
    canonical_url: str | None = None
    og_url: str | None = None

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
    ticker_terms: tuple[str, ...] = (),
) -> ExtractionResult:
    """Run extractors in order and accept only quality-validated article text."""
    metadata = _parse_html(raw_html)
    source_title = _first_present(title, metadata.title)
    attempts: list[ExtractionAttempt] = []
    best_attempt: ExtractionAttempt | None = None
    best_quality: ExtractionQuality | None = None

    extractors: tuple[tuple[str, Callable[..., ExtractionAttempt | None]], ...] = (
        ("trafilatura_standard", _extract_with_trafilatura_standard),
        ("trafilatura_favor_recall", _extract_with_trafilatura_favor_recall),
        ("trafilatura_baseline", _extract_with_trafilatura_baseline),
        ("internal_article_parser", _extract_with_internal_parser),
        ("readability", _extract_with_readability),
        ("last_resort_text", _extract_with_last_resort_text),
    )
    for method_name, extractor in extractors:
        try:
            attempt = extractor(raw_html=raw_html, url=url)
        except Exception as exc:  # noqa: BLE001 - one optional extractor must not stop the ensemble.
            attempt = _failed_attempt(method_name, type(exc).__name__)
        if attempt is None:
            attempt = _failed_attempt(method_name, "dependency_unavailable")
        attempts.append(attempt)
        if not attempt.success or not attempt.text.strip():
            continue
        quality = score_extraction_quality(
            attempt.text,
            source_title=source_title,
            final_url=url,
            ticker_terms=ticker_terms,
            paragraph_count=attempt.paragraph_count,
        )
        if best_quality is None or quality.score > best_quality.score:
            best_attempt = attempt
            best_quality = quality
        if quality.accepted_as_full_text:
            attempt_metadata = attempt.metadata
            return ExtractionResult(
                title=_first_present(str(attempt_metadata.get("title") or ""), source_title),
                main_text=attempt.text.strip(),
                publish_date=_first_present(str(attempt_metadata.get("date") or ""), metadata.publish_date),
                author=_first_present(str(attempt_metadata.get("author") or ""), metadata.author),
                extraction_status="success",
                extraction_basis="full_text",
                extractor=attempt.method_name,
                url=url,
                attempts=tuple(attempts),
                extraction_quality_score=quality.score,
                extraction_quality_grade=quality.grade,
                extraction_quality_reasons=quality.reasons,
                accepted_as_full_text=True,
                paragraph_count=attempt.paragraph_count,
                canonical_url=metadata.canonical_url,
                og_url=metadata.og_url,
            )

    fallback = choose_extraction_basis(snippet=snippet, title=source_title)
    blocked = _is_wrapper_url(url) or bool(best_quality and best_quality.grade == "blocked_or_shell")
    if blocked:
        fallback_grade = "blocked_or_shell"
    elif best_attempt and best_attempt.text.strip():
        fallback_grade = "weak_text"
    elif fallback.extraction_basis == "snippet":
        fallback_grade = "snippet_only"
    else:
        fallback_grade = "title_only"
    reasons = list(best_quality.reasons if best_quality else ())
    if not reasons:
        reasons.append("no_usable_article_body")
    if fallback.extraction_basis == "snippet":
        reasons.append("snippet_fallback")
    elif fallback.extraction_basis == "title":
        reasons.append("title_fallback")
    else:
        reasons.append("no_text_available")
    return ExtractionResult(
        title=source_title,
        main_text=fallback.main_text,
        publish_date=metadata.publish_date,
        author=metadata.author,
        extraction_status="fallback" if fallback.extraction_basis != "failed" else "failed",
        extraction_basis=fallback.extraction_basis,
        extraction_error="; ".join(_unique(reasons)),
        extractor="fallback",
        url=url,
        attempts=tuple(attempts),
        extraction_quality_score=best_quality.score if best_quality else 0.0,
        extraction_quality_grade=fallback_grade,
        extraction_quality_reasons=tuple(_unique(reasons)),
        accepted_as_full_text=False,
        paragraph_count=best_attempt.paragraph_count if best_attempt else 0,
        canonical_url=metadata.canonical_url,
        og_url=metadata.og_url,
    )


def score_extraction_quality(
    text: str,
    *,
    source_title: str | None = None,
    final_url: str | None = None,
    ticker_terms: tuple[str, ...] = (),
    paragraph_count: int | None = None,
) -> ExtractionQuality:
    """Score whether extracted text resembles a usable publisher article."""
    clean = _collapse_ws(text)
    lowered = clean.lower()
    lines = [_collapse_ws(line) for line in text.splitlines() if _collapse_ws(line)]
    sentences = _sentences(clean)
    paragraphs = paragraph_count if paragraph_count is not None else len(lines)
    reasons: list[str] = []

    if _is_wrapper_url(final_url or ""):
        return ExtractionQuality(
            0.0,
            "blocked_or_shell",
            ("blocked_or_shell", "google_wrapper_unresolved"),
            False,
        )

    shell_hits = [phrase for phrase in SHELL_PHRASES if phrase in lowered]
    shell_ratio = sum(lowered.count(phrase) * len(phrase) for phrase in shell_hits) / max(1, len(clean))
    if shell_hits and (len(clean) < 500 or shell_ratio >= 0.15):
        rejection_reason = (
            "paywall_or_login"
            if any(term in lowered for term in ("sign in", "subscribe", "subscription"))
            else "blocked_or_shell"
        )
        return ExtractionQuality(
            0.0,
            "blocked_or_shell",
            tuple(["blocked_or_shell", rejection_reason, *shell_hits[:3]]),
            False,
        )

    score = 10.0
    length = len(clean)
    if length >= 1600:
        score += 30
        reasons.append("substantial_text_length")
    elif length >= 800:
        score += 24
        reasons.append("good_text_length")
    elif length >= 400:
        score += 16
        reasons.append("moderate_text_length")
    elif length >= 220:
        score += 8
        reasons.append("short_article_text")
    else:
        score -= 12
        reasons.append("too_short")

    if len(sentences) >= 10:
        score += 20
        reasons.append("many_article_sentences")
    elif len(sentences) >= 5:
        score += 14
        reasons.append("several_article_sentences")
    elif len(sentences) >= 3:
        score += 8
        reasons.append("limited_sentence_count")
    else:
        score -= 8
        reasons.append("too_few_sentences")

    if paragraphs >= 5:
        score += 10
        reasons.append("multiple_paragraphs")
    elif paragraphs >= 2:
        score += 6
        reasons.append("more_than_one_paragraph")
    else:
        reasons.append("single_paragraph")

    article_sentence_ratio = sum(_looks_like_article_sentence(sentence) for sentence in sentences) / max(1, len(sentences))
    if article_sentence_ratio >= 0.6:
        score += 12
        reasons.append("article_like_prose")
    elif article_sentence_ratio < 0.3:
        score -= 10
        reasons.append("limited_article_like_prose")

    normalized_terms = tuple(term.lower().strip() for term in ticker_terms if term and term.strip())
    if normalized_terms and any(_term_present(lowered, term) for term in normalized_terms):
        score += 12
        reasons.append("ticker_or_company_mentioned")
    elif normalized_terms:
        score -= 8
        reasons.append("missing_ticker_context")

    title_overlap = _title_overlap(source_title, clean)
    if title_overlap >= 0.45:
        score += 12
        reasons.append("strong_title_overlap")
    elif title_overlap >= 0.2:
        score += 7
        reasons.append("some_title_overlap")

    boilerplate_hits = sum(lowered.count(phrase) for phrase in BOILERPLATE_PHRASES)
    boilerplate_ratio = boilerplate_hits / max(1, len(sentences))
    if boilerplate_ratio >= 0.5:
        score -= 25
        reasons.append("boilerplate_heavy")
    elif boilerplate_hits:
        score -= min(10, boilerplate_hits * 3)
        reasons.append("boilerplate_detected")

    duplicate_ratio = _duplicate_line_ratio(lines)
    if duplicate_ratio >= 0.35:
        score -= 20
        reasons.append("high_duplicate_line_ratio")
    elif duplicate_ratio >= 0.15:
        score -= 8
        reasons.append("duplicate_lines_detected")

    score = round(max(0.0, min(100.0, score)), 2)
    if score >= 75:
        grade = "strong_full_text"
    elif score >= 52:
        grade = "usable_full_text"
    else:
        grade = "weak_text"
    return ExtractionQuality(score, grade, tuple(_unique(reasons)), grade in FULL_TEXT_GRADES)


def extraction_dependency_status() -> dict[str, bool]:
    """Return optional extractor availability without network access."""
    return {
        "trafilatura_available": importlib.util.find_spec("trafilatura") is not None,
        "readability_available": importlib.util.find_spec("readability") is not None,
        "newspaper3k_available": _newspaper3k_available(),
        "internal_parser_available": True,
    }


def extract_url_metadata(raw_html: str) -> dict[str, str | None]:
    """Return safe canonical and Open Graph URLs found in supplied HTML."""
    parsed = _parse_html(raw_html)
    return {
        "canonical_url": parsed.canonical_url,
        "og_url": parsed.og_url,
    }


def choose_extraction_basis(
    *,
    full_text: str | None = None,
    snippet: str | None = None,
    title: str | None = None,
) -> ExtractionResult:
    """Choose the best provided text basis without fetching content."""
    if full_text and full_text.strip():
        return ExtractionResult(
            title=title,
            main_text=full_text.strip(),
            publish_date=None,
            author=None,
            extraction_status="success",
            extraction_basis="full_text",
            extractor="provided",
            extraction_quality_score=100.0,
            extraction_quality_grade="strong_full_text",
            extraction_quality_reasons=("provided_full_text",),
            accepted_as_full_text=True,
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
            extraction_quality_grade="snippet_only",
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
            extraction_quality_grade="title_only",
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
        extraction_quality_grade="title_only",
    )


def _extract_with_trafilatura_standard(*, raw_html: str, url: str) -> ExtractionAttempt | None:
    return _trafilatura_extract(raw_html=raw_html, url=url, method_name="trafilatura_standard")


def _extract_with_trafilatura_favor_recall(*, raw_html: str, url: str) -> ExtractionAttempt | None:
    return _trafilatura_extract(
        raw_html=raw_html,
        url=url,
        method_name="trafilatura_favor_recall",
        favor_recall=True,
    )


def _extract_with_trafilatura_baseline(*, raw_html: str, url: str) -> ExtractionAttempt | None:
    trafilatura = _load_trafilatura()
    if trafilatura is None:
        return None
    baseline = getattr(trafilatura, "baseline", None)
    if baseline is None:
        return _failed_attempt("trafilatura_baseline", "baseline_unavailable")
    result = baseline(raw_html)
    text = ""
    if isinstance(result, str):
        text = result
    elif isinstance(result, (tuple, list)):
        text = next((value for value in reversed(result) if isinstance(value, str) and value.strip()), "")
    return _attempt("trafilatura_baseline", text)


def _trafilatura_extract(
    *,
    raw_html: str,
    url: str,
    method_name: str,
    favor_recall: bool = False,
) -> ExtractionAttempt | None:
    trafilatura = _load_trafilatura()
    if trafilatura is None:
        return None
    extracted = trafilatura.extract(
        raw_html,
        url=url,
        output_format="json",
        include_comments=False,
        include_tables=False,
        favor_recall=favor_recall,
    )
    if not extracted:
        return _failed_attempt(method_name, "no_extractable_text")
    try:
        data: dict[str, Any] = json.loads(extracted)
    except (json.JSONDecodeError, TypeError):
        data = {"text": str(extracted)}
    return _attempt(
        method_name,
        str(data.get("text") or ""),
        metadata={
            "title": data.get("title"),
            "date": data.get("date"),
            "author": data.get("author"),
        },
    )


def _extract_with_internal_parser(*, raw_html: str, url: str) -> ExtractionAttempt:
    del url
    parser = _parse_html(raw_html)
    text = "\n\n".join(parser.paragraphs)
    return _attempt(
        "internal_article_parser",
        text,
        paragraph_count=len(parser.paragraphs),
        metadata={"title": parser.title, "date": parser.publish_date, "author": parser.author},
    )


def _extract_with_readability(*, raw_html: str, url: str) -> ExtractionAttempt | None:
    del url
    if importlib.util.find_spec("readability") is None:
        return None
    try:
        from readability import Document  # type: ignore
    except ImportError:
        return None
    document = Document(raw_html)
    parser = _parse_html(document.summary())
    return _attempt(
        "readability",
        "\n\n".join(parser.paragraphs),
        paragraph_count=len(parser.paragraphs),
        metadata={"title": document.short_title()},
    )


def _extract_with_last_resort_text(*, raw_html: str, url: str) -> ExtractionAttempt:
    del url
    parser = _VisibleTextParser()
    parser.feed(raw_html or "")
    lines = [_collapse_ws(value) for value in parser.blocks if len(_collapse_ws(value)) >= 25]
    return _attempt("last_resort_text", "\n".join(lines), paragraph_count=len(lines))


def _attempt(
    method_name: str,
    text: str,
    *,
    paragraph_count: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ExtractionAttempt:
    cleaned = text.strip()
    return ExtractionAttempt(
        method_name=method_name,
        text=cleaned,
        text_length=len(cleaned),
        sentence_count=len(_sentences(cleaned)),
        paragraph_count=paragraph_count if paragraph_count is not None else _paragraph_count(cleaned),
        metadata=metadata or {},
        success=bool(cleaned),
        failure_reason=None if cleaned else "no_extractable_text",
    )


def _failed_attempt(method_name: str, reason: str) -> ExtractionAttempt:
    return ExtractionAttempt(method_name=method_name, failure_reason=reason)


def _load_trafilatura():
    if importlib.util.find_spec("trafilatura") is None:
        return None
    try:
        import trafilatura  # type: ignore
    except ImportError:
        return None
    return trafilatura


@dataclass
class _ParsedHtml:
    title: str | None = None
    publish_date: str | None = None
    author: str | None = None
    canonical_url: str | None = None
    og_url: str | None = None
    paragraphs: list[str] = field(default_factory=list)


def _parse_html(raw_html: str) -> _ParsedHtml:
    parser = _ArticleHTMLParser()
    parser.feed(raw_html or "")
    return _ParsedHtml(
        title=parser.title,
        publish_date=parser.publish_date,
        author=parser.author,
        canonical_url=parser.canonical_url,
        og_url=parser.og_url,
        paragraphs=parser.paragraphs,
    )


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: str | None = None
        self.publish_date: str | None = None
        self.author: str | None = None
        self.canonical_url: str | None = None
        self.og_url: str | None = None
        self.paragraphs: list[str] = []
        self._in_title = False
        self._in_paragraph = False
        self._buffer: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
            self._buffer = []
        elif tag == "p":
            self._in_paragraph = True
            self._buffer = []
        elif tag == "link" and "canonical" in (attrs_dict.get("rel") or "").lower():
            self.canonical_url = _safe_http_url(attrs_dict.get("href"))
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
            elif name == "og:url":
                self.og_url = _safe_http_url(content)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
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
        if not self._ignored_depth and (self._in_title or self._in_paragraph):
            self._buffer.append(data)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[str] = []
        self._buffer: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in IGNORED_TAGS:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        if tag.lower() in {"p", "div", "section", "article", "main", "li"} and self._buffer:
            self.blocks.append(" ".join(self._buffer))
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self._buffer.append(data.strip())


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) >= 20]


def _paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n|\n", text) if part.strip()])


def _looks_like_article_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    words = sentence.split()
    return len(words) >= 7 and (
        any(verb in lowered for verb in ARTICLE_VERBS)
        or bool(re.search(r"\b(?:company|stock|shares|revenue|market|business|investors?)\b", lowered))
    )


def _title_overlap(title: str | None, text: str) -> float:
    if not title:
        return 0.0
    title_tokens = set(re.findall(r"[a-z0-9]{3,}", title.lower()))
    if not title_tokens:
        return 0.0
    text_tokens = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return len(title_tokens & text_tokens) / len(title_tokens)


def _duplicate_line_ratio(lines: list[str]) -> float:
    normalized = [line.lower() for line in lines if len(line) >= 20]
    if len(normalized) < 2:
        return 0.0
    return 1.0 - (len(set(normalized)) / len(normalized))


def _term_present(text: str, term: str) -> bool:
    if not term:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def _is_wrapper_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"news.google.com", "www.news.google.com"}


def _safe_http_url(value: str | None) -> str | None:
    if not value:
        return None
    clean = value.strip()
    return clean if urlparse(clean).scheme in {"http", "https"} else None


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


def _unique(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return unique_values
