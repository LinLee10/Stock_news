"""Sentiment scoring abstractions with no import-time model downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import Article, SentimentResult, TickerMention


SENTIMENT_BASES = {"full_text", "snippet", "title"}
POSITIVE_TERMS = {"beat", "beats", "growth", "record", "strong", "upgrade", "surge", "profit"}
NEGATIVE_TERMS = {"miss", "misses", "weak", "downgrade", "fall", "falls", "lawsuit", "loss"}


@dataclass(frozen=True)
class SentimentScore:
    article_id: str
    label: str
    score: float
    confidence: float
    model_name: str
    sentiment_basis: str
    evidence_window: str
    ticker: str | None = None


class SentimentScorer(Protocol):
    model_name: str

    def score_article(self, article: Article) -> SentimentScore:
        """Score one article."""

    def score_ticker_mentions(
        self,
        article: Article,
        mentions: list[TickerMention],
    ) -> list[SentimentScore]:
        """Score ticker-specific windows where possible."""


class RuleBasedSentimentScorer:
    """Deterministic scorer for tests and dry runs."""

    model_name = "rule_based_fake"

    def score_article(self, article: Article) -> SentimentScore:
        text, basis = _select_text(article)
        return _score_text(
            article_id=article.article_id or article.canonical_url,
            text=text,
            basis=basis,
            model_name=self.model_name,
        )

    def score_ticker_mentions(
        self,
        article: Article,
        mentions: list[TickerMention],
    ) -> list[SentimentScore]:
        text, basis = _select_text(article)
        scores = []
        for mention in mentions:
            window = _ticker_window(text, mention.ticker) or text
            scores.append(
                _score_text(
                    article_id=article.article_id or article.canonical_url,
                    text=window,
                    basis=basis,
                    model_name=self.model_name,
                    ticker=mention.ticker,
                )
            )
        return scores


class FinBertSentimentScorer:
    """Optional FinBERT scorer. Model loading is explicit and local-only by default."""

    def __init__(
        self,
        *,
        pipeline=None,
        model_name: str = "ProsusAI/finbert",
        load_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self._pipeline = pipeline
        if load_model and self._pipeline is None:
            self._pipeline = self._load_pipeline_local_only(model_name)

    def score_article(self, article: Article) -> SentimentScore:
        text, basis = _select_text(article)
        return self._score(
            article_id=article.article_id or article.canonical_url,
            text=text,
            basis=basis,
            ticker=None,
        )

    def score_ticker_mentions(
        self,
        article: Article,
        mentions: list[TickerMention],
    ) -> list[SentimentScore]:
        text, basis = _select_text(article)
        return [
            self._score(
                article_id=article.article_id or article.canonical_url,
                text=_ticker_window(text, mention.ticker) or text,
                basis=basis,
                ticker=mention.ticker,
            )
            for mention in mentions
        ]

    def _score(
        self,
        *,
        article_id: str,
        text: str,
        basis: str,
        ticker: str | None,
    ) -> SentimentScore:
        if self._pipeline is None:
            raise RuntimeError("FinBERT scorer requires an injected or explicitly loaded pipeline")
        evidence = _evidence_window(text)
        raw = self._pipeline(evidence)
        first = raw[0] if isinstance(raw, list) else raw
        label = str(first.get("label", "neutral")).lower()
        confidence = float(first.get("score", 0.0))
        signed_score = confidence
        if "neg" in label:
            signed_score = -confidence
            label = "negative"
        elif "pos" in label:
            label = "positive"
        else:
            signed_score = 0.0
            label = "neutral"
        return SentimentScore(
            article_id=article_id,
            label=label,
            score=max(-1.0, min(1.0, signed_score)),
            confidence=max(0.0, min(1.0, confidence)),
            model_name=self.model_name,
            sentiment_basis=basis,
            evidence_window=evidence,
            ticker=ticker,
        )

    @staticmethod
    def _load_pipeline_local_only(model_name: str):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("transformers is not installed") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def analyze_sentiment(article_id: str, text: str, basis: str) -> SentimentResult:
    """Compatibility helper returning the original SentimentResult model."""
    score = _score_text(
        article_id=article_id,
        text=text,
        basis=basis,
        model_name=RuleBasedSentimentScorer.model_name,
    )
    return SentimentResult(
        article_id=article_id,
        score=score.score,
        label=score.label,
        confidence=score.confidence,
        basis=score.sentiment_basis,
        model=score.model_name,
    )


def _select_text(article: Article) -> tuple[str, str]:
    if article.full_text and article.full_text.strip():
        return article.full_text.strip(), "full_text"
    if article.snippet and article.snippet.strip():
        return article.snippet.strip(), "snippet"
    return article.title.strip(), "title"


def _score_text(
    *,
    article_id: str,
    text: str,
    basis: str,
    model_name: str,
    ticker: str | None = None,
) -> SentimentScore:
    if basis not in SENTIMENT_BASES:
        raise ValueError(f"sentiment_basis must be one of {sorted(SENTIMENT_BASES)}")
    words = {word.strip(".,:;!?").casefold() for word in text.split()}
    positive = len(words & POSITIVE_TERMS)
    negative = len(words & NEGATIVE_TERMS)
    raw_score = positive - negative
    score = max(-1.0, min(1.0, raw_score / 3.0))
    label = "neutral"
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    confidence = min(1.0, 0.5 + abs(score) / 2.0)
    return SentimentScore(
        article_id=article_id,
        label=label,
        score=score,
        confidence=confidence,
        model_name=model_name,
        sentiment_basis=basis,
        evidence_window=_evidence_window(text),
        ticker=ticker,
    )


def _ticker_window(text: str, ticker: str, radius: int = 8) -> str | None:
    tokens = text.split()
    ticker_norm = ticker.casefold()
    for index, token in enumerate(tokens):
        if token.strip(".,:;!?()[]").casefold() == ticker_norm:
            start = max(0, index - radius)
            end = min(len(tokens), index + radius + 1)
            return " ".join(tokens[start:end])
    return None


def _evidence_window(text: str, max_chars: int = 240) -> str:
    return " ".join(text.split())[:max_chars]
