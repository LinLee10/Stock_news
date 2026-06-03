import importlib
import sys
import types
from datetime import datetime


def _import_news_deduplicator_without_external_feature_imports(monkeypatch):
    fake_vector_store = types.ModuleType("services.vector_store")
    fake_vector_store.NewsItem = object
    fake_vector_store.vector_store = object()

    fake_feature_flags = types.ModuleType("config.feature_flags")
    fake_feature_flags.is_vector_search_enabled = lambda: False

    monkeypatch.setitem(sys.modules, "services.vector_store", fake_vector_store)
    monkeypatch.setitem(sys.modules, "config.feature_flags", fake_feature_flags)
    monkeypatch.delitem(sys.modules, "services.news_deduplicator", raising=False)

    return importlib.import_module("services.news_deduplicator")


def test_canonicalize_url_removes_tracking_and_preserves_case(monkeypatch):
    deduplicator = _import_news_deduplicator_without_external_feature_imports(monkeypatch)

    canonical = deduplicator.canonicalize_url(
        "HTTPS://Example.COM/News/Article-ABC"
        "?utm_source=newsletter&token=MiXeD&id=42&utm_medium=email#section"
    )

    assert canonical == "https://example.com/News/Article-ABC?id=42&token=MiXeD"


def test_dedupe_headlines_simple_uses_canonical_urls(monkeypatch):
    deduplicator = _import_news_deduplicator_without_external_feature_imports(monkeypatch)
    published_at = datetime(2024, 1, 15, 10, 0, 0)
    headlines = [
        (
            "Company reports quarterly earnings",
            "https://example.com/news/article?id=42&utm_source=feed",
            published_at,
        ),
        (
            "Different headline from same article URL",
            "https://EXAMPLE.com/news/article?utm_medium=social&id=42",
            published_at,
        ),
    ]

    assert deduplicator.dedupe_headlines_simple(headlines) == [headlines[0]]
