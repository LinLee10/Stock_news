# News Pipeline Rebuild Plan

## Direction

This is a selective rebuild inside the existing repository, not a full wipe. The goal is to introduce a clean, testable core package named `news_pipeline` while preserving useful existing code, tests, configuration, and operational knowledge. Existing modules should be migrated or wrapped incrementally only when their behavior is understood and covered.

## Principles

- Build the new pipeline beside the current implementation, then switch callers over in small reversible phases.
- Keep local development no-network by default, with provider calls behind explicit adapters and fakes.
- Treat provider usage, quotas, provenance, extraction basis, and forecast basis as first-class data.
- Prefer real article text and conservative outputs over synthetic precision.
- Keep secrets out of logs, tests, docs examples, and import-time side effects.

## Target Architecture

The new package should live at `news_pipeline/` and own the canonical pipeline flow:

1. Discover article candidates from providers.
2. Canonicalize and group URLs.
3. Fetch and extract article content.
4. Deduplicate candidates.
5. Persist canonical article, provider, extraction, sentiment, mention, and forecast records to SQLite.
6. Generate daily pre-market outputs.
7. Send email through a provider adapter.

Suggested package layout:

```text
news_pipeline/
  config.py
  db/
    schema.py
    sqlite_store.py
  providers/
    base.py
    rss.py
    alpha_vantage.py
    marketaux.py
    gnews.py
  extraction/
    trafilatura_extractor.py
    newspaper_fallback.py
  dedupe/
    urls.py
    titles.py
    semantic.py
  sentiment/
    analyzer.py
  forecasts/
    direction.py
  reports/
    daily_email.py
  email/
    resend_sender.py
    smtp_sender.py
  telemetry/
    provider_usage.py
```

## Provider Strategy

Use hybrid discovery:

- RSS is the default free baseline and should support curated financial feeds.
- Marketaux is the preferred paid provider later if coverage and pricing justify it.
- Alpha Vantage news is auxiliary, not the primary discovery source, because quota and article depth are limiting factors.
- GNews is optional later if it fills a measurable coverage gap.

Each provider adapter should return normalized candidate records with provider name, provider article id when present, URL, title, snippet, symbols, published timestamp, source name, and raw provider metadata. Provider-specific data should be stored but not leak into pipeline internals.

## Extraction

Article extraction should prefer Trafilatura first and use Newspaper3k as fallback. Extraction records must include:

- `extractor`: `trafilatura`, `newspaper3k`, or `none`
- `basis`: `full_text`, `snippet`, or `title`
- text length
- failure reason when extraction fails
- final URL after redirects when available

Sentiment and downstream reporting must explicitly record whether the score came from full text, snippet, or title.

## SQLite Canonical Store

SQLite is the local canonical store for the rebuilt pipeline. It should be deterministic, easy to inspect, and safe for local runs.

Core tables:

- `providers`: provider metadata and enabled state.
- `provider_usage`: every provider call, quota cost, status, latency, and error class.
- `article_candidates`: raw normalized discoveries.
- `canonical_articles`: deduped canonical article records.
- `article_sources`: provider/source links for each canonical article.
- `extractions`: extracted full text or fallback basis metadata.
- `sentiment_scores`: score, model, basis, confidence, and timestamps.
- `mentions`: symbol/name mentions with source article id and confidence.
- `forecasts`: next market close direction and confidence.
- `email_runs`: generated pre-market email runs and delivery status.

Provider usage and quota tracking must be queryable before a run, during a run, and after failures. A failed request still gets a `provider_usage` row.

## Deduplication

Dedup should combine four signals:

1. URL canonicalization: normalize scheme/host, remove tracking query params, sort retained params, remove fragments, and preserve meaningful path/query case.
2. Provider grouping: group identical provider ids, source names, syndicated URLs, and canonical URLs.
3. Title similarity: use a deterministic fuzzy similarity threshold for near-identical titles.
4. Semantic similarity: optional embedding or lightweight semantic comparison for articles with sufficient text.

The dedupe result should store the winning canonical article id, duplicate reason, score, and all source candidate ids.

## Sentiment

Sentiment should run on full article text when available. If extraction fails, fall back in this order:

1. `full_text`
2. `snippet`
3. `title`

Every sentiment row must include the fallback basis. Reports should avoid presenting title-only sentiment as equally reliable as full-text sentiment.

## Forecasts

Forecast output should be next market close direction and confidence. It must not emit fake precise price targets.

Allowed examples:

- direction: `up`, `down`, `flat`, or `uncertain`
- horizon: `next_market_close`
- confidence: calibrated probability or bucketed confidence
- drivers: recent sentiment, mention momentum, source quality, corroboration, and price context

Disallowed output:

- exact next close price
- synthetic target price
- over-precise percentage moves unsupported by data

## Daily Pre-Market Email

The daily email should be generated before market open and include:

- portfolio table
- watchlist forecast table
- mention leaders
- emerging names
- article links
- CSV attachments
- chart attachments

The email should clearly identify stale or partial data, including provider failures, quota limits, extraction fallbacks, and sentiment basis. CSV and chart attachment generation should be deterministic and testable without sending email.

## Email Sender

Resend is the recommended email sender for the rebuild because it gives a clean API and better operational visibility. SMTP or existing email code should remain as fallback adapters.

Sender selection should happen behind an interface:

- `ResendSender`: primary recommended implementation.
- `SmtpSender`: fallback.
- `LocalFileSender`: no-network test/dry-run implementation.

No sender should read secrets at import time or print configuration values.

## Phased Plan

### Phase 1: Safety Baseline

- Stop config imports from printing secrets.
- Fix syntax/import errors blocking tests.
- Add focused no-network tests for config import safety and URL canonicalization.
- Keep behavior changes minimal.

### Phase 2: Core Skeleton

- Add `news_pipeline` package with empty interfaces and SQLite schema tests.
- Add provider adapter protocols and no-network fakes.
- Add initial `provider_usage` table and tests.
- No production caller switch yet.

### Phase 3: Discovery and Quota Tracking

- Implement RSS provider adapter first.
- Add Alpha Vantage auxiliary adapter behind strict quota tracking.
- Stub Marketaux and GNews adapters for later paid/optional activation.
- Persist provider usage for every attempted call.

### Phase 4: Extraction

- Add Trafilatura extraction adapter.
- Add Newspaper3k fallback adapter.
- Persist extraction basis and failure reasons.
- Test full text, snippet fallback, and title fallback paths.

### Phase 5: Dedupe and Canonical Articles

- Implement URL canonicalization and candidate grouping.
- Add title similarity dedupe.
- Add semantic similarity behind an optional feature flag.
- Persist duplicate decisions and source mappings.

### Phase 6: Sentiment and Mentions

- Run sentiment using the best available extraction basis.
- Persist sentiment basis explicitly.
- Add mention extraction for configured portfolio/watchlist names and emerging names.
- Add confidence metadata for mention matches.

### Phase 7: Forecasts

- Produce next market close direction and confidence only.
- Add calibration tests and no fake target-price fields.
- Store forecast drivers for auditability.

### Phase 8: Daily Email

- Build deterministic pre-market report data.
- Generate portfolio/watchlist tables, mention leaders, emerging names, links, CSVs, and charts.
- Load first-class tracked symbols from `news_pipeline.tickers`: portfolio holdings drive the portfolio 30 day sentiment table, and watchlist symbols drive the next market close direction/confidence table. Each tracked ticker carries company names and aliases for headline/full-text matching, supporting article links, mention leader calculations, CSV attachments, and chart labels.
- Add `LocalFileSender` tests before real sender integration.
- Add Resend sender, then SMTP/existing sender fallback.

#### Manual VS Code Dry-Run Workflow

Until live provider wiring and email sending are explicitly enabled, the daily manual workflow is to open the repository in VS Code and run `python -B -m news_pipeline.cli dry-run-daily --run-date YYYY-MM-DD --artifacts-dir artifacts` from the integrated terminal. This command uses only local RSS fixtures, writes report artifacts under `artifacts/runs/YYYY-MM-DD/`, and does not call live APIs or send email.

### Phase 9: Migration

- Wire existing entrypoints to `news_pipeline` behind a feature flag or explicit command.
- Keep rollback path to existing implementation.
- Remove duplicated legacy paths only after usage is migrated and tests cover the new path.

## Non-Goals

- No full repository wipe.
- No deletion of reports, data, plots, logs, models, or review bundles until separately reviewed.
- No paid provider calls during rebuild tests.
- No exact price-target forecasts.
- No import-time secret loading side effects beyond silent environment loading.

## Open Questions

- Which RSS feeds are canonical for portfolio and watchlist coverage?
- Should SQLite live under `data/`, `state/`, or a new ignored local path?
- Which current CSVs are true user input versus generated output?
- Which existing chart/report code should be wrapped versus replaced?
- What confidence threshold should gate emerging-name inclusion in the email?
