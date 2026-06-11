# News Pipeline Context

## Purpose

This document is the durable context for future Codex work on the production news
pipeline. Read it after `AGENTS.md`, then verify all assumptions against the
current repository state.

The project goal is to build a ticker-based market intelligence pipeline, not
just an email digest. The backend should collect a broad, source-diverse
article and event corpus, deduplicate it, extract readable text, classify
events, score ticker-specific sentiment, store durable event memory, and
generate a short, readable briefing.

## Production Path

- Use `news_pipeline` only for production pipeline work.
- Treat legacy `main.py` and old scraper paths as reference-only code.
- Do not modify legacy paths unless a prompt explicitly requires it.
- The current repository state is the source of truth when this document and
  implementation details differ.

## Current Architecture

The production pipeline currently includes:

1. RSS and direct-source discovery.
2. SEC official-source discovery.
3. A source registry and budget-aware scheduler.
4. Google News as a recall backstop only.
5. External free-tier API adapters.
6. A ticker query planner with company-name routing.
7. Source quality scoring and filtering.
8. Ticker matching with confidence and match basis.
9. Article and event-type classification.
10. URL deduplication and event clustering.
11. An article extraction ensemble with fallback handling.
12. Extraction quality scoring and grading.
13. Weighted, ticker-specific sentiment coverage.
14. External sentiment benchmark comparison.
15. Summaries and ranked reads.
16. A capped email briefing.
17. SMTP sending behind an explicit confirmation flag.
18. Project-root `.env.local` loading through CLI startup.
19. Diagnostic artifacts and a report contract.
20. SQLite and artifact-backed event memory.
21. Coverage-aware Alpha Vantage benchmark allocation.
22. Configurable comparison with event memory from multiple prior dated runs.
23. Cross-run exact and fuzzy event identity matching.

## API And Source State

External APIs are free-tier but quota-limited. Available environment variable
names may include:

- `NYT_API_KEY`
- `MARKETAUX_API_KEY`
- `GNEWS_KEY`
- `FINNHUB_KEY`
- `NEWSAPI_KEY`
- `ALPHA_VANTAGE_KEY`

Never use Alpaca or other trading keys for this project.

SEC, RSS, PR Newswire, GlobeNewswire, and Business Wire public sources
generally require no API key.

Known working external APIs:

- Marketaux
- NYT Article Search
- GNews
- Finnhub
- NewsAPI
- Alpha Vantage News Sentiment

NYT is a high-quality context-news source. It is not required to provide
ticker coverage and should not be treated as primary ticker sentiment unless
an article is company-specific.

Alpha Vantage News Sentiment is an external benchmark, not ground truth. Its
vendor sentiment, relevance, ticker sentiment, topics, source, URL, title, and
publication time should remain distinguishable from internal sentiment.

## Current Source-Layer Status

External API routing materially reduced dependence on Google News. Remaining
source-layer risks include:

- Provider and ticker concentration in raw collections.
- Finnhub raw-volume concentration.
- AMD concentration.
- Coverage gaps or Google dependence for tickers such as MU and META,
  depending on the run.
- Free-tier quota and rate-limit variability.
- Source-quality classification gaps for unfamiliar publishers.
- Alpha Vantage allocation uses current direct coverage plus prior-run
  benchmark coverage and Google-dependence signals. A first-ever run has less
  historical evidence and must bootstrap from portfolio priority, current
  coverage gaps, and known weak-coverage names.
- Cross-run event identity uses exact ticker/canonical-URL matching first, then
  bounded fuzzy title matching for the same ticker within a configurable
  publication window. The default lookback is three days. Generic or weak
  titles can still produce uncertain classifications.

The Alpha Vantage News Sentiment benchmark and initial durable event memory are
implemented in the current worktree. Alpha Vantage request allocation now
prioritizes portfolio names, weak-coverage names, Google-dominated names, and
tickers with few prior benchmark records instead of relying on alphabetical
order. Reports compare current event-memory records with every completed prior
dated run inside the configured lookback and report `history_building` cleanly
when none exist.
Event-memory records retain the original event title, a normalized title,
publication date bucket, and a deterministic identity fingerprint containing
ticker, title, event and article types, company, source family, and date.
Comparison diagnostics distinguish exact URL repeats, fuzzy event repeats, and
likely new events.

The next recommended phase is to calibrate fuzzy identity thresholds and
lookback duration from labeled multi-run examples, consolidate duplicate prior
events before sentiment baselining, add multi-run sentiment trend summaries and
benchmark calibration, and consider deeper SEC filing parsing as a separately
scoped future phase.

## Event And Sentiment Memory

Event memory should retain enough provenance to compare current reporting with
historical records. Durable records currently target:

- Article ID and canonical URL.
- Publication timestamp.
- Ticker and company.
- Source provider and source family.
- Article type and event type.
- Cluster ID.
- Ticker-match confidence.
- Extraction basis and quality grade.
- Internal sentiment.
- External sentiment provider and score.
- Event summary.
- Run ID and run date.

Historical comparison reads all earlier dated run databases inside a bounded
lookback in read-only mode. The CLI flag `--event-memory-lookback-days`
controls the window and defaults to three days. Canonical URL equality for the
same ticker is resolved across all prior records before fuzzy matching.
Otherwise, normalized titles may match only when the ticker is the same and
publication dates fall inside the lookback. The current similarity threshold,
prior runs, and prior record count are reported in diagnostics. Comparisons
record exact URL repeats, fuzzy repeats, likely new events, and per-ticker
changes in average internal sentiment. Same-day reruns are intentionally not
treated as an earlier reporting period.

SEC candidates are classified at least for:

- `8-K`: material event.
- `10-Q`: quarterly report.
- `10-K`: annual report.
- `6-K`: foreign issuer report.
- `13D` or `13G`: ownership event.
- `DEF 14A`: proxy event.
- `S-1`: registration event.
- `424B`: offering or prospectus event.

Do not add deep filing parsing unless explicitly requested.

## Design Principles

- Hard-cap API requests, fetch budgets, and email length.
- Do not hard-delete useful raw knowledge early.
- Keep raw and post-dedup diagnostics separate.
- Apply soft balancing and downweighting after deduplication when a provider or
  ticker dominates.
- Keep the email short even when the backend corpus is large.
- Use Google News for recall, not as the backbone.
- Use NYT for context, not guaranteed ticker sentiment.
- Use Alpha Vantage sentiment as a benchmark, not truth.
- Preserve provider-specific metadata without coupling pipeline internals to a
  provider response format.
- Prefer full text, then snippets, then titles; always retain the scoring basis.
- Do not build price prediction until durable event memory and sentiment
  history are established and explicitly approved.

## Safety Rules

- Never print, log, summarize, or expose secret values.
- Never inspect `.env.local` contents.
- Do not read or modify files covered by the repository secret-file rules.
- Never commit environment files, artifacts, logs, SQLite databases, generated
  reports, or generated CSV outputs.
- `dry-run-daily` must never send email.
- `send-daily-report` must require `--confirm-send` for real delivery.
- Never run `--confirm-send` unless the user explicitly instructs it.
- External APIs must require explicit global/provider flags and bounded request
  budgets.
- Make no paid or unflagged API calls.
- Do not add forecasting, price prediction, or trading execution unless a
  later prompt explicitly requests and scopes it.
- Do not commit or push unless explicitly instructed.

## Generated Outputs

Run artifacts belong under `artifacts/runs/YYYY-MM-DD/`. Typical diagnostics
include the report contract, provider validation, source diagnostics,
extraction diagnostics, dedupe clusters, sentiment coverage, benchmark
comparison, SEC event candidates, and event-memory JSON/CSV.

These outputs are runtime artifacts. Do not treat them as source files or
commit them.

## Standard Validation

Run the complete production-pipeline test suite:

```bash
python -B -m unittest discover -s tests/news_pipeline
```

Run an email preview without sending:

```bash
python -B -m news_pipeline.cli send-daily-report \
  --artifacts-dir artifacts \
  --to collin.le2014@gmail.com
```

Never add `--confirm-send` unless explicitly instructed.

## Documentation Maintenance

At the end of every major implementation phase, update this document with:

1. The new current architecture or behavior.
2. Newly validated providers or workflows.
3. Known remaining problems.
4. The next recommended phase.

Keep this file concise enough to read at the start of future tasks. Put
detailed implementation history in task-specific documentation rather than
turning this document into a changelog.
