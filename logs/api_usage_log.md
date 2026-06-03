# API Usage Log

## Purpose
Time-series log of API rate limit states and usage patterns for observability and capacity planning.

## Format
`YYYY-MM-DD HH:MM TZ — LimitState=<State> — <Notes>`

## Entries

2025-08-31 02:00 America/Los_Angeles — LimitState=Exhausted — All free-tier API calls used. Renewal time: unknown (to research). Treat 429s today as expected limit, not a bug.
2025-08-31 02:30 America/Los_Angeles — LimitState=Mixed — Final analysis: NewsAPI 100/100 calls exhausted (02:22 UTC). Alpha Vantage 0 calls (cache-protected). Yahoo Finance 100% cache hits. 72h probe scheduled for AV reset timing. Circuit breakers operational.