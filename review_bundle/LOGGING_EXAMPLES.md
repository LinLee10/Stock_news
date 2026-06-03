# Structured Logging and Auditability - Examples and Implementation

## Overview

The vNext features now include comprehensive structured logging that tracks:
- **Symbol**: Which ticker symbol is being processed
- **Step**: Current processing step (validate, fetch_data, analyze, persist, etc.)
- **Duration**: Time taken for operations in milliseconds
- **Counts**: Input and output record counts (count_in/count_out)
- **Source Links**: URLs or file paths accessed during processing

## Log Storage Location

All audit logs are written to: `data/audit_logs/vnext_audit_YYYYMMDD.jsonl`

Example: `data/audit_logs/vnext_audit_20250818.jsonl`

## Log Entry Structure

Each log entry follows this JSON structure:

```json
{
  "timestamp": "2025-08-19T01:02:30.699598",
  "log_id": "599acf51-902c-4438-9d9d-9bdc6029b199",
  "feature": "symbol_intake",
  "operation": "intake_job", 
  "symbol": "AAPL",
  "step": "validate_symbol",
  "duration_ms": null,
  "count_in": 1,
  "count_out": null,
  "source_links": [],
  "status": "in_progress",
  "metadata": {"ticker": "AAPL"},
  "error_message": null
}
```

## Real Log Examples

### 1. Symbol Intake Operation

**Complete flow for AAPL symbol intake:**

```jsonl
{"timestamp": "2025-08-19T01:02:30.699598", "log_id": "599acf51-902c-4438-9d9d-9bdc6029b199", "feature": "symbol_intake", "operation": "intake_job", "symbol": "AAPL", "step": "started", "duration_ms": null, "count_in": null, "count_out": null, "source_links": [], "status": "started", "metadata": {"test": true}, "error_message": null}

{"timestamp": "2025-08-19T01:02:30.699825", "log_id": "599acf51-902c-4438-9d9d-9bdc6029b199", "feature": "symbol_intake", "operation": "intake_job", "symbol": "AAPL", "step": "validate_symbol", "duration_ms": null, "count_in": 1, "count_out": null, "source_links": [], "status": "in_progress", "metadata": {"ticker": "AAPL"}, "error_message": null}

{"timestamp": "2025-08-19T01:02:30.804544", "log_id": "599acf51-902c-4438-9d9d-9bdc6029b199", "feature": "symbol_intake", "operation": "intake_job", "symbol": "AAPL", "step": "fetch_price_data", "duration_ms": null, "count_in": null, "count_out": 90, "source_links": ["https://api.example.com/aapl"], "status": "in_progress", "metadata": {}, "error_message": null}

{"timestamp": "2025-08-19T01:02:30.907186", "log_id": "599acf51-902c-4438-9d9d-9bdc6029b199", "feature": "symbol_intake", "operation": "intake_job", "symbol": "AAPL", "step": "fetch_news", "duration_ms": null, "count_in": null, "count_out": 15, "source_links": ["https://news.example.com/aapl"], "status": "in_progress", "metadata": {}, "error_message": null}

{"timestamp": "2025-08-19T01:02:31.010889", "log_id": "599acf51-902c-4438-9d9d-9bdc6029b199", "feature": "", "operation": "", "symbol": null, "step": "completed", "duration_ms": 311, "count_in": null, "count_out": null, "source_links": [], "status": "completed", "metadata": {}, "error_message": null}
```

**Key metrics captured:**
- **Symbol**: AAPL
- **Total Duration**: 311ms 
- **Data Retrieved**: 90 price data points, 15 news articles
- **Source Links**: API endpoints accessed
- **Steps**: validate_symbol → fetch_price_data → fetch_news → completed

### 2. Recommendation Generation

**MSFT recommendation generation:**

```jsonl
{"timestamp": "2025-08-19T01:02:31.011402", "log_id": "d014d71e-7708-4555-a6b5-3cb89ee0a909", "feature": "recommendations", "operation": "generate_recs", "symbol": "MSFT", "step": "started", "duration_ms": null, "count_in": null, "count_out": null, "source_links": [], "status": "started", "metadata": {}, "error_message": null}

{"timestamp": "2025-08-19T01:02:31.011564", "log_id": "d014d71e-7708-4555-a6b5-3cb89ee0a909", "feature": "recommendations", "operation": "generate_recs", "symbol": "MSFT", "step": "load_technical_data", "duration_ms": null, "count_in": 1, "count_out": null, "source_links": [], "status": "in_progress", "metadata": {}, "error_message": null}

{"timestamp": "2025-08-19T01:02:31.011664", "log_id": "d014d71e-7708-4555-a6b5-3cb89ee0a909", "feature": "recommendations", "operation": "generate_recs", "symbol": "MSFT", "step": "calculate_signals", "duration_ms": null, "count_in": null, "count_out": 1, "source_links": [], "status": "in_progress", "metadata": {"action": "buy", "confidence": 0.75}, "error_message": null}

{"timestamp": "2025-08-19T01:02:31.011806", "log_id": "d014d71e-7708-4555-a6b5-3cb89ee0a909", "feature": "", "operation": "", "symbol": null, "step": "completed", "duration_ms": 0, "count_in": null, "count_out": null, "source_links": [], "status": "completed", "metadata": {}, "error_message": null}
```

**Key metrics captured:**
- **Symbol**: MSFT
- **Recommendation**: buy action with 0.75 confidence
- **Steps**: load_technical_data → calculate_signals → completed
- **Duration**: <1ms (very fast recommendation)

## Instrumented Components

### 1. Symbol Intake Service (`services/symbol_intake.py`)
- **Feature**: `symbol_intake`
- **Operations**: `intake_job`
- **Steps**: load_registry, registry_loaded, record_updated/inserted, save_registry, fetch_result
- **Source Links**: Registry file paths
- **Counts**: Registry record counts

### 2. Job Queue Service (`services/job_queue.py`)
- **Feature**: `job_queue`
- **Operations**: `enqueue_job`
- **Steps**: write_job_file, job_enqueued
- **Source Links**: Job file paths
- **Counts**: Queue positions

### 3. API Endpoints

#### Symbol Intake API (`api/symbols.py`)
- **Feature**: `api`
- **Operations**: `request_symbols_intake`
- **Steps**: request_validated, ticker_validated, services_initialized, duplicate_check, job_enqueued, response_prepared
- **Metadata**: Ticker, company name, priority, queue position

#### Recommendations API (`api/recommendations.py`)
- **Feature**: `api`
- **Operations**: `request_recommendations`
- **Steps**: params_parsed, scope validation, service calls
- **Metadata**: Scope (watchlist/portfolio), include_details, max_age_hours

### 4. News Processing (`services/news_clustering.py`)
- **Feature**: `news_processing`
- **Operations**: `clustering`, `clustering_ticker`
- **Steps**: start_clustering, clusters_created, headlines_enhanced, clusters_stored, clustering_complete
- **Source Links**: Cluster storage files
- **Counts**: Headlines in/out, cluster counts
- **Metadata**: Similarity thresholds, ticker-specific metrics

### 5. Recommendation Engine (`services/reco_engine.py`)
- **Feature**: `recommendations`
- **Operations**: `generate_recs`
- **Steps**: engine_initialized, data_converted, recommendation_generated, result_serialized
- **Metadata**: Position status, recommendation action/confidence, technical indicators

### 6. Earnings Analysis (`services/earnings_service.py`)
- **Feature**: `earnings`
- **Operations**: `analyze_earnings`
- **Steps**: analysis_initialized, implied_move_calculated, historical_move_calculated, direction_classified, stats_persisted
- **Source Links**: Earnings stats file
- **Metadata**: Implied moves, direction predictions, confidence scores

## Admin Log Viewing Endpoints

### 1. GET /api/v1/admin/logs
View recent audit logs with filtering:

```bash
curl "http://localhost:5000/api/v1/admin/logs?hours=24&feature=symbol_intake&status=completed"
```

**Response structure:**
```json
{
  "query": {
    "hours": 24,
    "feature_filter": "symbol_intake",
    "operation_filter": "all",
    "status_filter": "completed",
    "generated_at": "2025-08-19T01:00:00"
  },
  "logs": [...],
  "summary": {
    "total_operations": 125,
    "by_feature": {"symbol_intake": 45, "recommendations": 80},
    "by_status": {"completed": 120, "failed": 5},
    "avg_duration_ms": 245,
    "total_symbols_processed": 15
  },
  "total_entries": 10,
  "available_filters": {
    "features": ["symbol_intake", "recommendations", "earnings", "news_processing"],
    "operations": ["intake_job", "generate_recs", "clustering"],
    "statuses": ["started", "completed", "failed", "in_progress"]
  }
}
```

### 2. GET /api/v1/admin/logs/summary
Quick summary of recent operations:

```bash
curl "http://localhost:5000/api/v1/admin/logs/summary?hours=24"
```

### 3. GET /api/v1/admin/logs/operation/{operation_id}
Get all log entries for a specific operation:

```bash
curl "http://localhost:5000/api/v1/admin/logs/operation/599acf51-902c-4438-9d9d-9bdc6029b199"
```

### 4. GET /api/v1/admin/logs/export
Export logs in JSONL format:

```bash
curl "http://localhost:5000/api/v1/admin/logs/export?hours=168&format=jsonl"
```

## Log Analysis Use Cases

### 1. Performance Monitoring
- Track operation durations to identify bottlenecks
- Monitor API response times
- Identify slow data source calls

### 2. Usage Analytics
- Count symbols processed per day
- Track most requested tickers
- Monitor feature adoption

### 3. Error Diagnosis
- Filter by `status=failed` to find errors
- Trace operation flow for debugging
- Identify problematic data sources

### 4. Audit Trail
- Full traceability of all operations
- Source attribution for data fetching
- Complete step-by-step operation history

### 5. Capacity Planning
- Monitor queue sizes and processing times
- Track data volumes (count_in/count_out)
- Identify peak usage patterns

## Log Retention

- **Daily rotation**: New log file each day
- **Format**: JSONL (newline-delimited JSON)
- **Persistence**: Logs stored on filesystem for analysis
- **Access**: Admin API endpoints for programmatic access