# Cleanup Deletion Manifest

## Scope

This cleanup removes generated or obsolete repository clutter after the `news_pipeline` baseline passed its targeted unittest suite. It does not remove active source architecture folders, requirements files, tests, secrets, real env files, or model artifacts.

## Baseline

- Branch: `news-pipeline-rebuild`
- Pre-cleanup test gate: `python -B -m unittest discover -s tests/news_pipeline`
- Result: passed, 46 tests
- Real secret/env files were not read, printed, modified, staged, or deleted.

## Manifest

| Path | Category | Risk | Reason | Final Action |
|---|---|---:|---|---|
| `**/__pycache__/` | Python bytecode cache | Low | Generated interpreter cache, reproducible. | Delete |
| `.pytest_cache/` | Test cache | Low | Generated pytest cache, reproducible. | Delete |
| `**/.DS_Store` | macOS metadata | Low | OS-generated metadata, not project source. | Delete |
| `**/*.pyc` | Python bytecode | Low | Generated interpreter bytecode, reproducible. | Delete |
| `TREE.txt` | Generated inventory | Low | Static tree snapshot superseded by live repository state. | Delete |
| `report.html` | Generated report output | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `report.txt` | Generated report output | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `portfolio_collage.png` | Generated image | Low | Root-level generated chart/image output. | Delete |
| `watchlist_collage.png` | Generated image | Low | Root-level generated chart/image output. | Delete |
| `test_enhanced_charts.png` | Generated test image | Low | Generated chart test artifact. | Delete |
| `plots/*.png` | Generated plot images | Low | Generated charts; new report artifacts write under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `plots/prediction_charts/` | Generated chart folder | Low | Generated prediction chart output. | Delete |
| `plots/sentiment_trends/` | Generated chart folder | Low | Generated sentiment chart output. | Delete |
| `report 2/` | Old audit/report artifact | Medium | Duplicate old audit report folder superseded by current docs and rebuild plan. | Delete |
| `report 3/` | Old audit/report artifact | Medium | Duplicate old audit report folder superseded by current docs and rebuild plan. | Delete |
| `log.txt` | Runtime log | Low | Runtime output, not source. | Delete |
| `pipeline_run.log` | Runtime log | Low | Runtime output, not source. | Delete |
| `logs/api_usage_log.md` | Runtime/security-sensitive log | Medium | Usage log can expose operational details; obsolete for source tree. | Delete |
| `review_bundle/` | Generated duplicate snapshot | Medium | Duplicate snapshot of old repo files; contains stale syntax errors and blocks full parse checks. | Delete |
| `review_bundle/ENTRYPOINTS_AND_IMPORTS.md` | Review bundle note | Medium | Potentially useful note captured by filename here; bundle is obsolete duplicate snapshot. | Delete with bundle |
| `review_bundle/FILE_INDEX.md` | Review bundle note | Medium | Potentially useful note captured by filename here; bundle is obsolete duplicate snapshot. | Delete with bundle |
| `review_bundle/TESTS_SUMMARY.md` | Review bundle note | Medium | Potentially useful note captured by filename here; bundle is obsolete duplicate snapshot. | Delete with bundle |
| `review_bundle/.env.example`, `review_bundle/secrets.env.sample` | Sample/template env-like files in obsolete snapshot | Low | Sample/template files only, deleted solely as part of obsolete generated bundle removal. | Delete with bundle |
| `article_sentiment_feed.csv` | Generated root CSV | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `daily_mentions.csv` | Generated root CSV | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `daily_sentiment_summary.csv` | Generated root CSV | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `predictions_log.csv` | Generated root CSV | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `sentiment_history.csv` | Generated root CSV | Low | Root-level generated output; new outputs belong under `artifacts/runs/YYYY-MM-DD/`. | Delete |
| `data/av_bulk_cache/` | Generated provider cache | Medium | Cached provider data; new canonical local storage/reporting are under `news_pipeline` and `artifacts/runs`. | Delete |
| `data/yf_bulk_cache/` | Generated provider cache | Medium | Cached provider data; new canonical local storage/reporting are under `news_pipeline` and `artifacts/runs`. | Delete |
| `data/yf_http_cache.sqlite` | Generated HTTP cache | Medium | Generated cache database if present. | Delete if present |
| `data/audit_logs/` | Generated audit log folder | Medium | Runtime audit logs, not source. | Delete |

## Kept

- `AGENTS.md`, `README.md`, `docs/`, requirements files, `pytest.ini`, `conftest.py`, `tests/`, `fakes/`, `news_pipeline/`
- Source architecture folders: `config/`, `services/`, `api/`, `ingest/`, `news_ingest/`, `microservices/`, `docker/`, `kubernetes/`, `terraform/`, `scripts/`
- Main source entrypoints: `main.py`, `news_scraper.py`, `prediction.py`, `charts.py`, `email_report.py`
- `models/`, including `models/reg_model.pkl`
- `secrets/` and real env/secret files
