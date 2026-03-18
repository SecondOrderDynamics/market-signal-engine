# Architecture Overview

## Pipeline
The daily workflow is orchestrated by `daily_scan_runner.py` and keeps existing scanner/evaluator logic intact:

1. Run `explosive_play_scanner.py` to generate candidate scans and append explosive CSV history.
2. Run `event_intel_scanner.py` for explosive tickers to generate event-intel output/history.
3. Merge explosive + event-intel rows via `event_intel_merge.merge_event_intel(...)`.
4. Run `explosive_play_evaluator.py --auto-evaluate` to evaluate eligible historical rows.
5. Emit a structured end-of-run summary and a `daily_summary_*.json` file.

## Data Flow
Primary CSV flow:

- `explosive_play_candidates.csv` -> explosive candidates from the latest run.
- `event_intel_scan.csv` -> event-intel output for explosive tickers.
- `merged_watchlist.csv` -> explosive rows enriched with event-intel columns.
- `explosive_play_eval.csv` -> evaluator output for eligible historical rows.

Primary history CSV flow:

- `explosive_play_historical_scans.csv`
- `event_intel_history.csv`
- `merged_watchlist_history.csv`

Optional SQLite flow:

- Enabled in `daily_scan_config.json` (`storage.enable_sqlite`).
- Appends the same run data to:
  - `explosive_history`
  - `event_intel_history`
  - `merged_watchlist_history`
  - `evaluation_results`
- Schema is simple and adaptive:
  - Columns are inferred from DataFrame dtypes.
  - Missing columns are added with `ALTER TABLE`.
  - Metadata columns are added automatically: `run_id`, `inserted_at_utc`.

## Config and Overrides
`daily_scan_runner.py` reads `daily_scan_config.json` by default.

- Config controls paths, logging, evaluator defaults, and optional SQLite settings.
- CLI flags still work and override config values.
- Existing commands remain compatible (`py .\daily_scan_runner.py`, `--quiet`, path overrides).

## GUI Readiness
The pipeline is now callable as Python functions (not only CLI subprocess chaining):

- `resolve_config(args)` -> runtime config object.
- `run_daily_pipeline(config)` -> executes the full workflow and returns a structured `DailyRunSummary`.
- `write_summary_json(summary)` -> writes GUI-friendly summary JSON.

A future GUI can call `run_daily_pipeline(...)` directly, then render:

- status (`success` / `failed`)
- top candidates
- count metrics
- output file locations
- log location

without changing scanner or scoring logic.

## Local API + HTML Frontend
`local_dashboard_server.py` adds a lightweight local API and static frontend layer.

- Static UI: `dashboard_ui/index.html`, `dashboard_ui/app.js`, `dashboard_ui/styles.css`
- API endpoints:
  - `GET /api/health`
  - `GET /api/latest-run`
  - `GET /api/runs?limit=N`
  - `GET /api/top-candidates?limit=N`
  - `GET /api/merged-watchlist?limit=N`
  - `GET /api/evaluation`
  - `GET /api/files`

This provides a simple bridge for future GUI work: start with local JSON endpoints, then later replace/extend handlers with richer app services without touching scanner scoring code.
