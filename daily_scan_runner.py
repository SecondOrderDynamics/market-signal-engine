#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from event_intel_merge import merge_event_intel
from history_storage import SQLiteHistoryStore, append_csv_compatible


DEFAULT_CONFIG_PATH = "daily_scan_config.json"
DEFAULT_EXPLOSIVE_OUTPUT = "explosive_play_candidates.csv"
DEFAULT_EXPLOSIVE_HISTORY = "explosive_play_historical_scans.csv"
DEFAULT_EVENT_OUTPUT = "event_intel_scan.csv"
DEFAULT_EVENT_HISTORY = "event_intel_history.csv"
DEFAULT_MERGED_OUTPUT = "merged_watchlist.csv"
DEFAULT_MERGED_HISTORY = "merged_watchlist_history.csv"
DEFAULT_EVAL_OUTPUT = "explosive_play_eval.csv"
DEFAULT_LOG_DIR = "logs"
DEFAULT_SUMMARY_DIR = "logs"
DEFAULT_LAST_RUN_STATUS = "last_run_status.json"
DEFAULT_TOP_N = "5,10,20"
DEFAULT_WEIGHTS = "0.65,0.25,0.10"
DEFAULT_TOP_CANDIDATES = 10
DEFAULT_TIMEOUT_EXPLOSIVE_SECONDS = 45 * 60
DEFAULT_TIMEOUT_EVENT_SECONDS = 20 * 60
DEFAULT_TIMEOUT_EVALUATOR_SECONDS = 10 * 60


@dataclass
class RunnerConfig:
    python: str
    explosive_output: str
    explosive_history: str
    event_output: str
    event_history: str
    merged_output: str
    merged_history: str
    eval_output: str
    eval_top_n: str
    eval_weights: str
    auto_evaluate: bool
    log_dir: str
    summary_dir: str
    quiet: bool
    enable_sqlite: bool
    sqlite_path: str
    sqlite_table_explosive: str
    sqlite_table_event_intel: str
    sqlite_table_merged: str
    sqlite_table_evaluation: str


@dataclass
class DailyRunSummary:
    timestamp_utc: str
    run_id: str
    status: str
    success: bool
    error_message: str
    top_candidates: list[dict[str, Any]]
    candidate_counts: dict[str, int]
    evaluation_counts: dict[str, int]
    output_files: dict[str, str]
    log_file: str
    merge_status: str
    last_completed_stage: str


class RunLogger:
    def __init__(self, log_path: Path, quiet: bool = False) -> None:
        self.log_path = log_path
        self.quiet = quiet
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def log(self, message: str, always_console: bool = False) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if always_console or not self.quiet:
            print(line)


def log_stage_start(logger: RunLogger, stage_name: str) -> datetime:
    started_at = datetime.now(timezone.utc)
    logger.log(f"{stage_name} started | start_utc={started_at.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    return started_at


def log_stage_end(logger: RunLogger, stage_name: str, started_at: datetime, status: str = "finished") -> None:
    ended_at = datetime.now(timezone.utc)
    elapsed_seconds = max(0.0, (ended_at - started_at).total_seconds())
    logger.log(
        f"{stage_name} {status} | end_utc={ended_at.strftime('%Y-%m-%dT%H:%M:%SZ')} | elapsed_seconds={elapsed_seconds:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily explosive/event/evaluation workflow.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to JSON config file.")
    parser.add_argument("--python", default=None, help="Python executable used to run child scripts.")
    parser.add_argument("--explosive-output", default=None, help="Explosive scan CSV output path.")
    parser.add_argument("--explosive-history", default=None, help="Explosive history CSV path.")
    parser.add_argument("--event-output", default=None, help="Event intel scan CSV output path.")
    parser.add_argument("--event-history", default=None, help="Event intel history CSV path.")
    parser.add_argument("--merged-output", default=None, help="Merged watchlist CSV path.")
    parser.add_argument("--merged-history", default=None, help="Merged watchlist history CSV path.")
    parser.add_argument("--eval-output", default=None, help="Evaluator output CSV path.")
    parser.add_argument("--eval-weights", default=None, help="Composite weights passed to evaluator.")
    parser.add_argument("--eval-top-n", default=None, help="Top-N groups for evaluator (for example 5,10,20).")
    parser.add_argument("--log-dir", default=None, help="Directory for timestamped daily run logs.")
    parser.add_argument("--summary-dir", default=None, help="Directory for final daily summary JSON files.")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output (full details still logged).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without executing scanners, network calls, or writing files"
    )
    parser.add_argument("--enable-sqlite", action="store_true", help="Enable SQLite history persistence.")
    parser.add_argument("--sqlite-path", default=None, help="SQLite file path for optional local storage.")
    args = parser.parse_args()

    if args.dry_run and not args.quiet:
        print("=== DRY RUN MODE ===")

    return args


def _cfg_get(data: dict[str, Any], path: Sequence[str], default: Any) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_json_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def resolve_config(args: argparse.Namespace) -> RunnerConfig:
    cfg = load_json_config(args.config)
    tables = _cfg_get(cfg, ["storage", "tables"], {}) or {}

    return RunnerConfig(
        python=args.python or sys.executable,
        explosive_output=args.explosive_output or _cfg_get(cfg, ["paths", "explosive_output"], DEFAULT_EXPLOSIVE_OUTPUT),
        explosive_history=args.explosive_history or _cfg_get(cfg, ["paths", "explosive_history"], DEFAULT_EXPLOSIVE_HISTORY),
        event_output=args.event_output or _cfg_get(cfg, ["paths", "event_output"], DEFAULT_EVENT_OUTPUT),
        event_history=args.event_history or _cfg_get(cfg, ["paths", "event_history"], DEFAULT_EVENT_HISTORY),
        merged_output=args.merged_output or _cfg_get(cfg, ["paths", "merged_output"], DEFAULT_MERGED_OUTPUT),
        merged_history=args.merged_history or _cfg_get(cfg, ["paths", "merged_history"], DEFAULT_MERGED_HISTORY),
        eval_output=args.eval_output or _cfg_get(cfg, ["paths", "eval_output"], DEFAULT_EVAL_OUTPUT),
        eval_top_n=args.eval_top_n or _cfg_get(cfg, ["evaluation", "top_n"], DEFAULT_TOP_N),
        eval_weights=args.eval_weights or _cfg_get(cfg, ["evaluation", "composite_weights"], DEFAULT_WEIGHTS),
        auto_evaluate=bool(_cfg_get(cfg, ["evaluation", "auto_evaluate"], True)),
        log_dir=args.log_dir or _cfg_get(cfg, ["paths", "log_dir"], DEFAULT_LOG_DIR),
        summary_dir=args.summary_dir or _cfg_get(cfg, ["paths", "summary_dir"], DEFAULT_SUMMARY_DIR),
        quiet=bool(args.quiet),
        enable_sqlite=bool(args.enable_sqlite or _cfg_get(cfg, ["storage", "enable_sqlite"], False)),
        sqlite_path=args.sqlite_path or _cfg_get(cfg, ["storage", "sqlite_path"], "scan_history.db"),
        sqlite_table_explosive=str(tables.get("explosive", "explosive_history")),
        sqlite_table_event_intel=str(tables.get("event_intel", "event_intel_history")),
        sqlite_table_merged=str(tables.get("merged", "merged_watchlist_history")),
        sqlite_table_evaluation=str(tables.get("evaluation", "evaluation_results")),
    )


def safe_read_csv(path: str, logger: RunLogger) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        logger.log(f"Missing CSV: {path}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        logger.log(f"Failed to read {path}: {e}")
        return None


def run_command(
    command: Sequence[str],
    logger: RunLogger,
    hard_fail: bool = False,
    timeout_seconds: Optional[int] = None,
) -> int:
    logger.log(f"RUN: {' '.join(command)}")
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        logger.log(
            f"Command timed out after {timeout_seconds} second(s): {' '.join(command)}"
        )
        if e.stdout:
            for line in str(e.stdout).splitlines():
                logger.log(f"[stdout] {line}")
        if e.stderr:
            for line in str(e.stderr).splitlines():
                logger.log(f"[stderr] {line}")
        if hard_fail:
            return 124
        return 0

    if proc.stdout.strip():
        for line in proc.stdout.splitlines():
            logger.log(f"[stdout] {line}")
    if proc.stderr.strip():
        for line in proc.stderr.splitlines():
            logger.log(f"[stderr] {line}")

    if proc.returncode != 0:
        logger.log(f"Command failed with exit code {proc.returncode}")
        if hard_fail:
            return proc.returncode
    return 0


def persist_to_sqlite(
    store: Optional[SQLiteHistoryStore],
    table: str,
    df: Optional[pd.DataFrame],
    run_id: str,
    logger: RunLogger,
) -> None:
    if store is None or df is None or df.empty:
        return
    try:
        inserted = store.append_dataframe(table_name=table, df=df, run_id=run_id)
        logger.log(f"SQLite append -> {table}: {inserted} row(s)")
    except Exception as e:
        logger.log(f"[WARN] SQLite append failed for {table}: {e}")


def build_top_candidates(df: pd.DataFrame, top_n: int = DEFAULT_TOP_CANDIDATES) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    sorted_df = df.sort_values(by=["total_score", "rel_volume"], ascending=[False, False]).head(top_n).copy()
    keep = [c for c in ["ticker", "total_score", "event_intel_score", "anomaly_score", "anomaly_label"] if c in sorted_df.columns]
    if keep:
        sorted_df = sorted_df[keep]
    return sorted_df.fillna("").to_dict(orient="records")


def run_explosive_scan(config: RunnerConfig, logger: RunLogger) -> tuple[int, Optional[pd.DataFrame]]:
    rc = run_command(
        [
            config.python,
            "explosive_play_scanner.py",
            "--output",
            config.explosive_output,
            "--history",
            config.explosive_history,
        ],
        logger,
        hard_fail=True,
        timeout_seconds=DEFAULT_TIMEOUT_EXPLOSIVE_SECONDS,
    )
    if rc != 0:
        return rc, None
    return 0, safe_read_csv(config.explosive_output, logger)


def run_event_intel_scan(config: RunnerConfig, explosive_df: pd.DataFrame, logger: RunLogger) -> int:
    if explosive_df.empty or "ticker" not in explosive_df.columns:
        logger.log("Skipping event-intel scan (no explosive tickers available)")
        return 0
    return run_command(
        [
            config.python,
            "event_intel_scanner.py",
            "--input-csv",
            config.explosive_output,
            "--ticker-column",
            "ticker",
            "--output",
            config.event_output,
            "--history",
            config.event_history,
        ],
        logger,
        hard_fail=False,
        timeout_seconds=DEFAULT_TIMEOUT_EVENT_SECONDS,
    )


def run_auto_evaluator(config: RunnerConfig, logger: RunLogger) -> int:
    cmd = [
        config.python,
        "explosive_play_evaluator.py",
        "--explosive-history",
        config.explosive_history,
        "--event-history",
        config.event_history,
        "--output",
        config.eval_output,
        "--top-n",
        config.eval_top_n,
    ]
    if config.auto_evaluate:
        cmd.append("--auto-evaluate")
    if config.eval_weights:
        cmd.extend(["--composite-weights", config.eval_weights])
    return run_command(cmd, logger, hard_fail=True, timeout_seconds=DEFAULT_TIMEOUT_EVALUATOR_SECONDS)


def run_daily_pipeline(config: RunnerConfig) -> DailyRunSummary:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_path = Path(config.log_dir) / f"daily_scan_{run_id}.log"
    summary_path = Path(config.summary_dir) / f"daily_summary_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(log_path=log_path, quiet=config.quiet)
    store = SQLiteHistoryStore(config.sqlite_path) if config.enable_sqlite else None

    status = "success"
    success = True
    error_message = ""
    merge_status = ""
    last_completed_stage = "startup"

    explosive_df = pd.DataFrame()
    merged_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    total_historical_rows = 0
    eligible_rows = 0
    skipped_rows = 0

    try:
        show_console_summary = not config.quiet
        logger.log("Starting daily scan workflow", always_console=show_console_summary)

        # 1) explosive scan
        explosive_stage_started = log_stage_start(logger, "explosive_scanner")
        try:
            rc, explosive = run_explosive_scan(config, logger)
        except Exception:
            log_stage_end(logger, "explosive_scanner", explosive_stage_started, status="failed")
            raise
        if rc != 0 or explosive is None:
            log_stage_end(logger, "explosive_scanner", explosive_stage_started, status="failed")
            status = "failed"
            success = False
            error_message = "explosive scanner step failed"
            return DailyRunSummary(
                timestamp_utc=timestamp_utc,
                run_id=run_id,
                status=status,
                success=success,
                error_message=error_message,
                top_candidates=[],
                candidate_counts={"explosive_found": 0, "merged_total": 0},
                evaluation_counts={"historical_total": 0, "eligible": 0, "skipped_insufficient_forward": 0},
                output_files={
                    "explosive_output": config.explosive_output,
                    "event_output": config.event_output,
                    "merged_output": config.merged_output,
                    "evaluation_output": config.eval_output,
                    "summary_output": str(summary_path),
                },
                log_file=str(log_path),
                merge_status="",
                last_completed_stage=last_completed_stage,
            )
        log_stage_end(logger, "explosive_scanner", explosive_stage_started, status="finished")
        explosive_df = explosive
        persist_to_sqlite(store, config.sqlite_table_explosive, explosive_df, run_id, logger)
        last_completed_stage = "explosive_scan"

        # 2) event-intel scan for explosive tickers
        event_stage_started = log_stage_start(logger, "event_intel_scanner")
        try:
            event_rc = run_event_intel_scan(config, explosive_df, logger)
        except Exception:
            log_stage_end(logger, "event_intel_scanner", event_stage_started, status="failed")
            raise
        if event_rc != 0:
            logger.log("Event intel step failed; continuing with graceful merge fallback", always_console=show_console_summary)
            log_stage_end(logger, "event_intel_scanner", event_stage_started, status="failed_nonfatal")
        else:
            log_stage_end(logger, "event_intel_scanner", event_stage_started, status="finished")
        event_df = safe_read_csv(config.event_output, logger)
        persist_to_sqlite(store, config.sqlite_table_event_intel, event_df, run_id, logger)
        last_completed_stage = "event_intel_scan"

        # 3) merge event intel into explosive results
        merge_stage_started = log_stage_start(logger, "merge_event_intel")
        try:
            merged_df, merge_summary = merge_event_intel(explosive_df, config.event_output, composite_weights=None)
        except Exception:
            log_stage_end(logger, "merge_event_intel", merge_stage_started, status="failed")
            raise
        merge_status = merge_summary.message
        merged_df = merged_df.sort_values(by=["total_score", "rel_volume"], ascending=[False, False]).reset_index(drop=True)
        merged_df.to_csv(config.merged_output, index=False)
        append_csv_compatible(merged_df, config.merged_history)
        persist_to_sqlite(store, config.sqlite_table_merged, merged_df, run_id, logger)
        log_stage_end(logger, "merge_event_intel", merge_stage_started, status="finished")
        last_completed_stage = "merge"

        # 4) auto evaluate
        eval_stage_started = log_stage_start(logger, "auto_evaluator")
        try:
            eval_rc = run_auto_evaluator(config, logger)
        except Exception:
            log_stage_end(logger, "auto_evaluator", eval_stage_started, status="failed")
            raise
        if eval_rc != 0:
            status = "failed"
            success = False
            error_message = "evaluator step failed"
            log_stage_end(logger, "auto_evaluator", eval_stage_started, status="failed")
        else:
            log_stage_end(logger, "auto_evaluator", eval_stage_started, status="finished")
        eval_read = safe_read_csv(config.eval_output, logger)
        if eval_read is not None:
            eval_df = eval_read
        persist_to_sqlite(store, config.sqlite_table_evaluation, eval_df, run_id, logger)
        last_completed_stage = "evaluation"

        # 5) summarize counts
        explosive_history_df = safe_read_csv(config.explosive_history, logger)
        if explosive_history_df is not None:
            total_historical_rows = len(explosive_history_df.index)
        eligible_rows = len(eval_df.index)
        skipped_rows = max(0, total_historical_rows - eligible_rows)

        top_candidates = build_top_candidates(merged_df)
        logger.log("", always_console=show_console_summary)
        logger.log("End-of-run summary", always_console=show_console_summary)
        logger.log(f"Total explosive candidates found: {len(explosive_df.index)}", always_console=show_console_summary)
        logger.log(f"Total merged candidates: {len(merged_df.index)}", always_console=show_console_summary)
        logger.log(f"Total historical rows: {total_historical_rows}", always_console=show_console_summary)
        logger.log(f"Rows eligible for evaluation: {eligible_rows}", always_console=show_console_summary)
        logger.log(f"Rows skipped due to insufficient forward data: {skipped_rows}", always_console=show_console_summary)
        logger.log(f"Workflow status: {status}", always_console=show_console_summary)
        logger.log(f"Log file: {log_path}", always_console=True)
        last_completed_stage = "summary"

        return DailyRunSummary(
            timestamp_utc=timestamp_utc,
            run_id=run_id,
            status=status,
            success=success,
            error_message=error_message,
            top_candidates=top_candidates,
            candidate_counts={
                "explosive_found": len(explosive_df.index),
                "merged_total": len(merged_df.index),
            },
            evaluation_counts={
                "historical_total": total_historical_rows,
                "eligible": eligible_rows,
                "skipped_insufficient_forward": skipped_rows,
            },
            output_files={
                "explosive_output": config.explosive_output,
                "event_output": config.event_output,
                "merged_output": config.merged_output,
                "evaluation_output": config.eval_output,
                "summary_output": str(summary_path),
            },
            log_file=str(log_path),
            merge_status=merge_status,
            last_completed_stage=last_completed_stage,
        )
    except Exception as e:
        logger.log(f"Hard failure: {type(e).__name__}: {e}", always_console=True)
        return DailyRunSummary(
            timestamp_utc=timestamp_utc,
            run_id=run_id,
            status="failed",
            success=False,
            error_message=f"{type(e).__name__}: {e}",
            top_candidates=[],
            candidate_counts={"explosive_found": len(explosive_df.index), "merged_total": len(merged_df.index)},
            evaluation_counts={
                "historical_total": total_historical_rows,
                "eligible": eligible_rows,
                "skipped_insufficient_forward": skipped_rows,
            },
            output_files={
                "explosive_output": config.explosive_output,
                "event_output": config.event_output,
                "merged_output": config.merged_output,
                "evaluation_output": config.eval_output,
                "summary_output": str(summary_path),
            },
            log_file=str(log_path),
            merge_status=merge_status,
            last_completed_stage=last_completed_stage,
        )
    finally:
        logger.close()


def write_summary_json(summary: DailyRunSummary) -> None:
    summary_output = summary.output_files.get("summary_output", "")
    if not summary_output:
        return
    path = Path(summary_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)


def write_last_run_status(
    summary: DailyRunSummary,
    started_at_utc: datetime,
    ended_at_utc: datetime,
    path: str = DEFAULT_LAST_RUN_STATUS,
) -> None:
    duration_seconds = max(0.0, (ended_at_utc - started_at_utc).total_seconds())
    output_file_names = sorted(
        {
            Path(v).name
            for v in (summary.output_files or {}).values()
            if str(v or "").strip()
        }
    )
    payload = {
        "start_time_utc": started_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time_utc": ended_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": round(duration_seconds, 3),
        "status": summary.status,
        "success": bool(summary.success),
        "last_completed_stage": summary.last_completed_stage,
        "output_file_names": output_file_names,
        "output_files": summary.output_files,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = parse_args()
    config = resolve_config(args)

    if args.dry_run:
        print("DRY RUN: configuration loaded successfully")
        print(f"DRY RUN: config file = {args.config}")
        print("DRY RUN: pipeline execution skipped")
        return 0

    started_at_utc = datetime.now(timezone.utc)
    summary = run_daily_pipeline(config)
    ended_at_utc = datetime.now(timezone.utc)
    write_summary_json(summary)
    write_last_run_status(summary, started_at_utc, ended_at_utc)
    return 0 if summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
