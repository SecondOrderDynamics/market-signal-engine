#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from daily_scan_runner import resolve_config
from history_storage import SQLiteHistoryStore


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local smoke test for daily runner config, filesystem paths, and optional SQLite wiring."
    )
    parser.add_argument("--config", default="daily_scan_config.json", help="Path to daily runner config JSON.")
    parser.add_argument("--python", default=None, help="Override python executable.")
    parser.add_argument("--enable-sqlite", action="store_true", help="Force-enable SQLite checks.")
    parser.add_argument("--sqlite-path", default=None, help="Override sqlite DB path.")
    parser.add_argument("--quiet", action="store_true", help="Print only failures and final status.")
    return parser.parse_args()


def to_runner_namespace(args: argparse.Namespace) -> SimpleNamespace:
    # Mirror daily_scan_runner CLI namespace fields so resolve_config can be reused.
    return SimpleNamespace(
        config=args.config,
        python=args.python,
        explosive_output=None,
        explosive_history=None,
        event_output=None,
        event_history=None,
        merged_output=None,
        merged_history=None,
        eval_output=None,
        eval_weights=None,
        eval_top_n=None,
        log_dir=None,
        summary_dir=None,
        quiet=False,
        enable_sqlite=args.enable_sqlite,
        sqlite_path=args.sqlite_path,
    )


def check_python_executable(python_exe: str) -> CheckResult:
    try:
        proc = subprocess.run([python_exe, "--version"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return CheckResult("python_executable", False, f"python returned exit code {proc.returncode}")
        version_text = (proc.stdout or proc.stderr or "").strip()
        return CheckResult("python_executable", True, version_text or "python executable is reachable")
    except Exception as e:
        return CheckResult("python_executable", False, f"failed to run python executable: {e}")


def check_required_scripts(base_dir: Path) -> CheckResult:
    required = [
        "daily_scan_runner.py",
        "explosive_play_scanner.py",
        "event_intel_scanner.py",
        "explosive_play_evaluator.py",
        "event_intel_merge.py",
    ]
    missing = [name for name in required if not (base_dir / name).exists()]
    if missing:
        return CheckResult("required_scripts", False, f"missing: {', '.join(missing)}")
    return CheckResult("required_scripts", True, "all required scripts are present")


def ensure_dir_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="smoke_", suffix=".tmp", dir=str(path), delete=True) as _:
            pass
        return True, f"writable directory: {path}"
    except Exception as e:
        return False, f"not writable: {path} ({e})"


def check_parent_writable(path_str: str) -> tuple[bool, str]:
    p = Path(path_str)
    parent = p.parent if str(p.parent) not in ("", ".") else Path(".")
    return ensure_dir_writable(parent)


def check_paths(config) -> list[CheckResult]:
    results: list[CheckResult] = []

    ok, detail = ensure_dir_writable(Path(config.log_dir))
    results.append(CheckResult("log_dir", ok, detail))

    ok, detail = ensure_dir_writable(Path(config.summary_dir))
    results.append(CheckResult("summary_dir", ok, detail))

    file_targets = {
        "explosive_output_parent": config.explosive_output,
        "explosive_history_parent": config.explosive_history,
        "event_output_parent": config.event_output,
        "event_history_parent": config.event_history,
        "merged_output_parent": config.merged_output,
        "merged_history_parent": config.merged_history,
        "eval_output_parent": config.eval_output,
    }
    for name, target in file_targets.items():
        ok, detail = check_parent_writable(target)
        results.append(CheckResult(name, ok, detail))
    return results


def check_sqlite_wiring(config) -> CheckResult:
    if not config.enable_sqlite:
        return CheckResult("sqlite_wiring", True, "sqlite disabled; skipped")
    try:
        store = SQLiteHistoryStore(config.sqlite_path)
        payload = pd.DataFrame(
            [
                {
                    "smoke_key": "runner_config",
                    "smoke_value": "ok",
                }
            ]
        )
        inserted = store.append_dataframe(table_name="smoke_test_runs", df=payload, run_id="smoke")
        return CheckResult("sqlite_wiring", True, f"sqlite append ok ({inserted} row) at {config.sqlite_path}")
    except Exception as e:
        return CheckResult("sqlite_wiring", False, f"sqlite check failed: {e}")


def print_results(results: list[CheckResult], quiet: bool) -> None:
    for r in results:
        if quiet and r.ok:
            continue
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")

    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    print(f"\nSmoke test summary: {passed} passed, {failed} failed, total {len(results)} checks")


def main() -> int:
    args = parse_args()
    runner_ns = to_runner_namespace(args)
    config = resolve_config(runner_ns)

    base_dir = Path(".")
    results: list[CheckResult] = []
    results.append(check_python_executable(config.python))
    results.append(check_required_scripts(base_dir))
    results.extend(check_paths(config))
    results.append(check_sqlite_wiring(config))

    print_results(results, quiet=args.quiet)
    failed = [r for r in results if not r.ok]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
