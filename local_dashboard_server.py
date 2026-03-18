#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


class DashboardDataService:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.logs_dir = self.workspace / "logs"

    def _summary_files(self) -> list[Path]:
        if not self.logs_dir.exists():
            return []
        return sorted(
            self.logs_dir.glob("daily_summary_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def latest_summary(self) -> dict[str, Any]:
        files = self._summary_files()
        if not files:
            return {}
        try:
            return json.loads(files[0].read_text(encoding="utf-8"))
        except Exception:
            return {}

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in self._summary_files()[: max(limit, 1)]:
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(
                {
                    "run_id": payload.get("run_id", ""),
                    "timestamp_utc": payload.get("timestamp_utc", ""),
                    "status": payload.get("status", "unknown"),
                    "success": bool(payload.get("success", False)),
                    "explosive_found": int((payload.get("candidate_counts") or {}).get("explosive_found", 0) or 0),
                    "merged_total": int((payload.get("candidate_counts") or {}).get("merged_total", 0) or 0),
                    "eligible": int((payload.get("evaluation_counts") or {}).get("eligible", 0) or 0),
                    "skipped": int((payload.get("evaluation_counts") or {}).get("skipped_insufficient_forward", 0) or 0),
                    "summary_file": str(p),
                    "log_file": str(payload.get("log_file", "")),
                }
            )
        return out

    def _resolve_output_path(self, summary: dict[str, Any], output_key: str, fallback_name: str) -> Path:
        output_files = summary.get("output_files") or {}
        raw = str(output_files.get(output_key, fallback_name) or fallback_name)
        p = Path(raw)
        return p if p.is_absolute() else self.workspace / p

    def merged_watchlist_preview(self, limit: int = 25) -> list[dict[str, Any]]:
        summary = self.latest_summary()
        merged_path = self._resolve_output_path(summary, "merged_output", "merged_watchlist.csv")
        if not merged_path.exists():
            return []
        try:
            df = pd.read_csv(merged_path)
        except Exception:
            return []
        cols = [c for c in ["ticker", "total_score", "event_intel_score", "anomaly_score", "anomaly_label"] if c in df.columns]
        if not cols:
            return []
        preview = df.sort_values(by=[c for c in ["total_score", "rel_volume"] if c in df.columns], ascending=False).head(max(limit, 1))
        return preview[cols].fillna("").to_dict(orient="records")

    def evaluation_overview(self) -> dict[str, Any]:
        summary = self.latest_summary()
        eval_path = self._resolve_output_path(summary, "evaluation_output", "explosive_play_eval.csv")
        if not eval_path.exists():
            return {
                "rows": 0,
                "eligible_rows": int((summary.get("evaluation_counts") or {}).get("eligible", 0) or 0),
                "skipped_rows": int((summary.get("evaluation_counts") or {}).get("skipped_insufficient_forward", 0) or 0),
                "mean_fwd5d_close": None,
                "median_fwd5d_close": None,
            }
        try:
            df = pd.read_csv(eval_path)
        except Exception:
            return {
                "rows": 0,
                "eligible_rows": int((summary.get("evaluation_counts") or {}).get("eligible", 0) or 0),
                "skipped_rows": int((summary.get("evaluation_counts") or {}).get("skipped_insufficient_forward", 0) or 0),
                "mean_fwd5d_close": None,
                "median_fwd5d_close": None,
            }

        fwd = pd.to_numeric(df.get("fwd5d_close"), errors="coerce")
        mean_val = None if fwd.dropna().empty else round(float(fwd.dropna().mean()), 4)
        med_val = None if fwd.dropna().empty else round(float(fwd.dropna().median()), 4)
        return {
            "rows": int(len(df.index)),
            "eligible_rows": int((summary.get("evaluation_counts") or {}).get("eligible", 0) or 0),
            "skipped_rows": int((summary.get("evaluation_counts") or {}).get("skipped_insufficient_forward", 0) or 0),
            "mean_fwd5d_close": mean_val,
            "median_fwd5d_close": med_val,
        }

    def files_status(self) -> dict[str, Any]:
        summary = self.latest_summary()
        checks = {
            "explosive_output": self._resolve_output_path(summary, "explosive_output", "explosive_play_candidates.csv"),
            "event_output": self._resolve_output_path(summary, "event_output", "event_intel_scan.csv"),
            "merged_output": self._resolve_output_path(summary, "merged_output", "merged_watchlist.csv"),
            "evaluation_output": self._resolve_output_path(summary, "evaluation_output", "explosive_play_eval.csv"),
        }
        out: dict[str, Any] = {}
        for key, p in checks.items():
            out[key] = {
                "path": str(p),
                "exists": bool(p.exists()),
                "last_modified": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if p.exists()
                else None,
                "size_bytes": int(p.stat().st_size) if p.exists() else 0,
            }
        return out


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, data_service: DashboardDataService, **kwargs) -> None:
        self.data_service = data_service
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
            return
        super().do_GET()

    def _handle_api(self, parsed) -> None:
        query = parse_qs(parsed.query or "")
        try:
            if parsed.path == "/api/health":
                self._write_json(
                    {
                        "ok": True,
                        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                return

            if parsed.path == "/api/latest-run":
                self._write_json(self.data_service.latest_summary())
                return

            if parsed.path == "/api/runs":
                limit = int(query.get("limit", ["20"])[0] or 20)
                self._write_json({"runs": self.data_service.recent_runs(limit=limit)})
                return

            if parsed.path == "/api/top-candidates":
                limit = int(query.get("limit", ["10"])[0] or 10)
                latest = self.data_service.latest_summary()
                top = (latest.get("top_candidates") or [])[: max(limit, 1)]
                self._write_json({"top_candidates": top})
                return

            if parsed.path == "/api/merged-watchlist":
                limit = int(query.get("limit", ["25"])[0] or 25)
                self._write_json({"rows": self.data_service.merged_watchlist_preview(limit=limit)})
                return

            if parsed.path == "/api/evaluation":
                self._write_json(self.data_service.evaluation_overview())
                return

            if parsed.path == "/api/files":
                self._write_json({"files": self.data_service.files_status()})
                return

            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as e:
            self._write_json({"error": f"{type(e).__name__}: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _write_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local API + HTML dashboard for daily scan outputs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind.")
    parser.add_argument("--workspace", default=".", help="Workspace directory containing logs and CSV outputs.")
    parser.add_argument("--ui-dir", default="dashboard_ui", help="Directory containing static dashboard files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    ui_dir = Path(args.ui_dir).resolve()

    if not ui_dir.exists():
        print(f"UI directory not found: {ui_dir}")
        return 2

    data_service = DashboardDataService(workspace=workspace)
    handler = partial(
        DashboardRequestHandler,
        directory=str(ui_dir),
        data_service=data_service,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Local dashboard running at http://{args.host}:{args.port}")
    print(f"Serving UI from: {ui_dir}")
    print(f"Reading data from workspace: {workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
