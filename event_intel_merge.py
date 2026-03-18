from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


EVENT_COLUMNS = [
    "ticker",
    "scan_time_utc",
    "event_intel_score",
    "sec_insider_score",
    "anomaly_score",
    "anomaly_label",
    "anomaly_flags",
    "baseline_sample_size",
    "event_tags",
    "reason",
]


@dataclass
class MergeSummary:
    event_rows: int
    merged_rows: int
    event_path: str
    used_composite: bool
    message: str


def _prepare_event_df(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "ticker" not in df.columns:
        return None

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # keep only the latest scan per ticker
    if "scan_time_utc" in df.columns:
        df["scan_time_utc_parsed"] = pd.to_datetime(df["scan_time_utc"], errors="coerce")
        df = df.sort_values("scan_time_utc_parsed").drop_duplicates(subset=["ticker"], keep="last")

    keep_cols = [c for c in EVENT_COLUMNS if c in df.columns]
    df = df[keep_cols]

    # rename to avoid clobbering explosive_play reason column
    if "reason" in df.columns:
        df = df.rename(columns={"reason": "event_intel_reason"})

    if "event_tags" not in df.columns:
        df["event_tags"] = ""

    return df


def _apply_composite_score(
    df: pd.DataFrame,
    weights: Tuple[float, float, float],
) -> pd.DataFrame:
    total_w, event_w, anomaly_w = weights
    if total_w == 0 and event_w == 0 and anomaly_w == 0:
        return df

    def _score(row: pd.Series) -> float:
        total = float(row.get("total_score", 0.0) or 0.0)
        event = float(row.get("event_intel_score", 0.0) or 0.0)
        anomaly = float(row.get("anomaly_score", 0.0) or 0.0)
        return (total * total_w) + (event * event_w) + (anomaly * anomaly_w)

    df["composite_score"] = df.apply(_score, axis=1).round(2)
    return df


def merge_event_intel(
    base_df: pd.DataFrame,
    event_csv_path: str,
    composite_weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[pd.DataFrame, MergeSummary]:
    base = base_df.copy()
    if "ticker" in base.columns:
        base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()

    event_df = _prepare_event_df(event_csv_path)
    if event_df is None or event_df.empty:
        for col in [
            "event_intel_score",
            "sec_insider_score",
            "anomaly_score",
            "anomaly_label",
            "anomaly_flags",
            "baseline_sample_size",
            "event_tags",
            "event_intel_reason",
        ]:
            if col not in base.columns:
                base[col] = pd.NA
        summary = MergeSummary(
            event_rows=0,
            merged_rows=len(base.index),
            event_path=event_csv_path,
            used_composite=False,
            message=f"event-intel data not available at {event_csv_path}; left-joined empty columns",
        )
        return base, summary

    # If base rows already have event-intel columns from a prior merge,
    # drop only overlapping event columns so latest event_df values remain canonical.
    overlap_cols = [c for c in event_df.columns if c != "ticker" and c in base.columns]
    if overlap_cols:
        base = base.drop(columns=overlap_cols)

    merged = base.merge(event_df, on="ticker", how="left")

    for col in [
        "event_intel_score",
        "sec_insider_score",
        "anomaly_score",
        "anomaly_label",
        "anomaly_flags",
        "baseline_sample_size",
        "event_tags",
        "event_intel_reason",
    ]:
        if col not in merged.columns:
            merged[col] = pd.NA

    used_composite = False
    if composite_weights is not None:
        merged = _apply_composite_score(merged, composite_weights)
        used_composite = True

    summary = MergeSummary(
        event_rows=len(event_df.index),
        merged_rows=len(merged.index),
        event_path=event_csv_path,
        used_composite=used_composite,
        message="merged event-intel data",
    )
    return merged, summary
