#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_MARKET_WEIGHT = 0.75
DEFAULT_EVENT_WEIGHT = 0.25


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def build_alignment_bonus(row: pd.Series) -> float:
    bonus = 0.0

    market_score = float(row.get("market_score", 0.0) or 0.0)
    event_score = float(row.get("event_score", 0.0) or 0.0)
    rel_volume = float(row.get("rel_volume", 0.0) or 0.0)
    news_score = float(row.get("news_score", 0.0) or 0.0)
    breakout_score = float(row.get("breakout_score", 0.0) or 0.0)
    days_since_latest_event = row.get("days_since_latest_event")
    event_tags = str(row.get("event_tags", "") or "").lower()

    if market_score >= 12 and event_score >= 3:
        bonus += 1.0
    elif market_score >= 10 and event_score >= 2:
        bonus += 0.5

    if rel_volume >= 2.0 and event_score >= 2.5:
        bonus += 0.5

    if news_score >= 2.5 and ("buy" in event_tags or "open_market_buy" in event_tags):
        bonus += 0.5

    if breakout_score >= 1.0 and event_score >= 2.0:
        bonus += 0.25

    try:
        if pd.notna(days_since_latest_event) and float(days_since_latest_event) <= 5:
            bonus += 0.25
    except Exception:
        pass

    return round(min(bonus, 2.0), 2)


def classify_event_bias(event_tags: str) -> str:
    tags = {t.strip().lower() for t in str(event_tags or "").split(",") if t.strip()}
    bullish = any(t in tags for t in ["open_market_buy", "buy", "cluster_buying", "cluster"])
    bearish = any(t in tags for t in ["open_market_sale", "sell"])
    admin_only = tags and tags.issubset({"administrative", "option_exercise", "tax_withholding", "form4", "insider", "parsed", "unparsed"})

    if admin_only:
        return "Administrative"
    if bullish and bearish:
        return "Mixed"
    if bullish:
        return "Bullish"
    if bearish:
        return "Bearish"
    return "Neutral"


def classify_confidence(row: pd.Series) -> str:
    final_score = float(row.get("final_score", 0.0) or 0.0)
    market_score = float(row.get("market_score", 0.0) or 0.0)
    event_score = float(row.get("event_score", 0.0) or 0.0)
    alignment_bonus = float(row.get("alignment_bonus", 0.0) or 0.0)
    event_bias = str(row.get("event_bias", "Neutral"))

    if final_score >= 13 and market_score >= 11 and event_score >= 3 and event_bias == "Bullish":
        return "A+ Setup"
    if final_score >= 11 and market_score >= 10 and alignment_bonus >= 0.5:
        return "High Alignment"
    if market_score >= 10 and event_score < 2:
        return "Technical First"
    if market_score >= 8 and event_score >= 2.5:
        return "Watchlist"
    if event_score >= 3 and market_score < 8:
        return "Event First"
    return "Low Conviction"


def build_reason(row: pd.Series) -> str:
    parts: list[str] = []
    market_reason = str(row.get("market_reason", "") or "").strip()
    event_reason = str(row.get("event_reason", "") or "").strip()
    event_bias = str(row.get("event_bias", "Neutral"))
    alignment_bonus = float(row.get("alignment_bonus", 0.0) or 0.0)

    if market_reason:
        parts.append(market_reason)
    if event_reason:
        parts.append(event_reason)
    if event_bias != "Neutral":
        parts.append(f"event bias: {event_bias.lower()}")
    if alignment_bonus >= 1.0:
        parts.append("strong multi-signal alignment")
    elif alignment_bonus > 0:
        parts.append("some cross-signal alignment")

    if not parts:
        return "merged market and event signals"
    deduped: list[str] = []
    seen = set()
    for p in parts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return "; ".join(deduped)


def merge_signals(market_df: pd.DataFrame, event_df: pd.DataFrame, market_weight: float, event_weight: float) -> pd.DataFrame:
    market_df = market_df.copy()
    event_df = event_df.copy()

    market_ticker_col = pick_col(market_df, ["ticker"])
    event_ticker_col = pick_col(event_df, ["ticker"])
    market_score_col = pick_col(market_df, ["total_score", "market_score"])
    event_score_col = pick_col(event_df, ["event_intel_score", "insider_score", "event_score"])

    market_df[market_ticker_col] = normalize_ticker_series(market_df[market_ticker_col])
    event_df[event_ticker_col] = normalize_ticker_series(event_df[event_ticker_col])

    market_reason_col = pick_col(market_df, ["reason", "market_reason"], required=False)
    event_reason_col = pick_col(event_df, ["reason", "event_reason"], required=False)
    event_tags_col = pick_col(event_df, ["event_tags"], required=False)
    signal_count_col = pick_col(event_df, ["signal_count"], required=False)
    days_since_col = pick_col(event_df, ["days_since_latest_event"], required=False)

    market_keep = [market_ticker_col, market_score_col]
    for c in ["price", "market_cap", "float_shares", "rel_volume", "news_score", "short_interest_score", "compression_score", "breakout_score", "trend_score", "premarket_gap_score", "repeat_count_30d", market_reason_col]:
        if c and c in market_df.columns and c not in market_keep:
            market_keep.append(c)

    event_keep = [event_ticker_col, event_score_col]
    for c in ["company_name", "sec_insider_score", "congressional_score", "procurement_score", "ownership_score", event_tags_col, signal_count_col, days_since_col, event_reason_col, "confidence_label"]:
        if c and c in event_df.columns and c not in event_keep:
            event_keep.append(c)

    market_small = market_df[market_keep].rename(columns={
        market_ticker_col: "ticker",
        market_score_col: "market_score",
        market_reason_col: "market_reason" if market_reason_col else "reason",
    })
    event_small = event_df[event_keep].rename(columns={
        event_ticker_col: "ticker",
        event_score_col: "event_score",
        event_reason_col: "event_reason" if event_reason_col else "reason",
    })

    merged = market_small.merge(event_small, on="ticker", how="left")

    merged["event_score"] = pd.to_numeric(merged.get("event_score", 0.0), errors="coerce").fillna(0.0)
    merged["market_score"] = pd.to_numeric(merged["market_score"], errors="coerce").fillna(0.0)

    for col in ["rel_volume", "news_score", "breakout_score", "signal_count", "days_since_latest_event"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if "event_tags" not in merged.columns:
        merged["event_tags"] = ""
    if "event_reason" not in merged.columns:
        merged["event_reason"] = ""
    if "market_reason" not in merged.columns:
        merged["market_reason"] = ""

    merged["event_bias"] = merged["event_tags"].apply(classify_event_bias)
    merged["alignment_bonus"] = merged.apply(build_alignment_bonus, axis=1)
    merged["base_weighted_score"] = (merged["market_score"] * market_weight) + (merged["event_score"] * event_weight)
    merged["final_score"] = (merged["base_weighted_score"] + merged["alignment_bonus"]).round(2)
    merged["confidence_label"] = merged.apply(classify_confidence, axis=1)
    merged["merged_reason"] = merged.apply(build_reason, axis=1)
    merged["scan_time_utc"] = utc_now_iso()

    front = [
        "scan_time_utc",
        "ticker",
        "price",
        "market_score",
        "event_score",
        "alignment_bonus",
        "final_score",
        "confidence_label",
        "event_bias",
        "signal_count",
        "days_since_latest_event",
        "event_tags",
        "market_reason",
        "event_reason",
        "merged_reason",
    ]
    ordered = [c for c in front if c in merged.columns] + [c for c in merged.columns if c not in front]
    merged = merged[ordered].sort_values(["final_score", "market_score"], ascending=False)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge explosive play market signals with event-intel signals.")
    parser.add_argument("--market-csv", default="explosive_play_candidates.csv", help="Path to market scanner CSV.")
    parser.add_argument("--event-csv", default="event_intel_scan.csv", help="Path to event-intel CSV.")
    parser.add_argument("--output", default="merged_watchlist.csv", help="Path to merged output CSV.")
    parser.add_argument("--history", default="merged_watchlist_history.csv", help="Optional history CSV append path.")
    parser.add_argument("--market-weight", type=float, default=DEFAULT_MARKET_WEIGHT)
    parser.add_argument("--event-weight", type=float, default=DEFAULT_EVENT_WEIGHT)
    parser.add_argument("--top", type=int, default=25, help="How many rows to print to console.")
    args = parser.parse_args()

    total_weight = args.market_weight + args.event_weight
    if total_weight <= 0:
        print("Weights must sum to more than zero.", file=sys.stderr)
        return 1

    market_weight = args.market_weight / total_weight
    event_weight = args.event_weight / total_weight

    try:
        market_df = safe_read_csv(args.market_csv)
        event_df = safe_read_csv(args.event_csv)
        merged = merge_signals(market_df, event_df, market_weight, event_weight)
    except Exception as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1

    print("Top merged watchlist results:\n")
    preview_cols = [c for c in [
        "ticker", "market_score", "event_score", "alignment_bonus", "final_score",
        "confidence_label", "event_bias", "signal_count", "days_since_latest_event", "merged_reason"
    ] if c in merged.columns]
    print(merged[preview_cols].head(args.top).to_string(index=False))

    out_path = Path(args.output)
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged output to: {out_path}")

    if args.history:
        hist_path = Path(args.history)
        write_header = not hist_path.exists()
        merged.to_csv(hist_path, mode="a", index=False, header=write_header)
        print(f"Appended merged results to history file: {hist_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
