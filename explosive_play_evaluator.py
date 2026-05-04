#/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Force yfinance timezone cache into a local writable folder.
# This avoids sqlite cache writes in restricted profile locations.
LOCAL_TZ_CACHE = Path("price_cache")
LOCAL_TZ_CACHE.mkdir(parents=True, exist_ok=True)
try:
    # Applies to cookie/ISIN sqlite-backed caches in newer yfinance builds.
    if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
        yf.cache.set_cache_location(str(LOCAL_TZ_CACHE))
    # Explicit timezone cache path for compatibility with older/newer builds.
    yf.set_tz_cache_location(str(LOCAL_TZ_CACHE))
except Exception:
    pass


# Defaults mirror existing scanner outputs
DEFAULT_EXPLOSIVE_HISTORY = "explosive_play_historical_scans.csv"
DEFAULT_EVENT_HISTORY = "event_intel_history.csv"
DEFAULT_OUTPUT = "explosive_play_eval.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate explosive play rankings with optional event-intel enrichment."
    )
    parser.add_argument("--explosive-history", default=DEFAULT_EXPLOSIVE_HISTORY, help="Path to explosive_play_historical_scans.csv")
    parser.add_argument("--event-history", default=DEFAULT_EVENT_HISTORY, help="Path to event_intel_history.csv (for enrichment)")
    parser.add_argument("--start-date", help="Filter scans on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Filter scans on/before this date (YYYY-MM-DD)")
    parser.add_argument("--composite-weights", help="Weights as total,event,anomaly. Example: 0.65,0.25,0.10")
    parser.add_argument("--top-n", default="5,10,20", help="Comma list for hit-rate and return summaries")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Evaluation CSV output path")
    parser.add_argument("--max-forward-days", type=int, default=5, help="Max forward window (trading days) to fetch")
    parser.add_argument("--auto-evaluate", action="store_true", help="Evaluate all historical rows where 5 trading days of forward data are available")
    return parser.parse_args()


def parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.to_datetime(value).normalize()
    except Exception:
        return None


def load_explosive_history(path: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scan_date" not in df.columns:
        raise ValueError("explosive history missing scan_date column")
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["scan_date"])
    if start is not None:
        df = df[df["scan_date"] >= start]
    if end is not None:
        df = df[df["scan_date"] <= end]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df.reset_index(drop=True)


def load_event_history(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "ticker" not in df.columns or "scan_time_utc" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    scan_ts = pd.to_datetime(df["scan_time_utc"], errors="coerce")
    if getattr(scan_ts.dt, "tz", None) is None:
        scan_ts = scan_ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    df["scan_date"] = scan_ts.dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=["scan_date"])
    df = df.sort_values("scan_time_utc")
    return df


def select_event_row(event_df: pd.DataFrame, ticker: str, scan_date: pd.Timestamp) -> Optional[pd.Series]:
    if event_df.empty:
        return None
    subset = event_df[(event_df["ticker"] == ticker) & (event_df["scan_date"] <= scan_date)]
    if subset.empty:
        return None
    return subset.iloc[-1]


def download_price_cache(tickers: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    start_str = (start - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end_str = (end + pd.Timedelta(days=8)).strftime("%Y-%m-%d")

    def log_empty_reason(symbol: str) -> None:
        try:
            # raise_errors=True surfaces the underlying failure cause when available.
            yf.Ticker(symbol).history(
                start=start_str,
                end=end_str,
                interval="1d",
                raise_errors=True,
            )
            print(f"[WARN] {symbol}: empty price response from yfinance")
        except Exception as err:
            print(f"[WARN] {symbol}: yfinance download failed: {type(err).__name__}: {err}")

    for ticker in sorted(set(tickers)):
        try:
            df = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval="1d",
                progress=False,
                auto_adjust=False,
                prepost=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna()
            if not df.empty:
                df["date"] = pd.to_datetime(df.index).normalize()
                cache[ticker] = df
            else:
                log_empty_reason(ticker)
        except Exception as e:
            print(f"[WARN] {ticker}: yfinance download failed: {type(e).__name__}: {e}")
            continue
    return cache


def forward_returns(prices: pd.DataFrame, scan_date: pd.Timestamp, horizons: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if prices.empty:
        for h in horizons:
            out[f"fwd{h}d_close"] = math.nan
            out[f"fwd{h}d_max"] = math.nan
        out["hit_5pct_5d"] = out["hit_10pct_5d"] = out["hit_15pct_5d"] = math.nan
        return out

    after = prices[prices["date"] >= scan_date].reset_index(drop=True)
    if after.empty:
        for h in horizons:
            out[f"fwd{h}d_close"] = math.nan
            out[f"fwd{h}d_max"] = math.nan
        out["hit_5pct_5d"] = out["hit_10pct_5d"] = out["hit_15pct_5d"] = math.nan
        return out

    close0 = float(after["Close"].iloc[0])
    high_after = after["High"].astype(float).values
    closes = after["Close"].astype(float).values

    for h in horizons:
        if len(closes) > h:
            ret_close = (closes[h] - close0) / close0
            max_move = (max(high_after[1 : h + 1]) - close0) / close0 if h >= 1 else 0.0
        else:
            ret_close = math.nan
            max_move = math.nan
        out[f"fwd{h}d_close"] = round(ret_close, 4) if not math.isnan(ret_close) else math.nan
        out[f"fwd{h}d_max"] = round(max_move, 4) if not math.isnan(max_move) else math.nan

    window = min(len(high_after) - 1, max(horizons))
    window_high = max(high_after[1 : window + 1]) if window >= 1 else float("nan")
    if math.isnan(window_high):
        out["hit_5pct_5d"] = out["hit_10pct_5d"] = out["hit_15pct_5d"] = math.nan
    else:
        move = (window_high - close0) / close0
        out["hit_5pct_5d"] = move >= 0.05
        out["hit_10pct_5d"] = move >= 0.10
        out["hit_15pct_5d"] = move >= 0.15
    return out


def has_forward_window(prices: pd.DataFrame, scan_date: pd.Timestamp, horizon_days: int = 5) -> bool:
    if prices.empty:
        return False
    after = prices[prices["date"] >= scan_date]
    # Need scan-day close + N trading closes to compute close-to-close N-day return.
    return len(after.index) > horizon_days


def spearman_corr(score: pd.Series, target: pd.Series) -> float:
    s = score.dropna()
    t = target.dropna()
    aligned = s.to_frame("s").join(t.to_frame("t"), how="inner")
    if len(aligned) < 3:
        return float("nan")
    return aligned["s"].rank().corr(aligned["t"].rank(), method="pearson")


def suggest_regime_weights(eval_df: pd.DataFrame, window_days: int = 90) -> dict:
    """Return suggested composite weights based on rolling Spearman performance."""
    if len(eval_df) < 30 or "composite_score" not in eval_df.columns:
        return {"total": 0.65, "event": 0.25, "anomaly": 0.10, "regime": "insufficient_data"}
    
    eval_df = eval_df.copy()
    eval_df["scan_date"] = pd.to_datetime(eval_df["scan_date"])
    cutoff = eval_df["scan_date"].max() - pd.Timedelta(days=window_days)
    recent = eval_df[eval_df["scan_date"] >= cutoff].copy()
    
    if recent.empty:
        return {"total": 0.65, "event": 0.25, "anomaly": 0.10, "regime": "insufficient_data"}
    
    # Simple regime proxy
    avg_hit10 = recent["hit_10pct_5d"].mean(skipna=True)
    regime = "high_vol" if avg_hit10 < 0.25 else "normal"
    
    # Grid search best weights
    best_corr = -1.0
    best_w = (0.65, 0.25, 0.10)
    for t in [0.5, 0.6, 0.65, 0.7, 0.75]:
        for e in [0.15, 0.2, 0.25, 0.3, 0.35]:
            for a in [0.05, 0.1, 0.15]:
                if abs(t + e + a - 1.0) > 0.01:
                    continue
                recent["test_comp"] = (recent["total_score"] * t +
                                       recent.get("event_intel_score", pd.Series(0)).fillna(0) * e +
                                       recent.get("anomaly_score", pd.Series(0)).fillna(0) * a)
                corr = spearman_corr(recent["test_comp"], recent["fwd5d_close"])
                if not math.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_w = (t, e, a)
    
    return {
        "total": round(best_w[0], 2),
        "event": round(best_w[1], 2),
        "anomaly": round(best_w[2], 2),
        "regime": regime,
        "spearman_corr": round(best_corr, 3),
        "sample_size": len(recent)
    }


def summarize_top_n(df: pd.DataFrame, score_col: str, top_ns: Sequence[int]) -> List[str]:
    lines: List[str] = []
    for n in top_ns:
        head = df.sort_values(score_col, ascending=False).head(n)
        if head.empty:
            lines.append(f"Top {n} on {score_col}: no data")
            continue
        hit = head["hit_10pct_5d"].mean(skipna=True)
        avg = head["fwd5d_close"].mean(skipna=True)
        med = head["fwd5d_close"].median(skipna=True)
        lines.append(f"Top {n} {score_col}: hit10= {hit:.2%} avg5d= {avg:.2%} med5d= {med:.2%}")
    return lines


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if args.auto_evaluate and (args.start_date or args.end_date):
        print("[INFO] --auto-evaluate uses full history; ignoring --start-date/--end-date filters.")
        start = None
        end = None

    top_ns = [int(x) for x in args.top_n.split(",") if x.strip().isdigit()]
    horizons = [1, 3, 5]

    comp_weights: Optional[Tuple[float, float, float]] = None
    if args.composite_weights:
        parts = [p.strip() for p in args.composite_weights.split(",")]
        if len(parts) != 3:
            raise ValueError("Composite weights must be three comma-separated numbers")
        comp_weights = tuple(float(p) for p in parts)  # type: ignore[assignment]

    explosive_df = load_explosive_history(args.explosive_history, start, end)
    if explosive_df.empty:
        raise SystemExit("No explosive history rows in the requested window.")
    total_historical_rows = len(explosive_df.index)

    event_df = load_event_history(args.event_history)

    tickers = explosive_df["ticker"].unique().tolist()
    price_cache = download_price_cache(tickers, explosive_df["scan_date"].min(), explosive_df["scan_date"].max() + pd.Timedelta(days=args.max_forward_days))

    records: List[dict] = []
    skipped_insufficient_forward = 0
    for _, row in explosive_df.iterrows():
        ticker = row["ticker"]
        scan_date = row["scan_date"]
        base_score = float(row.get("total_score", 0.0) or 0.0)

        event_row = select_event_row(event_df, ticker, scan_date)
        event_score = float(event_row["event_intel_score"]) if event_row is not None and not pd.isna(event_row.get("event_intel_score")) else math.nan
        anomaly_score = float(event_row["anomaly_score"]) if event_row is not None and not pd.isna(event_row.get("anomaly_score")) else math.nan
        sec_score = float(event_row["sec_insider_score"]) if event_row is not None and not pd.isna(event_row.get("sec_insider_score")) else math.nan

        if comp_weights:
            comp_total, comp_event, comp_anom = comp_weights
            composite = base_score * comp_total
            if not math.isnan(event_score):
                composite += event_score * comp_event
            if not math.isnan(anomaly_score):
                composite += anomaly_score * comp_anom
        else:
            composite = math.nan

        prices = price_cache.get(ticker, pd.DataFrame())
        if args.auto_evaluate and not has_forward_window(prices, scan_date, horizon_days=5):
            skipped_insufficient_forward += 1
            continue

        returns = forward_returns(prices, scan_date, horizons)
        if args.auto_evaluate and pd.isna(returns.get("fwd5d_close")):
            skipped_insufficient_forward += 1
            continue

        rec = {
            "scan_date": scan_date,
            "ticker": ticker,
            "total_score": base_score,
            "composite_score": round(composite, 2) if not math.isnan(composite) else math.nan,
            "event_intel_score": event_score,
            "sec_insider_score": sec_score,
            "anomaly_score": anomaly_score,
        }
        rec.update(returns)
        records.append(rec)

    eval_columns = [
        "scan_date",
        "ticker",
        "total_score",
        "composite_score",
        "event_intel_score",
        "sec_insider_score",
        "anomaly_score",
        "fwd1d_close",
        "fwd1d_max",
        "fwd3d_close",
        "fwd3d_max",
        "fwd5d_close",
        "fwd5d_max",
        "hit_5pct_5d",
        "hit_10pct_5d",
        "hit_15pct_5d",
    ]
    eval_df = pd.DataFrame(records)
    if eval_df.empty:
        eval_df = pd.DataFrame(columns=eval_columns + ["base_rank", "merged_rank"])
        corr_base = float("nan")
        corr_merge = float("nan")
    else:
        eval_df["base_rank"] = eval_df["total_score"].rank(method="dense", ascending=False)
        eval_df["merged_rank"] = eval_df["composite_score"].rank(method="dense", ascending=False)
        corr_base = spearman_corr(eval_df["total_score"], eval_df["fwd5d_close"])
        corr_merge = spearman_corr(eval_df["composite_score"], eval_df["fwd5d_close"]) if comp_weights else float("nan")

    eval_df.to_csv(args.output, index=False)

    print("\nEvaluation summary")
    print(f"Rows: {len(eval_df)} | Date window: {start.date() if start else 'min'} to {end.date() if end else 'max'}")
    print(f"Total historical scans: {total_historical_rows}")
    print(f"Rows eligible for evaluation: {len(eval_df)}")
    print(f"Rows skipped due to insufficient forward data: {skipped_insufficient_forward}")
    print(f"Composite weights: {comp_weights if comp_weights else 'not provided; merged ranking not evaluated'}")
    print(f"Spearman corr (base total_score vs 5d close): {corr_base:.3f}" if not math.isnan(corr_base) else "Spearman corr base: n/a")
    if comp_weights:
        print(f"Spearman corr (composite vs 5d close): {corr_merge:.3f}" if not math.isnan(corr_merge) else "Spearman corr composite: n/a")

    # === NEW: Regime-adaptive weight suggestion ===
    if args.auto_evaluate and not eval_df.empty:
        suggested = suggest_regime_weights(eval_df)
        print("\n=== REGIME-ADAPTIVE WEIGHTS SUGGESTED ===")
        print(f"Suggested weights: total={suggested['total']}, event={suggested['event']}, anomaly={suggested['anomaly']}")
        print(f"Regime: {suggested['regime']} | Spearman: {suggested['spearman_corr']:.3f} | Sample: {suggested['sample_size']}")
        # Optionally write to a file for daily_scan_runner to pick up next time
        weights_path = Path("suggested_weights.json")
        import json
        with weights_path.open("w") as f:
            json.dump(suggested, f, indent=2)
        print(f"Wrote suggested_weights.json")

    if comp_weights:
        print("\nHit-rate / return snapshots:")
        for line in summarize_top_n(eval_df, "total_score", top_ns):
            print("Base " + line)
        for line in summarize_top_n(eval_df, "composite_score", top_ns):
            print("Merge " + line)
    else:
        print("\nHit-rate summary skipped (composite weights not supplied).")

    print(f"\nWrote detailed evaluation rows to: {args.output}")


if __name__ == "__main__":
    main()
