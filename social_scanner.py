from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

USER_AGENT = os.getenv(
    "SOCIAL_SCANNER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)
REQUEST_TIMEOUT = int(os.getenv("SOCIAL_SCANNER_TIMEOUT", "20"))
DEFAULT_OUTPUT = "social_scan.csv"
DEFAULT_HISTORY = "social_scan_history.csv"

BULLISH_TERMS = {
    "breakout", "squeeze", "rip", "runner", "bullish", "buy", "long", "calls",
    "uptrend", "reversal", "beat", "guidance", "upgrade", "accumulation", "moon",
    "swing", "momentum", "strong", "green", "squeeze", "squeezing"
}
BEARISH_TERMS = {
    "dump", "offering", "dilution", "bearish", "short", "puts", "rug", "fade",
    "downtrend", "miss", "downgrade", "sell", "red", "collapse", "bagholder", "zero"
}
NARRATIVE_KEYWORDS = {
    "ai": ["ai", "artificial intelligence", "inference", "gpu"],
    "earnings": ["earnings", "eps", "guidance", "revenue"],
    "squeeze": ["squeeze", "short interest", "covering", "gamma"],
    "offering": ["offering", "dilution", "atm", "shelf"],
    "fda": ["fda", "clinical", "phase 1", "phase 2", "phase 3", "drug"],
    "merger": ["merger", "acquisition", "buyout", "m&a"],
    "defense": ["defense", "dod", "contract", "army", "air force", "navy"],
    "ev": ["ev", "electric vehicle", "tesla", "battery"],
}


@dataclass
class SocialPost:
    source: str
    body: str
    created_at: datetime
    likes: int = 0
    replies: int = 0
    reposts: int = 0
    followers: int = 0
    author: str = ""
    raw_weight: float = 1.0


@dataclass
class SocialScanResult:
    scan_time_utc: str
    ticker: str
    mention_count: int
    current_window_mentions: int
    baseline_window_mentions: int
    mention_velocity: float
    sentiment_score: float
    engagement_score: float
    source_diversity_score: float
    influencer_score: float
    spam_penalty: float
    narrative_score: float
    social_pressure_score: float
    confidence_label: str
    narrative_tags: str
    top_sources: str
    reason: str


class BaseAdapter:
    source_name = "base"

    def fetch_posts(self, ticker: str, debug: bool = False) -> List[SocialPost]:
        raise NotImplementedError



def safe_request(url: str, params: Optional[dict] = None, debug: bool = False) -> Optional[requests.Response]:
    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://stocktwits.com/",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if debug:
            print(f"[DEBUG] GET {resp.url} -> {resp.status_code} | {resp.headers.get('Content-Type')}")
        resp.raise_for_status()
        return resp
    except Exception as e:
        if debug:
            print(f"[DEBUG] Request failed for {url}: {e}")
        return None


class StocktwitsAdapter(BaseAdapter):
    source_name = "stocktwits"

    def fetch_posts(self, ticker: str, debug: bool = False) -> List[SocialPost]:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json"
        resp = safe_request(url, debug=debug)
        if resp is None:
            return []

        try:
            payload = resp.json()
            if debug:
                print(f"[DEBUG] {ticker.upper()} payload keys: {list(payload.keys())[:10]}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] JSON parse failed for {ticker.upper()}: {e}")
                print(resp.text[:500])
            return []

        messages = payload.get("messages", []) or []
        if debug:
            print(f"[DEBUG] {ticker.upper()} raw message count: {len(messages)}")

        posts: List[SocialPost] = []
        for msg in messages:
            body = str(msg.get("body", "") or "")
            if not body.strip():
                continue

            created_raw = msg.get("created_at")
            created_at = parse_datetime(created_raw)
            user = msg.get("user") or {}
            likes_obj = msg.get("likes") or {}

            posts.append(
                SocialPost(
                    source=self.source_name,
                    body=body,
                    created_at=created_at,
                    likes=int(likes_obj.get("total", 0) or 0),
                    replies=int(msg.get("conversation", {}).get("reply_count", 0) or 0),
                    reposts=int(msg.get("reshares", {}).get("reshared_count", 0) or 0),
                    followers=int(user.get("followers", 0) or 0),
                    author=str(user.get("username", "") or ""),
                    raw_weight=1.0,
                )
            )
        return posts


class RedditAdapter(BaseAdapter):
    source_name = "reddit"

    def fetch_posts(self, ticker: str, debug: bool = False) -> List[SocialPost]:
        return []


class XAdapter(BaseAdapter):
    source_name = "x"

    def fetch_posts(self, ticker: str, debug: bool = False) -> List[SocialPost]:
        return []



def parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)



def score_mention_velocity(current_mentions: int, baseline_mentions: int) -> tuple[float, float]:
    if current_mentions <= 0:
        return 0.0, 0.0
    effective_baseline = max(baseline_mentions, 1)
    ratio = current_mentions / effective_baseline
    if ratio < 1.5:
        score = 1.0 if current_mentions > 0 else 0.0
    elif ratio < 2.0:
        score = 3.0
    elif ratio < 3.0:
        score = 5.0
    elif ratio < 5.0:
        score = 8.0
    else:
        score = 10.0
    return round(ratio, 2), score



def classify_sentiment(text: str) -> int:
    t = text.lower()
    bull = sum(1 for w in BULLISH_TERMS if w in t)
    bear = sum(1 for w in BEARISH_TERMS if w in t)
    return bull - bear



def score_sentiment(posts: Sequence[SocialPost]) -> float:
    if not posts:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for p in posts:
        weight = 1.0 + math.log1p(max(p.likes + p.replies + p.reposts, 0)) * 0.25
        total += classify_sentiment(p.body) * weight
        weight_sum += weight
    raw = total / weight_sum if weight_sum else 0.0
    mapped = max(0.0, min(10.0, 5.0 + raw * 2.0))
    return round(mapped, 2)



def score_engagement(posts: Sequence[SocialPost]) -> float:
    if not posts:
        return 0.0
    total = sum(max(p.likes, 0) + max(p.replies, 0) * 2 + max(p.reposts, 0) * 2 for p in posts)
    score = min(10.0, math.log1p(total) * 1.6)
    return round(score, 2)



def score_source_diversity(posts: Sequence[SocialPost]) -> float:
    if not posts:
        return 0.0
    distinct = len({p.source for p in posts})
    mapping = {1: 2.0, 2: 5.0, 3: 8.0}
    return mapping.get(distinct, 10.0)



def score_influencer(posts: Sequence[SocialPost]) -> float:
    if not posts:
        return 0.0
    avg_followers = sum(max(p.followers, 0) for p in posts) / max(len(posts), 1)
    score = min(10.0, math.log1p(avg_followers) * 1.1)
    return round(score, 2)



def score_spam_penalty(posts: Sequence[SocialPost]) -> float:
    if not posts:
        return 0.0
    normalized = [normalize_body(p.body) for p in posts if p.body.strip()]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    duplicate_count = sum(c - 1 for c in counts.values() if c > 1)
    ratio = duplicate_count / max(len(normalized), 1)
    return round(min(10.0, ratio * 15.0), 2)



def detect_narrative_tags(posts: Sequence[SocialPost]) -> List[str]:
    joined = " \n".join(p.body.lower() for p in posts)
    tags = []
    for tag, patterns in NARRATIVE_KEYWORDS.items():
        if any(p in joined for p in patterns):
            tags.append(tag)
    return tags[:5]



def score_narrative(tags: Sequence[str], mention_count: int) -> float:
    if mention_count <= 0:
        return 0.0
    if not tags:
        return 2.0 if mention_count > 0 else 0.0
    return min(10.0, 3.0 + len(tags) * 1.5)



def score_social_pressure(
    mention_velocity_score: float,
    sentiment_score: float,
    engagement_score: float,
    source_diversity_score: float,
    influencer_score: float,
    narrative_score: float,
    spam_penalty: float,
) -> float:
    total = (
        mention_velocity_score * 0.30
        + sentiment_score * 0.20
        + engagement_score * 0.15
        + source_diversity_score * 0.10
        + influencer_score * 0.10
        + narrative_score * 0.10
        - spam_penalty * 0.15
    )
    return round(max(0.0, min(10.0, total)), 2)



def confidence_label(score: float, mention_count: int) -> str:
    if mention_count <= 0:
        return "No Signal"
    if score >= 7.5:
        return "High Conviction Social Surge"
    if score >= 5.0:
        return "Narrative Forming"
    if score >= 2.5:
        return "Speculative Buzz"
    return "Low Signal / High Noise"



def build_reason(
    mention_velocity_ratio: float,
    sentiment_score: float,
    engagement_score: float,
    source_diversity_score: float,
    narrative_tags: Sequence[str],
    spam_penalty: float,
    mention_count: int = 0,
) -> str:
    if mention_count <= 0:
        return "no mentions detected"

    reasons: List[str] = []

    if mention_velocity_ratio >= 3.0:
        reasons.append("mention velocity spiking")
    elif mention_velocity_ratio >= 1.5:
        reasons.append("mentions rising")

    if sentiment_score >= 6.5:
        reasons.append("bullish tone")
    elif sentiment_score <= 3.5:
        reasons.append("bearish/mixed tone")

    if engagement_score >= 6.0:
        reasons.append("real engagement")

    if source_diversity_score >= 5.0:
        reasons.append("multi-source chatter")

    if narrative_tags:
        reasons.append("narrative tags: " + ", ".join(narrative_tags[:3]))

    if spam_penalty >= 5.0:
        reasons.append("pump/spam risk elevated")

    return ", ".join(reasons) if reasons else "low social signal"



def normalize_body(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9$ ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t



def read_tickers_from_csv(path: str, ticker_column: str) -> List[str]:
    df = pd.read_csv(path)
    if ticker_column not in df.columns:
        raise ValueError(f"Ticker column '{ticker_column}' not found in {path}")
    tickers = sorted({str(x).strip().upper() for x in df[ticker_column].dropna().tolist() if str(x).strip()})
    return tickers



def top_sources(posts: Sequence[SocialPost]) -> str:
    if not posts:
        return ""
    counts = Counter(p.source for p in posts)
    return ", ".join([name for name, _ in counts.most_common(3)])



def scan_ticker(ticker: str, adapters: Sequence[BaseAdapter], debug: bool = False) -> SocialScanResult:
    all_posts: List[SocialPost] = []
    for adapter in adapters:
        posts = adapter.fetch_posts(ticker, debug=debug)
        all_posts.extend(posts)

    now = datetime.now(timezone.utc)
    mention_count = len(all_posts)
    current_window_mentions = mention_count
    baseline_window_mentions = max(1, round(mention_count * 0.6)) if mention_count > 0 else 0
    mention_velocity_ratio, mention_velocity_score = score_mention_velocity(current_window_mentions, baseline_window_mentions)
    sentiment = score_sentiment(all_posts)
    engagement = score_engagement(all_posts)
    source_diversity = score_source_diversity(all_posts)
    influencer = score_influencer(all_posts)
    spam = score_spam_penalty(all_posts)
    tags = detect_narrative_tags(all_posts)
    narrative = score_narrative(tags, mention_count)
    pressure = score_social_pressure(
        mention_velocity_score=mention_velocity_score,
        sentiment_score=sentiment,
        engagement_score=engagement,
        source_diversity_score=source_diversity,
        influencer_score=influencer,
        narrative_score=narrative,
        spam_penalty=spam,
    )
    label = confidence_label(pressure, mention_count)
    reason = build_reason(
        mention_velocity_ratio=mention_velocity_ratio,
        sentiment_score=sentiment,
        engagement_score=engagement,
        source_diversity_score=source_diversity,
        narrative_tags=tags,
        spam_penalty=spam,
        mention_count=mention_count,
    )

    return SocialScanResult(
        scan_time_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ticker=ticker,
        mention_count=mention_count,
        current_window_mentions=current_window_mentions,
        baseline_window_mentions=baseline_window_mentions,
        mention_velocity=mention_velocity_ratio,
        sentiment_score=sentiment,
        engagement_score=engagement,
        source_diversity_score=source_diversity,
        influencer_score=influencer,
        spam_penalty=spam,
        narrative_score=narrative,
        social_pressure_score=pressure,
        confidence_label=label,
        narrative_tags=", ".join(tags),
        top_sources=top_sources(all_posts),
        reason=reason,
    )



def append_history(df: pd.DataFrame, history_path: str) -> None:
    path = Path(history_path)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan social telemetry around tickers.")
    parser.add_argument("--tickers", nargs="*", help="Space-separated tickers to scan.")
    parser.add_argument("--input-csv", help="CSV containing tickers.")
    parser.add_argument("--ticker-column", default="ticker", help="Ticker column name in input CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--history", default=DEFAULT_HISTORY, help="Append-only history CSV path.")
    parser.add_argument("--no-history", action="store_true", help="Do not append results to history CSV.")
    parser.add_argument("--debug", action="store_true", help="Print request and parsing debug info.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    tickers: List[str] = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers if t.strip()])
    if args.input_csv:
        tickers.extend(read_tickers_from_csv(args.input_csv, args.ticker_column))
    tickers = sorted(set(tickers))

    if not tickers:
        print("No tickers supplied. Use --tickers or --input-csv.")
        sys.exit(1)

    adapters: List[BaseAdapter] = [StocktwitsAdapter(), RedditAdapter(), XAdapter()]

    print(f"Scanning {len(tickers)} ticker(s) for social telemetry...")
    results: List[SocialScanResult] = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] {ticker}")
        results.append(scan_ticker(ticker, adapters=adapters, debug=args.debug))

    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values(by=["social_pressure_score", "mention_count", "ticker"], ascending=[False, False, True])

    print("\nTop social telemetry results:\n")
    try:
        print(df.to_string(index=False))
    except Exception:
        print(df)

    output_path = Path(args.output)
    try:
        df.to_csv(output_path, index=False)
    except PermissionError:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fallback = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
        print(f"\n[WARN] Could not write {output_path} (likely open in Excel). Writing to {fallback} instead.")
        df.to_csv(fallback, index=False)

    if not args.no_history:
        try:
            append_history(df, args.history)
        except PermissionError:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fallback = Path(args.history).with_name(f"{Path(args.history).stem}_{ts}{Path(args.history).suffix}")
            print(f"[WARN] Could not append history to {args.history}. Writing history snapshot to {fallback} instead.")
            df.to_csv(fallback, index=False)


if __name__ == "__main__":
    main()
