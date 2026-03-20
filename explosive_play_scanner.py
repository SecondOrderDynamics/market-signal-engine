#!/usr/bin/env python3
"""
Explosive Play Scanner v2
-------------------------
Scans for small-/mid-cap stocks that may be under accumulation and capable of
momentum expansion.

What it does:
- Pulls a candidate universe from Finviz (small caps, unusual volume)
- Downloads OHLCV data with yfinance
- Scores symbols for:
    * Relative volume
    * Multi-day volume expansion
    * Tight price consolidation
    * Breakout proximity
    * Trend improvement (20D > 50D / reclaim behavior)
    * News activity
    * Short-interest proxy
    * Premarket gap behavior (best-effort)
    * Retail/social mention proxy (best-effort)
    * Insider activity proxy (best-effort)
- Outputs a ranked CSV watchlist
- Appends daily results to a historical scan file

Notes:
- This is a research scanner, not a prediction engine.
- Free data sources are noisy, incomplete, and occasionally ridiculous.
- yfinance data fields vary by ticker.
- OpenInsider and social scraping can break over time.

Install:
    pip install pandas numpy yfinance requests beautifulsoup4 lxml finvizfinance

Usage:
    python explosive_play_scanner_v2.py

Optional environment variables:
    NEWSAPI_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from event_intel_merge import merge_event_intel

try:
    from finvizfinance.screener.overview import Overview
except Exception:
    Overview = None


# ----------------------------
# Configuration
# ----------------------------
MIN_PRICE = 1.0
MAX_PRICE = 20.0
MIN_AVG_VOL = 300_000
MIN_REL_VOL = 1.5
MIN_DOLLAR_VOL = 1_000_000
MAX_RESULTS = 100
SLEEP_BETWEEN_REQUESTS = 0.35
REQUEST_TIMEOUT = 15

HISTORY_FILE = "explosive_play_historical_scans.csv"
DAILY_OUTPUT_FILE = "explosive_play_candidates.csv"
DEFAULT_EVENT_INTEL = "event_intel_scan.csv"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ExplosivePlayScanner/2.0"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
FEAR_GREED_ENDPOINTS = [
    "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
    "https://production.dataviz.cnn.io/index/fearandgreed/now",
]
_FEAR_GREED_CACHE: Optional[float] = None
_FEAR_GREED_CACHE_TS: float = 0.0
_FEAR_GREED_CACHE_TTL: float = 3600.0  # refresh after 1 hour


@dataclass
class ScanResult:
    scan_date: str
    ticker: str
    price: float
    market_cap: Optional[float]
    float_shares: Optional[float]
    rel_volume: float
    dollar_volume: float
    vol_expansion_score: float
    compression_score: float
    breakout_score: float
    trend_score: float
    news_score: float
    short_interest_score: float
    premarket_gap_score: float
    social_score: float
    insider_score: float
    total_score: float
    reason: str


# ----------------------------
# Utility helpers
# ----------------------------
def safe_request(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[requests.Response]:
    try:
        request_headers = {"User-Agent": USER_AGENT}
        if headers:
            request_headers.update(headers)
        resp = requests.get(url, params=params, headers=request_headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


# ----------------------------
# Universe builder
# ----------------------------
def get_finviz_universe() -> pd.DataFrame:
    """Return a starting universe from Finviz."""
    if Overview is None:
        raise RuntimeError(
            "finvizfinance is not installed or failed to import. Run: pip install finvizfinance"
        )

    overview = Overview()
    filters_dict = {
        "Price": "Over $1",
        "Average Volume": "Over 300K",
        "Relative Volume": "Over 1.5",
        "Market Cap.": "+Small (over $300mln)",
    }

    overview.set_filter(filters_dict=filters_dict)
    df = overview.screener_view()
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker"])

    if "Ticker" not in df.columns:
        raise RuntimeError("Unexpected Finviz response; no 'Ticker' column found.")

    return df[["Ticker"]].drop_duplicates().head(MAX_RESULTS).reset_index(drop=True)


# ----------------------------
# Data helpers
# ----------------------------
def safe_get_info(ticker: str) -> Dict:
    try:
        tk = yf.Ticker(ticker)
        return tk.info or {}
    except Exception:
        return {}



def download_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna().copy()
    except Exception:
        return pd.DataFrame()


def batch_download_histories(tickers: List[str], period: str = "6mo") -> Dict[str, pd.DataFrame]:
    """Download OHLCV for all tickers in a single yfinance call.

    Falls back to an empty dict on failure; individual tickers will then
    be fetched on-demand by scan_ticker via download_history().
    """
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=True,
            group_by="ticker",
        )
        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                # Single-ticker downloads don't gain a ticker-level MultiIndex layer.
                df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                result[ticker] = df.dropna()
            except Exception:
                result[ticker] = pd.DataFrame()
        return result
    except Exception:
        return {}


# ----------------------------
# News / catalyst scoring
# ----------------------------
def get_recent_news_items(ticker: str) -> List[dict]:
    """Use NewsAPI if available; otherwise fall back to yfinance news if possible."""
    if NEWSAPI_KEY:
        resp = safe_request(
            "https://newsapi.org/v2/everything",
            params={
                "q": f'"{ticker}"',
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 20,
                "apiKey": NEWSAPI_KEY,
            },
        )
        if resp is not None:
            try:
                payload = resp.json()
                return payload.get("articles", []) or []
            except Exception:
                pass

    try:
        tk = yf.Ticker(ticker)
        news = getattr(tk, "news", None)
        if isinstance(news, list):
            return news
    except Exception:
        pass

    return []



_HIGH_CATALYST_KW = frozenset(["fda", "approval", "phase 3", "acquisition", "merger", "buyout"])
_MED_CATALYST_KW = frozenset(["phase 1", "phase 2", "trial", "contract", "partnership", "buyback", "grant", "guidance", "spinoff", "restructuring"])
_LOW_CATALYST_KW = frozenset(["earnings", "ai", "upgrade", "outlook"])


def news_score_from_items(news_items: List[dict]) -> float:
    count = len(news_items)
    score = 0.0
    if count >= 10:
        score += 2.0
    elif count >= 5:
        score += 1.0
    elif count >= 2:
        score += 0.5

    # Weighted catalyst scoring: high-impact events count 3x more than low-signal terms.
    # Each article contributes at most its highest matching tier.
    catalyst_score = 0.0
    for item in news_items[:20]:
        blob = " ".join(
            str(item.get(k, "")) for k in ["title", "description", "summary"]
        ).lower()
        if any(kw in blob for kw in _HIGH_CATALYST_KW):
            catalyst_score += 1.0
        elif any(kw in blob for kw in _MED_CATALYST_KW):
            catalyst_score += 0.5
        elif any(kw in blob for kw in _LOW_CATALYST_KW):
            catalyst_score += 0.2

    # Map to the same 0-1 bonus range as before (previously 0.5 or 1.0 based on raw hit count).
    score += min(1.0, catalyst_score / 4.0)

    return min(score, 3.0)


# ----------------------------
# Indicator logic
# ----------------------------
def relative_volume(df: pd.DataFrame, window: int = 20) -> float:
    if len(df) < window + 1:
        return 0.0
    today_vol = float(df["Volume"].iloc[-1])
    avg_vol = float(df["Volume"].iloc[-window - 1 : -1].mean())
    if avg_vol <= 0:
        return 0.0
    return today_vol / avg_vol



def dollar_volume(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["Close"].iloc[-1] * df["Volume"].iloc[-1])



def volume_expansion_score(df: pd.DataFrame) -> float:
    if len(df) < 10:
        return 0.0
    vols = df["Volume"].tail(5).astype(float).values
    if np.any(vols <= 0):
        return 0.0

    x = np.arange(len(vols))
    slope = np.polyfit(x, vols, 1)[0]
    base = np.mean(vols)
    normalized = slope / base if base > 0 else 0.0

    monotonic_bonus = 0.5 if (vols[-1] > vols[-2] > vols[-3]) else 0.0
    score = max(0.0, normalized * 20 + monotonic_bonus)
    return min(score, 3.0)



def compression_score(df: pd.DataFrame) -> float:
    if len(df) < 20:
        return 0.0
    recent = df.tail(10).copy()
    high = recent["High"].max()
    low = recent["Low"].min()
    close = recent["Close"].iloc[-1]
    if close <= 0:
        return 0.0
    range_pct = (high - low) / close

    if range_pct <= 0.05:
        return 3.0
    if range_pct <= 0.08:
        return 2.0
    if range_pct <= 0.12:
        return 1.0
    return 0.0



def breakout_score(df: pd.DataFrame) -> float:
    if len(df) < 25:
        return 0.0
    close = float(df["Close"].iloc[-1])
    prior_20h = float(df["High"].iloc[-21:-1].max())
    if prior_20h <= 0:
        return 0.0
    distance = (prior_20h - close) / prior_20h

    if -0.03 <= distance <= 0.01:
        return 3.0
    if 0.01 < distance <= 0.03:
        return 2.0
    if 0.03 < distance <= 0.06:
        return 1.0
    return 0.0



def trend_score(df: pd.DataFrame) -> float:
    if len(df) < 60:
        return 0.0
    close = df["Close"].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    last = close.iloc[-1]

    score = 0.0
    if last > ma20:
        score += 1.0
    if last > ma50:
        score += 1.0
    if ma20 > ma50:
        score += 1.0
    return score



def short_interest_score(info: Dict) -> float:
    short_ratio = info.get("shortRatio")
    short_percent_float = info.get("shortPercentOfFloat")

    score = 0.0
    if isinstance(short_ratio, (int, float)):
        if short_ratio > 10:
            score += 1.5
        elif short_ratio > 5:
            score += 1.0

    if isinstance(short_percent_float, (int, float)):
        if short_percent_float > 0.25:
            score += 1.5
        elif short_percent_float > 0.15:
            score += 1.0

    return min(score, 3.0)



def premarket_gap_score(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    today_open = float(df["Open"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    if prev_close <= 0:
        return 0.0

    gap = (today_open - prev_close) / prev_close
    if gap > 0.10:
        return 2.0
    if gap > 0.05:
        return 1.0
    return 0.0



def _extract_fear_greed_value(payload: object) -> Optional[float]:
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if key_l in {"score", "value", "fear_and_greed"} and isinstance(value, (int, float)):
                    v = float(value)
                    if 0.0 <= v <= 100.0:
                        return v
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


def get_fear_greed_index() -> Optional[float]:
    global _FEAR_GREED_CACHE, _FEAR_GREED_CACHE_TS
    if _FEAR_GREED_CACHE is not None and time.monotonic() - _FEAR_GREED_CACHE_TS < _FEAR_GREED_CACHE_TTL:
        return _FEAR_GREED_CACHE

    for endpoint in FEAR_GREED_ENDPOINTS:
        resp = safe_request(endpoint)
        if resp is None:
            continue
        try:
            payload = resp.json()
        except Exception:
            continue

        value = _extract_fear_greed_value(payload)
        if value is not None:
            _FEAR_GREED_CACHE = value
            _FEAR_GREED_CACHE_TS = time.monotonic()
            return value

    return None


def fear_greed_risk_on_score(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if value >= 75:
        return 1.0
    if value >= 60:
        return 0.8
    if value >= 50:
        return 0.6
    if value >= 40:
        return 0.4
    if value >= 25:
        return 0.2
    return 0.0


def reddit_buzz_score(ticker: str) -> float:
    ticker_up = ticker.upper()
    resp = safe_request(
        REDDIT_SEARCH_URL,
        params={
            "q": f"(${ticker_up} OR {ticker_up}) (stock OR stocks OR earnings OR market)",
            "sort": "new",
            "limit": 40,
            "type": "link",
            "raw_json": 1,
        },
        headers={
            "Accept": "application/json",
            "Referer": "https://www.reddit.com/",
        },
    )
    if resp is None:
        return 0.0

    try:
        payload = resp.json()
    except Exception:
        return 0.0

    children = payload.get("data", {}).get("children", []) or []
    if not children:
        return 0.0

    ticker_pattern = re.compile(rf"(?<![A-Z0-9])\$?{re.escape(ticker_up)}(?![A-Z0-9])", re.IGNORECASE)
    now_ts = time.time()
    mentions = 0
    total_engagement = 0.0
    for child in children:
        data = child.get("data") or {}
        title = str(data.get("title", "") or "")
        selftext = str(data.get("selftext", "") or "")
        body = f"{title} {selftext}".strip()
        if not body or not ticker_pattern.search(body):
            continue
        mentions += 1
        ups = float(data.get("ups", 0) or 0)
        comments = float(data.get("num_comments", 0) or 0)
        # Decay engagement over 2 weeks: a post from 14 days ago counts ~10% as much as a fresh one.
        created = float(data.get("created_utc", now_ts) or now_ts)
        age_hours = max(0.0, (now_ts - created) / 3600.0)
        recency_weight = max(0.1, 1.0 - age_hours / 336.0)
        total_engagement += (ups + 1.5 * comments) * recency_weight

    if mentions == 0:
        return 0.0

    mention_component = min(1.0, mentions / 12.0)
    engagement_component = min(1.0, float(np.log1p(total_engagement)) / 6.0)
    return round(mention_component * 0.65 + engagement_component * 0.35, 3)


def news_buzz_score(news_items: List[dict], ticker: str) -> float:
    if not news_items:
        return 0.0

    ticker_up = ticker.upper()
    ticker_pattern = re.compile(rf"(?<![A-Z0-9])\$?{re.escape(ticker_up)}(?![A-Z0-9])", re.IGNORECASE)
    catalyst_terms = [
        "earnings",
        "guidance",
        "upgrade",
        "approval",
        "contract",
        "partnership",
        "acquisition",
        "merger",
        "fda",
        "phase 1",
        "phase 2",
        "phase 3",
    ]

    mentions = 0
    catalyst_hits = 0
    for item in news_items[:20]:
        blob = " ".join(str(item.get(k, "")) for k in ["title", "description", "summary"])
        if not blob.strip() or not ticker_pattern.search(blob):
            continue
        mentions += 1
        blob_l = blob.lower()
        if any(term in blob_l for term in catalyst_terms):
            catalyst_hits += 1

    if mentions == 0:
        return 0.0

    coverage_component = min(1.0, mentions / 10.0)
    catalyst_component = min(1.0, catalyst_hits / max(mentions, 1))
    return round(coverage_component * 0.6 + catalyst_component * 0.4, 3)


def social_score(ticker: str, news_items: Optional[List[dict]] = None) -> float:
    """
    Best-effort sentiment proxy using Reddit + news + market regime.
    Kept on the historical 0.0/0.5/1.0/1.5 scale for score stability.
    """
    news_items = news_items if news_items is not None else get_recent_news_items(ticker)
    reddit = reddit_buzz_score(ticker)
    news = news_buzz_score(news_items, ticker)
    regime = fear_greed_risk_on_score(get_fear_greed_index())

    blended = reddit * 0.55 + news * 0.30 + regime * 0.15
    if blended >= 0.75:
        return 1.5
    if blended >= 0.50:
        return 1.0
    if blended >= 0.25:
        return 0.5
    return 0.0



def insider_activity_score(ticker: str) -> float:
    """Best-effort OpenInsider scrape for recent purchase activity.

    Counts explicit buy (P) and sell (S) transaction codes in table cells rather
    than checking for the presence of the word 'sale' in page text — which almost
    always appears in site navigation/headers regardless of actual transactions.
    """
    url = f"https://www.openinsider.com/screener?s={ticker}"
    resp = safe_request(url)
    if resp is None:
        return 0.0

    text = resp.text
    # OpenInsider renders each transaction code in its own <td> cell.
    buy_count = len(re.findall(r"<td[^>]*>\s*P\s*</td>", text, re.IGNORECASE))
    sell_count = len(re.findall(r"<td[^>]*>\s*S\s*</td>", text, re.IGNORECASE))

    if buy_count == 0 and sell_count == 0:
        # Fallback: look for the verbose label used in some page variants.
        tl = text.lower()
        buy_count = tl.count("p - purchase")
        sell_count = tl.count("s - sale")

    if buy_count == 0:
        return 0.0
    if buy_count > sell_count:
        # More buys than sells: score rises with the net imbalance, capped at 2.0.
        return min(2.0, 1.0 + (buy_count - sell_count) * 0.25)
    # Buys present but matched or outnumbered by sells.
    return 0.5



def human_reason(metrics: Dict[str, float]) -> str:
    reasons: List[str] = []
    if metrics["rel_volume"] >= 2:
        reasons.append("unusual volume")
    if metrics["vol_expansion_score"] >= 1.5:
        reasons.append("multi-day volume expansion")
    if metrics["compression_score"] >= 2:
        reasons.append("tight consolidation")
    if metrics["breakout_score"] >= 2:
        reasons.append("near breakout")
    if metrics["trend_score"] >= 2:
        reasons.append("trend improving")
    if metrics["news_score"] >= 1:
        reasons.append("elevated news flow")
    if metrics["short_interest_score"] >= 1:
        reasons.append("short squeeze potential")
    if metrics["premarket_gap_score"] >= 1:
        reasons.append("gap strength")
    if metrics["social_score"] >= 1:
        reasons.append("retail attention")
    if metrics["insider_score"] >= 1:
        reasons.append("insider activity")
    return ", ".join(reasons) if reasons else "mixed setup"


def parse_composite_weights(text: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not text:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Composite weights must be three comma-separated numbers: total,event,anomaly")
    weights = tuple(float(p) for p in parts)
    return weights  # type: ignore[return-value]


# ----------------------------
# Core scan
# ----------------------------
def scan_ticker(ticker: str, hist: Optional[pd.DataFrame] = None) -> Optional[ScanResult]:
    if hist is None or hist.empty:
        hist = download_history(ticker, period="6mo", interval="1d")
    if hist.empty or len(hist) < 60:
        return None

    price = float(hist["Close"].iloc[-1])
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    rv = relative_volume(hist)
    dv = dollar_volume(hist)
    if rv < MIN_REL_VOL or dv < MIN_DOLLAR_VOL:
        return None

    info = safe_get_info(ticker)
    market_cap = info.get("marketCap")
    float_shares = info.get("floatShares")
    avg_vol = info.get("averageVolume") or info.get("averageVolume10days") or 0
    if avg_vol and avg_vol < MIN_AVG_VOL:
        return None

    news_items = get_recent_news_items(ticker)

    vscore = volume_expansion_score(hist)
    cscore = compression_score(hist)
    bscore = breakout_score(hist)
    tscore = trend_score(hist)
    nscore = news_score_from_items(news_items)
    sscore = short_interest_score(info)
    pgscore = premarket_gap_score(hist)
    socscore = social_score(ticker, news_items=news_items)
    iscore = insider_activity_score(ticker)

    float_bonus = 0.0
    if isinstance(float_shares, (int, float)) and float_shares > 0:
        if float_shares < 20_000_000:
            float_bonus = 2.0
        elif float_shares < 50_000_000:
            float_bonus = 1.0

    total = (
        min(rv, 5.0) * 1.2
        + vscore * 1.2
        + cscore * 1.0
        + bscore * 1.2
        + tscore * 0.8
        + nscore * 0.8
        + sscore * 1.0
        + pgscore * 0.6
        + socscore * 0.5
        + iscore * 0.8
        + float_bonus
    )

    metrics = {
        "rel_volume": rv,
        "vol_expansion_score": vscore,
        "compression_score": cscore,
        "breakout_score": bscore,
        "trend_score": tscore,
        "news_score": nscore,
        "short_interest_score": sscore,
        "premarket_gap_score": pgscore,
        "social_score": socscore,
        "insider_score": iscore,
    }

    return ScanResult(
        scan_date=datetime.now().strftime("%Y-%m-%d"),
        ticker=ticker,
        price=round(price, 2),
        market_cap=market_cap,
        float_shares=float_shares,
        rel_volume=round(rv, 2),
        dollar_volume=round(dv, 2),
        vol_expansion_score=round(vscore, 2),
        compression_score=round(cscore, 2),
        breakout_score=round(bscore, 2),
        trend_score=round(tscore, 2),
        news_score=round(nscore, 2),
        short_interest_score=round(sscore, 2),
        premarket_gap_score=round(pgscore, 2),
        social_score=round(socscore, 2),
        insider_score=round(iscore, 2),
        total_score=round(total, 2),
        reason=human_reason(metrics),
    )


# ----------------------------
# Historical persistence
# ----------------------------
def append_to_history(df: pd.DataFrame, history_file: str = HISTORY_FILE) -> None:
    # Ensure every persisted row has a scan_date for evaluator alignment.
    if "scan_date" not in df.columns:
        df = df.copy()
        df["scan_date"] = datetime.now().strftime("%Y-%m-%d")

    path = os.path.abspath(history_file)
    if os.path.exists(path):
        try:
            history = pd.read_csv(path)
        except Exception:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()

    merged_cols = list(history.columns)
    for col in df.columns:
        if col not in merged_cols:
            merged_cols.append(col)

    combined = pd.concat(
        [
            history.reindex(columns=merged_cols),
            df.reindex(columns=merged_cols),
        ],
        ignore_index=True,
    )
    combined.to_csv(path, index=False)



def add_repeat_appearance_count(df: pd.DataFrame, history_file: str = HISTORY_FILE) -> pd.DataFrame:
    if not os.path.exists(history_file):
        df["repeat_count_30d"] = 1
        return df

    try:
        history = pd.read_csv(history_file)
        if "scan_date" not in history.columns or "ticker" not in history.columns:
            df["repeat_count_30d"] = 1
            return df

        history["scan_date"] = pd.to_datetime(history["scan_date"], errors="coerce")
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
        recent = history[history["scan_date"] >= cutoff]
        counts = recent.groupby("ticker").size().rename("repeat_count_30d")
        df = df.merge(counts, how="left", left_on="ticker", right_index=True)
        df["repeat_count_30d"] = df["repeat_count_30d"].fillna(0).astype(int) + 1
        return df
    except Exception:
        df["repeat_count_30d"] = 1
        return df


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explosive play scanner with optional event-intel enrichment.")
    parser.add_argument("--event-intel-csv", default=DEFAULT_EVENT_INTEL, help="Path to event_intel_scan.csv for enrichment")
    parser.add_argument("--no-event-intel", action="store_true", help="Skip event-intel merge even if CSV exists")
    parser.add_argument(
        "--composite-weights",
        help="Optional composite weights as 'total,event,anomaly'. Example: 0.65,0.25,0.10",
    )
    parser.add_argument("--output", default=DAILY_OUTPUT_FILE, help="Output CSV path (ranked merged results)")
    parser.add_argument("--history", default=HISTORY_FILE, help="Historical append CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building universe from Finviz...")
    universe = get_finviz_universe()
    tickers = universe["Ticker"].dropna().astype(str).tolist()
    print(f"Universe size: {len(tickers)}")

    print("Downloading market data in batch...")
    hist_cache = batch_download_histories(tickers)
    if hist_cache:
        print(f"Batch download complete ({len(hist_cache)} tickers).")
    else:
        print("Batch download failed or returned empty; will fetch per-ticker.")

    results: List[ScanResult] = []
    for i, ticker in enumerate(tickers, start=1):
        try:
            print(f"[{i}/{len(tickers)}] Scanning {ticker}...")
            res = scan_ticker(ticker, hist=hist_cache.get(ticker))
            if res:
                results.append(res)
        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not results:
        print("No candidates found.")
        return

    df = pd.DataFrame([asdict(r) for r in results])
    df = add_repeat_appearance_count(df, history_file=args.history)

    # slight reward for names showing up repeatedly over the last month
    df["total_score"] = df["total_score"] + np.minimum(df["repeat_count_30d"] - 1, 5) * 0.4
    df["total_score"] = df["total_score"].round(2)

    composite_weights = None
    if args.composite_weights:
        composite_weights = parse_composite_weights(args.composite_weights)

    # Optional enrichment with latest event-intel snapshot; falls back to empty columns if missing.
    merged_df, merge_summary = merge_event_intel(
        df,
        event_csv_path=args.event_intel_csv,
        composite_weights=None if args.no_event_intel else composite_weights,
    )

    sort_columns = ["total_score", "repeat_count_30d", "rel_volume"]
    if not args.no_event_intel and composite_weights is not None and "composite_score" in merged_df.columns:
        sort_columns = ["composite_score"] + sort_columns

    merged_df = merged_df.sort_values(
        sort_columns,
        ascending=[False] + [False] * (len(sort_columns) - 1),
    ).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\nTop candidates:\n")
    try:
        print(merged_df.head(25).to_string(index=False))
    except Exception:
        print(merged_df.head(25))

    merged_df.to_csv(args.output, index=False)
    append_to_history(merged_df, history_file=args.history)

    print(f"\nSaved ranked results to: {args.output}")
    print(f"Appended results to history file: {args.history}")
    print(f"Event-intel merge: {merge_summary.message} (rows={merge_summary.event_rows}, used_composite={merge_summary.used_composite})")


if __name__ == "__main__":
    main()
