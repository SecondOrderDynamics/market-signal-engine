from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd


NUMERIC_COLUMNS = (
    "sec_insider_score",
    "congressional_score",
    "procurement_score",
    "ownership_score",
    "event_intel_score",
    "signal_count",
    "days_since_latest_event",
)

SPIKE_METRICS = {
    "event_intel_score": ("event score spike", 0.35, "high"),
    "sec_insider_score": ("insider activity spike", 0.20, "high"),
    "procurement_score": ("procurement activity spike", 0.12, "high"),
    "congressional_score": ("congressional activity spike", 0.08, "high"),
    "ownership_score": ("ownership activity spike", 0.05, "high"),
    "signal_count": ("signal count spike", 0.10, "high"),
    "days_since_latest_event": ("events are fresher than usual", 0.10, "low"),
}


def _parse_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _split_multi_value_field(value: object) -> set[str]:
    if value is None:
        return set()
    return {
        part.strip().lower()
        for part in str(value).split(",")
        if part and part.strip()
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EventBaselineAssessment:
    baseline_sample_size: int = 0
    baseline_window_days: int = 0
    baseline_event_intel_median: Optional[float] = None
    baseline_signal_count_median: Optional[float] = None
    baseline_days_since_latest_event_median: Optional[float] = None
    event_intel_score_z: float = 0.0
    signal_count_z: float = 0.0
    freshness_z: float = 0.0
    anomaly_score: float = 0.0
    anomaly_label: str = "No Baseline"
    anomaly_flags: str = ""
    anomaly_reason: str = ""
    novel_tag_count: int = 0
    novel_source_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class EventBaselineModel:
    def __init__(self, history_df: Optional[pd.DataFrame] = None) -> None:
        self.history_df = self._prepare_history(history_df)

    @classmethod
    def from_csv(cls, history_path: str) -> "EventBaselineModel":
        path = Path(history_path)
        if not path.exists():
            return cls(pd.DataFrame())
        try:
            history_df = pd.read_csv(path)
        except Exception:
            history_df = pd.DataFrame()
        return cls(history_df)

    def assess_row(self, row: Mapping[str, object]) -> EventBaselineAssessment:
        ticker = str(row.get("ticker", "") or "").strip().upper()
        if not ticker or self.history_df.empty:
            return EventBaselineAssessment(anomaly_reason="baseline unavailable: no historical rows")

        history = self.history_df[self.history_df["ticker"] == ticker].copy()
        if history.empty:
            return EventBaselineAssessment(anomaly_reason="baseline unavailable: no prior scans for ticker")

        sample_size = len(history.index)
        baseline_event_median = self._median(history, "event_intel_score")
        baseline_signal_median = self._median(history, "signal_count")
        baseline_freshness_median = self._median(history, "days_since_latest_event")
        window_days = self._baseline_window_days(history)

        metric_z: dict[str, float] = {}
        for metric, (_, _, direction) in SPIKE_METRICS.items():
            z_score, _ = self._robust_z_score(
                current_value=_safe_float(row.get(metric)),
                history_series=history[metric] if metric in history.columns else pd.Series(dtype="float64"),
                direction=direction,
            )
            metric_z[metric] = z_score

        current_tags = _split_multi_value_field(row.get("event_tags"))
        current_sources = _split_multi_value_field(row.get("top_sources"))
        seen_tags = self._collect_seen_values(history, "event_tags")
        seen_sources = self._collect_seen_values(history, "top_sources")
        novel_tags = sorted(current_tags - seen_tags)
        novel_sources = sorted(current_sources - seen_sources)

        # Exponential saturation: ~0.18 at n=1, ~0.45 at n=3, ~0.70 at n=6, ~0.86 at n=10.
        # Much more conservative than linear (min(1, n/6)) which hits 1.0 at just 6 samples.
        sample_reliability = 1.0 - math.exp(-sample_size / 5.0)
        raw_score = 0.0
        flags: list[str] = []
        ranked_reasons: list[tuple[float, str]] = []

        for metric, (label, weight, _) in SPIKE_METRICS.items():
            positive_z = max(0.0, metric_z.get(metric, 0.0))
            raw_score += weight * min(positive_z, 4.0)
            if positive_z >= 1.0:
                flags.append(metric)
                ranked_reasons.append((positive_z, label))

        # Scale novelty bonus by sample_reliability so a single prior scan can't
        # inflate the score to "Anomalous" just because everything looks new.
        novelty_bonus = min(1.5, len(novel_tags) * 0.35 + len(novel_sources) * 0.25) * sample_reliability
        anomaly_score = _clamp((raw_score * sample_reliability * 3.0) + novelty_bonus, 0.0, 10.0)

        if novel_tags:
            flags.append("new_tags")
            ranked_reasons.append((1.5, "new event tags vs history"))
        if novel_sources:
            flags.append("new_sources")
            ranked_reasons.append((1.0, "new sources vs history"))

        label = self._label(anomaly_score=anomaly_score, sample_size=sample_size)
        reason = self._build_reason(
            sample_size=sample_size,
            anomaly_score=anomaly_score,
            ranked_reasons=ranked_reasons,
            novel_tags=novel_tags,
            novel_sources=novel_sources,
        )

        return EventBaselineAssessment(
            baseline_sample_size=sample_size,
            baseline_window_days=window_days,
            baseline_event_intel_median=baseline_event_median,
            baseline_signal_count_median=baseline_signal_median,
            baseline_days_since_latest_event_median=baseline_freshness_median,
            event_intel_score_z=metric_z.get("event_intel_score", 0.0),
            signal_count_z=metric_z.get("signal_count", 0.0),
            freshness_z=metric_z.get("days_since_latest_event", 0.0),
            anomaly_score=round(anomaly_score, 2),
            anomaly_label=label,
            anomaly_flags=", ".join(dict.fromkeys(flags)),
            anomaly_reason=reason,
            novel_tag_count=len(novel_tags),
            novel_source_count=len(novel_sources),
        )

    @staticmethod
    def _prepare_history(history_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if history_df is None or history_df.empty:
            return pd.DataFrame(columns=["ticker", "scan_time_utc", *NUMERIC_COLUMNS, "event_tags", "top_sources"])

        df = history_df.copy()
        if "ticker" not in df.columns:
            df["ticker"] = ""
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"] != ""].copy()

        if "confidence_label" in df.columns:
            df = df[df["confidence_label"].astype(str).str.strip().str.lower() != "scan error"].copy()

        if "scan_time_utc" in df.columns:
            df["scan_time_utc_parsed"] = df["scan_time_utc"].apply(_parse_datetime)
        else:
            df["scan_time_utc_parsed"] = None

        for column in NUMERIC_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
            df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in ("event_tags", "top_sources"):
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].fillna("").astype(str)

        return df

    @staticmethod
    def _baseline_window_days(history: pd.DataFrame) -> int:
        if "scan_time_utc_parsed" not in history.columns:
            return 0
        parsed = history["scan_time_utc_parsed"].dropna()
        if parsed.empty:
            return 0
        span = parsed.max() - parsed.min()
        return max(0, int(span.total_seconds() // 86400))

    @staticmethod
    def _median(history: pd.DataFrame, column: str) -> Optional[float]:
        if column not in history.columns:
            return None
        series = pd.to_numeric(history[column], errors="coerce").dropna()
        if series.empty:
            return None
        return round(float(series.median()), 2)

    @staticmethod
    def _collect_seen_values(history: pd.DataFrame, column: str) -> set[str]:
        if column not in history.columns:
            return set()
        seen: set[str] = set()
        for value in history[column].dropna():
            seen.update(_split_multi_value_field(value))
        return seen

    @staticmethod
    def _robust_z_score(
        current_value: Optional[float],
        history_series: pd.Series,
        direction: str,
    ) -> tuple[float, Optional[float]]:
        if current_value is None:
            return 0.0, None

        series = pd.to_numeric(history_series, errors="coerce").dropna()
        if series.empty:
            return 0.0, None

        median = float(series.median())
        mad = float((series - median).abs().median())
        std = float(series.std(ddof=0)) if len(series.index) > 1 else 0.0
        floor = max(abs(median) * 0.15, 0.75)
        scale = mad * 1.4826 if mad > 0 else 0.0
        scale = max(scale, std, floor)
        if scale <= 0:
            return 0.0, round(median, 2)

        delta = current_value - median
        if direction == "low":
            delta = median - current_value
        z_score = _clamp(delta / scale, -6.0, 6.0)
        return round(z_score, 2), round(median, 2)

    @staticmethod
    def _label(anomaly_score: float, sample_size: int) -> str:
        if sample_size <= 0:
            return "No Baseline"
        if sample_size < 3:
            return "Baseline Building"
        if anomaly_score >= 6.5:
            return "Extreme vs Baseline"
        if anomaly_score >= 4.5:
            return "Anomalous vs Baseline"
        if anomaly_score >= 2.0:
            return "Elevated vs Baseline"
        return "Within Baseline"

    @staticmethod
    def _build_reason(
        sample_size: int,
        anomaly_score: float,
        ranked_reasons: Iterable[tuple[float, str]],
        novel_tags: list[str],
        novel_sources: list[str],
    ) -> str:
        if sample_size <= 0:
            return "baseline unavailable"
        if sample_size < 3:
            return f"baseline still building from {sample_size} prior scan(s)"
        if anomaly_score < 2.0:
            return "current activity is within the recent ticker baseline"

        ordered = sorted(ranked_reasons, key=lambda item: item[0], reverse=True)
        phrases: list[str] = []
        for _, text in ordered[:3]:
            if text not in phrases:
                phrases.append(text)
        if novel_tags:
            phrases.append("new tags: " + ", ".join(novel_tags[:3]))
        if novel_sources:
            phrases.append("new sources: " + ", ".join(novel_sources[:2]))
        return ", ".join(phrases) if phrases else "elevated versus recent baseline"
