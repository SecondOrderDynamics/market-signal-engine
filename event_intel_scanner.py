from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from event_baseline_model import EventBaselineModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "TRADE_SCANNER/1.0 RyanKersey research@example.com",
)
HTTP_TIMEOUT = int(os.getenv("EVENT_INTEL_TIMEOUT", "20"))
DEFAULT_OUTPUT = "event_intel_scan.csv"
DEFAULT_HISTORY = "event_intel_history.csv"
SAM_API_KEY = os.getenv("SAM_API_KEY", "")
HOUSE_PTR_CSV = os.getenv("HOUSE_PTR_CSV", "")

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/plain, */*",
}
GENERIC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
}

PROCUREMENT_KEYWORDS = {
    "defense": ["air force", "army", "navy", "space force", "dod", "missile", "cyber"],
    "health": ["nih", "hhs", "va medical", "cms", "cdc"],
    "energy": ["doe", "energy", "grid", "battery"],
    "space": ["nasa", "space", "launch", "satellite"],
    "infrastructure": ["construction", "bridge", "transportation", "faa", "dot"],
}

BUY_CODES = {"P"}
SELL_CODES = {"S"}
NEUTRAL_CODES = {"A", "D", "F", "G", "I", "M", "J", "K", "W", "X"}


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class EventIntelSignal:
    signal_type: str
    title: str
    source: str
    event_date: datetime
    score: float
    tags: List[str]
    details: str


@dataclass
class EventIntelResult:
    scan_time_utc: str
    ticker: str
    company_name: str
    sec_insider_score: float
    congressional_score: float
    procurement_score: float
    ownership_score: float
    event_intel_score: float
    signal_count: int
    days_since_latest_event: Optional[int]
    confidence_label: str
    event_tags: str
    top_sources: str
    reason: str
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


@dataclass
class CompanyLookup:
    ticker: str
    cik_str: str
    cik_int: int
    title: str


@dataclass
class InsiderForm4Event:
    filing_date: datetime
    accession_number: str
    primary_document: str
    transaction_codes: List[str]
    acquired_shares: float
    disposed_shares: float
    net_shares: float
    avg_price: Optional[float]
    reporting_owner_count: int


@dataclass
class ReportingOwnerInfo:
    name: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    officer_title: str


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)



def parse_date(value: Optional[str]) -> datetime:
    if not value:
        return now_utc()
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return now_utc()



def days_ago(dt: datetime) -> int:
    return max(0, int((now_utc() - dt).total_seconds() // 86400))



def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))



def ensure_parent_dir(path_str: str) -> None:
    path = Path(path_str)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)



def request_json(url: str, headers: Optional[dict] = None, params: Optional[dict] = None, debug: bool = False) -> Optional[dict]:
    hdrs = headers or GENERIC_HEADERS
    try:
        resp = requests.get(url, headers=hdrs, params=params, timeout=HTTP_TIMEOUT)
        if debug:
            print(f"[DEBUG] GET {resp.url} -> {resp.status_code} | {resp.headers.get('Content-Type')}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        if debug:
            print(f"[DEBUG] JSON request failed for {url}: {e}")
        return None



def request_text(url: str, headers: Optional[dict] = None, params: Optional[dict] = None, debug: bool = False) -> Optional[str]:
    hdrs = headers or GENERIC_HEADERS
    try:
        resp = requests.get(url, headers=hdrs, params=params, timeout=HTTP_TIMEOUT)
        if debug:
            print(f"[DEBUG] GET {resp.url} -> {resp.status_code} | {resp.headers.get('Content-Type')}")
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        if debug:
            print(f"[DEBUG] Text request failed for {url}: {e}")
        return None


# -----------------------------------------------------------------------------
# SEC lookups and Form 4 parsing
# -----------------------------------------------------------------------------


class SecTickerMap:
    URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self) -> None:
        self._cache: Dict[str, CompanyLookup] = {}
        self._loaded = False

    def load(self, debug: bool = False) -> None:
        if self._loaded:
            return
        payload = request_json(self.URL, headers=GENERIC_HEADERS, debug=debug)
        if not payload:
            self._loaded = True
            return
        for _, item in payload.items():
            ticker = str(item.get("ticker", "") or "").upper().strip()
            cik_int = int(item.get("cik_str", 0) or 0)
            title = str(item.get("title", "") or "").strip()
            if not ticker or cik_int <= 0:
                continue
            self._cache[ticker] = CompanyLookup(
                ticker=ticker,
                cik_str=f"{cik_int:010d}",
                cik_int=cik_int,
                title=title,
            )
        self._loaded = True

    def get(self, ticker: str, debug: bool = False) -> Optional[CompanyLookup]:
        self.load(debug=debug)
        return self._cache.get(ticker.upper().strip())


class SecInsiderAdapter:
    def __init__(self, ticker_map: SecTickerMap) -> None:
        self.ticker_map = ticker_map

    def fetch_signals(self, ticker: str, debug: bool = False) -> Tuple[List[EventIntelSignal], str]:
        lookup = self.ticker_map.get(ticker, debug=debug)
        if not lookup:
            return [], "ticker not found in SEC map"

        submissions_url = f"https://data.sec.gov/submissions/CIK{lookup.cik_str}.json"
        payload = request_json(submissions_url, headers=SEC_HEADERS, debug=debug)
        if not payload:
            return [], "SEC submissions unavailable"

        recent = ((payload.get("filings") or {}).get("recent") or {})
        forms = recent.get("form", []) or []
        filing_dates = recent.get("filingDate", []) or []
        accessions = recent.get("accessionNumber", []) or []
        primary_docs = recent.get("primaryDocument", []) or []

        events: List[EventIntelSignal] = []
        form4_count = 0

        for form, filing_date, accession, primary_doc in zip(forms, filing_dates, accessions, primary_docs):
            form = str(form or "")
            if form not in {"4", "4/A"}:
                continue
            form4_count += 1
            filed_dt = parse_date(str(filing_date or ""))
            filing_age = days_ago(filed_dt)
            if filing_age > 60:
                continue

            parsed_event = self._fetch_and_parse_form4(
                lookup=lookup,
                accession_number=str(accession or ""),
                primary_document=str(primary_doc or ""),
                filing_date=filed_dt,
                debug=debug,
            )
            if not parsed_event:
                recency_score = 4.0 if filing_age <= 2 else 2.5 if filing_age <= 7 else 1.5
                events.append(
                    EventIntelSignal(
                        signal_type="sec_form4_activity",
                        title=f"Recent Form 4 filing for {lookup.ticker}",
                        source="SEC EDGAR",
                        event_date=filed_dt,
                        score=recency_score,
                        tags=["insider", "form4", "unparsed"],
                        details=f"Form {form} filed {filed_dt.date()} but transaction details were not parsed.",
                    )
                )
                continue

            events.extend(self._score_form4_event(lookup, parsed_event))

        if not events and form4_count == 0:
            return [], "no recent Form 4 filings"
        if not events and form4_count > 0:
            return [], "recent Form 4 filings found, but no actionable transactions parsed"

        return events, "ok"

    def _fetch_and_parse_form4(
        self,
        lookup: CompanyLookup,
        accession_number: str,
        primary_document: str,
        filing_date: datetime,
        debug: bool = False,
    ) -> Optional[InsiderForm4Event]:
        if not accession_number:
            return None

        accession_nodash = accession_number.replace("-", "")
        candidates = self._discover_filing_xml_candidates(lookup, accession_nodash, primary_document, debug=debug)
        if debug:
            print(f"[DEBUG] {lookup.ticker} accession {accession_number} candidate XMLs: {candidates[:5]}")
        if not candidates:
            return None

        text = None
        resolved_doc = primary_document
        for candidate in candidates:
            url = f"https://www.sec.gov/Archives/edgar/data/{lookup.cik_int}/{accession_nodash}/{candidate}"
            text = request_text(url, headers=SEC_HEADERS, debug=debug)
            if text and "ownershipDocument" in text:
                resolved_doc = candidate
                break
            text = None

        if not text:
            return None

        transaction_codes: List[str] = []
        acquired = 0.0
        disposed = 0.0
        prices: List[float] = []
        owners: List[ReportingOwnerInfo] = []

        try:
            root = ET.fromstring(text.encode("utf-8", errors="ignore"))
            owners = self._extract_reporting_owners(root)
            for txn in root.findall(".//nonDerivativeTransaction") + root.findall(".//derivativeTransaction"):
                code = (txn.findtext(".//transactionCoding/transactionCode") or "").strip().upper()
                if code:
                    transaction_codes.append(code)

                shares_val = self._xml_float(txn.findtext(".//transactionAmounts/transactionShares/value"))
                price_val = self._xml_float(txn.findtext(".//transactionAmounts/transactionPricePerShare/value"))
                acquired_disposed = (txn.findtext(".//transactionAmounts/transactionAcquiredDisposedCode/value") or "").strip().upper()

                if price_val and price_val > 0:
                    prices.append(price_val)
                if acquired_disposed == "A":
                    acquired += shares_val
                elif acquired_disposed == "D":
                    disposed += shares_val
        except Exception:
            owner_count = len(re.findall(r"<reportingOwner>", text, flags=re.IGNORECASE))
            owners = [ReportingOwnerInfo(name="", is_director=False, is_officer=False, is_ten_percent_owner=False, officer_title="") for _ in range(max(owner_count,1))]
            transaction_codes.extend(re.findall(r"<transactionCode>\s*([A-Z])\s*</transactionCode>", text, flags=re.IGNORECASE))
            acquired += sum(float(x) for x in re.findall(r"<transactionAcquiredDisposedCode>\s*<value>A</value>.*?<transactionShares>\s*<value>([\d\.]+)</value>", text, flags=re.IGNORECASE | re.DOTALL))
            disposed += sum(float(x) for x in re.findall(r"<transactionAcquiredDisposedCode>\s*<value>D</value>.*?<transactionShares>\s*<value>([\d\.]+)</value>", text, flags=re.IGNORECASE | re.DOTALL))
            prices.extend(float(x) for x in re.findall(r"<transactionPricePerShare>\s*<value>([\d\.]+)</value>", text, flags=re.IGNORECASE))

        if not transaction_codes and acquired == 0 and disposed == 0:
            return None

        avg_price = round(sum(prices) / len(prices), 4) if prices else None
        return InsiderForm4Event(
            filing_date=filing_date,
            accession_number=accession_number,
            primary_document=resolved_doc,
            transaction_codes=[c.upper() for c in transaction_codes if c],
            acquired_shares=round(acquired, 2),
            disposed_shares=round(disposed, 2),
            net_shares=round(acquired - disposed, 2),
            avg_price=avg_price,
            reporting_owner_count=max(len(owners), 1),
        )

    def _discover_filing_xml_candidates(
        self,
        lookup: CompanyLookup,
        accession_nodash: str,
        primary_document: str,
        debug: bool = False,
    ) -> List[str]:
        ranked_candidates: List[Tuple[int, str]] = []

        if primary_document and primary_document.lower().endswith(".xml"):
            ranked_candidates.append((6, primary_document))

        index_url = f"https://www.sec.gov/Archives/edgar/data/{lookup.cik_int}/{accession_nodash}/index.json"
        payload = request_json(index_url, headers=SEC_HEADERS, debug=debug)
        items = (((payload or {}).get("directory") or {}).get("item") or [])
        for item in items:
            name = str((item or {}).get("name", "") or "")
            lower = name.lower()
            if not lower.endswith(".xml"):
                continue
            if lower == "primary_doc.xml":
                continue
            rank = 0
            if "ownership" in lower:
                rank += 10
            if "form4" in lower or lower.endswith("doc4.xml") or lower.endswith("edgardoc.xml"):
                rank += 8
            if primary_document and lower == primary_document.lower():
                rank += 6
            if "xsl" in lower:
                rank -= 5
            if lower.startswith("schema") or lower.endswith("_hdr.sgml"):
                rank -= 10
            ranked_candidates.append((rank, name))

        seen = set()
        ranked: List[str] = []
        for _, name in sorted(ranked_candidates, key=lambda x: (-x[0], x[1])):
            if name not in seen:
                ranked.append(name)
                seen.add(name)
        return ranked

    @staticmethod
    def _extract_reporting_owners(root: ET.Element) -> List[ReportingOwnerInfo]:
        owners: List[ReportingOwnerInfo] = []
        for owner in root.findall(".//reportingOwner"):
            name = (owner.findtext(".//rptOwnerName") or "").strip()
            rel = owner.find(".//reportingOwnerRelationship")
            owners.append(
                ReportingOwnerInfo(
                    name=name,
                    is_director=(owner.findtext(".//isDirector") or "0").strip() in {"1", "true", "True"},
                    is_officer=(owner.findtext(".//isOfficer") or "0").strip() in {"1", "true", "True"},
                    is_ten_percent_owner=(owner.findtext(".//isTenPercentOwner") or "0").strip() in {"1", "true", "True"},
                    officer_title=(owner.findtext(".//officerTitle") or "").strip(),
                )
            )
        return owners

    @staticmethod
    def _xml_float(value: Optional[str]) -> float:
        try:
            return float(str(value or "0").replace(",", ""))
        except Exception:
            return 0.0

    def _score_form4_event(self, lookup: CompanyLookup, event: InsiderForm4Event) -> List[EventIntelSignal]:
        tags = ["insider", "form4", "parsed"]
        age = days_ago(event.filing_date)
        recency = 4.5 if age <= 2 else 3.5 if age <= 7 else 2.0 if age <= 21 else 1.0
        cluster_bonus = 1.25 if event.reporting_owner_count >= 2 else 0.0
        size_bonus = min(3.0, math.log10(max(abs(event.net_shares), 1.0)))

        codes = set(event.transaction_codes)
        signals: List[EventIntelSignal] = []
        codes_text = ','.join(sorted(codes)) or 'n/a'

        if codes & BUY_CODES or event.net_shares > 0:
            score = clamp(recency + cluster_bonus + size_bonus, 0.0, 10.0)
            tags_buy = tags + ["buy"]
            if event.reporting_owner_count >= 2:
                tags_buy.append("cluster")
            if 'P' in codes:
                tags_buy.append('open_market_buy')
            price_text = f" at ~${event.avg_price:.2f}" if event.avg_price else ""
            signals.append(
                EventIntelSignal(
                    signal_type="insider_accumulation",
                    title=f"Insider accumulation detected for {lookup.ticker}",
                    source="SEC EDGAR",
                    event_date=event.filing_date,
                    score=score,
                    tags=tags_buy,
                    details=(
                        f"Form 4 filing shows net acquired shares {event.net_shares:,.0f}{price_text}; "
                        f"owners involved: {event.reporting_owner_count}; codes: {codes_text}."
                    ),
                )
            )

        if codes & SELL_CODES or event.net_shares < 0:
            score = clamp(recency + 0.75 + size_bonus, 0.0, 10.0)
            tags_sell = tags + ["sell"]
            if 'S' in codes:
                tags_sell.append('open_market_sale')
            if 'M' in codes:
                tags_sell.append('option_exercise')
            if 'F' in codes:
                tags_sell.append('tax_withholding')
            signals.append(
                EventIntelSignal(
                    signal_type="insider_distribution",
                    title=f"Insider selling detected for {lookup.ticker}",
                    source="SEC EDGAR",
                    event_date=event.filing_date,
                    score=score,
                    tags=tags_sell,
                    details=(
                        f"Form 4 filing shows net disposed shares {abs(event.net_shares):,.0f}; "
                        f"owners involved: {event.reporting_owner_count}; codes: {codes_text}."
                    ),
                )
            )

        if not signals:
            score = clamp(recency + 0.25, 0.0, 10.0)
            signals.append(
                EventIntelSignal(
                    signal_type="insider_activity",
                    title=f"Recent insider filing activity for {lookup.ticker}",
                    source="SEC EDGAR",
                    event_date=event.filing_date,
                    score=score,
                    tags=tags + ["administrative"],
                    details=f"Recent Form 4 activity detected; transaction codes: {codes_text}.",
                )
            )

        return signals


# -----------------------------------------------------------------------------
# Procurement adapters
# -----------------------------------------------------------------------------


class SamOpportunityAdapter:
    """
    Practical compromise: use SAM opportunities as a procurement-intelligence hook.
    Actual contract-award APIs on SAM.gov also exist, but they require API keys and
    more account setup. This adapter is therefore optional and degrades gracefully.
    """

    SEARCH_URL = "https://api.sam.gov/opportunities/v2/search"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key.strip()

    def fetch_signals(self, ticker: str, company_name: str, debug: bool = False) -> Tuple[List[EventIntelSignal], str]:
        if not self.api_key:
            return [], "SAM_API_KEY not set"
        if not company_name:
            return [], "company name unavailable"

        keyword = company_name.replace(",", " ").replace(" Inc", "").replace(" Corporation", "").strip()
        posted_from = (now_utc() - timedelta(days=14)).strftime("%m/%d/%Y")
        params = {
            "api_key": self.api_key,
            "q": keyword,
            "postedFrom": posted_from,
            "limit": 25,
            "offset": 0,
        }
        payload = request_json(self.SEARCH_URL, params=params, debug=debug)
        if not payload:
            return [], "SAM opportunities unavailable"

        opportunities = payload.get("opportunitiesData") or payload.get("data") or []
        if not isinstance(opportunities, list):
            opportunities = []

        signals: List[EventIntelSignal] = []
        for opp in opportunities[:10]:
            raw_text = " ".join(
                str(opp.get(k, "") or "")
                for k in ["title", "description", "fullParentPathName", "departmentName", "subTier", "organizationType"]
            )
            if keyword.lower() not in raw_text.lower():
                continue
            event_date = parse_date(opp.get("postedDate") or opp.get("responseDeadLine"))
            age = days_ago(event_date)
            recency = 4.0 if age <= 3 else 2.5 if age <= 10 else 1.0
            tags = ["procurement", "sam"]
            narrative_bonus = 0.0
            raw_lower = raw_text.lower()
            for tag, words in PROCUREMENT_KEYWORDS.items():
                if any(w in raw_lower for w in words):
                    tags.append(tag)
                    narrative_bonus += 1.0
            score = clamp(recency + min(2.0, narrative_bonus), 0.0, 10.0)
            title = str(opp.get("title", "SAM opportunity match") or "SAM opportunity match").strip()
            dept = str(opp.get("departmentName", "") or "").strip()
            details = f"SAM opportunity matched company name '{keyword}'."
            if dept:
                details += f" Department: {dept}."
            signals.append(
                EventIntelSignal(
                    signal_type="procurement_signal",
                    title=title,
                    source="SAM.gov",
                    event_date=event_date,
                    score=score,
                    tags=sorted(set(tags)),
                    details=details,
                )
            )

        return signals, "ok" if signals else "no recent SAM opportunity matches"


# -----------------------------------------------------------------------------
# Congressional transaction adapter (offline CSV import)
# -----------------------------------------------------------------------------


class CongressionalCsvAdapter:
    """
    Offline adapter for House/Senate transaction exports. This keeps the project
    honest: no fake API promises, but a clean hook for public disclosure imports.
    Expected columns (flexible names):
        ticker / asset_ticker / symbol
        transaction_date / tx_date / date
        amount / amount_range / amount_range_low
        representative / member
        owner / owner_name
        asset_description / description
    """

    def __init__(self, csv_path: str = "") -> None:
        self.csv_path = csv_path.strip()

    def fetch_signals(self, ticker: str, debug: bool = False) -> Tuple[List[EventIntelSignal], str]:
        if not self.csv_path:
            return [], "HOUSE_PTR_CSV not set"
        path = Path(self.csv_path)
        if not path.exists():
            return [], f"congressional CSV not found: {self.csv_path}"

        try:
            df = pd.read_csv(path)
        except Exception as e:
            return [], f"failed to read congressional CSV: {e}"

        df.columns = [str(c).strip().lower() for c in df.columns]
        ticker_cols = [c for c in ["ticker", "asset_ticker", "symbol"] if c in df.columns]
        if not ticker_cols:
            return [], "no ticker column found in congressional CSV"
        ticker_col = ticker_cols[0]

        date_col = next((c for c in ["transaction_date", "tx_date", "date"] if c in df.columns), None)
        amount_col = next((c for c in ["amount", "amount_range", "amount_range_low", "range"] if c in df.columns), None)
        member_col = next((c for c in ["representative", "member", "officeholder", "reporting_person"] if c in df.columns), None)
        desc_col = next((c for c in ["asset_description", "description", "asset_name"] if c in df.columns), None)

        filt = df[df[ticker_col].astype(str).str.upper().str.contains(fr"\b{re.escape(ticker.upper())}\b", regex=True, na=False)]
        if filt.empty:
            return [], "no congressional transactions for ticker"

        signals: List[EventIntelSignal] = []
        for _, row in filt.head(10).iterrows():
            event_date = parse_date(row.get(date_col) if date_col else None)
            age = days_ago(event_date)
            score = 3.0 if age <= 14 else 2.0 if age <= 45 else 1.0
            amount_text = str(row.get(amount_col, "") or "").strip() if amount_col else ""
            member_text = str(row.get(member_col, "") or "").strip() if member_col else ""
            desc_text = str(row.get(desc_col, "") or "").strip() if desc_col else ""
            details = "Congressional transaction disclosure match"
            if member_text:
                details += f" by {member_text}"
            if amount_text:
                details += f" amount {amount_text}"
            if desc_text:
                details += f" for {desc_text}"
            signals.append(
                EventIntelSignal(
                    signal_type="congressional_transaction",
                    title=f"Congressional transaction disclosure for {ticker.upper()}",
                    source="House/Senate disclosure import",
                    event_date=event_date,
                    score=score,
                    tags=["congress", "disclosure"],
                    details=details + ".",
                )
            )
        return signals, "ok"


# -----------------------------------------------------------------------------
# Scoring and merge logic
# -----------------------------------------------------------------------------


class EventIntelEngine:
    def __init__(
        self,
        sec_adapter: SecInsiderAdapter,
        sam_adapter: Optional[SamOpportunityAdapter] = None,
        congress_adapter: Optional[CongressionalCsvAdapter] = None,
    ) -> None:
        self.sec_adapter = sec_adapter
        self.sam_adapter = sam_adapter or SamOpportunityAdapter("")
        self.congress_adapter = congress_adapter or CongressionalCsvAdapter("")

    def scan_ticker(self, ticker: str, debug: bool = False) -> EventIntelResult:
        ticker = ticker.upper().strip()
        lookup = self.sec_adapter.ticker_map.get(ticker, debug=debug)
        company_name = lookup.title if lookup else ""

        sec_signals, sec_status = self.sec_adapter.fetch_signals(ticker, debug=debug)
        sam_signals, sam_status = self.sam_adapter.fetch_signals(ticker, company_name, debug=debug)
        congress_signals, congress_status = self.congress_adapter.fetch_signals(ticker, debug=debug)
        ownership_signals: List[EventIntelSignal] = []  # reserved for 13D/13G module later

        sec_score = self._aggregate_bucket(sec_signals)
        procurement_score = self._aggregate_bucket(sam_signals)
        congressional_score = self._aggregate_bucket(congress_signals)
        ownership_score = self._aggregate_bucket(ownership_signals)

        event_intel_score = clamp(
            sec_score * 0.40
            + procurement_score * 0.30
            + ownership_score * 0.20
            + congressional_score * 0.10,
            0.0,
            10.0,
        )

        all_signals = sec_signals + sam_signals + congress_signals + ownership_signals
        latest_event = max((s.event_date for s in all_signals), default=None)
        latest_days = days_ago(latest_event) if latest_event else None
        tags = sorted({tag for sig in all_signals for tag in sig.tags})
        sources = sorted({sig.source for sig in all_signals})
        confidence = self._label(event_intel_score, sec_score, procurement_score, congressional_score)
        reason = self._build_reason(
            ticker=ticker,
            sec_score=sec_score,
            procurement_score=procurement_score,
            congressional_score=congressional_score,
            latest_days=latest_days,
            sec_status=sec_status,
            sam_status=sam_status,
            congress_status=congress_status,
            signals=all_signals,
        )

        return EventIntelResult(
            scan_time_utc=now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ticker=ticker,
            company_name=company_name,
            sec_insider_score=round(sec_score, 2),
            congressional_score=round(congressional_score, 2),
            procurement_score=round(procurement_score, 2),
            ownership_score=round(ownership_score, 2),
            event_intel_score=round(event_intel_score, 2),
            signal_count=len(all_signals),
            days_since_latest_event=latest_days,
            confidence_label=confidence,
            event_tags=", ".join(tags),
            top_sources=", ".join(sources),
            reason=reason,
        )

    @staticmethod
    def _aggregate_bucket(signals: Sequence[EventIntelSignal]) -> float:
        if not signals:
            return 0.0
        ordered = sorted((s.score for s in signals), reverse=True)
        top = ordered[0]
        bonus = sum(ordered[1:3]) * 0.15
        return round(clamp(top + bonus, 0.0, 10.0), 2)

    @staticmethod
    def _label(event_intel_score: float, sec_score: float, procurement_score: float, congressional_score: float) -> str:
        if event_intel_score >= 7.5:
            return "High Conviction Event"
        if sec_score >= 6.0 and procurement_score >= 4.0:
            return "Multi-Source Alignment"
        if sec_score >= 5.5:
            return "Insider Activity Watch"
        if procurement_score >= 4.5:
            return "Procurement Watch"
        if congressional_score >= 2.5:
            return "Congressional Context"
        return "Low Event Signal"

    @staticmethod
    def _build_reason(
        ticker: str,
        sec_score: float,
        procurement_score: float,
        congressional_score: float,
        latest_days: Optional[int],
        sec_status: str,
        sam_status: str,
        congress_status: str,
        signals: Sequence[EventIntelSignal],
    ) -> str:
        if not signals:
            return f"no event-intel signals; SEC={sec_status}; SAM={sam_status}; Congress={congress_status}"
        phrases: List[str] = []
        if sec_score >= 6.0:
            phrases.append("meaningful insider filing activity")
        elif sec_score > 0:
            phrases.append("recent insider filings")
        if procurement_score >= 4.5:
            phrases.append("procurement signal present")
        if congressional_score >= 2.5:
            phrases.append("congressional disclosure overlap")
        if latest_days is not None:
            phrases.append(f"latest event {latest_days} day(s) ago")
        top_signal = max(signals, key=lambda s: s.score)
        phrases.append(top_signal.title.lower())
        return ", ".join(phrases)


# -----------------------------------------------------------------------------
# Input / output
# -----------------------------------------------------------------------------



def load_tickers_from_csv(path: str, ticker_column: str = "ticker") -> List[str]:
    df = pd.read_csv(path)
    if ticker_column not in df.columns:
        raise ValueError(f"Column '{ticker_column}' not found in {path}")
    tickers = sorted({str(x).upper().strip() for x in df[ticker_column].dropna().tolist() if str(x).strip()})
    return tickers



def append_history(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    ensure_parent_dir(path)
    incoming_df = pd.DataFrame(rows)
    path_obj = Path(path)

    if not path_obj.exists():
        incoming_df.to_csv(path_obj, index=False)
        return

    try:
        with open(path_obj, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
    except Exception:
        existing_header = []

    incoming_header = list(incoming_df.columns)
    if existing_header == incoming_header:
        incoming_df.to_csv(path_obj, mode="a", header=False, index=False)
        return

    try:
        existing_df = pd.read_csv(path_obj)
    except Exception:
        existing_df = pd.DataFrame(columns=existing_header)

    merged_columns = list(existing_df.columns)
    for column in incoming_header:
        if column not in merged_columns:
            merged_columns.append(column)

    combined = pd.concat(
        [
            existing_df.reindex(columns=merged_columns),
            incoming_df.reindex(columns=merged_columns),
        ],
        ignore_index=True,
    )
    combined.to_csv(path_obj, index=False)



def write_csv_safely(df: pd.DataFrame, path: str) -> str:
    ensure_parent_dir(path)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stem = Path(path).stem
        suffix = Path(path).suffix or ".csv"
        alt = f"{stem}_{now_utc().strftime('%Y%m%dT%H%M%SZ')}{suffix}"
        df.to_csv(alt, index=False)
        return alt


def apply_baseline_enrichment(results: Sequence[EventIntelResult], history_path: str) -> None:
    model = EventBaselineModel.from_csv(history_path)
    for result in results:
        assessment = model.assess_row(asdict(result))
        for key, value in assessment.to_dict().items():
            setattr(result, key, value)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRADE_SCANNER Event Intel Scanner")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols to scan")
    parser.add_argument("--input-csv", help="Optional CSV with a ticker column")
    parser.add_argument("--ticker-column", default="ticker", help="Ticker column for --input-csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--history", default=DEFAULT_HISTORY, help="Historical append CSV path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-history", action="store_true", help="Do not append to history CSV")
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tickers: List[str] = []
    if args.tickers:
        tickers.extend(args.tickers)
    if args.input_csv:
        tickers.extend(load_tickers_from_csv(args.input_csv, args.ticker_column))
    tickers = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})

    if not tickers:
        parser.error("Provide --tickers or --input-csv")

    print(f"Scanning {len(tickers)} ticker(s) for event intelligence...")

    ticker_map = SecTickerMap()
    sec_adapter = SecInsiderAdapter(ticker_map)
    sam_adapter = SamOpportunityAdapter(api_key=SAM_API_KEY)
    congress_adapter = CongressionalCsvAdapter(csv_path=HOUSE_PTR_CSV)
    engine = EventIntelEngine(sec_adapter, sam_adapter=sam_adapter, congress_adapter=congress_adapter)

    results: List[EventIntelResult] = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] {ticker}")
        try:
            results.append(engine.scan_ticker(ticker, debug=args.debug))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[WARN] Failed scanning {ticker}: {e}", file=sys.stderr)
            results.append(
                EventIntelResult(
                    scan_time_utc=now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ticker=ticker,
                    company_name="",
                    sec_insider_score=0.0,
                    congressional_score=0.0,
                    procurement_score=0.0,
                    ownership_score=0.0,
                    event_intel_score=0.0,
                    signal_count=0,
                    days_since_latest_event=None,
                    confidence_label="Scan Error",
                    event_tags="",
                    top_sources="",
                    reason=str(e),
                )
            )

    apply_baseline_enrichment(results, args.history)
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows).sort_values(by=["event_intel_score", "sec_insider_score", "procurement_score"], ascending=False)

    print("\nTop event-intel results:\n")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.head(20).to_string(index=False))

    actual_output = write_csv_safely(df, args.output)
    if not args.no_history:
        try:
            append_history(args.history, df.to_dict(orient="records"))
        except PermissionError:
            history_fallback = Path(args.history).with_name(
                f"{Path(args.history).stem}_{now_utc().strftime('%Y%m%dT%H%M%SZ')}{Path(args.history).suffix or '.csv'}"
            )
            df.to_csv(history_fallback, index=False)
            print(f"[WARN] Could not append history to {args.history}. Wrote history snapshot to {history_fallback} instead.")

    print(f"\nSaved output to: {actual_output}")
    if actual_output != args.output:
        print("Note: target file was locked, so a timestamped file was written instead.")


if __name__ == "__main__":
    main()
