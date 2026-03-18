from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def append_csv_compatible(df: pd.DataFrame, history_path: str) -> None:
    path = Path(history_path)
    if path.exists():
        try:
            history_df = pd.read_csv(path)
        except Exception:
            history_df = pd.DataFrame()
    else:
        history_df = pd.DataFrame()

    merged_columns = list(history_df.columns)
    for col in df.columns:
        if col not in merged_columns:
            merged_columns.append(col)

    combined = pd.concat(
        [
            history_df.reindex(columns=merged_columns),
            df.reindex(columns=merged_columns),
        ],
        ignore_index=True,
    )
    combined.to_csv(path, index=False)


def _sqlite_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dtype):
        return "REAL"
    return "TEXT"


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


@dataclass
class SQLiteHistoryStore:
    db_path: str

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def append_dataframe(self, table_name: str, df: pd.DataFrame, run_id: Optional[str] = None) -> int:
        if df is None or df.empty:
            return 0

        payload = df.copy()
        payload["run_id"] = str(run_id or "")
        payload["inserted_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Keep sqlite writes simple and stable by stringifying datetime-like data.
        for col in payload.columns:
            if pd.api.types.is_datetime64_any_dtype(payload[col]):
                payload[col] = payload[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table(conn, table_name, payload)
            table_columns = self._table_columns(conn, table_name)
            payload = payload.reindex(columns=table_columns)
            payload.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
        return len(payload.index)

    def _ensure_table(self, conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
        existing_cols = self._table_columns(conn, table_name)
        if not existing_cols:
            column_defs = []
            for col in df.columns:
                col_type = _sqlite_type_for_series(df[col])
                column_defs.append(f"{_quote_identifier(col)} {col_type}")
            ddl = f"CREATE TABLE IF NOT EXISTS {_quote_identifier(table_name)} ({', '.join(column_defs)})"
            conn.execute(ddl)
            return

        for col in df.columns:
            if col in existing_cols:
                continue
            col_type = _sqlite_type_for_series(df[col])
            ddl = f"ALTER TABLE {_quote_identifier(table_name)} ADD COLUMN {_quote_identifier(col)} {col_type}"
            conn.execute(ddl)

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> list[str]:
        try:
            rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        except Exception:
            return []
        return [str(r[1]) for r in rows]
