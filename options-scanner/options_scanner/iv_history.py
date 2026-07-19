"""Persistent scan-history store for the percentile signal score.

The scanner is otherwise stateless; this is the one piece that
remembers past scans so today's IV excess can be ranked against a
ticker's own recent history ("92nd percentile of IV richness over the
past 30 days").

Storage is a single SQLite file (stdlib, no new dependency) under
options-scanner/cache/. Each scan appends one row per contract,
keyed by (ticker, scan_date); re-running the same ticker on the same
day replaces that day's rows so reruns don't inflate the distribution.

Percentile is computed against the pooled distribution of the
ticker's iv_excess over the trailing window. Until enough history
accumulates the score returns NaN (cold start) — the UI/CLI render
those as blank.

A custom DB path can be supplied via the OSC_IV_HISTORY_DB env var
(used by tests to avoid touching the real cache).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "cache" / "iv_history.db"
_MIN_HISTORY = 30          # pooled observations required before percentiles mean anything
_REQUIRED_COLS = ("type", "strike", "expiration", "dte", "iv_excess")
_NEW_COLS = ("mid", "delta", "ann_yield_pct")  # added post-launch; nullable for old rows


def _db_path() -> Path:
    return Path(os.environ.get("OSC_IV_HISTORY_DB", str(_DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS iv_history (
            ticker     TEXT NOT NULL,
            scan_date  TEXT NOT NULL,
            type       TEXT,
            strike     REAL,
            expiration TEXT,
            dte        INTEGER,
            iv_excess  REAL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_date "
        "ON iv_history (ticker, scan_date)"
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(iv_history)")}
    for col in _NEW_COLS:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE iv_history ADD COLUMN {col} REAL")
    return conn


def record_scan(ticker: str, df: pd.DataFrame,
                scan_day: date | None = None) -> None:
    """Persist today's chain snapshot for `ticker` (idempotent per day).

    No-op if df is empty or missing the columns we record — keeps the
    store from ever breaking a scan.
    """
    if df is None or df.empty or not ticker:
        return
    if not set(_REQUIRED_COLS) <= set(df.columns):
        return
    scan_date = (scan_day or date.today()).isoformat()
    ticker = ticker.upper()

    def _opt(col: str, r) -> float | None:
        if col not in df.columns:
            return None
        v = r[col]
        return float(v) if pd.notna(v) else None

    rows = [
        (ticker, scan_date, str(r["type"]), float(r["strike"]),
         str(r["expiration"]), int(r["dte"]), float(r["iv_excess"]),
         _opt("mid", r), _opt("delta", r), _opt("ann_yield_pct", r))
        for _, r in df.iterrows()
        if pd.notna(r["iv_excess"])
    ]
    if not rows:
        return
    try:
        with _connect() as conn:
            conn.execute(
                "DELETE FROM iv_history WHERE ticker = ? AND scan_date = ?",
                (ticker, scan_date),
            )
            conn.executemany(
                "INSERT INTO iv_history "
                "(ticker, scan_date, type, strike, expiration, dte, iv_excess, "
                " mid, delta, ann_yield_pct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
    except sqlite3.Error:
        return


def _pool(ticker: str, window_days: int) -> np.ndarray:
    """Trailing-window pool of the ticker's historical iv_excess values."""
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    try:
        with _connect() as conn:
            cur = conn.execute(
                "SELECT iv_excess FROM iv_history "
                "WHERE ticker = ? AND scan_date >= ?",
                (ticker.upper(), cutoff),
            )
            vals = [row[0] for row in cur.fetchall() if row[0] is not None]
    except sqlite3.Error:
        return np.empty(0)
    return np.asarray(vals, dtype=float)


def history_for(ticker: str, window_days: int = 30) -> pd.DataFrame:
    """Raw trailing-window scan-history rows for `ticker`.

    Lets the UI/CLI show what's actually been recorded — the only other
    reader of this store (`percentile_for`) collapses it to one number
    per row. Empty (right columns, zero rows) if nothing's recorded yet
    or the DB is unreachable, mirroring `_pool()`'s fail-open behavior.
    """
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    cols = ["scan_date", "type", "strike", "expiration", "dte", "iv_excess",
            "mid", "delta", "ann_yield_pct"]
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                "SELECT scan_date, type, strike, expiration, dte, iv_excess, "
                "       mid, delta, ann_yield_pct "
                "FROM iv_history WHERE ticker = ? AND scan_date >= ? "
                "ORDER BY scan_date, type, strike",
                conn, params=(ticker.upper(), cutoff),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    return df


def percentile_for(ticker: str, iv_excess, window_days: int = 30) -> np.ndarray:
    """Percentile rank (0–100) of each iv_excess value within the ticker's
    trailing-window pool. Returns NaN for every row during cold start
    (fewer than _MIN_HISTORY pooled observations)."""
    values = np.asarray(pd.Series(iv_excess).to_numpy(), dtype=float)
    pool = _pool(ticker, window_days)
    if pool.size < _MIN_HISTORY:
        return np.full(values.shape, np.nan)
    pool.sort()
    ranks = np.searchsorted(pool, values, side="right")
    return 100.0 * ranks / pool.size
