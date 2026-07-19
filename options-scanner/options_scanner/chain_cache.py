"""Persistent raw-chain snapshot cache, so a same-day scan doesn't have
to hit the network twice.

Storage is a single SQLite file (stdlib, no new dependency) under
options-scanner/cache/ — same idiom as iv_history.py. Each row holds one
ticker's RAW (pre-enrichment) chain for one day, pickled whole. Caching
at the raw-chain boundary (before earnings annotation / surface fit /
scoring) rather than the enriched result means a cache hit still gets
re-enriched against whatever surface-fit/score settings the caller has
selected right now — only the expensive network/Selenium fetch is
skipped.

Keyed by (ticker, scan_date, provider) — NOT opt_type: every real caller
fetches opt_type="both" today (fit_both_sides defaults True everywhere),
so the cache always stores the two-sided chain and callers filter to one
side afterward, same as the enrichment layer already does. min_dte/
max_dte are stored as plain columns and checked by *range containment*
in load_snapshot, not as part of the key, so a wide background-scanned
range satisfies any narrower request.

A custom DB path can be supplied via the OSC_CHAIN_CACHE_DB env var
(used by tests to avoid touching the real cache).
"""

from __future__ import annotations

import io
import os
import pickle
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "cache" / "chain_cache.db"


def _db_path() -> Path:
    return Path(os.environ.get("OSC_CHAIN_CACHE_DB", str(_DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chain_snapshots (
            ticker     TEXT NOT NULL,
            scan_date  TEXT NOT NULL,
            provider   TEXT NOT NULL,
            min_dte    INTEGER NOT NULL,
            max_dte    INTEGER,
            fetched_at TEXT NOT NULL,
            payload    BLOB NOT NULL,
            PRIMARY KEY (ticker, scan_date, provider)
        )
        """
    )
    return conn


def _covers(cached_min: int, cached_max: int | None,
            req_min: int, req_max: int | None) -> bool:
    """Whether a cached [cached_min, cached_max] range covers the
    requested [req_min, req_max] range. None max = no upper limit."""
    if cached_min > req_min:
        return False
    if req_max is None:
        return cached_max is None
    return cached_max is None or cached_max >= req_max


def save_snapshot(ticker: str, df: pd.DataFrame, min_dte: int,
                   max_dte: int | None, provider: str,
                   scan_day: date | None = None) -> None:
    """Persist today's raw chain for (ticker, provider).

    No-op if df is empty/None, or if an existing same-day row for this
    (ticker, provider) already covers [min_dte, max_dte] — so a narrower
    live Scan never shrinks a wider background-scanned snapshot. Rows
    are never merged across two fetches; the wider one simply wins.
    """
    if df is None or df.empty or not ticker or not provider:
        return
    scan_date = (scan_day or date.today()).isoformat()
    ticker = ticker.upper()
    fetched_at = datetime.now().astimezone().isoformat()

    buf = io.BytesIO()
    pickle.dump(df, buf, protocol=pickle.HIGHEST_PROTOCOL)
    payload = buf.getvalue()

    try:
        with _connect() as conn:
            cur = conn.execute(
                "SELECT min_dte, max_dte FROM chain_snapshots "
                "WHERE ticker = ? AND scan_date = ? AND provider = ?",
                (ticker, scan_date, provider),
            )
            row = cur.fetchone()
            if row is not None and _covers(row[0], row[1], min_dte, max_dte):
                return  # existing snapshot already at least as wide
            conn.execute(
                "INSERT INTO chain_snapshots "
                "(ticker, scan_date, provider, min_dte, max_dte, fetched_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, scan_date, provider) DO UPDATE SET "
                "min_dte=excluded.min_dte, max_dte=excluded.max_dte, "
                "fetched_at=excluded.fetched_at, payload=excluded.payload",
                (ticker, scan_date, provider, min_dte, max_dte, fetched_at, payload),
            )
    except sqlite3.Error:
        return


def load_snapshot(ticker: str, min_dte: int, max_dte: int | None,
                   provider: str, scan_day: date | None = None
                   ) -> tuple[pd.DataFrame, datetime] | None:
    """Return (df, fetched_at) if today's (ticker, provider) snapshot
    covers [min_dte, max_dte], else None. Fails open (None) on any
    storage error — a broken cache must never block a live scan."""
    if not ticker or not provider:
        return None
    scan_date = (scan_day or date.today()).isoformat()
    ticker = ticker.upper()
    try:
        with _connect() as conn:
            cur = conn.execute(
                "SELECT min_dte, max_dte, fetched_at, payload FROM chain_snapshots "
                "WHERE ticker = ? AND scan_date = ? AND provider = ?",
                (ticker, scan_date, provider),
            )
            row = cur.fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    cached_min, cached_max, fetched_at_raw, payload = row
    if not _covers(cached_min, cached_max, min_dte, max_dte):
        return None
    try:
        df = pickle.loads(payload)
        fetched_at = datetime.fromisoformat(fetched_at_raw)
    except Exception:
        return None
    return df, fetched_at


def has_fresh_snapshot(ticker: str, min_dte: int, max_dte: int | None,
                        provider: str, scan_day: date | None = None) -> bool:
    """Cheap existence check — same coverage rule as load_snapshot, no
    payload deserialize."""
    if not ticker or not provider:
        return False
    scan_date = (scan_day or date.today()).isoformat()
    ticker = ticker.upper()
    try:
        with _connect() as conn:
            cur = conn.execute(
                "SELECT min_dte, max_dte FROM chain_snapshots "
                "WHERE ticker = ? AND scan_date = ? AND provider = ?",
                (ticker, scan_date, provider),
            )
            row = cur.fetchone()
    except sqlite3.Error:
        return False
    if row is None:
        return False
    return _covers(row[0], row[1], min_dte, max_dte)
