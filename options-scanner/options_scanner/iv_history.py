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
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from options_scanner import sqlite_util

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "cache" / "iv_history.db"
_MIN_HISTORY = 30          # pooled observations required before percentiles mean anything
_MIN_HISTORY_BUCKETED = 15  # lower bar per delta×DTE bucket — same-bucket obs are scarcer
_REQUIRED_COLS = ("type", "strike", "expiration", "dte", "iv_excess")
# Added post-launch; nullable for old rows. (name, sql_type) so TEXT columns
# (mc_status etc.) don't get forced into REAL by the old plain-tuple form.
_NEW_COLS: tuple[tuple[str, str], ...] = (
    ("mid", "REAL"), ("delta", "REAL"), ("ann_yield_pct", "REAL"), ("iv", "REAL"),
    ("open_interest", "REAL"), ("volume", "REAL"),
    # Monte Carlo inputs — needed to rebuild a Position from a stored row
    # later (the live scan's df has these, but nothing persisted them
    # until now).
    ("spot", "REAL"), ("earnings_next_date", "TEXT"),
    # Directional stance at scan time (see options_scanner.market_view),
    # e.g. "Bullish" — used to derive Monte Carlo drift. NULL wherever
    # the scanning tab has no buy/sell x calls/puts/both concept to
    # derive one from (GEX, Spreads) — same "no state -> neutral"
    # fallback as any other missing input.
    ("market_view", "TEXT"),
    # Monte Carlo status/metadata. mc_status NULL == pending — every
    # pre-existing row is implicitly pending the moment this column
    # exists, no backfill UPDATE needed.
    ("mc_status", "TEXT"), ("mc_computed_at", "TEXT"),
    ("mc_duration_ms", "REAL"), ("mc_error", "TEXT"),
    # Monte Carlo results — side-invariant (contract properties, not
    # duplicated per buy/sell).
    ("mc_fair_value", "REAL"), ("mc_breakeven_move_pct", "REAL"),
    # Monte Carlo results — side-asymmetric (P&L-shaped metrics that
    # genuinely differ between holding the contract long vs. short).
    ("mc_prob_profit_buy", "REAL"), ("mc_prob_profit_sell", "REAL"),
    ("mc_expected_pnl_buy", "REAL"), ("mc_expected_pnl_sell", "REAL"),
    ("mc_cvar5_buy", "REAL"), ("mc_cvar5_sell", "REAL"),
    ("mc_var5_buy", "REAL"), ("mc_var5_sell", "REAL"),
    ("mc_sortino_buy", "REAL"), ("mc_sortino_sell", "REAL"),
    ("mc_edge_vs_market_buy", "REAL"), ("mc_edge_vs_market_sell", "REAL"),
)

_IV_RANK_WINDOW_DAYS = 365  # cap on how far back IV Rank looks
_IV_RANK_MIN_DAYS = 2       # need >=1 prior day + today to form a min/max range

_DELTA_STEP = 0.05
_DTE_BANDS: tuple[tuple[int, int | None], ...] = ((0, 14), (15, 45), (46, 90), (91, None))


def _dte_band(dte: int) -> str:
    """Which DTE bucket `dte` falls into, as a stable string key."""
    for lo, hi in _DTE_BANDS:
        if dte >= lo and (hi is None or dte <= hi):
            return f"{lo}-{hi if hi is not None else '+'}"
    return f"{_DTE_BANDS[-1][0]}-+"


def _bucket(delta: float, dte: int) -> tuple[float, str]:
    """Cohort key for a contract: |delta| rounded to the nearest 0.05,
    paired with its DTE band. Two contracts in the same bucket are
    "similar enough" to pool for a historical-percentile comparison."""
    rounded_delta = round(abs(delta) / _DELTA_STEP) * _DELTA_STEP
    return (round(rounded_delta, 10), _dte_band(int(dte)))


def _db_path() -> Path:
    return Path(os.environ.get("OSC_IV_HISTORY_DB", str(_DEFAULT_DB)))


@contextmanager
def _connect() -> Generator[sqlite3.Connection]:
    with sqlite_util.connect(_db_path()) as conn:
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
        for col, sql_type in _NEW_COLS:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE iv_history ADD COLUMN {col} {sql_type}")
        yield conn


def record_scan(ticker: str, df: pd.DataFrame,
                scan_day: date | None = None,
                earnings_next_date: date | None = None,
                market_view: str | None = None) -> None:
    """Persist today's chain snapshot for `ticker` (idempotent per day).

    No-op if df is empty or missing the columns we record — keeps the
    store from ever breaking a scan.

    `earnings_next_date` and `market_view` are ticker-scan-level
    (broadcast to every row, like `spot` already is per-row on `df`) —
    all three are Monte Carlo inputs the background worker needs to
    rebuild a Position (and now a drift) later, since the live df is
    gone by the time it runs. `market_view` is the outlook-card stance
    string (see options_scanner.market_view.stance_for) for whichever
    tab/CLI invocation did the scanning; callers with no buy/sell x
    calls/puts/both concept to derive one from (GEX, Spreads) simply
    don't pass it, leaving it NULL — resolves to no drift, same as any
    other "no state found" row. Rows recorded here never get their
    `mc_*` result columns filled in directly — those start implicitly
    pending (`mc_status IS NULL`) and are filled in later by the worker.
    """
    if df is None or df.empty or not ticker:
        return
    if not set(_REQUIRED_COLS) <= set(df.columns):
        return
    scan_date = (scan_day or date.today()).isoformat()
    ticker = ticker.upper()
    earnings_next_iso = earnings_next_date.isoformat() if earnings_next_date else None

    def _opt(col: str, r) -> float | None:
        if col not in df.columns:
            return None
        v = r[col]
        return float(v) if pd.notna(v) else None

    rows = [
        (ticker, scan_date, str(r["type"]), float(r["strike"]),
         str(r["expiration"]), int(r["dte"]), float(r["iv_excess"]),
         _opt("mid", r), _opt("delta", r), _opt("ann_yield_pct", r),
         _opt("iv", r), _opt("open_interest", r), _opt("volume", r),
         _opt("spot", r), earnings_next_iso, market_view)
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
                " mid, delta, ann_yield_pct, iv, open_interest, volume, "
                " spot, earnings_next_date, market_view) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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


def _pool_rows(ticker: str, window_days: int, value_col: str) -> pd.DataFrame:
    """Trailing-window (delta, dte, value_col) rows for bucketed pooling.
    Rows missing delta/dte/value_col (pre-PR1 legacy rows, or a value that
    was never recorded) are dropped — they can't be bucketed or ranked."""
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    cols = ["delta", "dte", value_col]
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                f"SELECT delta, dte, {value_col} FROM iv_history "
                "WHERE ticker = ? AND scan_date >= ?",
                conn, params=(ticker.upper(), cutoff),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    return df.dropna(subset=["delta", "dte", value_col])


def _bucketed_percentile(ticker: str, values, deltas, dtes,
                         window_days: int, value_col: str,
                         transform=lambda v, d: v) -> np.ndarray:
    """Shared engine for percentile_for/ann_delta_percentile_for: pool
    `value_col`'s trailing history bucketed by delta×DTE, apply `transform`
    to both the pool and the query values (identity for IV+pp, ÷|delta|
    for Ann%/Delta), then rank each query value within its own bucket's
    pool. NaN when the bucket has fewer than _MIN_HISTORY_BUCKETED rows."""
    values = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    deltas = np.asarray(pd.Series(deltas).to_numpy(), dtype=float)
    dtes = np.asarray(pd.Series(dtes).to_numpy(), dtype=float)
    out = np.full(values.shape, np.nan)

    pool = _pool_rows(ticker, window_days, value_col)
    if pool.empty:
        return out
    pool = pool.copy()
    pool["_bucket"] = [_bucket(d, t) for d, t in zip(pool["delta"], pool["dte"])]
    pool["_value"] = transform(pool[value_col].to_numpy(), pool["delta"].to_numpy())

    for i, (v, d, t) in enumerate(zip(values, deltas, dtes)):
        if not (np.isfinite(v) and np.isfinite(d) and np.isfinite(t)):
            continue
        sub = pool.loc[pool["_bucket"] == _bucket(d, t), "_value"].to_numpy()
        sub = sub[np.isfinite(sub)]
        if sub.size < _MIN_HISTORY_BUCKETED:
            continue
        q = transform(np.array([v]), np.array([d]))[0]
        if not np.isfinite(q):
            continue
        sub = np.sort(sub)
        out[i] = 100.0 * np.searchsorted(sub, q, side="right") / sub.size
    return out


# The 14 Monte Carlo metric columns — exactly the keys mc_batch.py's
# compute_mc_for_row() returns, and exactly what record_mc_results() writes.
_MC_METRIC_COLS = (
    "mc_fair_value", "mc_breakeven_move_pct",
    "mc_prob_profit_buy", "mc_prob_profit_sell",
    "mc_expected_pnl_buy", "mc_expected_pnl_sell",
    "mc_cvar5_buy", "mc_cvar5_sell",
    "mc_var5_buy", "mc_var5_sell",
    "mc_sortino_buy", "mc_sortino_sell",
    "mc_edge_vs_market_buy", "mc_edge_vs_market_sell",
)


def mc_results_for(ticker: str, scan_date: date | None = None) -> pd.DataFrame:
    """MC status + result columns for every (type, strike, expiration) row
    recorded for `ticker` on `scan_date` (default today).

    Used by fetch.py to attach already-computed MC metrics back onto a
    freshly-scanned chain. On a brand-new scan every row will show
    mc_status IS NULL (pending — the UI renders this as "TBD") since the
    background worker hasn't caught up yet; a same-day rescan may already
    carry real values if the worker ran in between.
    """
    scan_date_iso = (scan_date or date.today()).isoformat()
    cols = ["type", "strike", "expiration", "mc_status", *_MC_METRIC_COLS]
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                "SELECT type, strike, expiration, mc_status, "
                + ", ".join(_MC_METRIC_COLS)
                + " FROM iv_history WHERE ticker = ? AND scan_date = ?",
                conn, params=(ticker.upper(), scan_date_iso),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    return df


def pending_mc_rows(limit: int, priority_ticker: str | None = None) -> pd.DataFrame:
    """Rows still needing Monte Carlo computation (mc_status IS NULL), up
    to `limit`, ordered so `priority_ticker`'s rows (if any) come first —
    lets the background worker prioritize whatever the user just scanned
    without a separate priority column on every row.

    `rowid` (SQLite's implicit row identifier) is the update handle for
    `record_mc_results`/`record_mc_error` — avoids reconstructing a
    composite (ticker, scan_date, type, strike, expiration) WHERE clause.
    """
    cols = ["rowid", "ticker", "scan_date", "type", "strike", "expiration",
            "dte", "mid", "iv", "spot", "earnings_next_date", "market_view"]
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                "SELECT rowid, ticker, scan_date, type, strike, expiration, "
                "       dte, mid, iv, spot, earnings_next_date, market_view "
                "FROM iv_history WHERE mc_status IS NULL "
                "ORDER BY (ticker = ?) DESC, scan_date DESC LIMIT ?",
                conn, params=(priority_ticker.upper() if priority_ticker else "", limit),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    return df


def mc_rows_for_keys(ticker: str, keys: list[tuple[str, float, str]],
                     scan_date: date | None = None) -> pd.DataFrame:
    """Still-pending rows (mc_status IS NULL) among an exact set of
    `(type, strike, expiration)` keys for `ticker`'s `scan_date` (default
    today) — same shape as `pending_mc_rows`, but scoped to precisely the
    rows a caller wants to block on (e.g. the top-N rows about to be
    displayed), rather than a priority-ordered slice of the whole
    backlog. Keys already `mc_status='done'`/`'error'` are simply absent
    from the result — the caller doesn't need to distinguish "already
    computed" from "not requested".

    Empty `keys` (or no matches) returns an empty frame — never queries
    with an empty IN (...) clause.
    """
    cols = ["rowid", "ticker", "scan_date", "type", "strike", "expiration",
            "dte", "mid", "iv", "spot", "earnings_next_date", "market_view"]
    if not keys:
        return pd.DataFrame(columns=cols)
    scan_date_iso = (scan_date or date.today()).isoformat()
    # One placeholder pair per key: type = ? AND strike = ? AND expiration = ?
    key_clause = " OR ".join(["(type = ? AND strike = ? AND expiration = ?)"] * len(keys))
    params: list = [ticker.upper(), scan_date_iso]
    for t, strike, expiration in keys:
        params.extend([t, float(strike), expiration])
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                "SELECT rowid, ticker, scan_date, type, strike, expiration, "
                "       dte, mid, iv, spot, earnings_next_date, market_view "
                "FROM iv_history WHERE ticker = ? AND scan_date = ? "
                f"AND mc_status IS NULL AND ({key_clause})",
                conn, params=params,
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    return df


def record_mc_results(rowid: int, results: dict[str, float],
                       duration_ms: float) -> None:
    """Persist a completed Monte Carlo computation for one row (matched by
    SQLite rowid, from `pending_mc_rows`). Sets mc_status='done'. Fails
    open — a storage error here must never crash the background worker.
    """
    set_clause = ", ".join(f"{col} = ?" for col in _MC_METRIC_COLS)
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE iv_history SET mc_status = 'done', "
                f"mc_computed_at = ?, mc_duration_ms = ?, {set_clause} "
                "WHERE rowid = ?",
                (
                    datetime.now().astimezone().isoformat(),
                    duration_ms,
                    *(results.get(col) for col in _MC_METRIC_COLS),
                    rowid,
                ),
            )
    except sqlite3.Error:
        return


def record_mc_error(rowid: int, error: str) -> None:
    """Mark one row's Monte Carlo computation as failed (matched by
    rowid). Fails open, same as record_mc_results."""
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE iv_history SET mc_status = 'error', mc_error = ? "
                "WHERE rowid = ?",
                (error, rowid),
            )
    except sqlite3.Error:
        return


def history_for(ticker: str, window_days: int = 30) -> pd.DataFrame:
    """Raw trailing-window scan-history rows for `ticker`.

    Lets the UI/CLI show what's actually been recorded — the only other
    reader of this store (`percentile_for`) collapses it to one number
    per row. Empty (right columns, zero rows) if nothing's recorded yet
    or the DB is unreachable, mirroring `_pool()`'s fail-open behavior.
    """
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    cols = ["scan_date", "type", "strike", "expiration", "dte", "iv_excess",
            "mid", "delta", "ann_yield_pct", "iv", "open_interest", "volume"]
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                "SELECT scan_date, type, strike, expiration, dte, iv_excess, "
                "       mid, delta, ann_yield_pct, iv, open_interest, volume "
                "FROM iv_history WHERE ticker = ? AND scan_date >= ? "
                "ORDER BY scan_date, type, strike",
                conn, params=(ticker.upper(), cutoff),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    return df


def percentile_for(ticker: str, iv_excess, window_days: int = 30,
                    deltas=None, dtes=None) -> np.ndarray:
    """Percentile rank (0–100) of each iv_excess value within the ticker's
    trailing-window pool.

    Without `deltas`/`dtes`: pools the ticker's WHOLE chain history
    together (original, unbucketed behavior — kept for backward
    compatibility). NaN for every row during cold start (fewer than
    _MIN_HISTORY pooled observations).

    With `deltas`/`dtes` supplied: pools only same delta×DTE-bucket
    history (see `_bucket`), so a 7-DTE 0.10-delta contract is ranked
    against similar contracts, not diluted by a 90-DTE 0.60-delta one.
    NaN when that bucket has fewer than _MIN_HISTORY_BUCKETED rows.
    """
    if deltas is None or dtes is None:
        values = np.asarray(pd.Series(iv_excess).to_numpy(), dtype=float)
        pool = _pool(ticker, window_days)
        if pool.size < _MIN_HISTORY:
            return np.full(values.shape, np.nan)
        pool.sort()
        ranks = np.searchsorted(pool, values, side="right")
        return 100.0 * ranks / pool.size
    return _bucketed_percentile(ticker, iv_excess, deltas, dtes,
                                window_days, "iv_excess")


def ann_delta_percentile_for(ticker: str, ann_yield_pct, deltas, dtes,
                              window_days: int = 30) -> np.ndarray:
    """Percentile rank (0–100) of each contract's Ann% ÷ |Delta| within
    the ticker's own delta×DTE-bucketed trailing history. Requires PR1's
    `ann_yield_pct`/`delta` columns; rows recorded before that (NULL)
    are excluded from the pool. NaN when the bucket has fewer than
    _MIN_HISTORY_BUCKETED rows, or when a row's own delta is ~0."""
    def _yield_per_delta(v, d):
        safe_d = np.where(np.abs(d) > 1e-6, np.abs(d), np.nan)
        return v / safe_d
    return _bucketed_percentile(ticker, ann_yield_pct, deltas, dtes,
                                window_days, "ann_yield_pct",
                                transform=_yield_per_delta)


def _daily_representative_iv(ticker: str, window_days: int) -> pd.DataFrame:
    """One row per scan_date: that day's median IV across every recorded
    contract for `ticker` — a simple, robust stand-in for "the" ticker's
    implied-vol level that day. Median (not a single ATM contract) so a
    strike/expiration rolling out of the chain between scans can't create
    a gap or a discontinuity in the series. Rows with NULL `iv` (pre-PR
    legacy rows) are excluded. Empty (right columns, zero rows) on no
    history or a storage error, mirroring this module's fail-open style."""
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    cols = ["scan_date", "iv"]
    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                "SELECT scan_date, iv FROM iv_history "
                "WHERE ticker = ? AND scan_date >= ? AND iv IS NOT NULL",
                conn, params=(ticker.upper(), cutoff),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=cols)
    if df.empty:
        return df
    return df.groupby("scan_date", as_index=False)["iv"].median()


def iv_rank_for(ticker: str, today_iv: float | None,
                 window_days: int = _IV_RANK_WINDOW_DAYS
                 ) -> tuple[float | None, date | None, date | None]:
    """0-100 IV Rank: where `today_iv` (the ticker's representative IV
    for today — see _daily_representative_iv) sits between the min and
    max representative IV seen for `ticker` over the trailing
    `window_days` (default 365), INCLUDING today.

    Rank = 100 * (today_iv - min) / (max - min), clamped to [0, 100].

    Returns (rank, earliest_date_used, latest_date_used). The date range
    reflects whatever history actually exists — it can be much shorter
    than `window_days` when a ticker has only recently started being
    scanned, which is by design (a rank computed over 3 weeks of history
    is still a real rank, just not a 365-day one — the caller surfaces
    the date range so that distinction isn't hidden).

    Returns (None, None, None) when there's no usable history at all,
    and (None, only_date, only_date) when today would be the *only*
    data point (a min/max range of a single value isn't a rank).
    """
    if today_iv is None or not (today_iv == today_iv):  # None or NaN
        return None, None, None
    daily = _daily_representative_iv(ticker, window_days)
    today = date.today()
    if daily.empty:
        return None, today, today
    prior_dates = [datetime.strptime(d, "%Y-%m-%d").date()
                   for d in daily["scan_date"]]
    all_dates = prior_dates + [today]
    d_from, d_to = min(all_dates), max(all_dates)
    if len(set(all_dates)) < _IV_RANK_MIN_DAYS:
        return None, d_from, d_to

    values = list(daily["iv"]) + [today_iv]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return None, d_from, d_to  # no meaningful spread yet

    rank = 100.0 * (today_iv - lo) / (hi - lo)
    return max(0.0, min(100.0, rank)), d_from, d_to
