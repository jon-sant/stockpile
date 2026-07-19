"""Tests for the SQLite scan-history store behind the percentile score.

Uses a throwaway DB via the OSC_IV_HISTORY_DB env var so the real
cache is never touched.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from options_scanner import iv_history


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_IV_HISTORY_DB", str(tmp_path / "h.db"))


def _snapshot(n: int, base: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-06-19"] * n,
        "dte": [30] * n,
        "iv_excess": [base + 0.001 * i for i in range(n)],
        "mid": [1.0 + 0.01 * i for i in range(n)],
        "delta": [0.3 + 0.01 * i for i in range(n)],
        "ann_yield_pct": [10.0 + 0.1 * i for i in range(n)],
        "iv": [0.25 + 0.001 * i for i in range(n)],
        "open_interest": [100 + i for i in range(n)],
        "volume": [10 + i for i in range(n)],
        "spot": [98.0] * n,
    })


def _snapshot_legacy(n: int, base: float = 0.0) -> pd.DataFrame:
    """Pre-PR1 shape — no mid/delta/ann_yield_pct columns at all."""
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-06-19"] * n,
        "dte": [30] * n,
        "iv_excess": [base + 0.001 * i for i in range(n)],
    })


def test_record_is_idempotent_per_day():
    df = _snapshot(10)
    iv_history.record_scan("AMD", df, scan_day=date(2026, 5, 26))
    iv_history.record_scan("AMD", df, scan_day=date(2026, 5, 26))
    pool = iv_history._pool("AMD", window_days=365)
    assert pool.size == 10  # not 20 — same-day rerun replaced, not appended


def test_percentile_cold_start_returns_nan():
    iv_history.record_scan("AMD", _snapshot(5), scan_day=date.today())
    pct = iv_history.percentile_for("AMD", pd.Series([0.0, 0.1]))
    assert np.isnan(pct).all()


def test_percentile_ranks_against_pool():
    # Seed ≥30 observations across several recent days.
    today = date.today()
    for d in range(5):
        iv_history.record_scan(
            "AMD", _snapshot(10, base=0.0),
            scan_day=today - timedelta(days=d + 1),
        )
    pool = iv_history._pool("AMD", window_days=30)
    assert pool.size >= 30

    lo, hi = float(pool.min()), float(pool.max())
    pct = iv_history.percentile_for(
        "AMD", pd.Series([lo - 1.0, hi + 1.0]), window_days=30)
    assert pct[0] <= 5.0       # below the whole pool
    assert pct[1] >= 99.0      # above the whole pool


def test_record_noop_when_columns_missing():
    # Missing iv_excess → silently skipped, no crash, nothing stored.
    iv_history.record_scan("AMD", pd.DataFrame({"type": ["call"]}),
                           scan_day=date.today())
    assert iv_history._pool("AMD", window_days=365).size == 0


def test_percentile_for_empty_ticker_is_nan():
    pct = iv_history.percentile_for("NOPE", pd.Series([0.1, 0.2, 0.3]))
    assert np.isnan(pct).all()


def test_legacy_shaped_df_still_records_and_reads():
    # Old-style df (no mid/delta/ann_yield_pct/iv/open_interest/volume)
    # must not crash record_scan, and history_for must still return the
    # full (new) column set with NULL/NaN in the columns the legacy df
    # never had.
    iv_history.record_scan("AMD", _snapshot_legacy(5), scan_day=date.today())
    hist = iv_history.history_for("AMD")
    assert len(hist) == 5
    assert {"mid", "delta", "ann_yield_pct", "iv",
            "open_interest", "volume"} <= set(hist.columns)
    assert hist["mid"].isna().all()
    assert hist["delta"].isna().all()
    assert hist["ann_yield_pct"].isna().all()
    assert hist["open_interest"].isna().all()
    assert hist["volume"].isna().all()


def test_oi_and_volume_round_trip():
    iv_history.record_scan("AMD", _snapshot(3), scan_day=date.today())
    hist = iv_history.history_for("AMD")
    assert list(hist["open_interest"]) == [100.0, 101.0, 102.0]
    assert list(hist["volume"]) == [10.0, 11.0, 12.0]
    assert list(hist["iv"].round(3)) == [0.25, 0.251, 0.252]


# ── Monte Carlo columns ──────────────────────────────────────────────────


def test_spot_and_earnings_next_date_round_trip():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19),
                           earnings_next_date=date(2026, 8, 5))
    mc = iv_history.mc_results_for("AMD", scan_date=date(2026, 7, 19))
    assert not mc.empty
    pending = iv_history.pending_mc_rows(limit=10)
    assert list(pending["spot"]) == [98.0, 98.0]
    assert list(pending["earnings_next_date"]) == ["2026-08-05", "2026-08-05"]


def test_fresh_scan_rows_are_pending():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19))
    mc = iv_history.mc_results_for("AMD", scan_date=date(2026, 7, 19))
    assert mc["mc_status"].isna().all()
    for col in iv_history._MC_METRIC_COLS:
        assert mc[col].isna().all()


def test_legacy_rows_have_no_spot_and_are_pending():
    # A row recorded before this session's migration (no spot/earnings/mc_*
    # columns) must still read back as pending, not crash.
    iv_history.record_scan("AMD", _snapshot_legacy(2), scan_day=date(2026, 7, 19))
    pending = iv_history.pending_mc_rows(limit=10)
    assert len(pending) == 2
    assert pending["spot"].isna().all()
    assert pending["earnings_next_date"].isna().all()


def test_record_mc_results_marks_done_and_writes_metrics():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19))
    pending = iv_history.pending_mc_rows(limit=10)
    rowid = int(pending.iloc[0]["rowid"])
    results = {col: 1.5 for col in iv_history._MC_METRIC_COLS}
    iv_history.record_mc_results(rowid, results, duration_ms=42.0)

    mc = iv_history.mc_results_for("AMD", scan_date=date(2026, 7, 19))
    done_row = mc.iloc[0]
    assert done_row["mc_status"] == "done"
    for col in iv_history._MC_METRIC_COLS:
        assert done_row[col] == 1.5
    # The other row is untouched — still pending.
    assert pd.isna(mc.iloc[1]["mc_status"])

    still_pending = iv_history.pending_mc_rows(limit=10)
    assert rowid not in set(still_pending["rowid"])
    assert len(still_pending) == 1


def test_record_mc_error_marks_error_not_pending():
    iv_history.record_scan("AMD", _snapshot(1), scan_day=date(2026, 7, 19))
    rowid = int(iv_history.pending_mc_rows(limit=10).iloc[0]["rowid"])
    iv_history.record_mc_error(rowid, "boom: no positive IV")

    mc = iv_history.mc_results_for("AMD", scan_date=date(2026, 7, 19))
    assert mc.iloc[0]["mc_status"] == "error"
    # Errored rows are no longer "pending" (mc_status IS NULL) — they don't
    # get retried forever in a tight loop.
    assert iv_history.pending_mc_rows(limit=10).empty


def test_pending_mc_rows_prioritizes_ticker():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19))
    iv_history.record_scan("TSLA", _snapshot(2), scan_day=date(2026, 7, 19))
    pending = iv_history.pending_mc_rows(limit=10, priority_ticker="TSLA")
    assert list(pending["ticker"])[:2] == ["TSLA", "TSLA"]


def test_market_view_round_trip():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19),
                           market_view="Bullish")
    pending = iv_history.pending_mc_rows(limit=10)
    assert list(pending["market_view"]) == ["Bullish", "Bullish"]


def test_market_view_defaults_to_none():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19))
    pending = iv_history.pending_mc_rows(limit=10)
    assert pending["market_view"].isna().all()


def test_legacy_rows_have_no_market_view():
    iv_history.record_scan("AMD", _snapshot_legacy(2), scan_day=date(2026, 7, 19))
    pending = iv_history.pending_mc_rows(limit=10)
    assert pending["market_view"].isna().all()


def test_mc_rows_for_keys_returns_exact_matches_only():
    iv_history.record_scan("AMD", _snapshot(3), scan_day=date(2026, 7, 19),
                           market_view="Bearish")
    # _snapshot(3): strikes 100/101/102, all type "call", expiration 2026-06-19.
    keys = [("call", 100.0, "2026-06-19"), ("call", 999.0, "2026-06-19")]
    scoped = iv_history.mc_rows_for_keys("AMD", keys, scan_date=date(2026, 7, 19))
    assert len(scoped) == 1
    assert scoped.iloc[0]["strike"] == 100.0
    assert scoped.iloc[0]["market_view"] == "Bearish"


def test_mc_rows_for_keys_empty_keys_returns_empty():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19))
    result = iv_history.mc_rows_for_keys("AMD", [], scan_date=date(2026, 7, 19))
    assert result.empty


def test_mc_rows_for_keys_excludes_already_done_rows():
    iv_history.record_scan("AMD", _snapshot(2), scan_day=date(2026, 7, 19))
    pending = iv_history.pending_mc_rows(limit=10)
    rowid = int(pending.iloc[0]["rowid"])
    results = {col: 1.0 for col in iv_history._MC_METRIC_COLS}
    iv_history.record_mc_results(rowid, results, duration_ms=10.0)

    keys = [("call", 100.0, "2026-06-19"), ("call", 101.0, "2026-06-19")]
    scoped = iv_history.mc_rows_for_keys("AMD", keys, scan_date=date(2026, 7, 19))
    # Row 0 (strike 100.0) is already done — only row 1 (still pending) comes back.
    assert len(scoped) == 1
    assert scoped.iloc[0]["strike"] == 101.0


def test_schema_migration_preserves_existing_rows():
    # Simulate a pre-PR1 database: build the table with only the original
    # 7 columns, seed rows directly, then confirm _connect()'s ALTER TABLE
    # path adds the 3 new columns without touching existing data.
    import sqlite3

    db_path = iv_history._db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE iv_history (
            ticker     TEXT NOT NULL,
            scan_date  TEXT NOT NULL,
            type       TEXT,
            strike     REAL,
            expiration TEXT,
            dte        INTEGER,
            iv_excess  REAL
        )
    """)
    conn.execute(
        "INSERT INTO iv_history VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("AMD", date.today().isoformat(), "call", 100.0, "2026-06-19", 30, 0.05),
    )
    conn.commit()
    conn.close()

    # Any call through _connect() (e.g. via history_for) triggers migration.
    hist = iv_history.history_for("AMD")
    assert len(hist) == 1
    assert hist.iloc[0]["iv_excess"] == pytest.approx(0.05)
    assert hist.iloc[0]["strike"] == pytest.approx(100.0)
    assert pd.isna(hist.iloc[0]["mid"])
    assert pd.isna(hist.iloc[0]["delta"])
    assert pd.isna(hist.iloc[0]["ann_yield_pct"])


def test_dte_band_boundaries():
    assert iv_history._dte_band(14) == "0-14"
    assert iv_history._dte_band(15) == "15-45"
    assert iv_history._dte_band(45) == "15-45"
    assert iv_history._dte_band(46) == "46-90"
    assert iv_history._dte_band(90) == "46-90"
    assert iv_history._dte_band(91) == "91-+"
    assert iv_history._dte_band(365) == "91-+"


def test_bucket_uses_absolute_delta():
    # A put's negative delta buckets the same as the equivalent-magnitude
    # call delta — cohorting is about |delta|, not direction.
    assert iv_history._bucket(-0.30, 30) == iv_history._bucket(0.30, 30)
    assert iv_history._bucket(0.28, 30) == iv_history._bucket(0.30, 30)  # rounds to 0.30
    assert iv_history._bucket(0.30, 30) != iv_history._bucket(0.30, 60)  # diff DTE band


def _bucketed_snapshot(n: int, delta: float, dte: int, base_iv: float = 0.0,
                       base_ann: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["put"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-06-19"] * n,
        "dte": [dte] * n,
        "iv_excess": [base_iv + 0.001 * i for i in range(n)],
        "mid": [1.0] * n,
        "delta": [delta] * n,
        "ann_yield_pct": [base_ann + 0.1 * i for i in range(n)],
    })


def test_bucketed_percentile_isolates_cohorts():
    today = date.today()
    # Bucket A: delta 0.20, dte 30 — low iv_excess distribution (0..0.019).
    # Bucket B: delta 0.45, dte 60 — high iv_excess distribution (5..5.019).
    for d in range(3):
        iv_history.record_scan(
            "SPY", _bucketed_snapshot(20, delta=0.20, dte=30, base_iv=0.0),
            scan_day=today - timedelta(days=d + 1))
        iv_history.record_scan(
            "SPY", _bucketed_snapshot(20, delta=0.45, dte=60, base_iv=5.0),
            scan_day=today - timedelta(days=d + 10))

    # A query at bucket A's own level should rank high within A, even
    # though it would rank at 0 if pooled against bucket B's distribution.
    pct = iv_history.percentile_for(
        "SPY", pd.Series([0.02]), deltas=pd.Series([0.20]),
        dtes=pd.Series([30]))
    assert pct[0] >= 90.0  # near/above the top of bucket A's own pool

    pct_b = iv_history.percentile_for(
        "SPY", pd.Series([0.02]), deltas=pd.Series([0.45]),
        dtes=pd.Series([60]))
    assert np.isnan(pct_b).all() or pct_b[0] <= 5.0  # far below bucket B's pool


def test_bucketed_percentile_cold_start_per_bucket():
    # 28 total rows for the ticker, but split across two buckets (14 each)
    # so neither bucket alone reaches _MIN_HISTORY_BUCKETED (15).
    today = date.today()
    for d in range(2):
        iv_history.record_scan(
            "QQQ", _bucketed_snapshot(7, delta=0.20, dte=30),
            scan_day=today - timedelta(days=d + 1))
        iv_history.record_scan(
            "QQQ", _bucketed_snapshot(7, delta=0.45, dte=60),
            scan_day=today - timedelta(days=d + 10))
    pct = iv_history.percentile_for(
        "QQQ", pd.Series([0.0]), deltas=pd.Series([0.20]),
        dtes=pd.Series([30]))
    assert np.isnan(pct).all()


def test_percentile_for_backward_compatible_without_bucket_args():
    # Omitting deltas/dtes keeps the original whole-pool behavior.
    today = date.today()
    for d in range(5):
        iv_history.record_scan(
            "AMD", _snapshot(10, base=0.0), scan_day=today - timedelta(days=d + 1))
    pct = iv_history.percentile_for("AMD", pd.Series([100.0]))
    assert pct[0] >= 99.0


def test_ann_delta_percentile_ranks_within_bucket():
    today = date.today()
    for d in range(3):
        iv_history.record_scan(
            "SPY", _bucketed_snapshot(20, delta=0.20, dte=30, base_ann=10.0),
            scan_day=today - timedelta(days=d + 1))
    # ann_yield_pct/|delta| for the seeded rows ranges ~ (10/0.2)=50 to
    # (11.9/0.2)=59.5; a query well above that should rank near 100.
    pct = iv_history.ann_delta_percentile_for(
        "SPY", pd.Series([20.0]), pd.Series([0.20]), pd.Series([30]))
    assert pct[0] >= 90.0  # 20/0.2=100, far above the pool


def test_ann_delta_percentile_skips_legacy_null_rows():
    # Legacy rows (recorded before PR1) have NULL ann_yield_pct/delta and
    # must not crash or pollute the pool.
    iv_history.record_scan("AMD", _snapshot_legacy(20), scan_day=date.today())
    pct = iv_history.ann_delta_percentile_for(
        "AMD", pd.Series([50.0]), pd.Series([0.30]), pd.Series([30]))
    assert np.isnan(pct).all()


def test_record_and_read_round_trip_includes_new_columns():
    iv_history.record_scan("AMD", _snapshot(3), scan_day=date.today())
    hist = iv_history.history_for("AMD")
    assert len(hist) == 3
    assert hist["mid"].notna().all()
    assert hist["delta"].notna().all()
    assert hist["ann_yield_pct"].notna().all()
    row0 = hist.iloc[0]
    assert row0["mid"] == pytest.approx(1.0)
    assert row0["delta"] == pytest.approx(0.3)
    assert row0["ann_yield_pct"] == pytest.approx(10.0)


# ── IV Rank ──────────────────────────────────────────────────────────────────

def _iv_snapshot(n: int, iv: float) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-06-19"] * n,
        "dte": [30] * n,
        "iv_excess": [0.001 * i for i in range(n)],
        "iv": [iv] * n,
    })


def test_iv_rank_none_when_no_history_at_all():
    rank, d_from, d_to = iv_history.iv_rank_for("AMD", 0.30)
    assert rank is None
    assert d_from == d_to == date.today()


def test_iv_rank_none_with_only_one_day_of_data():
    # Recording *today's* iv into the DB doesn't help iv_rank_for on its
    # own — it needs a PRIOR day plus today's live value to form a range.
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.30), scan_day=date.today())
    rank, d_from, d_to = iv_history.iv_rank_for("AMD", 0.30)
    assert rank is None
    assert d_from == d_to == date.today()


def test_iv_rank_at_extremes():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.20),
                           scan_day=today - timedelta(days=10))
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.40),
                           scan_day=today - timedelta(days=5))
    # today's IV at the historical max -> rank 100
    rank_hi, d_from, d_to = iv_history.iv_rank_for("AMD", 0.40)
    assert rank_hi == pytest.approx(100.0)
    assert d_from == today - timedelta(days=10)
    assert d_to == today
    # today's IV at the historical min -> rank 0
    rank_lo, _, _ = iv_history.iv_rank_for("AMD", 0.20)
    assert rank_lo == pytest.approx(0.0)


def test_iv_rank_midpoint():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.20),
                           scan_day=today - timedelta(days=10))
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.40),
                           scan_day=today - timedelta(days=5))
    rank, _, _ = iv_history.iv_rank_for("AMD", 0.30)
    assert rank == pytest.approx(50.0)


def test_iv_rank_clamped_when_today_is_new_extreme():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.20),
                           scan_day=today - timedelta(days=10))
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.40),
                           scan_day=today - timedelta(days=5))
    rank, _, _ = iv_history.iv_rank_for("AMD", 0.90)  # new all-time high
    assert rank == pytest.approx(100.0)
    rank_lo, _, _ = iv_history.iv_rank_for("AMD", 0.01)  # new all-time low
    assert rank_lo == pytest.approx(0.0)


def test_iv_rank_none_when_no_spread():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.30),
                           scan_day=today - timedelta(days=5))
    rank, d_from, d_to = iv_history.iv_rank_for("AMD", 0.30)
    assert rank is None  # every observed value is identical, no range
    assert d_from == today - timedelta(days=5)
    assert d_to == today


def test_iv_rank_none_for_nan_or_none_today_iv():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.30),
                           scan_day=today - timedelta(days=5))
    assert iv_history.iv_rank_for("AMD", float("nan")) == (None, None, None)
    assert iv_history.iv_rank_for("AMD", None) == (None, None, None)


def test_iv_rank_uses_median_across_contracts_same_day():
    today = date.today()
    mixed = pd.DataFrame({
        "type": ["call"] * 3,
        "strike": [100.0, 105.0, 110.0],
        "expiration": ["2026-06-19"] * 3,
        "dte": [30] * 3,
        "iv_excess": [0.0, 0.001, 0.002],
        "iv": [0.10, 0.20, 0.90],  # median 0.20, mean ~0.4 — must use median
    })
    iv_history.record_scan("AMD", mixed, scan_day=today - timedelta(days=10))
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.40),
                           scan_day=today - timedelta(days=5))
    # Pool is [0.20 (day1 median, NOT the ~0.4 mean), 0.40 (day2)] + today.
    # If the mean were used instead, day1 would contribute ~0.4 too, and
    # this query would land at a different rank (or None, zero spread).
    rank, _, _ = iv_history.iv_rank_for("AMD", 0.30)
    assert rank == pytest.approx(50.0)


def test_iv_rank_window_limits_lookback():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.05),
                           scan_day=today - timedelta(days=400))  # outside window
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.15),
                           scan_day=today - timedelta(days=200))  # inside window
    iv_history.record_scan("AMD", _iv_snapshot(3, iv=0.25),
                           scan_day=today - timedelta(days=5))   # inside window
    rank, d_from, _ = iv_history.iv_rank_for("AMD", 0.40, window_days=365)  # new high
    assert d_from == today - timedelta(days=200)  # earliest IN-WINDOW date, not 400
    assert rank == pytest.approx(100.0)


def test_iv_rank_ignores_rows_with_null_iv():
    # Legacy rows (recorded before this column existed) have NULL iv.
    iv_history.record_scan("AMD", _snapshot_legacy(3), scan_day=date.today() - timedelta(days=5))
    rank, d_from, d_to = iv_history.iv_rank_for("AMD", 0.30)
    assert rank is None  # no usable prior history -> same as no history
    assert d_from == d_to == date.today()
