"""Tests for the persistent raw-chain snapshot store behind background
scanning + screen auto-populate.

Uses a throwaway DB via the OSC_CHAIN_CACHE_DB env var so the real
cache is never touched.
"""

from datetime import date

import pandas as pd
import pytest

from options_scanner import chain_cache


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_CHAIN_CACHE_DB", str(tmp_path / "c.db"))


def _chain(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-06-19"] * n,
        "dte": [30] * n,
        "bid": [1.0] * n,
        "ask": [1.2] * n,
    })


def test_round_trip_preserves_data():
    df = _chain(5)
    chain_cache.save_snapshot("AMD", df, 30, 90, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    result = chain_cache.load_snapshot("AMD", 30, 90, "yahoo-headless",
                                       scan_day=date(2026, 5, 26))
    assert result is not None
    loaded, fetched_at = result
    pd.testing.assert_frame_equal(loaded, df)
    assert fetched_at is not None


def test_miss_on_different_scan_day():
    chain_cache.save_snapshot("AMD", _chain(), 30, 90, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    result = chain_cache.load_snapshot("AMD", 30, 90, "yahoo-headless",
                                       scan_day=date(2026, 5, 27))
    assert result is None


def test_miss_on_different_provider():
    chain_cache.save_snapshot("AMD", _chain(), 30, 90, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    result = chain_cache.load_snapshot("AMD", 30, 90, "schwab",
                                       scan_day=date(2026, 5, 26))
    assert result is None


def test_miss_when_requested_range_not_covered():
    chain_cache.save_snapshot("AMD", _chain(), 30, 60, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    # requested min_dte below stored min_dte -> not covered
    assert chain_cache.load_snapshot(
        "AMD", 10, 60, "yahoo-headless", scan_day=date(2026, 5, 26)) is None
    # requested max_dte above stored max_dte -> not covered
    assert chain_cache.load_snapshot(
        "AMD", 30, 90, "yahoo-headless", scan_day=date(2026, 5, 26)) is None


def test_hit_on_subset_range():
    chain_cache.save_snapshot("AMD", _chain(), 7, 365, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    result = chain_cache.load_snapshot("AMD", 30, 60, "yahoo-headless",
                                       scan_day=date(2026, 5, 26))
    assert result is not None


def test_no_upper_limit_range():
    chain_cache.save_snapshot("AMD", _chain(), 7, None, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    # A concrete upper limit is satisfied by an unbounded stored range.
    assert chain_cache.load_snapshot(
        "AMD", 30, 90, "yahoo-headless", scan_day=date(2026, 5, 26)) is not None
    # A stored bounded range does NOT satisfy an unbounded request.
    chain_cache.save_snapshot("MSFT", _chain(), 7, 90, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    assert chain_cache.load_snapshot(
        "MSFT", 30, None, "yahoo-headless", scan_day=date(2026, 5, 26)) is None


def test_narrower_save_does_not_shrink_wider_snapshot():
    wide = _chain(5)
    narrow = _chain(2)
    chain_cache.save_snapshot("AMD", wide, 7, 365, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    chain_cache.save_snapshot("AMD", narrow, 30, 60, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    result = chain_cache.load_snapshot("AMD", 7, 365, "yahoo-headless",
                                       scan_day=date(2026, 5, 26))
    assert result is not None
    loaded, _ = result
    pd.testing.assert_frame_equal(loaded, wide)  # not overwritten by narrow


def test_wider_save_overwrites_narrower_snapshot():
    narrow = _chain(2)
    wide = _chain(5)
    chain_cache.save_snapshot("AMD", narrow, 30, 60, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    chain_cache.save_snapshot("AMD", wide, 7, 365, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    result = chain_cache.load_snapshot("AMD", 7, 365, "yahoo-headless",
                                       scan_day=date(2026, 5, 26))
    assert result is not None
    loaded, _ = result
    pd.testing.assert_frame_equal(loaded, wide)


def test_has_fresh_snapshot_matches_load_snapshot():
    assert chain_cache.has_fresh_snapshot(
        "AMD", 30, 90, "yahoo-headless", scan_day=date(2026, 5, 26)) is False
    chain_cache.save_snapshot("AMD", _chain(), 30, 90, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    assert chain_cache.has_fresh_snapshot(
        "AMD", 30, 90, "yahoo-headless", scan_day=date(2026, 5, 26)) is True
    assert chain_cache.has_fresh_snapshot(
        "AMD", 10, 90, "yahoo-headless", scan_day=date(2026, 5, 26)) is False


def test_save_noop_on_empty_or_missing_df():
    chain_cache.save_snapshot("AMD", pd.DataFrame(), 30, 90, "yahoo-headless")
    chain_cache.save_snapshot("AMD", None, 30, 90, "yahoo-headless")
    assert chain_cache.load_snapshot("AMD", 30, 90, "yahoo-headless") is None


def test_ticker_is_case_insensitive():
    chain_cache.save_snapshot("amd", _chain(), 30, 90, "yahoo-headless",
                              scan_day=date(2026, 5, 26))
    assert chain_cache.load_snapshot(
        "AMD", 30, 90, "yahoo-headless", scan_day=date(2026, 5, 26)) is not None
