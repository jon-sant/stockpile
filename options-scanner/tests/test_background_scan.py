"""Tests for the daily background-scan scheduler.

Uses throwaway DBs via the OSC_IV_HISTORY_DB / OSC_CHAIN_CACHE_DB env
vars, and always monkeypatches chain.fetch_chain — no test here touches
real Yahoo/Selenium.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from options_scanner import background_scan, chain_cache, iv_history


@pytest.fixture(autouse=True)
def _temp_dbs(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_IV_HISTORY_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("OSC_CHAIN_CACHE_DB", str(tmp_path / "c.db"))


def _iv_snapshot(n: int, dte: int) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-06-19"] * n,
        "dte": [dte] * n,
        "iv_excess": [0.001 * i for i in range(n)],
    })


def _chain(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "dte": [30] * n,
    })


# ── _target_universe ────────────────────────────────────────────────────────

def test_target_universe_unions_dte_ranges_within_window():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=90), scan_day=today - timedelta(days=5))
    universe = background_scan._target_universe(window_days=14)
    assert dict((t, (lo, hi)) for t, lo, hi in universe) == {"AMD": (30, 90)}


def test_target_universe_excludes_tickers_outside_window():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    iv_history.record_scan("OLD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=30))
    tickers = {t for t, _, _ in background_scan._target_universe(window_days=14)}
    assert tickers == {"AMD"}


def test_target_universe_empty_when_no_history():
    assert background_scan._target_universe() == []


# ── _is_trading_day ─────────────────────────────────────────────────────────

def test_is_trading_day_weekday():
    assert background_scan._is_trading_day(date(2026, 7, 20)) is True  # Monday


def test_is_trading_day_weekend():
    assert background_scan._is_trading_day(date(2026, 7, 25)) is False  # Saturday
    assert background_scan._is_trading_day(date(2026, 7, 26)) is False  # Sunday


# ── _run_pass ────────────────────────────────────────────────────────────────

def test_run_pass_full_success(monkeypatch):
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    iv_history.record_scan("MSFT", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))

    calls = []

    def _fake_fetch_chain(ticker, opt_type, min_dte, max_dte, provider):
        calls.append((ticker, provider))
        return _chain()

    monkeypatch.setattr("options_scanner.chain.fetch_chain", _fake_fetch_chain)
    ok = background_scan._run_pass(today)
    assert ok is True
    assert {c[0] for c in calls} == {"AMD", "MSFT"}
    assert all(c[1] == "yahoo-headless" for c in calls)
    assert chain_cache.has_fresh_snapshot("AMD", 30, 30, "yahoo-headless", scan_day=today)
    assert chain_cache.has_fresh_snapshot("MSFT", 30, 30, "yahoo-headless", scan_day=today)


def test_run_pass_partial_failure_returns_false(monkeypatch):
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    iv_history.record_scan("BAD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))

    def _fake_fetch_chain(ticker, opt_type, min_dte, max_dte, provider):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return _chain()

    monkeypatch.setattr("options_scanner.chain.fetch_chain", _fake_fetch_chain)
    ok = background_scan._run_pass(today)
    assert ok is False
    assert chain_cache.has_fresh_snapshot("AMD", 30, 30, "yahoo-headless", scan_day=today)
    assert not chain_cache.has_fresh_snapshot("BAD", 30, 30, "yahoo-headless", scan_day=today)


def test_run_pass_empty_chain_counts_as_failure(monkeypatch):
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    monkeypatch.setattr("options_scanner.chain.fetch_chain",
                        lambda *a, **k: pd.DataFrame())
    assert background_scan._run_pass(today) is False


def test_run_pass_skips_already_cached_tickers(monkeypatch):
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    chain_cache.save_snapshot("AMD", _chain(), 30, 30, "yahoo-headless", scan_day=today)

    calls = []
    monkeypatch.setattr(
        "options_scanner.chain.fetch_chain",
        lambda ticker, **k: calls.append(ticker) or _chain(),
    )
    ok = background_scan._run_pass(today)
    assert ok is True
    assert calls == []  # already had a fresh snapshot, no fetch needed


def test_run_pass_no_targets_is_success():
    assert background_scan._run_pass(date.today()) is True


# ── _pass_succeeded_for ─────────────────────────────────────────────────────

def test_pass_succeeded_for_true_when_all_cached():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    chain_cache.save_snapshot("AMD", _chain(), 30, 30, "yahoo-headless", scan_day=today)
    assert background_scan._pass_succeeded_for(today) is True


def test_pass_succeeded_for_false_when_partially_cached():
    today = date.today()
    iv_history.record_scan("AMD", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    iv_history.record_scan("MSFT", _iv_snapshot(3, dte=30), scan_day=today - timedelta(days=1))
    chain_cache.save_snapshot("AMD", _chain(), 30, 30, "yahoo-headless", scan_day=today)
    assert background_scan._pass_succeeded_for(today) is False


def test_pass_succeeded_for_true_when_no_universe():
    assert background_scan._pass_succeeded_for(date.today()) is True


# ── _tick ────────────────────────────────────────────────────────────────────

def test_tick_noop_on_non_trading_day(monkeypatch):
    monkeypatch.setattr(background_scan, "date",
                        type("D", (), {"today": staticmethod(lambda: date(2026, 7, 25))}))
    calls = []
    monkeypatch.setattr(background_scan, "_run_pass", lambda d: calls.append(d) or True)
    background_scan._tick()
    assert calls == []


def test_tick_runs_pass_when_incomplete(monkeypatch):
    fixed_today = date(2026, 7, 20)  # Monday
    monkeypatch.setattr(background_scan, "date",
                        type("D", (), {"today": staticmethod(lambda: fixed_today)}))
    monkeypatch.setattr(background_scan, "_pass_succeeded_for", lambda d: False)
    calls = []
    monkeypatch.setattr(background_scan, "_run_pass", lambda d: calls.append(d) or True)
    background_scan._tick()
    assert calls == [fixed_today]


def test_tick_noop_when_already_succeeded(monkeypatch):
    fixed_today = date(2026, 7, 20)  # Monday
    monkeypatch.setattr(background_scan, "date",
                        type("D", (), {"today": staticmethod(lambda: fixed_today)}))
    monkeypatch.setattr(background_scan, "_pass_succeeded_for", lambda d: True)
    calls = []
    monkeypatch.setattr(background_scan, "_run_pass", lambda d: calls.append(d) or True)
    background_scan._tick()
    assert calls == []
