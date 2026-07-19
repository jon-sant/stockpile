"""Tests for backtest/engine.py — the fixed-delta rolling short-option
backtester. `fetch_history` is monkeypatched (no network in this
sandbox) with deterministic synthetic price paths so cycle count and
P&L sign are exactly predictable.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import options_scanner.backtest.engine as engine
from options_scanner.backtest.engine import BacktestConfig, run_backtest


def _price_series(start_price, daily_pct_change, n_days=120):
    idx = pd.bdate_range("2025-01-01", periods=n_days)
    prices = [start_price]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1.0 + daily_pct_change))
    return pd.DataFrame({"Close": prices}, index=idx)


def _patch_fetch_history(monkeypatch, df):
    monkeypatch.setattr(engine, "fetch_history",
                        lambda ticker, start=None, end=None: df)


def test_run_backtest_produces_multiple_cycles_on_flat_price(monkeypatch):
    df = _price_series(100.0, 0.0, n_days=150)
    _patch_fetch_history(monkeypatch, df)
    config = BacktestConfig(
        ticker="FLAT", opt_type="put", target_delta=0.30, dte_days=15,
        start_date=date(2025, 3, 1), end_date=date(2025, 5, 1),
    )
    result = run_backtest(config)
    assert len(result.trades) >= 2
    assert result.win_rate == result.win_rate  # not NaN


def test_run_backtest_declining_price_hurts_put_seller(monkeypatch):
    # Price steadily declines -> short OTM puts drift ITM -> the seller
    # pays back more intrinsic than the premium collected -> net losses.
    df = _price_series(100.0, -0.01, n_days=150)
    _patch_fetch_history(monkeypatch, df)
    config = BacktestConfig(
        ticker="DOWN", opt_type="put", target_delta=0.30, dte_days=15,
        start_date=date(2025, 3, 1), end_date=date(2025, 5, 1),
    )
    result = run_backtest(config)
    assert len(result.trades) >= 1
    assert result.avg_ann_pct < 0


def test_run_backtest_declining_price_helps_call_seller(monkeypatch):
    # Price declines -> short OTM calls stay OTM -> seller keeps most
    # of the premium -> net gains (opposite of the put case above).
    df = _price_series(100.0, -0.01, n_days=150)
    _patch_fetch_history(monkeypatch, df)
    config = BacktestConfig(
        ticker="DOWN", opt_type="call", target_delta=0.30, dte_days=15,
        start_date=date(2025, 3, 1), end_date=date(2025, 5, 1),
    )
    result = run_backtest(config)
    assert len(result.trades) >= 1
    assert result.avg_ann_pct > 0


def test_run_backtest_opt_type_casing_regression(monkeypatch):
    """Regression for the estimate_option_history casing bug: its
    intrinsic branch does an exact `== "Call"` check that bypasses
    bs_price's case-insensitivity, so the engine MUST capitalize
    opt_type at that call site. A rising price series selling calls
    should show LOSSES (calls go ITM, seller pays back intrinsic) — if
    the capitalize() were dropped, estimate_option_history would
    silently compute PUT intrinsic instead (near-zero for a rising
    underlying), which would make daily_value's time_value_per_share
    wrongly track ~bs_price instead of bs_price-intrinsic, and — more
    directly checkable here — every daily_value row's intrinsic_per_share
    would be the wrong (put) formula rather than the call formula.
    """
    df = _price_series(100.0, 0.01, n_days=150)  # steadily rising
    _patch_fetch_history(monkeypatch, df)
    config = BacktestConfig(
        ticker="UP", opt_type="call", target_delta=0.30, dte_days=15,
        start_date=date(2025, 3, 1), end_date=date(2025, 5, 1),
    )
    result = run_backtest(config)
    assert len(result.trades) >= 1

    for trade in result.trades:
        daily = trade.daily_value
        last_row = daily.iloc[-1]
        last_spot = df.loc[daily.index[-1], "Close"]
        expected_call_intrinsic = max(0.0, last_spot - trade.strike)
        expected_put_intrinsic = max(0.0, trade.strike - last_spot)
        # On a rising series ending well above most reasonable strikes,
        # the call-intrinsic and put-intrinsic formulas diverge sharply
        # whenever the position has gone meaningfully ITM — assert the
        # recorded value matches the CALL formula, not the PUT one.
        if expected_call_intrinsic > 0.01:
            assert last_row["intrinsic_per_share"] == pytest.approx(
                expected_call_intrinsic, abs=0.05)
            assert abs(last_row["intrinsic_per_share"]
                      - expected_put_intrinsic) > 0.01


def test_run_backtest_empty_price_history_returns_empty_result(monkeypatch):
    _patch_fetch_history(monkeypatch, pd.DataFrame(columns=["Close"]))
    config = BacktestConfig(
        ticker="NOPE", opt_type="put", target_delta=0.30, dte_days=15,
        start_date=date(2025, 3, 1), end_date=date(2025, 5, 1),
    )
    result = run_backtest(config)
    assert result.trades == []
    assert np.isnan(result.win_rate)


def test_run_backtest_window_too_short_for_one_cycle_returns_empty(monkeypatch):
    # Only 5 trading days of history fetched — nowhere near enough to
    # close a 15-day-DTE cycle, regardless of start/end_date.
    df = _price_series(100.0, 0.0, n_days=5)
    _patch_fetch_history(monkeypatch, df)
    config = BacktestConfig(
        ticker="FLAT", opt_type="put", target_delta=0.30, dte_days=15,
        start_date=date(2025, 1, 1), end_date=date(2025, 1, 3),
    )
    result = run_backtest(config)
    assert result.trades == []
