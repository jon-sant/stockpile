"""Tests for display/decay_curves.py — pure-data functions only, no
Streamlit/Altair rendering (matches this repo's existing convention of
unit-testing the pure function rather than the render call)."""

from datetime import date

import pandas as pd
import pytest

from options_scanner.backtest.engine import BacktestResult, BacktestTrade
from options_scanner.display.decay_curves import (
    build_decay_curve_data,
    regime_bucket,
)


def test_regime_bucket_quartile_assignment():
    # 8 evenly-spaced values -> clean quartile split.
    sigmas = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    buckets = regime_bucket(sigmas)
    assert list(buckets[:2]) == ["Low IV", "Low IV"]
    assert list(buckets[-2:]) == ["High IV", "High IV"]
    assert set(buckets) == {"Low IV", "Below-avg", "Above-avg", "High IV"}


def test_regime_bucket_falls_back_to_all_with_too_few_distinct_values():
    sigmas = pd.Series([0.30, 0.30, 0.30])
    buckets = regime_bucket(sigmas)
    assert list(buckets) == ["All", "All", "All"]


def _trade(entry_sigma, time_values):
    idx = pd.bdate_range("2026-01-01", periods=len(time_values))
    daily = pd.DataFrame({
        "total_value": [v * 100 for v in time_values],
        "intrinsic_per_share": [0.0] * len(time_values),
        "time_value_per_share": time_values,
    }, index=idx)
    return BacktestTrade(
        open_date=date(2026, 1, 1), close_date=date(2026, 1, 1 + len(time_values)),
        strike=100.0, entry_sigma=entry_sigma, premium_per_share=time_values[0],
        terminal_spot=100.0, pnl=0.0, ann_pct_realized=0.0, daily_value=daily,
    )


def test_build_decay_curve_data_normalizes_to_pct_of_entry():
    trade = _trade(0.30, [2.0, 1.5, 1.0, 0.5])
    result = BacktestResult(trades=[trade], win_rate=1.0, avg_ann_pct=10.0,
                            max_drawdown=0.0, sharpe=1.0, sortino=1.0)
    data = build_decay_curve_data(result)
    assert len(data) == 4
    assert list(data["day_offset"]) == [0, 1, 2, 3]
    assert data.iloc[0]["pct_remaining"] == pytest.approx(100.0)
    assert data.iloc[1]["pct_remaining"] == pytest.approx(75.0)
    assert data.iloc[2]["pct_remaining"] == pytest.approx(50.0)
    assert data.iloc[3]["pct_remaining"] == pytest.approx(25.0)


def test_build_decay_curve_data_skips_zero_entry_time_value():
    trade = _trade(0.30, [0.0, 0.0, 0.0])
    result = BacktestResult(trades=[trade], win_rate=1.0, avg_ann_pct=10.0,
                            max_drawdown=0.0, sharpe=1.0, sortino=1.0)
    data = build_decay_curve_data(result)
    assert data.empty


def test_build_decay_curve_data_empty_trades_returns_empty_df():
    result = BacktestResult(trades=[], win_rate=float("nan"),
                            avg_ann_pct=float("nan"), max_drawdown=0.0,
                            sharpe=float("nan"), sortino=float("nan"))
    data = build_decay_curve_data(result)
    assert data.empty
    assert list(data.columns) == ["trade_idx", "day_offset", "regime", "pct_remaining"]


def test_build_decay_curve_data_multiple_trades_tagged_by_regime():
    low_trade = _trade(0.10, [1.0, 0.8])
    high_trade = _trade(0.50, [1.0, 0.9])
    result = BacktestResult(trades=[low_trade, high_trade], win_rate=1.0,
                            avg_ann_pct=10.0, max_drawdown=0.0, sharpe=1.0,
                            sortino=1.0)
    data = build_decay_curve_data(result)
    # Only 2 distinct sigma values -> regime_bucket falls back to "All".
    assert set(data["regime"]) == {"All"}
    assert set(data["trade_idx"]) == {0, 1}
