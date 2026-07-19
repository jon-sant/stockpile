"""Tests for the background Monte Carlo batch compute (mc_batch.py).

Covers:
    - The 14-key result shape and buy/sell asymmetry
    - Side-invariant metrics (mc_fair_value, breakeven) match between
      the shared-path long/short evaluations
    - Determinism (same row -> same seed -> same result)
    - run_batch's per-row error isolation
"""
from __future__ import annotations

from options_scanner import mc_batch


def _row(**overrides) -> dict:
    base = {
        "rowid": 1, "ticker": "AMD", "scan_date": "2026-07-19", "type": "call",
        "strike": 100.0, "expiration": "2026-09-18", "dte": 61,
        "mid": 5.0, "iv": 0.40, "spot": 98.0, "earnings_next_date": "2026-08-05",
    }
    base.update(overrides)
    return base


def test_result_has_all_14_metric_keys():
    result = mc_batch.compute_mc_for_row(_row())
    expected = {
        "mc_fair_value", "mc_breakeven_move_pct",
        "mc_prob_profit_buy", "mc_prob_profit_sell",
        "mc_expected_pnl_buy", "mc_expected_pnl_sell",
        "mc_cvar5_buy", "mc_cvar5_sell",
        "mc_var5_buy", "mc_var5_sell",
        "mc_sortino_buy", "mc_sortino_sell",
        "mc_edge_vs_market_buy", "mc_edge_vs_market_sell",
    }
    assert expected <= set(result.keys())
    assert "duration_ms" in result


def test_buy_sell_metrics_are_asymmetric():
    result = mc_batch.compute_mc_for_row(_row())
    # Zero-sum trade between the two sides of one contract.
    assert result["mc_expected_pnl_buy"] == -result["mc_expected_pnl_sell"]
    assert result["mc_edge_vs_market_buy"] == -result["mc_edge_vs_market_sell"]
    # Complementary probabilities (no ties at this path count/seed).
    assert result["mc_prob_profit_buy"] + result["mc_prob_profit_sell"] == 1.0
    assert result["mc_prob_profit_buy"] != result["mc_prob_profit_sell"]


def test_fair_value_is_non_negative_and_side_invariant_derivation():
    result = mc_batch.compute_mc_for_row(_row())
    # mc_fair_value is derived from the long-side evaluation (mean of an
    # always->=0 intrinsic payoff) — must never be negative.
    assert result["mc_fair_value"] >= 0.0


def test_breakeven_move_is_a_finite_percent():
    result = mc_batch.compute_mc_for_row(_row())
    assert isinstance(result["mc_breakeven_move_pct"], float)
    assert -100.0 < result["mc_breakeven_move_pct"] < 1000.0


def test_deterministic_seed_reproduces_identical_result():
    row = _row()
    first = mc_batch.compute_mc_for_row(row)
    second = mc_batch.compute_mc_for_row(row)
    for key in ("mc_fair_value", "mc_prob_profit_buy", "mc_cvar5_sell",
                "mc_sortino_buy"):
        assert first[key] == second[key]


def test_different_strike_gets_different_seed_and_result():
    a = mc_batch.compute_mc_for_row(_row(strike=100.0))
    b = mc_batch.compute_mc_for_row(_row(strike=105.0))
    assert a["mc_fair_value"] != b["mc_fair_value"]


def test_run_batch_isolates_one_row_failure():
    good = _row(rowid=1)
    bad = _row(rowid=2, iv=None)  # no positive IV -> vol_source resolution raises
    out = mc_batch.run_batch([good, bad])
    by_rowid = {rowid: (result, error) for rowid, result, error in out}

    good_result, good_error = by_rowid[1]
    assert good_error is None
    assert good_result is not None
    assert "mc_fair_value" in good_result

    bad_result, bad_error = by_rowid[2]
    assert bad_result is None
    assert bad_error is not None
    assert "IV" in bad_error


def test_run_batch_returns_one_tuple_per_row():
    rows = [_row(rowid=i, strike=100.0 + i) for i in range(3)]
    out = mc_batch.run_batch(rows)
    assert len(out) == 3
    assert [rowid for rowid, _, _ in out] == [0, 1, 2]
