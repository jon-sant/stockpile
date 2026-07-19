"""Tests for compute/capital_allocator.py — edge scoring and the greedy
capital-waterfill allocator. No MC/Streamlit dependency here:
`candidates_from_board` (which does run a real simulation) is exercised
separately in test_capital_allocator_candidates.py; these tests build
`AllocationCandidate`s directly so the allocator's own arithmetic is
tested in isolation.
"""

import pandas as pd
import pytest

from options_scanner.compute.capital_allocator import (
    AllocationCandidate,
    allocate_capital,
    edge_score,
)


def _cand(ticker, strike, expected_pnl, cvar_5pct, collateral=10000.0,
          opt_type="put", composite_score=None):
    return AllocationCandidate(
        ticker=ticker, strike=strike, expiration="2026-08-21",
        opt_type=opt_type, premium=1.0, collateral_per_contract=collateral,
        prob_profit=0.7, expected_pnl=expected_pnl, cvar_5pct=cvar_5pct,
        composite_score=composite_score,
    )


def test_edge_score_reward_over_tail_risk():
    c = _cand("AAA", 100, expected_pnl=200.0, cvar_5pct=-400.0)
    assert edge_score(c) == pytest.approx(200.0 / 400.0)


def test_edge_score_floors_at_zero_for_negative_pnl():
    c = _cand("AAA", 100, expected_pnl=-50.0, cvar_5pct=-400.0)
    assert edge_score(c) == 0.0


def test_edge_score_no_cvar_uses_unit_downside():
    c = _cand("AAA", 100, expected_pnl=150.0, cvar_5pct=0.0)
    assert edge_score(c) == pytest.approx(150.0)


def test_allocate_capital_total_within_budget():
    candidates = [
        _cand("AAA", 100, expected_pnl=300.0, cvar_5pct=-300.0, collateral=10000.0),
        _cand("BBB", 50, expected_pnl=100.0, cvar_5pct=-500.0, collateral=5000.0),
    ]
    result = allocate_capital(candidates, budget=12000.0, max_pct_per_position=1.0)
    assert result["collateral"].sum() <= 12000.0


def test_allocate_capital_respects_max_pct_per_position():
    candidates = [
        _cand("AAA", 100, expected_pnl=1000.0, cvar_5pct=-100.0, collateral=1000.0),
    ]
    result = allocate_capital(candidates, budget=10000.0, max_pct_per_position=0.25)
    assert len(result) == 1
    assert result.iloc[0]["collateral"] <= 0.25 * 10000.0


def test_allocate_capital_higher_edge_allocated_first():
    candidates = [
        _cand("LOW", 100, expected_pnl=50.0, cvar_5pct=-500.0, collateral=5000.0),
        _cand("HIGH", 100, expected_pnl=400.0, cvar_5pct=-400.0, collateral=5000.0),
    ]
    # Budget affords only one of the two (each needs 5000, budget=5000).
    result = allocate_capital(candidates, budget=5000.0, max_pct_per_position=1.0)
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "HIGH"


def test_allocate_capital_zero_edge_gets_nothing():
    candidates = [
        _cand("ZERO", 100, expected_pnl=-10.0, cvar_5pct=-100.0, collateral=1000.0),
        _cand("POS", 100, expected_pnl=50.0, cvar_5pct=-100.0, collateral=1000.0),
    ]
    result = allocate_capital(candidates, budget=10000.0, max_pct_per_position=1.0)
    assert "ZERO" not in result["ticker"].tolist()
    assert "POS" in result["ticker"].tolist()


def test_allocate_capital_empty_candidates_returns_empty_df():
    result = allocate_capital([], budget=10000.0)
    assert result.empty
    assert list(result.columns) == [
        "ticker", "strike", "expiration", "opt_type", "contracts",
        "collateral", "edge_score", "expected_pnl", "prob_profit"]


def test_allocate_capital_zero_budget_returns_empty_df():
    candidates = [_cand("AAA", 100, expected_pnl=300.0, cvar_5pct=-300.0)]
    result = allocate_capital(candidates, budget=0.0)
    assert result.empty


def test_allocate_capital_unaffordable_collateral_skipped():
    candidates = [_cand("AAA", 100, expected_pnl=300.0, cvar_5pct=-300.0,
                        collateral=1_000_000.0)]
    result = allocate_capital(candidates, budget=1000.0, max_pct_per_position=1.0)
    assert result.empty


def test_allocate_capital_leftover_budget_reoffered_to_next():
    # max_pct_per_position=0.5 caps each pick's round to 5000, so the
    # top candidate can only take 1 contract per round even though the
    # full 10000 budget could otherwise buy 2 — the leftover 5000 must
    # get re-offered to the next-best candidate rather than sitting idle.
    candidates = [
        _cand("A", 100, expected_pnl=400.0, cvar_5pct=-400.0, collateral=5000.0),
        _cand("B", 100, expected_pnl=200.0, cvar_5pct=-400.0, collateral=5000.0),
    ]
    result = allocate_capital(candidates, budget=10000.0, max_pct_per_position=0.5)
    assert len(result) == 2
    assert set(result["ticker"]) == {"A", "B"}


def test_allocate_capital_corr_penalty_can_change_selection():
    # AAA and BBB are highly correlated; CCC is not. Budget affords two
    # picks. Without corr, AAA (best) and BBB (2nd best) win. With corr,
    # BBB's edge gets penalized enough by AAA's pick that CCC (3rd best,
    # uncorrelated) wins the second slot instead.
    candidates = [
        _cand("AAA", 100, expected_pnl=400.0, cvar_5pct=-400.0, collateral=5000.0),
        _cand("BBB", 100, expected_pnl=390.0, cvar_5pct=-400.0, collateral=5000.0),
        _cand("CCC", 100, expected_pnl=200.0, cvar_5pct=-400.0, collateral=5000.0),
    ]
    corr = pd.DataFrame(
        {"AAA": [1.0, 1.0, 0.0], "BBB": [1.0, 1.0, 0.0], "CCC": [0.0, 0.0, 1.0]},
        index=["AAA", "BBB", "CCC"],
    )

    no_corr = allocate_capital(candidates, budget=10000.0,
                               max_pct_per_position=0.5)
    assert set(no_corr["ticker"]) == {"AAA", "BBB"}

    with_corr = allocate_capital(candidates, budget=10000.0,
                                 max_pct_per_position=0.5, corr=corr)
    assert set(with_corr["ticker"]) == {"AAA", "CCC"}
