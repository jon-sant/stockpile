"""Integration test for compute/capital_allocator.candidates_from_board —
exercises the real (fast, no-network) Monte Carlo path, not mocked, since
the whole point is confirming the MC wiring produces usable candidates.
"""

from datetime import date, timedelta

import pandas as pd

from options_scanner.compute.capital_allocator import candidates_from_board


def _board_row(ticker, opt_type, strike, spot, dte_days=30):
    exp = (date.today() + timedelta(days=dte_days)).isoformat()
    return {
        "ticker": ticker, "type": opt_type, "strike": strike,
        "expiration": exp, "mid": 1.5, "iv": 0.35, "spot": spot,
        "signal_score": 0.05,
    }


def test_candidates_from_board_builds_put_and_call_candidates():
    board = pd.DataFrame([
        _board_row("AAA", "put", 90.0, spot=100.0),
        _board_row("BBB", "call", 110.0, spot=100.0),
    ])
    candidates = candidates_from_board(board, n_paths=200)
    assert len(candidates) == 2

    put_cand = next(c for c in candidates if c.opt_type == "put")
    call_cand = next(c for c in candidates if c.opt_type == "call")

    # CSP collateral = strike * 100; covered-call collateral = spot * 100.
    assert put_cand.collateral_per_contract == 90.0 * 100.0
    assert call_cand.collateral_per_contract == 100.0 * 100.0

    for c in candidates:
        assert 0.0 <= c.prob_profit <= 1.0
        assert c.composite_score == 0.05


def test_candidates_from_board_skips_bad_rows_without_crashing():
    board = pd.DataFrame([
        _board_row("AAA", "put", 90.0, spot=100.0),
        {"ticker": "BAD", "type": "put", "strike": 100.0,
         "expiration": "not-a-date", "mid": 1.0, "iv": 0.3, "spot": 100.0},
    ])
    candidates = candidates_from_board(board, n_paths=200)
    assert len(candidates) == 1
    assert candidates[0].ticker == "AAA"


def test_candidates_from_board_empty_returns_empty_list():
    board = pd.DataFrame(columns=["ticker", "type", "strike", "expiration",
                                  "mid", "iv", "spot"])
    assert candidates_from_board(board) == []
