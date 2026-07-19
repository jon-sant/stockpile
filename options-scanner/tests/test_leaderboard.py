"""Tests for the pure-formula helpers in display/leaderboard.py."""

import numpy as np
import pandas as pd

from options_scanner.display.leaderboard import _prem_pct_em, build_leaderboard


def test_prem_pct_em_matches_formula():
    board = pd.DataFrame({
        "mid": [2.0, 1.0],
        "spot": [100.0, 50.0],
        "iv": [0.30, 0.40],
        "dte": [30, 60],
    })
    result = _prem_pct_em(board)
    expected_move_0 = 100.0 * 0.30 * np.sqrt(30 / 365.0)
    expected_move_1 = 50.0 * 0.40 * np.sqrt(60 / 365.0)
    assert np.isclose(result.iloc[0], round(2.0 / expected_move_0, 2))
    assert np.isclose(result.iloc[1], round(1.0 / expected_move_1, 2))


def test_prem_pct_em_dte_zero_no_divide_by_zero():
    board = pd.DataFrame({
        "mid": [1.5],
        "spot": [100.0],
        "iv": [0.25],
        "dte": [0],
    })
    result = _prem_pct_em(board)
    assert np.isfinite(result.iloc[0])
    expected_move = 100.0 * 0.25 * np.sqrt(1 / 365.0)
    assert np.isclose(result.iloc[0], round(1.5 / expected_move, 2))


def test_prem_pct_em_missing_columns_returns_nan():
    board = pd.DataFrame({"mid": [1.0, 2.0]})
    result = _prem_pct_em(board)
    assert np.isnan(result).all()


def _row(strike, iv_pct, ann_delta_pct, oi=500, volume=100):
    return {
        "type": "put", "strike": strike, "expiration": "2026-08-21",
        "dte": 45, "bid": 1.0, "ask": 1.2, "mid": 1.1, "iv": 0.5,
        "iv_excess": 0.02, "delta": -0.40, "ann_yield_pct": 25.0,
        "open_interest": oi, "volume": volume, "earnings_count": 0,
        "last": 1.1, "spot": 150.0,
        "iv_percentile": iv_pct, "ann_delta_percentile": ann_delta_pct,
    }


def _percentile_results():
    return [
        {"error": None, "position": {"ticker": "AAA"},
         "df": pd.DataFrame([_row(140, 90.0, 85.0), _row(130, 20.0, 15.0)])},
        {"error": None, "position": {"ticker": "BBB"},
         "df": pd.DataFrame([_row(90, 40.0, 30.0)])},
    ]


def test_min_percentile_excludes_below_floor():
    board = build_leaderboard(_percentile_results(), "put", min_oi=25,
                              top_n=5, min_vol=10, min_percentile=50.0)
    assert sorted(board["strike"].tolist()) == [140.0]


def test_min_ann_delta_percentile_excludes_below_floor():
    board = build_leaderboard(_percentile_results(), "put", min_oi=25,
                              top_n=5, min_vol=10,
                              min_ann_delta_percentile=50.0)
    assert sorted(board["strike"].tolist()) == [140.0]


def test_percentile_filters_none_means_no_filtering():
    board = build_leaderboard(_percentile_results(), "put", min_oi=25,
                              top_n=5, min_vol=10)
    assert len(board) == 3
