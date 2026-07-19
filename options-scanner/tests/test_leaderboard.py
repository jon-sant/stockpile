"""Tests for the pure-formula helpers in display/leaderboard.py."""

import numpy as np
import pandas as pd

from options_scanner.display.leaderboard import _prem_pct_em


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
