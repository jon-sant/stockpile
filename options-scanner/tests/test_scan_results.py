"""Tests for the pure-formula helpers in display/scan_results.py."""

import numpy as np
import pandas as pd
import streamlit as st

from options_scanner.display.scan_results import _prem_pct_em, show_scan_results


def _percentile_df():
    return pd.DataFrame({
        "type": ["call"] * 4,
        "strike": [100.0, 105.0, 110.0, 115.0],
        "expiration": ["2026-08-21"] * 4,
        "dte": [34] * 4,
        "bid": [2.0, 1.5, 1.0, 0.5],
        "ask": [2.2, 1.7, 1.2, 0.7],
        "mid": [2.1, 1.6, 1.1, 0.6],
        "last": [2.1, 1.6, 1.1, 0.6],
        "iv": [0.3, 0.28, 0.26, 0.24],
        "iv_excess": [0.05, 0.01, -0.02, 0.08],
        "delta": [0.5, 0.4, 0.3, 0.2],
        "ann_yield_pct": [20.0, 8.0, 3.0, 25.0],
        "open_interest": [100] * 4,
        "volume": [10] * 4,
        "earnings_count": [0] * 4,
        "signal_score": [0.05, 0.01, -0.02, 0.08],
        "spot": [150.0] * 4,
        "iv_percentile": [90.0, 40.0, 10.0, 95.0],
        "ann_delta_percentile": [85.0, 30.0, 5.0, 92.0],
    })


def test_prem_pct_em_matches_formula():
    sub = pd.DataFrame({
        "mid": [2.0, 1.0],
        "spot": [100.0, 50.0],
        "iv": [0.30, 0.40],
        "dte": [30, 60],
    })
    result = _prem_pct_em(sub)
    expected_move_0 = 100.0 * 0.30 * np.sqrt(30 / 365.0)
    expected_move_1 = 50.0 * 0.40 * np.sqrt(60 / 365.0)
    assert np.isclose(result.iloc[0], round(2.0 / expected_move_0, 2))
    assert np.isclose(result.iloc[1], round(1.0 / expected_move_1, 2))


def test_prem_pct_em_dte_zero_no_divide_by_zero():
    sub = pd.DataFrame({
        "mid": [1.5],
        "spot": [100.0],
        "iv": [0.25],
        "dte": [0],
    })
    result = _prem_pct_em(sub)
    assert np.isfinite(result.iloc[0])
    # dte clamped to 1, so it matches the dte=1 formula exactly.
    expected_move = 100.0 * 0.25 * np.sqrt(1 / 365.0)
    assert np.isclose(result.iloc[0], round(1.5 / expected_move, 2))


def test_prem_pct_em_missing_columns_returns_nan():
    sub = pd.DataFrame({"mid": [1.0, 2.0]})
    result = _prem_pct_em(sub)
    assert np.isnan(result).all()


def _run_and_capture(df, **kwargs):
    captured = []
    orig_dataframe = st.dataframe
    st.dataframe = lambda styled, **kw: captured.append(
        styled.data if hasattr(styled, "data") else styled)
    try:
        show_scan_results(df, "call", False, None, 0, 10, 0, **kwargs)
    finally:
        st.dataframe = orig_dataframe
    return captured[-1]


def test_min_percentile_filter_excludes_below_floor():
    disp = _run_and_capture(_percentile_df(), min_percentile=50.0)
    # Rows with iv_percentile 90 and 95 pass; 40 and 10 are excluded.
    assert sorted(disp["Strike"].tolist()) == ["$100", "$115"]


def test_min_ann_delta_percentile_filter_excludes_below_floor():
    disp = _run_and_capture(_percentile_df(), min_ann_delta_percentile=50.0)
    assert sorted(disp["Strike"].tolist()) == ["$100", "$115"]


def test_percentile_filters_none_means_no_filtering():
    disp = _run_and_capture(_percentile_df())
    assert len(disp) == 4


def test_combined_percentile_filters():
    disp = _run_and_capture(
        _percentile_df(), min_percentile=50.0, min_ann_delta_percentile=50.0)
    assert sorted(disp["Strike"].tolist()) == ["$100", "$115"]
