"""Tests for display/rank_filter.py — the shared filter/sort/top-N
selection logic extracted from show_scan_results() (Enhancement 1: this
is also what the blocking-scan priority step calls, so it must match
what actually gets displayed, exactly).
"""
from __future__ import annotations

import pandas as pd

from options_scanner.display.rank_filter import filter_sort_top_n


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "type":          ["call", "call", "call", "put", "put"],
        "strike":        [100.0, 105.0, 110.0, 90.0, 95.0],
        "open_interest": [500, 50, 300, 10, 200],
        "volume":        [50, 5, 30, 1, 20],
        "iv_excess":     [0.03, 0.05, 0.01, 0.02, 0.04],
        "signal_score":  [0.03, 0.05, 0.01, 0.02, 0.04],
        "ann_yield_pct": [12.0, 20.0, 5.0, 8.0, 15.0],
        "iv_percentile": [60.0, 90.0, 30.0, 40.0, 80.0],
        "ann_delta_percentile": [55.0, 85.0, 25.0, 35.0, 75.0],
    })


def test_filters_to_requested_type():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0, top_n=10)
    assert set(sub["type"]) == {"call"}
    assert len(sub) == 3


def test_sell_mode_sorts_descending_by_signal_score():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0, top_n=10)
    assert list(sub["strike"]) == [105.0, 100.0, 110.0]  # 0.05, 0.03, 0.01


def test_buy_mode_sorts_ascending_by_signal_score():
    sub = filter_sort_top_n(_df(), "call", buy=True, min_oi=0, min_vol=0, top_n=10)
    assert list(sub["strike"]) == [110.0, 100.0, 105.0]  # 0.01, 0.03, 0.05


def test_falls_back_to_iv_excess_when_no_signal_score():
    df = _df().drop(columns=["signal_score"])
    sub = filter_sort_top_n(df, "call", buy=False, min_oi=0, min_vol=0, top_n=10)
    assert list(sub["strike"]) == [105.0, 100.0, 110.0]


def test_min_oi_and_min_vol_floors():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=100, min_vol=0, top_n=10)
    assert set(sub["strike"]) == {100.0, 110.0}  # strike 105 has OI=50


def test_top_n_caps_result_count():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0, top_n=1)
    assert len(sub) == 1
    assert sub.iloc[0]["strike"] == 105.0  # richest first in sell mode


def test_min_ivpp_filter():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0,
                            top_n=10, min_ivpp=4.0)  # 4pp = 0.04 iv_excess
    assert set(sub["strike"]) == {105.0}  # only 0.05 clears 4pp


def test_min_ann_filter():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0,
                            top_n=10, min_ann=10.0)
    assert set(sub["strike"]) == {100.0, 105.0}


def test_min_percentile_filter():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0,
                            top_n=10, min_percentile=50.0)
    assert set(sub["strike"]) == {100.0, 105.0}


def test_min_ann_delta_percentile_filter():
    sub = filter_sort_top_n(_df(), "call", buy=False, min_oi=0, min_vol=0,
                            top_n=10, min_ann_delta_percentile=50.0)
    assert set(sub["strike"]) == {100.0, 105.0}


def test_missing_percentile_columns_dont_crash():
    df = _df().drop(columns=["iv_percentile", "ann_delta_percentile"])
    sub = filter_sort_top_n(df, "call", buy=False, min_oi=0, min_vol=0,
                            top_n=10, min_percentile=50.0,
                            min_ann_delta_percentile=50.0)
    # Both floors silently no-op when the column isn't present.
    assert len(sub) == 3


def test_does_not_mutate_input_df():
    df = _df()
    original_len = len(df)
    filter_sort_top_n(df, "call", buy=False, min_oi=100, min_vol=0, top_n=1)
    assert len(df) == original_len
