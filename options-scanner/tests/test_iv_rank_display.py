"""Tests for the IV Rank column + tooltip wiring in the results grids."""

import pandas as pd
import streamlit as st

from options_scanner.display.leaderboard import _iv_rank_help as _leaderboard_help
from options_scanner.display.scan_results import _iv_rank_help, show_df


def _base_chain(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-08-21"] * n,
        "dte": [34] * n,
        "bid": [2.0] * n,
        "ask": [2.2] * n,
        "mid": [2.1] * n,
        "last": [2.1] * n,
        "iv": [0.30] * n,
        "iv_excess": [0.01] * n,
        "delta": [0.3] * n,
        "ann_yield_pct": [10.0] * n,
        "open_interest": [100] * n,
        "volume": [10] * n,
        "earnings_count": [0] * n,
        "signal_score": [0.01] * n,
        "spot": [150.0] * n,
    })


def _run_show_df_and_capture(sub, **kwargs):
    captured = {}
    orig_dataframe = st.dataframe

    def _fake(styled, **kw):
        captured["data"] = styled.data if hasattr(styled, "data") else styled
        captured["column_config"] = kw.get("column_config", {})

    st.dataframe = _fake
    try:
        show_df(sub, **kwargs)
    finally:
        st.dataframe = orig_dataframe
    return captured


def test_show_df_includes_iv_rank_column_when_present():
    sub = _base_chain(3)
    sub["iv_rank"] = [72.4, 72.4, 72.4]
    sub["iv_rank_date_from"] = ["2026-01-01"] * 3
    sub["iv_rank_date_to"] = ["2026-08-19"] * 3
    result = _run_show_df_and_capture(sub)
    assert "IV Rank" in result["data"].columns
    assert list(result["data"]["IV Rank"]) == [72.0, 72.0, 72.0]  # rounded


def test_show_df_iv_rank_blank_when_column_missing():
    sub = _base_chain(2)  # no iv_rank columns at all
    result = _run_show_df_and_capture(sub)
    assert "IV Rank" in result["data"].columns
    assert result["data"]["IV Rank"].isna().all()


def test_iv_rank_help_includes_exact_date_range():
    sub = _base_chain(2)
    sub["iv_rank_date_from"] = ["2026-01-01"] * 2
    sub["iv_rank_date_to"] = ["2026-08-19"] * 2
    help_text = _iv_rank_help(sub)
    assert "2026-01-01" in help_text
    assert "2026-08-19" in help_text


def test_iv_rank_help_notes_missing_history():
    sub = _base_chain(2)
    sub["iv_rank_date_from"] = [None, None]
    sub["iv_rank_date_to"] = [None, None]
    help_text = _iv_rank_help(sub)
    assert "2 distinct" in help_text


def test_iv_rank_help_generic_when_no_columns_at_all():
    sub = _base_chain(2)
    help_text = _iv_rank_help(sub)
    assert "0-100" in help_text
    assert "2026" not in help_text  # no fabricated date


def test_leaderboard_iv_rank_help_aggregates_across_tickers():
    board = pd.DataFrame({
        "iv_rank_date_from": ["2026-01-01", "2026-03-15", None],
        "iv_rank_date_to": ["2026-08-19", "2026-08-10", None],
    })
    help_text = _leaderboard_help(board)
    assert "2026-01-01" in help_text  # earliest across tickers
    assert "2026-08-19" in help_text  # latest across tickers
    assert "varies per ticker" in help_text


def test_leaderboard_iv_rank_help_generic_when_empty():
    board = pd.DataFrame({"iv_rank_date_from": [], "iv_rank_date_to": []})
    help_text = _leaderboard_help(board)
    assert "distinct scan days" in help_text
