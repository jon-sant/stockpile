"""Tests for options_scanner.market_view — the (Direction x Option Type)
stance table and its Monte Carlo drift mapping.
"""
from __future__ import annotations

from options_scanner import market_view


def test_stance_for_all_six_combinations():
    assert market_view.stance_for(False, "Calls") == "Bearish / neutral-down"
    assert market_view.stance_for(False, "Puts") == "Bullish / neutral-up"
    assert market_view.stance_for(False, "Both") == "Range-bound (short volatility)"
    assert market_view.stance_for(True, "Calls") == "Bullish"
    assert market_view.stance_for(True, "Puts") == "Bearish"
    assert market_view.stance_for(True, "Both") == "Volatility expansion (long vol)"


def test_stance_for_unknown_opt_type_is_none():
    assert market_view.stance_for(True, "Straddle") is None


def test_drift_for_directional_stances():
    assert market_view.drift_for_stance("Bullish") == market_view.DEFAULT_DRIFT_MAGNITUDE
    assert market_view.drift_for_stance("Bullish / neutral-up") == market_view.DEFAULT_DRIFT_MAGNITUDE
    assert market_view.drift_for_stance("Bearish") == -market_view.DEFAULT_DRIFT_MAGNITUDE
    assert market_view.drift_for_stance("Bearish / neutral-down") == -market_view.DEFAULT_DRIFT_MAGNITUDE


def test_drift_for_non_directional_stances_is_zero():
    assert market_view.drift_for_stance("Range-bound (short volatility)") == 0.0
    assert market_view.drift_for_stance("Volatility expansion (long vol)") == 0.0


def test_drift_for_none_or_unknown_is_zero():
    assert market_view.drift_for_stance(None) == 0.0
    assert market_view.drift_for_stance("") == 0.0
    assert market_view.drift_for_stance("nonsense") == 0.0


def test_drift_magnitude_is_configurable():
    assert market_view.drift_for_stance("Bullish", magnitude=0.25) == 0.25
    assert market_view.drift_for_stance("Bearish", magnitude=0.25) == -0.25


def test_outlook_table_round_trip_stance_to_drift():
    # Every stance in the table maps to a finite, correctly-signed drift.
    for (buy, opt_type), cfg in market_view.OUTLOOK_TABLE.items():
        drift = market_view.drift_for_stance(cfg["stance"])
        if cfg["tone"] == "pos":
            assert drift > 0
        elif cfg["tone"] == "neg":
            assert drift < 0
        else:
            assert drift == 0.0
