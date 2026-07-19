"""Tests for compute/correlation.py — pairwise return correlation and
the diversification penalty the capital allocator applies.
"""

import numpy as np
import pandas as pd
import pytest

from options_scanner.compute import correlation


def _price_series(log_returns, start=100.0):
    """Build a Close-price series whose day-over-day LOG returns exactly
    equal `log_returns` (avoids the (1+r) vs. log(1+r) precision drift
    that a naive percent-return construction would introduce)."""
    prices = [start]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    idx = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_pairwise_corr_perfectly_correlated(monkeypatch):
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 40).tolist()
    series = {"AAA": _price_series(returns), "BBB": _price_series(returns)}

    monkeypatch.setattr(correlation, "fetch_history",
                        lambda t, start=None, end=None: series[t])
    corr = correlation.pairwise_log_return_corr(["AAA", "BBB"])
    assert corr.loc["AAA", "BBB"] == pytest.approx(1.0, abs=1e-6)
    assert corr.loc["BBB", "AAA"] == pytest.approx(1.0, abs=1e-6)
    assert corr.loc["AAA", "AAA"] == 1.0


def test_pairwise_corr_anti_correlated(monkeypatch):
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 40).tolist()
    inverse = [-r for r in returns]
    series = {"AAA": _price_series(returns), "BBB": _price_series(inverse)}

    monkeypatch.setattr(correlation, "fetch_history",
                        lambda t, start=None, end=None: series[t])
    corr = correlation.pairwise_log_return_corr(["AAA", "BBB"])
    assert corr.loc["AAA", "BBB"] == pytest.approx(-1.0, abs=1e-6)


def test_pairwise_corr_nan_on_thin_history(monkeypatch):
    rng = np.random.default_rng(2)
    short_returns = rng.normal(0, 0.01, 5).tolist()  # below _MIN_OVERLAP_DAYS
    long_returns = rng.normal(0, 0.01, 40).tolist()
    series = {"AAA": _price_series(short_returns), "BBB": _price_series(long_returns)}

    monkeypatch.setattr(correlation, "fetch_history",
                        lambda t, start=None, end=None: series[t])
    corr = correlation.pairwise_log_return_corr(["AAA", "BBB"])
    assert pd.isna(corr.loc["AAA", "BBB"])


def test_pairwise_corr_fetch_failure_is_nan_not_zero(monkeypatch):
    def _fetch(t, start=None, end=None):
        if t == "BAD":
            raise ConnectionError("no network")
        return _price_series(np.random.default_rng(3).normal(0, 0.01, 40).tolist())

    monkeypatch.setattr(correlation, "fetch_history", _fetch)
    corr = correlation.pairwise_log_return_corr(["AAA", "BAD"])
    assert pd.isna(corr.loc["AAA", "BAD"])


def test_diversification_penalty_below_threshold_is_one():
    corr = pd.DataFrame({"AAA": [1.0, 0.3], "BBB": [0.3, 1.0]},
                        index=["AAA", "BBB"])
    penalty = correlation.diversification_penalty("AAA", ["BBB"], corr, threshold=0.7)
    assert penalty == 1.0


def test_diversification_penalty_at_threshold_is_one():
    corr = pd.DataFrame({"AAA": [1.0, 0.7], "BBB": [0.7, 1.0]},
                        index=["AAA", "BBB"])
    penalty = correlation.diversification_penalty("AAA", ["BBB"], corr, threshold=0.7)
    assert penalty == 1.0


def test_diversification_penalty_above_threshold_between_zero_and_one():
    corr = pd.DataFrame({"AAA": [1.0, 0.9], "BBB": [0.9, 1.0]},
                        index=["AAA", "BBB"])
    penalty = correlation.diversification_penalty("AAA", ["BBB"], corr, threshold=0.7)
    assert 0.0 < penalty < 1.0


def test_diversification_penalty_monotonically_decreasing_with_correlation():
    def _penalty_for(c):
        corr = pd.DataFrame({"AAA": [1.0, c], "BBB": [c, 1.0]},
                            index=["AAA", "BBB"])
        return correlation.diversification_penalty("AAA", ["BBB"], corr, threshold=0.7)

    p_low = _penalty_for(0.75)
    p_mid = _penalty_for(0.90)
    p_high = _penalty_for(1.0)
    assert p_low > p_mid > p_high


def test_diversification_penalty_nan_correlation_treated_as_no_penalty():
    corr = pd.DataFrame({"AAA": [1.0, np.nan], "BBB": [np.nan, 1.0]},
                        index=["AAA", "BBB"])
    penalty = correlation.diversification_penalty("AAA", ["BBB"], corr, threshold=0.7)
    assert penalty == 1.0


def test_diversification_penalty_compounds_across_multiple_correlated_picks():
    corr = pd.DataFrame(
        {"AAA": [1.0, 0.9, 0.9], "BBB": [0.9, 1.0, 0.1], "CCC": [0.9, 0.1, 1.0]},
        index=["AAA", "BBB", "CCC"],
    )
    single = correlation.diversification_penalty("AAA", ["BBB"], corr, threshold=0.7)
    double = correlation.diversification_penalty("AAA", ["BBB", "CCC"], corr, threshold=0.7)
    assert double < single


def test_diversification_penalty_missing_ticker_no_penalty():
    corr = pd.DataFrame({"AAA": [1.0]}, index=["AAA"])
    penalty = correlation.diversification_penalty("AAA", ["UNKNOWN"], corr)
    assert penalty == 1.0
