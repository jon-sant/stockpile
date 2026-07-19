"""Tests for stocks_shared.black_scholes.strike_for_delta — the bisection
inversion of bs_delta used by the backtest engine to find "the 0.30-delta
strike" for a rolling short-option strategy."""

import pytest

from stocks_shared.black_scholes import bs_delta, strike_for_delta


def test_strike_for_delta_call_round_trips():
    S, T, r, sigma = 100.0, 30 / 365.0, 0.045, 0.30
    target = 0.30
    K = strike_for_delta(S, T, r, sigma, "call", target)
    recovered = bs_delta(S, K, T, r, sigma, "call")
    assert recovered == pytest.approx(target, abs=1e-3)
    assert K > S  # 0.30-delta call is OTM (above spot)


def test_strike_for_delta_put_round_trips():
    S, T, r, sigma = 100.0, 30 / 365.0, 0.045, 0.30
    target = -0.30
    K = strike_for_delta(S, T, r, sigma, "put", target)
    recovered = bs_delta(S, K, T, r, sigma, "put")
    assert recovered == pytest.approx(target, abs=1e-3)
    assert K < S  # 0.30-delta put is OTM (below spot)


def test_strike_for_delta_case_insensitive():
    S, T, r, sigma = 100.0, 30 / 365.0, 0.045, 0.30
    k_lower = strike_for_delta(S, T, r, sigma, "call", 0.30)
    k_upper = strike_for_delta(S, T, r, sigma, "CALL", 0.30)
    assert k_lower == pytest.approx(k_upper)


def test_strike_for_delta_clamps_out_of_range_target():
    S, T, r, sigma = 100.0, 30 / 365.0, 0.045, 0.30
    # A target delta beyond what any call can reach (max delta is 1.0;
    # min is 0.0) clamps to the nearest search bound rather than raising
    # or looping forever.
    K_hi = strike_for_delta(S, T, r, sigma, "call", target_delta=1.5)
    K_lo = strike_for_delta(S, T, r, sigma, "call", target_delta=-5.0)
    assert K_hi == pytest.approx(S * 0.3)
    assert K_lo == pytest.approx(S * 3.0)


def test_strike_for_delta_deeper_delta_gives_further_otm_strike():
    S, T, r, sigma = 100.0, 30 / 365.0, 0.045, 0.30
    k_30 = strike_for_delta(S, T, r, sigma, "call", 0.30)
    k_10 = strike_for_delta(S, T, r, sigma, "call", 0.10)
    assert k_10 > k_30  # lower delta = further OTM = higher strike (call)
