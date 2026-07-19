"""Tests for compute/gex_summary.py — per-strike GEX, the gamma-flip
strike, the per-ticker summary dict, and the per-contract alignment score.
"""

import numpy as np
import pandas as pd

from options_scanner.compute.gex_summary import (
    compute_gex_summary,
    gamma_flip_strike,
    gex_alignment,
    per_strike_gex,
)


def _chain(strikes, gammas, oi, spot=100.0):
    n = len(strikes)
    types = ["call" if s >= spot else "put" for s in strikes]
    return pd.DataFrame({
        "type": types,
        "strike": strikes,
        "gamma": gammas,
        "open_interest": oi,
    })


def test_per_strike_gex_sign_by_type():
    df = _chain([105.0, 95.0], [0.05, 0.05], [100, 100], spot=100.0)
    result = per_strike_gex(df, spot=100.0)
    call_row = result[result["strike"] == 105.0].iloc[0]
    put_row = result[result["strike"] == 95.0].iloc[0]
    assert call_row["gex"] > 0
    assert put_row["gex"] < 0


def test_per_strike_gex_empty_without_gamma_column():
    df = pd.DataFrame({"type": ["call"], "strike": [100.0], "open_interest": [10]})
    result = per_strike_gex(df, spot=100.0)
    assert result.empty


def test_gamma_flip_strike_finds_sign_change():
    per_strike = pd.DataFrame({
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "gex": [-100.0, -50.0, 20.0, 60.0, 80.0],
        "open_interest": [10] * 5,
    })
    # cumulative net GEX: -100, -150, -130, -70, 10 — crosses zero between
    # the 105 and 110 strikes (the only sign change in the sequence).
    flip = gamma_flip_strike(per_strike, spot=100.0)
    assert 105.0 < flip < 110.0


def test_gex_alignment_high_near_wall_low_near_amp():
    # Strong positive GEX (wall) at 110, strong negative (amp) at 90.
    # A big OI cluster at each strike, with strike 105/95 as query rows
    # closer to the wall/amp respectively than to each other.
    df = _chain(
        strikes=[90.0, 95.0, 105.0, 110.0],
        gammas=[0.08, 0.02, 0.02, 0.08],
        oi=[500, 50, 50, 500],
        spot=100.0,
    )
    scores = gex_alignment(df, spot=100.0)
    assert len(scores) == 4
    idx = {s: i for i, s in enumerate(df["strike"])}
    # Row at strike 110 (at the wall) scores higher than row at strike 95
    # (near the amp zone).
    assert scores[idx[110.0]] > scores[idx[95.0]]
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_gex_alignment_all_nan_when_degenerate():
    df = pd.DataFrame({"type": ["call"], "strike": [100.0],
                       "open_interest": [10]})  # no gamma column
    scores = gex_alignment(df, spot=100.0)
    assert len(scores) == 1
    assert np.isnan(scores).all()


def test_gex_alignment_all_nan_when_gex_is_zero_everywhere():
    df = _chain([100.0, 105.0], [0.0, 0.0], [10, 10], spot=100.0)
    scores = gex_alignment(df, spot=100.0)
    assert np.isnan(scores).all()


def test_compute_gex_summary_regime_and_walls():
    df = _chain(
        strikes=[90.0, 95.0, 105.0, 110.0],
        gammas=[0.02, 0.02, 0.08, 0.08],
        oi=[100, 100, 500, 500],
        spot=100.0,
    )
    summary = compute_gex_summary(df, spot=100.0)
    assert summary is not None
    assert summary["regime"] in ("Pinning", "Amplifying")
    assert summary["top_wall"] is not None
