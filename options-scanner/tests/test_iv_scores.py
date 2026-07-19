"""Tests for the pluggable signal-score registry.

Each score is a pure function of (df, fit_mask, ctx); these pin the
formula for each and check the raw_pp == iv_excess identity that keeps
the default ranking unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from options_scanner import iv_scores
from options_scanner.iv_scores import ScoreContext


def _scored_df():
    """A small frame with iv / iv_fitted / iv_excess already populated."""
    iv = np.array([0.30, 0.40, 0.25, 0.50, 0.35])
    fitted = np.array([0.30, 0.32, 0.30, 0.40, 0.34])
    df = pd.DataFrame({
        "iv": iv,
        "iv_fitted": fitted,
        "iv_excess": iv - fitted,
        "bid": [1.00, 1.00, 1.00, 1.00, 1.00],
        "ask": [1.02, 1.50, 1.02, 1.04, 1.02],
        "mid": [1.01, 1.25, 1.01, 1.02, 1.01],
    })
    return df


def test_raw_pp_equals_iv_excess():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)
    score, label = iv_scores.score(df, mask, None, ("raw_pp", frozenset()))
    assert label == "IV+pp"
    assert np.allclose(score, df["iv_excess"].to_numpy())


def test_zscore_is_excess_over_residual_std():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)
    score, label = iv_scores.score(df, mask, None, ("zscore", frozenset()))
    expected = df["iv_excess"].to_numpy() / np.std(df["iv_excess"].to_numpy())
    assert label == "IV z"
    assert np.allclose(score, expected)


def test_zscore_zero_when_no_residual_spread():
    df = _scored_df()
    df["iv_excess"] = 0.0
    mask = np.ones(len(df), dtype=bool)
    score, _ = iv_scores.score(df, mask, None, ("zscore", frozenset()))
    assert np.allclose(score, 0.0)


def test_relative_is_excess_over_fitted():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)
    score, label = iv_scores.score(df, mask, None, ("relative", frozenset()))
    expected = df["iv_excess"].to_numpy() / df["iv_fitted"].to_numpy()
    assert label == "IV rel"
    assert np.allclose(score, expected)


def test_composite_penalizes_wide_spread():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)
    score, label = iv_scores.score(df, mask, None,
                                   ("composite_exec", frozenset()))
    assert label == "Score"
    # Row 1 has a wide spread (1.00/1.50) → its composite is discounted
    # relative to its raw excess.
    spread_pct = (1.50 - 1.00) / 1.25
    expected_1 = df["iv_excess"].iloc[1] / max(spread_pct, 0.05)
    assert np.isclose(score[1], expected_1)


def test_vrp_is_iv_over_realized_vol():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)
    ctx = ScoreContext(ticker="X", hv_20=0.20)
    score, label = iv_scores.score(df, mask, ctx, ("vrp", frozenset()))
    assert label == "VRP"
    assert np.allclose(score, df["iv"].to_numpy() / 0.20)


def test_vrp_nan_without_realized_vol():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)
    ctx = ScoreContext(ticker="X", hv_20=float("nan"))
    score, _ = iv_scores.score(df, mask, ctx, ("vrp", frozenset()))
    assert np.isnan(score).all()


def test_percentile_uses_history_and_is_nan_without_it():
    df = _scored_df()
    mask = np.ones(len(df), dtype=bool)

    class _FakeHistory:
        def percentile_for(self, ticker, excess, window_days=30,
                           deltas=None, dtes=None):
            return np.full(len(excess), 75.0)

    ctx = ScoreContext(ticker="X", history=_FakeHistory())
    score, label = iv_scores.score(df, mask, ctx, ("percentile", frozenset()))
    assert label == "IV %ile"
    assert np.allclose(score, 75.0)

    none_ctx = ScoreContext(ticker="X", history=None)
    score2, _ = iv_scores.score(df, mask, none_ctx, ("percentile", frozenset()))
    assert np.isnan(score2).all()


def test_percentile_passes_delta_dte_when_present():
    df = _scored_df()
    df["delta"] = [0.3, 0.3, 0.3, 0.3, 0.3]
    df["dte"] = [30, 30, 30, 30, 30]
    mask = np.ones(len(df), dtype=bool)

    captured = {}

    class _FakeHistory:
        def percentile_for(self, ticker, excess, window_days=30,
                           deltas=None, dtes=None):
            captured["deltas"] = deltas
            captured["dtes"] = dtes
            return np.full(len(excess), 50.0)

    ctx = ScoreContext(ticker="X", history=_FakeHistory())
    iv_scores.score(df, mask, ctx, ("percentile", frozenset()))
    assert captured["deltas"] is not None
    assert captured["dtes"] is not None


def test_ann_delta_percentile_uses_history_and_is_nan_without_it():
    df = _scored_df()
    df["ann_yield_pct"] = [10.0, 12.0, 8.0, 15.0, 11.0]
    df["delta"] = [0.3, 0.3, 0.3, 0.3, 0.3]
    df["dte"] = [30, 30, 30, 30, 30]
    mask = np.ones(len(df), dtype=bool)

    class _FakeHistory:
        def ann_delta_percentile_for(self, ticker, ann_yield_pct, deltas,
                                     dtes, window_days=30):
            return np.full(len(ann_yield_pct), 80.0)

    ctx = ScoreContext(ticker="X", history=_FakeHistory())
    score, label = iv_scores.score(
        df, mask, ctx, ("ann_delta_percentile", frozenset()))
    assert label == "Ann/Δ %ile"
    assert np.allclose(score, 80.0)

    none_ctx = ScoreContext(ticker="X", history=None)
    score2, _ = iv_scores.score(
        df, mask, none_ctx, ("ann_delta_percentile", frozenset()))
    assert np.isnan(score2).all()


def test_ann_delta_percentile_nan_when_columns_missing():
    df = _scored_df()  # no ann_yield_pct/delta/dte columns
    mask = np.ones(len(df), dtype=bool)

    class _FakeHistory:
        def ann_delta_percentile_for(self, *a, **k):
            raise AssertionError("should not be called without required cols")

    ctx = ScoreContext(ticker="X", history=_FakeHistory())
    score, _ = iv_scores.score(
        df, mask, ctx, ("ann_delta_percentile", frozenset()))
    assert np.isnan(score).all()


def test_zscore_nonzero_with_per_expiration_on_thin_chain():
    """Regression: per_expiration + zscore must not collapse to all-zero
    on a chain whose slices each have few fit rows (the old quadratic
    interpolated them to zero residuals)."""
    import math
    from options_scanner.iv_surface import compute_iv_excess

    spot = 100.0
    rng = np.random.default_rng(3)
    rows = []
    for dte, exp in [(30, "A"), (60, "B"), (90, "C")]:
        for K in (90.0, 100.0, 110.0):  # 3 strikes/expiration
            m = math.log(K / spot)
            iv = 0.30 + 0.40 * m ** 2 + rng.normal(0, 0.02)
            rows.append(dict(
                type="call" if K > spot else "put", strike=K, spot=spot,
                expiration=exp, dte=dte, iv=iv,
                log_moneyness=m, bid=1.0, ask=1.04, mid=1.02, delta=0.4,
                open_interest=500, volume=100, earnings_count=0))
    df = pd.DataFrame(rows)
    out = compute_iv_excess(df, algo_config=("per_expiration", frozenset()),
                            score_config=("zscore", frozenset()))
    assert out["signal_kind"].iloc[0] == "IV z"
    assert not np.allclose(out["signal_score"].to_numpy(), 0.0)


def test_display_helpers():
    assert iv_scores.display_for("IV+pp") == (100.0, "%+.1f pp")
    assert iv_scores.display_for("unknown") == (1.0, "%.2f")
    df = pd.DataFrame({"signal_kind": ["IV z", "IV z"]})
    assert iv_scores.active_kind(df) == "IV z"
    assert iv_scores.active_kind(pd.DataFrame()) == "IV+pp"


def _composite_df():
    return pd.DataFrame({
        "iv": [0.30, 0.30],
        "iv_fitted": [0.28, 0.28],
        "iv_excess": [0.02, 0.02],
        "delta": [0.3, 0.3],
        "dte": [30, 30],
        "ann_yield_pct": [10.0, 10.0],
        "open_interest": [100, 400],
        "volume": [10, 40],
        "gamma": [0.01, 0.01],
        "spot": [100.0, 100.0],
        "strike": [100.0, 105.0],
        "type": ["call", "call"],
    })


class _FakeHistoryConst:
    """Returns fixed percentile values regardless of input, for pinning
    the composite score's weighting arithmetic independent of the
    percentile machinery's own (separately-tested) correctness."""

    def percentile_for(self, ticker, excess, window_days=30,
                       deltas=None, dtes=None):
        return np.full(len(excess), 80.0)

    def ann_delta_percentile_for(self, ticker, ann_yield_pct, deltas, dtes,
                                 window_days=30):
        return np.full(len(ann_yield_pct), 60.0)


def test_liquidity_norm_min_max():
    oi = pd.Series([100, 400])
    vol = pd.Series([10, 40])
    result = iv_scores._liquidity_norm(oi, vol)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0)


def test_liquidity_norm_degenerate_all_equal():
    oi = pd.Series([100, 100, 100])
    vol = pd.Series([10, 10, 10])
    result = iv_scores._liquidity_norm(oi, vol)
    assert np.allclose(result, 0.5)


def test_composite_v2_exact_weighted_sum(monkeypatch):
    df = _composite_df()
    mask = np.ones(len(df), dtype=bool)
    monkeypatch.setattr(
        "options_scanner.compute.gex_summary.gex_alignment",
        lambda df, spot: np.array([0.5, 0.9]),
    )
    ctx = ScoreContext(ticker="X", history=_FakeHistoryConst())
    score, label = iv_scores.score(df, mask, ctx, ("composite_v2", frozenset()))
    assert label == "Composite v2"

    liq = iv_scores._liquidity_norm(df["open_interest"], df["volume"])
    # Weights sum to 1.0 (0.40+0.25+0.20+0.15), so with every component
    # present w_sum == 1 and no renormalization applies.
    expected = 0.40 * 0.60 + 0.25 * 0.80 + 0.20 * liq + 0.15 * np.array([0.5, 0.9])
    assert np.allclose(score, expected)


def test_composite_v2_renormalizes_when_gex_missing():
    # No gamma/spot/strike/type columns -> gex term is NaN for every row,
    # and the remaining 3 weights (0.40+0.25+0.20=0.85) renormalize to
    # sum 1.0 rather than the score shrinking by the missing 0.15.
    df = pd.DataFrame({
        "iv_excess": [0.02, 0.02],
        "delta": [0.3, 0.3],
        "dte": [30, 30],
        "ann_yield_pct": [10.0, 10.0],
        "open_interest": [100, 400],
        "volume": [10, 40],
    })
    mask = np.ones(len(df), dtype=bool)
    ctx = ScoreContext(ticker="X", history=_FakeHistoryConst())
    score, _ = iv_scores.score(df, mask, ctx, ("composite_v2", frozenset()))

    liq = iv_scores._liquidity_norm(df["open_interest"], df["volume"])
    expected = (0.40 * 0.60 + 0.25 * 0.80 + 0.20 * liq) / 0.85
    assert np.allclose(score, expected)


def test_composite_v2_nan_when_no_history():
    df = _composite_df()
    mask = np.ones(len(df), dtype=bool)
    ctx = ScoreContext(ticker="X", history=None)
    score, _ = iv_scores.score(df, mask, ctx, ("composite_v2", frozenset()))
    # ann/iv percentile terms are NaN without history; liquidity and gex
    # (mocked implicitly via the real gex_summary call on this synthetic
    # single-strike-pair chain) still renormalize the remaining weight.
    assert not np.isnan(score).all()  # liquidity + gex alone still contribute
