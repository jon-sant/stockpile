"""Signal-score registry for iv_surface.compute_iv_excess.

A score turns the fitted surface (iv, iv_fitted, iv_excess) plus
ticker-level context into a per-row `signal_score` — the number the
scanner ranks by. The default score is raw IV+pp, so ranking is
unchanged until the user picks another score.

These are screening signals, not mispricing claims: a high score
means the contract's IV stands above the fitted surface (IV-rich),
which may reflect demand, event risk, or a stale print as easily as a
tradeable edge.

ScoreConfig is a single (name, frozenset-of-kwargs) pair — hashable
for st.cache_data, like iv_algorithms.AlgorithmConfig.

Each score:  fn(df, fit_mask, ctx, **kwargs) -> (np.ndarray, label)
  - np.ndarray is signal_score aligned to df row order
  - label is the short column header naming the active score

Adding a new score
------------------
1. Write fn(df, fit_mask, ctx, **kwargs) -> (np.ndarray, str)
2. Add an entry to REGISTRY with fn, defaults, and label
3. It appears automatically in the UI Score dropdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# Hashable single-score config: (name, frozenset({(kwarg, val), ...}))
ScoreConfig = tuple[str, frozenset]


@dataclass
class ScoreContext:
    """Ticker-level context a score may need beyond the chain itself."""
    ticker: str | None = None
    hv_20: float = float("nan")
    history: object | None = None          # iv_history module (or None)
    window_days: int = 30
    extra: dict = field(default_factory=dict)


# ── Scores ──────────────────────────────────────────────────────────────────────

def _raw_pp(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """IV excess in raw percentage points (current behavior)."""
    return df["iv_excess"].to_numpy(dtype=float), "IV+pp"


def _zscore(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """IV excess expressed in standard deviations of the fit residuals.

    A +1.5σ reading means the same thing regardless of the ticker's
    baseline IV level — the most ticker-comparable framing.
    """
    excess = df["iv_excess"].to_numpy(dtype=float)
    resid = excess[np.asarray(fit_mask, dtype=bool)]
    sd = float(np.std(resid)) if resid.size else 0.0
    if sd < 1e-9:
        return np.zeros_like(excess), "IV z"
    return excess / sd, "IV z"


def _relative(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """IV excess as a fraction of the fitted IV ("10% above surface")."""
    excess = df["iv_excess"].to_numpy(dtype=float)
    fitted = df["iv_fitted"].to_numpy(dtype=float)
    rel = np.divide(excess, fitted, out=np.zeros_like(excess),
                    where=fitted > 1e-9)
    return rel, "IV rel"


def _composite_exec(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """IV excess discounted by bid-ask spread — penalizes unfillable quotes."""
    excess = df["iv_excess"].to_numpy(dtype=float)
    if not {"ask", "bid", "mid"} <= set(df.columns):
        return excess, "Score"
    mid = df["mid"].to_numpy(dtype=float)
    spread = df["ask"].to_numpy(dtype=float) - df["bid"].to_numpy(dtype=float)
    spread_pct = np.divide(spread, mid, out=np.full_like(spread, np.inf),
                           where=mid > 0)
    return excess / np.maximum(spread_pct, 0.05), "Score"


def _vrp(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """Volatility-risk-premium ratio: IV / 20-day realized vol.

    Orthogonal to iv_excess — answers "is this ticker's IV elevated vs.
    what the stock actually does?" rather than "rich vs. other options."
    Constant across a single ticker's chain; most useful cross-ticker.
    """
    iv = df["iv"].to_numpy(dtype=float)
    hv = getattr(ctx, "hv_20", float("nan"))
    if not (np.isfinite(hv) and hv > 0):
        return np.full_like(iv, np.nan), "VRP"
    return iv / hv, "VRP"


def _percentile(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """Percentile rank of each contract's IV excess within the ticker's
    own delta×DTE-bucketed trailing history (needs the scan-history store).

    Bucketed when `delta`/`dte` are present on `df` (the normal case —
    falls back to whole-chain pooling otherwise, matching
    `iv_history.percentile_for`'s own backward-compatible default)."""
    excess = df["iv_excess"]
    history = getattr(ctx, "history", None)
    ticker = getattr(ctx, "ticker", None)
    if history is None or not ticker:
        return np.full(len(df), np.nan), "IV %ile"
    has_bucket_cols = {"delta", "dte"} <= set(df.columns)
    pct = history.percentile_for(
        ticker, excess, window_days=getattr(ctx, "window_days", 30),
        deltas=df["delta"] if has_bucket_cols else None,
        dtes=df["dte"] if has_bucket_cols else None,
    )
    return np.asarray(pct, dtype=float), "IV %ile"


def _ann_delta_percentile(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """Percentile rank of each contract's Ann% ÷ |Delta| within the
    ticker's own delta×DTE-bucketed trailing history — the literal
    "is this yield historically rich for this ETF at this delta/DTE"
    signal (needs the scan-history store)."""
    history = getattr(ctx, "history", None)
    ticker = getattr(ctx, "ticker", None)
    needed = {"ann_yield_pct", "delta", "dte"}
    if history is None or not ticker or not needed <= set(df.columns):
        return np.full(len(df), np.nan), "Ann/Δ %ile"
    pct = history.ann_delta_percentile_for(
        ticker, df["ann_yield_pct"], df["delta"], df["dte"],
        window_days=getattr(ctx, "window_days", 30))
    return np.asarray(pct, dtype=float), "Ann/Δ %ile"


_V2_WEIGHTS = {"ann_delta_pct": 0.40, "iv_pct": 0.25,
              "liquidity": 0.20, "gex": 0.15}


def _liquidity_norm(oi: pd.Series, volume: pd.Series) -> np.ndarray:
    """Min-max normalized log1p(OI)+log1p(Vol) within the current
    chain/basket, 0..1. All-equal input -> 0.5 (not NaN/inf) — a
    degenerate/tiny chain shouldn't silently zero out the liquidity term."""
    raw = (np.log1p(oi.to_numpy(dtype=float))
           + np.log1p(volume.to_numpy(dtype=float)))
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-9:
        return np.full_like(raw, 0.5)
    return (raw - lo) / (hi - lo)


def _composite_v2(df: pd.DataFrame, fit_mask, ctx) -> tuple[np.ndarray, str]:
    """Weighted blend of Ann%/Delta percentile, IV percentile, liquidity,
    and GEX alignment — the historically-aware, cross-signal ranking
    the feedback review asked for. Composes the existing
    `_percentile`/`_ann_delta_percentile` scores (0-100 -> /100) and
    `gex_summary.gex_alignment` (already 0-1) rather than duplicating
    their logic, so this always matches what those scores show
    standalone.

    Note on call order: this runs inside `compute_iv_excess`, BEFORE
    `fetch.py`'s post-hoc `iv_percentile`/`ann_delta_percentile`/
    `gex_alignment` columns exist — so components are computed fresh
    here from `ctx.history` and the raw chain columns, not read off
    those columns.

    Renormalizes weights per row over whichever components are actually
    available/non-NaN, so a cold-start NaN in one term (e.g. no history
    yet) doesn't poison the whole score — dropping a 0.15-weight term
    entirely renormalizes the remaining weights to sum to 1.0 rather
    than shrinking the score by 0.15.
    """
    n = len(df)
    iv_pct_raw, _ = _percentile(df, fit_mask, ctx)
    ann_pct_raw, _ = _ann_delta_percentile(df, fit_mask, ctx)
    iv_pct = np.asarray(iv_pct_raw, dtype=float) / 100.0
    ann_pct = np.asarray(ann_pct_raw, dtype=float) / 100.0

    if {"open_interest", "volume"} <= set(df.columns):
        liquidity = _liquidity_norm(df["open_interest"], df["volume"])
    else:
        liquidity = np.full(n, np.nan)

    if n and {"gamma", "spot", "strike", "type"} <= set(df.columns):
        from options_scanner.compute import gex_summary
        gex = gex_summary.gex_alignment(df, float(df["spot"].iloc[0]))
    else:
        gex = np.full(n, np.nan)

    comps = np.stack([ann_pct, iv_pct, liquidity, gex], axis=1)  # (n, 4)
    weights = np.array([_V2_WEIGHTS["ann_delta_pct"], _V2_WEIGHTS["iv_pct"],
                        _V2_WEIGHTS["liquidity"], _V2_WEIGHTS["gex"]])
    avail = ~np.isnan(comps)
    w_sum = (avail * weights).sum(axis=1)
    weighted = np.where(avail, comps, 0.0) * weights
    score = np.divide(weighted.sum(axis=1), w_sum,
                      out=np.full(n, np.nan), where=w_sum > 0)
    return score, "Composite v2"


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: dict[str, dict] = {
    "raw_pp": {
        "fn":       _raw_pp,
        "defaults": {},
        "label":    "IV+pp — raw excess (current)",
        "enabled":  True,
    },
    "zscore": {
        "fn":       _zscore,
        "defaults": {},
        "label":    "Z-score — σ above the surface",
        "enabled":  True,
    },
    "relative": {
        "fn":       _relative,
        "defaults": {},
        "label":    "Relative — % above the surface",
        "enabled":  True,
    },
    "composite_exec": {
        "fn":       _composite_exec,
        "defaults": {},
        "label":    "Composite — excess ÷ spread cost",
        "enabled":  True,
    },
    "vrp": {
        "fn":       _vrp,
        "defaults": {},
        "label":    "VRP — IV vs. 20-day realized vol",
        "enabled":  True,
    },
    "percentile": {
        "fn":       _percentile,
        "defaults": {},
        "label":    "Percentile — IV richness vs. history",
        "enabled":  True,
    },
    "ann_delta_percentile": {
        "fn":       _ann_delta_percentile,
        "defaults": {},
        "label":    "Ann%/Delta Percentile — vs. own bucketed history",
        "enabled":  True,
    },
    "composite_v2": {
        "fn":       _composite_v2,
        "defaults": {},
        "label":    "Composite v2 — Ann/Δ + IV%ile + liquidity + GEX",
        "enabled":  True,
    },
}

# Default: raw IV+pp — reproduces current ranking exactly.
DEFAULT_CONFIG: ScoreConfig = ("raw_pp", frozenset())

# Default ranking key for the ★ star rating — independent of DEFAULT_CONFIG
# (which drives table ranking/sort). Composite v2 blends Ann/Δ, IV
# percentile, liquidity, and GEX, so stars stay meaningful even when the
# active ranking score is something narrow like raw IV+pp.
STAR_DEFAULT: ScoreConfig = ("composite_v2", frozenset())

# Per-label display spec: (multiplier applied to signal_score, column format).
# Keyed by the short label each score returns.
# ASCII-only formats — these feed both Streamlit column_config and the
# CLI's tabulate output, which prints to a cp1252 console on Windows.
SCORE_DISPLAY: dict[str, tuple[float, str]] = {
    "IV+pp":   (100.0, "%+.1f pp"),
    "IV z":    (1.0,   "%+.2f"),
    "IV rel":  (100.0, "%+.1f%%"),
    "Score":   (1.0,   "%+.2f"),
    "VRP":     (1.0,   "%.2f"),
    "IV %ile": (1.0,   "%.0f"),
    "Ann/Δ %ile": (1.0, "%.0f"),
    "Composite v2": (100.0, "%.1f"),
}


def display_for(label: str) -> tuple[float, str]:
    """Return (multiplier, column format) for rendering a score label."""
    return SCORE_DISPLAY.get(label, (1.0, "%.2f"))


def active_kind(df) -> str:
    """The score label carried by a scored chain, defaulting to IV+pp."""
    if "signal_kind" in getattr(df, "columns", []) and len(df):
        return str(df["signal_kind"].iloc[0])
    return "IV+pp"


def active_star_kind(df) -> str:
    """The score label driving the ★ rating, defaulting to Composite v2."""
    if "star_kind" in getattr(df, "columns", []) and len(df):
        return str(df["star_kind"].iloc[0])
    return "Composite v2"


# ── Dispatch ────────────────────────────────────────────────────────────────────

def score(df: pd.DataFrame, fit_mask, ctx: ScoreContext | None,
          config: ScoreConfig) -> tuple[np.ndarray, str]:
    """Run the configured score. Returns (signal_score array, label)."""
    name, kwargs_fs = config
    entry = REGISTRY.get(name, REGISTRY["raw_pp"])
    fn: Callable = entry["fn"]
    return fn(df, fit_mask, ctx, **dict(kwargs_fs))
