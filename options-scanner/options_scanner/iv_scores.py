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
}

# Default: raw IV+pp — reproduces current ranking exactly.
DEFAULT_CONFIG: ScoreConfig = ("raw_pp", frozenset())

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
}


def display_for(label: str) -> tuple[float, str]:
    """Return (multiplier, column format) for rendering a score label."""
    return SCORE_DISPLAY.get(label, (1.0, "%.2f"))


def active_kind(df) -> str:
    """The score label carried by a scored chain, defaulting to IV+pp."""
    if "signal_kind" in getattr(df, "columns", []) and len(df):
        return str(df["signal_kind"].iloc[0])
    return "IV+pp"


# ── Dispatch ────────────────────────────────────────────────────────────────────

def score(df: pd.DataFrame, fit_mask, ctx: ScoreContext | None,
          config: ScoreConfig) -> tuple[np.ndarray, str]:
    """Run the configured score. Returns (signal_score array, label)."""
    name, kwargs_fs = config
    entry = REGISTRY.get(name, REGISTRY["raw_pp"])
    fn: Callable = entry["fn"]
    return fn(df, fit_mask, ctx, **dict(kwargs_fs))
