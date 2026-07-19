"""Per-ticker Gamma Exposure (GEX) summary: total net GEX, regime,
zero-gamma flip, strongest pinning wall, and strongest amp zone.

Same math the GEX chart uses, distilled to single numbers so the
multi-ticker GEX summary table can rank tickers without re-rendering
the full chart for each one.

Convention: GEX is the dealer's net gamma exposure. Calls contribute
positive gamma (dealers long via market-making → short to hedge);
puts contribute negative gamma (dealers short via market-making →
long to hedge). Net positive GEX = "pinning" regime (price reverts
to the wall); net negative = "amplifying" regime (price runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# U.S. equity/ETF options carry 100 shares per contract. The 0.01 scales
# the raw dollar-gamma to the change in dealer delta per a 1% move in the
# underlying — the standard GEX definition (SpotGamma et al.).
_CONTRACT_SIZE = 100
_PCT_MOVE = 0.01


def per_strike_gex(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Net dealer GEX per strike (calls +, puts −), summed across the chain.

    Per contract: gamma × 100 (contract size) × open_interest × spot² × 0.01
    — the dollar change in dealer delta per a 1% move in the underlying.
    Calls add positive gamma, puts negative.

    Returns a frame with columns [strike, gex, open_interest] sorted by
    strike — the shared primitive the GEX chart, the strikes-of-interest
    table, and the summary below all build on. Empty frame when `df` is
    empty or has no `gamma` column.
    """
    cols = ["strike", "gex", "open_interest"]
    if df.empty or "gamma" not in df.columns:
        return pd.DataFrame(columns=cols)
    cp = df[df["type"].isin(("call", "put"))]
    if cp.empty:
        return pd.DataFrame(columns=cols)
    sign = cp["type"].map({"call": 1.0, "put": -1.0})
    gex = (sign * cp["gamma"] * cp["open_interest"]
           * _CONTRACT_SIZE * (spot * spot) * _PCT_MOVE)
    return (
        pd.DataFrame({"strike": cp["strike"], "gex": gex,
                      "open_interest": cp["open_interest"]})
        .groupby("strike", as_index=False)
        .agg({"gex": "sum", "open_interest": "sum"})
        .sort_values("strike")
        .reset_index(drop=True)
    )


def gamma_flip_strike(per_strike: pd.DataFrame, spot: float,
                      lo: float | None = None,
                      hi: float | None = None) -> float:
    """Strike where cumulative net GEX flips sign, nearest to spot.

    Walks strikes low→high accumulating net GEX; the flip (a.k.a. the
    zero-gamma level) is where that running total *crosses* zero, linearly
    interpolated between the two bracketing strikes. When several crossings
    exist the one nearest `spot` wins — the meaningful flip sits among the
    GEX-bearing strikes near spot, not out in a far low-OI tail.

    This deliberately differs from "lowest strike whose cumulative GEX is
    ≥ 0": when the running total *starts* positive that rule just returns
    the lowest strike (e.g. a bogus 245 on SPY), which isn't a flip at all.

    `lo`/`hi` optionally restrict the search to the GEX-bearing window.
    Returns NaN when the cumulative total never changes sign in range.
    """
    d = per_strike.sort_values("strike")
    if lo is not None:
        d = d[d["strike"] >= lo]
    if hi is not None:
        d = d[d["strike"] <= hi]
    if len(d) < 2:
        return float("nan")
    strikes = d["strike"].to_numpy(dtype=float)
    cum = d["gex"].cumsum().to_numpy(dtype=float)
    flips: list[float] = []
    for i in range(1, len(cum)):
        a, b = cum[i - 1], cum[i]
        if a == 0.0:
            flips.append(strikes[i - 1])
        elif (a < 0.0) != (b < 0.0):  # sign change between adjacent strikes
            t = a / (a - b)           # interpolate to the zero crossing
            flips.append(strikes[i - 1] + t * (strikes[i] - strikes[i - 1]))
    if cum[-1] == 0.0:
        flips.append(strikes[-1])
    if not flips:
        return float("nan")
    return min(flips, key=lambda k: abs(k - spot))


def compute_gex_summary(df: pd.DataFrame, spot: float) -> dict | None:
    """Reduce a GEX-enriched chain to one summary dict per ticker.

    Args:
        df: Chain DataFrame with `type`, `strike`, `open_interest`, and
            `gamma` columns.
        spot: Current underlying spot price.

    Returns:
        Dict with keys:
            total_gex:  Net sum of per-strike GEX across calls + puts.
            regime:     "Pinning" when total_gex >= 0, else "Amplifying".
            zero_gamma: Strike where cumulative GEX flips sign, nearest
                        spot (NaN if it never changes sign).
            top_wall:   Strike with the largest positive net GEX, or
                        None when no strike has positive net GEX.
            top_amp:    Strike with the largest negative net GEX, or
                        None when no strike has negative net GEX.

        Returns None when the input is empty, lacks a gamma column, or
        has zero total absolute GEX (degenerate chain).
    """
    per_strike = per_strike_gex(df, spot)
    if per_strike.empty or per_strike["gex"].abs().sum() == 0:
        return None

    total_gex = float(per_strike["gex"].sum())
    zero_gamma = gamma_flip_strike(per_strike, spot)

    walls = per_strike[per_strike["gex"] > 0]
    amps = per_strike[per_strike["gex"] < 0]
    top_wall = (float(walls.loc[walls["gex"].idxmax(), "strike"])
                if not walls.empty else None)
    top_amp = (float(amps.loc[amps["gex"].idxmin(), "strike"])
               if not amps.empty else None)

    return {
        "total_gex": total_gex,
        "regime": "Pinning" if total_gex >= 0 else "Amplifying",
        "zero_gamma": zero_gamma,
        "top_wall": top_wall,
        "top_amp": top_amp,
    }


_AMP_PENALTY_WEIGHT = 0.5  # tunable starting default, not load-bearing


def gex_alignment(df: pd.DataFrame, spot: float) -> np.ndarray:
    """Per-row [0,1] GEX alignment score, one value per row of `df`.

    High near the chain's strongest pinning wall (price tends to revert
    there), discounted near the strongest amp zone (price tends to run
    away from there). Turns `compute_gex_summary`'s per-ticker wall/amp
    strikes into a per-contract number a composite score can blend in.

    Returns NaN (not 0) for every row when GEX can't be computed at all
    (no gamma column, empty chain, or a degenerate all-zero-GEX chain) —
    so a downstream composite can skip the term entirely via
    renormalization instead of silently treating "unknown" as "bad".
    """
    n = len(df)
    per_strike = per_strike_gex(df, spot)
    if per_strike.empty or per_strike["gex"].abs().sum() == 0:
        return np.full(n, np.nan)

    walls = per_strike[per_strike["gex"] > 0]
    amps = per_strike[per_strike["gex"] < 0]
    top_wall = (float(walls.loc[walls["gex"].idxmax(), "strike"])
                if not walls.empty else None)
    top_amp = (float(amps.loc[amps["gex"].idxmin(), "strike"])
               if not amps.empty else None)

    strikes = df["strike"].to_numpy(dtype=float)
    norm = max(float(spot), 1e-9)

    wall_term = (1.0 / (1.0 + np.abs(strikes - top_wall) / norm)
                if top_wall is not None else np.zeros(n))
    amp_term = (1.0 / (1.0 + np.abs(strikes - top_amp) / norm)
               if top_amp is not None else np.zeros(n))
    return np.clip(wall_term - _AMP_PENALTY_WEIGHT * amp_term, 0.0, 1.0)
