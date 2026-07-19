"""Shared number-formatting helpers.

Leaf module — no internal imports, safe to import anywhere (display,
tabs, report, CLI). Centralizes strike formatting so 2.5-/0.25-wide
strikes (NVDA, TSLA, AAPL, SOFI, …) render their decimals everywhere
instead of being rounded to the nearest dollar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# d3-format string for option strikes on Altair/Vega and Plotly
# charts. The `~` trims trailing zeros so whole strikes render as
# "$145" while fractional strikes keep their decimals ("$142.5",
# "$12.75"). Mirrors `fmt_strike` below for f-string call sites.
STRIKE_D3_FORMAT = "$,.2~f"

# Legend for the ⚠ marker appended to short-dated, post-earnings
# expiration cells in the ranked tables. Shown as an always-visible
# caption under the table (st.dataframe has no per-cell hover; only the
# column header carries a tooltip), so the meaning isn't hidden.
EARNINGS_WARN_LEGEND = (
    "⚠ next to a date = ≤60 DTE and expiring after the next earnings — its "
    "IV+pp includes event premium (and it's excluded from the surface fit)."
)


def fmt_strike(strike) -> str:
    """Format an option strike as a dollar string.

    Shows decimals only when the strike isn't a whole number:
    145 -> "$145", 142.5 -> "$142.5", 12.75 -> "$12.75". Keeps
    big-ticker integer strikes clean while preserving fractional
    strikes. Mirrors `STRIKE_D3_FORMAT` used on the charts.
    """
    x = float(strike)
    if x.is_integer():
        return f"${x:,.0f}"
    return f"${x:,.2f}".rstrip("0").rstrip(".")


def stars_for(scores) -> np.ndarray:
    """0-5 star rating at 0.5 resolution, from each row's percentile rank
    of `scores` within the same set — "how does this compare to the rest
    of what's on screen right now," same top-N-of-this-basket framing as
    everywhere else in this codebase, not a fixed external calibration.

    NaN scores rank last (na_option="bottom") and are then blanked back
    out to NaN in the result, so a row with no score (e.g. cold-start
    percentile) doesn't quietly render as "0 stars" — it renders as no
    rating at all (fmt_stars("") for the UI).
    """
    s = pd.Series(scores)
    ranks = s.rank(pct=True, na_option="bottom").to_numpy(dtype=float)
    stars = np.round(ranks * 5.0 * 2.0) / 2.0
    stars[s.isna().to_numpy()] = np.nan
    return stars


def fmt_stars(n: float) -> str:
    """Render a 0-5, 0.5-resolution star rating as a short string:
    2.0 -> '★★', 2.5 -> '★★½'. NaN -> '' (blank, not '0 stars')."""
    if n != n:  # NaN
        return ""
    full = int(n)
    half = (n - full) >= 0.5
    return "★" * full + ("½" if half else "")


def strike_tick_values(strikes, lo=None, hi=None, max_ticks=16):
    """Axis tick positions aligned to the real option strikes.

    Vega-Lite's automatic ticks land on "nice" round steps (1, 2, 5, …), so
    $0.50-/$2.50-wide strikes (7.5, 152.5) never get a labeled tick. Passing
    the actual strikes as the axis `values` forces ticks onto them.

    Restricts to the [lo, hi] domain when given and thins uniformly to at
    most `max_ticks` so wide chains don't crowd the axis (the kept ticks are
    still real strikes). Returns an empty list when there are no strikes, so
    callers can fall back to Vega's default ticks.
    """
    vals = sorted({round(float(s), 4) for s in strikes if s is not None})
    if lo is not None:
        vals = [v for v in vals if v >= lo - 1e-9]
    if hi is not None:
        vals = [v for v in vals if v <= hi + 1e-9]
    if len(vals) > max_ticks:
        step = -(-len(vals) // max_ticks)   # ceil division
        vals = vals[::step]
    return vals
