"""Premium-decay curves segmented by entry IV regime — a display layer
on top of `backtest.BacktestResult`, not a new simulator. Answers "does
this ticker's premium decay differently depending on how rich IV was
when the position was opened?"
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from options_scanner.backtest import BacktestResult
from options_scanner.ui_theme import PALETTE

_REGIME_LABELS = ["Low IV", "Below-avg", "Above-avg", "High IV"]
_REGIME_COLORS = [PALETTE["success"], PALETTE["primary"],
                  PALETTE["accent"], PALETTE["destructive"]]


def regime_bucket(entry_sigma_values: pd.Series) -> pd.Series:
    """Quartile bucket of `entry_sigma` across one backtest run's trades
    — self-relative to that run's own window, not an external
    calibration (a backtest is a closed set of trades; "high IV" only
    means "high relative to this same ticker's other entries here").

    Falls back to a single "All" bucket when there aren't enough
    distinct values for 4 quartile bins (e.g. very few trades, or a
    flat vol series) rather than letting `pd.qcut` raise on duplicate
    bin edges.
    """
    if entry_sigma_values.nunique() < 4:
        return pd.Series(["All"] * len(entry_sigma_values),
                         index=entry_sigma_values.index)
    return pd.qcut(entry_sigma_values, 4, labels=_REGIME_LABELS)


def build_decay_curve_data(result: BacktestResult) -> pd.DataFrame:
    """One row per (trade, day-offset-from-entry, regime), with
    `time_value_per_share` normalized to % of that trade's entry-day
    time value — so trades of different premium levels average
    together meaningfully. Empty (right columns) when there are no
    trades or none have usable `daily_value` data."""
    cols = ["trade_idx", "day_offset", "regime", "pct_remaining"]
    if not result.trades:
        return pd.DataFrame(columns=cols)

    entry_sigmas = pd.Series([t.entry_sigma for t in result.trades])
    regimes = regime_bucket(entry_sigmas)

    rows: list[dict] = []
    for i, (trade, regime) in enumerate(zip(result.trades, regimes)):
        daily = trade.daily_value
        if (daily is None or daily.empty
                or "time_value_per_share" not in daily.columns):
            continue
        entry_tv = float(daily["time_value_per_share"].iloc[0])
        if entry_tv <= 1e-9:
            continue
        for offset, (_, row) in enumerate(daily.iterrows()):
            pct = float(row["time_value_per_share"]) / entry_tv * 100.0
            rows.append({"trade_idx": i, "day_offset": offset,
                        "regime": regime, "pct_remaining": pct})
    return pd.DataFrame(rows, columns=cols)


def render_decay_curves(result: BacktestResult) -> None:
    """Altair line chart, one line per entry-IV-regime bucket, x = days
    since entry, y = mean % of entry time-value remaining. No-op
    (informational message) when there isn't enough data to build one."""
    data = build_decay_curve_data(result)
    if data.empty:
        st.info("Not enough trade data to build decay curves.")
        return

    curve = (
        data.groupby(["regime", "day_offset"], observed=True)["pct_remaining"]
        .mean()
        .reset_index()
    )
    present_regimes = [r for r in _REGIME_LABELS if r in set(curve["regime"])]
    color_range = ([PALETTE["primary"]] if present_regimes == ["All"]
                   else [_REGIME_COLORS[_REGIME_LABELS.index(r)]
                         for r in present_regimes])
    domain = present_regimes if present_regimes else ["All"]

    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(
            x=alt.X("day_offset:Q", title="Days since entry"),
            y=alt.Y("pct_remaining:Q", title="% of entry time value remaining"),
            color=alt.Color("regime:N", title="IV regime at entry",
                            scale=alt.Scale(domain=domain, range=color_range)),
        )
        .properties(height=320, title="Premium decay by entry IV regime")
    )
    st.altair_chart(chart, width="stretch")
