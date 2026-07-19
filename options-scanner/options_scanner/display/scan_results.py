"""Ranked scan-results table renderer for the Single Ticker tab.

`show_scan_results` is the entry point — splits the input chain by
call/put, applies the OI/Vol filters, ranks each side, and delegates
the actual table render to `show_df`. `show_df` is also reused from
the Portfolio tab's per-position view.

The yellow row highlights and column tooltips come from
`display.chain_styling`; the source/timestamp caption below the
table comes from `display.scan_stamp.stamp_caption`.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from options_scanner import iv_scores
from options_scanner.format import EARNINGS_WARN_LEGEND, fmt_stars, fmt_strike, stars_for
from options_scanner.ui_theme import empty_state

from options_scanner.display import mc_columns
from options_scanner.display.chain_styling import (
    SPREAD_HELP,
    LAST_HELP,
    CELL_WARN,
    OI_HELP,
    vol_help_for,
    ivpp_help_for,
    last_outside_mask,
    low_oi_mask,
    low_vol_mask,
    wide_spread_mask,
)
from options_scanner.display.scan_stamp import stamp_caption


def _prem_pct_em(sub: pd.DataFrame) -> pd.Series:
    """Premium as a fraction of the option's own theoretical 1-sigma
    expected move: mid / (spot * iv * sqrt(dte/365)). >1 means the
    premium exceeds the modeled 1-sigma move; informational only, no
    ranking/filter tie-in. dte is clamped to >=1 to avoid a 0-DTE
    divide-by-zero (mirrors the ann_yield_pct convention elsewhere)."""
    if not {"mid", "spot", "iv", "dte"} <= set(sub.columns):
        return pd.Series([float("nan")] * len(sub), index=sub.index)
    expected_move = (sub["spot"] * sub["iv"]
                     * np.sqrt(sub["dte"].clip(lower=1) / 365.0))
    return (sub["mid"] / expected_move.replace(0, float("nan"))).round(2)


def _iv_rank_help(sub: pd.DataFrame) -> str:
    """Column-header tooltip for IV Rank — this table is always one
    ticker (Single Ticker tab, or one Portfolio position at a time), so
    the exact date range actually evaluated can be shown, not just a
    generic description."""
    base = ("0-100: where today's IV sits between this ticker's own "
            "min and max scanned IV, over up to 365 days of scan "
            "history (this app's own accumulated scans — no external "
            "historical-IV provider exists).")
    if "iv_rank_date_from" not in sub.columns or sub.empty:
        return base
    d_from = sub["iv_rank_date_from"].iloc[0]
    d_to = sub["iv_rank_date_to"].iloc[0]
    if pd.isna(d_from) or pd.isna(d_to):
        return base + " Blank until at least 2 distinct scan days exist."
    return f"{base} Range evaluated: {d_from} to {d_to}."


def show_df(sub: pd.DataFrame, roll_close_cost: float | None = None,
            min_oi: int = 0, min_vol: int = 0,
            buy: bool = False, opt_type: str = "option",
            show_mc: bool = False) -> None:
    """Render the styled table for one option-type subset (or one
    per-position view from the Portfolio tab).

    Empty input renders an `empty_state` callout so the user knows
    the table didn't fail to load — it just has nothing to show.
    When `roll_close_cost` is supplied (roll-an-existing-position
    flow), an extra Net Credit column is appended. `show_mc` gates the
    7 extra Monte Carlo columns (P(profit) through Sortino) behind the
    "Show Monte Carlo columns" checkbox — Exp P&L is always shown.
    """
    if sub.empty:
        empty_state(
            "No matches in this chain",
            "Try widening the delta band, lowering min OI/Volume, or "
            "extending the DTE range.",
        )
        return

    rank_col = {"Top": sub["_rank"]} if "_rank" in sub.columns else {}
    kind = iv_scores.active_kind(sub)
    star_kind = iv_scores.active_star_kind(sub)

    # ⚠ in the Expiration cell = short-dated (≤60 DTE) and expiring after the
    # next earnings — its IV+pp carries event premium and it's the slice
    # excluded from the surface fit.
    _ec = (sub["earnings_count"].fillna(0) if "earnings_count" in sub.columns
           else pd.Series(0, index=sub.index))

    def _exp_cell(e, d, c):
        base = datetime.strptime(e, "%Y-%m-%d").strftime("%b %d '%y")
        return f"{base} ⚠" if (c >= 1 and d <= 60) else base

    _has_warn = any(c >= 1 and d <= 60 for c, d in zip(_ec, sub["dte"]))

    cols = {
        **rank_col,
        "Strike": sub["strike"].apply(fmt_strike),
        "Expiration": [_exp_cell(e, d, c) for e, d, c
                       in zip(sub["expiration"], sub["dte"], _ec)],
        "DTE":    sub["dte"].astype(int),
        "Bid":    sub["bid"].round(2),
        "Ask":    sub["ask"].round(2),
        "Mid":    sub["mid"].round(2),
        "Spread": (sub["ask"] - sub["bid"]).round(2),
        "Last":   sub["last"].where(sub["last"] > 0) if "last" in sub.columns else pd.Series([float("nan")] * len(sub), index=sub.index),
        "IV%":    (sub["iv"] * 100).round(1),
        "IV+pp":  (sub["iv_excess"] * 100).round(1),
        "IV Rank": (sub["iv_rank"].round(0) if "iv_rank" in sub.columns
                    else pd.Series([float("nan")] * len(sub), index=sub.index)),
    }
    # When a non-default score drives the ranking, show it alongside IV+pp.
    if kind != "IV+pp":
        mult, _ = iv_scores.display_for(kind)
        cols[kind] = (sub["signal_score"] * mult).round(2)
    _star_scores = sub["star_score"] if "star_score" in sub.columns else sub["signal_score"]
    cols["★"] = [fmt_stars(s) for s in stars_for(_star_scores)]
    cols.update({
        "Delta":  sub["delta"].round(2),
        "Exp P&L": mc_columns.exp_pnl_column(sub, buy),
        "Ann%":   sub["ann_yield_pct"].round(1),
        "Ann% / Delta": (sub["ann_yield_pct"]
                         / sub["delta"].abs().replace(0, float("nan"))).round(1),
        "Prem/EM": _prem_pct_em(sub),
        "OI":     sub["open_interest"],
        "Vol":    sub["volume"],
    })
    if show_mc:
        cols.update(mc_columns.mc_extra_columns(sub, buy))
    disp = pd.DataFrame(cols)
    if roll_close_cost is not None:
        disp["NetCr"] = (sub["mid"] - roll_close_cost).round(2)

    wide = wide_spread_mask(sub["bid"], sub["ask"], sub["mid"])
    last_out = (last_outside_mask(sub["last"], sub["bid"], sub["ask"])
                if "last" in sub.columns else [False] * len(sub))
    lo = low_oi_mask(sub["open_interest"], min_oi)
    low_vol = low_vol_mask(sub["volume"], min_vol)

    styled = (
        disp.style
        .apply(lambda _: [CELL_WARN if w else "" for w in wide],
               subset=["Spread"])
        .apply(lambda _: [CELL_WARN if o else "" for o in last_out],
               subset=["Last"])
        .apply(lambda _: [CELL_WARN if l else "" for l in lo],
               subset=["OI"])
        .apply(lambda _: [CELL_WARN if v else "" for v in low_vol],
               subset=["Vol"])
    )

    col_cfg = {}
    if "_rank" in sub.columns:
        col_cfg["Top"] = st.column_config.NumberColumn("Top", format="%d",
                                                       width=45)
    col_cfg.update({
        "Strike":     st.column_config.TextColumn("Strike", width=75),
        "Expiration": st.column_config.TextColumn(
            "Expiration", width=115,
            help="⚠ = ≤60 DTE and expiring after the next earnings, so its "
                 "IV+pp includes event premium (and it's excluded from the "
                 "surface fit)."),
        "DTE":   st.column_config.NumberColumn("DTE", format="%d", width=55),
        "Bid":   st.column_config.NumberColumn("Bid", format="$%.2f",
                                               width=70),
        "Ask":   st.column_config.NumberColumn("Ask", format="$%.2f",
                                               width=70),
        "Mid":   st.column_config.NumberColumn("Mid", format="$%.2f",
                                               width=70),
        "Spread": st.column_config.NumberColumn("Spread", format="$%.2f",
                                                width=75, help=SPREAD_HELP),
        "Last":  st.column_config.NumberColumn("Last", format="$%.2f",
                                               width=70, help=LAST_HELP),
        "IV%":   st.column_config.NumberColumn("IV%", format="%.1f%%",
                                               width=70),
        "IV+pp": st.column_config.NumberColumn("IV+pp", format="%+.1f pp",
                                               width=75,
                                               help=ivpp_help_for(buy, opt_type)),
        "IV Rank": st.column_config.NumberColumn(
            "IV Rank", format="%.0f", width=75,
            help=_iv_rank_help(sub)),
        "★": st.column_config.TextColumn(
            "★", width=70,
            help=f"Star rating: percentile rank of the {star_kind} score "
                 "within this table, mapped to 0-5 stars at half-star "
                 "resolution. Independent of the ranking/sort column above "
                 "— set via the Stars (ranking key) dropdown in Advanced "
                 "surface fit."),
        "Delta": st.column_config.NumberColumn("Delta", format="%.2f",
                                               width=60),
        "Exp P&L": st.column_config.TextColumn(
            "Exp P&L", width=85, help=mc_columns.exp_pnl_help()),
        "Ann%":  st.column_config.NumberColumn("Ann%", format="%.1f%%",
                                               width=65),
        "Ann% / Delta": st.column_config.NumberColumn(
            "Ann% / Delta", format="%.1f", width=95,
            help="Ann% divided by |Delta| — yield per unit of directional "
                 "exposure. Higher = more yield for the assignment risk "
                 "taken on."),
        "Prem/EM": st.column_config.NumberColumn(
            "Prem/EM", format="%.2f", width=75,
            help="Premium ÷ (spot × IV × √(DTE/365)) — how much of the "
                 "option's own theoretical 1-sigma expected move you're "
                 "being paid. >1 means premium exceeds the modeled move."),
        "OI":    st.column_config.NumberColumn("OI", format="%d",
                                               width=65, help=OI_HELP),
        "Vol":   st.column_config.NumberColumn("Vol", format="%d",
                                               width=65,
                                               help=vol_help_for(min_vol)),
    })
    if show_mc:
        col_cfg.update({
            label: st.column_config.TextColumn(label, width=90, help=help_text)
            for label, help_text in mc_columns.mc_extra_help().items()
        })
    if kind != "IV+pp":
        _, fmt = iv_scores.display_for(kind)
        col_cfg[kind] = st.column_config.NumberColumn(
            kind, format=fmt, width=85,
            help="Active ranking score — the chain is ranked by this "
                 "column. IV+pp shown alongside for context.")
    if roll_close_cost is not None:
        col_cfg["NetCr"] = st.column_config.NumberColumn("Net Credit",
                                                         format="$%+.2f",
                                                         width=85)

    st.dataframe(styled, column_config=col_cfg, hide_index=True,
                 width="stretch")
    if _has_warn:
        st.caption(EARNINGS_WARN_LEGEND)
    stamp_caption()


def show_scan_results(df: pd.DataFrame, mode: str, buy: bool,
                      roll_close_cost: float | None,
                      min_oi: int, top_n: int,
                      min_vol: int = 0,
                      min_ivpp: float | None = None,
                      min_ann: float | None = None,
                      min_percentile: float | None = None,
                      min_ann_delta_percentile: float | None = None,
                      show_mc: bool = False) -> None:
    """Filter, rank, and render the top-N per option type.

    Splits the chain by `mode` ("call", "put", or "both"), sorts by
    signal_score (descending for sell mode, ascending for buy mode;
    defaults to iv_excess), applies the OI/Vol floors, takes the top
    N, and delegates to `show_df`. Adds a subheader when rendering
    both sides so the user knows which table is which.

    `min_ivpp`/`min_ann` are optional floors on IV+pp (pp) and Ann%
    (%). `min_percentile`/`min_ann_delta_percentile` are optional floors
    on the historical percentile columns (0-100) — "only show strikes
    historically rich for this ETF at this delta/DTE." All four default
    `None` (the UI default, left blank) meaning no filtering.
    """
    iv_asc = buy
    sort_col = "signal_score" if "signal_score" in df.columns else "iv_excess"
    type_labels = {"call": "Calls", "put": "Puts"}
    to_show = [mode] if mode in type_labels else list(type_labels.keys())

    for opt_type in to_show:
        sub = (
            df[df["type"] == opt_type]
            .sort_values([sort_col, "open_interest"], ascending=[iv_asc, False])
        )
        sub = sub[(sub["open_interest"] >= min_oi)
                  & (sub["volume"] >= min_vol)]
        if min_ivpp is not None:
            sub = sub[(sub["iv_excess"] * 100) >= min_ivpp]
        if min_ann is not None:
            sub = sub[sub["ann_yield_pct"] >= min_ann]
        if min_percentile is not None and "iv_percentile" in sub.columns:
            sub = sub[sub["iv_percentile"] >= min_percentile]
        if min_ann_delta_percentile is not None and "ann_delta_percentile" in sub.columns:
            sub = sub[sub["ann_delta_percentile"] >= min_ann_delta_percentile]
        sub = sub.head(top_n)
        sub = sub.copy()
        sub["_rank"] = range(1, len(sub) + 1)
        if len(to_show) > 1:
            st.subheader(type_labels[opt_type])
        show_df(sub, roll_close_cost, min_oi, min_vol,
                buy=buy, opt_type=opt_type, show_mc=show_mc)
