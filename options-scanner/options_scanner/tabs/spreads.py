"""Spreads tabs: Spreads (all 13 strategies), Directional, Neutral.

All three tabs share `_render_view` — the controls, scan handler, and
results display are identical; the only differences are the strategy
list, default filter values, and (for Neutral) the Max |Δ| slider.

The shared view also drives the Monte Carlo panel for any selected
spread row.

Note on naming: this module shares its short name with the top-level
`options_scanner.spreads` math module (where `scan_spreads` lives).
Imports always go through the fully-qualified `options_scanner.X`
path, so there's no ambiguity.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from options_scanner.display.payoff_chart import show_payoff_chart
from options_scanner.display.scan_stamp import stamp_caption
from options_scanner.display.spot_meta import (
    fetch_spot_meta,
    spot_help_text,
    spot_value_html,
)
from options_scanner import chain_cache
from options_scanner.fetch import fetch_and_enrich_cached
from options_scanner.format import fmt_strike
from options_scanner.mc_ui import LegSpec, position_from_legs, render_mc_panel
from options_scanner.ui_theme import (
    metric_card, render_schwab_reauth_hint, section_header,
)


_GREEK_HELP = {
    "Δ": "Net delta — directional exposure. Near 0 = delta-neutral.",
    "θ": "Net daily theta — time decay earned (positive) or paid (negative) per day.",
    "ν": "Net vega — profit/loss per 1-point rise in IV. Positive = benefits from IV expansion.",
}

# Cap displayed rows per strategy so pandas Styler stays under Streamlit's
# 262_144-cell render limit. Results are pre-sorted by the user's chosen
# ranking column, so the top N is always the best N.
_MAX_DISPLAY_ROWS = 100


def _show_spreads_table(
    sub: pd.DataFrame,
    strategy_name: str,
    spot: float,
    key_prefix: str = "sp",
) -> tuple[int | None, pd.DataFrame]:
    """Render the ranked spread table.

    Returns `(selected_row_index, displayed_sub)` — the index is relative to
    `displayed_sub`, which may be a `head()` of the original `sub` when
    results exceed `_MAX_DISPLAY_ROWS`. Caller must index back into
    `displayed_sub`, not the original frame.
    """
    if sub.empty:
        st.info(f"No {strategy_name} spreads found matching the filters.")
        return None, sub

    # Cap displayed rows — Styler has a per-render cell ceiling, and large
    # result sets crash the dataframe widget otherwise.
    total = len(sub)
    truncated = total > _MAX_DISPLAY_ROWS
    if truncated:
        sub = sub.head(_MAX_DISPLAY_ROWS).reset_index(drop=True)

    # Disclaimer captions
    if strategy_name == "Calendar / Diagonal":
        st.caption("⚠ Profit estimate assumes constant IV — actual P&L depends "
                   "on IV changes in the back month.")
    elif strategy_name == "Ratio Spread (1×2)":
        st.caption("⚠ Max loss is capped at 5× spread width for ranking — "
                   "actual loss is theoretically unlimited above the upper breakeven.")

    has_two_sides = strategy_name in ("Iron Condor", "Iron Butterfly")

    disp_rows = []
    for _, r in sub.iterrows():
        row_d = {
            "Expiration": r["expiration"],
            "DTE":        int(r["dte"]),
            "Short $":    fmt_strike(r['short_strike']),
            "Long $":     fmt_strike(r['long_strike']),
        }
        if has_two_sides:
            ss2 = r.get("short_strike2")
            ls2 = r.get("long_strike2")
            if ss2 and not pd.isna(ss2):
                row_d["Short $2"] = fmt_strike(ss2)
                row_d["Long $2"]  = fmt_strike(ls2)

        credit = float(r["net_credit"])
        row_d["Credit/Debit"] = credit
        row_d["Max Profit"]   = float(r["max_profit"])
        row_d["Max Loss"]     = float(r["max_loss"])
        row_d["R/R"]          = float(r["risk_reward"])
        row_d["POP%"]         = float(r["pop"]) * 100
        row_d["EV"]           = float(r["expected_value"])
        row_d["Ann%"]         = float(r["ann_yield_pct"])
        row_d["BE Move%"]     = float(r["be_move_pct"])
        row_d["Δ"]            = float(r["net_delta"])
        row_d["θ"]            = float(r["net_theta"])
        row_d["ν"]            = float(r["net_vega"])
        row_d["IV+pp"]        = float(r["short_iv_excess"]) * 100
        row_d["Earnings"]     = "⚠" if r.get("earnings_in_window") else ""
        disp_rows.append(row_d)

    disp = pd.DataFrame(disp_rows)

    # Row styling: precompute a 2-D style matrix once and apply with
    # axis=None. Avoids the per-row callback path that pandas uses for
    # axis=1, which dominates render time on large tables.
    n_rows = len(sub)
    row_bgs: list[str] = []
    for i in range(n_rows):
        orig = sub.iloc[i]
        pt = bool(orig["positive_theta"])
        pv = bool(orig["positive_vega"])
        pop = float(orig["pop"])
        rr = float(orig["risk_reward"])
        if pt and pv:
            bg = "background-color: rgba(34,197,94,0.30); outline: 2px solid #16a34a"
        elif pop >= 0.65 and rr >= 0.20:
            bg = "background-color: rgba(34,197,94,0.18)"
        elif pop >= 0.55 and rr >= 0.10:
            bg = "background-color: rgba(234,179,8,0.22)"
        else:
            bg = ""
        row_bgs.append(bg)

    earnings_mask = [bool(sub.iloc[i].get("earnings_in_window", False))
                     for i in range(n_rows)]

    style_matrix = pd.DataFrame("", index=disp.index, columns=disp.columns)
    for i, bg in enumerate(row_bgs):
        if bg:
            style_matrix.iloc[i, :] = bg
    if "Earnings" in disp.columns:
        for i, has_earn in enumerate(earnings_mask):
            if has_earn:
                style_matrix.at[i, "Earnings"] = (
                    "background-color: rgba(249,115,22,0.35)"
                )

    styled = disp.style.apply(lambda _: style_matrix, axis=None)

    col_cfg = {
        "DTE":        st.column_config.NumberColumn("DTE", format="%d", width="small"),
        "Credit/Debit": st.column_config.NumberColumn("Credit/Debit", format="$%+.2f"),
        "Max Profit": st.column_config.NumberColumn("Max Profit", format="$%.2f"),
        "Max Loss":   st.column_config.NumberColumn("Max Loss", format="$%.2f"),
        "R/R":        st.column_config.NumberColumn("R/R", format="%.2f",
                                                     help="max_profit / max_loss — higher is better"),
        "POP%":       st.column_config.NumberColumn("POP%", format="%.1f%%",
                                                     help="Probability of profit at expiration"),
        "EV":         st.column_config.NumberColumn("EV", format="$%+.2f",
                                                     help="Expected value = POP×MaxProfit − (1−POP)×MaxLoss"),
        "Ann%":       st.column_config.NumberColumn("Ann%", format="%.1f%%", width="small"),
        "BE Move%":   st.column_config.NumberColumn("BE Move%", format="%.1f%%",
                                                     help="How far spot must move to breach the lower breakeven"),
        "Δ":          st.column_config.NumberColumn("Δ", format="%.2f", width="small",
                                                     help=_GREEK_HELP["Δ"]),
        "θ":          st.column_config.NumberColumn("θ", format="%.4f", width="small",
                                                     help=_GREEK_HELP["θ"]),
        "ν":          st.column_config.NumberColumn("ν", format="%.3f", width="small",
                                                     help=_GREEK_HELP["ν"]),
        "IV+pp":      st.column_config.NumberColumn(
            "IV+pp", format="%+.1f pp", width="small",
            help=(
                "Short-leg IV residual vs the fitted surface. Spreads"
                " here are CREDIT-leaning — positive IV+pp on the short"
                " leg means you're collecting richer-than-fair premium"
                " on the side you sold. Look for +3 pp or higher."
            ),
        ),
        "Earnings":   st.column_config.TextColumn("Earn", width="small",
                                                   help="⚠ = earnings event before expiration"),
    }

    event = st.dataframe(
        styled,
        column_config=col_cfg,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"{key_prefix}_tbl_{strategy_name.replace(' ', '_').replace('/', '_').replace('×', 'x')}",
    )
    stamp_caption()
    if truncated:
        st.caption(
            f"Showing top {_MAX_DISPLAY_ROWS} of {total} total — "
            "tighten filters (POP, OI, width, |Δ|) to see different matches."
        )
    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    selected_idx = selected_rows[0] if selected_rows else None
    return selected_idx, sub


def _render_view(
    *,
    key_prefix: str,
    tab_label: str,
    available_strategies: list[str],
    default_strategies: list[str],
    default_min_dte: int,
    default_max_dte: int,
    default_min_pop_pct: int,
    default_sort_by: str,
    session_key: str,
    include_delta_filter: bool = False,
    default_max_abs_delta: float = 1.0,
) -> None:
    """Shared controls + scan + results rendering for all spread tabs."""
    from options_scanner.spreads import scan_spreads

    # ── Controls ──────────────────────────────────────────────────────────────
    with st.container(border=True):
        tc, _ = st.columns([1, 5])
        with tc:
            ticker = st.text_input("Ticker (ex: AAPL)", "", key=f"{key_prefix}_ticker")

    # Width-mode toggle determines $ vs % defaults dynamically
    width_mode_label = st.radio(
        "Width units", ["$", "% of spot"],
        horizontal=True, key=f"{key_prefix}_width_mode",
    )
    width_mode = "percent" if "%" in width_mode_label else "dollar"
    if width_mode == "percent":
        min_w_default, max_w_default = 0.5, 5.0
        min_w_step, max_w_step = 0.1, 0.5
        min_w_label = "Min Width (%)"
        max_w_label = "Max Width (%)"
    else:
        min_w_default, max_w_default = 5.0, 25.0
        min_w_step, max_w_step = 0.5, 1.0
        min_w_label = "Min Width ($)"
        max_w_label = "Max Width ($)"

    with st.container(border=True):
        d1, d2, w1, w2, oi_col = st.columns([1, 1, 1, 1, 1])
        with d1:
            min_dte = st.number_input("Min DTE", value=default_min_dte,
                                      min_value=1, key=f"{key_prefix}_min_dte")
        with d2:
            max_dte = st.number_input("Max DTE", value=default_max_dte,
                                      min_value=1, key=f"{key_prefix}_max_dte")
        with w1:
            min_width = st.number_input(min_w_label, value=min_w_default,
                                        min_value=0.1, step=min_w_step,
                                        key=f"{key_prefix}_min_width")
        with w2:
            max_width = st.number_input(max_w_label, value=max_w_default,
                                        min_value=0.1, step=max_w_step,
                                        key=f"{key_prefix}_max_width")
        with oi_col:
            min_oi = st.number_input("Min OI (each leg)", value=10,
                                     min_value=0, key=f"{key_prefix}_min_oi")

    with st.container(border=True):
        # Pre-filter the default list to the strategies actually available
        effective_default = [s for s in default_strategies if s in available_strategies]
        selected_strategies = st.multiselect(
            "Strategies to scan",
            options=available_strategies,
            default=effective_default,
            key=f"{key_prefix}_strategies",
        )

    # Delta-neutral slider (Neutral tab only)
    max_abs_delta = 1.0
    if include_delta_filter:
        max_abs_delta = st.slider(
            "Max |Δ| (delta-neutrality)",
            min_value=0.05, max_value=1.00,
            value=default_max_abs_delta, step=0.05,
            key=f"{key_prefix}_max_delta",
            help="Tighter values = more delta-neutral. 0.15 ≈ minimal "
                 "directional bias. 1.00 disables the filter.",
        )

    f1, f2, f3, f4, _, f5 = st.columns([2, 1, 1, 1, 1, 1.2], vertical_alignment="bottom")
    with f1:
        pop_range = st.slider(
            "POP % range",
            min_value=0, max_value=100,
            value=(default_min_pop_pct, 100),
            step=5,
            key=f"{key_prefix}_pop_range",
            help="Filter spreads whose probability of profit falls within "
                 "this range. Drag the right handle to exclude near-certain "
                 "trades; drag the left handle to set a minimum probability.",
        )
        min_pop_pct, max_pop_pct = pop_range
    with f2:
        sort_by = st.selectbox("Sort by",
                               ["Risk/Reward", "POP", "Expected Value", "Ann%"],
                               index=["Risk/Reward", "POP", "Expected Value", "Ann%"].index(default_sort_by),
                               key=f"{key_prefix}_sort_by")
    with f3:
        only_pos_theta = st.checkbox("θ > 0 only", key=f"{key_prefix}_pos_theta")
    with f4:
        only_pos_vega = st.checkbox("ν > 0 only", key=f"{key_prefix}_pos_vega")
    with f5:
        scanned = st.button(f"Scan {tab_label}", type="primary",
                            width='stretch',
                            key=f"{key_prefix}_scan_btn")

    # ── Scan ──────────────────────────────────────────────────────────────────
    # Also fires when the floating rescan button below was clicked on the
    # previous run (it sets `_{key_prefix}_rescan_trigger` and calls
    # st.rerun()).
    rescan_flag = f"_{key_prefix}_rescan_trigger"

    # ── Auto-populate from a same-day background scan ──────────────────────────
    # Zero-click: if a background pass already covers this ticker+DTE
    # window, run the (cheap, local) spread-building step against it
    # immediately instead of waiting on a live fetch.
    _auto_ticker = ticker.strip().upper()
    _auto_populate = (
        not scanned
        and not st.session_state.get(rescan_flag)
        and bool(_auto_ticker)
        and st.session_state.get(session_key, {}).get("ticker") != _auto_ticker
        and chain_cache.has_fresh_snapshot(
            _auto_ticker, int(min_dte), int(max_dte), "yahoo-headless")
    )
    if _auto_populate:
        st.session_state[rescan_flag] = True

    if scanned or st.session_state.pop(rescan_flag, False):
        ticker_clean = ticker.strip().upper()
        if not ticker_clean:
            st.error("Enter a ticker symbol.")
            st.session_state.pop(session_key, None)
            return
        if not selected_strategies:
            st.error("Select at least one strategy.")
            return

        if int(max_dte) < int(min_dte):
            st.error(
                f"Max DTE ({int(max_dte)}) must be ≥ Min DTE "
                f"({int(min_dte)})."
            )
            st.session_state.pop(session_key, None)
            return

        _was_auto_populate = _auto_populate
        _fetch_provider = ("yahoo-headless" if _was_auto_populate
                           else st.session_state.get("data_source", "yahoo"))
        _spinner_msg = (f"Loading this morning's {ticker_clean} scan…"
                        if _was_auto_populate
                        else f"Fetching {ticker_clean} option chain…")
        with st.spinner(_spinner_msg):
            df, earnings_dates, err, from_cache, fetched_at = fetch_and_enrich_cached(
                ticker_clean, "both", int(min_dte), int(max_dte),
                _fetch_provider,
                st.session_state.get("schwab_config"),
                moomoo_config=st.session_state.get("moomoo_config"),
                force_live=False,
            )

        if err:
            st.error(err)
            _scfg = st.session_state.get("schwab_config") or {}
            render_schwab_reauth_hint(st.session_state.get("data_source", "yahoo"),
                                      key="schwab_reauth_spreads",
                                      token_file=_scfg.get("token_file"))
            st.session_state.pop(session_key, None)
            return
        if df.empty:
            st.warning(f"No options found for {ticker_clean}.")
            st.session_state.pop(session_key, None)
            return

        with st.spinner("Building spreads…"):
            results_df, errors = scan_spreads(
                df,
                strategies=selected_strategies,
                min_dte=int(min_dte),
                max_dte=int(max_dte),
                min_width=float(min_width),
                max_width=float(max_width),
                min_oi=int(min_oi),
                min_pop=min_pop_pct / 100.0,
                max_pop=max_pop_pct / 100.0,
                sort_by=sort_by,
                only_positive_theta=only_pos_theta,
                only_positive_vega=only_pos_vega,
                earnings_dates=earnings_dates,
                max_abs_delta=max_abs_delta,
                width_mode=width_mode,
            )

        st.session_state["scan_ts"] = fetched_at if from_cache else datetime.now().astimezone()
        st.session_state["scan_provider"] = _fetch_provider
        st.session_state["scan_from_cache"] = from_cache
        st.session_state[session_key] = {
            "ticker": ticker_clean,
            "spot": float(df["spot"].iloc[0]),
            "earnings_dates": earnings_dates,
            "df": results_df,
            "errors": errors,
            "selected_strategies": selected_strategies,
            "min_pop_pct": min_pop_pct,
            "max_pop_pct": max_pop_pct,
            "max_abs_delta": max_abs_delta,
        }

    # ── Display ───────────────────────────────────────────────────────────────
    res = st.session_state.get(session_key)
    if not res:
        return

    for err in res.get("errors", []):
        st.warning(f"Builder failed — {err}")

    spot = res["spot"]
    df_r = res["df"]
    ticker_r = res["ticker"]

    # Floating rescan button — same fixed-position treatment as the
    # Single Ticker tab. The shared `[class*="st-key-rescan_pill"]` CSS
    # block in the global style section pins this to the header bar.
    with st.container(key=f"rescan_pill_{key_prefix}"):
        if st.button(f"↻ Rescan {ticker_r}", type="primary",
                     key=f"{key_prefix}_rescan_btn"):
            st.session_state[rescan_flag] = True
            st.rerun()

    section_header(
        title=f"{ticker_r} — spread candidates",
        subtitle="Ranked by your chosen criterion, filtered by POP and width.",
        eyebrow="RESULTS",
    )
    m1, m2, m3 = st.columns(3)
    ed = res["earnings_dates"]
    if ed:
        earn_days = (ed[0] - date.today()).days
        earn_label = f"{ed[0].strftime('%b %d')}"
        earn_sub   = f"in {earn_days}d"
    else:
        earn_label = "—"
        earn_sub   = "no upcoming events"
    with m1:
        _meta = fetch_spot_meta(
            ticker_r, st.session_state.get("scan_provider", "yahoo"),
        )
        metric_card("SPOT PRICE",
                    spot_value_html(spot, _meta["pct_change"]),
                    help_text=spot_help_text(_meta))
    with m2:
        metric_card("SPREADS FOUND", f"{len(df_r):,}",
                    help_text="After all filters & sorting")
    with m3:
        metric_card("NEXT EARNINGS", earn_label,
                    delta=earn_sub, delta_sign="neutral")
    st.markdown(
        "<div style='margin:0.85rem 0 0.35rem 0;'></div>",
        unsafe_allow_html=True,
    )

    if df_r.empty:
        delta_hint = (f", |Δ| ≤ {res['max_abs_delta']:.2f}"
                      if include_delta_filter else "")
        st.info(
            f"No spreads met the filters "
            f"(POP {res['min_pop_pct']}%–{res.get('max_pop_pct', 100)}%"
            f"{delta_hint}). Try widening the POP range or spread width, "
            "or selecting more strategies."
        )
        return

    for strategy_name in res["selected_strategies"]:
        sub = df_r[df_r["strategy"] == strategy_name].reset_index(drop=True)
        n = len(sub)
        has_theta_vega = (sub["positive_theta"] & sub["positive_vega"]).any() if not sub.empty else False
        label = f"{strategy_name} — {n} spread(s)"
        if has_theta_vega:
            label += "  ⭐ θ+ν"

        with st.expander(label, expanded=True):
            if has_theta_vega:
                st.caption("⭐ **Green-bordered rows** = positive theta AND vega — "
                           "earns time decay and benefits from rising IV.")
            if strategy_name == "Risk Reversal":
                st.caption("⚠ Max loss assumes put assignment "
                           "(capital-at-risk = put strike − net credit). "
                           "Theoretical upside is unbounded; max profit is "
                           "capped at 3× max loss for ranking.")
            if strategy_name in ("Long Straddle", "Long Strangle"):
                st.caption("ℹ Max profit is capped at 3× debit for ranking — "
                           "actual upside is unbounded.")
            selected_idx, displayed_sub = _show_spreads_table(
                sub, strategy_name, spot, key_prefix=key_prefix
            )

            if (selected_idx is not None
                    and 0 <= selected_idx < len(displayed_sub)):
                row = displayed_sub.iloc[selected_idx]
                st.markdown("**Payoff diagram**")
                safe_strat = (strategy_name.replace(" ", "_")
                              .replace("/", "_").replace("×", "x"))
                show_payoff_chart(
                    row, spot, key_prefix=f"{key_prefix}_{safe_strat}"
                )

                # ── Monte Carlo for the selected multi-leg strategy ─────────
                from options_scanner.spreads import build_legs_from_row
                raw_legs = build_legs_from_row(row)
                if raw_legs:
                    try:
                        exp = pd.to_datetime(row["expiration"]).date()
                        legs_spec = [
                            LegSpec(
                                opt_type=lg["type"],
                                strike=float(lg["strike"]),
                                expiration=exp,
                                side="long" if int(lg["qty"]) > 0 else "short",
                                mid=float(lg.get("entry_mid", 0.0)),
                                iv=float(lg["iv"]) if lg.get("iv") else None,
                                qty=abs(int(lg["qty"])),
                            )
                            for lg in raw_legs
                        ]
                        spread_position = position_from_legs(
                            # ticker_r is the .strip().upper() snapshot from
                            # scan time; `ticker` is the live widget value
                            # that drifts if the user edits the input box.
                            underlying=ticker_r,
                            spot=spot,
                            legs_spec=legs_spec,
                            earnings_dates=(),
                            risk_free_rate=0.045,
                        )
                        st.markdown("**Monte Carlo P&L distribution**")
                        render_mc_panel(
                            spread_position,
                            key=f"{key_prefix}_mc_{strategy_name.replace(' ', '_')}_{selected_idx}",
                            label=f"{strategy_name} — {len(legs_spec)}-leg position",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.caption(f"_MC unavailable for this row: {exc}_")

    with st.expander("Column & Greek key"):
        st.markdown("""
**Spread columns**

| Column | Meaning |
|--------|---------|
| Credit/Debit | Net premium received (+) or paid (−) per share to enter the spread. |
| Max Profit | Maximum gain per share at the best possible outcome. |
| Max Loss | Maximum loss per share (capped at 5× width for Ratio spreads). |
| R/R | Risk-reward ratio: Max Profit ÷ Max Loss. Higher is better. |
| POP% | Probability of Profit at expiration (Black-Scholes N(d₂) based). |
| EV | Expected Value = POP × Max Profit − (1−POP) × Max Loss. Positive EV is statistically favorable. |
| Ann% | Annualized return on capital at risk if the spread reaches max profit. |
| BE Move% | How far the stock price must move from spot to breach the lower breakeven. |
| Δ | Net delta — directional bias of the spread. Near 0 = delta-neutral. |
| θ | Net daily theta — premium earned (positive) or paid (negative) per calendar day. |
| ν | Net vega — P&L change per 1-point rise in IV. Positive = long volatility. |
| IV+pp | IV excess of the short leg above the fitted surface — positive means rich premium. |
| Earn | ⚠ = an earnings event falls before this expiration. |

**Row highlights**

| Color | Meaning |
|-------|---------|
| Green border ⭐ | Positive theta AND positive vega — earns decay and benefits from IV expansion (common in calendars). |
| Green fill | POP ≥ 65% and R/R ≥ 0.20 — high-probability, reasonable reward. |
| Yellow fill | POP ≥ 55% and R/R ≥ 0.10 — moderate probability. |
| Orange Earn cell | Earnings before expiration — IV may spike unpredictably. |
""")


def tab_spreads() -> None:
    """Power-user view — all 13 spread strategies available."""
    from options_scanner.spreads import STRATEGY_NAMES
    _render_view(
        key_prefix="sp",
        tab_label="Spreads",
        available_strategies=STRATEGY_NAMES,
        default_strategies=["Bull Put Spread", "Bear Call Spread", "Iron Condor"],
        default_min_dte=21, default_max_dte=60,
        default_min_pop_pct=60,
        default_sort_by="Risk/Reward",
        session_key="spreads_results",
    )


def tab_directional() -> None:
    """Bullish / bearish strategies only."""
    from options_scanner.spreads import DIRECTIONAL_STRATEGIES
    _render_view(
        key_prefix="dir",
        tab_label="Directional",
        available_strategies=DIRECTIONAL_STRATEGIES,
        default_strategies=["Bull Put Spread", "Bear Call Spread"],
        default_min_dte=21, default_max_dte=60,
        default_min_pop_pct=60,
        default_sort_by="Risk/Reward",
        session_key="directional_results",
    )


def tab_neutral() -> None:
    """Range-bound / delta-neutral strategies with a Max |Δ| slider."""
    from options_scanner.spreads import NEUTRAL_STRATEGIES
    _render_view(
        key_prefix="nu",
        tab_label="Neutral",
        available_strategies=NEUTRAL_STRATEGIES,
        default_strategies=["Iron Condor", "Calendar / Diagonal", "Long Strangle"],
        default_min_dte=30, default_max_dte=180,
        default_min_pop_pct=55,
        default_sort_by="Expected Value",
        session_key="neutral_results",
        include_delta_filter=True,
        default_max_abs_delta=0.15,
    )
