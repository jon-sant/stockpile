"""Single-ticker tab: the scanner's primary workflow.

Fetches an option chain for one ticker, fits the IV surface, and
ranks contracts by IV excess. Supports two flows:

- **Find new options** — sell IV-rich or buy IV-cheap candidates,
  filtered by DTE, OI, volume, and a delta-range slider.
- **Roll an existing position** — same scan plus a NetCr column
  showing the credit/debit from rolling out of a user-supplied
  strike + expiration.

Results panel includes the IV chart, chain table for the chosen
expiration, the full "Top candidates" ranking, a Monte Carlo trade
analyzer, and an HTML report download. (Gamma exposure lives in its
own GEX tab.)
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from options_scanner.compute.top_ranks import compute_top_ranks
from options_scanner.defaults import default_delta_range
from options_scanner.display.chain_table import show_chain_table
from options_scanner.display.iv_chart import show_iv_chart
from options_scanner.display.outlook_card import render_outlook_card
from options_scanner.display.scan_results import show_scan_results
from options_scanner.display.surface_diagnostics import show_surface_diagnostics
from options_scanner.display.spot_meta import (
    fetch_spot_meta,
    spot_help_text,
    spot_value_html,
)
from options_scanner.fetch import fetch_and_enrich
from options_scanner.format import fmt_strike
from options_scanner import iv_algorithms, iv_history, iv_scores
from options_scanner.iv_filters import DEFAULT_CONFIG as FILTER_DEFAULT, SurfaceFilterConfig
from options_scanner.mc_ui import position_from_chain_row, render_mc_panel
from options_scanner.recent_scans import build_label, load as load_recent, save as save_recent
from options_scanner.ui_theme import (
    badge, empty_state, metric_card, render_schwab_reauth_hint, section_header,
)


# Surface-fit presets — shared by the preset pill and the advanced section
# (which renders below the Scan button in tab_single). Both presets use
# the default filters, which include exclude_earnings as of 2026-06.
_SF_PRESETS = {
    "Global": (
        FILTER_DEFAULT,
        ("global_poly", frozenset()),
        ("raw_pp", frozenset()),
    ),
    "Per-expiry": (
        FILTER_DEFAULT,
        ("per_expiration", frozenset({("weights", "inv_spread")})),
        ("zscore", frozenset()),
    ),
}


def _surface_fit_controls() -> str:
    """Render the Surface-fit preset pill; return the selected preset name.

    The advanced-override checkbox and expander render below the Scan
    button in tab_single() so they don't clutter the primary scan flow.
    """
    with st.container(key="surface_fit_pill"):
        preset = st.radio(
            "Fit:", list(_SF_PRESETS.keys()), horizontal=True,
            key="s_sf_preset",
            help="Surface fit. Global = one polynomial across the chain + "
                 "raw IV+pp. Per-expiry (LGbengs) = per-expiration "
                 "spread-weighted fit, ranked by z-score. Both exclude "
                 "short-dated (≤60 DTE) earnings-spanning options from the fit.",
        )
    return preset


def tab_single() -> None:
    # ── Group 1: Ticker + flow + Recent Scans ─────────────────────────────────
    _recent = load_recent()
    _placeholder = "— recent scans —"

    def _apply_recall() -> None:
        """on_change callback for the Recent Scans selectbox.

        Runs synchronously before Streamlit reruns, so session-state writes
        here ARE reflected in widgets on the very next render — no separate
        st.rerun() or pre-fill pass needed. Setting s_recent_choice back to
        the placeholder is allowed inside callbacks (unlike post-render).
        """
        choice = st.session_state.get("s_recent_choice", _placeholder)
        if choice == _placeholder:
            return
        # Locate the matching entry by its display label.
        entry = next((e for e in _recent if build_label(e) == choice), None)
        if entry is None:
            return
        _step = 0.05
        _dmin = round(round(float(entry.get("delta_min", 0.10)) / _step) * _step, 10)
        _dmax = round(round(float(entry.get("delta_max", 0.75)) / _step) * _step, 10)
        st.session_state["s_ticker"] = entry["ticker"]
        st.session_state["s_flow"]   = (
            "Roll an existing position"
            if entry.get("flow") == "roll"
            else "Find new options"
        )
        st.session_state["s_min_dte"] = max(1, int(entry.get("min_dte", 30)))
        st.session_state["s_max_dte"] = int(entry.get("max_dte", 90))
        st.session_state["s_min_oi"]  = int(entry.get("min_oi", 1))
        st.session_state["s_min_vol"] = int(entry.get("min_vol", 1))
        # Not persisted with recent scans — runtime-only, so reset to blank
        # rather than carry over a stale filter from a previous scan.
        st.session_state["s_min_ivpp"] = None
        st.session_state["s_min_ann"]  = None
        st.session_state["s_min_percentile"] = None
        st.session_state["s_min_ann_delta_percentile"] = None
        st.session_state["s_delta"]   = (_dmin, _dmax)
        st.session_state["s_top"]     = int(entry.get("top_n", 10))
        if entry.get("flow") == "roll":
            st.session_state["s_roll_type"]   = entry.get("roll_type", "call")
            st.session_state["s_roll_strike"]  = float(entry.get("roll_strike", 0.0))
            try:
                from datetime import datetime as _dt
                st.session_state["s_roll_exp"] = _dt.strptime(
                    entry["roll_exp"], "%Y-%m-%d"
                ).date()
            except (KeyError, ValueError):
                pass
        else:
            st.session_state["s_action"] = (
                "Buy (IV-cheap candidates)"
                if entry.get("buy")
                else "Sell (IV-rich candidates)"
            )
            st.session_state["s_opt_type"] = entry.get("option_type", "Calls")
        # Reset the dropdown to placeholder — allowed inside on_change.
        st.session_state["s_recent_choice"] = _placeholder

    with st.container(border=True):
        tc, fc, rc = st.columns([1, 2.5, 3.5])
        with tc:
            ticker = st.text_input("Ticker (ex: AAPL)", "", key="s_ticker")
        with fc:
            flow = st.radio(
                "What do you want to do?",
                ["Find new options", "Roll an existing position"],
                horizontal=True,
                key="s_flow",
            )
        with rc:
            _options = [_placeholder] + [build_label(e) for e in _recent]
            st.selectbox(
                "Recent Scans",
                _options,
                index=0,
                key="s_recent_choice",
                on_change=_apply_recall,
                label_visibility="visible",
            )
    rolling = (flow == "Roll an existing position")

    # Defaults so the same scan code path handles both flows
    buy            = False
    option_type    = "Calls"
    roll_type_sel  = "call"
    roll_strike    = 0.0
    roll_exp       = date.today()

    # ── Group 2: Action-specific controls ─────────────────────────────────────
    with st.container(border=True):
        if rolling:
            rc1, rc2, rc3, _ = st.columns([1, 1, 1.2, 3])
            with rc1:
                roll_type_sel = st.selectbox("Position type", ["call", "put"],
                                             key="s_roll_type")
            with rc2:
                roll_strike = st.number_input("Current strike", value=0.0,
                                              min_value=0.0, step=1.0,
                                              key="s_roll_strike")
            with rc3:
                roll_exp = st.date_input("Current expiration", key="s_roll_exp")
        else:
            def _sync_s_delta() -> None:
                """Reset the delta band to the new direction's default when the
                Sell/Buy toggle flips. A range nudged within a direction sticks;
                flipping direction re-seeds it (sell ≈ OTM, buy ≈ near-ATM)."""
                _b = st.session_state.get("s_action", "").startswith("Buy")
                st.session_state["s_delta"] = default_delta_range(_b)

            a1, a2, a3 = st.columns([2.2, 1.8, 3.0])
            with a1:
                action = st.radio(
                    "Direction",
                    ["Sell (IV-rich candidates)", "Buy (IV-cheap candidates)"],
                    horizontal=True,
                    key="s_action",
                    on_change=_sync_s_delta,
                )
                buy = action.startswith("Buy")
            with a2:
                option_type = st.radio("Option Type",
                                       ["Calls", "Puts", "Both"],
                                       horizontal=True, key="s_opt_type")
            with a3:
                render_outlook_card(buy, option_type)

    # ── Group 3: Filters ──────────────────────────────────────────────────────
    with st.container(border=True):
        n1, n2, n3, n4, n5, n6, n7, n8 = st.columns(
            [1, 1, 1, 1, 1, 1, 2, 1], vertical_alignment="top",
        )
        with n1:
            min_dte = st.number_input("Min DTE", value=30, min_value=1,
                                      key="s_min_dte")
        with n2:
            max_dte_inp = st.number_input("Max DTE", value=90, min_value=0,
                                          help="0 = no limit; otherwise ≥ Min DTE",
                                          key="s_max_dte")
        with n3:
            min_oi = st.number_input("Min OI", value=1, min_value=0,
                                     key="s_min_oi")
        with n4:
            min_vol = st.number_input(
                "Min Vol", value=1, min_value=0,
                key="s_min_vol",
            )
        with n5:
            min_ivpp = st.number_input(
                "Min IV+pp", value=None, step=0.5, format="%.1f",
                placeholder="none", key="s_min_ivpp",
                help="Floor on IV+pp (pp). Blank = no filter.",
            )
        with n6:
            min_ann = st.number_input(
                "Min Ann%", value=None, step=1.0, format="%.1f",
                placeholder="none", key="s_min_ann",
                help="Floor on Ann% (annualized yield). Blank = no filter.",
            )
        with n7:
            delta_range = st.slider("Delta Range (abs value)", 0.0, 1.0,
                                    default_delta_range(False), step=0.05,
                                    key="s_delta")
        with n8:
            top_n = st.number_input("Top N", value=10, min_value=1,
                                    max_value=50, key="s_top")

        p1, p2, _p3 = st.columns([1, 1, 4])
        with p1:
            min_percentile = st.number_input(
                "Min IV %ile", value=None, step=5.0, format="%.0f",
                min_value=0.0, max_value=100.0,
                placeholder="none", key="s_min_percentile",
                help="Floor on IV %ile — how rich today's IV+pp is vs. this "
                     "ticker's own delta/DTE-bucketed history (0-100). "
                     "Blank = no filter.",
            )
        with p2:
            min_ann_delta_percentile = st.number_input(
                "Min Ann/Δ %ile", value=None, step=5.0, format="%.0f",
                min_value=0.0, max_value=100.0,
                placeholder="none", key="s_min_ann_delta_percentile",
                help="Floor on Ann%/Delta %ile — how rich today's yield-per-"
                     "delta is vs. this ticker's own bucketed history "
                     "(0-100). Blank = no filter.",
            )

    # ── Surface fit preset pill ───────────────────────────────────────────────
    preset = _surface_fit_controls()

    # ── Scan row: button + market hours warning ───────────────────────────────
    with st.container(key="scan_mh_row"):
        s_scan, s_mh = st.columns([1, 5], vertical_alignment="center")
        with s_scan:
            with st.container(key="scan_btn_lift"):
                scanned = st.button("Scan", type="primary",
                                    width='stretch', key="s_scan_btn")
        with s_mh:
            st.markdown(
                "<div style='display:flex; align-items:center; "
                "gap:0.75rem; padding-left:1rem;'>"
                + badge("MARKET HOURS RECOMMENDED", "warn")
                + "<span style='color:var(--osc-destructive); font-weight:700; "
                "font-size:0.78rem;'>"
                "Pre/post-market quotes may be stale or missing.</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Advanced surface fit (below scan row) ─────────────────────────────────
    advanced = st.checkbox(
        "Advanced surface fit — override the top-bar preset", value=False,
        key="s_sf_advanced",
        help="Compose the filter / algorithm / score stages by hand "
             "instead of using the title-bar preset.",
    )

    if not advanced:
        surface_filter_config, algo_config, score_config = _SF_PRESETS[preset]
    else:
        with st.expander("Advanced surface fit", expanded=True):
            st.caption(
                "Filters choose which options feed the regression; the "
                "algorithm fits the surface; the score is the number the "
                "chain is ranked by. All options still appear in the table."
            )

            # ── Filters ──────────────────────────────────────────────────────
            sf1, sf2 = st.columns([1, 2])
            with sf1:
                sf_otm = st.checkbox("OTM only", value=True, key="s_sf_otm",
                                     help="Calls K > S, puts K < S.")
                sf_use_spread = st.checkbox("Spread filter", value=True,
                                            key="s_sf_use_spread")
                sf_use_delta = st.checkbox("Delta range", value=True,
                                           key="s_sf_use_delta")
                sf_use_min_oi = st.checkbox("Min OI for fit", value=False,
                                            key="s_sf_use_min_oi")
                sf_excl_earn = st.checkbox("Exclude earnings", value=True,
                                           key="s_sf_excl_earn",
                                           help="Drop short-dated (≤60 DTE) "
                                                "options spanning the next "
                                                "earnings from the fit — their "
                                                "IV premium is legitimate event "
                                                "risk. Longer-dated contracts "
                                                "stay in.")
                sf_fresh = st.checkbox("Fresh quotes only", value=False,
                                       key="s_sf_fresh",
                                       help="Drop contracts that haven't "
                                            "traded in days and show no "
                                            "volume today — their broker IV "
                                            "is likely stale. Yahoo only; "
                                            "Schwab/Moomoo rows pass "
                                            "through.")
            with sf2:
                sf_spread_pct = st.number_input(
                    "Max spread % of mid", value=50, min_value=1, max_value=200,
                    step=1, key="s_sf_spread_pct", disabled=not sf_use_spread,
                )
                _dcols = st.columns(2)
                sf_delta_lo = _dcols[0].number_input(
                    "Min |Δ|", value=0.10, min_value=0.0, max_value=0.49,
                    step=0.01, format="%.2f", key="s_sf_delta_lo",
                    disabled=not sf_use_delta,
                )
                sf_delta_hi = _dcols[1].number_input(
                    "Max |Δ|", value=0.95, min_value=0.51, max_value=1.0,
                    step=0.01, format="%.2f", key="s_sf_delta_hi",
                    disabled=not sf_use_delta,
                )
                sf_min_oi_val = st.number_input(
                    "Min OI", value=1, min_value=1, key="s_sf_min_oi_val",
                    disabled=not sf_use_min_oi,
                )

            _sf: list[tuple[str, frozenset]] = []
            if sf_otm:
                _sf.append(("otm_only", frozenset()))
            if sf_use_spread:
                _sf.append(("spread_pct",
                            frozenset({("max_pct", sf_spread_pct / 100)})))
            if sf_use_delta:
                _sf.append(("delta_range", frozenset({("lo", sf_delta_lo),
                                                      ("hi", sf_delta_hi)})))
            if sf_use_min_oi:
                _sf.append(("min_oi", frozenset({("min_oi", sf_min_oi_val)})))
            if sf_excl_earn:
                _sf.append(("exclude_earnings", frozenset()))
            if sf_fresh:
                _sf.append(("fresh_quotes", frozenset()))
            surface_filter_config: SurfaceFilterConfig = tuple(_sf)

            # ── Algorithm + weighting ────────────────────────────────────────
            a1, a2 = st.columns(2)
            algo_names = list(iv_algorithms.REGISTRY.keys())
            algo_name = a1.selectbox(
                "Algorithm", algo_names,
                index=algo_names.index("global_poly"),
                format_func=lambda n: iv_algorithms.REGISTRY[n]["label"],
                key="s_sf_algo",
            )
            if not iv_algorithms.REGISTRY[algo_name].get("enabled", True):
                st.info(f"{iv_algorithms.REGISTRY[algo_name]['label']} isn't "
                        "available yet — using the global polynomial.")
                algo_name = "global_poly"
            weights = a2.selectbox(
                "Fit weighting", ["none", "oi", "inv_spread", "vega"],
                format_func={"none": "Equal", "oi": "By open interest",
                             "inv_spread": "By 1 / spread",
                             "vega": "By vega"}.get,
                key="s_sf_weights",
            )
            robust = st.selectbox(
                "Robust fit (outlier handling)",
                ["none", "huber", "tukey"],
                format_func={"none": "Off — plain least squares",
                             "huber": "Huber — downweight outliers",
                             "tukey": "Tukey — reject far outliers"}.get,
                key="s_sf_robust",
                help="Re-fits the surface a few times, reducing the "
                     "influence of contracts far from it — so a stale "
                     "high-IV print can't drag the surface toward "
                     "itself. Huber softens outliers; Tukey ignores "
                     "extreme ones entirely.",
            )
            algo_config = (algo_name, frozenset({("weights", weights),
                                                 ("robust", robust)}))

            # ── Score ────────────────────────────────────────────────────────
            score_names = list(iv_scores.REGISTRY.keys())
            score_name = st.selectbox(
                "Score (ranking key)", score_names,
                index=score_names.index("raw_pp"),
                format_func=lambda n: iv_scores.REGISTRY[n]["label"],
                key="s_sf_score",
            )
            if not iv_scores.REGISTRY[score_name].get("enabled", True):
                st.info(f"{iv_scores.REGISTRY[score_name]['label']} isn't "
                        "available yet — using raw IV+pp.")
                score_name = "raw_pp"
            score_config = (score_name, frozenset())

    # ── Run scan on button click, store in session state ──────────────────────
    # Also triggers when the sticky "Rescan" pill below the results was
    # clicked on the previous run — it sets `_rescan_trigger` and calls
    # st.rerun() so this top-of-script handler picks it up.
    if scanned or st.session_state.pop("_rescan_trigger", False):
        ticker_clean = ticker.strip().upper()
        if not ticker_clean:
            st.error("Enter a ticker symbol.")
            st.session_state.pop("single_results", None)
            return

        if 0 < int(max_dte_inp) < int(min_dte):
            st.error(
                f"Max DTE ({int(max_dte_inp)}) must be ≥ Min DTE "
                f"({int(min_dte)}), or 0 for no limit."
            )
            st.session_state.pop("single_results", None)
            return

        if rolling:
            eff_opt_fetch = roll_type_sel + "s"   # "calls" or "puts"
            eff_mode      = roll_type_sel          # "call"  or "put"
        else:
            opt_map  = {"Calls": "calls", "Puts": "puts", "Both": "both"}
            mode_map = {"Calls": "call",  "Puts": "put",  "Both": "both"}
            eff_opt_fetch = opt_map[option_type]
            eff_mode      = mode_map[option_type]

        max_dte_arg = int(max_dte_inp) if max_dte_inp > 0 else None
        delta_min, delta_max = delta_range

        with st.spinner(f"Fetching {ticker_clean} option chain…"):
            df, earnings_dates, err = fetch_and_enrich(
                ticker_clean, eff_opt_fetch, int(min_dte), max_dte_arg,
                st.session_state.get("data_source", "yahoo"),
                st.session_state.get("schwab_config"),
                surface_filter_config, algo_config, score_config,
                moomoo_config=st.session_state.get("moomoo_config"),
            )

        if err:
            st.error(f"**{ticker_clean}:** {err}")
            _scfg = st.session_state.get("schwab_config") or {}
            render_schwab_reauth_hint(st.session_state.get("data_source", "yahoo"),
                                      key="schwab_reauth_single",
                                      token_file=_scfg.get("token_file"))
            st.session_state.pop("single_results", None)
            return
        if df.empty:
            st.warning(
                f"**{ticker_clean}:** No options found for DTE {int(min_dte)}–"
                f"{int(max_dte_inp) if max_dte_inp else '∞'}. "
                "Try widening the DTE range or check that the ticker has listed options."
            )
            st.session_state.pop("single_results", None)
            return

        # Roll: look up close cost for the existing position
        roll_close_cost = None
        if rolling and roll_strike > 0:
            exp_yf = roll_exp.strftime("%Y-%m-%d")
            _provider = st.session_state.get("data_source", "yahoo")
            _scfg = st.session_state.get("schwab_config")
            with st.spinner("Looking up close cost…"):
                if _provider == "schwab":
                    from stocks_shared.schwab_live import (
                        get_client, fetch_option_chain_schwab
                    )
                    try:
                        _sclient = get_client(
                            _scfg["app_key"], _scfg["app_secret"],
                            _scfg["callback_url"], _scfg["token_file"],
                        )
                        chain = fetch_option_chain_schwab(
                            _sclient, ticker_clean, exp_yf
                        )
                    except ValueError as exc:
                        st.warning(f"Schwab roll lookup failed: {exc}")
                        chain = None
                else:
                    from stocks_shared.yahoo import fetch_option_chain
                    chain = fetch_option_chain(ticker_clean, exp_yf)
            if chain is not None:
                side_df = chain.calls if roll_type_sel == "call" else chain.puts
                row = side_df[side_df["strike"] == float(roll_strike)]
                if not row.empty:
                    bid  = float(row["bid"].iloc[0] or 0)
                    ask  = float(row["ask"].iloc[0] or 0)
                    last = float(row["lastPrice"].iloc[0] or 0)
                    roll_close_cost = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                else:
                    st.warning("Position not found in chain — NetCr column omitted.")
            else:
                st.warning(f"Could not fetch chain for {exp_yf} — NetCr column omitted.")

        st.session_state["scan_ts"] = datetime.now().astimezone()
        st.session_state["scan_provider"] = st.session_state.get(
            "data_source", "yahoo"
        )
        _adv = st.session_state.get("s_sf_advanced", False)
        if not _adv:
            st.session_state["scan_surface_label"] = st.session_state.get(
                "s_sf_preset", "Global"
            )
        else:
            _algo = st.session_state.get("s_sf_algo", "global_poly")
            _score = st.session_state.get("s_sf_score", "raw_pp")
            _algo_s = {"global_poly": "global poly",
                       "per_expiration": "per-expiry poly"}.get(_algo, _algo)
            _score_s = {"raw_pp": "raw IV+pp", "zscore": "z-score",
                        "relative": "relative", "composite": "composite",
                        "vrp": "VRP", "percentile": "percentile"}.get(
                            _score, _score)
            st.session_state["scan_surface_label"] = (
                f"Advanced · {_algo_s} · {_score_s}"
            )
        st.session_state["single_results"] = {
            "ticker": ticker_clean,
            "df": df,
            "earnings_dates": earnings_dates,
            "mode": eff_mode,
            "buy": buy,
            "roll_close_cost": roll_close_cost,
            "delta_min": delta_min,
            "delta_max": delta_max,
            "min_dte": int(min_dte),
            "max_dte": int(max_dte_inp),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
            "min_ivpp": min_ivpp,
            "min_ann": min_ann,
            "min_percentile": min_percentile,
            "min_ann_delta_percentile": min_ann_delta_percentile,
            "top_n": int(top_n),
            "roll_exp_str": roll_exp.strftime("%Y-%m-%d") if rolling else None,
            "roll_strike": roll_strike if rolling else None,
            "roll_type": roll_type_sel if rolling else None,
            "surface_filters": surface_filter_config,
            "algo_config": algo_config,
        }

        # Persist the scan parameters for the Recent Scans dropdown.
        # Shared filters are saved for both flows so they can be fully
        # restored on recall and so the dedup key distinguishes scans
        # that differ only in OI/vol/DTE.
        _shared = {
            "min_dte":   int(min_dte),
            "max_dte":   int(max_dte_inp),
            "min_oi":    int(min_oi),
            "min_vol":   int(min_vol),
            "delta_min": delta_min,
            "delta_max": delta_max,
            "top_n":     int(top_n),
        }
        if rolling:
            _entry = {
                "flow":        "roll",
                "ticker":      ticker_clean,
                "roll_type":   roll_type_sel,
                "roll_strike": roll_strike,
                "roll_exp":    roll_exp.strftime("%Y-%m-%d"),
                **_shared,
            }
        else:
            _entry = {
                "flow":        "find",
                "ticker":      ticker_clean,
                "buy":         buy,
                "option_type": option_type,
                "min_dte":     int(min_dte),
                "max_dte":     int(max_dte_inp),
                "min_oi":      int(min_oi),
                "min_vol":     int(min_vol),
                "delta_min":   delta_min,
                "delta_max":   delta_max,
                "top_n":       int(top_n),
            }
        _entry["label"] = build_label(_entry)
        save_recent(_entry)

    # ── Display results (persists across re-runs until next scan) ─────────────
    res = st.session_state.get("single_results")
    if not res:
        return

    ticker_r  = res["ticker"]
    mode_r    = res["mode"]
    buy_r     = res["buy"]
    rcc       = res["roll_close_cost"]
    # res["df"] is the full two-sided chain the surface was fit on (both
    # wings anchor the fit). Display/scoring (df_r → df_filt) uses only the
    # requested side; the diagnostics, GEX, and chart df_full get the full
    # fit set so they report what actually anchored the surface.
    df_fit_full = res["df"]
    df_r = (df_fit_full[df_fit_full["type"] == mode_r].reset_index(drop=True)
            if mode_r in ("call", "put") else df_fit_full)
    df_filt   = df_r[df_r["delta"].abs().between(
                    res["delta_min"], res["delta_max"])].copy()
    spot      = float(df_fit_full["spot"].iloc[0])

    st.markdown(
        "<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True,
    )
    section_header(
        title=f"{ticker_r} — scan results",
        subtitle="Spot, available expirations, and the next earnings event.",
        eyebrow="SUMMARY",
    )
    m1, m2, m3, m4 = st.columns(4)
    ed = res["earnings_dates"]
    if ed:
        earn_days = (ed[0] - date.today()).days
        earn_label = f"{ed[0].strftime('%b %d')}"
        earn_sub   = f"in {earn_days}d"
    else:
        earn_label = "—"
        earn_sub   = "no upcoming events"
    n_contracts = int(len(df_filt))
    action_lbl = "Find new" if not res["roll_close_cost"] else "Roll"
    direction_lbl = "BUY" if buy_r else "SELL"
    with m1:
        _meta = fetch_spot_meta(
            ticker_r, st.session_state.get("scan_provider", "yahoo"),
        )
        metric_card("SPOT PRICE",
                    spot_value_html(spot, _meta["pct_change"]),
                    help_text=spot_help_text(_meta))
    with m2:
        metric_card("EXPIRATIONS", f"{df_r['expiration'].nunique()}",
                    help_text=f"{n_contracts} contracts after filters")
    with m3:
        metric_card("NEXT EARNINGS", earn_label,
                    delta=earn_sub, delta_sign="neutral")
    with m4:
        metric_card("ACTION", f"{action_lbl}",
                    delta=f"{direction_lbl} · {mode_r.upper()}",
                    delta_sign="neutral")
    st.markdown(
        "<div style='margin:0.85rem 0 0.35rem 0;'></div>",
        unsafe_allow_html=True,
    )

    if rcc is not None:
        st.info(f"Rolling {res['roll_type']} {fmt_strike(res['roll_strike'])} "
                f"{res['roll_exp_str']} — close cost (mid): **${rcc:.2f}**")

    # Rescan button (fixed to header bar) + scan-criteria summary on
    # the same row. The button container is position:fixed via CSS so
    # it lifts out of document flow; _btn_col becomes a spacer in the
    # page body, and _sum_col holds the criteria caption to its right.
    _min_dte = res.get("min_dte", "?")
    _max_dte = res.get("max_dte", 0)
    _dte_str = f"DTE {_min_dte}–{_max_dte}" if _max_dte else f"DTE ≥{_min_dte}"
    _delta_str = f"Δ {res['delta_min']:.2f}–{res['delta_max']:.2f}"
    if rcc is not None:
        _mode_str = f"{res.get('roll_type','').upper()} {fmt_strike(res.get('roll_strike',0))} exp {res.get('roll_exp_str','')}"
        _summary = (f"**Roll** · {_mode_str} · {_dte_str} · "
                    f"OI≥{res['min_oi']} · Vol≥{res.get('min_vol',0)} · {_delta_str}")
    else:
        _dir = "BUY" if buy_r else "SELL"
        _type = {"call": "Calls", "put": "Puts", "both": "Both"}.get(mode_r, mode_r)
        _summary = (f"**{_type} · {_dir}** · {_dte_str} · "
                    f"OI≥{res['min_oi']} · Vol≥{res.get('min_vol',0)} · "
                    f"{_delta_str} · Top {res['top_n']}")

    _btn_col, _sum_col = st.columns([1, 5])
    with _btn_col:
        with st.container(key="rescan_pill_single"):
            if st.button(f"↻ Rescan {ticker_r}", type="primary",
                         key="s_rescan_btn"):
                st.session_state["_rescan_trigger"] = True
                st.rerun()
    with _sum_col:
        st.caption(_summary)

    show_iv_chart(df_filt, spot, mode_r, res["min_oi"], res["top_n"],
                   buy_r, ticker=ticker_r, key_prefix="s",
                   min_vol=res.get("min_vol", 0),
                   provider=st.session_state.get("scan_provider", "yahoo"),
                   earnings_dates=res.get("earnings_dates"),
                   surface_filters=res.get("surface_filters"),
                   df_full=df_fit_full,
                   delta_range=(res["delta_min"], res["delta_max"]))

    show_surface_diagnostics(df_fit_full, res.get("surface_filters"),
                             res.get("algo_config"))

    with st.expander(f"Scan history — {ticker_r}"):
        hist = iv_history.history_for(ticker_r)
        if hist.empty:
            st.caption(
                f"No recorded scan history yet for {ticker_r} — it "
                "accumulates one snapshot per day this ticker is scanned. "
                "The percentile score also needs 30+ pooled observations "
                "before it stops returning blank."
            )
        else:
            st.caption(
                f"{len(hist)} contract snapshot(s) across "
                f"{hist['scan_date'].nunique()} scan day(s), trailing 30 days."
            )
            daily = hist.groupby("scan_date")["iv_excess"].mean() * 100
            st.line_chart(daily, y_label="Mean IV+pp (%)")
            _disp = hist.copy()
            _disp["strike"] = _disp["strike"].apply(fmt_strike)
            _disp["iv_excess"] = (_disp["iv_excess"] * 100).round(1)
            st.dataframe(
                _disp.rename(columns={
                    "scan_date": "Scan Date", "type": "Type",
                    "strike": "Strike", "expiration": "Expiration",
                    "dte": "DTE", "iv_excess": "IV+pp",
                }),
                hide_index=True, width="stretch",
            )

    chosen_exp = st.session_state.get("s_chart_exp")
    if chosen_exp:
        df_chain = df_filt[df_filt["expiration"] == chosen_exp].copy()
        exp_lbl  = datetime.strptime(chosen_exp, "%Y-%m-%d").strftime("%b %d '%y")
        exp_date = datetime.strptime(chosen_exp, "%Y-%m-%d").date()
        earn_before = [d for d in res["earnings_dates"]
                       if date.today() < d <= exp_date]
        if earn_before:
            next_earn   = min(earn_before)
            earn_days   = (next_earn - date.today()).days
            earn_lbl    = next_earn.strftime("%b %d")
            chain_title = f"{exp_lbl} — next earnings {earn_lbl} ({earn_days}d)"
        else:
            chain_title = exp_lbl
        st.subheader(chain_title)
        top_ranks = compute_top_ranks(
            df_filt, mode_r, buy_r, res["min_oi"], res["top_n"],
            res.get("min_vol", 0),
        )
        show_chain_table(df_chain, buy_r, mode_r, rcc, res["min_oi"],
                          res.get("min_vol", 0), top_ranks=top_ranks)

    st.subheader("Top candidates — all chains")
    show_scan_results(df_filt, mode_r, buy_r, rcc,
                       res["min_oi"], res["top_n"],
                       res.get("min_vol", 0),
                       min_ivpp=res.get("min_ivpp"),
                       min_ann=res.get("min_ann"),
                       min_percentile=res.get("min_percentile"),
                       min_ann_delta_percentile=res.get("min_ann_delta_percentile"))

    # ── Monte Carlo trade analyzer ────────────────────────────────────────
    # Pick any candidate from the ranked table above and simulate its
    # full P&L distribution. Engine: 10k GBM paths with optional
    # earnings jumps. Pure NumPy — sub-second for typical 30-90 DTE.
    section_header(
        "Monte Carlo Trade Analyzer",
        eyebrow="DECISION SUPPORT",
        subtitle="Simulate the P&L distribution of any contract above. "
                 "P(profit), expected value, worst-5% CVaR, breakeven move.",
    )
    if df_filt.empty:
        empty_state(
            title="Nothing to analyze",
            subtitle="Run a scan to populate the candidate table, then pick "
                     "a row to simulate.",
        )
    else:
        # Apply the EXACT same filters and ranking the "Top candidates"
        # table uses, so the MC dropdown order matches the table order
        # row-for-row. show_scan_results does:
        #   1. filter to opt_type (or both)
        #   2. require open_interest >= min_oi AND volume >= min_vol
        #   3. sort by iv_excess (asc if buy / desc if sell), OI tie-break
        #   4. head(top_n)
        # Without these filters, the auto-filled top row could be a
        # low-liquidity option the table itself hides.
        if mode_r in ("call", "put"):
            df_mc_base = df_filt[df_filt["type"] == mode_r]
        else:
            df_mc_base = df_filt
        df_mc = (
            df_mc_base[
                (df_mc_base["open_interest"] >= res["min_oi"])
                & (df_mc_base["volume"] >= res.get("min_vol", 0))
            ]
            .sort_values(
                ["iv_excess", "open_interest"],
                ascending=[buy_r, False],
            )
            .head(res["top_n"])
            .reset_index(drop=True)
            .copy()
        )
        # Empty after filters → nothing to analyze. Surface the reason
        # explicitly rather than render an empty dropdown. Check BEFORE
        # the _label assignment below, since df.apply(..., axis=1) on an
        # empty frame returns an empty DataFrame (not Series), which then
        # crashes the single-column assignment.
        if df_mc.empty:
            empty_state(
                title="No candidates pass the table's filters",
                subtitle="Top candidates is empty for this ticker — relax "
                         "Min OI / Min Vol on the scan, or pick a ticker "
                         "with more option-chain liquidity.",
            )
            return
        # The first row is now exactly rank-1 from
        # "Top candidates — all chains" for the current scan direction.
        df_mc["_label"] = (
            df_mc.apply(lambda r: (
                f"{r.get('type', mode_r).upper()[0]}  "
                f"${r['strike']:>7.2f}  "
                f"exp {pd.to_datetime(r['expiration']).strftime('%b %d %y')}  "
                f"·  mid ${r.get('mid', 0):.2f}"
                f"  ·  IV {r.get('iv', 0) * 100:.0f}%"
                f"  ·  IV+pp {r.get('iv_excess', 0) * 100:+.1f}"
            ), axis=1)
        )
        # Mark the best one so the user knows the default isn't arbitrary.
        best_signal = df_mc.iloc[0]["iv_excess"] * 100
        best_label = df_mc.iloc[0]["_label"]
        arrow = "▼" if buy_r else "▲"
        st.caption(
            f"**Strongest signal (rank-1 from Top candidates):** {arrow} {best_label}  "
            f"(IV+pp {best_signal:+.1f}, pre-selected below)"
        )

        # Side defaults from scan direction (buy=long, sell=short).
        c_pick, c_side, c_qty, c_btn = st.columns([4, 1.2, 0.8, 1],
                                                    vertical_alignment="bottom")
        with c_pick:
            choice_idx = st.selectbox(
                "Candidate to analyze",
                df_mc.index,
                index=0,  # auto-fill: strongest-signal row from the sort above
                format_func=lambda i: df_mc.at[i, "_label"],
                key="s_mc_choice",
            )
        with c_side:
            side = st.radio(
                "Side", ["long", "short"],
                index=0 if buy_r else 1,
                horizontal=True, key="s_mc_side",
            )
        with c_qty:
            qty = st.number_input("Contracts", value=1, min_value=1,
                                  max_value=100, step=1, key="s_mc_qty")
        with c_btn:
            run_mc = st.button("Run MC", type="primary", key="s_mc_run")

        # Persist the trigger across reruns so the panel stays expanded.
        if run_mc:
            st.session_state["s_mc_armed"] = True

        if st.session_state.get("s_mc_armed", False) and choice_idx is not None:
            picked = df_mc.loc[choice_idx]
            opt_type = str(picked.get("type", "call")).lower()
            opt_type = "call" if opt_type.startswith("c") else "put"
            try:
                position = position_from_chain_row(
                    underlying=ticker_r,
                    spot=spot,
                    row={
                        "Strike": picked["strike"],
                        "Expiration": picked["expiration"],
                        "Mid": picked.get("mid", picked.get("ask", 0)),
                        "IV%": picked.get("iv", 0) * 100,
                    },
                    side=side,  # type: ignore[arg-type]
                    opt_type=opt_type,  # type: ignore[arg-type]
                    qty=int(qty),
                    earnings_dates=tuple(ed) if ed else (),
                    risk_free_rate=0.045,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Couldn't build position from this row: {exc}")
            else:
                render_mc_panel(
                    position,
                    key=f"s_mc_panel_{choice_idx}_{side}_{qty}",
                    label=f"{ticker_r} {opt_type.upper()} {fmt_strike(picked['strike'])} "
                          f"exp {pd.to_datetime(picked['expiration']).strftime('%b %d %y')}",
                )

    from options_scanner.report import render_html
    html = render_html(df_filt, ticker_r, spot, ed, mode_r, buy_r, rcc,
                       res["min_oi"], res.get("min_vol", 0))
    action_tag = "buy" if buy_r else "sell"
    type_tag   = mode_r if mode_r != "both" else "both"
    st.download_button(
        "⬇ Download HTML Report",
        data=html.encode("utf-8"),
        file_name=f"{ticker_r}_{type_tag}_{action_tag}_{date.today().strftime('%Y%m%d')}.html",
        mime="text/html",
        key="s_download",
    )

    with st.expander("Column & color key"):
        st.markdown("""
**Columns**

| Column | Meaning |
|--------|---------|
| Strike | Option strike price. |
| Expiration | Expiration date. |
| DTE | Days to expiration. |
| Bid / Ask | Market bid and ask prices. |
| Mid | Midpoint of bid and ask — the price you'd typically target. |
| IV% | Implied volatility, annualized. |
| IV+pp | How many percentage points the option's IV sits *above* the fitted volatility surface for its expiration. Positive = richer premium than peers at similar strike/DTE. |
| Delta | Black-Scholes delta. For calls: probability of expiring in the money (0–1). For puts: same magnitude, negative sign (−1–0). |
| Ann% | Annualized yield on capital at risk — calls vs. spot price, puts vs. strike. |
| OI | Open interest — total outstanding contracts. Higher = more liquid. |
| Vol | Volume — contracts traded today. |
| NetCr | Roll mode only: net credit received if you close the existing position and open this one. |

**Row shading (chain view)**

| Color | Meaning |
|-------|---------|
| Green | IV+pp is meaningfully above average — premium is rich relative to this chain. |
| Red | IV+pp is below average — premium is thin or cheap relative to this chain. |
| Gray | IV+pp is near average or within the ~3 pp noise floor — no strong signal. |

**Cell highlighting**

| Color | Column | Meaning |
|-------|--------|---------|
| Yellow cell | Bid / Ask | Spread exceeds 1.5× the median spread for this table — wider than typical, execution may cost more than expected. |
| Yellow cell | OI | Open interest is below 2× the minimum OI filter — limited liquidity, harder to fill at a good price. |
| Yellow cell | Vol | Fewer than 4 contracts traded today — very thin activity. |
""")
