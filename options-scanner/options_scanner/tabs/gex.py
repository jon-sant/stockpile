"""GEX tab: dealer-gamma exposure scanner across one or more tickers.

Multi-ticker mode renders a |Total GEX|-ranked summary and lets the
user drill into one ticker for the full GEX chart + strikes-of-
interest table. Single-ticker mode skips the summary and goes
straight to the chart.

DTE scope is user-set via Min/Max DTE inputs (default 0–60) across
both calls and puts — GEX is most reliable on near-term OI; long-dated
(LEAPS) gamma is thin. Note: 0DTE only flows through on Yahoo; the
Schwab fetcher drops same-day expiries.

This is diagnostic output, not a trade signal. See the README's
Gamma Exposure section for caveats.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from options_scanner.compute.gex_summary import compute_gex_summary
from options_scanner.display.gex_chart import show_gex_chart
from options_scanner.display.gex_strikes_table import (
    fmt_strike_with_dist,
    show_gex_strikes_of_interest,
)
from options_scanner.display.spot_meta import (
    fetch_spot_meta,
    spot_help_text,
    spot_value_html,
)
from options_scanner import chain_cache
from options_scanner.fetch import fetch_and_enrich_cached
from options_scanner.ui_theme import metric_card, render_schwab_reauth_hint


def tab_gex() -> None:
    """GEX-only scanner: fetch chains over the user-set DTE range
    (default 0–60) for one or more tickers and surface dealer-gamma
    context (walls, amp zones, gamma flip).

    Multi-ticker mode shows a summary table ranked by |Total GEX|;
    the user picks one ticker to drill into a full GEX chart and
    strikes-of-interest table.

    Diagnostic output, not a trade signal — see README's Gamma Exposure
    section for caveats.
    """
    with st.container(border=True):
        tc, dlo, dhi, sc, expl = st.columns(
            [2, 1, 1, 1, 3], vertical_alignment="bottom")
        with tc:
            tickers_input = st.text_input(
                "Ticker(s) — comma-separated",
                "SPY",
                key="g_ticker",
                help=(
                    "One or more tickers, e.g. `SPY, QQQ, NVDA, AAPL`. "
                    "Multi-ticker mode adds a summary table you can "
                    "sort, then drill into one ticker for the full chart."
                ),
            )
        with dlo:
            min_dte = st.number_input(
                "Min DTE", value=0, min_value=0, step=1, key="g_min_dte",
                help="Lowest days-to-expiration to include. 0 = same-day "
                     "expiries (0DTE carry the most dealer gamma). Note: "
                     "0DTE only flows through on Yahoo — Schwab drops them.",
            )
        with dhi:
            max_dte = st.number_input(
                "Max DTE", value=60, min_value=1, step=1, key="g_max_dte",
                help="Highest days-to-expiration to include. GEX is most "
                     "reliable near-term; long-dated (LEAPS) gamma is thin.",
            )
        with sc:
            with st.container(key="gex_scan_btn_lift"):
                scanned = st.button("Scan", type="primary",
                                    width='stretch',
                                    key="g_scan_btn")
        with expl:
            st.markdown(
                "<style>"
                ".gex-expl details > summary { list-style: none; cursor: pointer; }"
                ".gex-expl details > summary::-webkit-details-marker { display: none; }"
                "[data-testid='stMarkdownContainer']:has(.gex-expl)"
                " { margin-bottom: 0 !important; }"
                "</style>"
                "<div class='gex-expl' style='font-size:0.78rem; color:var(--osc-ink-3); "
                "padding:0 0 0 0.75rem; line-height:1.5; margin-bottom:0;'>"
                "<details>"
                "<summary>"
                "<b>Gamma Exposure (GEX)</b> shows the aggregate delta-hedging "
                "pressure dealers must apply at each strike."
                "&nbsp;<span style='color:var(--osc-ink-4);'>▾</span>"
                "</summary>"
                "<div style='margin-top:0.3rem;'>"
                "After scanning you'll see a bar chart by strike and a "
                "strikes-of-interest table with three key levels: "
                "<b>Gamma Wall</b> (largest positive GEX — price tends to pin here), "
                "<b>Amp Zone</b> (largest negative GEX — moves tend to accelerate), "
                "and <b>Gamma flip</b> (the zero-gamma level where the regime "
                "switches from pinning to amplifying). "
                "Diagnostic context, not a directional signal. "
                "<b>Note:</b> GEX assumes dealers are long calls / short "
                "puts — a market-wide tendency, not verifiable per strike; "
                "concentrated flows (e.g. covered-call funds) can flip the "
                "sign. See the README's caveats."
                "</div>"
                "</details>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Row 2: render-time filters — they re-slice the displayed results
        # live (no rescan), but live here so they're visible up front.
        f1, f2, f3 = st.columns([1.6, 1, 3.4], vertical_alignment="bottom")
        with f1:
            st.radio(
                "Side", ["Both", "Calls", "Puts"], horizontal=True,
                key="g_side",
                help="Dealer gamma is the combined call+put book; filter one "
                     "side to see which book a wall belongs to.",
            )
        with f2:
            st.number_input(
                "Min OI", value=0, min_value=0, key="g_min_oi",
                help="Drop strikes below this open interest before computing "
                     "GEX — trims noise bars from barely-traded contracts.",
            )
        with f3:
            st.caption("Side and Min OI apply live to the scanned results — "
                       "no rescan needed.")

    st.caption(
        f"Scans the **{int(min_dte)}–{int(max_dte)} DTE** chain across "
        "both calls and puts. GEX is most reliable on near-term chains "
        "where OI is dense; long-dated (LEAPS) gamma is thin and noisier."
    )

    # ── Auto-populate from a same-day background scan ──────────────────────────
    # Zero-click, but only when EVERY parsed ticker already has a fresh
    # yahoo-headless snapshot — a partial hit would still mean a live
    # (slow, Selenium) fetch for the missing tickers on page load, which
    # defeats the point. A partial hit just waits for the Scan button.
    _raw = tickers_input.strip().upper()
    _auto_tickers = [t.strip() for t in _raw.replace(";", ",").split(",") if t.strip()]
    _seen = set()
    _auto_tickers = [t for t in _auto_tickers if not (t in _seen or _seen.add(t))]
    _auto_populate = (
        not scanned
        and not st.session_state.get("_gex_rescan_trigger")
        and bool(_auto_tickers)
        and st.session_state.get("gex_results", {}).get("tickers") != _auto_tickers
        and all(chain_cache.has_fresh_snapshot(t, int(min_dte), int(max_dte),
                                               "yahoo-headless")
               for t in _auto_tickers)
    )
    if _auto_populate:
        st.session_state["_gex_rescan_trigger"] = True

    if scanned or st.session_state.pop("_gex_rescan_trigger", False):
        raw = tickers_input.strip().upper()
        tickers = [t.strip() for t
                   in raw.replace(";", ",").split(",")
                   if t.strip()]
        # Preserve user order, drop duplicates
        seen = set()
        tickers = [t for t in tickers
                   if not (t in seen or seen.add(t))]
        if not tickers:
            st.error("Enter one or more ticker symbols.")
            st.session_state.pop("gex_results", None)
            return
        if int(max_dte) < int(min_dte):
            st.error(f"Max DTE ({int(max_dte)}) must be ≥ Min DTE "
                     f"({int(min_dte)}).")
            return

        per_ticker: dict[str, dict] = {}
        failed: list[tuple[str, str]] = []
        fetch_errors = False  # fetch failures (vs. local no-options/no-GEX)
        _was_auto_populate = _auto_populate
        _fetch_provider = ("yahoo-headless" if _was_auto_populate
                           else st.session_state.get("data_source", "yahoo"))
        _any_from_cache = False
        _last_fetched_at = None
        progress = st.progress(
            0.0, text=f"Fetching {len(tickers)} ticker(s)…"
        )
        for i, t in enumerate(tickers, 1):
            progress.progress(
                i / len(tickers),
                text=f"Fetching {t} ({i}/{len(tickers)})…",
            )
            df, earnings, err, from_cache, fetched_at = fetch_and_enrich_cached(
                t, "both", int(min_dte), int(max_dte),
                _fetch_provider,
                st.session_state.get("schwab_config"),
                moomoo_config=st.session_state.get("moomoo_config"),
                force_live=False,
            )
            _any_from_cache = _any_from_cache or from_cache
            _last_fetched_at = fetched_at or _last_fetched_at
            if err:
                failed.append((t, err))
                fetch_errors = True
                continue
            if df.empty:
                failed.append((t, f"no options in {int(min_dte)}–"
                                  f"{int(max_dte)} DTE"))
                continue
            spot = float(df["spot"].iloc[0])
            summary = compute_gex_summary(df, spot)
            if summary is None:
                failed.append((t, "no GEX data (missing gamma/OI)"))
                continue
            per_ticker[t] = {"df": df, "spot": spot,
                             "earnings_dates": earnings, **summary}
        progress.empty()

        for t, msg in failed:
            st.warning(f"**{t}** skipped — {msg}")
        if not per_ticker:
            st.error("No tickers returned GEX data.")
            if fetch_errors:
                _scfg = st.session_state.get("schwab_config") or {}
                render_schwab_reauth_hint(
                    st.session_state.get("data_source", "yahoo"),
                    key="schwab_reauth_gex",
                    token_file=_scfg.get("token_file"),
                )
            st.session_state.pop("gex_results", None)
            return

        st.session_state["scan_ts"] = (
            _last_fetched_at if _any_from_cache else datetime.now().astimezone()
        )
        st.session_state["scan_provider"] = _fetch_provider
        st.session_state["scan_from_cache"] = _any_from_cache
        st.session_state["gex_results"] = {
            "tickers": list(per_ticker.keys()),
            "per_ticker": per_ticker,
            "min_dte": int(min_dte),
            "max_dte": int(max_dte),
        }

    res = st.session_state.get("gex_results")
    if not res:
        return

    per_ticker = res["per_ticker"]
    if not per_ticker:
        return

    # ── Render-time filters (widgets live in the controls card above) ─────
    side_label = st.session_state.get("g_side", "Both")
    g_min_oi = st.session_state.get("g_min_oi", 0)

    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[df["open_interest"] >= int(g_min_oi)]
        if side_label != "Both":
            sub = sub[sub["type"] == ("call" if side_label == "Calls"
                                      else "put")]
        return sub

    # Build summary df sorted by |Total GEX| descending so the most
    # gamma-exposed ticker is the default drill-down pick. Summaries are
    # recomputed from the filtered chains so the table tracks the filters.
    rows = []
    for t, info in per_ticker.items():
        spot = info["spot"]
        summary = compute_gex_summary(_apply_filters(info["df"]), spot)
        if summary is None:
            continue
        rows.append({
            "Ticker":    t,
            "Spot":      spot,
            "Total GEX": summary["total_gex"],
            "Regime":    summary["regime"],
            "Gamma flip": fmt_strike_with_dist(summary["zero_gamma"], spot),
            "Top Wall":  fmt_strike_with_dist(summary["top_wall"], spot),
            "Top Amp":   fmt_strike_with_dist(summary["top_amp"], spot),
        })
    if not rows:
        st.info("Nothing left after the Side / Min OI filters — loosen them "
                "to see GEX again.")
        return
    summary_df = pd.DataFrame(rows)
    summary_df = (summary_df
                  .assign(_abs=summary_df["Total GEX"].abs())
                  .sort_values("_abs", ascending=False)
                  .drop(columns=["_abs"])
                  .reset_index(drop=True))

    st.divider()

    n = len(per_ticker)
    rescan_label = (f"↻ Rescan {res['tickers'][0]}"
                    if n == 1 else f"↻ Rescan ({n})")
    with st.container(key="rescan_pill_gex"):
        if st.button(rescan_label, type="primary", key="g_rescan_btn"):
            st.session_state["_gex_rescan_trigger"] = True
            st.rerun()

    if n > 1:
        st.subheader("GEX summary")
        st.caption(
            "One row per ticker, sorted by absolute Total GEX (most "
            "dealer-gamma exposure first). The Gamma flip, Top Wall, and "
            "Top Amp cells include each strike's distance from spot."
        )
        st.dataframe(
            summary_df, hide_index=True, width='content',
            column_config={
                "Ticker":    st.column_config.TextColumn(),
                "Spot":      st.column_config.NumberColumn(format="$%.2f"),
                "Total GEX": st.column_config.NumberColumn(format="%,.0f"),
                "Regime":    st.column_config.TextColumn(),
                "Gamma flip": st.column_config.TextColumn(),
                "Top Wall":  st.column_config.TextColumn(),
                "Top Amp":   st.column_config.TextColumn(),
            },
        )

        drill = st.selectbox(
            "Drill into ticker",
            summary_df["Ticker"].tolist(),
            index=0,
            key="g_drill",
        )
        st.divider()
    else:
        drill = res["tickers"][0]

    info = per_ticker[drill]
    df_r = _apply_filters(info["df"])
    spot = info["spot"]

    if n == 1:
        m1, m2, m3 = st.columns(3)
        with m1:
            _meta = fetch_spot_meta(
                drill, st.session_state.get("scan_provider", "yahoo"),
            )
            metric_card("SPOT",
                        spot_value_html(spot, _meta["pct_change"]),
                        help_text=spot_help_text(_meta))
        with m2:
            metric_card("EXPIRATIONS",
                        f"{df_r['expiration'].nunique()}",
                        help_text=f"{res.get('min_dte', 0)}–"
                                  f"{res.get('max_dte', 60)} DTE")
        with m3:
            _earnings = info.get("earnings_dates") or []
            if _earnings:
                _earn_days = (_earnings[0] - date.today()).days
                _earn_label = _earnings[0].strftime("%b %d")
                _earn_sub   = f"in {_earn_days}d"
            else:
                _earn_label = "—"
                _earn_sub   = "no upcoming events"
            metric_card("NEXT EARNINGS", _earn_label,
                        delta=_earn_sub, delta_sign="neutral")
        st.divider()

    show_gex_chart(df_r, spot,
                    provider=st.session_state.get("scan_provider", "yahoo"),
                    ticker=drill)

    show_gex_strikes_of_interest(df_r, spot)
