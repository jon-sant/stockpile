"""Portfolio and Watchlist tabs — one shared scan engine, two entry points.

`tab_portfolio()` scans every position in an uploaded brokerage CSV;
`tab_watchlist()` scans a typed/saved basket of tickers (best-option
only — no positions, so no roll context). Both share `_render_scan_tab`:
fetch each ticker's chain, filter by DTE/OI/volume/delta, render the
cross-ticker leaderboard and a per-ticker section with action cards.

Controls (portfolio mode):
  Option type  — Calls / Puts / Both
  Scan mode    — Roll (surface roll candidates) / Best option (just top pick)
  Positions    — Open only (shares > 0) / All tickers (every ticker in CSV)
  Format       — auto-detected on upload; user can override
Watchlist mode swaps Scan mode/Positions for a Sell/Buy direction toggle
and the saved-watchlist load/save row.

The module keeps its CSV validation helpers (_validate_csv,
_show_validation) private — they're only meaningful in the upload context.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import date, datetime

import pandas as pd
import streamlit as st

from stocks_shared.yahoo import RateLimitError

from options_scanner.compute import capital_allocator, correlation
from options_scanner.display.iv_chart import show_iv_chart
from options_scanner.display.leaderboard import build_leaderboard, render_leaderboard
from options_scanner.display.portfolio_action_card import render_portfolio_action_card
from options_scanner.display.scan_results import show_scan_results
from options_scanner.display.spot_meta import (
    fetch_spot_meta,
    spot_help_text,
    spot_value_html,
)
from options_scanner.defaults import default_delta_range
from options_scanner import chain_cache
from options_scanner.fetch import fetch_position_cached
from options_scanner.portfolio import detect_brokerage
from options_scanner.ui_theme import (
    badge, metric_card, render_schwab_reauth_hint, section_header,
)
from options_scanner import watchlists

# Yahoo rate-limit handling: when a fetch raises RateLimitError, the scan
# waits once and retries that ticker rather than failing it — large
# baskets may pause this way several times and still finish. But if a
# ticker is *still* throttled after its wait, Yahoo is persistently
# rate-limiting the whole IP (it throttles per-IP, not per-symbol), so
# further waiting is pointless: the scan stops waiting and fails the
# remaining throttled tickers fast. Worst case ≈ one wait, not minutes
# per ticker.
_RL_WAIT_SECONDS = 60


@st.cache_data(show_spinner=False)
def _validate_csv(content: bytes, brokerage: str) -> tuple[list, int, str | None, list | None]:
    """Validate an uploaded CSV.

    Returns (issues, row_count, parse_error, positions_info):
    - issues:         list of ValidationIssue (stockpile only; [] for other formats)
    - row_count:      total tickers parsed (stockpile: data rows)
    - parse_error:    error string if parse failed, else None
    - positions_info: list of {ticker, shares, n_calls, n_puts} for non-stockpile;
                      None for stockpile (validator doesn't provide position detail)
    """
    if brokerage == "stockpile":
        from stocks_shared.validators import validate_stockpile_csv, count_data_rows
        text = content.decode("utf-8-sig")
        return validate_stockpile_csv(text), count_data_rows(text), None, None

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        from options_scanner.portfolio import get_portfolio
        positions = get_portfolio(tmp_path, brokerage, include_closed=True)
        info = [
            {
                "ticker": p["ticker"],
                "shares": p["shares"],
                "n_calls": sum(o["contracts"] for o in p["open_calls"]),
                "n_puts":  sum(o["contracts"] for o in p["open_puts"]),
            }
            for p in positions
        ]
        return [], len(positions), None, info
    except Exception as exc:
        return [], 0, str(exc), None
    finally:
        os.unlink(tmp_path)


def _show_validation(issues: list, row_count: int, parse_error: str | None,
                     brokerage: str,
                     positions_info: list | None = None) -> bool:
    """Render the validation panel.  Returns True if the file is scan-ready."""
    if parse_error:
        st.error(f"Could not parse CSV: {parse_error}")
        return False

    if brokerage != "stockpile":
        plural = "s" if row_count != 1 else ""
        st.success(f"Parsed successfully — {row_count} ticker{plural} found.")

        if positions_info:
            has_opts   = [p for p in positions_info
                          if p["n_calls"] > 0 or p["n_puts"] > 0]
            stock_only = [p for p in positions_info
                          if p["shares"] > 0
                          and p["n_calls"] == 0 and p["n_puts"] == 0]
            closed     = [p for p in positions_info
                          if p["shares"] <= 0
                          and p["n_calls"] == 0 and p["n_puts"] == 0]

            def _opt_label(p: dict) -> str:
                tags = []
                if p["n_calls"]: tags.append(f"{p['n_calls']}C")
                if p["n_puts"]:  tags.append(f"{p['n_puts']}P")
                return f"{p['ticker']} ({' '.join(tags)})"

            if has_opts:
                st.caption("**Open options:** "
                           + ", ".join(_opt_label(p) for p in has_opts))
            if stock_only:
                st.caption("**Stock only:** "
                           + ", ".join(f"{p['ticker']} ({p['shares']:g}sh)"
                                       for p in stock_only))
            if closed:
                st.caption("**Closed:** "
                           + ", ".join(p["ticker"] for p in closed))

        return True

    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if not issues:
        st.success(f"Valid — {row_count} rows, no issues found.")
        return True

    parts = []
    if errors:
        parts.append(f"{len(errors)} error{'s' if len(errors) != 1 else ''}")
    if warnings:
        parts.append(f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''}")
    summary = f"{row_count} rows — {', '.join(parts)}"

    if errors:
        st.error(summary)
    else:
        st.warning(summary)

    with st.expander("Show issues", expanded=bool(errors)):
        import pandas as pd
        df = pd.DataFrame([
            {
                "Row":     str(i.row) if i.row > 0 else "—",
                "Field":   i.field or "—",
                "Level":   i.severity.upper(),
                "Message": i.message,
            }
            for i in issues
        ])

        def _row_style(row):
            color = (
                "background-color: rgba(239,68,68,0.18)"
                if row["Level"] == "ERROR"
                else "background-color: rgba(234,179,8,0.22)"
            )
            return [color] * len(row)

        styled = df.style.apply(_row_style, axis=1)
        st.dataframe(styled, hide_index=True, width="stretch")

    return not errors


def _parse_watchlist(text: str) -> list[str]:
    """Parse a free-text watchlist into a deduped, normalized ticker list.

    Accepts commas, whitespace, or newlines as separators. Order is
    preserved (first occurrence wins).
    """
    from stocks_shared.yahoo import normalize_ticker
    raw = text.replace(",", " ").split()
    seen: dict[str, None] = {}
    for tok in raw:
        t = normalize_ticker(tok)
        if t:
            seen.setdefault(t, None)
    return list(seen.keys())


def _scan_one(pos: dict, opt_type_key: str, scan_mode_key: str,
              provider: str, scfg: dict | None,
              min_dte: int, max_dte: int,
              force_live: bool = True) -> dict:
    """Fetch + enrich one position and build its result dict.

    Shared by both input sources. Watchlist tickers pass a synthetic
    position (`shares=0`, no open options), so the roll-close-cost block
    below is skipped and the position scans as "best option" only.

    force_live=False lets a same-day snapshot (background-populated or
    from an earlier scan click this session) serve the fetch instead of
    a live call whenever it covers the requested DTE range — provider
    is still whatever the caller passes, but a cache hit only ever
    happens when that's "yahoo-headless" (see fetch.fetch_position_cached).
    Both the auto-populate path and an explicit Scan click pass False
    here so narrowing min_dte/max_dte after an earlier same-day scan
    reuses that snapshot instead of re-fetching live.
    """
    ticker = pos["ticker"]
    df, earnings_dates, err, from_cache, fetched_at = fetch_position_cached(
        ticker, int(min_dte), provider, scfg,
        moomoo_config=st.session_state.get("moomoo_config"),
        opt_type=opt_type_key,
        max_dte=int(max_dte),
        force_live=force_live,
    )

    # Roll close cost lookup — only in Roll mode, only for open options
    # on the relevant side(s).
    roll_close_costs = {}
    if scan_mode_key == "roll":
        opts_to_lookup = []
        if opt_type_key in ("calls", "both"):
            opts_to_lookup += [(opt, "calls") for opt in pos["open_calls"]]
        if opt_type_key in ("puts", "both"):
            opts_to_lookup += [(opt, "puts") for opt in pos["open_puts"]]

        _schwab_client = None
        if provider == "schwab" and opts_to_lookup:
            from stocks_shared.schwab_live import get_client
            try:
                _schwab_client = get_client(
                    scfg["app_key"], scfg["app_secret"],
                    scfg["callback_url"], scfg["token_file"],
                )
            except (ValueError, TypeError):
                pass

        for opt, side in opts_to_lookup:
            m, d, y = opt["expiration"].split("/")
            exp_yf = f"{y}-{m}-{d}"
            if provider == "schwab" and _schwab_client is not None:
                from stocks_shared.schwab_live import fetch_option_chain_schwab
                chain = fetch_option_chain_schwab(_schwab_client, ticker, exp_yf)
            else:
                from stocks_shared.yahoo import fetch_option_chain
                chain = fetch_option_chain(ticker, exp_yf)
            if chain is None:
                continue
            chain_df = chain.calls if side == "calls" else chain.puts
            row = chain_df[chain_df["strike"] == float(opt["strike"])]
            if not row.empty:
                bid  = float(row["bid"].iloc[0] or 0)
                ask  = float(row["ask"].iloc[0] or 0)
                last = float(row["lastPrice"].iloc[0] or 0)
                roll_close_costs[opt["symbol"]] = (
                    (bid + ask) / 2 if bid > 0 and ask > 0 else last
                )

    return {
        "position": pos,
        "error": err,
        "df": df,
        "spot": float(df["spot"].iloc[0]) if not df.empty else None,
        "earnings_dates": earnings_dates,
        "roll_close_costs": roll_close_costs,
        "from_cache": from_cache,
        "fetched_at": fetched_at,
    }


def _scan_all_parallel(positions: list, opt_type_key: str,
                       scan_mode_key: str, provider: str,
                       scfg: dict | None, min_dte: int, max_dte: int,
                       progress, force_live: bool = True) -> list:
    """Scan every position concurrently (yahoo-headless only).

    The headless provider keeps a bounded pool of long-lived browsers,
    so overlapping symbols just interleave their page loads there —
    unlike the API providers there's no shared throttle to trip, and
    no wait-and-retry choreography. Results come back in `positions`
    order; a failed ticker degrades to an error card instead of
    aborting the batch.
    """
    import concurrent.futures
    import functools

    try:  # same graceful degrade as tabs/trades.py
        from streamlit.runtime.scriptrunner import (
            add_script_run_ctx, get_script_run_ctx,
        )
        _init = functools.partial(add_script_run_ctx,
                                  ctx=get_script_run_ctx())
    except Exception:  # pragma: no cover — shields Streamlit layout drift
        _init = None

    def _job(pos):
        try:
            return _scan_one(pos, opt_type_key, scan_mode_key, provider,
                             scfg, min_dte, max_dte, force_live=force_live)
        except Exception as exc:  # noqa: BLE001 — degrade to an error card
            return {"position": pos, "error": f"{type(exc).__name__}: {exc}",
                    "df": pd.DataFrame(), "spot": None,
                    "earnings_dates": [], "roll_close_costs": {},
                    "from_cache": False, "fetched_at": None}

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, len(positions)), initializer=_init) as ex:
        futures = [ex.submit(_job, p) for p in positions]
        done = 0
        for _ in concurrent.futures.as_completed(futures):
            done += 1
            progress.progress(done / len(positions),
                              text=f"Scanning… ({done}/{len(positions)})")
    return [f.result() for f in futures]


def tab_portfolio() -> None:
    """Brokerage-CSV scan: every position in an export."""
    _render_scan_tab(is_watchlist=False, k="p")


def tab_watchlist() -> None:
    """Typed-basket scan: best option across a saved or ad-hoc list."""
    _render_scan_tab(is_watchlist=True, k="w")


def _render_scan_tab(is_watchlist: bool, k: str) -> None:
    """Shared engine behind the Portfolio and Watchlist tabs.

    Both tabs render every script run, so every widget and session key
    that appears in both modes is prefixed with `k` ("p" portfolio /
    "w" watchlist) to keep them collision-free and independent.
    """
    if is_watchlist:
        section_header(
            title="Watchlist scan",
            subtitle=(
                "Type or load a basket of tickers — we'll rank the richest "
                "(or cheapest) options across the lot. No broker account "
                "needed."
            ),
            eyebrow="STEP 01 · INPUT",
        )
    else:
        section_header(
            title="Portfolio scan",
            subtitle=(
                "Upload a brokerage CSV — we'll surface roll candidates and rich "
                "options ticker-by-ticker."
            ),
            eyebrow="STEP 01 · INPUT",
        )

    uploaded = None
    brokerage = None
    watchlist_tickers: list[str] = []

    if is_watchlist:
        # Saved watchlists row: load one (refills tickers + its filters), or
        # name the current basket and Save/Delete it. The selectbox on_change
        # runs before the rerun, so its session-state writes reach the widgets
        # on the next render. Save/Delete read the current tickers + filters
        # from session state (the widgets below persist across reruns).
        # Show a save/delete confirmation from the previous run (set just
        # before its st.rerun(), so the message survives the rerun), then clear.
        _flash = st.session_state.pop("_wl_flash", None)
        if _flash:
            st.success(_flash)

        _saved = watchlists.load()
        _names = {e["name"].lower() for e in _saved}
        _wl_placeholder = "— saved watchlists —"

        def _apply_watchlist() -> None:
            entry = next((e for e in _saved
                          if e["name"] == st.session_state.get(f"{k}_wl_saved")), None)
            if entry is None:
                return
            _step = 0.05
            _dmin = round(round(float(entry.get("delta_min", 0.10)) / _step) * _step, 10)
            _dmax = round(round(float(entry.get("delta_max", 0.70)) / _step) * _step, 10)
            st.session_state[f"{k}_watchlist"] = ", ".join(entry["tickers"])
            # Fill the name field too, so Save (re-save) and Delete light up
            # for the loaded watchlist.
            st.session_state[f"{k}_wl_name"] = entry["name"]
            st.session_state[f"{k}_min_dte"]  = max(1, int(entry.get("min_dte", 30)))
            st.session_state[f"{k}_max_dte"]  = max(1, int(entry.get("max_dte", 90)))
            st.session_state[f"{k}_min_oi"]   = max(0, int(entry.get("min_oi", 25)))
            st.session_state[f"{k}_min_vol"]  = max(0, int(entry.get("min_vol", 1)))
            # Not persisted with saved watchlists — runtime-only, so reset to
            # blank rather than carry over a stale filter.
            st.session_state[f"{k}_min_ivpp"] = None
            st.session_state[f"{k}_min_ann"]  = None
            st.session_state[f"{k}_min_percentile"] = None
            st.session_state[f"{k}_min_ann_delta_percentile"] = None
            st.session_state[f"{k}_delta"]    = (_dmin, _dmax)
            st.session_state[f"{k}_top"]      = max(1, int(entry.get("top_n", 5)))
            st.session_state[f"{k}_opt_type"] = entry.get("option_type", "Calls")
            st.session_state[f"{k}_action"] = (
                "Buy (IV-cheap candidates)" if entry.get("buy")
                else "Sell (IV-rich candidates)")
            # Reset to placeholder (allowed inside on_change) so the same
            # pick can be re-applied later after manual edits.
            st.session_state[f"{k}_wl_saved"] = _wl_placeholder

        _tk_col, _mid_col, _btn_col = st.columns([4, 2, 1])
        with _tk_col:
            _wl_text = st.text_area(
                "Tickers", key=f"{k}_watchlist", height=160,
                placeholder="AMD, NVDA, AAPL  (commas, spaces, or new lines)",
                help="Separate tickers by comma, space, or new line. Append ! "
                     "to a symbol to disable index normalization.",
            )
            st.caption("Separate with commas, spaces, or new lines.")
            watchlist_tickers = _parse_watchlist(_wl_text or "")
        with _mid_col:
            st.selectbox(
                "Saved watchlists",
                [_wl_placeholder] + [e["name"] for e in _saved],
                index=0, key=f"{k}_wl_saved", on_change=_apply_watchlist,
                help="Load a saved basket and the filters it was saved with.",
            )
            _wl_name = st.text_input(
                "Save as", key=f"{k}_wl_name",
                placeholder="e.g. Mag7  (.json ext will be added)",
                help="Name the current basket to reuse it later — the .json "
                     "extension is added automatically. Re-saving a name "
                     "overwrites it.",
            )
        _name = _wl_name.strip()
        with _btn_col:
            st.markdown("<div style='height:1.72rem'></div>", unsafe_allow_html=True)
            if st.button("💾 Save", use_container_width=True,
                         disabled=not (watchlist_tickers and _name)):
                _d = st.session_state.get(f"{k}_delta", (0.10, 0.70))
                _was_update = _name.lower() in _names
                watchlists.save({
                    "name": _name,
                    "tickers": watchlist_tickers,
                    "option_type": st.session_state.get(f"{k}_opt_type", "Calls"),
                    "min_dte": int(st.session_state.get(f"{k}_min_dte", 30)),
                    "max_dte": int(st.session_state.get(f"{k}_max_dte", 90)),
                    "min_oi": int(st.session_state.get(f"{k}_min_oi", 25)),
                    "min_vol": int(st.session_state.get(f"{k}_min_vol", 1)),
                    "delta_min": float(_d[0]),
                    "delta_max": float(_d[1]),
                    "top_n": int(st.session_state.get(f"{k}_top", 5)),
                    "buy": st.session_state.get(f"{k}_action", "").startswith("Buy"),
                })
                st.session_state["_wl_flash"] = (
                    f"{'Updated' if _was_update else 'Saved'} watchlist "
                    f"“{_name}” ({len(watchlist_tickers)} tickers).")
                st.rerun()
            if st.button("🗑 Delete", use_container_width=True,
                         disabled=_name.lower() not in _names):
                watchlists.delete(_name)
                st.session_state["_wl_flash"] = f"Deleted watchlist “{_name}”."
                st.rerun()

        if watchlist_tickers:
            st.caption(f"{len(watchlist_tickers)} ticker(s): "
                       + ", ".join(watchlist_tickers))
    else:
        # ── Upload row: file picker (50 %) + format selector (25 %) + spacer ──
        _up_col, _fmt_col, _ = st.columns([2, 1, 1])
        with _up_col:
            uploaded = st.file_uploader("Brokerage CSV export", type=["csv"])
            st.markdown(
                "<div style='margin: 0.2rem 0 0 0;'>"
                + badge("PROCESSED LOCALLY · NEVER UPLOADED", "positive")
                + "</div>",
                unsafe_allow_html=True,
            )

        # ── Auto-detect format on new file upload ────────────────────────────
        if uploaded is not None:
            _file_sig = f"{uploaded.name}:{len(uploaded.getvalue())}"
            if st.session_state.get("_port_file_sig") != _file_sig:
                _detected = detect_brokerage(uploaded.getvalue())
                st.session_state["_port_file_sig"] = _file_sig
                st.session_state["_port_auto_detected"] = _detected
                st.session_state["_port_detect_ran"] = True
                if _detected:
                    st.session_state["p_brokerage"] = _detected

        with _fmt_col:
            brokerage = st.selectbox(
                "Format",
                ["schwab", "robinhood", "fidelity", "merrill", "stockpile"],
                index=None,
                placeholder="Select format…",
                key="p_brokerage",
                help="Select your brokerage export format, or 'stockpile' for "
                     "a manually-entered transaction log.",
            )
            if uploaded is not None and st.session_state.get("_port_detect_ran"):
                _auto = st.session_state.get("_port_auto_detected")
                if _auto and brokerage == _auto:
                    st.markdown(
                        badge("AUTO-DETECTED", "positive")
                        + f"&nbsp;<span style='font-size:0.78rem'>{_auto}</span>",
                        unsafe_allow_html=True,
                    )
                elif not _auto and brokerage is None:
                    st.markdown(
                        badge("FORMAT UNKNOWN", "neutral")
                        + "&nbsp;<span style='font-size:0.78rem'>select manually</span>",
                        unsafe_allow_html=True,
                    )

    # ── Controls row 1: filter params ────────────────────────────────────────
    pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8 = st.columns(
        [1, 1, 1, 1, 1, 1, 2, 1])
    with pc1:
        port_min_dte = st.number_input("Min DTE", value=30, min_value=1,
                                       key=f"{k}_min_dte")
    with pc2:
        port_max_dte = st.number_input("Max DTE", value=90, min_value=1,
                                       key=f"{k}_max_dte")
    with pc3:
        port_min_oi = st.number_input("Min OI", value=25, min_value=0,
                                      key=f"{k}_min_oi")
    with pc4:
        port_min_vol = st.number_input("Min Vol", value=1, min_value=0,
                                       key=f"{k}_min_vol")
    with pc5:
        port_min_ivpp = st.number_input(
            "Min IV+pp", value=None, step=0.5, format="%.1f",
            placeholder="none", key=f"{k}_min_ivpp",
            help="Floor on IV+pp (pp). Blank = no filter.",
        )
    with pc6:
        port_min_ann = st.number_input(
            "Min Ann%", value=None, step=1.0, format="%.1f",
            placeholder="none", key=f"{k}_min_ann",
            help="Floor on Ann% (annualized yield). Blank = no filter.",
        )
    with pc7:
        port_delta_range = st.slider("Delta Range", 0.0, 1.0,
                                     default_delta_range(False),
                                     0.05, key=f"{k}_delta")
    with pc8:
        port_top = st.number_input("Top N per ticker", value=5, min_value=1,
                                   key=f"{k}_top")

    pp1, pp2, _pp3 = st.columns([1, 1, 4])
    with pp1:
        port_min_percentile = st.number_input(
            "Min IV %ile", value=None, step=5.0, format="%.0f",
            min_value=0.0, max_value=100.0,
            placeholder="none", key=f"{k}_min_percentile",
            help="Floor on IV %ile — how rich today's IV+pp is vs. this "
                 "ticker's own delta/DTE-bucketed history (0-100). "
                 "Blank = no filter.",
        )
    with pp2:
        port_min_ann_delta_percentile = st.number_input(
            "Min Ann/Δ %ile", value=None, step=5.0, format="%.0f",
            min_value=0.0, max_value=100.0,
            placeholder="none", key=f"{k}_min_ann_delta_percentile",
            help="Floor on Ann%/Delta %ile — how rich today's yield-per-"
                 "delta is vs. this ticker's own bucketed history (0-100). "
                 "Blank = no filter.",
        )
    with _pp3:
        port_show_mc = st.checkbox(
            "Show Monte Carlo columns", value=False, key=f"{k}_show_mc",
            help="P(profit), MC Fair Value, Premium vs Model, Breakeven "
                 "Move, CVaR, VaR, Sortino — computed by a background "
                 "worker, so these may show \"TBD\" until it catches up.",
        )

    # ── Controls row 2: scan semantics ───────────────────────────────────────
    # Watchlist mode is best-option only over a typed basket, so Scan mode
    # and the position-scope checkboxes (which are about CSV positions) hide.
    # The Sell/Buy toggle lives here too (watchlist only) — buying puts to
    # short, or calls across a basket. The CSV path stays sell-only.
    if is_watchlist:
        def _sync_p_delta() -> None:
            """Re-seed the delta band to the new direction's default on a
            Sell/Buy flip (sell ≈ OTM, buy ≈ near-ATM)."""
            _b = st.session_state.get(f"{k}_action", "").startswith("Buy")
            st.session_state[f"{k}_delta"] = default_delta_range(_b)

        wc1, wc2 = st.columns([2.2, 1.8])
        with wc1:
            direction_label = st.radio(
                "Direction",
                ["Sell (IV-rich candidates)", "Buy (IV-cheap candidates)"],
                horizontal=True, key=f"{k}_action", on_change=_sync_p_delta,
                help="Sell: rank IV-rich premium to write. Buy: rank IV-cheap "
                     "contracts to buy — e.g. puts to short a name, or calls "
                     "across the basket.",
            )
        with wc2:
            opt_type_label = st.radio(
                "Option type", ["Calls", "Puts", "Both"],
                horizontal=True, key=f"{k}_opt_type",
            )
        buy = direction_label.startswith("Buy")
        scan_mode_label = "Best option"
        scope_open = scope_stock = scope_closed = True
    else:
        buy = False
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            opt_type_label = st.radio(
                "Option type", ["Calls", "Puts", "Both"],
                horizontal=True, key=f"{k}_opt_type",
            )
        with sc2:
            scan_mode_label = st.radio(
                "Scan mode", ["Best option", "Roll"],
                horizontal=True, key="p_scan_mode",
                help="Best option: surface the top IV-rich pick with no roll context. "
                     "Roll: find candidates to roll existing open positions into.",
            )
        with sc3:
            st.caption("Positions to scan")
            scope_open   = st.checkbox("Open options", value=True,  key="p_scope_open",
                                       help="Has existing short calls or puts.")
            scope_stock  = st.checkbox("Stock only",   value=True,  key="p_scope_stock",
                                       help="Holds shares with no open options.")
            scope_closed = st.checkbox("Closed",       value=False, key="p_scope_closed",
                                       help="Previously held, now fully exited.")

    opt_type_key  = {"Calls": "calls", "Puts": "puts", "Both": "both"}[opt_type_label]
    scan_mode_key = {"Roll": "roll", "Best option": "best"}[scan_mode_label]
    _side = {"calls": "call", "puts": "put", "both": "both"}[opt_type_key]

    # Invalidate stored results when the file, format, watchlist, or scan
    # semantics change. Each tab keeps its own results + cache key.
    _results_key = "watchlist_results" if is_watchlist else "portfolio_results"
    _cache_key = (
        f"{uploaded.name}:{len(uploaded.getvalue())}" if uploaded else None,
        tuple(watchlist_tickers),
        brokerage,
        opt_type_key,
        scan_mode_key,
        buy,
        scope_open, scope_stock, scope_closed,
        int(port_min_dte),
        int(port_max_dte),
    )
    if st.session_state.get(f"_{k}_cache_key") != _cache_key:
        st.session_state.pop(_results_key, None)
        st.session_state[f"_{k}_cache_key"] = _cache_key

    # ── Validation (CSV source only; auto-runs when file + format are set) ────
    scan_ready = False
    if not is_watchlist and uploaded is not None and brokerage is not None:
        with st.container(border=True):
            st.caption(
                f"**Validation** — {uploaded.name}"
                + (" (stockpile format)" if brokerage == "stockpile" else "")
            )
            issues, row_count, parse_error, positions_info = _validate_csv(
                uploaded.getvalue(), brokerage
            )
            scan_ready = _show_validation(
                issues, row_count, parse_error, brokerage, positions_info,
            )

            if brokerage == "stockpile":
                st.caption(
                    "See the README for the full format spec and an example "
                    "row for every transaction type (BUY, SELL, STO, BTO, "
                    "STC, BTC, EXPIRED, ASSIGNED, EXERCISED, DIVIDEND, "
                    "SPLIT, TRANSFER_IN)."
                )

    _scan_disabled = (not watchlist_tickers if is_watchlist
                      else (uploaded is None or brokerage is None
                            or not scan_ready))
    _scan_label = "Scan Watchlist" if is_watchlist else "Scan Portfolio"

    # ── Auto-populate from a same-day background scan ──────────────────────────
    # Zero-click, but only when EVERY position's ticker already has a
    # fresh yahoo-headless snapshot — a partial hit would still mean a
    # live (slow) fetch for the rest on page load. Watchlist tickers are
    # known immediately; CSV positions need a validated upload first
    # (mirrors the click handler's own parsing below).
    _auto_positions: list | None = None
    if is_watchlist:
        if watchlist_tickers:
            _auto_positions = [
                {"ticker": t, "shares": 0, "open_calls": [], "open_puts": []}
                for t in watchlist_tickers
            ]
    elif uploaded is not None and brokerage is not None and scan_ready:
        from options_scanner.portfolio import get_portfolio
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as _f:
                _f.write(uploaded.getvalue())
                _auto_tmp_path = _f.name
            try:
                _auto_positions = get_portfolio(_auto_tmp_path, brokerage,
                                                include_closed=True)
                _auto_positions = [
                    p for p in _auto_positions
                    if (scope_open   and (p["open_calls"] or p["open_puts"]))
                    or (scope_stock  and p["shares"] > 0
                        and not p["open_calls"] and not p["open_puts"])
                    or (scope_closed and p["shares"] <= 0
                        and not p["open_calls"] and not p["open_puts"])
                ]
            finally:
                os.unlink(_auto_tmp_path)
        except Exception:
            _auto_positions = None  # let an explicit Scan click surface the real error

    _auto_populate = False
    if _auto_positions:
        _auto_tickers = tuple(p["ticker"] for p in _auto_positions)
        _prev = st.session_state.get(_results_key)
        _prev_tickers = (tuple(r["position"]["ticker"] for r in _prev["results"])
                         if _prev else None)
        if (_auto_tickers != _prev_tickers
                and all(chain_cache.has_fresh_snapshot(
                            t, int(port_min_dte), int(port_max_dte), "yahoo-headless")
                       for t in _auto_tickers)):
            _auto_populate = True
    _btn_col, _note_col = st.columns([1, 4], vertical_alignment="center")
    with _note_col:
        if st.session_state.get("data_source", "yahoo") == "yahoo":
            st.caption("Yahoo Finance option quotes can be unavailable while "
                       "the market is closed — scans may come back empty or "
                       "throttled until it reopens. Yahoo (headless) scrapes "
                       "the site directly and usually still gets quotes.")
    with _btn_col:
        _scan_clicked = st.button(_scan_label, type="primary",
                                  disabled=_scan_disabled)
    if _scan_clicked or _auto_populate:
        _was_auto_populate = _auto_populate and not _scan_clicked
        _provider = ("yahoo-headless" if _was_auto_populate
                    else st.session_state.get("data_source", "yahoo"))
        _scfg = st.session_state.get("schwab_config")

        if is_watchlist:
            # Synthetic positions: no shares, no open options → best-option scan.
            positions = [
                {"ticker": t, "shares": 0, "open_calls": [], "open_puts": []}
                for t in watchlist_tickers
            ]
            source_name = f"Watchlist ({len(positions)} tickers)"
        else:
            from options_scanner.portfolio import get_portfolio
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                f.write(uploaded.getvalue())
                tmp_path = f.name
            try:
                positions = get_portfolio(tmp_path, brokerage, include_closed=True)
                # Keep positions that match any checked bucket. Roll mode only
                # applies inside the "Open options" bucket; the other two buckets
                # always use "best option" logic (handled in the render loop via
                # the covered/roll_close flags, which are False/None when the
                # position has no open options).
                positions = [
                    p for p in positions
                    if (scope_open   and (p["open_calls"] or p["open_puts"]))
                    or (scope_stock  and p["shares"] > 0
                        and not p["open_calls"] and not p["open_puts"])
                    or (scope_closed and p["shares"] <= 0
                        and not p["open_calls"] and not p["open_puts"])
                ]
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
                os.unlink(tmp_path)
                return
            os.unlink(tmp_path)
            source_name = uploaded.name

        if not positions:
            st.warning("No positions to scan." if is_watchlist
                       else "No positions found in this CSV.")
            return

        st.success(f"Found {len(positions)} position(s): "
                   f"{', '.join(p['ticker'] for p in positions)}")

        progress = st.progress(0, text="Scanning…")
        results = []
        rl_give_up = False  # a wait didn't clear the throttle → stop waiting
        if _provider == "yahoo-headless" and len(positions) > 1:
            # Browser scrape has no shared API throttle — scan every
            # symbol at once instead of the serial wait-and-retry loop.
            results = _scan_all_parallel(
                positions, opt_type_key, scan_mode_key, _provider, _scfg,
                int(port_min_dte), int(port_max_dte), progress,
                force_live=False)
            positions = []  # skip the sequential loop below
        for i, pos in enumerate(positions):
            pct = (i + 1) / len(positions)
            progress.progress(pct, text=f"Scanning {pos['ticker']} "
                                        f"({i+1}/{len(positions)})…")
            waited = False  # this ticker already used its one wait
            while True:
                try:
                    res = _scan_one(
                        pos, opt_type_key, scan_mode_key, _provider, _scfg,
                        int(port_min_dte), int(port_max_dte),
                        force_live=False,
                    )
                    break
                except RateLimitError as exc:
                    if rl_give_up or waited:
                        rl_give_up = True
                        res = {"position": pos,
                               "error": (f"{exc}. Yahoo is still throttling "
                                         "— rescan in a few minutes, or "
                                         "switch the data source to Schwab."),
                               "df": pd.DataFrame(), "spot": None,
                               "earnings_dates": [], "roll_close_costs": {},
                               "from_cache": False, "fetched_at": None}
                        break
                    waited = True
                    for left in range(_RL_WAIT_SECONDS, 0, -1):
                        progress.progress(
                            pct,
                            text=(f"Yahoo rate limit — retrying "
                                  f"{pos['ticker']} in {left}s (the scan "
                                  f"will finish, just slower)…"))
                        time.sleep(1)
            results.append(res)

        progress.empty()
        _any_from_cache = any(r.get("from_cache") for r in results)
        _cached_ats = [r["fetched_at"] for r in results if r.get("fetched_at")]
        st.session_state["scan_ts"] = (
            max(_cached_ats) if _any_from_cache and _cached_ats
            else datetime.now().astimezone()
        )
        st.session_state["scan_provider"] = _provider
        st.session_state["scan_from_cache"] = _any_from_cache
        st.session_state[_results_key] = {
            "results": results,
            "uploaded_name": source_name,
            "opt_type": opt_type_key,
            "scan_mode": scan_mode_key,
            "buy": buy,
        }

    # ── Render stored results (survives widget interactions / re-runs) ────────
    stored = st.session_state.get(_results_key)
    if stored is None:
        return

    results          = stored["results"]
    uploaded_name    = stored["uploaded_name"]
    stored_opt_type  = stored.get("opt_type", "calls")
    stored_scan_mode = stored.get("scan_mode", "roll")
    stored_buy       = stored.get("buy", False)
    stored_side      = {"calls": "call", "puts": "put", "both": "both"}[stored_opt_type]

    # Every ticker failing is the expired-Schwab-token signature — surface
    # the re-auth fix once, above the results. A partial failure is more
    # likely a bad symbol or empty chain, so just name the misses.
    failed = [r["position"]["ticker"] for r in results if r.get("error")]
    if failed and len(failed) == len(results):
        _scfg = st.session_state.get("schwab_config") or {}
        render_schwab_reauth_hint(
            st.session_state.get("scan_provider", "yahoo"),
            key=f"schwab_reauth_{k}",
            token_file=_scfg.get("token_file"),
        )
    elif failed:
        st.warning(
            f"Could not fetch {len(failed)} of {len(results)} tickers "
            f"({', '.join(failed)}) — details in their sections below."
        )

    # ── Cross-ticker leaderboard — richest IV+pp across the whole basket ──────
    if len(results) > 1:
        section_header(
            title="Leaderboard",
            subtitle="Richest IV+pp contracts across every scanned ticker.",
            eyebrow="TOP CANDIDATES",
        )
        # Assisted put-selling (stub): a per-row "investigate" control on the
        # Puts board, gated to watchlist + sell + Schwab. Yahoo can only read
        # quotes; placing an order needs Schwab. See
        # options-scanner/assisted-put-selling-implementation-plan.md.
        _lb_provider = st.session_state.get("scan_provider", "yahoo")
        # Show the investigate control only when the *displayed* results came
        # from a Schwab scan AND Schwab is still the currently-selected source.
        # scan_provider is a per-scan snapshot, so toggling the data source away
        # hides the control and toggling back to Schwab restores it (no rescan).
        _allow_investigate = (
            is_watchlist and not stored_buy
            and _lb_provider == "schwab"
            and st.session_state.get("data_source") == "schwab"
        )
        render_leaderboard(results, stored_side, int(port_min_oi),
                           int(port_top), int(port_min_vol),
                           delta_range=port_delta_range, buy=stored_buy,
                           allow_investigate=_allow_investigate,
                           provider=_lb_provider,
                           min_ivpp=port_min_ivpp, min_ann=port_min_ann,
                           min_percentile=port_min_percentile,
                           min_ann_delta_percentile=port_min_ann_delta_percentile,
                           show_mc=port_show_mc)

        with st.expander("Capital Allocator", expanded=False):
            st.caption(
                "Greedy waterfill across the scanned basket, ranked by "
                "expected P&L per dollar of worst-5%-tail risk (a quick "
                "per-candidate Monte Carlo run). Sell-side only — CSP / "
                "covered-call collateral sizing."
            )
            if stored_buy:
                st.info("Capital allocation is sell-side only.")
            else:
                _budget = st.number_input(
                    "Capital to allocate ($)", value=10000.0, min_value=0.0,
                    step=1000.0, key=f"{k}_alloc_budget",
                )
                _run_alloc = st.button("Allocate", key=f"{k}_alloc_run")
                if _run_alloc and _budget > 0:
                    _alloc_sides = ([stored_side] if stored_side in ("call", "put")
                                    else ["call", "put"])
                    _alloc_boards = [
                        build_leaderboard(
                            results, s, int(port_min_oi), int(port_top),
                            int(port_min_vol), delta_range=port_delta_range,
                            buy=False, min_ivpp=port_min_ivpp,
                            min_ann=port_min_ann,
                            min_percentile=port_min_percentile,
                            min_ann_delta_percentile=port_min_ann_delta_percentile,
                        )
                        for s in _alloc_sides
                    ]
                    _alloc_boards = [b for b in _alloc_boards if not b.empty]
                    _alloc_board = (pd.concat(_alloc_boards, ignore_index=True)
                                    if _alloc_boards else pd.DataFrame())
                    if _alloc_board.empty:
                        st.info("No candidates to allocate against — loosen "
                                "the scan filters above.")
                    else:
                        _alloc_tickers = _alloc_board["ticker"].unique().tolist()
                        _corr = None
                        if len(_alloc_tickers) > 1:
                            with st.spinner("Checking cross-ticker correlation…"):
                                _corr = correlation.pairwise_log_return_corr(
                                    _alloc_tickers)
                        with st.spinner("Running per-candidate Monte Carlo…"):
                            _candidates = capital_allocator.candidates_from_board(
                                _alloc_board)
                        _alloc = capital_allocator.allocate_capital(
                            _candidates, float(_budget), corr=_corr)
                        if _alloc.empty:
                            st.info("No candidates cleared a positive "
                                    "risk-adjusted edge for this budget.")
                        else:
                            _disp = _alloc.rename(columns={
                                "ticker": "Ticker", "strike": "Strike",
                                "expiration": "Expiration", "opt_type": "Type",
                                "contracts": "Contracts", "collateral": "Collateral",
                                "edge_score": "Edge", "expected_pnl": "Exp P&L",
                            })
                            _disp["P(profit)"] = _alloc["prob_profit"] * 100.0
                            _disp = _disp.drop(columns=["prob_profit"])
                            st.dataframe(
                                _disp, hide_index=True, width="stretch",
                                column_config={
                                    "Strike": st.column_config.NumberColumn(
                                        "Strike", format="$%.2f"),
                                    "Collateral": st.column_config.NumberColumn(
                                        "Collateral", format="$%.2f"),
                                    "Edge": st.column_config.NumberColumn(
                                        "Edge", format="%.3f",
                                        help="Expected P&L / |worst-5% tail "
                                             "P&L| — higher is better."),
                                    "Exp P&L": st.column_config.NumberColumn(
                                        "Exp P&L", format="$%.2f"),
                                    "P(profit)": st.column_config.NumberColumn(
                                        "P(profit)", format="%.1f%%"),
                                },
                            )
                            st.caption(
                                f"Allocated ${_alloc['collateral'].sum():,.2f} "
                                f"of ${_budget:,.2f} across {len(_alloc)} "
                                "position(s)."
                            )
                            if _corr is not None:
                                _picked = _alloc["ticker"].tolist()
                                _pairs = [
                                    (a, b, _corr.loc[a, b])
                                    for i, a in enumerate(_picked)
                                    for b in _picked[i + 1:]
                                    if a in _corr.index and b in _corr.columns
                                    and pd.notna(_corr.loc[a, b])
                                    and abs(_corr.loc[a, b]) > 0.7
                                ]
                                if _pairs:
                                    _pair_strs = ", ".join(
                                        f"{a}/{b} ({c:+.2f})" for a, b, c in _pairs)
                                    st.warning(
                                        f"Correlated picks in this allocation: "
                                        f"{_pair_strs} — consider whether this "
                                        "concentrates risk more than it looks."
                                    )

    for res in results:
        pos    = res["position"]
        ticker = pos["ticker"]

        # Build a compact expander label summarising what this position holds.
        shares_str = f"{pos['shares']:g} shares" if pos["shares"] > 0 else "no shares"
        label_parts = [ticker, shares_str]
        if stored_opt_type in ("calls", "both") and pos["open_calls"]:
            n = sum(o["contracts"] for o in pos["open_calls"])
            label_parts.append(f"{n} short call(s)")
        if stored_opt_type in ("puts", "both") and pos["open_puts"]:
            n = sum(o["contracts"] for o in pos["open_puts"])
            label_parts.append(f"{n} short put(s)")
        label = " — ".join(label_parts)

        with st.expander(label, expanded=True):
            if res["error"]:
                st.error(res["error"])
                continue

            spot           = res["spot"]
            earnings_dates = res["earnings_dates"]
            df             = res["df"]

            if spot is None or df.empty:
                st.warning("No options data returned — Yahoo may be "
                           "throttling. Try again in a moment.")
                continue

            m1, m2, m3, m4 = st.columns(4)
            if earnings_dates:
                earn_days = (earnings_dates[0] - date.today()).days
                earn_label = f"{earnings_dates[0].strftime('%b %d')}"
                earn_sub   = f"in {earn_days}d"
            else:
                earn_label = "—"
                earn_sub   = "no upcoming events"
            with m1:
                _meta = fetch_spot_meta(
                    ticker, st.session_state.get("scan_provider", "yahoo"),
                )
                metric_card("SPOT",
                            spot_value_html(spot, _meta["pct_change"]),
                            help_text=spot_help_text(_meta))
            with m2:
                metric_card("SHARES", f"{pos['shares']:,g}" if pos["shares"] > 0 else "—")
            with m3:
                metric_card("EXPIRATIONS", f"{df['expiration'].nunique()}")
            with m4:
                metric_card("NEXT EARNINGS", earn_label,
                            delta=earn_sub, delta_sign="neutral")

            # Open option info rows (Roll mode only — irrelevant in Best mode)
            if stored_scan_mode == "roll":
                if stored_opt_type in ("calls", "both"):
                    for opt in pos["open_calls"]:
                        close = res["roll_close_costs"].get(opt["symbol"])
                        close_str = (f" — close mid: **${close:.2f}**"
                                     if close else "")
                        st.info(f"Open call: **{opt['symbol']}** "
                                f"({opt['contracts']} contract(s)){close_str}")
                if stored_opt_type in ("puts", "both"):
                    for opt in pos["open_puts"]:
                        close = res["roll_close_costs"].get(opt["symbol"])
                        close_str = (f" — close mid: **${close:.2f}**"
                                     if close else "")
                        st.info(f"Open put: **{opt['symbol']}** "
                                f"({opt['contracts']} contract(s)){close_str}")

            port_delta_min, port_delta_max = port_delta_range
            df_filt = df[df["delta"].abs().between(
                port_delta_min, port_delta_max)].copy()

            # Action card(s) — one per side being scanned.
            if stored_opt_type in ("calls", "both"):
                df_calls = (df_filt[df_filt["type"] == "call"]
                            if stored_opt_type == "both" else df_filt)
                roll_close_call = None
                if stored_scan_mode == "roll" and pos["open_calls"]:
                    roll_close_call = res["roll_close_costs"].get(
                        pos["open_calls"][0]["symbol"]
                    )
                render_portfolio_action_card(
                    ticker=ticker,
                    df_filt=df_calls,
                    spot=spot,
                    shares=int(pos["shares"]),
                    covered=bool(pos["open_calls"]) and stored_scan_mode == "roll",
                    roll_close=roll_close_call,
                    open_options=pos["open_calls"],
                    min_oi=int(port_min_oi),
                    min_vol=int(port_min_vol),
                    opt_type="calls",
                    buy=stored_buy,
                    min_ivpp=port_min_ivpp,
                    min_ann=port_min_ann,
                )

            if stored_opt_type in ("puts", "both"):
                df_puts = (df_filt[df_filt["type"] == "put"]
                           if stored_opt_type == "both" else df_filt)
                roll_close_put = None
                if stored_scan_mode == "roll" and pos["open_puts"]:
                    roll_close_put = res["roll_close_costs"].get(
                        pos["open_puts"][0]["symbol"]
                    )
                render_portfolio_action_card(
                    ticker=ticker,
                    df_filt=df_puts,
                    spot=spot,
                    shares=int(pos["shares"]),
                    covered=bool(pos["open_puts"]) and stored_scan_mode == "roll",
                    roll_close=roll_close_put,
                    open_options=pos["open_puts"],
                    min_oi=int(port_min_oi),
                    min_vol=int(port_min_vol),
                    opt_type="puts",
                    buy=stored_buy,
                    min_ivpp=port_min_ivpp,
                    min_ann=port_min_ann,
                )

            show_iv_chart(df_filt, spot, stored_side,
                           int(port_min_oi), int(port_top), stored_buy,
                           ticker=ticker, key_prefix=f"{k}_{ticker}",
                           min_vol=int(port_min_vol),
                           provider=st.session_state.get("scan_provider", "yahoo"))

            # For roll_close_cost in the candidates table: use the first open
            # option for the active side; None if no roll context or "both".
            _table_roll_close = None
            if stored_scan_mode == "roll":
                if stored_opt_type == "calls" and pos["open_calls"]:
                    _table_roll_close = res["roll_close_costs"].get(
                        pos["open_calls"][0]["symbol"]
                    )
                elif stored_opt_type == "puts" and pos["open_puts"]:
                    _table_roll_close = res["roll_close_costs"].get(
                        pos["open_puts"][0]["symbol"]
                    )

            st.markdown("**Top candidates**")
            show_scan_results(df_filt, stored_side, stored_buy, _table_roll_close,
                               int(port_min_oi), int(port_top),
                               int(port_min_vol),
                               min_ivpp=port_min_ivpp, min_ann=port_min_ann,
                               min_percentile=port_min_percentile,
                               min_ann_delta_percentile=port_min_ann_delta_percentile,
                               show_mc=port_show_mc)

    # Portfolio HTML download
    from options_scanner.report import render_portfolio_html
    port_html = render_portfolio_html(
        results, uploaded_name, int(port_min_oi), int(port_top),
        int(port_min_vol), opt_type=stored_opt_type,
        delta_range=port_delta_range, buy=stored_buy,
    )
    _report_kind = "Watchlist" if is_watchlist else "Portfolio"
    st.download_button(
        f"⬇ Download {_report_kind} Report",
        data=port_html.encode("utf-8"),
        file_name=(f"{_report_kind.lower()}_"
                   f"{date.today().strftime('%Y%m%d')}.html"),
        mime="text/html",
        key=f"{k}_dl_report",
    )
